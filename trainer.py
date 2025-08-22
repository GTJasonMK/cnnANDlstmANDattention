from __future__ import annotations

import math
import os
import time
import json
import hashlib
from dataclasses import asdict
from typing import Dict, Tuple, Optional

from configs.config import canonicalize_for_checkpoint, validate_full_config_strict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from configs.config import FullConfig


def _get_device(cfg: FullConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {name}")


def _build_optimizer(params, cfg) -> Optimizer:
    name = cfg.name.lower()
    if name == "adam":
        betas = tuple(cfg.betas) if cfg.betas else (0.9, 0.999)
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=betas)
    if name == "adamw":
        betas = tuple(cfg.betas) if cfg.betas else (0.9, 0.999)
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: Optimizer, cfg):
    name = (cfg.name or "").lower() if cfg else None
    if not name:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg.reduce_on_plateau_mode, patience=cfg.reduce_on_plateau_patience)
    raise ValueError(f"Unsupported scheduler: {name}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.wait = 0
        self.stop = False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        self.stop = self.wait >= self.patience
        return self.stop


class Trainer:
    """
    Generic trainer handling training/validation loops, checkpointing, early stopping and AMP.
    """

    def __init__(self, model: nn.Module, cfg: FullConfig, work_dir: str = ".") -> None:
        self.model = model
        self.cfg = cfg
        self.work_dir = work_dir
        self.device = _get_device(cfg)
        self.model.to(self.device)
        self.amp = (self.device.type == 'cuda') and cfg.train.mixed_precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.criterion = _build_loss(cfg.train.loss)
        self.optimizer = _build_optimizer(self.model.parameters(), cfg.train.optimizer)
        self.scheduler = _build_scheduler(self.optimizer, cfg.train.scheduler)
        self.early_stopper = EarlyStopping(cfg.train.early_stopping.patience, cfg.train.early_stopping.min_delta) if cfg.train.early_stopping.enabled else None
        os.makedirs(cfg.train.checkpoints.dir, exist_ok=True)
        self.best_val = math.inf
        # tensorboard (lazy import)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.tb = SummaryWriter(log_dir=self.cfg.train.log_dir)
        except Exception:
            self.tb = None

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns last epoch number (0 if none)."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])  # type: ignore
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])  # type: ignore
        self.best_val = ckpt.get("best_val", math.inf)
        return int(ckpt.get("epoch", 0))

    def _step(self, batch, train: bool = True) -> Tuple[torch.Tensor, float]:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        # use torch.amp autocast (cuda)
        autocast_ctx = torch.amp.autocast('cuda', enabled=self.amp)
        with autocast_ctx:
            preds = self.model(x)
            loss = self.criterion(preds, y)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.train.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return preds.detach(), float(loss.detach().item())

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, test_loader: Optional[DataLoader] = None):
        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
        t0 = time.time()
        for epoch in range(1, self.cfg.train.epochs + 1):
            self.model.train()
            running = 0.0
            for i, batch in enumerate(train_loader, start=1):
                _, loss = self._step(batch, train=True)
                running += loss
                if i % self.cfg.train.print_every == 0:
                    print(f"Epoch {epoch} Step {i}: train_loss={running / i:.6f}")
            avg_train = running / max(1, len(train_loader))
            history["train_loss"].append(avg_train)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                v_running = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        _, v_loss = self._step(batch, train=False)
                        v_running += v_loss
                val_loss = v_running / max(1, len(val_loader))
                history["val_loss"].append(val_loss)

            test_loss = None
            if test_loader is not None:
                self.model.eval()
                t_running = 0.0
                with torch.no_grad():
                    for batch in test_loader:
                        _, t_loss = self._step(batch, train=False)
                        t_running += t_loss
                test_loss = t_running / max(1, len(test_loader))
                history["test_loss"].append(test_loss)

            # scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else avg_train)
                else:
                    self.scheduler.step()
            # record lr
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # epoch summary print
            msg = f"Epoch {epoch}: train_loss={avg_train:.6f}"
            if val_loader is not None and val_loss is not None:
                msg += f", val_loss={val_loss:.6f}"
            if test_loader is not None and test_loss is not None:
                msg += f", test_loss={test_loss:.6f}"
            print(msg)

            # tensorboard logging
            if self.tb is not None:
                self.tb.add_scalar("loss/train", avg_train, epoch)
                if val_loss is not None:
                    self.tb.add_scalar("loss/val", val_loss, epoch)
                self.tb.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # checkpoint
            is_best = val_loss is not None and val_loss < self.best_val
            if is_best:
                self.best_val = val_loss  # type: ignore
            # 规范化+严格校验，确保保存的 cfg 可被严格评估端精确重建
            cfg_to_save = canonicalize_for_checkpoint(self.cfg)
            validate_full_config_strict(cfg_to_save)
            ckpt = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "cfg": asdict(cfg_to_save),
                "best_val": self.best_val,
            }
            # 按配置保存按-epoch 的快照
            if (not self.cfg.train.checkpoints.save_best_only) or is_best:
                epoch_path = os.path.join(self.cfg.train.checkpoints.dir, f"model_epoch{epoch}.pt")
                torch.save(ckpt, epoch_path)
            # 额外保存/更新 "model_best.pt"
            if is_best:
                best_path = os.path.join(self.cfg.train.checkpoints.dir, "model_best.pt")
                torch.save(ckpt, best_path)
                # 可选：将最佳模型另外导出到指定目录并按架构首字母命名（例如 sgs.pt）
                export_dir = getattr(self.cfg.train.checkpoints, 'export_best_dir', None)
                if export_dir:
                    os.makedirs(export_dir, exist_ok=True)
                    m = self.cfg.model
                    # Build human-readable slug
                    cnn_variant = str(getattr(m.cnn, 'variant', 'standard')).strip().lower()
                    # If TCN enabled, override cnn tag to 'tcn'
                    try:
                        tcn_enabled = bool(getattr(getattr(m, 'tcn', None), 'enabled', False))
                    except Exception:
                        tcn_enabled = False
                    if tcn_enabled or cnn_variant == 'tcn':
                        cnn_tag = 'tcn'
                    else:
                        cnn_tag = cnn_variant
                    rnn_type = str(getattr(m.lstm, 'rnn_type', 'lstm')).strip().lower()
                    attn_variant = str(getattr(m.attention, 'variant', 'standard')).strip().lower()
                    pos_mode = str(getattr(m.attention, 'positional_mode', 'none')).strip().lower()
                    ca_on = bool(getattr(m.cnn, 'use_channel_attention', False))
                    bi = bool(getattr(m.lstm, 'bidirectional', True))
                    heads = int(getattr(m.attention, 'num_heads', 4)) if hasattr(m, 'attention') else 0
                    slug = f"{cnn_tag}-{rnn_type}-{attn_variant}-pos{pos_mode}-ca{'on' if ca_on else 'off'}-{'bi' if bi else 'uni'}-H{heads}"
                    # Deterministic short hash from model config to avoid collisions
                    try:
                        cfg_model_dict = asdict(canonicalize_for_checkpoint(self.cfg)).get('model', {})
                    except Exception:
                        cfg_model_dict = {}
                    hash_src = json.dumps(cfg_model_dict, sort_keys=True, ensure_ascii=False)
                    sh = hashlib.sha1(hash_src.encode('utf-8')).hexdigest()[:8]
                    export_name = f"{slug}_{sh}.pt"
                    export_path = os.path.join(export_dir, export_name)
                    torch.save(ckpt, export_path)
                    try:
                        print(f"[checkpoint] Exported BEST to {export_path}")
                    except Exception:
                        pass

            # early stopping
            if self.early_stopper is not None and val_loss is not None:
                if self.early_stopper.step(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
        if self.tb is not None:
            self.tb.flush()
        # 始终在训练结束时保存最终模型
        # 结束时也保存规范化后的 cfg
        cfg_to_save = canonicalize_for_checkpoint(self.cfg)
        validate_full_config_strict(cfg_to_save)
        final_ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": asdict(cfg_to_save),
            "best_val": self.best_val,
        }
        last_path = os.path.join(self.cfg.train.checkpoints.dir, "model_last.pt")
        torch.save(final_ckpt, last_path)
        print(f"Saved final model to {last_path}")
        # 训练耗时统计
        history["train_time_sec"] = float(time.time() - t0)
        print(f"Training finished in {history['train_time_sec']:.2f} sec")
        return history

    def evaluate(self, loader: DataLoader, criterion: Optional[nn.Module] = None) -> float:
        criterion = criterion or self.criterion
        self.model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)
                loss = criterion(preds, y)
                running += float(loss.item())
        return running / max(1, len(loader))

    def predict(self, loader: DataLoader):
        self.model.eval()
        preds_list = []
        ys = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                preds = self.model(x)
                preds_list.append(preds.cpu())
                ys.append(y)
        return torch.cat(preds_list, dim=0), torch.cat(ys, dim=0)

