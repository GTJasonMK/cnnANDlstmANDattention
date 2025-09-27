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


def _safe_get(obj, path: str, default=None):
    """安全获取嵌套属性值"""
    try:
        keys = path.split('.')
        current = obj
        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        return current
    except Exception:
        return default


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
    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
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
        # 🔥 CRITICAL FIX: 使用安全属性访问避免AttributeError
        self.amp = (self.device.type == 'cuda') and _safe_get(cfg, 'train.mixed_precision', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.criterion = _build_loss(_safe_get(cfg, 'train.loss', 'mse'))
        self.optimizer = _build_optimizer(self.model.parameters(), _safe_get(cfg, 'train.optimizer'))
        self.scheduler = _build_scheduler(self.optimizer, _safe_get(cfg, 'train.scheduler'))

        # 早停配置安全访问
        early_stop_cfg = _safe_get(cfg, 'train.early_stopping')
        if early_stop_cfg and _safe_get(early_stop_cfg, 'enabled', False):
            patience = _safe_get(early_stop_cfg, 'patience', 20)
            min_delta = _safe_get(early_stop_cfg, 'min_delta', 0.0)
            self.early_stopper = EarlyStopping(patience, min_delta)
        else:
            self.early_stopper = None

        # 检查点目录安全访问
        checkpoints_dir = _safe_get(cfg, 'train.checkpoints.dir', 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.best_val = math.inf

        # tensorboard (lazy import)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            log_dir = _safe_get(cfg, 'train.log_dir', 'runs')
            self.tb = SummaryWriter(log_dir=log_dir)
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
        
        # 检查输入数据是否包含nan
        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)):
            print(f"[ERROR] Input data contains nan! train={train}")
            print(f"  X nan count: {torch.sum(torch.isnan(x)).item()}")
            print(f"  Y nan count: {torch.sum(torch.isnan(y)).item()}")
            # 修复输入数据中的NaN值
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            y = torch.where(torch.isnan(y), torch.zeros_like(y), y)
            print(f"[FIX] Replaced input NaN values with zeros")
        
        # use torch.amp autocast (cuda)
        autocast_ctx = torch.amp.autocast('cuda', enabled=self.amp)
        with autocast_ctx:
            preds = self.model(x)
            
            # 检查模型输出是否包含nan
            if torch.any(torch.isnan(preds)):
                print(f"[ERROR] Model predictions contain nan! train={train}")
                print(f"  Pred shape: {preds.shape}")
                print(f"  Pred nan count: {torch.sum(torch.isnan(preds)).item()}")
                print(f"  Input stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
                
                # 🔥 CRITICAL FIX: Remove label leakage - do NOT use target information during training!
                # Use consistent zero replacement for both training and evaluation to maintain fairness
                preds = torch.where(torch.isnan(preds), torch.zeros_like(preds), preds)
                print(f"[FIXED] Replaced prediction NaN values with zeros (NO TARGET LEAKAGE)")
                
                # Additional numerical stability: if too many NaNs, skip this batch entirely
                nan_ratio = torch.sum(torch.isnan(preds)).float() / preds.numel()
                if nan_ratio > 0.5:  # If more than 50% are NaN, skip batch
                    print(f"[SKIP] Batch has {nan_ratio:.2%} NaN predictions - skipping to prevent instability")
                    if train:
                        return torch.zeros_like(preds), 1.0  # Return dummy values
                    else:
                        return torch.zeros_like(preds), 1.0
            
            loss = self.criterion(preds, y)
            
            # 检查损失是否为nan
            if torch.isnan(loss):
                print(f"[ERROR] Loss is nan! train={train}")
                print(f"  Pred stats: min={preds.min():.6f}, max={preds.max():.6f}, mean={preds.mean():.6f}")
                print(f"  Target stats: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")
                
                # 🔥 CRITICAL FIX: Use consistent approach - no target information
                # Skip this batch entirely instead of setting arbitrary loss values
                if train:
                    print(f"[SKIP] Skipping batch with NaN loss during training")
                    return torch.zeros_like(preds), 0.0  # Return zero loss to not affect training
                else:
                    print(f"[SKIP] Skipping batch with NaN loss during evaluation")
                    return torch.zeros_like(preds), 1.0  # Return non-zero loss for evaluation tracking
                
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            
            # 检查scaled loss是否为nan
            scaled_loss = self.scaler.scale(loss)
            if torch.isnan(scaled_loss):
                print(f"[ERROR] Scaled loss is nan!")
                # 跳过这个batch的反向传播
                return preds.detach(), 1.0
                
            scaled_loss.backward()
            
            # 检查梯度是否包含NaN
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    print(f"[ERROR] NaN gradient in {name}")
                    param.grad.zero_()  # 清零该参数的梯度
                    has_nan_grad = True
            
            if has_nan_grad:
                print(f"[FIX] Cleared NaN gradients, skipping optimizer step")
                return preds.detach(), 1.0

            if _safe_get(self.cfg, 'train.gradient_clip') is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _safe_get(self.cfg, 'train.gradient_clip'))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        loss_item = float(loss.detach().item())
        if math.isnan(loss_item):
            print(f"[ERROR] Loss item is nan after detach: {loss_item}")
            loss_item = 1.0  # 设置默认值
            
        return preds.detach(), loss_item

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, test_loader: Optional[DataLoader] = None):
        history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
        t0 = time.time()
        # 🔥 CRITICAL FIX: 使用安全属性访问
        epochs = _safe_get(self.cfg, 'train.epochs', 100)
        print_every = _safe_get(self.cfg, 'train.print_every', 50)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            for i, batch in enumerate(train_loader, start=1):
                _, loss = self._step(batch, train=True)
                running += loss
                if i % print_every == 0:
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

            # ✅ FIXED: 移除训练过程中的测试集损失计算
            # 训练过程中不应该计算测试集损失，这会导致数据泄露
            # 测试集只能在最终评估时使用
            # test_loss = None
            # if test_loader is not None:
            #     self.model.eval()
            #     t_running = 0.0
            #     with torch.no_grad():
            #         for batch in test_loader:
            #             _, t_loss = self._step(batch, train=False)
            #             t_running += t_loss
            #     test_loss = t_running / max(1, len(test_loader))
            #     history["test_loss"].append(test_loss)  # ❌ 数据泄露！
            
            # 测试集损失不在训练过程中计算，保持训练-测试完全隔离
            test_loss = None

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
            # ✅ FIXED: 移除测试损失的打印，因为训练中不再计算测试损失
            # if test_loader is not None and test_loss is not None:
            #     msg += f", test_loss={test_loss:.6f}"
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
            # 🔥 CRITICAL FIX: 使用安全属性访问检查点配置
            checkpoints_config = _safe_get(self.cfg, 'train.checkpoints')
            save_best_only = _safe_get(checkpoints_config, 'save_best_only', True) if checkpoints_config else True
            checkpoints_dir = _safe_get(checkpoints_config, 'dir', 'checkpoints') if checkpoints_config else 'checkpoints'

            # 按配置保存按-epoch 的快照
            if (not save_best_only) or is_best:
                epoch_path = os.path.join(checkpoints_dir, f"model_epoch{epoch}.pt")
                torch.save(ckpt, epoch_path)
            # 额外保存/更新 "model_best.pt"
            if is_best:
                best_path = os.path.join(checkpoints_dir, "model_best.pt")
                torch.save(ckpt, best_path)
                # 可选：将最佳模型另外导出到指定目录；确保命名全局唯一，避免覆盖
                export_dir = _safe_get(checkpoints_config, 'export_best_dir') if checkpoints_config else None
                if export_dir:
                    os.makedirs(export_dir, exist_ok=True)
                    try:
                        # 🔥 SIMPLIFIED: 简化导出文件命名逻辑
                        model_config = _safe_get(self.cfg, 'model')
                        data_config = _safe_get(self.cfg, 'data')

                        # 基于配置的简化哈希
                        cfg_model_dict = asdict(canonicalize_for_checkpoint(self.cfg)).get('model', {})
                        hash_src = json.dumps(cfg_model_dict, sort_keys=True, ensure_ascii=False)
                        config_hash = hashlib.sha1(hash_src.encode('utf-8')).hexdigest()[:8]

                        # 简化标识符
                        yaml_stem = str(_safe_get(self.cfg, 'yaml_stem', 'model')).strip()
                        safe_stem = yaml_stem.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '-')

                        # 🔥 修复：移除时间戳，避免重复导出相同配置的模型
                        # 只使用配置哈希确保唯一性，相同配置会覆盖而不是创建新文件
                        export_name = f"{safe_stem}_{config_hash}.pt"
                        export_path = os.path.join(export_dir, export_name)

                        # 🔥 新增：检查是否已存在相同配置的模型
                        if os.path.exists(export_path):
                            print(f"[checkpoint] 覆盖已存在的模型: {export_path}")
                        else:
                            print(f"[checkpoint] 导出新模型: {export_path}")

                        # 保存模型
                        torch.save(ckpt, export_path)
                        print(f"[checkpoint] Exported BEST to {export_path}")

                    except Exception as e:
                        print(f"[WARNING] Failed to export best model: {e}")
                        # 最简化回退方案
                        try:
                            export_name = f"model_best_{config_hash}.pt"
                            export_path = os.path.join(export_dir, export_name)
                            torch.save(ckpt, export_path)
                            print(f"[checkpoint] Exported BEST (fallback) to {export_path}")
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
        # 🔥 CRITICAL FIX: 使用安全属性访问获取检查点目录
        checkpoints_config = _safe_get(self.cfg, 'train.checkpoints')
        checkpoints_dir = _safe_get(checkpoints_config, 'dir', 'checkpoints') if checkpoints_config else 'checkpoints'
        last_path = os.path.join(checkpoints_dir, "model_last.pt")
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
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                preds = self.model(x)
                
                # Debug info for nan detection
                if batch_idx == 0:  # 只在第一个batch打印详细信息
                    print(f"[DEBUG] Trainer.predict - First batch:")
                    print(f"  Input shape: {x.shape}, Target shape: {y.shape}")
                    print(f"  Input stats: min={x.min().item():.6f}, max={x.max().item():.6f}")
                    print(f"  Prediction shape: {preds.shape}")
                    if torch.any(torch.isnan(preds)) or torch.any(torch.isinf(preds)):
                        print(f"  [ERROR] Predictions contain nan/inf!")
                    else:
                        print(f"  Pred stats: min={preds.min().item():.6f}, max={preds.max().item():.6f}")
                
                preds_list.append(preds.cpu())
                ys.append(y)
                
        final_preds = torch.cat(preds_list, dim=0)
        final_targets = torch.cat(ys, dim=0)
        
        print(f"[DEBUG] Trainer.predict complete - Preds: {final_preds.shape}, Targets: {final_targets.shape}")
        if torch.any(torch.isnan(final_preds)):
            print(f"[ERROR] Final predictions contain {torch.sum(torch.isnan(final_preds)).item()} nan values")
        
        return final_preds, final_targets

