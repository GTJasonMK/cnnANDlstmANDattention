from __future__ import annotations

import argparse
import os
from typing import Optional
import random

import numpy as np
import torch

from configs.config import load_config
from dataProcess.data_preprocessor import load_array_from_path, create_dataloaders
from model_architecture import CNNLSTMAttentionModel
from trainer import Trainer
from eval.evaluator import evaluate_model
from visualizer import (
    plot_losses, plot_losses_logscale, plot_lr, plot_grad_norm, plot_param_count,
    plot_predictions, plot_residual_hist, plot_prediction_interval, plot_multihorizon_error,
    plot_attention_heatmap, plot_attention_multihead, plot_lstm_hidden_heatmap, plot_cnn_feature_maps,
    plot_series_distribution, plot_corr_heatmap, plot_split_distribution, plot_temporal_performance,
)


# 环境与种子设置
import random

def setup_env(cfg):
    # 随机种子
    seed = int(getattr(cfg.train, 'seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN 设置
    det = getattr(cfg.train, 'deterministic', None)
    bench = getattr(cfg.train, 'cudnn_benchmark', None)
    if det is not None:
        torch.backends.cudnn.deterministic = bool(det)
        # 当 deterministic=True，通常需要将 benchmark 关掉
        if det:
            torch.backends.cudnn.benchmark = False
    if bench is not None:
        torch.backends.cudnn.benchmark = bool(bench)

    # matmul 精度（PyTorch>=2.0）
    try:
        if getattr(cfg.train, 'matmul_precision', None):
            torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    except Exception:
        pass

    # 路径解析工具：仅当目标是相对路径时，才在 output_dir 下拼接；绝对路径保持不变；
    # 若目标已在 output_dir 之下，也保持不变，避免重复嵌套。
    def _resolve_path(base_out: str | None, desired: str | None, default_subdir: str) -> str:
        d = desired
        if not d:
            if base_out:
                return os.path.join(base_out, default_subdir)
            return default_subdir
        # 绝对路径直接返回
        if os.path.isabs(d):
            return d
        # 相对路径：如果设置了 base_out
        if base_out:
            abs_d = os.path.abspath(d)
            abs_base = os.path.abspath(base_out)
            try:
                common = os.path.commonpath([abs_d, abs_base])
            except Exception:
                common = ''
            # 若目标已经位于 base_out 之下，则保持不变；否则拼接 base_out
            if common == abs_base:
                return d
            return os.path.join(base_out, d)
        # 无 base_out 时，返回相对路径本身
        return d

    # 统一输出目录：output_dir > visual_save_dir/train.checkpoints.dir/train.log_dir
    base_out = getattr(cfg, 'output_dir', None)
    if base_out:
        os.makedirs(base_out, exist_ok=True)

    # image dir
    cfg.visual_save_dir = _resolve_path(base_out, getattr(cfg, 'visual_save_dir', None), 'image')
    # checkpoints dir
    ckpt_dir_cur = getattr(cfg.train.checkpoints, 'dir', None)
    cfg.train.checkpoints.dir = _resolve_path(base_out, ckpt_dir_cur, 'checkpoints')
    # tensorboard dir
    log_dir_cur = getattr(cfg.train, 'log_dir', None)
    cfg.train.log_dir = _resolve_path(base_out, log_dir_cur, 'runs')

    # 创建最终目录
    os.makedirs(cfg.visual_save_dir, exist_ok=True)
    os.makedirs(cfg.train.checkpoints.dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)

    # 可视化保存目录传给可视化模块（通过环境变量简化改动）
    os.environ["VIS_SAVE_DIR"] = getattr(cfg, 'visual_save_dir', 'image')


# 统一的数据/模型构建

def build_model_with_data(cfg, input_size: int, n_targets: int):
    # Build model from cfg but override num_features and targets
    m = cfg.model
    # 兼容：当选择 TCN 时，沿用 cnn.layers 作为 TCN 的层配置
    model = CNNLSTMAttentionModel(
        num_features=input_size,
        cnn_layers=[vars(l) if not isinstance(l, dict) else l for l in (m.tcn.layers if getattr(m, 'tcn', None) and getattr(m.tcn, 'enabled', False) else m.cnn.layers)],
        use_batchnorm=(m.tcn.use_batchnorm if getattr(m, 'tcn', None) and getattr(m.tcn, 'enabled', False) else m.cnn.use_batchnorm),
        cnn_dropout=(m.tcn.dropout if getattr(m, 'tcn', None) and getattr(m.tcn, 'enabled', False) else m.cnn.dropout),
        lstm_hidden=m.lstm.hidden_size,
        lstm_layers=m.lstm.num_layers,
        bidirectional=m.lstm.bidirectional,
        attn_enabled=m.attention.enabled,
        attn_heads=m.attention.num_heads,
        attn_dropout=m.attention.dropout,
        fc_hidden=m.fc_hidden,
        forecast_horizon=m.forecast_horizon,
        n_targets=n_targets,
        attn_add_pos_enc=m.attention.add_posional_encoding if hasattr(m.attention, 'add_posional_encoding') else m.attention.add_positional_encoding,
        lstm_dropout=m.lstm.dropout,
        cnn_variant=(
            'tcn' if (getattr(m, 'tcn', None) and getattr(m.tcn, 'enabled', False)) else getattr(m.cnn, 'variant', 'standard')
        ),
        attn_variant=getattr(m.attention, 'variant', 'standard'),
        multiscale_scales=getattr(m.attention, 'multiscale_scales', [1, 2]),
        multiscale_fuse=getattr(m.attention, 'multiscale_fuse', 'sum'),
        attn_positional_mode=getattr(m.attention, 'positional_mode', 'none'),
        local_window_size=getattr(m.attention, 'local_window_size', 64),
        local_dilation=getattr(m.attention, 'local_dilation', 1),
        cnn_use_channel_attention=getattr(m.cnn, 'use_channel_attention', False),
        cnn_channel_attention_type=getattr(m.cnn, 'channel_attention_type', 'eca'),
        normalization=getattr(m, 'normalization', None),
        decomposition=getattr(m, 'decomposition', None),
    )
    # 设置RNN类型（LSTM/GRU）
    try:
        model.rnn_type = getattr(m.lstm, 'rnn_type', 'lstm')
    except Exception:
        model.rnn_type = 'lstm'
    return model


def prepare_run(cfg):
    # 数据加载（一次读取，后续可复用数据数组）
    if cfg.data.data_path is None:
        raise ValueError("Please provide data.data_path in config pointing to CSV/NPZ/NPY")
    data = load_array_from_path(cfg.data.data_path)

    # DataLoader 性能参数（根据设备/用户配置推断）
    use_cuda = torch.cuda.is_available()
    pin_memory = True if use_cuda else False
    # 若用户已在 cfg.data.num_workers 指定，则尊重；否则自动给个较合理默认
    num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else (os.cpu_count() or 0)
    persistent_workers = True if (num_workers and num_workers > 0) else False
    prefetch_factor = 2 if (num_workers and num_workers > 0) else None

    train_loader, val_loader, test_loader, input_size, n_targets = create_dataloaders(
        data=data,
        sequence_length=cfg.data.sequence_length,
        horizon=cfg.data.horizon,
        feature_indices=cfg.data.feature_indices,
        target_indices=cfg.data.target_indices,
        normalize=cfg.data.normalize,
        batch_size=cfg.data.batch_size,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        num_workers=num_workers,
        shuffle_train=cfg.data.shuffle_train,
        drop_last=cfg.data.drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = build_model_with_data(cfg, input_size, n_targets)
    return data, model, (train_loader, val_loader, test_loader)


def run(cfg, resume_path: Optional[str] = None):
    setup_env(cfg)
    data, model, (train_loader, val_loader, test_loader) = prepare_run(cfg)

    trainer = Trainer(model, cfg)
    if resume_path:
        last_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resumed from epoch {last_epoch}")

    history = trainer.fit(train_loader, val_loader, test_loader)

    # 可视化与评估开关
    if getattr(cfg, 'visual_enabled', True):
        plot_losses(history)
        plot_losses_logscale(history)
        plot_lr(history)
        try:
            param_count = sum(p.numel() for p in model.parameters())
            plot_param_count(param_count)
        except Exception:
            pass

    # 预测与指标
    preds, targets = trainer.predict(test_loader)
    metrics = evaluate_model(model, test_loader, trainer.device)
    # 将训练耗时加入指标字典，便于搜索器解析
    if isinstance(history, dict) and 'train_time_sec' in history:
        metrics['train_time_sec'] = float(history['train_time_sec'])
    print({k: round(v, 6) if isinstance(v, (int, float)) else v for k, v in metrics.items()})

    if getattr(cfg, 'visual_enabled', True):
        plot_predictions(preds, targets)
        plot_residual_hist(preds, targets)
        plot_multihorizon_error(preds, targets)
        # 使用首次读取的 data 做数据分析，避免重复 IO
        try:
            plot_series_distribution(data)
            plot_corr_heatmap(data)
        except Exception:
            pass

        # 注意力图（若启用）
        model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(trainer.device, non_blocking=True)
                _, attn = model(x, return_attn=True)
                plot_attention_heatmap(attn)
                if attn is not None:
                    plot_attention_multihead(attn)
                break

    return history, metrics



def main(config_path: Optional[str], resume_path: Optional[str] = None):
    cfg = load_config(config_path)
    return run(cfg, resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--output_dir", type=str, default=None, help="Base output directory (overrides config.output_dir)")
    parser.add_argument("--image_dir", type=str, default=None, help="Image output directory (overrides visual_save_dir)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoints directory (overrides train.checkpoints.dir)")
    args = parser.parse_args()

    # 应用 CLI 覆盖
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.image_dir:
        cfg.visual_save_dir = args.image_dir
    if args.ckpt_dir:
        cfg.train.checkpoints.dir = args.ckpt_dir

    run(cfg, resume_path=args.resume)

