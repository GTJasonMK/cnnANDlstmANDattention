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
        # 当 deterministic=True,通常需要将 benchmark 关掉
        if det:
            torch.backends.cudnn.benchmark = False
    if bench is not None:
        torch.backends.cudnn.benchmark = bool(bench)

    # matmul 精度(PyTorch>=2.0)
    try:
        if getattr(cfg.train, 'matmul_precision', None):
            torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    except Exception:
        pass

    # 路径解析工具:仅当目标是相对路径时,才在 output_dir 下拼接;绝对路径保持不变;
    # 若目标已在 output_dir 之下,也保持不变,避免重复嵌套.
    def _resolve_path(base_out: str | None, desired: str | None, default_subdir: str) -> str:
        d = desired
        if not d:
            if base_out:
                return os.path.join(base_out, default_subdir)
            return default_subdir
        # 绝对路径直接返回
        if os.path.isabs(d):
            return d
        # 相对路径:如果设置了 base_out
        if base_out:
            abs_d = os.path.abspath(d)
            abs_base = os.path.abspath(base_out)
            try:
                common = os.path.commonpath([abs_d, abs_base])
            except Exception:
                common = ''
            # 若目标已经位于 base_out 之下,则保持不变;否则拼接 base_out
            if common == abs_base:
                return d
            return os.path.join(base_out, d)
        # 无 base_out 时,返回相对路径本身
        return d

    # 统一输出目录:output_dir > visual_save_dir/train.checkpoints.dir/train.log_dir
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

    # 可视化保存目录传给可视化模块(通过环境变量简化改动)
    os.environ["VIS_SAVE_DIR"] = getattr(cfg, 'visual_save_dir', 'image')


# 统一的数据/模型构建

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

def _is_tcn_enabled(m) -> bool:
    """检查是否启用TCN"""
    return bool(_safe_get(m, 'tcn.enabled', False))

def build_model_with_data(cfg, input_size: int, n_targets: int):
    """安全构建模型，使用简化的配置访问，支持小波变换的动态架构适配"""
    m = cfg.model

    # 🔥 NEW: 检测小波变换配置并计算特征扩展
    wavelet_cfg = _safe_get(cfg, 'data.wavelet', {})
    wavelet_enabled = _safe_get(wavelet_cfg, 'enabled', False)
    original_features = None
    expansion_factor = 1

    if wavelet_enabled:
        # 计算小波扩展倍数
        level = _safe_get(wavelet_cfg, 'level', 3)
        take = _safe_get(wavelet_cfg, 'take', 'all').lower()

        if take == 'all':
            expansion_factor = level + 1  # 近似 + 细节系数
        elif take == 'details':
            expansion_factor = level
        elif take == 'approx':
            expansion_factor = 1

        # 推断原始特征数
        if expansion_factor > 1:
            original_features = input_size // expansion_factor
            print(f"[INFO] 小波变换检测: 原始特征={original_features}, 扩展后={input_size}, 扩展倍数={expansion_factor}")

    # 🔥 SIMPLIFIED: 统一配置提取，减少重复访问
    # TCN配置
    tcn_enabled = _is_tcn_enabled(m)
    if tcn_enabled:
        cnn_config = {
            'layers': _safe_get(m, 'tcn.layers', []),
            'use_batchnorm': _safe_get(m, 'tcn.use_batchnorm', True),
            'dropout': _safe_get(m, 'tcn.dropout', 0.1),
            'variant': 'tcn'
        }
    else:
        cnn_config = {
            'layers': _safe_get(m, 'cnn.layers', []),
            'use_batchnorm': _safe_get(m, 'cnn.use_batchnorm', True),
            'dropout': _safe_get(m, 'cnn.dropout', 0.1),
            'variant': _safe_get(m, 'cnn.variant', 'standard')
        }

    # LSTM配置
    lstm_config = {
        'hidden_size': _safe_get(m, 'lstm.hidden_size', 128),
        'num_layers': _safe_get(m, 'lstm.num_layers', 2),
        'bidirectional': _safe_get(m, 'lstm.bidirectional', True),
        'dropout': _safe_get(m, 'lstm.dropout', 0.1),
        'rnn_type': _safe_get(m, 'lstm.rnn_type', 'lstm')
    }

    # 注意力配置
    attention_config = {
        'enabled': _safe_get(m, 'attention.enabled', True),
        'num_heads': _safe_get(m, 'attention.num_heads', 4),
        'dropout': _safe_get(m, 'attention.dropout', 0.1),
        'variant': _safe_get(m, 'attention.variant', 'standard'),
        'add_pos_enc': (_safe_get(m, 'attention.add_positional_encoding', False) or
                       _safe_get(m, 'attention.add_posional_encoding', False)),
        'positional_mode': _safe_get(m, 'attention.positional_mode', 'none'),
        'multiscale_scales': _safe_get(m, 'attention.multiscale_scales', [1, 2]),
        'multiscale_fuse': _safe_get(m, 'attention.multiscale_fuse', 'sum'),
        'local_window_size': _safe_get(m, 'attention.local_window_size', 64),
        'local_dilation': _safe_get(m, 'attention.local_dilation', 1),
        'st_mode': _safe_get(m, 'attention.st_mode', 'serial'),
        'st_fuse': _safe_get(m, 'attention.st_fuse', 'sum')
    }

    # 通道注意力配置
    channel_attention_config = {
        'use_channel_attention': _safe_get(m, 'cnn.use_channel_attention', False),
        'channel_attention_type': _safe_get(m, 'cnn.channel_attention_type', 'eca')
    }

    # 转换层配置为字典格式
    cnn_layers = [vars(l) if not isinstance(l, dict) else l for l in cnn_config['layers']]

    # 🔥 NEW: 动态架构适配 - 小波变换特征融合
    if wavelet_enabled and expansion_factor > 1 and original_features:
        print(f"[INFO] 启用小波特征融合: 添加1×1卷积层处理{expansion_factor}倍扩展的特征")

        # 根据CNN变体选择不同的融合策略
        cnn_variant = cnn_config.get('variant', 'standard')

        if cnn_variant in ['depthwise', 'dilated', 'inception']:
            # 高级CNN变体：使用更大的融合层输出通道
            fusion_out_channels = max(96, original_features * 3)
        elif cnn_variant == 'tcn':
            # TCN：保持适中的通道数，避免过度复杂化
            fusion_out_channels = max(64, original_features * 2)
        else:
            # 标准CNN：使用基础的融合层配置
            fusion_out_channels = max(64, original_features * 2)

        # 在现有CNN层前添加特征融合层
        fusion_layer = {
            'out_channels': fusion_out_channels,
            'kernel_size': 1,  # 1×1卷积用于特征融合
            'activation': 'relu',
            'pool': None,  # 不做池化，保持时序长度
            'stride': 1,
            'padding': 0
        }

        # 将融合层插入到第一层之前
        cnn_layers.insert(0, fusion_layer)

        # 智能调整后续层的配置
        if len(cnn_layers) > 1:
            # 根据融合层输出调整第二层输入期望
            second_layer = cnn_layers[1]
            if 'out_channels' in second_layer:
                if cnn_variant in ['inception']:
                    # Inception需要更多通道来处理复杂特征
                    second_layer['out_channels'] = min(second_layer['out_channels'] * 2, 256)
                elif cnn_variant in ['depthwise', 'dilated']:
                    # Depthwise和Dilated适度增加
                    second_layer['out_channels'] = min(int(second_layer['out_channels'] * 1.5), 192)
                else:
                    # 标准CNN保守增加
                    second_layer['out_channels'] = min(second_layer['out_channels'] + 32, 128)

                print(f"[INFO] 针对{cnn_variant}变体调整第二层通道数为: {second_layer['out_channels']}")

        # 如果原始CNN层数较少，添加额外的处理层
        if len(cnn_layers) <= 2:
            additional_layer = {
                'out_channels': fusion_out_channels // 2,
                'kernel_size': 3,
                'activation': 'relu',
                'pool': 'max' if cnn_variant != 'tcn' else None,
                'pool_kernel_size': 2
            }
            cnn_layers.append(additional_layer)
            print(f"[INFO] 为小波特征处理添加额外CNN层，输出通道: {additional_layer['out_channels']}")

    elif wavelet_enabled and expansion_factor == 1:
        print(f"[INFO] 小波变换启用但无维度扩展 (take={_safe_get(wavelet_cfg, 'take', 'approx')})")

    # 构建模型
    model = CNNLSTMAttentionModel(
        num_features=input_size,
        cnn_layers=cnn_layers,
        use_batchnorm=cnn_config['use_batchnorm'],
        cnn_dropout=cnn_config['dropout'],
        # LSTM配置
        lstm_hidden=lstm_config['hidden_size'],
        lstm_layers=lstm_config['num_layers'],
        bidirectional=lstm_config['bidirectional'],
        lstm_dropout=lstm_config['dropout'],
        rnn_type=lstm_config['rnn_type'],
        # 注意力配置
        attn_enabled=attention_config['enabled'],
        attn_heads=attention_config['num_heads'],
        attn_dropout=attention_config['dropout'],
        attn_variant=attention_config['variant'],
        attn_add_pos_enc=attention_config['add_pos_enc'],
        attn_positional_mode=attention_config['positional_mode'],
        multiscale_scales=attention_config['multiscale_scales'],
        multiscale_fuse=attention_config['multiscale_fuse'],
        local_window_size=attention_config['local_window_size'],
        local_dilation=attention_config['local_dilation'],
        st_mode=attention_config['st_mode'],
        st_fuse=attention_config['st_fuse'],
        # 通道注意力配置
        cnn_use_channel_attention=channel_attention_config['use_channel_attention'],
        cnn_channel_attention_type=channel_attention_config['channel_attention_type'],
        # 其他配置
        cnn_variant=cnn_config['variant'],
        fc_hidden=_safe_get(m, 'fc_hidden', 128),
        forecast_horizon=_safe_get(m, 'forecast_horizon', 1),
        n_targets=n_targets,
        # 高级特性
        normalization=_safe_get(m, 'normalization', None),
        decomposition=_safe_get(m, 'decomposition', None),
    )

    return model


def prepare_run(cfg):
    # 数据加载(一次读取,后续可复用数据数组)
    if cfg.data.data_path is None:
        raise ValueError("Please provide data.data_path in config pointing to CSV/NPZ/NPY")
    
    # Enhanced data loading with column information
    data, column_names = load_array_from_path(cfg.data.data_path, return_columns=True)

    # DataLoader 性能参数(根据设备/用户配置推断)
    use_cuda = torch.cuda.is_available()
    pin_memory = True if use_cuda else False
    # 若用户已在 cfg.data.num_workers 指定,则尊重;否则自动给个较合理默认
    num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else (os.cpu_count() or 0)
    persistent_workers = True if (num_workers and num_workers > 0) else False
    prefetch_factor = 2 if (num_workers and num_workers > 0) else None

    # 将 cfg 传递给 data_preprocessor 以便读取 wavelet 配置
    try:
        # 附加一个一次性属性供 create_dataloaders 读取
        create_dataloaders.__caller_cfg__ = cfg  # type: ignore[attr-defined]
    except Exception:
        pass
    
    # Enhanced dataloader creation with smart feature selection
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
        # Enhanced parameters
        feature_names=getattr(cfg.data, 'feature_names', None),
        target_names=getattr(cfg.data, 'target_names', None),
        column_names=column_names,
        auto_detect_features=getattr(cfg.data, 'auto_detect_features', True),
    )
    
    try:
        delattr(create_dataloaders, '__caller_cfg__')  # 清理
    except Exception:
        pass

    model = build_model_with_data(cfg, input_size, n_targets)
    return data, model, (train_loader, val_loader, test_loader)


def run(cfg, resume_path: Optional[str] = None):
    setup_env(cfg)
    data, model, (train_loader, val_loader, test_loader) = prepare_run(cfg)

    trainer = Trainer(model, cfg)
    if resume_path:
        last_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resumed from epoch {last_epoch}")

    history = trainer.fit(train_loader, val_loader)  # 🔥 CRITICAL FIX: 移除test_loader，避免数据泄露

    # 预测与指标计算
    preds, targets = trainer.predict(test_loader)
    metrics = evaluate_model(model, test_loader, trainer.device)
    # 将训练耗时加入指标字典,便于搜索器解析
    if isinstance(history, dict) and 'train_time_sec' in history:
        metrics['train_time_sec'] = float(history['train_time_sec'])
    print({k: round(v, 6) if isinstance(v, (int, float)) else v for k, v in metrics.items()})

    # 可视化与评估
    if getattr(cfg, 'visual_enabled', True):
        # 获取可视化保存目录
        visual_dir = os.environ.get("VIS_SAVE_DIR", "image")
        os.makedirs(visual_dir, exist_ok=True)
        
        # 传统训练历史图表
        if 'train_loss' in history and 'val_loss' in history:
            plot_losses(history['train_loss'], history['val_loss'], 
                       save_path=os.path.join(visual_dir, "training_losses.png"))
            plot_losses_logscale(history['train_loss'], history['val_loss'],
                                save_path=os.path.join(visual_dir, "training_losses_log.png"))
        if 'lr' in history:
            plot_lr(history['lr'], save_path=os.path.join(visual_dir, "learning_rate.png"))
        try:
            param_count = sum(p.numel() for p in model.parameters())
            # plot_param_count expects a dictionary of parameter counts
            param_dict = {'Total Parameters': param_count}
            plot_param_count(param_dict, save_path=os.path.join(visual_dir, "parameter_count.png"))
        except Exception:
            pass

        # ===== 新增:增强的预测性能分析 =====
        # 1. 综合性能仪表盘
        try:
            # 这些函数在visualizer.py中不存在，暂时跳过
            # from visualizer import plot_prediction_performance, plot_multi_step_analysis, create_interactive_dashboard
            print("[INFO] Enhanced performance visualization functions not available, using basic visualization")
        except Exception as e:
            print(f"[WARN] Enhanced performance visualization failed: {e}")
        
        # 回退到传统可视化
        try:
            print(f"[DEBUG] Predictions shape: {preds.shape}")
            print(f"[DEBUG] Targets shape: {targets.shape}")
            
            # 展平多维数据用于可视化，确保数据为1维
            if len(preds.shape) > 1:
                preds_flat = preds.reshape(-1)  # 使用reshape(-1)更可靠
                targets_flat = targets.reshape(-1)
            else:
                preds_flat = preds
                targets_flat = targets
            
            print(f"[DEBUG] After flattening - Predictions: {preds_flat.shape}, Targets: {targets_flat.shape}")
            
            # 确保numpy数组格式
            if hasattr(preds_flat, 'numpy'):
                preds_flat = preds_flat.numpy()
            if hasattr(targets_flat, 'numpy'):
                targets_flat = targets_flat.numpy()
                
            plot_predictions(targets_flat, preds_flat, 
                           save_path=os.path.join(visual_dir, "predictions_vs_targets.png"))  # 参数顺序：targets, predictions
            residuals = preds_flat - targets_flat
            plot_residual_hist(residuals, save_path=os.path.join(visual_dir, "residuals_histogram.png"))
            # plot_multihorizon_error需要特殊的错误字典格式，暂时跳过
            # plot_multihorizon_error(preds, targets)
        except Exception as e:
            print(f"[WARN] Basic visualization failed: {e}")

        # 数据分析图表
        try:
            plot_series_distribution(data, save_path=os.path.join(visual_dir, "data_distribution.png"))
            # 计算相关矩阵
            import pandas as pd
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
                corr_matrix = df.corr().values
                plot_corr_heatmap(corr_matrix, save_path=os.path.join(visual_dir, "correlation_heatmap.png"))
        except Exception:
            pass

        # 注意力图(若启用)
        try:
            model.eval()
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(trainer.device, non_blocking=True)
                    # 检查模型是否支持return_attn参数
                    try:
                        _, attn = model(x, return_attn=True)
                        plot_attention_heatmap(attn, save_path=os.path.join(visual_dir, "attention_heatmap.png"))
                        if attn is not None:
                            plot_attention_multihead(attn, save_path=os.path.join(visual_dir, "attention_multihead.png"))
                    except TypeError:
                        # 模型不支持return_attn参数，使用普通前向传播
                        print("[INFO] Model doesn't support return_attn parameter, skipping attention visualization")
                        break
                    break  # 只处理第一个batch
        except Exception as e:
            print(f"[WARN] Attention visualization failed: {e}")

        # 输出保存信息
        print(f"[INFO] Training visualizations saved to: {visual_dir}")
        try:
            import glob
            saved_images = glob.glob(os.path.join(visual_dir, "*.png"))
            if saved_images:
                print(f"[INFO] Generated {len(saved_images)} visualization images:")
                for img in saved_images:
                    print(f"  - {os.path.basename(img)}")
            else:
                print(f"[WARN] No visualization images found in {visual_dir}")
        except Exception:
            pass

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

    # 🔥 关键修复：手动读取experiment_metadata并添加到配置中
    if args.config:
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                raw_yaml = yaml.safe_load(f)
                experiment_metadata = raw_yaml.get('experiment_metadata', {})
                if experiment_metadata:
                    setattr(cfg, 'experiment_metadata', experiment_metadata)
                    print(f"[INFO] 读取实验元数据: {experiment_metadata}")
        except Exception as e:
            print(f"[WARNING] 无法读取实验元数据: {e}")

    # 记录 YAML stem,便于下游导出命名唯一化
    try:
        if args.config:
            import os
            from pathlib import Path
            yaml_stem = Path(args.config).stem
            # 🔥 修复：清理批量训练脚本添加的后缀，保持原始YAML名称
            if yaml_stem.endswith('_patched'):
                yaml_stem = yaml_stem[:-8]  # 移除'_patched'

            # 进一步清理可能的数字后缀（如果有的话）
            import re
            yaml_stem = re.sub(r'_\d+$', '', yaml_stem)  # 移除末尾的_数字

            print(f"[DEBUG] 清理后的yaml_stem: {yaml_stem}")
            setattr(cfg, 'yaml_stem', yaml_stem)
            setattr(cfg, 'yaml_path', os.path.abspath(args.config))
    except Exception:
        pass
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.image_dir:
        cfg.visual_save_dir = args.image_dir
    if args.ckpt_dir:
        cfg.train.checkpoints.dir = args.ckpt_dir

    run(cfg, resume_path=args.resume)

