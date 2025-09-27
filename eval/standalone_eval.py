from __future__ import annotations
"""
Standalone evaluator for CNN+LSTM(+Attention) time-series forecasting models.

Independence:
- Does NOT import main.py, trainer.py, config.py or data_preprocessor.py
- Only relies on core model definition files present in the project root:
  * model_architecture.py
  * cnn_feature_extractor.py
  * lstm_processor.py
  * attention_mechanism.py

Features:
- Load checkpoint (.pt) saved by Trainer (expects keys: model_state, cfg, epoch, best_val)
- Infer model structure from checkpoint cfg (with CLI overrides)
- Load test data (CSV/NPZ/NPY), normalize (standard/minmax/none), windowize
- Evaluate metrics: MSE, MAE, RMSE, MAPE, R2
- Visualizations: predictions vs truth, residual histogram, temporal performance, attention heatmap (if available)
- Robust argument validation & helpful error messages

Usage:
  python standalone_eval.py \
    --checkpoint checkpoints/model_best.pt \
    --data data/test.csv \
    --output_dir results/ \
    [--batch_size 128 --device cuda --sequence_length 64 --horizon 3 --normalize standard \
     --feature_indices 0,1,2 --target_indices 3 --save_csv]
"""

import argparse
import json
import os
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import traceback

# Only import core model and unified utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_architecture import CNNLSTMAttentionModel
from dataProcess.data_preprocessor import load_array_from_path, validate_and_process_indices as validate_indices
from utils.metrics import calculate_regression_metrics
from utils.config_utils import load_yaml_config
from utils.system_utils import get_device, setup_environment
from utils.logging_utils import setup_centralized_logging, log_warning, log_error, log_info, log_debug, _dlog_centralized


# -----------------------------
# Debug helpers
# -----------------------------

def _dbg_enabled() -> bool:
    return (os.environ.get("EVAL_DEBUG", "0") != "0")


def _dlog(msg: str):
    """Enhanced debug logging using centralized system"""
    _dlog_centralized(msg)


def _strict_mode() -> bool:
    return (os.environ.get("EVAL_STRICT", "1") not in ("0", "false", "False", ""))

def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in (None, "0", "false", "False", "")


# -----------------------------
# IO helpers
# -----------------------------

# 使用统一的工具，删除重复的数据加载和指标计算函数
# 这些函数现在由 utils/ 模块提供

def validate_and_process_indices(
    data_shape: Tuple[int, int],
    feature_indices: Optional[List[int]] = None,
    target_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    auto_detect: bool = True
) -> Tuple[List[int], List[int], Optional[List[str]], Optional[List[str]]]:
    """使用统一的索引验证工具"""
    feature_indices, target_indices = validate_indices(
        data_shape, feature_indices, target_indices, 
        feature_names, target_names, column_names, auto_detect
    )
    
    # 处理名称解析（保持原有接口兼容性）
    resolved_feature_names = None
    resolved_target_names = None
    if column_names:
        resolved_feature_names = [column_names[i] for i in feature_indices]
        resolved_target_names = [column_names[i] for i in target_indices]
    
    return feature_indices, target_indices, resolved_feature_names, resolved_target_names


# -----------------------------
# Dataset (windowing + normalization)
# -----------------------------

class NormalizationStats:
    def __init__(self, mean=None, std=None, min=None, max=None):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max


def fit_stats(arr: np.ndarray, mode: str) -> NormalizationStats:
    mode = (mode or "none").lower()
    if mode == "none":
        return NormalizationStats()
    if mode == "standard":
        mean = arr.mean(axis=0).astype(np.float32)
        std = (arr.std(axis=0) + 1e-8).astype(np.float32)
        return NormalizationStats(mean=mean, std=std)
    if mode == "minmax":
        mn = arr.min(axis=0).astype(np.float32)
        mx = arr.max(axis=0).astype(np.float32)
        return NormalizationStats(min=mn, max=mx)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def apply_normalize(arr: np.ndarray, stats: NormalizationStats, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return arr
    if mode == "standard":
        return (arr - stats.mean) / stats.std
    if mode == "minmax":
        return (arr - stats.min) / (stats.max - stats.min + 1e-8)
    return arr


def windowize(arr: np.ndarray, seq_len: int, horizon: int, feature_idx, target_idx):
    """Return X, Y windows.
    X: (N, T, F_sel)
    Y: (N, H) if one target else (N, H, C)
    """
    N, F = arr.shape
    T = int(seq_len)
    H = int(horizon)
    if N < T + H:
        raise ValueError(f"Data too short for sequence_length={T} and horizon={H}: N={N}")
    if not feature_idx:
        feature_idx = list(range(F))
    if not target_idx:
        target_idx = [F - 1]
    out_len = N - T - H + 1
    X = np.zeros((out_len, T, len(feature_idx)), dtype=np.float32)
    if len(target_idx) == 1:
        Y = np.zeros((out_len, H), dtype=np.float32)
    else:
        Y = np.zeros((out_len, H, len(target_idx)), dtype=np.float32)
    for i in range(out_len):
        X[i] = arr[i:i+T, feature_idx]
        tgt = arr[i+T:i+T+H, target_idx]
        if tgt.ndim == 1 or tgt.shape[1] == 1:
            Y[i] = tgt.reshape(H)
        else:
            Y[i] = tgt
    return X, Y


# -----------------------------
# Metrics
# -----------------------------

def _to_np(a: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a


def regression_metrics(preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray) -> dict:
    """使用统一的回归指标计算工具"""
    return calculate_regression_metrics(preds, targets)



# -----------------------------
# Robust casting helpers
# -----------------------------

def _as_int(val, default: Optional[int] = None) -> Optional[int]:
    try:
        if val is not None:
            return int(val)
        if default is None:
            return None
        return int(default)
    except Exception:
        return None if default is None else int(default)


def _as_float(val, default: Optional[float] = None) -> Optional[float]:
    try:
        if val is not None:
            return float(val)
        if default is None:
            return None
        return float(default)
    except Exception:
        return None if default is None else float(default)


def _as_bool(val, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return bool(default)


def _to_int_list(val):
    """将任意列表/元组转换为 int 列表;过滤 None/非法项;若无有效项返回 None."""
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                if x is None:
                    continue
                out.append(int(x))
            except Exception:
                continue
        return out if len(out) > 0 else None
    return None

# -----------------------------
# Visualization (minimal, self-contained)
# -----------------------------


def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _inverse_value(val: float, feature_idx: int, stats: NormalizationStats, mode: str) -> float:
    mode = (mode or "none").lower()
    if mode == "standard" and stats.mean is not None and stats.std is not None:
        return float(val * stats.std[feature_idx] + stats.mean[feature_idx])
    if mode == "minmax" and stats.min is not None and stats.max is not None:
        return float(val * (stats.max[feature_idx] - stats.min[feature_idx]) + stats.min[feature_idx])
    return float(val)



# -----------------------------
# Checkpoint & Model construction
# -----------------------------

def _extract_arch_meta(cfg_dict: dict) -> dict:
    m = cfg_dict.get("model", {}) or {}
    cnn = m.get("cnn", {}) or {}
    tcn = m.get("tcn", {}) or {}
    lstm = m.get("lstm", {}) or {}
    attn = m.get("attention", {}) or {}

    # 判断是否启用 TCN:显式 variant=tcn 或 tcn.enabled=True
    cnn_variant = ((cnn.get("variant") or "standard")).lower()
    tcn_enabled = _as_bool(tcn.get("enabled"), False)
    if tcn_enabled:
        cnn_variant = "tcn"

    attn_variant = ((attn.get("variant") or "standard")).lower()
    rnn_type = (str(lstm.get("rnn_type", "lstm")).lower() if lstm.get("rnn_type") is not None else "lstm")

    return {
        "cnn_variant": cnn_variant,
        "tcn_enabled": bool(tcn_enabled),
        "attn_variant": attn_variant,
        "attn_enabled": _as_bool(attn.get("enabled"), False),
        "rnn_type": rnn_type,
        "multiscale_scales": attn.get("multiscale_scales", [1, 2]),
        "multiscale_fuse": attn.get("multiscale_fuse", "sum"),
        "fc_hidden": _as_int(m.get("fc_hidden"), 128),
        "forecast_horizon": _as_int(m.get("forecast_horizon"), 1),
    }


def _strict_get(d: dict, key: str, typename: str):
    if key not in d:
        raise ValueError(f"缺少必需字段: {key}")
    val = d[key]
    if val is None:
        raise ValueError(f"字段 {key} 不能为 None(需与训练时一致)")
    return val


def _strict_int(d: dict, key: str) -> int:
    val = _strict_get(d, key, "int")
    try:
        return int(val)
    except Exception:
        raise ValueError(f"字段 {key} 需要为整数,但得到: {val}")


def _strict_bool(d: dict, key: str) -> bool:
    val = _strict_get(d, key, "bool")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    raise ValueError(f"字段 {key} 需要为布尔,但得到: {val}")


def _strict_list(d: dict, key: str):
    val = _strict_get(d, key, "list")
    if not isinstance(val, list):
        raise ValueError(f"字段 {key} 需要为列表,但得到: {type(val)}")
    return val


def _validate_layers_strict(layers, required_keys, path: str):
    if not layers:
        raise ValueError(f"{path} 至少需要一层配置")
    for i, l in enumerate(layers):
        if not isinstance(l, dict):
            raise ValueError(f"{path}[{i}] 需要为字典,但得到: {type(l)}")
        for k in required_keys:
            if l.get(k, None) is None:
                raise ValueError(f"{path}[{i}] 缺少必需字段: {k}")


def build_model_from_cfg_strict(cfg_dict: Dict, input_size: int, n_targets: int) -> CNNLSTMAttentionModel:
    """严格从 checkpoint 的 cfg 重建训练时完全一致的模型结构.
    - 不使用默认值,不做降级;字段缺失/None 直接抛错
    - CNN 变体:standard/depthwise/dilated
    - TCN:variant=tcn 或 tcn.enabled=True 时使用 tcn 配置
    - RNN:lstm/gru
    - 注意力:standard/multiscale(multiscale 需提供 scales/fuse)
    """
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint 中 cfg 不是字典或为空,请保存完整配置后再评估")

    m = _strict_get(cfg_dict, "model", "dict")
    if not isinstance(m, dict):
        raise ValueError("cfg['model'] 必须为字典")

    # 基本结构字段
    fc_hidden = _strict_int(m, "fc_hidden")
    forecast_horizon = _strict_int(m, "forecast_horizon")

    # 子模块配置
    cnn = _strict_get(m, "cnn", "dict")
    if not isinstance(cnn, dict):
        raise ValueError("cfg['model']['cnn'] 必须为字典")
    lstm = _strict_get(m, "lstm", "dict")
    if not isinstance(lstm, dict):
        raise ValueError("cfg['model']['lstm'] 必须为字典")
    attn = _strict_get(m, "attention", "dict")
    if not isinstance(attn, dict):
        raise ValueError("cfg['model']['attention'] 必须为字典")

    cnn_variant = _strict_get(cnn, "variant", "str").lower()

    # 初始化变量
    base_layers = []
    use_batchnorm = True
    dropout = 0.0

    if cnn_variant == "tcn":
        tcn = _strict_get(m, "tcn", "dict")
        if not isinstance(tcn, dict):
            raise ValueError("cfg['model']['tcn'] 必须为字典,且需包含 TCN 层配置")
        layers = _strict_list(tcn, "layers")
        _validate_layers_strict(layers, ["out_channels", "kernel_size", "dilation", "activation"], "cfg.model.tcn.layers")
        base_layers = layers
        use_batchnorm = _strict_bool(tcn, "use_batchnorm")
        dropout = float(_strict_get(tcn, "dropout", "float"))
    else:
        # CNN 变体
        layers = _strict_list(cnn, "layers")
        req = ["out_channels", "kernel_size"]
        if cnn_variant == "dilated":
            # 要求每层明确提供 dilation_rates
            req.append("dilation_rates")
        _validate_layers_strict(layers, req, "cfg.model.cnn.layers")
        base_layers = layers
        use_batchnorm = _strict_bool(cnn, "use_batchnorm")
        dropout = float(_strict_get(cnn, "dropout", "float"))

    # RNN
    rnn_type = _strict_get(lstm, "rnn_type", "str").lower()
    lstm_hidden = _strict_int(lstm, "hidden_size")
    lstm_layers = _strict_int(lstm, "num_layers")
    bidirectional = _strict_bool(lstm, "bidirectional")
    lstm_dropout = float(_strict_get(lstm, "dropout", "float"))

    # Attention
    attn_enabled = _strict_bool(attn, "enabled")
    attn_variant = _strict_get(attn, "variant", "str").lower()
    attn_heads = _strict_int(attn, "num_heads")
    attn_dropout = float(_strict_get(attn, "dropout", "float"))
    # 位置编码字段名兼容两种拼写,但必须提供其中之一
    if "add_positional_encoding" in attn:
        attn_add_pos_enc = _strict_bool(attn, "add_positional_encoding")
    elif "add_posional_encoding" in attn:
        attn_add_pos_enc = _strict_bool(attn, "add_posional_encoding")
    else:
        raise ValueError("attention 缺少 add_positional_encoding 字段(或旧拼写 add_posional_encoding)")
    # 其他注意力相关可选字段(存在则读取,不存在使用默认)
    attn_pos_mode = str(attn.get("positional_mode", "none")).lower()
    local_window_size = int(attn.get("local_window_size", 64))
    local_dilation = int(attn.get("local_dilation", 1))
    st_mode = str(attn.get("st_mode", "serial"))
    st_fuse = str(attn.get("st_fuse", "sum"))

    multiscale_scales = None
    multiscale_fuse = None
    if attn_variant == "multiscale":
        multiscale_scales = _strict_list(attn, "multiscale_scales")
        if not all(isinstance(x, int) for x in multiscale_scales):
            raise ValueError("multiscale_scales 需要为整数列表")
        multiscale_fuse = _strict_get(attn, "multiscale_fuse", "str")

    # CNN 通道注意力(可选)
    ca_on = bool(cnn.get("use_channel_attention", False))
    ca_type = str(cnn.get("channel_attention_type", "eca")).lower()

    # 可选归一化与趋势分解配置(若存在则按原样传入以对齐权重)
    normalization_cfg = m.get("normalization", None)
    decomposition_cfg = m.get("decomposition", None)

    # 构建模型(严格模式:完全按 cfg).若 TCN 配置写在 cnn.layers(老格式)且 cnn_variant==tcn,会在前面严格校验环节已抛错.
    model = CNNLSTMAttentionModel(
        num_features=int(input_size),
        cnn_layers=[(l if isinstance(l, dict) else dict(l)) for l in base_layers],
        use_batchnorm=use_batchnorm,
        cnn_dropout=float(dropout),
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        attn_enabled=attn_enabled,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        fc_hidden=fc_hidden,
        forecast_horizon=forecast_horizon,
        n_targets=int(n_targets),
        attn_add_pos_enc=attn_add_pos_enc,
        lstm_dropout=lstm_dropout,
        cnn_variant=cnn_variant,
        attn_variant=attn_variant,
        multiscale_scales=multiscale_scales if multiscale_scales is not None else [1],
        multiscale_fuse=multiscale_fuse if multiscale_fuse is not None else "sum",
        # 新增:对齐训练期的可选参数,避免严格加载时出现 Unexpected key
        attn_positional_mode=attn_pos_mode,
        local_window_size=local_window_size,
        local_dilation=local_dilation,
        st_mode=st_mode,
        st_fuse=st_fuse,
        cnn_use_channel_attention=ca_on,
        cnn_channel_attention_type=ca_type,
        normalization=normalization_cfg,
        decomposition=decomposition_cfg,
    )
    # 设置 RNN 类型
    model.rnn_type = rnn_type
    return model


# -----------------------------
# Main evaluation routine
# -----------------------------

def evaluate(checkpoint: str, data_path: str, output_dir: str,
             device: str = "cuda", batch_size: int = 128,
             sequence_length: Optional[int] = None,
             horizon: Optional[int] = None,
             normalize: Optional[str] = None,
             feature_indices=None,
             target_indices=None,
             save_csv: bool = False,
             feature_names=None,
             target_names=None,
             auto_detect_features: bool = True,
             return_predictions: bool = False,
             enable_advanced_viz: bool = True,
             save_attention: bool = True,
             outlier_strategy: str = "iqr"):
    ensure_dir(output_dir)
    
    # Setup centralized logging
    log_path = os.path.join(output_dir, "result.log")
    setup_centralized_logging(log_file_path=log_path, capture_warnings=True)
    log_info(f"Starting evaluation: {checkpoint}")

    # Load checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    _dlog(f"Loaded ckpt keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    _dlog(f"cfg_dict_type={type(cfg_dict)} model_keys={list((cfg_dict.get('model') or {}).keys()) if isinstance(cfg_dict, dict) else 'N/A'} data_keys={list((cfg_dict.get('data') or {}).keys()) if isinstance(cfg_dict, dict) else 'N/A'}")

    # Enhanced data loading with column information
    arr, column_names = load_array_from_path(data_path, return_columns=True)

    # Infer preprocessing params from checkpoint cfg with CLI override
    data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    seq_len = _as_int(sequence_length, _as_int(data_cfg.get("sequence_length"), 64))
    model_cfg = cfg_dict.get("model", {}) if isinstance(cfg_dict, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    hz = _as_int(horizon, _as_int(model_cfg.get("forecast_horizon"), _as_int(data_cfg.get("horizon"), 1)))
    norm = (normalize if normalize is not None else data_cfg.get("normalize", "standard"))
    _dlog(f"seq_len={seq_len} horizon={hz} normalize={norm}")

    # Enhanced feature/target selection with intelligent defaults
    feat_idx = feature_indices
    if feat_idx is None:
        fi = data_cfg.get("feature_indices", None)
        feat_idx = _to_int_list(fi)
    
    targ_idx = target_indices
    if targ_idx is None:
        ti = data_cfg.get("target_indices", None)
        targ_idx = _to_int_list(ti)
    
    # Enhanced: Use intelligent feature selection if nothing specified
    if (feat_idx is None and targ_idx is None and 
        feature_names is None and target_names is None and
        auto_detect_features and column_names):
        _dlog("Using intelligent auto-detection for features and targets")
        feat_idx, targ_idx, resolved_feature_names, resolved_target_names = validate_and_process_indices(
            data_shape=arr.shape,
            feature_indices=feat_idx,
            target_indices=targ_idx,
            feature_names=feature_names,
            target_names=target_names,
            column_names=column_names,
            auto_detect=True
        )
        # Update feature_names for downstream use
        if resolved_feature_names:
            feature_names = resolved_feature_names
    elif (feature_names or target_names) and column_names:
        # Name-based selection
        _dlog("Using name-based feature selection")
        feat_idx, targ_idx, resolved_feature_names, resolved_target_names = validate_and_process_indices(
            data_shape=arr.shape,
            feature_indices=feat_idx,
            target_indices=targ_idx,
            feature_names=feature_names,
            target_names=target_names,
            column_names=column_names,
            auto_detect=False
        )
        # Update feature_names for downstream use
        if resolved_feature_names:
            feature_names = resolved_feature_names
    else:
        # Traditional fallback
        if targ_idx is None:
            # Align with training default: when target_indices is None, use LAST feature as target
            targ_idx = [arr.shape[1] - 1]

    _dlog(f"Final indices: feat={feat_idx} targ={targ_idx}")

    # Fit stats on test data (注意:理想情况下应使用训练时的标准化参数,但多数checkpoint未保存)
    # 修复：检查是否有模型特定的归一化统计量传入
    custom_seed = os.environ.get('EVAL_CUSTOM_SEED')
    if custom_seed and custom_seed.isdigit():
        custom_mean_key = f'EVAL_CUSTOM_MEAN_{custom_seed}'
        custom_std_key = f'EVAL_CUSTOM_STD_{custom_seed}'
        
        if custom_mean_key in os.environ and custom_std_key in os.environ:
            try:
                # 使用传入的模型特定统计量
                mean_vals = np.array([float(x) for x in os.environ[custom_mean_key].split(',')], dtype=np.float32)
                std_vals = np.array([float(x) for x in os.environ[custom_std_key].split(',')], dtype=np.float32)
                stats = NormalizationStats(mean=mean_vals, std=std_vals, min=None, max=None)
                _dlog(f"[INFO] Using model-specific normalization stats from batch_eval (seed={custom_seed})")
                _dlog(f"[INFO] Custom mean range: {mean_vals.min():.6f}~{mean_vals.max():.6f}")
                _dlog(f"[INFO] Custom std range: {std_vals.min():.6f}~{std_vals.max():.6f}")
            except Exception as e:
                _dlog(f"[WARN] Failed to parse custom stats: {e}, falling back to test data stats")
                stats = fit_stats(arr, norm)
        else:
            _dlog(f"[WARN] Custom seed provided but stats not found, falling back to test data stats")
            stats = fit_stats(arr, norm)
    else:
        # 为了一致性，我们记录这个潜在问题并继续使用测试数据统计量
        _dlog(f"data_shape={arr.shape} first_row_sample={arr[0][:5] if arr.size>0 else 'EMPTY'}")
        
        # 改进：尝试从checkpoint获取训练时的归一化统计量
        train_stats_available = False
        if 'data_stats' in ckpt and isinstance(ckpt['data_stats'], dict):
            try:
                train_mean = ckpt['data_stats'].get('mean')
                train_std = ckpt['data_stats'].get('std')
                if train_mean is not None and train_std is not None:
                    stats = NormalizationStats(
                        mean=np.array(train_mean, dtype=np.float32), 
                        std=np.array(train_std, dtype=np.float32)
                    )
                    train_stats_available = True
                    _dlog(f"[INFO] Using training normalization stats from checkpoint")
            except Exception as e:
                _dlog(f"[WARN] Failed to load training stats from checkpoint: {e}")
        
        if not train_stats_available:
            _dlog(f"[WARNING] Using test data statistics for normalization. This may cause inconsistency with training.")
            stats = fit_stats(arr, norm)
            
            # 验证统计量的合理性
            if norm == "standard" and stats.mean is not None and stats.std is not None:
                mean_range = np.ptp(stats.mean)  # peak-to-peak
                std_range = np.ptp(stats.std)
                if mean_range < 1e-8 and std_range < 1e-8:
                    _dlog("[WARNING] Normalization statistics appear degenerate - this may cause zero predictions")
                elif np.any(stats.std < 1e-8):
                    _dlog("[WARNING] Some standard deviations are near zero - this may cause numerical issues")
            elif norm == "minmax" and stats.min is not None and stats.max is not None:
                ranges = stats.max - stats.min
                if np.any(ranges < 1e-8):
                    _dlog("[WARNING] Some feature ranges are near zero - this may cause numerical issues")
    arr_norm = apply_normalize(arr, stats, norm)

    # Optional: Wavelet transform (align with training)
    def _maybe_get(dct, *keys, default=None):
        cur = dct
        try:
            for k in keys:
                if cur is None:
                    return default
                if isinstance(cur, dict):
                    cur = cur.get(k)
                else:
                    return default
            return cur if cur is not None else default
        except Exception:
            return default

    wave_cfg = _maybe_get(data_cfg, 'wavelet', default=None)
    if wave_cfg is None:
        wave_cfg = _maybe_get(cfg_dict, 'data', 'wavelet', default=None)
    _wv_enabled = False
    try:
        _wv_enabled = bool(_maybe_get(wave_cfg or {}, 'enabled', default=False))
    except Exception:
        _wv_enabled = False

    if _wv_enabled:
        try:
            import pywt  # type: ignore
            wavelet = str(_maybe_get(wave_cfg, 'wavelet', default='db4'))
            level = int(_maybe_get(wave_cfg, 'level', default=3))
            mode = str(_maybe_get(wave_cfg, 'mode', default='symmetric'))
            take = str(_maybe_get(wave_cfg, 'take', default='all')).lower()
            
            # 改进的小波模式判断：基于配置而非文件名
            # 检查配置中是否明确禁用小波变换
            should_skip_wavelet = False
            try:
                # 检查配置中的小波设置
                if isinstance(wave_cfg, dict):
                    explicit_enabled = wave_cfg.get('enabled', None)
                    if explicit_enabled is False:
                        should_skip_wavelet = True
                        _dlog(f"Skipping wavelet: explicitly disabled in config")
                    elif take == 'none':
                        should_skip_wavelet = True
                        _dlog(f"Skipping wavelet: take='none' specified")
                        
                # 如果仍然启用，检查文件名作为次要判断
                if not should_skip_wavelet:
                    checkpoint_name = os.path.basename(checkpoint) if isinstance(checkpoint, str) else ""
                    if 'wavelet-off' in checkpoint_name.lower() or 'wav-off' in checkpoint_name.lower():
                        should_skip_wavelet = True
                        _dlog(f"Skipping wavelet based on filename: {checkpoint_name}")
                        
            except Exception:
                pass
                
            if should_skip_wavelet:
                _wv_enabled = False
            else:
                _dlog(f"Applying wavelet transform: wavelet={wavelet}, level={level}, mode={mode}, take={take}")
                N, F = arr_norm.shape
                outs = []
                for f in range(F):
                    xcol = arr_norm[:, f]
                    coeffs = pywt.wavedec(xcol, wavelet=wavelet, level=level, mode=mode)
                    if take == 'approx':
                        sel = [coeffs[0]]
                    elif take == 'details':
                        sel = coeffs[1:]
                    else:
                        sel = coeffs
                    parts = []
                    for c in sel:
                        xi = np.linspace(0, len(c) - 1, num=len(c))
                        xN = np.linspace(0, len(c) - 1, num=N)
                        parts.append(np.interp(xN, xi, c))
                    band = np.stack(parts, axis=1)
                    outs.append(band)
                arr_norm = np.concatenate(outs, axis=1).astype(np.float32)
                _dlog(f"wavelet enabled -> features expanded to {arr_norm.shape[1]}")
        except Exception as e:
            _dlog(f"[WARN] wavelet requested but failed: {e}")

    # 智能通道数自动修复：如果输入通道数与模型期望不匹配，尝试自动修复
    try:
        # 从权重快速推断期望的输入通道数
        exp_in_channels_quick = None
        state_quick = ckpt.get('model_state', {}) if isinstance(ckpt, dict) else {}
        if isinstance(state_quick, dict):
            for _k, _v in state_quick.items():
                try:
                    if isinstance(_v, torch.Tensor) and _v.ndim == 3 and _k.endswith('.weight'):
                        c_in = int(_v.shape[1])
                        if exp_in_channels_quick is None or c_in > exp_in_channels_quick:
                            exp_in_channels_quick = c_in
                except Exception:
                    continue
        
        cur_in_channels = int(arr_norm.shape[1])
        
        # 如果通道数不匹配，尝试智能修复
        if exp_in_channels_quick is not None and cur_in_channels != exp_in_channels_quick:
            _dlog(f"Channel mismatch detected: current={cur_in_channels}, expected={exp_in_channels_quick}")
            
            # 情况1: 当前通道数少于期望，且为整倍数关系，尝试小波展开
            if (cur_in_channels < exp_in_channels_quick and 
                exp_in_channels_quick % cur_in_channels == 0 and
                not _wv_enabled):
                
                expansion_factor = exp_in_channels_quick // cur_in_channels
                if expansion_factor <= 8:  # 合理的扩展倍数
                    try:
                        import pywt
                        level_guess = max(1, expansion_factor - 1)
                        wavelet = str(_maybe_get(wave_cfg or {}, 'wavelet', default='db4'))
                        mode = str(_maybe_get(wave_cfg or {}, 'mode', default='symmetric'))
                        
                        _dlog(f"Auto-applying wavelet to match channels: factor={expansion_factor}, level={level_guess}")
                        
                        N, F = arr_norm.shape
                        outs = []
                        for f in range(F):
                            xcol = arr_norm[:, f]
                            coeffs = pywt.wavedec(xcol, wavelet=wavelet, level=level_guess, mode=mode)
                            # 取全部系数以达到期望的扩展倍数
                            sel = coeffs
                            parts = []
                            for c in sel:
                                xi = np.linspace(0, len(c) - 1, num=len(c))
                                xN = np.linspace(0, len(c) - 1, num=N)
                                parts.append(np.interp(xN, xi, c))
                            outs.append(np.stack(parts, axis=1))
                        
                        arr_norm_expanded = np.concatenate(outs, axis=1).astype(np.float32)
                        
                        # 如果扩展后的通道数匹配或接近，采用
                        if arr_norm_expanded.shape[1] == exp_in_channels_quick:
                            arr_norm = arr_norm_expanded
                            _wv_enabled = True
                            _dlog(f"Auto-expansion successful: {cur_in_channels} -> {arr_norm.shape[1]}")
                        elif abs(arr_norm_expanded.shape[1] - exp_in_channels_quick) < abs(cur_in_channels - exp_in_channels_quick):
                            # 即使不完全匹配，如果更接近就采用
                            arr_norm = arr_norm_expanded
                            _wv_enabled = True
                            _dlog(f"Auto-expansion partial success: {cur_in_channels} -> {arr_norm.shape[1]} (target: {exp_in_channels_quick})")
                            
                    except Exception as ex:
                        _dlog(f"Auto-expansion failed: {ex}")
                        
            # 情况2: 当前通道数多于期望，尝试特征选择
            elif cur_in_channels > exp_in_channels_quick and feat_idx is None:
                _dlog(f"Auto-selecting first {exp_in_channels_quick} features to match model")
                feat_idx = list(range(exp_in_channels_quick))
                arr_norm = arr_norm[:, feat_idx]
                
    except Exception as e:
        _dlog(f"[WARN] Auto channel repair failed: {e}")
    # Windowize
    _dlog(f"feature_idx={feat_idx} target_idx={targ_idx}")
    X_np, Y_np = windowize(arr_norm, seq_len, hz,
                           feat_idx if feat_idx is not None else list(range(arr_norm.shape[1])),
                           targ_idx if targ_idx is not None else [arr_norm.shape[1]-1])
    _dlog(f"windowized X={X_np.shape} Y={Y_np.shape}")

    # 读取期望的 IO 形状(从权重中推断),与数据/配置进行严格对齐
    state = ckpt.get("model_state", {})
    exp_in_channels = None
    exp_out_dim = None
    if isinstance(state, dict):
        # 推断输入通道:寻找第一个 conv1d 权重(形状为 [out_c, in_c, k])
        # 🔥 CRITICAL FIX: 应该找第一层的输入通道数，而不是最大的
        # 按layer顺序找第一个卷积层的输入通道数
        for k, v in state.items():
            try:
                if isinstance(v, torch.Tensor) and v.ndim == 3 and k.endswith(".weight"):
                    # 查找第一层：cnn.net.0.0.weight 或类似模式
                    if ('cnn.net.0.0' in k or 'cnn.layers.0' in k or 
                        k.startswith('cnn.') and '.0.' in k and 'weight' in k):
                        exp_in_channels = int(v.shape[1])
                        break  # 找到第一层就停止
            except Exception:
                continue
        
        # 如果上面的模式匹配失败，作为备用方案，找最小的输入通道数（通常是第一层）
        if exp_in_channels is None:
            min_in_channels = None
            for k, v in state.items():
                try:
                    if isinstance(v, torch.Tensor) and v.ndim == 3 and k.endswith(".weight"):
                        c_in = int(v.shape[1])
                        if min_in_channels is None or c_in < min_in_channels:
                            min_in_channels = c_in
                except Exception:
                    continue
            exp_in_channels = min_in_channels
        # 推断输出维度:寻找 head.*.weight 的二维矩阵,取“最后一层 head 的 out_features”
        head_out_dim = None
        head_max_idx = -1
        for k, v in state.items():
            try:
                if isinstance(v, torch.Tensor) and v.ndim == 2 and k.startswith("head") and k.endswith(".weight"):
                    # 解析 head.<idx>.weight 的 idx,选择最大的 idx 作为最终输出层
                    parts = k.split('.')
                    idx = -1
                    if len(parts) >= 3 and parts[0] == 'head':
                        try:
                            idx = int(parts[1])
                        except Exception:
                            idx = -1
                    if idx >= head_max_idx:
                        head_max_idx = idx
                        head_out_dim = int(v.shape[0])
            except Exception:
                continue
        exp_out_dim = head_out_dim
    # 若仍未能推断到 exp_in_channels,再进行一次宽松扫描(不强制 .weight 后缀)
    if exp_in_channels is None and isinstance(state, dict):
        try:
            cand = None
            for _k, _v in state.items():
                try:
                    if isinstance(_v, torch.Tensor) and _v.ndim == 3:
                        c_in = int(_v.shape[1])
                        if cand is None or c_in > cand:
                            cand = c_in
                except Exception:
                    continue
            exp_in_channels = cand
            if exp_in_channels is not None:
                _dlog(f"exp_in_channels (loose scan) = {exp_in_channels}")
        except Exception:
            pass


    # 构建模型(严格一致).若 cfg 信息缺失,将抛错并停止评估.
    input_size = int(X_np.shape[2])
    n_targets = int(1 if Y_np.ndim == 2 else Y_np.shape[2])
    _dlog(f"input_size={input_size} n_targets={n_targets} exp_in={exp_in_channels} exp_out={exp_out_dim}")

    # 严格校验:数据特征数与训练时一致;输出维度与 horizon*n_targets 一致
    if _strict_mode():
        if exp_in_channels is not None and input_size != exp_in_channels:
            # 可选:自动对齐输入特征数(仅在特征多于训练且未显式提供 feature_indices 时)
            if (
                input_size > exp_in_channels and
                feature_indices is None and
                os.environ.get('EVAL_AUTO_ALIGN_INPUT', '').lower() in ('1','true','yes','on')
            ):
                _dlog(f"auto-align input features: take first {exp_in_channels} of {input_size}")
                feat_idx = list(range(int(exp_in_channels)))
                X_np, Y_np = windowize(arr_norm, seq_len, hz,
                                       feat_idx,
                                       targ_idx if targ_idx is not None else [arr_norm.shape[1]-1])
                input_size = int(X_np.shape[2])
            # 新增:当特征少于训练值且为整倍数,自动进行小波展开补足通道(不再要求 _wv_enabled 为 False)
            elif (
                exp_in_channels is not None and
                input_size < exp_in_channels and
                exp_in_channels % max(1, int(input_size)) == 0 and
                os.environ.get('EVAL_WAVELET_AUTO_EXPAND', '1').lower() in ('1','true','yes','on')
            ):
                try:
                    ratio = int(exp_in_channels) // int(input_size)
                    level_guess = max(1, ratio - 1)
                    wavelet = str(_maybe_get(wave_cfg or {}, 'wavelet', default='db4'))
                    mode = str(_maybe_get(wave_cfg or {}, 'mode', default='symmetric'))
                    _dlog(f"auto-expand by wavelet: input_size={input_size}, expected={exp_in_channels}, ratio={ratio}, level~{level_guess}")
                    import pywt  # type: ignore
                    N, F = arr_norm.shape
                    outs = []
                    for f in range(F):
                        xcol = arr_norm[:, f]
                        coeffs = pywt.wavedec(xcol, wavelet=wavelet, level=level_guess, mode=mode)
                        # 取全部系数(近似+细节),与 ratio=level+1 对齐
                        sel = coeffs
                        parts = []
                        for c in sel:
                            xi = np.linspace(0, len(c) - 1, num=len(c))
                            xN = np.linspace(0, len(c) - 1, num=N)
                            parts.append(np.interp(xN, xi, c))
                        outs.append(np.stack(parts, axis=1))
                    arr_norm = np.concatenate(outs, axis=1).astype(np.float32)
                    _wv_enabled = True
                    # 重新窗口化
                    X_np, Y_np = windowize(
                        arr_norm, seq_len, hz,
                        feat_idx if feat_idx is not None else list(range(arr_norm.shape[1])),
                        targ_idx if targ_idx is not None else [arr_norm.shape[1]-1]
                    )
                    input_size = int(X_np.shape[2])
                    _dlog(f"auto-expand applied -> input_size={input_size}")
                except Exception as _we:
                    _dlog(f"[WARN] auto-expand by wavelet failed: {_we}")
            if exp_in_channels is not None and input_size != exp_in_channels:
                raise RuntimeError(
                    f"输入特征数与训练时不一致: now={input_size}, expected={exp_in_channels}. "
                    f"请使用与训练相同的特征列(或设置相同的 feature_indices),并确保数据预处理一致.")
        hz_check = _as_int(horizon, None) or _as_int(cfg_dict.get("model", {}).get("forecast_horizon"), None)
        if exp_out_dim is not None and hz_check is not None:
            expected_targets = exp_out_dim // int(hz_check) if exp_out_dim % int(hz_check) == 0 else None
            cur_out_dim = int(n_targets) * int(hz_check)
            if expected_targets is None or cur_out_dim != exp_out_dim:
                # 可选1:仅对齐目标列个数(在 horizon 与 exp_out_dim 可整除时)
                if _env_true("EVAL_AUTO_ALIGN_TARGETS") and expected_targets is not None and target_indices is None:
                    _dlog(f"auto-align targets: take first {expected_targets} of {arr_norm.shape[1]} columns")
                    targ_idx = list(range(int(expected_targets)))
                    X_np, Y_np = windowize(
                        arr_norm, seq_len, hz_check,
                        feat_idx if feat_idx is not None else list(range(arr_norm.shape[1])),
                        targ_idx
                    )
                    n_targets = int(1 if Y_np.ndim == 2 else Y_np.shape[2])
                    cur_out_dim = int(n_targets) * int(hz_check)
                    if cur_out_dim != exp_out_dim:
                        raise RuntimeError(
                            f"自动对齐失败: 重新计算后 hz*targets={cur_out_dim}, 仍不等于 {exp_out_dim}. "
                            f"请显式提供训练时的 target_indices(长度={expected_targets}).")
                    else:
                        _dlog("auto-align OK, proceeding with strict load")
                # 可选2:自动推断 horizon 与目标列个数(当 horizon 与 exp_out_dim 不整除时)
                elif _env_true("EVAL_AUTO_INFER_HZ") and target_indices is None and exp_out_dim is not None:
                    def _divisors(n: int):
                        ds = []
                        for d in range(1, n + 1):
                            if n % d == 0:
                                ds.append(d)
                        return ds
                    cand_hz = _divisors(int(exp_out_dim))
                    # 优先使用 CLI 的 horizon 或 cfg 的 horizon(若可用且可整除并满足目标列数不超过现有列数)
                    pref = []
                    cli_hz = _as_int(horizon, None)
                    cfg_hz = _as_int(cfg_dict.get("model", {}).get("forecast_horizon"), None)
                    if cli_hz in cand_hz and (exp_out_dim // int(cli_hz)) <= arr_norm.shape[1]:
                        pref.append(int(cli_hz))
                    if cfg_hz in cand_hz and (exp_out_dim // int(cfg_hz)) <= arr_norm.shape[1]:
                        pref.append(int(cfg_hz))
                    chosen_hz = None
                    if pref:
                        chosen_hz = pref[0]
                    else:
                        # 选择一个使 n_targets <= 当前特征列数的 horizon,偏好更大的 horizon(更接近训练可能的设定)
                        for hz_c in sorted(cand_hz, reverse=True):
                            n_t = exp_out_dim // hz_c
                            if n_t <= arr.shape[1]:
                                chosen_hz = hz_c
                                break
                    if chosen_hz is None:
                        raise RuntimeError(
                            f"无法自动推断 horizon:exp_out_dim={exp_out_dim} 与可用列数={arr.shape[1]} 不匹配."
                            f"请显式提供与训练一致的 horizon 与 target_indices.")
                    expected_targets2 = exp_out_dim // int(chosen_hz)
                    _dlog(f"auto-infer horizon: choose hz={chosen_hz}, targets={expected_targets2}")
                    targ_idx = list(range(int(expected_targets2)))
                    # 覆盖运行时 horizon 与 cfg 的 forecast_horizon(仅当前评估进程内生效)
                    hz_check = int(chosen_hz)
                    if isinstance(cfg_dict.get("model"), dict):
                        cfg_dict["model"]["forecast_horizon"] = int(chosen_hz)
                    X_np, Y_np = windowize(
                        arr_norm, seq_len, hz_check,
                        feat_idx if feat_idx is not None else list(range(arr.shape[1])),
                        targ_idx
                    )
                    n_targets = int(1 if Y_np.ndim == 2 else Y_np.shape[2])
                    cur_out_dim = int(n_targets) * int(hz_check)
                    if cur_out_dim != exp_out_dim:
                        raise RuntimeError(
                            f"自动推断失败: 重新计算后 hz*targets={cur_out_dim}, 仍不等于 {exp_out_dim}. "
                            f"请显式提供训练时的 horizon 与 target_indices.")
                    else:
                        _dlog("auto-infer horizon OK, proceeding with strict load")
                else:
                    raise RuntimeError(
                        f"输出维度不一致: now(hz*targets)={cur_out_dim}, expected={exp_out_dim}. "
                        f"请确保 forecast_horizon 与目标列数与训练一致.")

    model = build_model_from_cfg_strict(cfg_dict, input_size=input_size, n_targets=n_targets)

    # Load weights
    try:
        # 严格模式:必须完全匹配
        missing = model.load_state_dict(ckpt["model_state"], strict=True)
        _dlog("Model weights loaded successfully (strict mode)")
    except RuntimeError as e:
        if _strict_mode():
            raise RuntimeError(
                "严格加载权重失败(形状不匹配).请确保评估时的数据特征数、目标列与 forecast_horizon 与训练完全一致.\n"
                f"详细信息: {e}")
        else:
            _dlog("state_dict strict load failed; attempting partial load")
            log_warning(f"Strict load failed due to shape mismatch: {e}\nAttempting partial load (excluding head)...")
            state2 = ckpt["model_state"].copy()
            to_del = [k for k in list(state2.keys()) if k.startswith("head.")]
            for k in to_del:
                del state2[k]
            # 实际执行部分加载
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state2, strict=False)
                _dlog(f"Partial load completed. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                log_info(f"Partial model weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            except Exception as partial_e:
                log_error(f"Partial load also failed: {partial_e}")
                raise RuntimeError(f"Both strict and partial weight loading failed: {e}") from partial_e
    # 最后防线:已构建并加载权重后,再次用模型权重推断期望输入通道;若仍不匹配,尝试展开
    try:
        exp_c_in_final = None
        for _n, _p in model.named_parameters():
            if hasattr(_p, 'ndim') and _p.ndim == 3 and _n.endswith('.weight'):
                c_in = int(_p.shape[1])
                if exp_c_in_final is None or c_in > exp_c_in_final:
                    exp_c_in_final = c_in
        cur_in = int(X_np.shape[2])
        if exp_c_in_final is not None and cur_in != exp_c_in_final:
            if exp_c_in_final % max(1, cur_in) == 0 and os.environ.get('EVAL_WAVELET_AUTO_EXPAND', '1').lower() in ('1','true','yes','on'):
                ratio = int(exp_c_in_final) // int(cur_in)
                level_guess = max(1, ratio - 1)
                wavelet = str(_maybe_get(wave_cfg or {}, 'wavelet', default='db4'))
                mode = str(_maybe_get(wave_cfg or {}, 'mode', default='symmetric'))
                _dlog(f"post-build auto-expand: cur_in={cur_in}, expected={exp_c_in_final}, ratio={ratio}, level~{level_guess}")
                import pywt  # type: ignore
                N, F = arr_norm.shape
                outs = []
                for f in range(F):
                    xcol = arr_norm[:, f]
                    coeffs = pywt.wavedec(xcol, wavelet=wavelet, level=level_guess, mode=mode)
                    sel = coeffs
                    parts = []
                    for c in sel:
                        xi = np.linspace(0, len(c) - 1, num=len(c))
                        xN = np.linspace(0, len(c) - 1, num=N)
                        parts.append(np.interp(xN, xi, c))
                    outs.append(np.stack(parts, axis=1))
                arr_norm = np.concatenate(outs, axis=1).astype(np.float32)
                X_np, Y_np = windowize(
                    arr_norm, seq_len, hz,
                    feat_idx if feat_idx is not None else list(range(arr_norm.shape[1])),
                    targ_idx if targ_idx is not None else [arr_norm.shape[1]-1]
                )
                _dlog(f"post-build auto-expand applied -> input_size={int(X_np.shape[2])}")
            else:
                raise RuntimeError(
                    f"输入特征数与模型权重不匹配:X_in={cur_in}, weight expects={exp_c_in_final}. 请检查是否遗漏小波展开或特征选择.")
    except Exception as _e3:
        _dlog(f"[WARN] post-build expand failed: {_e3}")

    # Device
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    model.to(dev)
    model.eval()

    # Batching inference
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    preds_list = []
    
    _dlog(f"Starting inference: X.shape={X.shape}, Y.shape={Y.shape}, device={dev}")
    _dlog(f"X stats: min={X.min().item():.6f}, max={X.max().item():.6f}, mean={X.mean().item():.6f}")
    _dlog(f"Model training mode: {model.training}")
    
    # Debug model parameters to check if they are properly loaded
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad and p.numel() > 0)
    _dlog(f"Model total parameters: {total_params}, non-zero: {nonzero_params}")
    
    if nonzero_params == 0:
        log_warning("All model parameters are zero! This indicates model loading failed.")
    
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = X[start:end].to(dev, non_blocking=True)
            
            # 添加输入检查
            if start == 0:  # 只检查第一个batch
                _dlog(f"First batch input stats: min={xb.min().item():.6f}, max={xb.max().item():.6f}")
                # 检查输入是否全为0
                if torch.all(xb == 0):
                    log_warning("Input data is all zeros! Check data preprocessing.")
                elif torch.all(torch.abs(xb) < 1e-6):
                    log_warning("Input data is near zero - this may indicate normalization issues.")
            
            # 增强预测过程检查
            try:
                out = model(xb)
            except Exception as model_error:
                log_error(f"Model forward pass failed: {model_error}")
                # 创建零预测作为fallback
                expected_shape = (xb.shape[0], hz * n_targets)
                out = torch.zeros(expected_shape, device=dev)
            
            # 添加输出检查
            if start == 0:  # 只检查第一个batch
                _dlog(f"First batch output stats: min={out.min().item():.6f}, max={out.max().item():.6f}")
                _dlog(f"First batch output shape: {out.shape}")
                
                # 检查输出是否异常
                if torch.all(out == 0):
                    log_warning("Model output is all zeros! This indicates a model prediction problem.")
                    # 尝试诊断原因
                    _dlog("Attempting model diagnostic...")
                    
                    # 检查模型是否在正确模式
                    if model.training:
                        log_warning("Model is in training mode during inference!")
                        model.eval()
                        try:
                            out_eval = model(xb)
                            if not torch.all(out_eval == 0):
                                log_info("Fixed: Setting model to eval mode resolved zero predictions")
                                out = out_eval
                        except:
                            pass
                            
                    # 检查梯度计算
                    if torch.is_grad_enabled():
                        log_warning("Gradient computation is enabled during inference!")
                        
                    # 检查设备一致性
                    model_device = next(model.parameters()).device
                    if model_device != xb.device:
                        log_warning(f"Device mismatch: model on {model_device}, input on {xb.device}")
                        
                elif torch.all(torch.isnan(out)):
                    log_warning("Model output contains NaN values!")
                elif torch.all(torch.abs(out) < 1e-8):
                    log_warning("Model output is near zero - this may indicate model issues.")
            
            preds_list.append(out.detach().cpu())
    preds = torch.cat(preds_list, dim=0)
    _dlog(f"inference done, preds_shape={tuple(preds.shape)}")
    
    # 检查并处理NaN预测
    nan_count = torch.isnan(preds).sum().item()
    if nan_count > 0:
        _dlog(f"[WARN] Found {nan_count} NaN values in predictions, replacing with target mean...")
        # 使用目标均值替换NaN预测
        target_mean = Y.mean().item() if not torch.isnan(Y.mean()) else 0.0
        preds = torch.where(torch.isnan(preds), target_mean, preds)
        _dlog(f"[INFO] Replaced NaN predictions with target mean: {target_mean:.6f}")
    
    # 检查预测范围的合理性
    pred_min, pred_max = preds.min().item(), preds.max().item()
    target_min, target_max = Y.min().item(), Y.max().item()
    _dlog(f"[INFO] Prediction range: [{pred_min:.6f}, {pred_max:.6f}]")
    _dlog(f"[INFO] Target range: [{target_min:.6f}, {target_max:.6f}]")
    
    # 如果预测值全部相同，记录警告
    if abs(pred_max - pred_min) < 1e-8:
        _dlog(f"[WARN] All predictions are identical: {pred_min:.6f}")
    
    metrics = regression_metrics(preds, Y)

    # Standardized metric keys (test_*) and backward-compatible aliases
    std_metrics = {
        'test_loss': float(metrics.get('mse', 0.0)),  # treat loss as MSE by default
        'test_mse': float(metrics.get('mse', 0.0)),
        'test_mae': float(metrics.get('mae', 0.0)),
        'test_rmse': float(metrics.get('rmse', 0.0)) if 'rmse' in metrics else float(np.sqrt(max(metrics.get('mse', 0.0), 0))),
        'test_r2': float(metrics.get('r2', 0.0)),
    }
    # Keep original keys for backward compatibility
    merged_metrics = {**metrics, **std_metrics}

    # Print metrics
    print(json.dumps({k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 6)) for k, v in merged_metrics.items()}, ensure_ascii=False))

    # 高级可视化系统 (新增)
    if enable_advanced_viz:
        try:
            # 导入高级可视化模块
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
            from visualization.advanced_evaluation_visualizer import create_advanced_visualizer
            
            log_info("正在生成高级评估可视化...")
            visualizer = create_advanced_visualizer(outlier_strategy=outlier_strategy)
            
            # 准备数据
            pred_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds
            target_np = Y.detach().cpu().numpy() if torch.is_tensor(Y) else Y
            
            # 创建时间戳
            timestamps = np.arange(len(pred_np))
            
            # 提取注意力权重(如果模型支持)
            attention_weights = None
            if save_attention:
                try:
                    with torch.no_grad():
                        model.eval()
                        # 取一小批数据获取注意力
                        sample_batch = X[:min(4, len(X))].to(dev, non_blocking=True)
                        output_with_attn = model(sample_batch, return_attn=True)
                        
                        # 处理模型返回值 - 可能是 (output, attn) 或者只是 output
                        if isinstance(output_with_attn, tuple):
                            if len(output_with_attn) == 2:
                                _, attn = output_with_attn
                                if attn is not None:
                                    # 处理不同的注意力权重格式
                                    if isinstance(attn, torch.Tensor):
                                        attention_weights = attn.detach().cpu().numpy()
                                    elif isinstance(attn, (list, tuple)):
                                        # 如果是多头注意力,可能返回tuple/list
                                        if len(attn) > 0 and isinstance(attn[0], torch.Tensor):
                                            attention_weights = attn[0].detach().cpu().numpy()
                                        else:
                                            log_warning(f"注意力权重格式不支持: 元素类型 {type(attn[0]) if len(attn) > 0 else 'empty'}")
                                    else:
                                        log_warning(f"注意力权重格式不支持: {type(attn)}")
                                else:
                                    log_info("模型返回了空的注意力权重")
                            else:
                                log_warning(f"模型返回的tuple长度不正确: {len(output_with_attn)}")
                        else:
                            log_warning(f"模型返回格式不是tuple: {type(output_with_attn)}")
                            # 可能模型不支持return_attn,尝试直接解包
                            try:
                                # 某些模型可能直接返回多个值而不打包成tuple
                                output_with_attn = list(output_with_attn) if hasattr(output_with_attn, '__iter__') else [output_with_attn]
                                if len(output_with_attn) >= 2:
                                    attn = output_with_attn[1]
                                    if isinstance(attn, torch.Tensor):
                                        attention_weights = attn.detach().cpu().numpy()
                            except:
                                pass
                            
                except Exception as e:
                    log_warning(f"无法提取注意力权重: {e}")
            
            # 生成可视化
            model_name = os.path.splitext(os.path.basename(checkpoint))[0]
            viz_files = visualizer.create_comprehensive_evaluation_dashboard(
                predictions=pred_np,
                targets=target_np,
                timestamps=timestamps,
                feature_names=feature_names,
                model_name=model_name,
                attention_weights=attention_weights,
                confidence_intervals=None,
                original_data=arr,
                output_dir=os.path.join(output_dir, 'advanced_visualization')
            )
            
            print(f"[SUCCESS] 高级可视化完成,生成了 {len(viz_files)} 个文件")
            for key, path in viz_files.items():
                print(f"  - {key}: {os.path.relpath(path)}")
                
        except Exception as e:
            log_error(f"高级可视化失败: {e}")
            if _dbg_enabled():
                import traceback
                traceback.print_exc()


    # Optional CSV exports
    if save_csv:
        try:
            import pandas as pd  # optional dependency
            y_np = Y_np.reshape(len(Y_np), -1)
            p_np = preds.detach().cpu().numpy().reshape(len(preds), -1)
            df = pd.DataFrame(np.concatenate([y_np, p_np], axis=1))
            ensure_dir(output_dir)
            df.to_csv(os.path.join(output_dir, "preds_vs_targets.csv"), index=False)
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_warning(f"Failed to save CSV/JSON: {e}")

    # 返回metrics和可选的预测数据
    if return_predictions:
        return metrics, preds, Y
    else:
        return metrics


# -----------------------------
# CLI
# -----------------------------

def parse_indices(s: str):
    if not s:
        return None
    return [int(x) for x in s.split(',') if x.strip() != '']


def main():
    p = argparse.ArgumentParser(description="Standalone evaluator for CNN+LSTM(+Attention) model")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint (.pt) saved by Trainer")
    p.add_argument("--data", required=True, type=str, help="Path to test data (CSV/NPZ/NPY)")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to save visualizations/exports")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    p.add_argument("--sequence_length", type=int, default=None, help="Override sequence length (default from ckpt cfg)")
    p.add_argument("--horizon", type=int, default=None, help="Override forecast horizon (default from ckpt cfg)")
    p.add_argument("--normalize", type=str, default=None, choices=["standard", "minmax", "none"], help="Override normalization mode")
    p.add_argument("--feature_indices", type=str, default=None, help="Comma-separated feature indices (default from ckpt cfg or auto-detect)")
    p.add_argument("--target_indices", type=str, default=None, help="Comma-separated target indices (default from ckpt cfg or auto-detect)")
    p.add_argument("--feature_names", type=str, default=None, help="Comma-separated feature column names (for name-based selection)")
    p.add_argument("--target_names", type=str, default=None, help="Comma-separated target column names (for name-based selection)")
    p.add_argument("--auto_detect_features", action="store_true", default=True, help="Enable automatic feature/target detection (default: True)")
    p.add_argument("--disable_auto_detect", action="store_true", help="Disable automatic feature detection")
    p.add_argument("--save_csv", action="store_true", help="Save predictions and metrics to CSV/JSON as well")
    p.add_argument("--outlier_strategy", type=str, default="iqr", choices=["iqr", "z_score", "percentile", "none"], help="Outlier detection strategy for robust visualization")
    p.add_argument("--enable_advanced_viz", action="store_true", default=True, help="Enable advanced time series evaluation visualization")
    p.add_argument("--disable_advanced_viz", action="store_true", help="Disable advanced visualization (override --enable_advanced_viz)")
    p.add_argument("--save_attention", action="store_true", default=True, help="Extract and save attention weights for visualization")
    args = p.parse_args()

    try:
        # 处理可视化参数
        enable_viz = args.enable_advanced_viz and not args.disable_advanced_viz
        
        # Handle auto-detection flag
        auto_detect = args.auto_detect_features
        if args.disable_auto_detect:
            auto_detect = False
        
        metrics = evaluate(
            checkpoint=args.checkpoint,
            data_path=args.data,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
            normalize=args.normalize,
            feature_indices=parse_indices(args.feature_indices),
            target_indices=parse_indices(args.target_indices),
            feature_names=(args.feature_names.split(',') if args.feature_names else None),
            target_names=(args.target_names.split(',') if args.target_names else None),
            auto_detect_features=auto_detect,
            save_csv=args.save_csv,
            # 新增高级可视化参数
            enable_advanced_viz=enable_viz,
            save_attention=args.save_attention,
            outlier_strategy=args.outlier_strategy,
        )
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

