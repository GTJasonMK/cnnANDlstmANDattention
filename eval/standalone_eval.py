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
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import traceback

# Only import core model
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_architecture import CNNLSTMAttentionModel


# -----------------------------
# Debug helpers
# -----------------------------

def _dbg_enabled() -> bool:
    import os as _os
    return (_os.environ.get("EVAL_DEBUG", "0") != "0")


def _dlog(msg: str):
    if _dbg_enabled():
        print(f"[DEBUG] {msg}")


def _strict_mode() -> bool:
    import os as _os
    return (_os.environ.get("EVAL_STRICT", "1") != "0")

def _env_true(name: str, default: str = "0") -> bool:
    import os as _os
    return _os.environ.get(name, default) not in (None, "0", "false", "False", "")


# -----------------------------
# IO helpers
# -----------------------------

def load_array_from_path(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    lower = path.lower()
    if lower.endswith(".csv"):
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("Reading CSV requires pandas. Please install: pip install pandas") from e
        df = pd.read_csv(path)
        return df.values.astype(np.float32)
    if lower.endswith(".npz"):
        z = np.load(path)
        key = list(z.keys())[0]
        return z[key].astype(np.float32)
    if lower.endswith(".npy"):
        return np.load(path).astype(np.float32)
    raise ValueError("Unsupported data file format. Use CSV, NPZ, or NPY.")


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


def windowize(arr: np.ndarray, seq_len: int, horizon: int, feature_idx: List[int], target_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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


def regression_metrics(preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray) -> Dict[str, float]:
    p = _to_np(preds)
    t = _to_np(targets)
    # Align dims: (N, H[, C]) -> flatten all dims
    p_f = p.reshape(-1)
    t_f = t.reshape(-1)
    mse = float(np.mean((p_f - t_f) ** 2))
    mae = float(np.mean(np.abs(p_f - t_f)))
    rmse = float(np.sqrt(mse))
    # Avoid division by zero in MAPE
    mape = float(np.mean(np.abs((t_f - p_f) / (np.where(np.abs(t_f) < 1e-8, 1e-8, t_f)))))
    # R^2
    ss_res = np.sum((t_f - p_f) ** 2)
    ss_tot = np.sum((t_f - np.mean(t_f)) ** 2) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}



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


def _to_int_list(val) -> Optional[List[int]]:
    """将任意列表/元组转换为 int 列表；过滤 None/非法项；若无有效项返回 None。"""
    if isinstance(val, (list, tuple)):
        out: List[int] = []
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

import matplotlib.pyplot as plt

def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def savefig(path: str):
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pred_vs_true(preds: torch.Tensor, targets: torch.Tensor, out_dir: str, count: int = 5):
    p = preds.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    n = min(count, p.shape[0])
    H = p.shape[1]
    is_multi = (p.ndim == 3)
    for i in range(n):
        plt.figure(figsize=(7, 3))
        if is_multi:
            plt.plot(range(H), t[i].mean(axis=-1), label="target(avg)")
            plt.plot(range(H), p[i].mean(axis=-1), label="pred(avg)")
        else:
            plt.plot(range(H), t[i], label="target")
            plt.plot(range(H), p[i], label="pred")
        plt.xlabel("Horizon"); plt.ylabel("Value"); plt.title(f"Sample {i}")
        plt.legend(); plt.tight_layout()
        savefig(os.path.join(out_dir, f"pred_vs_true_{i}.png"))


def plot_residual_hist(preds: torch.Tensor, targets: torch.Tensor, out_dir: str, bins: int = 50):
    p = preds.detach().cpu().numpy().reshape(-1)
    t = targets.detach().cpu().numpy().reshape(-1)
    res = p - t
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=bins, alpha=0.85, color="steelblue", edgecolor="k")
    plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.title("Residual Histogram")
    plt.tight_layout(); savefig(os.path.join(out_dir, "residual_hist.png"))


def plot_temporal_performance(preds: torch.Tensor, targets: torch.Tensor, out_dir: str, window: int = 50):
    p = preds.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    if p.ndim == 3:
        p = p.mean(axis=2)
        t = t.mean(axis=2)
    # Flatten horizon, then moving average abs error
    err = np.abs(p - t).reshape(-1)
    if err.size == 0:
        return
    w = max(1, min(window, err.size))
    ma = np.convolve(err, np.ones(w)/w, mode='valid')
    plt.figure(figsize=(8, 3))
    plt.plot(ma)
    plt.xlabel("Time (flattened windows)"); plt.ylabel("MAE (moving avg)")
    plt.title("Temporal Performance"); plt.tight_layout()
    savefig(os.path.join(out_dir, "temporal_perf.png"))


def plot_attention_heatmap(attn: torch.Tensor, out_dir: str, prefix: str = "attn_heatmap_"):
    if attn is None:
        return
    # attn: (B, H, T, T)
    w = attn.detach().cpu().numpy()
    b = min(4, w.shape[0])
    for i in range(b):
        plt.figure(figsize=(5, 4))
        plt.imshow(w[i].mean(axis=0), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(); plt.xlabel("Key"); plt.ylabel("Query"); plt.title(f"Attention (batch {i})")
        plt.tight_layout(); savefig(os.path.join(out_dir, f"{prefix}{i}.png"))


def _inverse_value(val: float, feature_idx: int, stats: NormalizationStats, mode: str) -> float:
    mode = (mode or "none").lower()
    if mode == "standard" and stats.mean is not None and stats.std is not None:
        return float(val * stats.std[feature_idx] + stats.mean[feature_idx])
    if mode == "minmax" and stats.min is not None and stats.max is not None:
        return float(val * (stats.max[feature_idx] - stats.min[feature_idx]) + stats.min[feature_idx])
    return float(val)


def plot_feature_sequences(
    arr_raw: np.ndarray,
    preds: torch.Tensor,
    target_idx: List[int],
    seq_len: int,
    horizon: int,
    stats: NormalizationStats,
    normalize_mode: str,
    out_dir: str,
    feature_names: Optional[List[str]] = None,
    inverse: bool = True,
):
    """Plot original vs predicted sequences for each target feature.

    preds: (Nw, H) or (Nw, H, C)
    arr_raw: (N, F) in original scale
    """
    ensure_dir(out_dir)
    P = preds.detach().cpu().numpy()
    if P.ndim == 2:
        P = P[:, :, None]
    Nw, H, C = P.shape
    N, F = arr_raw.shape
    # Accumulate predictions per time step with averaging for overlaps
    pred_sum = np.zeros((C, N), dtype=np.float32)
    pred_cnt = np.zeros((C, N), dtype=np.int32)

    for i in range(Nw):
        for h in range(H):
            t = i + seq_len + h
            if t >= N:
                continue
            for c in range(C):
                val = float(P[i, h, c])
                j = target_idx[c]
                if inverse:
                    val = _inverse_value(val, j, stats, normalize_mode)
                pred_sum[c, t] += val
                pred_cnt[c, t] += 1

    # Average where multiple predictions exist
    with np.errstate(invalid='ignore'):
        pred_avg = np.where(pred_cnt > 0, pred_sum / np.maximum(pred_cnt, 1), np.nan)

    # Ground truth for target features (original scale)
    gt = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        gt[c] = arr_raw[:, target_idx[c]]

    # Build names
    names = feature_names if (isinstance(feature_names, list) and len(feature_names) == F) else [f"f{k}" for k in range(F)]

    # Plot each target feature
    for c in range(C):
        j = target_idx[c]
        plt.figure(figsize=(10, 3))
        x = np.arange(N)
        plt.plot(x, gt[c], label=f"true ({names[j]})", color='C0', alpha=0.9)
        plt.plot(x, pred_avg[c], label=f"pred ({names[j]})", color='C1', alpha=0.9)
        plt.xlabel("Time"); plt.ylabel("Value")
        plt.title(f"Feature {j}: {names[j]} (seq_len={seq_len}, horizon={horizon})")
        plt.legend(); plt.tight_layout()
        savefig(os.path.join(out_dir, f"feature_sequence_{j}.png"))


# -----------------------------
# Checkpoint & Model construction
# -----------------------------

def _extract_arch_meta(cfg_dict: Dict) -> Dict[str, Any]:
    m = cfg_dict.get("model", {}) or {}
    cnn = m.get("cnn", {}) or {}
    tcn = m.get("tcn", {}) or {}
    lstm = m.get("lstm", {}) or {}
    attn = m.get("attention", {}) or {}

    # 判断是否启用 TCN：显式 variant=tcn 或 tcn.enabled=True
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


def _strict_get(d: Dict[str, Any], key: str, typename: str) -> Any:
    if key not in d:
        raise ValueError(f"缺少必需字段: {key}")
    val = d[key]
    if val is None:
        raise ValueError(f"字段 {key} 不能为 None（需与训练时一致）")
    return val


def _strict_int(d: Dict[str, Any], key: str) -> int:
    val = _strict_get(d, key, "int")
    try:
        return int(val)
    except Exception:
        raise ValueError(f"字段 {key} 需要为整数，但得到: {val}")


def _strict_bool(d: Dict[str, Any], key: str) -> bool:
    val = _strict_get(d, key, "bool")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    raise ValueError(f"字段 {key} 需要为布尔，但得到: {val}")


def _strict_list(d: Dict[str, Any], key: str) -> List[Any]:
    val = _strict_get(d, key, "list")
    if not isinstance(val, list):
        raise ValueError(f"字段 {key} 需要为列表，但得到: {type(val)}")
    return val


def _validate_layers_strict(layers: List[Dict[str, Any]], required_keys: List[str], path: str):
    if not layers:
        raise ValueError(f"{path} 至少需要一层配置")
    for i, l in enumerate(layers):
        if not isinstance(l, dict):
            raise ValueError(f"{path}[{i}] 需要为字典，但得到: {type(l)}")
        for k in required_keys:
            if l.get(k, None) is None:
                raise ValueError(f"{path}[{i}] 缺少必需字段: {k}")


def build_model_from_cfg_strict(cfg_dict: Dict, input_size: int, n_targets: int) -> CNNLSTMAttentionModel:
    """严格从 checkpoint 的 cfg 重建训练时完全一致的模型结构。
    - 不使用默认值，不做降级；字段缺失/None 直接抛错
    - CNN 变体：standard/depthwise/dilated
    - TCN：variant=tcn 或 tcn.enabled=True 时使用 tcn 配置
    - RNN：lstm/gru
    - 注意力：standard/multiscale（multiscale 需提供 scales/fuse）
    """
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint 中 cfg 不是字典或为空，请保存完整配置后再评估")

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

    # TCN 分支或 CNN 分支
    base_layers: List[Dict[str, Any]]
    use_batchnorm: bool
    dropout: float

    if cnn_variant == "tcn":
        tcn = _strict_get(m, "tcn", "dict")
        if not isinstance(tcn, dict):
            raise ValueError("cfg['model']['tcn'] 必须为字典，且需包含 TCN 层配置")
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
    # 位置编码字段名兼容两种拼写，但必须提供其中之一
    if "add_positional_encoding" in attn:
        attn_add_pos_enc = _strict_bool(attn, "add_positional_encoding")
    elif "add_posional_encoding" in attn:
        attn_add_pos_enc = _strict_bool(attn, "add_posional_encoding")
    else:
        raise ValueError("attention 缺少 add_positional_encoding 字段（或旧拼写 add_posional_encoding）")

    multiscale_scales = None
    multiscale_fuse = None
    if attn_variant == "multiscale":
        multiscale_scales = _strict_list(attn, "multiscale_scales")
        if not all(isinstance(x, int) for x in multiscale_scales):
            raise ValueError("multiscale_scales 需要为整数列表")
        multiscale_fuse = _strict_get(attn, "multiscale_fuse", "str")

    # 构建模型（严格模式：完全按 cfg）。若 TCN 配置写在 cnn.layers（老格式）且 cnn_variant==tcn，会在前面严格校验环节已抛错。
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
             feature_indices: Optional[List[int]] = None,
             target_indices: Optional[List[int]] = None,
             save_csv: bool = False,
             plot_sequences: bool = False,
             feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    ensure_dir(output_dir)

    # Load checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    _dlog(f"Loaded ckpt keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    _dlog(f"cfg_dict_type={type(cfg_dict)} model_keys={list((cfg_dict.get('model') or {}).keys()) if isinstance(cfg_dict, dict) else 'N/A'} data_keys={list((cfg_dict.get('data') or {}).keys()) if isinstance(cfg_dict, dict) else 'N/A'}")

    # Load data
    arr = load_array_from_path(data_path)

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

    feat_idx = feature_indices
    if feat_idx is None:
        fi = data_cfg.get("feature_indices", None)
        feat_idx = _to_int_list(fi)
    targ_idx = target_indices
    if targ_idx is None:
        ti = data_cfg.get("target_indices", None)
        targ_idx = _to_int_list(ti)
        if targ_idx is None:
            # Align with training default: when target_indices is None, use ALL features as targets
            targ_idx = list(range(arr.shape[1]))

    # Fit stats on test data (note: may differ from training-time stats)
    _dlog(f"data_shape={arr.shape} first_row_sample={arr[0][:5] if arr.size>0 else 'EMPTY'}")
    stats = fit_stats(arr, norm)
    arr_norm = apply_normalize(arr, stats, norm)

    # Windowize
    _dlog(f"feature_idx={feat_idx} target_idx={targ_idx}")
    X_np, Y_np = windowize(arr_norm, seq_len, hz,
                           feat_idx if feat_idx is not None else list(range(arr.shape[1])),
                           targ_idx if targ_idx is not None else [arr.shape[1]-1])
    _dlog(f"windowized X={X_np.shape} Y={Y_np.shape}")

    # 读取期望的 IO 形状（从权重中推断），与数据/配置进行严格对齐
    state = ckpt.get("model_state", {})
    exp_in_channels = None
    exp_out_dim = None
    if isinstance(state, dict):
        # 推断输入通道：寻找第一个 conv1d 权重（形状为 [out_c, in_c, k]）
        for k, v in state.items():
            try:
                if isinstance(v, torch.Tensor) and v.ndim == 3 and k.startswith("cnn") and k.endswith(".weight"):
                    exp_in_channels = int(v.shape[1])
                    break
            except Exception:
                continue
        # 推断输出维度：寻找 head.*.weight 的二维矩阵，取“最后一层 head 的 out_features”
        head_out_dim = None
        head_max_idx = -1
        for k, v in state.items():
            try:
                if isinstance(v, torch.Tensor) and v.ndim == 2 and k.startswith("head") and k.endswith(".weight"):
                    # 解析 head.<idx>.weight 的 idx，选择最大的 idx 作为最终输出层
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

    # 构建模型（严格一致）。若 cfg 信息缺失，将抛错并停止评估。
    input_size = int(X_np.shape[2])
    n_targets = int(1 if Y_np.ndim == 2 else Y_np.shape[2])
    _dlog(f"input_size={input_size} n_targets={n_targets} exp_in={exp_in_channels} exp_out={exp_out_dim}")

    # 严格校验：数据特征数与训练时一致；输出维度与 horizon*n_targets 一致
    if _strict_mode():
        if exp_in_channels is not None and input_size != exp_in_channels:
            raise RuntimeError(
                f"输入特征数与训练时不一致: now={input_size}, expected={exp_in_channels}. "
                f"请使用与训练相同的特征列（或设置相同的 feature_indices），并确保数据预处理一致。")
        hz_check = _as_int(horizon, None) or _as_int(cfg_dict.get("model", {}).get("forecast_horizon"), None)
        if exp_out_dim is not None and hz_check is not None:
            expected_targets = exp_out_dim // int(hz_check) if exp_out_dim % int(hz_check) == 0 else None
            cur_out_dim = int(n_targets) * int(hz_check)
            if expected_targets is None or cur_out_dim != exp_out_dim:
                # 可选1：仅对齐目标列个数（在 horizon 与 exp_out_dim 可整除时）
                if _env_true("EVAL_AUTO_ALIGN_TARGETS") and expected_targets is not None and target_indices is None:
                    _dlog(f"auto-align targets: take first {expected_targets} of {arr.shape[1]} columns")
                    targ_idx = list(range(int(expected_targets)))
                    X_np, Y_np = windowize(
                        arr_norm, seq_len, hz_check,
                        feat_idx if feat_idx is not None else list(range(arr.shape[1])),
                        targ_idx
                    )
                    n_targets = int(1 if Y_np.ndim == 2 else Y_np.shape[2])
                    cur_out_dim = int(n_targets) * int(hz_check)
                    if cur_out_dim != exp_out_dim:
                        raise RuntimeError(
                            f"自动对齐失败: 重新计算后 hz*targets={cur_out_dim}, 仍不等于 {exp_out_dim}. "
                            f"请显式提供训练时的 target_indices（长度={expected_targets}）。")
                    else:
                        _dlog("auto-align OK, proceeding with strict load")
                # 可选2：自动推断 horizon 与目标列个数（当 horizon 与 exp_out_dim 不整除时）
                elif _env_true("EVAL_AUTO_INFER_HZ") and target_indices is None and exp_out_dim is not None:
                    def _divisors(n: int):
                        ds = []
                        for d in range(1, n + 1):
                            if n % d == 0:
                                ds.append(d)
                        return ds
                    cand_hz = _divisors(int(exp_out_dim))
                    # 优先使用 CLI 的 horizon 或 cfg 的 horizon（若可用且可整除并满足目标列数不超过现有列数）
                    pref = []
                    cli_hz = _as_int(horizon, None)
                    cfg_hz = _as_int(cfg_dict.get("model", {}).get("forecast_horizon"), None)
                    if cli_hz in cand_hz and (exp_out_dim // int(cli_hz)) <= arr.shape[1]:
                        pref.append(int(cli_hz))
                    if cfg_hz in cand_hz and (exp_out_dim // int(cfg_hz)) <= arr.shape[1]:
                        pref.append(int(cfg_hz))
                    chosen_hz = None
                    if pref:
                        chosen_hz = pref[0]
                    else:
                        # 选择一个使 n_targets <= 当前特征列数的 horizon，偏好更大的 horizon（更接近训练可能的设定）
                        for hz_c in sorted(cand_hz, reverse=True):
                            n_t = exp_out_dim // hz_c
                            if n_t <= arr.shape[1]:
                                chosen_hz = hz_c
                                break
                    if chosen_hz is None:
                        raise RuntimeError(
                            f"无法自动推断 horizon：exp_out_dim={exp_out_dim} 与可用列数={arr.shape[1]} 不匹配。"
                            f"请显式提供与训练一致的 horizon 与 target_indices。")
                    expected_targets2 = exp_out_dim // int(chosen_hz)
                    _dlog(f"auto-infer horizon: choose hz={chosen_hz}, targets={expected_targets2}")
                    targ_idx = list(range(int(expected_targets2)))
                    # 覆盖运行时 horizon 与 cfg 的 forecast_horizon（仅当前评估进程内生效）
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
                            f"请显式提供训练时的 horizon 与 target_indices。")
                    else:
                        _dlog("auto-infer horizon OK, proceeding with strict load")
                else:
                    raise RuntimeError(
                        f"输出维度不一致: now(hz*targets)={cur_out_dim}, expected={exp_out_dim}. "
                        f"请确保 forecast_horizon 与目标列数与训练一致。")

    model = build_model_from_cfg_strict(cfg_dict, input_size=input_size, n_targets=n_targets)

    # Load weights
    try:
        # 严格模式：必须完全匹配
        missing = model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        if _strict_mode():
            raise RuntimeError(
                "严格加载权重失败（形状不匹配）。请确保评估时的数据特征数、目标列与 forecast_horizon 与训练完全一致。\n"
                f"详细信息: {e}")
        else:
            _dlog("state_dict strict load failed; attempting partial load")
            print(f"[WARN] Strict load failed due to shape mismatch: {e}\nAttempting partial load (excluding head)...")
            state2 = ckpt["model_state"].copy()
            to_del = [k for k in list(state2.keys()) if k.startswith("head.")]
            for k in to_del:
                del state2[k]
            try:
                missing = model.load_state_dict(state2, strict=False)
            except RuntimeError:
                _dlog("partial load still failed; fallback to shape-matched loading")
                cur = model.state_dict()
                filtered = {}
                for k, v in ckpt["model_state"].items():
                    if k in cur and hasattr(v, 'shape') and hasattr(cur[k], 'shape') and tuple(v.shape) == tuple(cur[k].shape):
                        filtered[k] = v
                _dlog(f"shape-matched params: {len(filtered)}/{len(ckpt['model_state'])}")
                missing = model.load_state_dict(filtered, strict=False)

    # Device
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    model.to(dev)
    model.eval()

    # Batching inference
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    preds_list: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = X[start:end].to(dev, non_blocking=True)
            out = model(xb)
            preds_list.append(out.detach().cpu())
    preds = torch.cat(preds_list, dim=0)
    _dlog(f"inference done, preds_shape={tuple(preds.shape)}")
    metrics = regression_metrics(preds, Y)

    # Print metrics
    print(json.dumps({k: round(v, 6) for k, v in metrics.items()}, ensure_ascii=False))

    # Visualizations
    try:
        plot_pred_vs_true(preds, Y, out_dir=output_dir, count=min(5, len(preds)))
        plot_residual_hist(preds, Y, out_dir=output_dir)
        plot_temporal_performance(preds, Y, out_dir=output_dir)
        # Attention heatmap (single batch)
        with torch.no_grad():
            xb0 = X[:min(len(X), batch_size)].to(dev, non_blocking=True)
            out, attn = model(xb0, return_attn=True)
            plot_attention_heatmap(attn, out_dir=output_dir)
        # Feature sequences (optional)
        if plot_sequences:
            # Try to infer original (unnormalized) series for display
            arr_disp = arr
            if normalize and normalize.lower() != 'none':
                arr_disp = arr  # arr is raw already (stats were fit on raw arr), so display arr
            plot_feature_sequences(
                arr_raw=arr_disp,
                preds=preds,
                target_idx=(targ_idx if targ_idx is not None else [arr.shape[1]-1]),
                seq_len=seq_len,
                horizon=hz,
                stats=stats,
                normalize_mode=norm,
                out_dir=os.path.join(output_dir, 'feature_sequences'),
                feature_names=feature_names,
                inverse=True,
            )
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

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
            print(f"[WARN] Failed to save CSV/JSON: {e}")

    return metrics


# -----------------------------
# CLI
# -----------------------------

def parse_indices(s: Optional[str]) -> Optional[List[int]]:
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
    p.add_argument("--feature_indices", type=str, default=None, help="Comma-separated feature indices (default from ckpt cfg or all)")
    p.add_argument("--target_indices", type=str, default=None, help="Comma-separated target indices (default from ckpt cfg or last)")
    p.add_argument("--save_csv", action="store_true", help="Save predictions and metrics to CSV/JSON as well")
    p.add_argument("--plot_sequences", action="store_true", help="Plot per-feature sequences (original vs predicted)")
    p.add_argument("--feature_names", type=str, default=None, help="Comma-separated feature names (length must match data columns)")
    args = p.parse_args()

    try:
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
            save_csv=args.save_csv,
            plot_sequences=args.plot_sequences,
            feature_names=(args.feature_names.split(',') if args.feature_names else None),
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

