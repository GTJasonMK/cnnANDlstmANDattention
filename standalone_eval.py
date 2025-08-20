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
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn

# Only import core model
from model_architecture import CNNLSTMAttentionModel


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

def build_model_from_cfg(cfg_dict: Dict, input_size: int, n_targets: int) -> CNNLSTMAttentionModel:
    """Map stored cfg (asdict(FullConfig)) to model constructor args."""
    m = cfg_dict.get("model", {})
    cnn = m.get("cnn", {})
    lstm = m.get("lstm", {})
    attn = m.get("attention", {})
    cnn_layers = cnn.get("layers", [])
    # layers may be list of dicts already
    model = CNNLSTMAttentionModel(
        num_features=input_size,
        cnn_layers=[(l if isinstance(l, dict) else dict(l)) for l in cnn_layers],
        use_batchnorm=bool(cnn.get("use_batchnorm", True)),
        cnn_dropout=float(cnn.get("dropout", 0.0)),
        lstm_hidden=int(lstm.get("hidden_size", 128)),
        lstm_layers=int(lstm.get("num_layers", 2)),
        bidirectional=bool(lstm.get("bidirectional", True)),
        attn_enabled=bool(attn.get("enabled", False)),
        attn_heads=int(attn.get("num_heads", 4)),
        attn_dropout=float(attn.get("dropout", 0.1)),
        fc_hidden=int(m.get("fc_hidden", 128)),
        forecast_horizon=int(m.get("forecast_horizon", 1)),
        n_targets=int(n_targets),
        attn_add_pos_enc=bool(attn.get("add_positional_encoding", False)),
        lstm_dropout=float(lstm.get("dropout", 0.0)),
    )
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

    # Load data
    arr = load_array_from_path(data_path)

    # Infer preprocessing params from checkpoint cfg with CLI override
    data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict, dict) else {}
    seq_len = int(sequence_length if sequence_length is not None else data_cfg.get("sequence_length", 64))
    hz = int(horizon if horizon is not None else cfg_dict.get("model", {}).get("forecast_horizon", data_cfg.get("horizon", 1)))
    norm = (normalize if normalize is not None else data_cfg.get("normalize", "standard"))

    feat_idx = feature_indices
    if feat_idx is None:
        fi = data_cfg.get("feature_indices", None)
        feat_idx = list(map(int, fi)) if isinstance(fi, list) else None
    targ_idx = target_indices
    if targ_idx is None:
        ti = data_cfg.get("target_indices", None)
        if isinstance(ti, list):
            targ_idx = list(map(int, ti))
        else:
            # Align with training default: when target_indices is None, use ALL features as targets
            targ_idx = list(range(arr.shape[1]))

    # Fit stats on test data (note: may differ from training-time stats)
    stats = fit_stats(arr, norm)
    arr_norm = apply_normalize(arr, stats, norm)

    # Windowize
    X_np, Y_np = windowize(arr_norm, seq_len, hz,
                           feat_idx if feat_idx is not None else list(range(arr.shape[1])),
                           targ_idx if targ_idx is not None else [arr.shape[1]-1])

    # Build model
    input_size = X_np.shape[2]
    n_targets = 1 if Y_np.ndim == 2 else Y_np.shape[2]
    model = build_model_from_cfg(cfg_dict, input_size=input_size, n_targets=n_targets)

    # Load weights
    try:
        missing = model.load_state_dict(ckpt["model_state"], strict=False)
        if hasattr(missing, 'missing_keys') and missing.missing_keys:
            print(f"[WARN] Missing keys when loading state_dict: {missing.missing_keys}")
        if hasattr(missing, 'unexpected_keys') and missing.unexpected_keys:
            print(f"[WARN] Unexpected keys when loading state_dict: {missing.unexpected_keys}")
    except RuntimeError as e:
        # Common mismatch: output head size differs due to horizon/targets. Try to load partial weights
        print(f"[WARN] Strict load failed due to shape mismatch: {e}\nAttempting partial load (excluding head)...")
        # Filter out head.* parameters
        state = ckpt["model_state"].copy()
        to_del = [k for k in list(state.keys()) if k.startswith("head.")]
        for k in to_del:
            del state[k]
        missing = model.load_state_dict(state, strict=False)
        if hasattr(missing, 'missing_keys') and missing.missing_keys:
            print(f"[WARN] Missing keys after partial load: {missing.missing_keys}")
        if hasattr(missing, 'unexpected_keys') and missing.unexpected_keys:
            print(f"[WARN] Unexpected keys after partial load: {missing.unexpected_keys}")

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

