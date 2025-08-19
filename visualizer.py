from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

DEFAULT_SAVE_DIR = "image"

def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _savefig(filename: str, save_dir: str = DEFAULT_SAVE_DIR):
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, filename)
    try:
        import matplotlib.pyplot as plt  # local import safe
        plt.savefig(full, dpi=150, bbox_inches="tight")
    except Exception as e:
        # 不阻断主流程，仅提示
        print(f"[WARN] 保存图像失败: {full}. 原因: {e}")



def plot_losses(history: Dict[str, list], save: bool = True, filename: str = "loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("train_loss", []), label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training/Validation Loss")
    plt.tight_layout()
    if save:
        _savefig(filename)


def plot_lr(history: Dict[str, list], save: bool = True, filename: str = "lr_curve.png"):
    if "lr" not in history:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR Schedule")
    plt.tight_layout()
    if save:
        _savefig(filename)


def plot_predictions(preds: torch.Tensor, targets: torch.Tensor, start: int = 0, count: int = 5,
                     save: bool = True, filename_prefix: str = "pred_vs_true_"):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    n = min(count, preds.shape[0])
    horizon = preds.shape[1]
    for i in range(n):
        plt.figure(figsize=(8, 3))
        plt.plot(range(horizon), targets[start + i], label="target")
        plt.plot(range(horizon), preds[start + i], label="pred")
        plt.xlabel("Horizon")
        plt.ylabel("Value")
        plt.legend()
        plt.title(f"Sample {start + i}")
        plt.tight_layout()
        if save:
            _savefig(f"{filename_prefix}{start + i}.png")


def plot_attention_heatmap(attn_weights: torch.Tensor, title: str = "Attention Weights",
                           save: bool = True, filename_prefix: str = "attn_heatmap_"):
    if attn_weights is None:
        return
    # attn_weights: (B, H, T, T)
    w = attn_weights.mean(dim=1).detach().cpu().numpy()  # (B, T, T)
    b = min(4, w.shape[0])
    for i in range(b):
        plt.figure(figsize=(5, 4))
        plt.imshow(w[i], aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.xlabel("Key Time")
        plt.ylabel("Query Time")
        plt.title(f"{title} (batch {i})")
        plt.tight_layout()
        if save:
            _savefig(f"{filename_prefix}{i}.png")


def plot_feature_importance(weights: np.ndarray, feature_names: Optional[list] = None):
    idx = np.argsort(-np.abs(weights))
    w_sorted = weights[idx]
    names = feature_names or [f"f{i}" for i in range(len(weights))]
    names_sorted = [names[i] for i in idx]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(w_sorted)), w_sorted)
    plt.xticks(range(len(w_sorted)), names_sorted, rotation=45, ha='right')
    plt.title("Feature Importance (abs) from first FC layer weights")
    plt.tight_layout()

