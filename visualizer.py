from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# 注意：不要在模块导入时读取环境变量，避免在 main.setup_env 之后无法生效
DEFAULT_SAVE_DIR = "image"

def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _savefig(filename: str, save_dir: Optional[str] = None):
    """保存图像到指定目录。
    优先级：参数传入的 save_dir > 环境变量 VIS_SAVE_DIR > DEFAULT_SAVE_DIR
    """
    # 运行时动态读取环境变量，确保 main.setup_env 设置的目录能生效
    if save_dir is None:
        save_dir = os.environ.get("VIS_SAVE_DIR", DEFAULT_SAVE_DIR)
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, filename)
    try:
        import matplotlib.pyplot as plt  # local import safe
        plt.savefig(full, dpi=150, bbox_inches="tight")
        plt.close()  # 关闭当前figure，避免内存占用累积
    except Exception as e:
        # 不阻断主流程，仅提示
        print(f"[WARN] 保存图像失败: {full}. 原因: {e}")



def plot_losses(history: Dict[str, list], save: bool = True, filename: str = "loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("train_loss", []), label="train")
    if "val_loss" in history and len(history.get("val_loss", [])) > 0:
        plt.plot(history["val_loss"], label="val")
    if "test_loss" in history and len(history.get("test_loss", [])) > 0:
        plt.plot(history["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training/Validation/Test Loss")
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
    # 支持 (B,H,T,T) 或 (w_t, w_f) 元组
    if isinstance(attn_weights, (tuple, list)) and len(attn_weights) == 2:
        w_t, w_f = attn_weights
        # 时间注意力
        plot_attention_heatmap(w_t, title+" (Temporal)", save, filename_prefix+"temporal_")
        # 空间注意力：形状 (B,H,F,F)，轴标签换成 Feature
        if w_f is not None:
            w = w_f.mean(dim=1).detach().cpu().numpy()
            b = min(4, w.shape[0])
            for i in range(b):
                plt.figure(figsize=(5, 4))
                plt.imshow(w[i], aspect='auto', origin='lower', cmap='magma')
                plt.colorbar(); plt.xlabel("Key Feature"); plt.ylabel("Query Feature")
                plt.title(f"{title} (Spatial, batch {i})"); plt.tight_layout()
                if save: _savefig(f"{filename_prefix}spatial_{i}.png")
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


# ----------------------------
# 训练过程可视化增强
# ----------------------------

def plot_losses_logscale(history: Dict[str, list], save: bool = True, filename: str = "loss_curve_log.png"):
    """绘制训练/验证/测试损失的对数尺度曲线。
    Args:
        history: 包含 'train_loss'、可选 'val_loss'、可选 'test_loss' 的字典
        save: 是否保存图像
        filename: 保存文件名
    """
    plt.figure(figsize=(8, 4))
    train = history.get("train_loss", [])
    val = history.get("val_loss", [])
    test = history.get("test_loss", [])
    if len(train) > 0:
        plt.semilogy(train, label="train")
    if len(val) > 0:
        plt.semilogy(val, label="val")
    if len(test) > 0:
        plt.semilogy(test, label="test")
    plt.xlabel("Epoch"); plt.ylabel("Loss (log scale)")
    plt.legend(); plt.title("Training/Validation/Test Loss (log)")
    plt.tight_layout()
    if save: _savefig(filename)


def plot_grad_norm(grad_norms: list, save: bool = True, filename: str = "grad_norm.png"):
    """绘制梯度范数随训练步骤变化。
    Args:
        grad_norms: 每个训练 step 的梯度范数列表（需在外部收集）
    """
    if len(grad_norms) == 0: return
    plt.figure(figsize=(8, 3))
    plt.plot(grad_norms)
    plt.xlabel("Step"); plt.ylabel("Grad Norm")
    plt.title("Gradient Norm over Steps"); plt.tight_layout()
    if save: _savefig(filename)


def plot_param_count(model_param_count: int, save: bool = True, filename: str = "param_count.png"):
    """显示模型参数数量（以柱状图呈现一个数值，便于随模型变化对比）。
    Args:
        model_param_count: 参数总数
    """
    plt.figure(figsize=(4, 3))
    plt.bar(["params"], [model_param_count])
    plt.ylabel("Count"); plt.title("Model Parameters")
    for i, v in enumerate([model_param_count]):
        plt.text(i, v, f"{v:,}", ha='center', va='bottom')
    plt.tight_layout()
    if save: _savefig(filename)

# ----------------------------
# 预测结果分析
# ----------------------------

def plot_residual_hist(preds: torch.Tensor, targets: torch.Tensor, bins: int = 50,
                       save: bool = True, filename: str = "residual_hist.png"):
    """绘制残差（预测-真实）的直方图。"""
    res = (preds.detach().cpu().numpy() - targets.detach().cpu().numpy()).reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=bins, alpha=0.8, color='steelblue', edgecolor='k')
    plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.title("Residual Histogram")
    plt.tight_layout();
    if save: _savefig(filename)


def plot_multihorizon_error(preds: torch.Tensor, targets: torch.Tensor,
                            save: bool = True, filename: str = "horizon_error_bar.png"):
    """不同预测步长的误差对比柱状图（使用 MAE）。
    适用于 preds/targets 形状 (B, H[, C])。"""
    p = preds.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    if p.ndim == 3:  # 多目标，取平均
        p = p.mean(axis=2); t = t.mean(axis=2)
    mae_per_h = np.mean(np.abs(p - t), axis=0)
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(mae_per_h)), mae_per_h)
    plt.xlabel("Horizon step"); plt.ylabel("MAE")
    plt.title("Error by Forecast Horizon"); plt.tight_layout()
    if save: _savefig(filename)

# 置信区间绘制（需要外部提供均值和上下界）

def plot_prediction_interval(mean: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                             save: bool = True, filename: str = "prediction_interval.png"):
    """绘制预测均值及置信区间带（均值、上下界为 1D 或相同长度）。"""
    mean = np.asarray(mean); lower = np.asarray(lower); upper = np.asarray(upper)
    assert mean.shape == lower.shape == upper.shape
    x = np.arange(len(mean))
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="mean", color='C0')
    plt.fill_between(x, lower, upper, color='C0', alpha=0.2, label="interval")
    plt.xlabel("Step"); plt.ylabel("Value"); plt.legend(); plt.title("Prediction Interval")
    plt.tight_layout();
    if save: _savefig(filename)

# ----------------------------
# 模型解释性可视化
# ----------------------------

def plot_attention_multihead(attn_weights, max_batches: int = 2,
                             save: bool = True, filename_prefix: str = "attn_head_"):
    """显示多头注意力权重。
    支持：
      - 时序注意力 (B,H,T,T) 或 (B,T,T)；
      - 时空注意力返回的二元组 (w_t, w_f)，其中：
        w_t: (B,H,T,T) 或 (B,T,T)，时间维注意力
        w_f: (B,H,F,F) 或 (B,T,F,F) 或 (B,F,F)，特征维注意力
    """
    if attn_weights is None:
        return

    def _plot_time(wt):
        if wt is None: return
        if hasattr(wt, 'detach'):
            wt = wt.detach().cpu().numpy()
        wt = np.asarray(wt)
        if wt.ndim == 3:  # (B,T,T) -> add head=1
            wt = wt[:, None, ...]
        B, H, T, _ = wt.shape
        for b in range(min(B, max_batches)):
            for h in range(H):
                plt.figure(figsize=(5, 4))
                plt.imshow(wt[b, h], aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(); plt.xlabel("Key Time"); plt.ylabel("Query Time")
                plt.title(f"Temporal Attention b{b} h{h}"); plt.tight_layout()
                if save: _savefig(f"{filename_prefix}temporal_b{b}_h{h}.png")

    def _plot_feature(wf):
        if wf is None: return
        if hasattr(wf, 'detach'):
            wf = wf.detach().cpu().numpy()
        wf = np.asarray(wf)
        # 可能形状：(B,H,F,F) / (B,T,F,F) / (B,F,F)
        if wf.ndim == 4 and wf.shape[1] != wf.shape[2]:
            # (B,T,F,F) -> 按时间平均到 (B,F,F)
            wf = wf.mean(axis=1)
        if wf.ndim == 3:  # (B,F,F)
            B = wf.shape[0]
            for b in range(min(B, max_batches)):
                plt.figure(figsize=(5, 4))
                plt.imshow(wf[b], aspect='auto', origin='lower', cmap='magma')
                plt.colorbar(); plt.xlabel("Key Feature"); plt.ylabel("Query Feature")
                plt.title(f"Spatial Attention b{b}"); plt.tight_layout()
                if save: _savefig(f"{filename_prefix}spatial_b{b}.png")
        elif wf.ndim == 4:  # (B,H,F,F)
            B, H = wf.shape[:2]
            for b in range(min(B, max_batches)):
                for h in range(H):
                    plt.figure(figsize=(5, 4))
                    plt.imshow(wf[b, h], aspect='auto', origin='lower', cmap='magma')
                    plt.colorbar(); plt.xlabel("Key Feature"); plt.ylabel("Query Feature")
                    plt.title(f"Spatial Attention b{b} h{h}"); plt.tight_layout()
                    if save: _savefig(f"{filename_prefix}spatial_b{b}_h{h}.png")

    # 分派
    if isinstance(attn_weights, (tuple, list)) and len(attn_weights) == 2:
        w_t, w_f = attn_weights
        _plot_time(w_t)
        _plot_feature(w_f)
    else:
        _plot_time(attn_weights)


def plot_lstm_hidden_heatmap(hidden_seq: torch.Tensor, save: bool = True,
                              filename: str = "lstm_hidden_heatmap.png"):
    """LSTM 序列输出激活热力图。
    Args:
        hidden_seq: (B, T, H)
    """
    hs = hidden_seq.detach().cpu().numpy()
    hs = hs.mean(axis=0)  # (T, H) 按 batch 平均
    plt.figure(figsize=(7, 4))
    plt.imshow(hs.T, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(); plt.xlabel("Time"); plt.ylabel("Hidden Dim")
    plt.title("LSTM Hidden Activations (avg over batch)"); plt.tight_layout()
    if save: _savefig(filename)


def plot_cnn_feature_maps(conv_outputs: torch.Tensor, max_channels: int = 8,
                          save: bool = True, filename_prefix: str = "cnn_feat_"):
    """CNN 不同通道的时序激活可视化。
    Args:
        conv_outputs: (B, T, C)
    """
    y = conv_outputs.detach().cpu().numpy()
    y = y[0]  # 取第一个样本 (T, C)
    T, C = y.shape
    show_c = min(C, max_channels)
    for c in range(show_c):
        plt.figure(figsize=(7, 2))
        plt.plot(y[:, c])
        plt.xlabel("Time"); plt.ylabel(f"Ch{c}")
        plt.title(f"CNN Feature Channel {c}"); plt.tight_layout()
        if save: _savefig(f"{filename_prefix}{c}.png")

# ----------------------------
# 数据分析图表
# ----------------------------

def plot_series_distribution(arr: np.ndarray, save: bool = True,
                             filename: str = "series_distribution.png"):
    """原始序列各特征的直方图(拼接)。 arr: (N, F)"""
    N, F = arr.shape
    cols = min(4, F); rows = int(np.ceil(F / cols))
    plt.figure(figsize=(4*cols, 2.8*rows))
    for i in range(F):
        ax = plt.subplot(rows, cols, i+1)
        ax.hist(arr[:, i], bins=40, color='C0', alpha=0.85)
        ax.set_title(f"f{i}")
    plt.tight_layout();
    if save: _savefig(filename)


def plot_corr_heatmap(arr: np.ndarray, save: bool = True,
                      filename: str = "corr_heatmap.png"):
    """特征相关性矩阵热力图。 arr: (N, F)"""
    import pandas as pd
    df = pd.DataFrame(arr)
    corr = df.corr().values
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(); plt.title("Feature Correlation")
    plt.xlabel("Feature"); plt.ylabel("Feature")
    plt.tight_layout();
    if save: _savefig(filename)


def plot_split_distribution(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray,
                            save: bool = True, filename: str = "split_distribution.png"):
    """训练/验证/测试数据的分布对比（以整体均值与标准差为例）。"""
    def stats(a):
        return np.mean(a, axis=0), np.std(a, axis=0)
    m1, s1 = stats(train_arr); m2, s2 = stats(val_arr); m3, s3 = stats(test_arr)
    means = np.array([m1.mean(), m2.mean(), m3.mean()])
    stds = np.array([s1.mean(), s2.mean(), s3.mean()])
    x = np.arange(3)
    plt.figure(figsize=(7, 4))
    plt.bar(x-0.15, means, width=0.3, label='mean')
    plt.bar(x+0.15, stds, width=0.3, label='std')
    plt.xticks(x, ['train','val','test'])
    plt.title("Data Split Distribution (mean/std)"); plt.legend(); plt.tight_layout()
    if save: _savefig(filename)

# ----------------------------
# 性能评估可视化
# ----------------------------

def plot_temporal_performance(errors_over_time: np.ndarray, window: int = 50,
                              save: bool = True, filename: str = "temporal_perf.png"):
    """模型随时间的性能变化（滑动平均 MAE）。
    Args:
        errors_over_time: 逐时间点绝对误差 (N,)
        window: 平滑窗口大小
    """
    e = np.asarray(errors_over_time)
    if e.ndim != 1: e = e.reshape(-1)
    if len(e) == 0: return
    w = window
    ma = np.convolve(np.abs(e), np.ones(w)/w, mode='valid')
    plt.figure(figsize=(8, 3))
    plt.plot(ma)
    plt.xlabel("Time"); plt.ylabel("MAE (moving avg)")
    plt.title("Temporal Performance"); plt.tight_layout()
    if save: _savefig(filename)

