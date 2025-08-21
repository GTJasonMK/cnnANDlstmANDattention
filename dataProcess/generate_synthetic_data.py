from __future__ import annotations

"""
生成可用于本项目的数据文件的模拟数据脚本。

- 输出矩阵形状为 (N, F+1)，最后一列为目标（target）列，
  与 data_preprocessor 的默认 target_indices=None（取最后一列）兼容。
- 支持 CSV/NPY/NPZ 格式保存。

用法示例：

1) 生成默认数据保存到 data/simulated.csv
   python generate_synthetic_data.py

2) 自定义参数：
   python generate_synthetic_data.py --n-samples 6000 --num-features 6 \
       --noise-std 0.08 --trend-slope 0.002 --no-seasonal \
       --save-format csv --out data/my_series.csv

3) 生成 NPZ：
   python generate_synthetic_data.py --save-format npz --out data/simulated.npz

生成后可在 example_config.yaml 中将 data.data_path 指向该文件。
"""

import argparse
import os
from typing import Tuple

import numpy as np


def _set_seed(seed: int):
    np.random.seed(seed)


def _make_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _seasonal_component(t: np.ndarray, n_components: int = 2) -> np.ndarray:
    """组合多个不同频率/相位的正弦成分。返回 shape (len(t),)。"""
    y = np.zeros_like(t, dtype=np.float32)
    for k in range(n_components):
        freq = 0.01 * (k + 1) * (1.0 + 0.3 * np.random.rand())
        phase = 2 * np.pi * np.random.rand()
        amp = 0.5 + np.random.rand()
        y = y + amp * np.sin(2 * np.pi * freq * t + phase)
    return y


def _build_features(n_samples: int, num_features: int, seasonal: bool, trend_slope: float, noise_std: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32)
    X = np.zeros((n_samples, num_features), dtype=np.float32)
    for i in range(num_features):
        base = np.zeros_like(t)
        if seasonal:
            base += _seasonal_component(t, n_components=2 + (i % 2))
        if trend_slope != 0.0:
            base += trend_slope * (t - t.mean())
        noise = noise_std * np.random.randn(n_samples).astype(np.float32)
        X[:, i] = base + noise
    return X


def _induce_correlation(X: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """通过随机正交矩阵混合特征引入相关性。strength ∈ [0, 1]。"""
    n_features = X.shape[1]
    if n_features < 2 or strength <= 0.0:
        return X
    A = np.random.randn(n_features, n_features).astype(np.float32)
    # QR 得到正交矩阵 Q
    Q, _ = np.linalg.qr(A)
    M = (1.0 - strength) * np.eye(n_features, dtype=np.float32) + strength * Q.astype(np.float32)
    return X @ M


def _build_target(X: np.ndarray, seasonal: bool, noise_std: float) -> np.ndarray:
    """基于特征的加权和 + 额外季节项 与 噪声 生成目标列。"""
    n_features = X.shape[1]
    w = np.abs(np.random.randn(n_features).astype(np.float32))
    w = w / (w.sum() + 1e-8)
    y = X @ w
    if seasonal:
        t = np.arange(X.shape[0], dtype=np.float32)
        y = y + 0.5 * np.sin(2 * np.pi * 0.005 * t + 1.5)
    y = y + (0.5 * noise_std) * np.random.randn(X.shape[0]).astype(np.float32)
    return y.astype(np.float32)


def generate_series(n_samples: int = 5000, num_features: int = 4, noise_std: float = 0.1,
                    trend_slope: float = 0.0, seasonal: bool = True, correlated: bool = True,
                    corr_strength: float = 0.6, seed: int = 42) -> Tuple[np.ndarray, list]:
    """
    生成多特征时间序列与目标列（最后一列）。

    返回：
        data: shape (N, num_features+1)，最后一列为 target
        columns: 列名列表 [f0, f1, ..., target]
    """
    _set_seed(seed)
    X = _build_features(n_samples, num_features, seasonal, trend_slope, noise_std)
    if correlated:
        X = _induce_correlation(X, strength=corr_strength)
    y = _build_target(X, seasonal, noise_std)
    data = np.concatenate([X, y[:, None]], axis=1).astype(np.float32)
    columns = [f"f{i}" for i in range(num_features)] + ["target"]
    return data, columns


def save_data(data: np.ndarray, columns: list, path: str, save_format: str):
    _make_dir_for(path)
    save_format = save_format.lower()
    if save_format == "csv":
        try:
            import pandas as pd  # lazy import
        except Exception as e:
            raise RuntimeError("保存 CSV 需要 pandas，请先安装：pip install pandas") from e
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False)
    elif save_format == "npz":
        np.savez(path, data=data)
    elif save_format == "npy":
        np.save(path, data)
    else:
        raise ValueError("save_format 仅支持 csv|npz|npy")


def parse_args():
    p = argparse.ArgumentParser(description="生成模拟多特征时间序列数据（最后一列为 target）")
    p.add_argument("--n-samples", type=int, default=5000, help="样本点个数 N")
    p.add_argument("--num-features", type=int, default=4, help="输入特征数（不含 target）")
    p.add_argument("--noise-std", type=float, default=0.1, help="高斯噪声标准差")
    p.add_argument("--trend-slope", type=float, default=0.0, help="线性趋势斜率（每步增量）")
    p.add_argument("--seasonal", dest="seasonal", action="store_true", help="启用季节项")
    p.add_argument("--no-seasonal", dest="seasonal", action="store_false", help="禁用季节项")
    p.set_defaults(seasonal=True)
    p.add_argument("--correlated", dest="correlated", action="store_true", help="引入特征相关性")
    p.add_argument("--no-correlated", dest="correlated", action="store_false", help="不引入相关性")
    p.set_defaults(correlated=True)
    p.add_argument("--corr-strength", type=float, default=0.6, help="相关性强度 [0,1]")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--save-format", type=str, default="csv", choices=["csv", "npz", "npy"], help="保存格式")
    p.add_argument("--out", type=str, default="data/simulated.csv", help="输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    data, columns = generate_series(
        n_samples=args.__dict__["n_samples"],
        num_features=args.__dict__["num_features"],
        noise_std=args.__dict__["noise_std"],
        trend_slope=args.__dict__["trend_slope"],
        seasonal=args.seasonal,
        correlated=args.correlated,
        corr_strength=args.__dict__["corr_strength"],
        seed=args.seed,
    )
    save_data(data, columns, args.out, args.__dict__["save_format"])
    print(f"Saved synthetic data: {args.out}  shape={data.shape}  format={args.__dict__['save_format']}")


if __name__ == "__main__":
    main()

