from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


@dataclass
class NormalizationStats:
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]
    min: Optional[np.ndarray]
    max: Optional[np.ndarray]


class TimeSeriesDataset(Dataset):
    """
    Windowed time series dataset for multivariate sequences.

    Accepts either a numpy array of shape (N, F) or a path to a CSV/NPZ file.
    Produces samples of shape (T, F_selected) with targets of shape (H, T_selected) or (H,).
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        horizon: int,
        feature_indices: Optional[List[int]] = None,
        target_indices: Optional[List[int]] = None,
        normalize: str = "standard",
        stats: Optional[NormalizationStats] = None,
    ) -> None:
        super().__init__()
        if data.ndim != 2:
            raise ValueError("Data must be (N, F)")
        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.features = data.astype(np.float32)
        self.feature_idx = feature_indices
        self.target_idx = target_indices
        self.normalize = normalize
        self.stats = stats or self._fit_stats(self.features)
        self.features = self._apply_normalize(self.features, self.stats)

        self.input_idx = feature_indices if feature_indices is not None else list(range(self.features.shape[1]))
        if target_indices is None:
            self.target_idx = list(range(self.features.shape[1]))

        self.length = self.features.shape[0] - self.sequence_length - self.horizon + 1
        self.length = max(self.length, 0)

    def _fit_stats(self, arr: np.ndarray) -> NormalizationStats:
        if self.normalize == "none":
            return NormalizationStats(None, None, None, None)
        if self.normalize == "standard":
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-8
            return NormalizationStats(mean=mean, std=std, min=None, max=None)
        if self.normalize == "minmax":
            min_v = arr.min(axis=0)
            max_v = arr.max(axis=0)
            return NormalizationStats(mean=None, std=None, min=min_v, max=max_v)
        raise ValueError(f"Unsupported normalize: {self.normalize}")

    def _apply_normalize(self, arr: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        if self.normalize == "none":
            return arr
        if self.normalize == "standard":
            return (arr - stats.mean) / stats.std
        if self.normalize == "minmax":
            return (arr - stats.min) / (stats.max - stats.min + 1e-8)
        return arr

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        start = idx
        end = idx + self.sequence_length
        target_start = end
        target_end = target_start + self.horizon
        x = self.features[start:end, self.input_idx]
        y = self.features[target_start:target_end, self.target_idx]
        if y.shape[1] == 1:
            y = y.squeeze(1)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_array_from_path(path: str) -> np.ndarray:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        return df.values.astype(np.float32)
    if path.lower().endswith(".npz"):
        data = np.load(path)
        # assume first array
        key = list(data.keys())[0]
        return data[key].astype(np.float32)
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    raise ValueError("Unsupported data file format. Use CSV, NPZ, or NPY.")


def create_dataloaders(
    data: np.ndarray,
    sequence_length: int,
    horizon: int,
    feature_indices: Optional[List[int]],
    target_indices: Optional[List[int]],
    normalize: str,
    batch_size: int,
    train_split: float,
    val_split: float,
    num_workers: int = 0,
    shuffle_train: bool = True,
    drop_last: bool = False,
):
    """Create train/val/test DataLoaders with normalization fit on train split only.

    Splitting is chronological to avoid leakage from overlapping windows.
    """
    # Base dataset for determining total number of windows
    base_ds = TimeSeriesDataset(
        data=data,
        sequence_length=sequence_length,
        horizon=horizon,
        feature_indices=feature_indices,
        target_indices=target_indices,
        normalize=normalize,
    )

    n_total = len(base_ds)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    # Chronological indices
    idx_train = list(range(0, n_train))
    idx_val = list(range(n_train, n_train + n_val))
    idx_test = list(range(n_train + n_val, n_total))

    # Fit normalization on raw series up to the last index used by training inputs
    if normalize == "standard":
        end_idx = min(data.shape[0], n_train + sequence_length - 1)
        train_slice = data[:end_idx]
        mean = train_slice.mean(axis=0).astype(np.float32)
        std = (train_slice.std(axis=0) + 1e-8).astype(np.float32)
        stats = NormalizationStats(mean=mean, std=std, min=None, max=None)
    elif normalize == "minmax":
        end_idx = min(data.shape[0], n_train + sequence_length - 1)
        train_slice = data[:end_idx]
        min_v = train_slice.min(axis=0).astype(np.float32)
        max_v = train_slice.max(axis=0).astype(np.float32)
        stats = NormalizationStats(mean=None, std=None, min=min_v, max=max_v)
    else:
        stats = NormalizationStats(None, None, None, None)

    # Final datasets with fixed stats
    full_ds = TimeSeriesDataset(
        data=data,
        sequence_length=sequence_length,
        horizon=horizon,
        feature_indices=feature_indices,
        target_indices=target_indices,
        normalize=normalize,
        stats=stats,
    )

    train_set = Subset(full_ds, idx_train)
    val_set = Subset(full_ds, idx_val)
    test_set = Subset(full_ds, idx_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    input_size = len(full_ds.input_idx)
    n_targets = len(full_ds.target_idx)

    return train_loader, val_loader, test_loader, input_size, n_targets

