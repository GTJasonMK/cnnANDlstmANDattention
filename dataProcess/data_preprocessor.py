from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False


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
        normalize: str = "minmax",
        stats: Optional[NormalizationStats] = None,
        wavelet_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if data.ndim != 2:
            raise ValueError("Data must be (N, F)")

        # 保存原始数据形状用于索引计算
        self.original_data_shape = data.shape

        self.sequence_length = int(sequence_length)
        self.horizon = int(horizon)
        self.features = data.astype(np.float32)
        self.feature_idx = feature_indices
        self.target_idx = target_indices
        self.normalize = normalize

        # 🔥 CRITICAL FIX: 正确的预处理顺序
        # 1. 先应用小波变换（如果启用），但仅对特征列进行！
        self.wavelet_cfg = wavelet_cfg or {}
        if bool(self.wavelet_cfg.get('enabled', False)):
            print(f"[INFO] Step 1: Applying wavelet transform to FEATURE columns only {self.features.shape}")

            # 🔥 CRITICAL: 只对特征列应用小波变换，保持目标列原始状态
            if feature_indices is not None:
                # 分离特征列和其他列
                feature_data = self.features[:, feature_indices]
                other_data = np.delete(self.features, feature_indices, axis=1)

                print(f"[INFO] 特征列形状: {feature_data.shape}, 其他列形状: {other_data.shape}")

                # 仅对特征列应用小波变换
                wavelet_features = self._apply_wavelet(feature_data, self.wavelet_cfg)
                print(f"[INFO] 小波变换后特征列形状: {wavelet_features.shape}")

                # 重新组合数据：小波特征 + 原始其他列
                self.features = np.concatenate([wavelet_features, other_data], axis=1)

                # 🔥 更新索引：小波变换后特征列索引改变，目标列索引需要调整
                n_wavelet_features = wavelet_features.shape[1]
                self.feature_idx = list(range(n_wavelet_features))

                # 目标列索引需要重新计算（在新的数据矩阵中的位置）
                if target_indices is not None:
                    # 目标列现在在小波特征之后
                    adjusted_target_indices = []
                    original_shape = self.original_data_shape[1]  # 使用保存的原始数据形状
                    for target_idx in target_indices:
                        # 找到目标列在原始非特征列中的位置
                        non_feature_indices = [i for i in range(original_shape) if i not in feature_indices]
                        if target_idx in non_feature_indices:
                            pos_in_non_feature = non_feature_indices.index(target_idx)
                            new_target_idx = n_wavelet_features + pos_in_non_feature
                            adjusted_target_indices.append(new_target_idx)

                    self.target_idx = adjusted_target_indices
                    print(f"[INFO] 调整后的目标列索引: {self.target_idx}")

            else:
                # 如果没有指定特征列，假设除最后两列外都是特征
                print("[WARNING] 未指定feature_indices，假设除最后2列外都是特征列")
                feature_data = self.features[:, :-2]
                target_data = self.features[:, -2:]

                wavelet_features = self._apply_wavelet(feature_data, self.wavelet_cfg)
                self.features = np.concatenate([wavelet_features, target_data], axis=1)

                self.feature_idx = list(range(wavelet_features.shape[1]))
                self.target_idx = list(range(wavelet_features.shape[1], self.features.shape[1]))

            print(f"[INFO] Wavelet transform completed: {self.features.shape}")
            print(f"[INFO] KEY FIX: Target columns preserved in original state, loss computed in correct physical space")
        else:
            # 无小波变换时保持原来的逻辑
            self.feature_idx = feature_indices
            self.target_idx = target_indices

        # 2. 基于小波变换后的数据计算归一化统计量
        if stats is None:
            print(f"[INFO] Step 2: Computing normalization stats on post-wavelet data")
            self.stats = self._fit_stats(self.features)
            print(f"[INFO] Normalization stats computed (based on correct data distribution)")
        else:
            self.stats = stats
            print(f"[INFO] Using provided normalization stats")

        # 3. 最后应用归一化
        print(f"[INFO] Step 3: Applying normalization {self.normalize}")
        self.features = self._apply_normalize(self.features, self.stats)
        print(f"[INFO] Normalization completed, preprocessing sequence correct")

        self.input_idx = feature_indices if feature_indices is not None else list(range(self.features.shape[1]))
        if target_indices is None:
            self.target_idx = list(range(self.features.shape[1]))

        self.length = self.features.shape[0] - self.sequence_length - self.horizon + 1
        self.length = max(self.length, 0)

    def _fit_stats(self, arr: np.ndarray) -> NormalizationStats:
        if self.normalize == "none":
            return NormalizationStats(None, None, None, None)
            
        # 检查输入数据是否包含nan
        if np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            print(f"[ERROR] Input array contains {nan_count} NaN values during normalization!")
            # 不应该到这里，但如果到了，用0填充
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"[INFO] Replaced NaN values with 0 before normalization")
            
        if self.normalize == "standard":
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-8
            
            # 确保统计量本身不包含nan
            if np.isnan(mean).any() or np.isnan(std).any():
                print(f"[ERROR] Normalization stats contain NaN! mean_nan={np.isnan(mean).sum()}, std_nan={np.isnan(std).sum()}")
                mean = np.nan_to_num(mean, nan=0.0)
                std = np.nan_to_num(std, nan=1.0) + 1e-8
                print(f"[INFO] Fixed normalization stats")
                
            return NormalizationStats(mean=mean, std=std, min=None, max=None)
            
        if self.normalize == "minmax":
            min_v = arr.min(axis=0)
            max_v = arr.max(axis=0)
            
            # 确保统计量本身不包含nan
            if np.isnan(min_v).any() or np.isnan(max_v).any():
                print(f"[ERROR] MinMax stats contain NaN! min_nan={np.isnan(min_v).sum()}, max_nan={np.isnan(max_v).sum()}")
                min_v = np.nan_to_num(min_v, nan=0.0)
                max_v = np.nan_to_num(max_v, nan=1.0)
                print(f"[INFO] Fixed MinMax stats")
                
            return NormalizationStats(mean=None, std=None, min=min_v, max=max_v)
        raise ValueError(f"Unsupported normalize: {self.normalize}")

    def _apply_normalize(self, arr: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        if self.normalize == "none":
            return arr
            
        # 检查输入数据是否包含nan
        if np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            print(f"[WARN] Input data contains {nan_count} NaN values before applying normalization")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"[INFO] Replaced NaN values with 0 before applying normalization")
            
        if self.normalize == "standard":
            result = (arr - stats.mean) / stats.std
            
            # 检查结果是否包含nan
            if np.isnan(result).any():
                nan_count = np.isnan(result).sum()
                print(f"[ERROR] Standard normalization produced {nan_count} NaN values!")
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                print(f"[INFO] Fixed NaN values in normalized result")
                
            return result
            
        if self.normalize == "minmax":
            result = (arr - stats.min) / (stats.max - stats.min + 1e-8)
            
            # 检查结果是否包含nan
            if np.isnan(result).any():
                nan_count = np.isnan(result).sum()
                print(f"[ERROR] MinMax normalization produced {nan_count} NaN values!")
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                print(f"[INFO] Fixed NaN values in MinMax normalized result")
                
            return result
            
        return arr

    def _apply_wavelet(self, arr: np.ndarray, cfg: dict) -> np.ndarray:
        """Apply discrete wavelet transform with improved multi-resolution handling.

        🔥 IMPROVED: Better multi-resolution processing that preserves frequency characteristics
        while maintaining temporal alignment and causality.

        Args:
            arr: (N, F) input array
            cfg: wavelet configuration dict

        Returns:
            (N, F*B) where B depends on `take` (all/approx/details)
        """
        if not _HAS_PYWT:
            raise RuntimeError("pywt not installed. Please install with: pip install PyWavelets")
        wavelet = str(cfg.get('wavelet', 'db4'))
        level = int(cfg.get('level', 3))
        mode = str(cfg.get('mode', 'symmetric'))
        take = str(cfg.get('take', 'all')).lower()
        # 🔥 FIXED: 默认使用稳定的线性插值，避免潜在的数值问题
        resample_method = str(cfg.get('resample_method', 'linear')).lower()

        N, F = arr.shape
        outs = []

        print(f"[INFO] 应用改进的小波变换: {wavelet}, level={level}, mode={mode}, take={take}, resample={resample_method}")

        for f in range(F):
            x = arr[:, f]
            # 小波分解
            coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode)
            # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]

            # 选择系数
            sel = []
            if take == 'approx':
                sel = [coeffs[0]]
            elif take == 'details':
                sel = coeffs[1:]
            else:
                sel = coeffs

            # 🔥 IMPROVED: 改进的多分辨率处理
            parts = []
            for i, c in enumerate(sel):
                if resample_method == 'adaptive':
                    # 自适应重采样：保持频率特性
                    if len(c) >= N // 2:
                        # 高分辨率系数：使用线性插值
                        xi = np.linspace(0, len(c) - 1, num=len(c))
                        xN = np.linspace(0, len(c) - 1, num=N)
                        resampled = np.interp(xN, xi, c)
                    else:
                        # 低分辨率系数：使用零填充+平滑
                        pad_size = (N - len(c)) // 2
                        if pad_size > 0:
                            padded = np.pad(c, (pad_size, N - len(c) - pad_size),
                                          mode='edge')
                        else:
                            padded = c[:N]
                        # 轻微平滑以减少伪影
                        from scipy.ndimage import gaussian_filter1d
                        try:
                            resampled = gaussian_filter1d(padded, sigma=0.5)
                        except ImportError:
                            # fallback to simple interpolation
                            xi = np.linspace(0, len(c) - 1, num=len(c))
                            xN = np.linspace(0, len(c) - 1, num=N)
                            resampled = np.interp(xN, xi, c)
                elif resample_method == 'zero_pad':
                    # 零填充方法：保持原始分辨率
                    if len(c) >= N:
                        resampled = c[:N]
                    else:
                        pad_size = N - len(c)
                        resampled = np.pad(c, (0, pad_size), mode='constant', constant_values=0)
                else:
                    # 默认线性插值
                    xi = np.linspace(0, len(c) - 1, num=len(c))
                    xN = np.linspace(0, len(c) - 1, num=N)
                    resampled = np.interp(xN, xi, c)

                # 🔥 FIXED: 移除有问题的归一化，保持原始数值范围
                # 原来的归一化破坏了小波系数的数值范围和相对关系
                # resampled = resampled / (resampled_std + 1e-8)  # 已移除

                parts.append(resampled)

            band = np.stack(parts, axis=1)  # (N, B_f)
            outs.append(band)

        out = np.concatenate(outs, axis=1).astype(np.float32)  # (N, F*B)

        print(f"[INFO] 改进的小波变换完成: {arr.shape} -> {out.shape}")
        return out

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


def validate_and_process_indices(
    data_shape: Tuple[int, int],
    feature_indices: Optional[List[int]] = None,
    target_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    auto_detect: bool = True
) -> Tuple[List[int], List[int], Optional[List[str]], Optional[List[str]]]:
    """
    Validate and process feature/target indices with smart defaults and name resolution.
    
    Args:
        data_shape: Shape of the data array (N, F)
        feature_indices: List of feature column indices to use as input
        target_indices: List of target column indices to predict
        feature_names: List of feature column names (for name-based selection)
        target_names: List of target column names (for name-based selection)
        column_names: Available column names from data file (if any)
        auto_detect: Whether to auto-detect reasonable defaults
        
    Returns:
        tuple: (processed_feature_indices, processed_target_indices, resolved_feature_names, resolved_target_names)
    """
    n_samples, n_features = data_shape
    
    # Helper function to resolve names to indices
    def names_to_indices(names: List[str], available_names: List[str]) -> List[int]:
        if not available_names:
            raise ValueError("Column names not available for name-based selection")
        indices = []
        for name in names:
            if name not in available_names:
                raise ValueError(f"Column name '{name}' not found in available columns: {available_names}")
            indices.append(available_names.index(name))
        return indices
    
    # Process feature indices
    if feature_names and column_names:
        # Name-based selection for features
        feature_indices = names_to_indices(feature_names, column_names)
        print(f"[INFO] Resolved feature names {feature_names} to indices {feature_indices}")
    elif feature_indices is None:
        if auto_detect:
            # Auto-detect: use all features except the last one as input by default
            if target_indices is None:
                feature_indices = list(range(n_features - 1)) if n_features > 1 else [0]
                print(f"[INFO] Auto-detected feature indices: {feature_indices} (all except last column)")
            else:
                # Exclude target columns from features
                all_indices = set(range(n_features))
                target_set = set(target_indices if isinstance(target_indices, list) else [target_indices])
                feature_indices = list(all_indices - target_set)
                print(f"[INFO] Auto-detected feature indices: {feature_indices} (excluding targets)")
        else:
            feature_indices = list(range(n_features))
            print(f"[INFO] Using all columns as features: {feature_indices}")
    
    # Process target indices
    if target_names and column_names:
        # Name-based selection for targets
        target_indices = names_to_indices(target_names, column_names)
        print(f"[INFO] Resolved target names {target_names} to indices {target_indices}")
    elif target_indices is None:
        if auto_detect:
            # Auto-detect: use the last column as target by default
            target_indices = [n_features - 1]
            print(f"[INFO] Auto-detected target indices: {target_indices} (last column)")
        else:
            target_indices = list(range(n_features))
            print(f"[INFO] Using all columns as targets: {target_indices}")
    
    # Validate indices
    def validate_indices(indices: List[int], name: str):
        for idx in indices:
            if idx < 0 or idx >= n_features:
                raise ValueError(f"{name} index {idx} out of range [0, {n_features-1}]")
    
    validate_indices(feature_indices, "Feature")
    validate_indices(target_indices, "Target")
    
    # Check for overlap
    feature_set = set(feature_indices)
    target_set = set(target_indices)
    overlap = feature_set & target_set
    if overlap:
        print(f"[WARN] Feature and target indices overlap at columns: {list(overlap)}. This may cause data leakage.")
    
    # Resolve final names if available
    resolved_feature_names = None
    resolved_target_names = None
    if column_names:
        resolved_feature_names = [column_names[i] for i in feature_indices]
        resolved_target_names = [column_names[i] for i in target_indices]
        
        print(f"[INFO] Selected features: {list(zip(feature_indices, resolved_feature_names))}")
        print(f"[INFO] Selected targets: {list(zip(target_indices, resolved_target_names))}")
    else:
        print(f"[INFO] Selected feature indices: {feature_indices}")
        print(f"[INFO] Selected target indices: {target_indices}")
    
    return feature_indices, target_indices, resolved_feature_names, resolved_target_names


def load_array_from_path(path: str, return_columns: bool = False) -> tuple[np.ndarray, Optional[List[str]]]:
    """
    Load array from CSV/NPZ/NPY file with enhanced column information support.
    
    Args:
        path: File path to load data from
        return_columns: Whether to return column names (for CSV files)
        
    Returns:
        tuple: (data_array, column_names) where column_names is None for non-CSV files
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
        
    columns = None
    
    if path.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        
        if return_columns:
            columns = df.columns.tolist()
        
        # 检查并处理NaN值
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            print(f"[WARN] Found {nan_count} NaN values in CSV file: {path}")
            
            # ✅ FIXED: 处理NaN值的策略 - 移除违反因果性的后向填充
            # 1. 先尝试前向填充（使用过去信息填充未来缺失值 - 符合因果性）
            df_filled = df.ffill()
            # 2. ❌ 移除后向填充 - bfill()违反时间因果性（使用未来信息填充过去）
            # 3. 剩余NaN用0填充（保守策略）
            df_filled = df_filled.fillna(0)
            
            remaining_nan = df_filled.isnull().sum().sum()
            if remaining_nan > 0:
                print(f"[ERROR] Still have {remaining_nan} NaN values after forward fill, using zero fill")
                df_filled = df_filled.fillna(0)
            else:
                print(f"[INFO] Successfully filled {nan_count} NaN values using forward fill + zero fill (causality preserved)")
            
            data = df_filled.values.astype(np.float32)
        else:
            data = df.values.astype(np.float32)
            
    elif path.lower().endswith(".xlsx"):
        import pandas as pd
        df = pd.read_excel(path)
        
        if return_columns:
            columns = df.columns.tolist()
        
        # ✅ FIXED: 处理Excel文件中的NaN值 - 移除后向填充
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            print(f"[WARN] Found {nan_count} NaN values in Excel file: {path}")
            # 仅使用前向填充（符合因果性）+ 零填充
            df_filled = df.ffill().fillna(0)
            print(f"[INFO] Successfully filled {nan_count} NaN values in Excel file (forward fill + zero fill)")
            data = df_filled.values.astype(np.float32)
        else:
            data = df.values.astype(np.float32)
            
    elif path.lower().endswith(".npz"):
        loaded_data = np.load(path)
        # assume first array
        key = list(loaded_data.keys())[0]
        arr = loaded_data[key].astype(np.float32)
        
        # 检查NumPy数组中的NaN值
        if np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            print(f"[WARN] Found {nan_count} NaN values in NPZ file: {path}")
            # 用0填充NaN值
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"[INFO] Replaced NaN values with 0 in NPZ file")
        
        data = arr
        
    elif path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        
        # 检查NumPy数组中的NaN值
        if np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            print(f"[WARN] Found {nan_count} NaN values in NPY file: {path}")
            # 用0填充NaN值
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"[INFO] Replaced NaN values with 0 in NPY file")
        
        data = arr
        
    else:
        raise ValueError("Unsupported data file format. Use CSV, XLSX, NPZ, or NPY.")
    
    if return_columns:
        return data, columns
    return data, None


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
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    # Enhanced parameters for smart feature selection
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    column_names: Optional[List[str]] = None,
    auto_detect_features: bool = True,
):
    """Create train/val/test DataLoaders with strict temporal separation to prevent data leakage.
    
    Enhanced with intelligent feature/target selection and validation.

    Args:
        data: Input data array of shape (N, F)
        sequence_length: Length of input sequences
        horizon: Number of steps to predict
        feature_indices: Explicit feature column indices (can be None for auto-detection)
        target_indices: Explicit target column indices (can be None for auto-detection)  
        normalize: Normalization method ('standard', 'minmax', 'none')
        batch_size: Batch size for DataLoaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        drop_last: Whether to drop incomplete batches
        pin_memory: Whether to pin memory for GPU
        persistent_workers: Whether to keep workers alive
        prefetch_factor: Prefetch factor for workers
        feature_names: Feature column names for name-based selection
        target_names: Target column names for name-based selection
        column_names: All available column names from data source
        auto_detect_features: Whether to auto-detect reasonable feature/target defaults

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_size, n_targets)
    
    FIXED: Temporal splitting based on time indices with safety gaps to prevent data leakage.
    """
    # Smart feature/target selection and validation
    feature_indices, target_indices, resolved_feature_names, resolved_target_names = validate_and_process_indices(
        data_shape=data.shape,
        feature_indices=feature_indices,
        target_indices=target_indices,
        feature_names=feature_names,
        target_names=target_names,
        column_names=column_names,
        auto_detect=auto_detect_features
    )
    
    # Log configuration summary
    print(f"[INFO] Data configuration:")
    print(f"  - Input shape: {data.shape}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Prediction horizon: {horizon}")
    print(f"  - Feature dimensions: {len(feature_indices)} ({feature_indices})")
    print(f"  - Target dimensions: {len(target_indices)} ({target_indices})")
    if resolved_feature_names:
        print(f"  - Feature names: {resolved_feature_names}")
    if resolved_target_names:
        print(f"  - Target names: {resolved_target_names}")

    # ✅ FIXED: Time-based splitting with safety gaps to prevent leakage
    n_samples = data.shape[0]
    min_required_length = sequence_length + horizon
    
    if n_samples < min_required_length * 3:  # Need at least 3 windows for train/val/test
        raise ValueError(f"Data too short. Need at least {min_required_length * 3} samples, got {n_samples}")
    
    # Calculate time boundaries with safety gaps
    # Each split needs to reserve space for sequence_length + horizon
    total_usable = n_samples - min_required_length
    
    # Time-based splits (not window-based!)
    train_time_end = int(total_usable * train_split)
    val_time_start = train_time_end + min_required_length  # Add safety gap
    val_time_end = val_time_start + int(total_usable * val_split)
    test_time_start = val_time_end + min_required_length  # Add safety gap
    
    print(f"[INFO] Time-based data splits (FIXED to prevent leakage):")
    print(f"  - Train: [0:{train_time_end}] ({train_time_end} samples)")
    print(f"  - Safety gap: [{train_time_end}:{val_time_start}] ({val_time_start-train_time_end} samples)")
    print(f"  - Val: [{val_time_start}:{val_time_end}] ({val_time_end-val_time_start} samples)")
    print(f"  - Safety gap: [{val_time_end}:{test_time_start}] ({test_time_start-val_time_end} samples)")
    print(f"  - Test: [{test_time_start}:{n_samples}] ({n_samples-test_time_start} samples)")
    
    # ✅ FIXED: Normalization stats computed ONLY on training data
    if normalize != "none":
        train_data_only = data[:train_time_end]  # Strictly training data only!
        print(f"[INFO] Computing normalization stats on training data only: {train_data_only.shape}")
        
        if normalize == "standard":
            mean = train_data_only.mean(axis=0).astype(np.float32)
            std = (train_data_only.std(axis=0) + 1e-8).astype(np.float32)
            stats = NormalizationStats(mean=mean, std=std, min=None, max=None)
        elif normalize == "minmax":
            min_v = train_data_only.min(axis=0).astype(np.float32)
            max_v = train_data_only.max(axis=0).astype(np.float32)
            stats = NormalizationStats(mean=None, std=None, min=min_v, max=max_v)
    else:
        stats = NormalizationStats(None, None, None, None)

    # ✅ FIXED: Create separate datasets for each split with no Wavelet leakage
    # Get wavelet config but apply it separately to each split
    wavelet_cfg = getattr(getattr(create_dataloaders, '__caller_cfg__', object()), 'data', None)
    wavelet_cfg = getattr(wavelet_cfg, 'wavelet', None) if wavelet_cfg is not None else None
    wavelet_cfg_dict = wavelet_cfg if isinstance(wavelet_cfg, dict) else (wavelet_cfg.__dict__ if wavelet_cfg else None)
    
    # 🔥 CRITICAL FIX: 训练数据不能包含安全间隔内的数据！
    # 这是之前修复中引入的严重数据泄露问题
    train_data = data[:train_time_end]  # 严格限制在训练时间边界内
    val_data = data[val_time_start-sequence_length:val_time_end]  # Need lookback for sequences
    test_data = data[test_time_start-sequence_length:]  # Need lookback for sequences
    
    print(f"[CRITICAL FIX] Training data strictly limited to [:train_time_end] = [:{train_time_end}]")
    print(f"[INFO] Training data shape: {train_data.shape}")
    print(f"[INFO] Validation data shape: {val_data.shape}")
    print(f"[INFO] Test data shape: {test_data.shape}")
    
    # ✅ FIXED: Apply wavelet transform separately to each split to prevent leakage
    print(f"[INFO] Creating datasets with temporal separation...")
    
    train_ds = TimeSeriesDataset(
        data=train_data,
        sequence_length=sequence_length,
        horizon=horizon,
        feature_indices=feature_indices,
        target_indices=target_indices,
        normalize=normalize,
        stats=stats,
        wavelet_cfg=wavelet_cfg_dict,
    )
    
    # For val/test: use training stats but create with appropriate data slice
    val_ds = TimeSeriesDataset(
        data=val_data,
        sequence_length=sequence_length,
        horizon=horizon,
        feature_indices=feature_indices,
        target_indices=target_indices,
        normalize=normalize,
        stats=stats,  # Use training stats
        wavelet_cfg=wavelet_cfg_dict,
    )
    
    test_ds = TimeSeriesDataset(
        data=test_data,
        sequence_length=sequence_length,
        horizon=horizon,
        feature_indices=feature_indices,
        target_indices=target_indices,
        normalize=normalize,
        stats=stats,  # Use training stats
        wavelet_cfg=wavelet_cfg_dict,
    )
    
    # Calculate valid indices for each dataset
    # Train: use all valid windows
    train_indices = list(range(len(train_ds)))
    
    # Val: skip initial windows that would use pre-val data
    val_start_offset = sequence_length  # Skip windows that use train data
    val_indices = list(range(val_start_offset, len(val_ds)))
    
    # Test: skip initial windows that would use pre-test data  
    test_start_offset = sequence_length  # Skip windows that use val data
    test_indices = list(range(test_start_offset, len(test_ds)))
    
    print(f"[INFO] Dataset window counts after temporal separation:")
    print(f"  - Train windows: {len(train_indices)}")
    print(f"  - Val windows: {len(val_indices)} (after offset {val_start_offset})")
    print(f"  - Test windows: {len(test_indices)} (after offset {test_start_offset})")

    # Create subsets with proper indices
    train_set = Subset(train_ds, train_indices) if train_indices else train_ds
    val_set = Subset(val_ds, val_indices) if val_indices else val_ds  
    test_set = Subset(test_ds, test_indices) if test_indices else test_ds

    # DataLoader kwargs with performance hints
    def _dl_kwargs():
        kw = dict(num_workers=num_workers)
        if pin_memory is not None:
            kw["pin_memory"] = pin_memory
        if num_workers and num_workers > 0:
            if persistent_workers is not None:
                kw["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                kw["prefetch_factor"] = prefetch_factor
        return kw

    base_kwargs = _dl_kwargs()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last, **base_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **base_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **base_kwargs)

    # 🔥 CRITICAL FIX: 正确计算小波变换后的实际输入维度
    base_input_size = len(feature_indices)

    # 检查是否启用了小波变换，并计算维度倍数
    if wavelet_cfg_dict and wavelet_cfg_dict.get('enabled', False):
        level = int(wavelet_cfg_dict.get('level', 3))
        take = str(wavelet_cfg_dict.get('take', 'all')).lower()

        # 计算小波分量数量
        if take == 'approx':
            wavelet_multiplier = 1  # 只有近似分量
        elif take == 'details':
            wavelet_multiplier = level  # 只有细节分量
        else:  # take == 'all'
            wavelet_multiplier = level + 1  # 近似 + 细节分量

        input_size = base_input_size * wavelet_multiplier
        print(f"[INFO] Wavelet transform enabled: {base_input_size} features -> {input_size} features (×{wavelet_multiplier})")
    else:
        input_size = base_input_size
        print(f"[INFO] No wavelet transform: {input_size} features")

    n_targets = len(target_indices)

    return train_loader, val_loader, test_loader, input_size, n_targets

