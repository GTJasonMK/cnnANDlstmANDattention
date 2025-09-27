"""
统一的指标计算工具
整合所有重复的指标计算逻辑
"""
from __future__ import annotations

from typing import Dict, Union
import numpy as np
import torch


def _to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """统一转换为numpy数组"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def calculate_regression_metrics(predictions: Union[torch.Tensor, np.ndarray], 
                               targets: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    统一的回归指标计算函数
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        dict: 包含MSE, MAE, RMSE, MAPE, R²的字典
    """
    pred = _to_numpy(predictions)
    true = _to_numpy(targets)
    
    # 处理NaN和无穷大值
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        print(f"[WARN] Found {np.sum(np.isnan(pred) | np.isinf(pred))} invalid predictions")
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isnan(true)) or np.any(np.isinf(true)):
        print(f"[WARN] Found {np.sum(np.isnan(true) | np.isinf(true))} invalid targets")
        true = np.nan_to_num(true, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 展平为一维数组
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    
    if len(pred_flat) == 0 or len(true_flat) == 0:
        return {
            "mse": 0.0, "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "test_mse": 0.0, "test_mae": 0.0, "test_rmse": 0.0, "test_r2": 0.0, "test_loss": 0.0
        }
    
    try:
        # 基本指标
        mse = float(np.mean((pred_flat - true_flat) ** 2))
        mae = float(np.mean(np.abs(pred_flat - true_flat)))
        rmse = float(np.sqrt(max(mse, 0)))
        
        # MAPE（避免除零）
        true_safe = np.where(np.abs(true_flat) < 1e-8, 1e-8, true_flat)
        mape = float(np.mean(np.abs((true_flat - pred_flat) / true_safe)))
        
        # R²
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 0.0
        
        # 确保指标在合理范围
        mse = max(0.0, mse)
        mae = max(0.0, mae)
        rmse = max(0.0, rmse)
        mape = max(0.0, mape)
        r2 = max(-1.0, min(1.0, r2))
        
        return {
            "mse": mse,
            "mae": mae, 
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            # 兼容性字段
            "test_mse": mse,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "test_loss": mse
        }
        
    except Exception as e:
        print(f"[ERROR] Metrics calculation failed: {e}")
        return {
            "mse": float('inf'), "mae": float('inf'), "rmse": float('inf'), 
            "mape": float('inf'), "r2": 0.0,
            "test_mse": float('inf'), "test_mae": float('inf'), 
            "test_rmse": float('inf'), "test_r2": 0.0, "test_loss": float('inf')
        }