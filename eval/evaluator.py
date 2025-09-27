from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入统一的指标计算工具
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_regression_metrics
from utils.system_utils import get_device


def regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """使用统一的回归指标计算工具"""
    return calculate_regression_metrics(preds, targets)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device = None) -> Dict[str, float]:
    """评估模型性能 - 使用统一的设备管理"""
    if device is None:
        device = get_device()
        
    model.eval()
    preds_list = []
    ys = []
    
    try:
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(device)
                preds = model(x)
                
                # 检查单个批次的输出
                if torch.any(torch.isnan(preds)) or torch.any(torch.isinf(preds)):
                    print(f"[ERROR] Model output contains nan/inf in batch {batch_idx}")
                    print(f"  Input shape: {x.shape}, Output shape: {preds.shape}")
                    print(f"  Input stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
                    print(f"  Output stats: min={preds.min().item():.6f}, max={preds.max().item():.6f}, mean={preds.mean().item():.6f}")
                
                preds_list.append(preds.cpu())
                ys.append(y)
                
                # 限制处理的批次数量以避免内存问题
                if batch_idx > 100:  # 最多处理101个批次
                    break
                    
        if not preds_list:
            print("[ERROR] No predictions collected")
            return {"mse": float('nan'), "mae": float('nan'), "rmse": float('nan'), "mape": float('nan'), "r2": float('nan')}
        
        preds = torch.cat(preds_list, dim=0)
        targets = torch.cat(ys, dim=0)
        
        print(f"[DEBUG] Final tensors - Preds: {preds.shape}, Targets: {targets.shape}")
        print(f"[DEBUG] Preds stats: min={preds.min().item():.6f}, max={preds.max().item():.6f}, mean={preds.mean().item():.6f}")
        print(f"[DEBUG] Targets stats: min={targets.min().item():.6f}, max={targets.max().item():.6f}, mean={targets.mean().item():.6f}")
        
        return regression_metrics(preds, targets)
        
    except Exception as e:
        print(f"[ERROR] evaluate_model failed: {e}")
        import traceback
        traceback.print_exc()
        return {"mse": float('nan'), "mae": float('nan'), "rmse": float('nan'), "mape": float('nan'), "r2": float('nan')}

