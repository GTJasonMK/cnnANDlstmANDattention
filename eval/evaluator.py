from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        mse = torch.mean((preds - targets) ** 2).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        rmse = mse ** 0.5
        mape = torch.mean(torch.abs((targets - preds) / (targets + 1e-8))).item()
        # R^2 (coefficient of determination)
        t = targets.detach()
        ss_res = torch.sum((t - preds) ** 2).item()
        ss_tot = (torch.sum((t - torch.mean(t)) ** 2).item()) + 1e-8
        r2 = float(1.0 - ss_res / ss_tot)
        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds_list = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)
            preds_list.append(preds.cpu())
            ys.append(y)
    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(ys, dim=0)
    return regression_metrics(preds, targets)

