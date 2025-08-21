from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series.
    - 训练时：per-instance 统计，标准化后再仿射变换
    - 推理时：反标准化恢复原分布
    输入: (B, T, D)  输出: (B, T, D)
    """
    def __init__(self, d_model: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
            self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self._cache = None  # 缓存 (mean, std) 供反归一化

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        self._cache = (mean, std)
        y = (x - mean) / std
        if self.affine:
            y = y * self.gamma + self.beta
        return y

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        if self._cache is None:
            return y
        mean, std = self._cache
        if self.affine:
            y = (y - self.beta) / (self.gamma + self.eps)
        x = y * std + mean
        return x

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        return self.denormalize(x) if reverse else self.normalize(x)

