from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    """
    轻量 Conformer Block（Macaron-FFN + MHSA + DWConv + 残差 + 归一化）。
    适配 (B, T, D) 输入输出。
    参数：
      d_model: 通道维
      num_heads: 注意力头数
      conv_kernel: depthwise conv 的核大小
      dropout: 丢弃率
    说明：
      - 该实现为简化版，便于与现有模块无缝拼接
    """
    def __init__(self, d_model: int, num_heads: int = 4, conv_kernel: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn1 = _FFN(d_model, dropout)
        self.attn = _MHSA(d_model, num_heads, dropout)
        self.conv = _ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = _FFN(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Macaron-FFN
        y = x + 0.5 * self.drop(self.ffn1(self.norm1(x)))
        # MHSA
        a, _ = self.attn(self.norm2(y), need_weights=False, attn_mask=attn_mask)
        y = y + self.drop(a)
        # ConvModule
        y = y + self.drop(self.conv(self.norm3(y)))
        # FFN
        y = y + 0.5 * self.drop(self.ffn2(self.norm4(y)))
        return y


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)


class _MHSA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        from .attention_mechanism import MultiHeadSelfAttention
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
    def forward(self, x: torch.Tensor, need_weights=False, attn_mask=None):
        return self.mhsa(x, need_weights=need_weights, attn_mask=attn_mask)


class _ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.glu = nn.GLU(dim=-1)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=(kernel - 1)//2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor):
        # (B,T,D)
        y = self.pw1(x)
        y = self.glu(y)
        y = y.transpose(1, 2)  # (B,D,T)
        y = self.dwconv(y)
        y = self.bn(y)
        y = self.act(y)
        y = y.transpose(1, 2)
        y = self.pw2(y)
        return self.drop(y)

