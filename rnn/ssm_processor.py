from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SSMProcessor(nn.Module):
    """
    兼容 LSTMProcessor/GRUProcessor 接口的占位 SSM 模块（轻量近似）。
    说明：
      - 该实现不是完整的 S4/Mamba，只是一个高效的一阶线性状态更新近似，
        作为占位版本，接口和形状兼容，便于后续替换为真实 SSM（如 mamba-ssm）。
    输入/输出：同 LSTMProcessor
    参数：
      input_size: 输入特征维
      hidden_size: 状态维（输出维）
      num_layers: 堆叠层数
      bidirectional: 是否双向（这里通过两个方向的 SSM 并行实现）
      dropout: 层间 dropout
      return_sequence: 是否返回序列
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.0,
                 return_sequence: bool = True) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.return_sequence = bool(return_sequence)

        self.in_proj = nn.Linear(input_size, self.hidden_size)
        self.layers = nn.ModuleList([
            _SSMLayer(self.hidden_size) for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.bidirectional:
            self.layers_bw = nn.ModuleList([
                _SSMLayer(self.hidden_size) for _ in range(self.num_layers)
            ])
        else:
            self.layers_bw = None

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got {tuple(x.shape)}")
        B,T,F = x.shape
        y = self.in_proj(x)
        y = self.dropout(y)
        for l in self.layers:
            y = l(y)
            y = self.dropout(y)
        if self.bidirectional:
            y_bw = self.in_proj(torch.flip(x, dims=[1]))
            for l in self.layers_bw:
                y_bw = l(y_bw)
            y_bw = torch.flip(y_bw, dims=[1])
            y = torch.cat([y, y_bw], dim=-1)
        if self.return_sequence:
            return y, (None, None)
        return y[:, -1, :], (None, None)


class _SSMLayer(nn.Module):
    """简化的一阶状态空间层： s_t = A ⊙ s_{t-1} + B ⊙ x_t,  y_t = C ⊙ s_t
    其中 A,B,C 为可学习向量，⊙ 为逐元素乘。
    通过 softplus/ sigmoid 约束 A∈(0,1)，稳定长序列。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.rand(dim))
        self.B = nn.Parameter(torch.randn(dim))
        self.C = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,D = x.shape
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out = []
        a = torch.sigmoid(self.A)  # (0,1)
        for t in range(T):
            s = a * s + self.B * x[:, t, :]
            y = self.C * s + self.bias
            out.append(y.unsqueeze(1))
        return torch.cat(out, dim=1)

