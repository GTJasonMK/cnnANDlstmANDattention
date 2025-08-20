from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_mechanism import MultiHeadSelfAttention


class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-Scale Temporal Attention (MSTA)

    核心思想：在多个时间尺度上（通过下采样得到）分别进行自注意力建模，
    再将各尺度的信息对齐回原始时间长度进行融合。该方法有助于同时捕捉
    短期与长期依赖关系，常常在时间序列预测任务中带来更稳健的性能提升。

    组成：
      - 多尺度构建：对输入 (B,T,D) 采用平均池化在时间维下采样为 (B,T/s,D)
      - 每个尺度上使用独立的轻量级自注意力（可共享或不共享参数，这里默认独立）
      - 插值上采样回原始步长 T，并在特征维上融合（求和或拼接+线性投影）

    参数：
      d_model: 输入/输出通道维度
      num_heads: 注意力头数
      dropout: dropout 概率
      scales: 多尺度下采样因子，例如 [1, 2, 4]（1 表示原始尺度）
      fuse: 融合策略，"sum" 或 "concat"（concat 后接线性投影回 d_model）

    参考：
      - Multi-Scale Transformer 变体在时序任务中的应用（2023-2024 多篇工作）
      - 该实现为简化版，便于与现有代码集成与消融对比
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        scales: Optional[List[int]] = None,
        fuse: str = "sum",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.scales = scales or [1, 2]
        self.fuse = fuse

        self.attn_blocks = nn.ModuleList([
            MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in self.scales
        ])
        if self.fuse == "concat":
            self.proj = nn.Linear(d_model * len(self.scales), d_model)
        else:
            self.proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def _downsample(self, x: torch.Tensor, s: int) -> torch.Tensor:
        if s == 1:
            return x
        # x: (B,T,D) -> (B,D,T)
        x_ch = x.transpose(1, 2)
        x_ds = F.avg_pool1d(x_ch, kernel_size=s, stride=s, padding=0)
        return x_ds.transpose(1, 2)

    def _upsample_to_T(self, x_s: torch.Tensor, T: int) -> torch.Tensor:
        # x_s: (B,T_s,D) -> (B,D,T_s) -> interpolate to T -> (B,T,D)
        x_ch = x_s.transpose(1, 2)
        x_up = F.interpolate(x_ch, size=T, mode="linear", align_corners=False)
        return x_up.transpose(1, 2)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        B, T, D = x.shape
        outs = []
        attn_ws = []
        for s, attn in zip(self.scales, self.attn_blocks):
            x_s = self._downsample(x, s)
            y_s, w_s = attn(x_s, need_weights=need_weights)
            y_up = self._upsample_to_T(y_s, T)
            outs.append(y_up)
            if need_weights:
                # 将权重直接上采样到 T×T 近似对齐（简化处理）
                if w_s is not None:
                    # w_s: (B,H,T_s,T_s) -> 上采样到 (B,H,T,T)
                    w = F.interpolate(
                        F.interpolate(w_s, size=(T, w_s.shape[-1]), mode="nearest"),
                        size=(T, T), mode="nearest"
                    )
                    attn_ws.append(w)
        if self.fuse == "concat":
            y = torch.cat(outs, dim=-1)
            y = self.proj(y)
        else:
            y = torch.stack(outs, dim=0).sum(dim=0)
        y = self.dropout(y)
        if need_weights:
            if len(attn_ws) == 0:
                return y, None
            # 简单平均不同尺度的注意力
            W = torch.stack(attn_ws, dim=0).mean(dim=0)
            return y, W
        return y, None

