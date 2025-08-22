from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalAttention(nn.Module):
    """
    Spatial-Temporal Attention for multivariate time series.

    - Temporal attention over sequence length T (like standard MHA)
    - Spatial (feature-wise) attention over feature dimension F
    - Two fusion modes: serial (temporal -> spatial) or parallel (sum/concat)

    Input:  x (B, T, F)
    Output: y (B, T, F), optionally return (attn_t, attn_f)
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 mode: str = 'serial', fuse: str = 'sum') -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mode = mode
        self.fuse = fuse

        # Temporal MHA (over T)
        self.qkv_t = nn.Linear(d_model, 3 * d_model)
        self.proj_t = nn.Linear(d_model, d_model)
        # Spatial branch will use feature-wise self-attention (per time step) without projections
        self.dropout = nn.Dropout(dropout)

    def _mhsa(self, x: torch.Tensor, qkv: nn.Linear, proj: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        qkv_out = qkv(x)
        q, k, v = qkv_out.chunk(3, dim=-1)
        def split_heads(t):
            return t.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        q = split_heads(q); k = split_heads(k); v = split_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, N, D)
        y = proj(ctx)
        y = self.dropout(y)
        return y, w  # (B,N,D), (B,H,N,N)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        # x: (B, T, F)
        # Map features to d_model if needed (assume already processed by CNN/LSTM giving d_model)
        # Temporal attention on T
        y_t, w_t = self._mhsa(x, self.qkv_t, self.proj_t)
        # Spatial attention across feature dimension, computed independently for each time step
        # scores: (B, T, D, D)
        attn_scores_f = torch.einsum('bti,btj->btij', x, x) / (x.size(-1) ** 0.5)
        w_f = torch.softmax(attn_scores_f, dim=-1)
        w_f = self.dropout(w_f)
        # apply to values (use x as values): (B,T,D)
        y_f = torch.einsum('btij,btj->bti', w_f, x)

        if self.mode == 'serial':
            y = y_t
            y = y + y_f  # residual sum
        else:  # parallel
            if self.fuse == 'concat':
                y = torch.cat([y_t, y_f], dim=-1)
                # project back to d_model
                y = nn.functional.linear(y, torch.eye(self.d_model*2, self.d_model, device=y.device))
            else:
                y = y_t + y_f
        if need_weights:
            return y, (w_t, w_f)
        return y, None

