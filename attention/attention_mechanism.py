from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention over temporal dimension.

    Input/Output shapes:
      - Input: (B, T, D)
      - Output: (B, T, D)
    Also returns attention weights if requested.

    Args:
        d_model: Input/Output feature size.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights and output proj.
        add_positional_encoding: If True, add sinusoidal positional encodings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        add_positional_encoding: bool = False,
        positional_mode: str = "none",  # none|absolute|alibi|rope
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.add_pos_enc = add_positional_encoding
        self.positional_mode = positional_mode

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** 0.5

    def _positional_encoding(self, length: int, d_model: int, device) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device))) / d_model
        )
        pe = torch.zeros(length, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (T, D)

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, D)
            need_weights: if True, also return averaged attention weights (B, H, T, T)
            attn_mask: Optional mask (T, T) or (B, T, T) with -inf on masked positions
        Returns:
            y: (B, T, D)
            weights: (B, H, T, T) or None
        """
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {D}")
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # 位置编码/偏置
        if self.positional_mode in ("absolute",) or self.add_pos_enc:
            pe = self._positional_encoding(T, self.d_k, x.device)
            pe = pe.unsqueeze(0).unsqueeze(0)  # (1,1,T,d_k)
            q = q + pe
            k = k + pe
        elif self.positional_mode == "alibi":
            # ALiBi：对注意力得分加线性位置偏置（简化实现：每个头使用相同斜率）
            pass  # 在打分处添加
        elif self.positional_mode == "rope":
            # RoPE：对 q,k 进行旋转位置编码（简化：以最后维做二维旋转）
            q, k = _apply_rope(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if self.positional_mode == "alibi":
            bias = _alibi_bias(T, q.device)  # (1,1,T,T)
            attn_scores = attn_scores + bias
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)  # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_proj(context)
        y = self.dropout(y)
        return (y, attn_weights) if need_weights else (y, None)

def _alibi_bias(T: int, device) -> torch.Tensor:
    # 简化实现：单斜率，越远惩罚越大
    pos = torch.arange(T, device=device)
    rel = pos[None, :] - pos[:, None]  # (T,T)
    rel = -torch.abs(rel).float()
    return rel.view(1, 1, T, T)


def _apply_rope(q: torch.Tensor, k: torch.Tensor):
    # q,k: (B,H,T,d_k)
    def _rope(x):
        B,H,T,D = x.shape
        half = D // 2
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        pos = torch.arange(T, device=x.device).float()
        theta = 10000 ** (-torch.arange(0, half, device=x.device).float() / half)
        angle = pos[:, None] * theta[None, :]
        sin = torch.sin(angle)[None, None, ...]
        cos = torch.cos(angle)[None, None, ...]
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        return torch.cat([xr1, xr2, x[..., half*2:]], dim=-1)
    return _rope(q), _rope(k)


