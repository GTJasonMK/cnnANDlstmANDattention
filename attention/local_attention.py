from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSelfAttention(nn.Module):
    """
    局部窗口自注意力（支持膨胀/dilated窗口）。
    - 输入: (B, T, D)
    - 输出: (B, T, D)
    参数：
      d_model: 通道维
      num_heads: 注意力头数
      dropout: 丢弃率
      window_size: 窗口大小（奇数更合适）
      dilation: 膨胀步长（>=1）
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 window_size: int = 64, dilation: int = 1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window = int(window_size)
        self.dilation = max(1, int(dilation))

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5

    def forward(self, x: torch.Tensor, need_weights: bool = False,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # (B,H,T,d_k)
        k = k.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # 构造局部窗口索引
        radius = self.window // 2
        idxs = torch.arange(T, device=x.device)
        attn_w_list = []
        ctx = torch.zeros(B, self.num_heads, T, self.d_k, device=x.device)
        for t in range(T):
            # 中心 t 的窗口 [t - radius*dil, t + radius*dil]，步长=dilation
            left = max(0, t - radius * self.dilation)
            right = min(T - 1, t + radius * self.dilation)
            r = torch.arange(left, right + 1, self.dilation, device=x.device)
            qt = q[:, :, t:t+1, :]                # (B,H,1,d_k)
            kt = torch.index_select(k, 2, r)      # (B,H,W,d_k)
            vt = torch.index_select(v, 2, r)      # (B,H,W,d_k)
            scores = torch.matmul(qt, kt.transpose(-2, -1)) / self.scale  # (B,H,1,W)
            weights = torch.softmax(scores, dim=-1)
            if need_weights:
                # 归一化到全局 T：这里仅返回窗口注意力，可在外部拼接或忽略
                attn_w_list.append(weights)
            ct = torch.matmul(weights, vt)  # (B,H,1,d_k)
            ctx[:, :, t, :] = ct.squeeze(2)

        out = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out(out)
        out = self.drop(out)
        if need_weights:
            # 拼接为 (B,H,T,maxW) 的局部权重，仅供调试
            try:
                W = max(w.shape[-1] for w in attn_w_list) if len(attn_w_list) else 0
                if W > 0:
                    pad_w = [F.pad(w, (0, W - w.shape[-1])) for w in attn_w_list]
                    ws = torch.cat(pad_w, dim=2)  # (B,H,T,W)
                else:
                    ws = None
            except Exception:
                ws = None
            return out, ws
        return out, None

