from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as _weight_norm


class _CausalConv1d(nn.Module):
    """因果卷积封装：只在左侧做 padding，保持输出步长与输入一致。
    支持 dilation，使得感受野按指数增长。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,
                 bias: bool = True, use_weightnorm: bool = True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # 手动做不对称 padding
            dilation=dilation,
            bias=bias,
        )
        self.conv = _weight_norm(conv) if use_weightnorm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  -> 在左侧 pad
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """标准 TCN block：Conv(ks, dil) + Act + Norm + Dropout x 2 + 残差。
    - 因果卷积防止未来信息泄露
    - 残差分支用 1x1 调整通道数以匹配主分支
    - 归一化可选 BatchNorm1d
    - 权重归一化可选（施加于卷积层）
    - 可选门控激活（gated）：第二条分支作为门控，输出 y * sigmoid(gate)
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: int, dilation: int,
                 activation: str = "relu", dropout: float = 0.0,
                 use_batchnorm: bool = True, use_weightnorm: bool = True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_p = float(dropout)
        self.activation = activation.lower()
        self.gated = (self.activation == 'gated')

        self.conv1 = _CausalConv1d(in_c, out_c, kernel_size, dilation, use_weightnorm=use_weightnorm)
        self.bn1 = nn.BatchNorm1d(out_c) if use_batchnorm else nn.Identity()
        self.conv2 = _CausalConv1d(out_c, out_c, kernel_size, dilation, use_weightnorm=use_weightnorm)
        self.bn2 = nn.BatchNorm1d(out_c) if use_batchnorm else nn.Identity()

        if self.gated:
            self.gate1 = _CausalConv1d(in_c, out_c, kernel_size, dilation, use_weightnorm=use_weightnorm)
            self.gate2 = _CausalConv1d(out_c, out_c, kernel_size, dilation, use_weightnorm=use_weightnorm)
        else:
            self.gate1 = nn.Identity()
            self.gate2 = nn.Identity()

        self.drop1 = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.drop2 = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # 残差分支：若通道数不一致，用 1x1 卷积对齐
        self.downsample = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            # 门控：对 x 应用 sigmoid 做门控
            return torch.sigmoid(x)
        if self.activation == "relu":
            return F.relu(x, inplace=True)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation in ("silu", "swish"):
            return F.silu(x)
        return F.relu(x, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.conv1(x)
        if self.gated:
            g1 = self.gate1(x)
            y = y * torch.sigmoid(g1)
        y = self.bn1(y)
        y = self._act(y)
        y = self.drop1(y)

        y2 = self.conv2(y)
        if self.gated:
            g2 = self.gate2(y)
            y2 = y2 * torch.sigmoid(g2)
        y2 = self.bn2(y2)
        y2 = self._act(y2)
        y2 = self.drop2(y2)

        res = self.downsample(x)
        return y2 + res


class TCNFeatureExtractor(nn.Module):
    """
    Temporal Convolutional Network (TCN) 特征提取器。

    接口与 CNNFeatureExtractor 对齐：
      - __init__(in_channels, layer_configs, use_batchnorm=True, dropout=0.0)
      - forward(x) 接收 (B, T, F) 返回 (B, T, C)

    layer_configs: List[dict] 每层支持字段：
      - out_channels: int（必须）
      - kernel_size: int（默认 3）
      - dilation: int（若未提供，按指数 1,2,4,... 推断）
      - activation: str（relu|gelu|silu）
      - use_weightnorm: bool（默认 True）
    """
    def __init__(
        self,
        in_channels: int,
        layer_configs: List[dict],
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_batchnorm = use_batchnorm
        self.dropout_p = float(dropout)

        layers: List[nn.Module] = []
        current_c = in_channels
        # 若未显式给出 dilation，则采用 1,2,4,...
        default_dil = 1
        for i, cfg in enumerate(layer_configs):
            # 输出通道为必需字段
            raw_out = cfg.get("out_channels", None)
            if raw_out is None:
                raise ValueError(f"TCN layer缺少 out_channels: 第{i}层 cfg={cfg}")
            try:
                out_c = int(raw_out)
            except Exception:
                raise ValueError(f"TCN layer的 out_channels 非法: 第{i}层 cfg={cfg}")

            # kernel_size：None 或未提供时使用默认 3
            raw_k = cfg.get("kernel_size", None)
            k = int(raw_k) if raw_k is not None else 3

            # dilation：当为 None 或未提供时，采用默认指数增长序列
            raw_d = cfg.get("dilation", None)
            d = default_dil if (raw_d is None) else int(raw_d)

            # 激活：允许 None，回退 relu
            act = str(cfg.get("activation", None) or "relu").lower()

            # 权重归一化：允许 None，回退 True
            use_wn = cfg.get("use_weightnorm", None)
            use_wn = True if use_wn is None else bool(use_wn)

            block = TemporalBlock(
                in_c=current_c,
                out_c=out_c,
                kernel_size=k,
                dilation=d,
                activation=act,
                dropout=self.dropout_p,
                use_batchnorm=self.use_batchnorm,
                use_weightnorm=use_wn,
            )
            layers.append(block)
            current_c = out_c
            # 缺省时下一层 dilation 翻倍
            if "dilation" not in cfg or cfg.get("dilation") is None:
                default_dil = max(1, default_dil * 2)

        self.net = nn.Sequential(*layers)

    @property
    def out_channels(self) -> int:
        if not isinstance(self.net, nn.Sequential) or len(self.net) == 0:
            return self.in_channels
        # 最后一层的 out 通道
        for m in reversed(self.net):
            if isinstance(m, TemporalBlock):
                # 通过其第二个卷积的 out_channels 反推
                if hasattr(m, 'conv2') and hasattr(m.conv2, 'conv'):
                    return m.conv2.conv.out_channels  # type: ignore
        return self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got shape {tuple(x.shape)}")
        # 转为 (B, C, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        # 回到 (B, T, C)
        y = y.transpose(1, 2)
        return y

