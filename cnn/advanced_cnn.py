from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for efficient feature extraction.

    Based on MobileNet architecture, reduces parameters while maintaining performance.
    Particularly effective for time series where channel-wise and temporal patterns
    can be learned separately.

    Reference: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)
    Applied to time series in "Efficient Deep Learning for Time Series Forecasting" (2023)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: Optional[int] = None, dilation: int = 1,
                 bias: bool = True, activation: str = "relu"):
        super().__init__()

        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation

        # Depthwise convolution - learns spatial patterns for each channel separately
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )

        # Pointwise convolution - learns channel interactions
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Activation
        if activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        return x


class DilatedConvBlock(nn.Module):
    """
    Dilated Convolution Block for multi-scale temporal pattern capture.

    Uses different dilation rates to capture patterns at multiple time scales
    without increasing computational cost significantly.

    Reference: "WaveNet: A Generative Model for Raw Audio" (2016)
    Applied to time series in "Temporal Convolutional Networks for Action Segmentation" (2023)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation_rates: List[int] = [1, 2, 4], use_residual: bool = True,
                 activation: str = "relu", dropout: float = 0.1):
        super().__init__()

        # 仅当输入与输出通道一致时启用残差，避免形状不匹配
        self.use_residual = use_residual and (in_channels == out_channels)
        # 容错：dilation_rates 可能为 None 或空
        if not dilation_rates:
            dilation_rates = [1, 2, 4]
        self.dilation_rates = [int(d) for d in dilation_rates]
        self.out_channels = out_channels

        # 多分支膨胀卷积
        self.dilated_convs = nn.ModuleList()
        branch_channels = max(1, out_channels // max(1, len(self.dilation_rates)))
        for dilation in self.dilation_rates:
            padding = ((kernel_size - 1) // 2) * dilation
            conv = nn.Conv1d(
                in_channels, branch_channels,
                kernel_size=kernel_size, padding=padding, dilation=dilation
            )
            self.dilated_convs.append(conv)

        # 分支拼接后的通道数
        self.concat_channels = branch_channels * len(dilation_rates)

        # 归一化与激活
        self.bn = nn.BatchNorm1d(self.concat_channels)
        if activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 将拼接后的通道通过 1x1 卷积投影到精确的 out_channels，确保与外部声明一致
        if self.concat_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv1d(self.concat_channels, out_channels, kernel_size=1)

        # 残差投影（仅在启用残差且需要投影时使用）
        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用多分支膨胀卷积并拼接
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))

        out = torch.cat(dilated_outputs, dim=1)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        # 投影到精确的 out_channels
        out = self.proj(out)

        # 残差连接
        if self.use_residual:
            residual = self.residual_proj(x)
            out = out + residual

        return out


class AdvancedCNNFeatureExtractor(nn.Module):
    """
    Advanced CNN Feature Extractor with modern architectural improvements.

    Combines:
    1. Depthwise Separable Convolutions for efficiency
    2. Dilated Convolutions for multi-scale pattern capture
    3. Residual connections for better gradient flow
    4. Optional Channel Attention (ECA/SE)
    5. Optional Inception-style multi-branch blocks

    """

    def __init__(self, in_channels: int, layer_configs: List[dict],
                 use_batchnorm: bool = True, dropout: float = 0.0,
                 architecture_type: str = "standard",
                 use_channel_attention: bool = False,
                 channel_attention_type: str = "eca"):
        super().__init__()

        self.in_channels = in_channels
        self.use_batchnorm = use_batchnorm
        self.dropout_p = float(dropout)
        self.architecture_type = architecture_type.lower()
        self.use_ca = bool(use_channel_attention)
        self.ca_type = str(channel_attention_type).lower()

        layers: List[nn.Module] = []
        current_c = in_channels

        for i, cfg in enumerate(layer_configs):
            out_c = int(cfg.get("out_channels"))
            k = int(cfg.get("kernel_size", 3))
            s = int(cfg.get("stride", 1))
            activation = str(cfg.get("activation", "relu")).lower()

            if self.architecture_type == "depthwise":
                conv_block = self._build_depthwise_block(current_c, out_c, k, s, activation, cfg)
            elif self.architecture_type == "dilated":
                raw_rates = cfg.get("dilation_rates", None)
                if raw_rates is None:
                    dilation_rates = [1, 2, 4]
                else:
                    if isinstance(raw_rates, (list, tuple)):
                        dilation_rates = [int(x) for x in raw_rates if x is not None]
                    else:
                        dilation_rates = [int(raw_rates)]
                    if len(dilation_rates) == 0:
                        dilation_rates = [1, 2, 4]
                conv_block = self._build_dilated_block(current_c, out_c, k, dilation_rates, activation, cfg)
            elif self.architecture_type == "inception":
                conv_block = self._build_inception_block(current_c, out_c, k, activation, cfg)
            else:
                conv_block = self._build_standard_block(current_c, out_c, k, s, activation, cfg)

            # optional channel attention
            if self.use_ca:
                conv_block = nn.Sequential(conv_block, self._build_channel_attention(out_c))

            layers.append(conv_block)
            current_c = out_c

        self.net = nn.Sequential(*layers)

    def _build_depthwise_block(self, in_c: int, out_c: int, k: int, s: int,
                              activation: str, cfg: dict) -> nn.Module:
        block = []

        # Depthwise separable convolution
        dsconv = DepthwiseSeparableConv1d(
            in_c, out_c, kernel_size=k, stride=s, activation=activation
        )
        block.append(dsconv)

        if self.use_batchnorm:
            block.append(nn.BatchNorm1d(out_c))

        # Pooling
        pool = cfg.get("pool", "max")
        if pool:
            kpool = int(cfg.get("pool_kernel_size", 2))
            if str(pool).lower() == "max":
                block.append(nn.MaxPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))
            elif str(pool).lower() == "avg":
                block.append(nn.AvgPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))

        if self.dropout_p > 0:
            block.append(nn.Dropout(p=self.dropout_p))

        return nn.Sequential(*block)

    def _build_dilated_block(self, in_c: int, out_c: int, k: int,
                            dilation_rates: List[int], activation: str, cfg: dict) -> nn.Module:
        return DilatedConvBlock(
            in_c, out_c, kernel_size=k, dilation_rates=dilation_rates,
            activation=activation, dropout=self.dropout_p
        )

    def _build_standard_block(self, in_c: int, out_c: int, k: int, s: int,
                             activation: str, cfg: dict) -> nn.Module:
        block = []
        d = int(cfg.get("dilation", 1))
        p = ((k - 1) // 2) * d if cfg.get("padding") is None else int(cfg.get("padding"))
        conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d)
        block.append(conv)
        if self.use_batchnorm:
            block.append(nn.BatchNorm1d(out_c))
        if activation == "relu":
            block.append(nn.ReLU(inplace=True))
        elif activation == "gelu":
            block.append(nn.GELU())
        elif activation == "swish":
            block.append(nn.SiLU())
        pool = cfg.get("pool", "max")
        if pool:
            kpool = int(cfg.get("pool_kernel_size", 2))
            if str(pool).lower() == "max":
                block.append(nn.MaxPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))
            elif str(pool).lower() == "avg":
                block.append(nn.AvgPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))
        if self.dropout_p > 0:
            block.append(nn.Dropout(p=self.dropout_p))
        return nn.Sequential(*block)

    def _build_inception_block(self, in_c: int, out_c: int, k: int, activation: str, cfg: dict) -> nn.Module:
        """Inception-style block: multi-branch conv with varying kernel sizes/dilations + concat + BN + Act.
        分支设定：kernel_sizes（默认 [3,5,7]）、dilations（默认 [1,2]），最后 1x1 投影到 out_c。
        可通过 cfg['inception_kernel_sizes'] / cfg['inception_dilations'] 自定义。
        """
        ks = cfg.get('inception_kernel_sizes', [3, 5, 7])
        ds = cfg.get('inception_dilations', [1, 2])
        branches: List[nn.Module] = []
        # 每个分支输出通道近似均分
        n_br = max(1, len(ks) * len(ds))
        br_c = max(1, out_c // n_br)
        for kk in ks:
            for dd in ds:
                p = ((int(kk) - 1) // 2) * int(dd)
                branches.append(
                    nn.Conv1d(in_c, br_c, kernel_size=int(kk), padding=p, dilation=int(dd))
                )
        m = nn.ModuleList(branches)
        post = []
        post.append(nn.BatchNorm1d(br_c * n_br) if self.use_batchnorm else nn.Identity())
        if activation == 'relu':
            post.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            post.append(nn.GELU())
        elif activation == 'swish':
            post.append(nn.SiLU())
        post.append(nn.Conv1d(br_c * n_br, out_c, kernel_size=1))
        if self.dropout_p > 0:
            post.append(nn.Dropout(self.dropout_p))
        return nn.Sequential(
            _ParallelConcat1D(m),
            *post
        )

    def _build_channel_attention(self, channels: int) -> nn.Module:
        if self.ca_type == 'se':
            return _SEBlock(channels)
        return _ECA1D(channels)

    @property
    def out_channels(self) -> int:
        """Number of channels produced by the last conv layer."""
        if not isinstance(self.net, nn.Sequential) or len(self.net) == 0:
            return self.in_channels

        last_out = self.in_channels
        for m in self.net:
            if isinstance(m, nn.Sequential):
                for sub in m:
                    if isinstance(sub, nn.Conv1d):
                        last_out = sub.out_channels
                    elif isinstance(sub, DepthwiseSeparableConv1d):
                        # depthwise separable conv ends with pointwise conv with out_channels
                        last_out = sub.pointwise.out_channels
            elif isinstance(m, nn.Conv1d):
                last_out = m.out_channels
        return last_out

class _ParallelConcat1D(nn.Module):
    def __init__(self, branches: nn.ModuleList):
        super().__init__()
        self.branches = branches
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        return torch.cat(outs, dim=1)

class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, max(1, channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, channels // reduction), channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class _ECA1D(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)               # (B,C,1)
        y = y.transpose(1,2)           # (B,1,C)
        y = self.conv(y)               # (B,1,C)
        y = self.sigmoid(y).transpose(1,2)  # (B,C,1)
        return x * y


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, features)
        Returns:
            Tensor (batch, seq_len, out_channels)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got shape {tuple(x.shape)}")

        # Convert to (B, C, T) for conv1d
        x = x.transpose(1, 2)
        y = self.net(x)
        # Convert back to (B, T, C)
        y = y.transpose(1, 2)
        return y
