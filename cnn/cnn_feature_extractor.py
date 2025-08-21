from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    Temporal 1D CNN feature extractor for multivariate time series.

    Expects input shape (batch, seq_len, num_features) and returns a tensor
    of shape (batch, seq_len, channels_last) aligned per time-step, enabling
    subsequent recurrent/attention layers. Uses causal padding by default to
    preserve sequence length.

    Args:
        in_channels: Number of input features.
        layer_configs: List of dicts or simple namespace with fields
            out_channels, kernel_size, stride, padding, dilation, activation,
            pool ("max"/"avg"/None), pool_kernel_size.
        use_batchnorm: Whether to apply BatchNorm1d after conv.
        dropout: Dropout probability applied after activation/pooling.
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
        for cfg in layer_configs:
            out_c = int(cfg.get("out_channels"))
            k = int(cfg.get("kernel_size", 3))
            s = int(cfg.get("stride", 1))
            d = int(cfg.get("dilation", 1))
            # same padding on length with dilation
            if cfg.get("padding") is None:
                p = ((k - 1) // 2) * d
            else:
                p = int(cfg.get("padding"))

            conv = nn.Conv1d(current_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d)
            block: List[nn.Module] = [conv]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_c))

            act_name = str(cfg.get("activation", "relu")).lower()
            if act_name == "relu":
                block.append(nn.ReLU(inplace=True))
            elif act_name == "gelu":
                block.append(nn.GELU())
            elif act_name == "elu":
                block.append(nn.ELU(inplace=True))
            else:
                raise ValueError(f"Unsupported activation: {act_name}")

            pool = cfg.get("pool", "max")
            if pool:
                kpool = int(cfg.get("pool_kernel_size", 2))
                if str(pool).lower() == "max":
                    block.append(nn.MaxPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))
                elif str(pool).lower() == "avg":
                    block.append(nn.AvgPool1d(kernel_size=kpool, stride=1, padding=(kpool - 1) // 2))
                else:
                    raise ValueError(f"Unsupported pool: {pool}")

            if self.dropout_p > 0:
                block.append(nn.Dropout(p=self.dropout_p))

            layers.append(nn.Sequential(*block))
            current_c = out_c

        self.net = nn.Sequential(*layers)

    @property
    def out_channels(self) -> int:
        """Number of channels produced by the last conv layer."""
        if not isinstance(self.net, nn.Sequential) or len(self.net) == 0:
            return self.in_channels
        # last Conv1d out_channels
        for m in reversed(self.net):
            for sub in m:
                if isinstance(sub, nn.Conv1d):
                    return sub.out_channels
        return self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, features)
        Returns:
            Tensor (batch, seq_len, out_channels)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got shape {tuple(x.shape)}")
        # to (B, C, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        # back to (B, T, C)
        y = y.transpose(1, 2)
        return y

