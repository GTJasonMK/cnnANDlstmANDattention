from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from cnn_feature_extractor import CNNFeatureExtractor
from lstm_processor import LSTMProcessor
from gru_processor import GRUProcessor
from attention_mechanism import MultiHeadSelfAttention
from advanced_cnn import AdvancedCNNFeatureExtractor
from improved_attention import MultiScaleTemporalAttention


class CNNLSTMAttentionModel(nn.Module):
    """
    End-to-end CNN + LSTM + (optional) Self-Attention model for multivariate TS forecasting.

    Input: (B, T, F)
    Output: (B, horizon) for univariate target or (B, horizon, n_targets)

    Args:
        num_features: Input feature dimension.
        cnn_layers: List of CNN layer configs.
        lstm_hidden: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        bidirectional: Whether to use BiLSTM.
        attn_heads: Number of attention heads (if attention enabled).
        attn_dropout: Attention dropout.
        attn_enabled: Whether to include attention block.
        fc_hidden: Hidden size for final MLP head.
        forecast_horizon: Number of steps to predict.
        n_targets: Number of target variables. Default 1.
    """

    def __init__(
        self,
        num_features: int,
        cnn_layers: List[dict],
        use_batchnorm: bool,
        cnn_dropout: float,
        lstm_hidden: int,
        lstm_layers: int,
        bidirectional: bool,
        attn_enabled: bool,
        attn_heads: int,
        attn_dropout: float,
        fc_hidden: int,
        forecast_horizon: int,
        n_targets: int = 1,
        attn_add_pos_enc: bool = False,
        lstm_dropout: Optional[float] = None,
        cnn_variant: str = "standard",  # "standard"|"depthwise"|"dilated"
        attn_variant: str = "standard",  # "standard"|"multiscale"
        multiscale_scales: Optional[List[int]] = None,
        multiscale_fuse: str = "sum",
    ) -> None:
        super().__init__()
        self.horizon = int(forecast_horizon)
        self.n_targets = int(n_targets)
        self.attn_enabled = bool(attn_enabled)

        # CNN backbone variant
        if (cnn_variant or "standard").lower() == "standard":
            self.cnn = CNNFeatureExtractor(
                in_channels=num_features,
                layer_configs=cnn_layers,
                use_batchnorm=use_batchnorm,
                dropout=cnn_dropout,
            )
        else:
            self.cnn = AdvancedCNNFeatureExtractor(
                in_channels=num_features,
                layer_configs=cnn_layers,
                use_batchnorm=use_batchnorm,
                dropout=cnn_dropout,
                architecture_type=cnn_variant,
            )

        # Recurrent backbone: LSTM (default) or GRU (via cnn_variant "gru"? use attn_variant? We'll use new param rnn_type)
        rnn_type = getattr(self, 'rnn_type', 'lstm')
        if rnn_type.lower() == 'gru':
            self.rnn = GRUProcessor(
                input_size=self.cnn.out_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                bidirectional=bidirectional,
                dropout=(cnn_dropout if lstm_dropout is None else float(lstm_dropout)),
                return_sequence=True,
            )
        else:
            self.rnn = LSTMProcessor(
                input_size=self.cnn.out_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                bidirectional=bidirectional,
                dropout=(cnn_dropout if lstm_dropout is None else float(lstm_dropout)),
                return_sequence=True,
            )
        attn_dim = self.rnn.output_size
        if self.attn_enabled:
            if (attn_variant or "standard").lower() == "multiscale":
                self.attn = MultiScaleTemporalAttention(
                    d_model=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
                    scales=multiscale_scales or [1, 2], fuse=multiscale_fuse
                )
            else:
                self.attn = MultiHeadSelfAttention(
                    d_model=attn_dim, num_heads=attn_heads, dropout=attn_dropout, add_positional_encoding=attn_add_pos_enc
                )
        else:
            self.attn = None

        self.head = nn.Sequential(
            nn.Linear(attn_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cnn_dropout),
            nn.Linear(fc_hidden, self.horizon * self.n_targets),
        )

        # Save variants for reference/serialization
        self.cnn_variant = cnn_variant
        self.attn_variant = attn_variant
        self.multiscale_scales = multiscale_scales or [1, 2]
        self.multiscale_fuse = multiscale_fuse
        self.rnn_type = 'lstm'  # default; Can be switched via config and main

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.dim() != 3:
            raise ValueError(f"Expected input (B, T, F), got {tuple(x.shape)}")
        y = self.cnn(x)  # (B, T, C)
        y, _ = self.rnn(y)  # (B, T, H)
        attn_w = None
        if self.attn is not None:
            y, attn_w = self.attn(y, need_weights=return_attn)
        # take last step representation for forecasting head
        last = y[:, -1, :]
        out = self.head(last)  # (B, horizon*n_targets)
        if self.n_targets == 1:
            out = out.view(-1, self.horizon)
        else:
            out = out.view(-1, self.horizon, self.n_targets)
        return (out, attn_w) if return_attn else out

