from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from cnn.cnn_feature_extractor import CNNFeatureExtractor
from rnn.lstm_processor import LSTMProcessor
from rnn.gru_processor import GRUProcessor
from rnn.ssm_processor import SSMProcessor
from attention.attention_mechanism import MultiHeadSelfAttention
from cnn.advanced_cnn import AdvancedCNNFeatureExtractor
from attention.improved_attention import MultiScaleTemporalAttention
from attention.local_attention import LocalSelfAttention
from attention.conformer_block import ConformerBlock
from attention.spatiotemporal_attention import SpatialTemporalAttention
from cnn.tcn_feature_extractor import TCNFeatureExtractor

from normalization.revin import RevIN
from preprocess.decomposition import MovingAverageDecomposition


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
        cnn_variant: str = "standard",  # "standard"|"depthwise"|"dilated"|"tcn"
        attn_variant: str = "standard",  # "standard"|"multiscale"
        multiscale_scales: Optional[List[int]] = None,
        multiscale_fuse: str = "sum",
        # 新增：注意力位置编码模式、本地注意力参数、CNN通道注意力
        attn_positional_mode: str = "none",
        local_window_size: int = 64,
        local_dilation: int = 1,
        # 时空注意力参数（可选，默认安全值）
        st_mode: str = "serial",
        st_fuse: str = "sum",
        cnn_use_channel_attention: bool = False,
        cnn_channel_attention_type: str = "eca",
        # 第二阶段：可选归一化与分解配置（字典，向后兼容）
        normalization: Optional[dict] = None,
        decomposition: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.horizon = int(forecast_horizon)
        self.n_targets = int(n_targets)
        self.attn_enabled = bool(attn_enabled)

        # CNN/TCN 特征提取骨干
        _backbone = (cnn_variant or "standard").lower()
        if _backbone == "standard":
            self.cnn = CNNFeatureExtractor(
                in_channels=num_features,
                layer_configs=cnn_layers,
                use_batchnorm=use_batchnorm,
                dropout=cnn_dropout,
            )
        elif _backbone in ("depthwise", "dilated", "inception"):
            self.cnn = AdvancedCNNFeatureExtractor(
                in_channels=num_features,
                layer_configs=cnn_layers,
                use_batchnorm=use_batchnorm,
                dropout=cnn_dropout,
                architecture_type=_backbone,
                use_channel_attention=cnn_use_channel_attention,
                channel_attention_type=cnn_channel_attention_type,
            )
        elif _backbone == "tcn":
            self.cnn = TCNFeatureExtractor(
                in_channels=num_features,
                layer_configs=cnn_layers,
                use_batchnorm=use_batchnorm,
                dropout=cnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported cnn/tcn variant: {cnn_variant}")

        # Recurrent backbone: LSTM (default) or GRU/SSM via config
        rnn_type = getattr(self, 'rnn_type', 'lstm')
        rnn_type = (rnn_type or 'lstm').lower()
        if rnn_type == 'gru':
            rnn_cls = GRUProcessor
        elif rnn_type in ('ssm','mamba','ssm-mamba'):
            rnn_cls = SSMProcessor
        else:
            rnn_cls = LSTMProcessor
        # 计算实际 CNN 输出维度（更稳健）：通过一次小样本前向推断确定维度
        try:
            with torch.no_grad():
                _probe = torch.zeros(1, 8, num_features)
                _cnn_out = self.cnn(_probe)
                cnn_out_dim = int(_cnn_out.size(-1))
        except Exception:
            # 回退到静态属性
            cnn_out_dim = int(getattr(self.cnn, 'out_channels', num_features))
        self.rnn = rnn_cls(
            input_size=cnn_out_dim,
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
                _variant = (attn_variant or "standard").lower()
                if _variant == "local":
                    self.attn = LocalSelfAttention(
                        d_model=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
                        window_size=local_window_size, dilation=local_dilation,
                    )
                elif _variant == "conformer":
                    # 用 N 层轻量 Conformer 叠加（此处用1层；可在配置中扩展层数）
                    self.attn = nn.Sequential(ConformerBlock(attn_dim, attn_heads))
                elif _variant == "spatiotemporal":
                    self.attn = SpatialTemporalAttention(
                        d_model=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
                        mode=st_mode,
                        fuse=st_fuse,
                    )
                else:
                    self.attn = MultiHeadSelfAttention(
                        d_model=attn_dim, num_heads=attn_heads, dropout=attn_dropout,
                        add_positional_encoding=attn_add_pos_enc, positional_mode=attn_positional_mode,
                    )
        else:
            self.attn = None

        self.head = nn.Sequential(
            nn.Linear(attn_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cnn_dropout),
            nn.Linear(fc_hidden, self.horizon * self.n_targets),
        )
        # 可选 RevIN 与趋势分解模块（作为包装层）
        self._use_revin = bool((normalization or {}).get('revin', {}).get('enabled', False))
        if self._use_revin:
            self.revin = RevIN(d_model=num_features)
        else:
            self.revin = None
        decomp_cfg = decomposition or {}
        self._use_decomp = bool(decomp_cfg.get('enabled', False))
        if self._use_decomp:
            method = str(decomp_cfg.get('method', 'ma')).lower()
            kernel = int(decomp_cfg.get('kernel', 25))
            alpha = float(decomp_cfg.get('alpha', 0.3))
            self.decomp = MovingAverageDecomposition(kernel=kernel, method=method, alpha=alpha)
            # 两路共享同一骨干，输出后再融合（concat + 线性）
            self.trend_backbone = self.cnn
            self.resid_backbone = self.cnn
            self.fuse = nn.Linear(attn_dim * 2, attn_dim)
        else:
            self.decomp = None
            self.trend_backbone = None
            self.resid_backbone = None
            self.fuse = None

        # Save variants for reference/serialization
        self.cnn_variant = cnn_variant
        self.attn_variant = attn_variant
        self.multiscale_scales = multiscale_scales or [1, 2]
        self.multiscale_fuse = multiscale_fuse
        self.rnn_type = 'lstm'  # default; Can be switched via config and main

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.dim() != 3:
            raise ValueError(f"Expected input (B, T, F), got {tuple(x.shape)}")
        # 可选 RevIN 标准化
        if self.revin is not None:
            x = self.revin(x)
        # 可选趋势分解
        if self.decomp is not None:
            resid, trend = self.decomp(x)
            yr = self.resid_backbone(resid)
            yt = self.trend_backbone(trend)
            yr, _ = self.rnn(yr)
            yt, _ = self.rnn(yt)
            y = torch.cat([yr, yt], dim=-1)
            y = self.fuse(y)
        else:
            y = self.cnn(x)  # (B, T, C)
            y, _ = self.rnn(y)  # (B, T, H)
        attn_w = None
        if self.attn is not None:
            # 注意：ConformerBlock 返回 (B,T,D)，MultiHead/Local 返回 (y, weights)
            if isinstance(self.attn, nn.Sequential) and isinstance(self.attn[0], ConformerBlock):
                y = self.attn(y)
            else:
                y, attn_w = self.attn(y, need_weights=return_attn)
        # take last step representation for forecasting head
        last = y[:, -1, :]
        out = self.head(last)  # (B, horizon*n_targets)
        if self.n_targets == 1:
            out = out.view(-1, self.horizon)
        else:
            out = out.view(-1, self.horizon, self.n_targets)
        return (out, attn_w) if return_attn else out

