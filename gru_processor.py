from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GRUProcessor(nn.Module):
    """
    GRU module for processing per-step features into temporal representations.

    Interface mirrors LSTMProcessor for drop-in replacement.

    Args:
        input_size: Feature dimension.
        hidden_size: GRU hidden size per direction.
        num_layers: Number of stacked GRU layers.
        bidirectional: If True, use BiGRU.
        dropout: Dropout between GRU layers (PyTorch semantics).
        return_sequence: If True, return (B, T, H) sequence; else only last step (B, H).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
        return_sequence: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.return_sequence = bool(return_sequence)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F)
            hx: Optional initial hidden state (num_layers * num_directions, B, hidden_size)
        Returns:
            outputs: (B, T, H) if return_sequence else (B, H)
            hn: final hidden state
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got {tuple(x.shape)}")
        outputs, hn = self.gru(x, hx) if hx is not None else self.gru(x)
        if self.return_sequence:
            return outputs, hn
        else:
            last = outputs[:, -1, :]
            return last, hn

