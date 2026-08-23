"""MLP classification head on top of frozen ESM-2 features.

Architecture (per specification):

    Linear(d_in -> d_hidden)
    Residual block:
        x -> Linear -> LayerNorm -> GELU -> Dropout(p) -> (+ residual add back to x)
    Linear(d_hidden -> 1 logit)

The residual connection requires matching dimensions, so the input projection
first maps the concatenated ESM-2 feature vector into the hidden width and the
residual stream lives entirely in that hidden space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """``x + Dropout(GELU(LayerNorm(Linear(x))))`` with matched in/out dims."""

    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.norm(self.fc(x))))


class VariantPathogenicityMLP(nn.Module):
    """Residual MLP producing a single pathogenicity logit per variant."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        n_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(max(1, n_blocks))]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)
