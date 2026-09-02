"""Fusion architectures for multi-modal variant pathogenicity (Phase 5 start).

PROJECT_PLAN.md Phase 5 fixes the starting architecture — freeze both branch
encoders, project each branch to a shared dimension, then fuse with a shallow
head — and mandates a GateWave-style gated alternative (from the MVmamba base
paper) plus mandatory branch-only baselines.

Heads implemented here (all operate on pre-extracted frozen features):

* :class:`BranchHead`            — single-branch residual MLP baseline.
* :class:`ConcatFusionHead`      — the plan's default: BatchNorm -> Linear ->
  ReLU -> Dropout -> Linear over the concatenated projections.
* :class:`GateWaveFusionHead`    — MVmamba's design adapted to two modality
  feature vectors: sigmoid gate balancing the branches + softmax dynamic
  per-feature gating + Gated Linear Unit + residual connection.

The second "modality" in our current pipeline is the external-prior branch
(published-model scores incl. REVEL/EVE/CADD/AlphaMissense + gnomAD AF); the
LLM clinical-text branch plugs into the exact same interface later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import ResidualMLPBlock


class BranchProjection(nn.Module):
    """Linear projection of one frozen-branch feature vector to shared dim."""

    def __init__(self, input_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BranchHead(nn.Module):
    """Single-branch baseline: projection + residual MLP block(s) + score."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 dropout: float = 0.15, n_blocks: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *[ResidualMLPBlock(hidden_dim, dropout=dropout)
              for _ in range(max(1, n_blocks))],
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(-1)


class ConcatFusionHead(nn.Module):
    """Plan-default fusion: concat -> BN -> Linear -> ReLU -> Dropout -> Linear.

    Both inputs are first projected to ``shared_dim``; the shallow MLP head is
    deliberately small to avoid overfitting on the few-thousand-row MMR data.
    """

    def __init__(self, dims: tuple[int, ...], shared_dim: int = 128,
                 dropout: float = 0.2, norm: str = "batch") -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [BranchProjection(d, shared_dim) for d in dims])
        width = shared_dim * len(dims)
        # The attribute stays named `bn` whichever norm it holds: renaming it
        # would change every existing Stage-2 checkpoint's state-dict keys.
        # "layer" exists for the Stage-2b fine-tune, which runs at micro-batch
        # 1 on a 15 GiB card -- BatchNorm1d cannot compute batch statistics
        # over a single sample and raises in train mode.
        if norm == "batch":
            self.bn: nn.Module = nn.BatchNorm1d(width)
        elif norm == "layer":
            self.bn = nn.LayerNorm(width)
        else:
            raise ValueError(f"norm must be 'batch' or 'layer'; got {norm!r}")
        self.norm_kind = norm
        self.fc1 = nn.Linear(width, shared_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(shared_dim, 1)

    def forward(self, *branch_inputs: torch.Tensor) -> torch.Tensor:
        if len(branch_inputs) != len(self.branches):
            raise ValueError(f"Expected {len(self.branches)} branch tensors, "
                             f"got {len(branch_inputs)}.")
        projected = [proj(x) for proj, x in zip(self.branches, branch_inputs)]
        h = torch.cat(projected, dim=-1)
        h = self.drop(self.act(self.fc1(self.bn(h))))
        return self.fc2(h).squeeze(-1)


class GateWaveFusionHead(nn.Module):
    """MVmamba GateWave adapted to two related modality feature vectors.

    Pipeline (per the base paper's bi-modal gate, here ESM vs prior branch):

    1. Each branch is projected to ``shared_dim``.
    2. A **sigmoid gate** computes scalar branch weights from the mean-pooled
       branch vectors and rebalances the two streams (analogous to their
       WT-vs-VT balance gate).
    3. A **softmax dynamic gate** produces per-dimension weights from the
       concatenated streams, weighting each modality's features individually.
    4. A **Gated Linear Unit** (two linear towers with sigmoid gate) mixes the
       reweighted concatenation.
    5. **Residual connection**: GLU output is added back to the projected
       concatenation before the final score.

    The plan cites the paper's ablation as the reason this exists: removing
    the gate cost ~0.001 AUC, removing the wavelet component ~0.007 AUC on
    18,731 clinical variants — quantified evidence for how much complexity is
    warranted. We reproduce that ablation on our own data via the
    ``use_gate`` / ``use_glu`` switches.
    """

    def __init__(self, dims: tuple[int, ...], shared_dim: int = 128,
                 dropout: float = 0.2,
                 use_gate: bool = True, use_glu: bool = True) -> None:
        super().__init__()
        if len(dims) != 2:
            raise ValueError("GateWave fusion expects exactly two branches "
                             "(MVmamba's WT/VT analogue: esm vs prior).")
        self.use_gate = bool(use_gate)
        self.use_glu = bool(use_glu)
        self.shared_dim = shared_dim
        self.branches = nn.ModuleList(
            [BranchProjection(d, shared_dim) for d in dims])

        if self.use_gate:
            # Sigmoid balance gate over mean-pooled branch summaries.
            self.balance_gate = nn.Linear(shared_dim * len(dims), len(dims))
        # Softmax dynamic per-feature gating across modalities.
        self.dynamic_gate = nn.Linear(shared_dim * len(dims),
                                      shared_dim * len(dims))
        if self.use_glu:
            self.glu_value = nn.Linear(shared_dim * len(dims), shared_dim)
            self.glu_gate = nn.Linear(shared_dim * len(dims), shared_dim)
        else:
            self.plain_mix = nn.Linear(shared_dim * len(dims), shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(shared_dim, 1)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        a = self.branches[0](x_a)
        b = self.branches[1](x_b)
        concat = torch.cat([a, b], dim=-1)

        if self.use_gate:
            weights = torch.softmax(self.balance_gate(concat), dim=-1)
            a = a * weights[:, 0:1]
            b = b * weights[:, 1:2]

        gated_concat = concat * torch.softmax(
            self.dynamic_gate(concat).view(-1, 2, self.shared_dim), dim=1
        ).view(-1, 2 * self.shared_dim)

        if self.use_glu:
            mixed = self.glu_value(gated_concat) * torch.sigmoid(
                self.glu_gate(gated_concat))
        else:
            mixed = self.plain_mix(gated_concat)

        h = self.norm(mixed + (a + b) / 2.0)   # residual connection
        h = self.drop(h)
        return self.out(h).squeeze(-1)


def build_fusion_head(name: str, dims: tuple[int, ...], shared_dim: int = 128,
                      dropout: float = 0.2, norm: str = "batch") -> nn.Module:
    """Factory used by the transfer/fusion benchmark CLI.

    ``norm`` applies to the concat head only -- GateWave already normalises
    with :class:`~torch.nn.LayerNorm`, so it is safe at micro-batch 1 as-is.
    """
    name = name.lower().strip()
    if name == "concat":
        return ConcatFusionHead(dims, shared_dim, dropout, norm=norm)
    if name == "gatewave":
        return GateWaveFusionHead(dims, shared_dim, dropout)
    raise ValueError(f"Unknown fusion head '{name}' (expected concat|gatewave).")


__all__ = [
    "BranchProjection", "BranchHead", "ConcatFusionHead",
    "GateWaveFusionHead", "build_fusion_head",
]
