"""Loss formulations for class-imbalance handling.

* :class:`FocalLoss`      -- binary focal loss  ``L = -alpha_t (1 - p_t)^gamma log(p_t)``
* :class:`WeightedBCELoss`-- standard BCE with a fixed training-fold
  ``pos_weight``.

Both operate on raw logits for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss computed directly from logits.

    Parameters
    ----------
    alpha:
        Class-balancing weight applied to the positive class (``alpha_t`` is
        ``alpha`` for positives and ``1 - alpha`` for negatives).
    gamma:
        Focusing parameter; ``gamma > 0`` down-weights easy examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0; got {gamma}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        if sample_weights is not None:
            loss = loss * sample_weights
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """BCE-with-logits with a fixed, training-fold class weight.

    Computing the weight per mini-batch makes the objective fluctuate with
    batch composition.  A fixed fold-level value is stable and preserves the
    intended class balance.
    """

    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class BCELoss(WeightedBCELoss):
    """Unweighted BCE-with-logits with optional source/sample weighting."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(pos_weight=1.0, reduction=reduction)


def build_loss(loss_type: str, focal_gamma: float = 2.0, focal_alpha: float = 0.25,
               pos_weight: float = 1.0) -> nn.Module:
    """Instantiate the configured criterion."""
    loss_type = loss_type.lower().strip()
    if loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if loss_type == "wbce":
        return WeightedBCELoss(pos_weight=pos_weight)
    if loss_type == "bce":
        return BCELoss()
    raise ValueError(f"Unknown loss type '{loss_type}'; expected 'focal', 'wbce', or 'bce'.")
