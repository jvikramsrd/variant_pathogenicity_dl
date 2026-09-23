"""Multi-task losses. Each term only sees the rows that actually carry its supervision.

    classify   cross-entropy against a probability target: a one-hot class, or 0.5/0.5 for
               ClinVar's "Pathogenic/Likely pathogenic" and "Benign/Likely benign" (never
               forced to one side). Optional class weights and label smoothing.
    ordinal    squared earth-mover's distance between predicted and target distributions on
               the ordered scale P > LP > VUS > LB > B: calling a benign variant "pathogenic"
               costs more than calling it "likely benign". Off by default (weight 0).
    acmg       binary cross-entropy per code, only on documents that state codes, only on
               codes applicable to the gene (vpdl.slm.text.acmg.GENE_SPECS)
    types      binary cross-entropy per evidence type on evidence units (weak labels)
    polarity   cross-entropy on units with a polarity label (weak labels)

Weak-label terms are down-weighted by default and are an ablation, not an
assumption (docs/slm/GENOMIC_SLM_EXPERIMENT_PLAN.md, EXP-012).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

__all__ = ["LossWeights", "soft_cross_entropy", "ordinal_emd", "masked_bce", "multitask_loss"]


@dataclass
class LossWeights:
    classify: float = 1.0
    ordinal: float = 0.0
    acmg: float = 0.5
    types: float = 0.3
    polarity: float = 0.3
    label_smoothing: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def soft_cross_entropy(logits, target, weight=None, label_smoothing: float = 0.0):
    import torch
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    if label_smoothing:
        target = target * (1 - label_smoothing) + label_smoothing / target.size(-1)
    per_row = -(target * log_probs)
    if weight is not None:
        per_row = per_row * weight.unsqueeze(0)
    return per_row.sum(-1).mean()


def ordinal_emd(logits, target):
    import torch
    probs = torch.softmax(logits.float(), dim=-1)
    return ((probs.cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1).mean()


def masked_bce(logits, target, mask):
    import torch
    mask = mask.float()
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                                                reduction="none")
    return (loss * mask).sum() / mask.sum()


def multitask_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: LossWeights,
                   class_weight=None, unit_outputs: Mapping[str, Any] | None = None):
    """(total, parts) — parts are floats for logging."""
    import torch
    parts: dict[str, float] = {}
    terms = []
    target = batch["class_target"]
    has_target = target.sum(-1) > 0
    if has_target.any() and weights.classify:
        loss = soft_cross_entropy(outputs["class_logits"][has_target], target[has_target],
                                  class_weight, weights.label_smoothing)
        terms.append(weights.classify * loss)
        parts["classify"] = float(loss.detach())
        if weights.ordinal:
            emd = ordinal_emd(outputs["class_logits"][has_target], target[has_target])
            terms.append(weights.ordinal * emd)
            parts["ordinal"] = float(emd.detach())
    if "acmg_logits" in outputs and weights.acmg and "acmg_target" in batch:
        mask = batch["acmg_supervised"].unsqueeze(-1).float() * batch["acmg_applicable"]
        loss = masked_bce(outputs["acmg_logits"], batch["acmg_target"], mask)
        terms.append(weights.acmg * loss)
        parts["acmg"] = float(loss.detach())
    if unit_outputs is not None and "unit_types" in batch:
        if weights.types:
            loss = masked_bce(unit_outputs["type_logits"], batch["unit_types"],
                              torch.ones_like(batch["unit_types"]))
            terms.append(weights.types * loss)
            parts["types"] = float(loss.detach())
        if weights.polarity:
            valid = batch["unit_polarity"] >= 0
            if valid.any():
                loss = torch.nn.functional.cross_entropy(
                    unit_outputs["polarity_logits"][valid].float(), batch["unit_polarity"][valid])
                terms.append(weights.polarity * loss)
                parts["polarity"] = float(loss.detach())
    if not terms:
        zero = outputs["class_logits"].sum() * 0.0
        return zero, parts
    total = torch.stack([t.float() for t in terms]).sum()
    parts["total"] = float(total.detach())
    return total, parts
