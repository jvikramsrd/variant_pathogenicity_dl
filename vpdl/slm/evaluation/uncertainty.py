"""Uncertainty that is measured, not decorative, and the model's right to abstain.

Signals (per example):

    entropy               of the calibrated five-class distribution
    max_prob, margin      top-1 probability; gap to the runner-up
    mutual_information    disagreement between stochastic passes (MC dropout) or ensemble
                          members: total entropy minus mean member entropy — the part of
                          the uncertainty that more training data could reduce
    score_std             spread of the pathogenic-vs-benign score across passes/members
    evidence_count        informative evidence units in the input (non-neutral polarity):
                          "no evidence" is a reason to abstain even when the classifier is
                          confident (it may be confident about a template)

An uncertainty is useful only if it predicts error. :func:`evaluate_uncertainty`
reports how well each signal separates wrong from right predictions (AUROC)
and the risk-coverage curve: error rate among the most-confident fraction of
examples, for every coverage level, summarised as AURC (lower is better).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vpdl.evaluate import roc_auc

__all__ = ["entropy", "max_prob", "margin", "mutual_information", "score_std", "risk_coverage",
           "evaluate_uncertainty", "abstain", "ABSTAIN_REASONS"]

_EPS = 1e-12
ABSTAIN_REASONS = ("low_confidence", "high_disagreement", "no_informative_evidence")


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1)
    return -(p * np.log(p)).sum(-1)


def max_prob(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs, dtype=float).max(-1)


def margin(probs: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probs, dtype=float), axis=-1)
    return ordered[..., -1] - ordered[..., -2]


def mutual_information(samples: np.ndarray) -> np.ndarray:
    """samples [S, N, C] -> [N]: entropy of the mean minus mean entropy (>= 0)."""
    samples = np.asarray(samples, dtype=float)
    return np.maximum(entropy(samples.mean(0)) - entropy(samples).mean(0), 0.0)


def score_std(samples: np.ndarray) -> np.ndarray:
    from vpdl.slm.evaluation.metrics import pathogenic_score
    samples = np.asarray(samples, dtype=float)
    return np.stack([pathogenic_score(s) for s in samples]).std(0)


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> dict[str, Any]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-confidence, kind="stable")
    errors = (~correct[order]).astype(float)
    n = len(errors)
    if n == 0:
        return {"aurc": float("nan"), "curve": []}
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    points = [{"coverage": float(c), "risk": float(r)} for c, r in
              zip(coverage[np.linspace(0, n - 1, min(n, 21)).astype(int)],
                  risk[np.linspace(0, n - 1, min(n, 21)).astype(int)])]
    return {"aurc": float(risk.mean()), "curve": points,
            "risk_at_full_coverage": float(risk[-1])}


def evaluate_uncertainty(probs: np.ndarray, y: Sequence[int],
                         signals: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    keep = y >= 0
    probs, y = probs[keep], y[keep]
    wrong = (probs.argmax(1) != y).astype(int)
    measures = {"entropy": entropy(probs), "one_minus_max_prob": 1 - max_prob(probs),
                "one_minus_margin": 1 - margin(probs)}
    for name, values in (signals or {}).items():
        measures[name] = np.asarray(values, dtype=float)[keep]
    result = {"n": int(len(y)), "error_rate": float(wrong.mean()) if len(y) else float("nan"),
              "error_detection_auroc": {name: roc_auc(wrong, values) for name, values in measures.items()},
              "risk_coverage": risk_coverage(max_prob(probs), wrong == 0)}
    return result


def abstain(probs: np.ndarray, min_confidence: float = 0.5, max_mutual_information: float | None = None,
            mutual_info: Sequence[float] | None = None, evidence_count: Sequence[int] | None = None,
            min_evidence: int = 1) -> tuple[np.ndarray, list[list[str]]]:
    """(abstain mask, reasons per example). Thresholds come from validation, like any other."""
    probs = np.asarray(probs, dtype=float)
    reasons: list[list[str]] = [[] for _ in range(len(probs))]
    confidence = max_prob(probs)
    for i in np.flatnonzero(confidence < min_confidence):
        reasons[i].append("low_confidence")
    if max_mutual_information is not None and mutual_info is not None:
        for i in np.flatnonzero(np.asarray(mutual_info) > max_mutual_information):
            reasons[i].append("high_disagreement")
    if evidence_count is not None:
        for i in np.flatnonzero(np.asarray(evidence_count) < min_evidence):
            reasons[i].append("no_informative_evidence")
    mask = np.array([bool(r) for r in reasons])
    return mask, reasons
