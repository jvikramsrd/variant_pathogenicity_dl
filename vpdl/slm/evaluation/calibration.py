"""Calibration of five-class probabilities — fitted on validation, never on test.

Multi-class:
    temperature   one T for all logits (ranking-preserving; the default)
    vector        per-class scale and bias on the logits (more flexible, more to overfit)
Binary (the pathogenic-vs-benign score): Platt, isotonic, temperature — the DL
branch's implementations (``vpdl.dl.calibration``), reused.

Metrics: top-label ECE, class-wise ECE, multi-class Brier, NLL, reliability
tables — pooled and per group (gene, disease, and the VUS class). Every fit
records ``fitted_on``; :func:`fit_calibrator` refuses rows marked test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from vpdl.dl.calibration import CALIBRATORS as BINARY_CALIBRATORS
from vpdl.dl.calibration import calibration_report as binary_calibration_report

__all__ = ["softmax", "TemperatureScaling", "VectorScaling", "fit_calibrator", "top_label_ece",
           "classwise_ece", "multiclass_brier", "reliability_table", "calibration_summary",
           "grouped_calibration", "BINARY_CALIBRATORS", "binary_calibration_report"]

_EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _nll(logits: np.ndarray, y: np.ndarray) -> float:
    p = softmax(logits)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1)).mean())


@dataclass
class TemperatureScaling:
    temperature: float = 1.0
    fitted_on: str = ""

    def fit(self, logits: np.ndarray, y: Sequence[int]) -> "TemperatureScaling":
        y = np.asarray(y)
        lo, hi = np.log(0.05), np.log(20.0)
        ratio = (np.sqrt(5) - 1) / 2
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        for _ in range(80):
            if _nll(logits / np.exp(c), y) < _nll(logits / np.exp(d), y):
                hi = d
            else:
                lo = c
            c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        self.temperature = float(np.exp((lo + hi) / 2))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.temperature)


@dataclass
class VectorScaling:
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fitted_on: str = ""
    l2: float = 1e-3

    def fit(self, logits: np.ndarray, y: Sequence[int]) -> "VectorScaling":
        from scipy.optimize import minimize
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y)
        k = logits.shape[1]
        onehot = np.eye(k)[y]

        def objective(theta):
            w, b = theta[:k], theta[k:]
            z = logits * w + b
            p = softmax(z)
            loss = -np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1)).mean()
            loss += self.l2 * (((w - 1) ** 2).sum() + (b ** 2).sum())
            grad_z = (p - onehot) / len(y)
            grad_w = (grad_z * logits).sum(0) + 2 * self.l2 * (w - 1)
            grad_b = grad_z.sum(0) + 2 * self.l2 * b
            return loss, np.concatenate([grad_w, grad_b])

        result = minimize(objective, np.concatenate([np.ones(k), np.zeros(k)]), jac=True,
                          method="L-BFGS-B")
        self.scale, self.bias = result.x[:k], result.x[k:]
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) * self.scale + self.bias)


MULTICLASS = {"temperature": TemperatureScaling, "vector": VectorScaling}


def fit_calibrator(method: str, logits: np.ndarray, y: Sequence[int], split: Sequence[str]):
    """Fit on rows whose split is validation. Test rows are refused, not filtered silently."""
    split = np.asarray(split).astype(str)
    if np.isin(split, ["test", "broad_test"]).any():
        raise ValueError("calibration must be fitted on validation predictions only; "
                         "test rows were passed")
    if method not in MULTICLASS:
        raise ValueError(f"unknown calibrator {method!r}; known {sorted(MULTICLASS)}")
    y = np.asarray(y)
    keep = y >= 0
    calibrator = MULTICLASS[method]().fit(np.asarray(logits)[keep], y[keep])
    calibrator.fitted_on = ",".join(sorted(set(split)))
    return calibrator


def reliability_table(probs: np.ndarray, y: Sequence[int], n_bins: int = 10) -> list[dict[str, float]]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    confidence = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        members = index == b
        n = int(members.sum())
        rows.append({"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]), "n": n,
                     "confidence": float(confidence[members].mean()) if n else float("nan"),
                     "accuracy": float(correct[members].mean()) if n else float("nan")})
    return rows


def top_label_ece(probs: np.ndarray, y: Sequence[int], n_bins: int = 10) -> float:
    rows = [r for r in reliability_table(probs, y, n_bins) if r["n"]]
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["accuracy"] - r["confidence"]) for r in rows) / total) if total else float("nan")


def classwise_ece(probs: np.ndarray, y: Sequence[int], n_bins: int = 10) -> dict[str, float]:
    from vpdl.dl.calibration import expected_calibration_error
    from vpdl.slm.schema import CLASSES
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    return {name: expected_calibration_error((y == k).astype(float), probs[:, k], n_bins)
            for k, name in enumerate(CLASSES)}


def multiclass_brier(probs: np.ndarray, y: Sequence[int]) -> float:
    probs = np.asarray(probs, dtype=float)
    return float(((probs - np.eye(probs.shape[1])[np.asarray(y)]) ** 2).sum(1).mean())


def calibration_summary(probs: np.ndarray, y: Sequence[int], n_bins: int = 10) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    keep = y >= 0
    probs, y = probs[keep], y[keep]
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "top_label_ece": top_label_ece(probs, y, n_bins),
            "classwise_ece": classwise_ece(probs, y, n_bins), "brier": multiclass_brier(probs, y),
            "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1)).mean()),
            "reliability": reliability_table(probs, y, n_bins)}


def grouped_calibration(probs: np.ndarray, y: Sequence[int], groups: Sequence[str],
                        min_n: int = 30) -> dict[str, Any]:
    """Per-group summaries (genes, diseases); groups under `min_n` rows are pooled as 'other'."""
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y)
    names, counts = np.unique(groups, return_counts=True)
    small = set(names[counts < min_n])
    folded = np.where(np.isin(groups, list(small)), "other(<%d)" % min_n, groups)
    return {name: {k: v for k, v in calibration_summary(probs[folded == name], y[folded == name]).items()
                   if k != "reliability"} for name in np.unique(folded)}
