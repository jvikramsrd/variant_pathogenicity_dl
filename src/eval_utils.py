"""Evaluation utilities: bootstrap confidence intervals and threshold tuning.

PROJECT_PLAN.md Phase 3 (step 6) makes MCC the primary metric with AUC/AUPR
secondary and requires **bootstrap CIs (10,000 iterations)** plus decision
thresholds tuned on validation data instead of assuming 0.5.

Everything here is metric-only: pure numpy, no torch, fully deterministic
under a fixed seed.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    SUPPORTED_METRICS,
    compute_metric,
    evaluation_report,
    optimal_threshold,
    safe_mcc,
)

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 10_000


def optimal_threshold_by_mcc(y_true: Sequence[int], y_score: Sequence[float],
                             n_grid: int = 2001) -> Tuple[float, float]:
    """Find the score threshold maximising MCC on *validation* data.

    Thin wrapper over :func:`src.metrics.optimal_threshold` kept for the
    existing call sites. Returns ``(threshold, mcc)``; a degenerate input
    yields ``(0.5, 0.0)`` as before.
    """
    thr, value = optimal_threshold(y_true, y_score, metric="mcc", n_grid=n_grid)
    return thr, (0.0 if not np.isfinite(value) else value)


def bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric: str | Callable[[Sequence[int], Sequence[float]], float] = "roc_auc",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
    alpha: float = 0.05,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Percentile-bootstrap CI for one discrimination/threshold metric.

    Parameters
    ----------
    metric:
        Any name in :data:`src.metrics.SUPPORTED_METRICS` -- which now covers
        the whole reporting panel (accuracy, precision, recall, specificity,
        macro/weighted F1, ...) and not just the original five -- or a
        callable ``(y_true_sample, y_score_sample) -> float``.
    threshold:
        Decision cut-off used by the thresholded metrics (mcc/f1/bal-acc).

    Returns ``{"point", "lower", "upper", "n"}``.  Resamples are drawn
    stratified-by-class when both classes have >= 2 members (keeps every
    replicate two-class); otherwise plain i.i.d. resampling is used.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n = len(truth)
    if n == 0:
        return {"point": np.nan, "lower": np.nan, "upper": np.nan, "n": 0}
    if not callable(metric) and metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown bootstrap metric '{metric}'. "
            f"Supported: {', '.join(SUPPORTED_METRICS)}.")

    def _compute(t: np.ndarray, s: np.ndarray) -> float:
        return compute_metric(metric, t, s, threshold=threshold)

    point = _compute(truth, scores)

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(truth == 1)
    neg_idx = np.flatnonzero(truth == 0)
    stratify = len(pos_idx) >= 2 and len(neg_idx) >= 2

    stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        if stratify:
            p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            m = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([p, m])
        else:
            idx = rng.integers(0, n, size=n)
        stats[b] = _compute(truth[idx], scores[idx])
    valid = stats[np.isfinite(stats)]
    if len(valid) == 0:
        lower = upper = np.nan
    else:
        lower = float(np.quantile(valid, alpha / 2))
        upper = float(np.quantile(valid, 1 - alpha / 2))
    n_dropped = n_bootstrap - len(valid)
    if n_dropped:
        logger.debug("bootstrap: %d/%d replicates produced non-finite %s",
                     n_dropped, n_bootstrap, metric)
    return {
        "point": float(point),
        "lower": lower,
        "upper": upper,
        "ci_alpha": alpha,
        "n": int(n),
    }


def primary_metric_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    val_y_true: Optional[Sequence[int]] = None,
    val_y_prob: Optional[Sequence[float]] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
    ci_metrics: Sequence[str] = ("roc_auc", "pr_auc", "mcc"),
) -> Dict[str, float]:
    """Headline report: the full metric panel plus bootstrap CIs.

    The body of the report is :func:`src.metrics.evaluation_report` -- every
    contract metric at both the fixed 0.5 threshold (``t050_*``) and the
    selected threshold (``tval_*``), with calibration and cohort counts.
    Bootstrap CIs are added for *ci_metrics* on top.

    The decision threshold is tuned by MCC on the **validation** vectors when
    provided; otherwise the test vectors double as tuning input (explicitly
    flagged via ``threshold_source`` so callers never confuse the two).
    """
    out: Dict[str, float] = dict(evaluation_report(
        y_true, y_prob,
        val_y_true=val_y_true, val_y_prob=val_y_prob,
        threshold_metric="mcc",
    ))
    if not out.get("available", False):
        return out

    thr = float(out["threshold"])
    for name in ci_metrics:
        ci = bootstrap_ci(y_true, y_prob, metric=name, threshold=thr,
                          n_bootstrap=n_bootstrap, seed=seed)
        out[name] = ci["point"]
        out[f"{name}_ci_low"] = ci["lower"]
        out[f"{name}_ci_high"] = ci["upper"]
    return out


__all__ = [
    "DEFAULT_N_BOOTSTRAP",
    "safe_mcc",
    "optimal_threshold_by_mcc",
    "bootstrap_ci",
    "primary_metric_report",
]
