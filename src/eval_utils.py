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
from sklearn import metrics as sk_metrics

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 10_000


def safe_mcc(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """MCC that returns 0.0 (not NaN) for degenerate single-class predictions."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if len(np.unique(truth)) < 2 or len(np.unique(pred)) < 2:
        return 0.0
    return float(sk_metrics.matthews_corrcoef(truth, pred))


def optimal_threshold_by_mcc(y_true: Sequence[int], y_score: Sequence[float],
                             n_grid: int = 2001) -> Tuple[float, float]:
    """Find the score threshold maximising MCC on *validation* data.

    Sweeps a dense probability grid between the observed score extremes
    (cheap, monotone-free, no external deps) and returns ``(threshold, mcc)``.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return 0.5, 0.0
    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5, safe_mcc(truth, (scores >= 0.5).astype(int))
    grid = np.linspace(lo, hi, n_grid)
    best_t, best_mcc = 0.5, -np.inf
    for t in grid:
        mcc = safe_mcc(truth, (scores >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t, best_mcc


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
        One of ``{"roc_auc", "pr_auc", "mcc", "balanced_accuracy", "f1"}`` or
        a callable ``(y_true_sample, y_score_sample) -> float``.
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

    def _compute(t: np.ndarray, s: np.ndarray) -> float:
        if callable(metric):
            return float(metric(t, s))
        if metric == "roc_auc":
            if len(np.unique(t)) < 2:
                return np.nan
            return float(sk_metrics.roc_auc_score(t, s))
        if metric == "pr_auc":
            if len(np.unique(t)) < 2:
                return np.nan
            return float(sk_metrics.average_precision_score(t, s))
        if metric == "mcc":
            return safe_mcc(t, (s >= threshold).astype(int))
        if metric == "balanced_accuracy":
            return float(sk_metrics.balanced_accuracy_score(t, (s >= threshold).astype(int)))
        if metric == "f1":
            return float(sk_metrics.f1_score(t, (s >= threshold).astype(int), zero_division=0))
        raise ValueError(f"Unknown bootstrap metric '{metric}'.")

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
) -> Dict[str, float]:
    """Plan-accurate headline report: MCC primary, AUC/AUPR secondary, CIs.

    The decision threshold is tuned by MCC on the **validation** vectors when
    provided; otherwise the test vectors double as tuning input (explicitly
    flagged via ``threshold_source`` so callers never confuse the two).
    """
    if val_y_true is None or val_y_prob is None:
        thr, tuned_mcc = optimal_threshold_by_mcc(y_true, y_prob)
        source = "test(fallback)"
    else:
        thr, tuned_mcc = optimal_threshold_by_mcc(val_y_true, val_y_prob)
        source = "validation"

    out: Dict[str, float] = {
        "threshold": thr,
        "threshold_source": source,
    }
    for name in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(y_true, y_prob, metric=name,
                          n_bootstrap=n_bootstrap, seed=seed)
        out[f"{name}"] = ci["point"]
        out[f"{name}_ci_low"] = ci["lower"]
        out[f"{name}_ci_high"] = ci["upper"]
    mcc_ci = bootstrap_ci(y_true, y_prob, metric="mcc", threshold=thr,
                          n_bootstrap=n_bootstrap, seed=seed)
    out["mcc"] = mcc_ci["point"]
    out["mcc_ci_low"] = mcc_ci["lower"]
    out["mcc_ci_high"] = mcc_ci["upper"]
    return out


__all__ = [
    "DEFAULT_N_BOOTSTRAP",
    "safe_mcc",
    "optimal_threshold_by_mcc",
    "bootstrap_ci",
    "primary_metric_report",
]
