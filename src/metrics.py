"""Authoritative metric engine for every experiment in this project.

This module is the single definition of "how a model is scored". It is pure
``numpy``/``scikit-learn``/``pandas`` -- deliberately no ``torch``, no
``matplotlib`` -- so that ablation drivers, benchmark scripts and tests can
import it without paying for a deep-learning stack.

Why it exists
-------------
Metrics previously lived in three places that had drifted apart:

* ``calibration.discrimination_metrics`` -- ROC-AUC, PR-AUC, MCC, balanced
  accuracy, F1.
* ``calibration.full_report`` -- the above plus ECE/MCE/Brier.
* ``eval_utils.primary_metric_report`` -- ROC-AUC, PR-AUC, MCC with bootstrap
  CIs and a validation-tuned threshold.

None of them reported accuracy, precision, recall, specificity, macro/weighted
F1 or a confusion matrix, so no run could produce the evaluation panel the
project's reporting contract requires. ``eval_utils.bootstrap_ci`` also
hard-coded its metric dispatch, so the gap could not be closed by callers.

``calibration`` and ``eval_utils`` now re-export from here; this module owns
the definitions.

Reporting contract
------------------
Every primary experiment and ablation reports discrimination (AUROC, AUPRC),
threshold metrics (accuracy, precision, recall/sensitivity, specificity,
F1-pathogenic, macro/weighted F1, balanced accuracy, MCC), the confusion
matrix, and calibration (Brier, ECE, MCE) -- at **both** the fixed 0.5
threshold and a threshold selected on validation data only.

Degenerate cohorts
------------------
Undersized or single-class cohorts do not get a plausible-looking number. They
are flagged ``available=False`` with an ``unavailable_reason`` and NaN metrics,
so a journal table can print "n/a" rather than a misleading 1.000.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

logger = logging.getLogger(__name__)

#: Cohorts smaller than this are reported as unavailable rather than scored.
MIN_COHORT_N: int = 20
#: Cohorts with fewer than this many members of either class are unavailable
#: for threshold-dependent and ranking metrics.
MIN_COHORT_PER_CLASS: int = 5

#: Metrics that consume ranked scores and need no decision threshold.
THRESHOLD_FREE_METRICS: Tuple[str, ...] = ("roc_auc", "pr_auc")
#: Metrics computed from hard predictions at a decision threshold.
THRESHOLDED_METRICS: Tuple[str, ...] = (
    "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity",
    "specificity", "npv", "f1_pathogenic", "f1_macro", "f1_weighted", "mcc",
)
#: Calibration metrics computed from probabilities against labels.
CALIBRATION_METRICS: Tuple[str, ...] = ("brier", "ece_uniform", "ece_adaptive", "mce")


# --------------------------------------------------------------------------- #
# Availability guarding
# --------------------------------------------------------------------------- #
def cohort_availability(
    y_true: Sequence[int],
    min_n: int = MIN_COHORT_N,
    min_per_class: int = MIN_COHORT_PER_CLASS,
) -> Tuple[bool, str]:
    """Return ``(available, reason)`` for a cohort's label vector.

    A cohort is scoreable only when it is large enough *and* carries both
    classes in sufficient number. Anything else yields NaN metrics with an
    explicit reason instead of a number that looks like a result.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    n = int(truth.size)
    if n == 0:
        return False, "empty_cohort"
    if n < min_n:
        return False, f"n<{min_n}"
    n_pos = int((truth == 1).sum())
    n_neg = int((truth == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return False, "single_class"
    if min(n_pos, n_neg) < min_per_class:
        return False, f"minority_class<{min_per_class}"
    return True, ""


def _nan_panel(keys: Sequence[str]) -> Dict[str, float]:
    return {k: float("nan") for k in keys}


# --------------------------------------------------------------------------- #
# Calibration primitives (moved here from calibration.py; it re-exports them)
# --------------------------------------------------------------------------- #
def _bin_assignments(probs: np.ndarray, n_bins: int, strategy: str) -> Tuple[np.ndarray, int]:
    """Assign each probability to a bin; return ``(bin_index, n_effective_bins)``."""
    probs = np.asarray(probs, dtype=np.float64)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n_bins - 1)
        return idx, n_bins
    if strategy == "adaptive":
        # Equal-mass bins via quantiles; collapse duplicate edges gracefully.
        quantiles = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(quantiles)
        if len(edges) < 2:
            return np.zeros_like(probs, dtype=int), 1
        idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, len(edges) - 2)
        return idx, len(edges) - 1
    raise ValueError(f"Unknown binning strategy '{strategy}'.")


def calibration_bin_stats(
    y_prob: Sequence[float],
    y_true: Sequence[int],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Per-bin confidence/accuracy statistics used by ECE/MCE and plotting."""
    probs = np.asarray(y_prob, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.int64)
    idx, n_eff = _bin_assignments(probs, n_bins, strategy)

    confidence = np.zeros(n_eff)
    accuracy = np.zeros(n_eff)
    counts = np.zeros(n_eff, dtype=int)
    for b in range(n_eff):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b]:
            confidence[b] = probs[mask].mean()
            accuracy[b] = truth[mask].mean()
    gaps = np.abs(confidence - accuracy)
    return {"confidence": confidence, "accuracy": accuracy,
            "counts": counts, "gaps": gaps, "n_bins": n_eff}


def expected_calibration_error(
    y_prob: Sequence[float], y_true: Sequence[int], n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """``ECE = sum_b (n_b / N) |conf(b) - acc(b)|``."""
    stats = calibration_bin_stats(y_prob, y_true, n_bins=n_bins, strategy=strategy)
    total = stats["counts"].sum()
    if total == 0:
        return float("nan")
    return float((stats["counts"] / total * stats["gaps"]).sum())


def maximum_calibration_error(
    y_prob: Sequence[float], y_true: Sequence[int], n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Largest absolute confidence-accuracy gap across bins."""
    stats = calibration_bin_stats(y_prob, y_true, n_bins=n_bins, strategy=strategy)
    nonempty = stats["counts"] > 0
    if not nonempty.any():
        return float("nan")
    return float(stats["gaps"][nonempty].max())


def brier_score(y_prob: Sequence[float], y_true: Sequence[int]) -> float:
    """Mean squared error of probabilities against binary labels."""
    probs = np.asarray(y_prob, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((probs - truth) ** 2))


def calibration_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    fixed_bins: int = 15,
    adaptive_bins: int = 10,
) -> Dict[str, float]:
    """Brier + ECE (uniform and adaptive binning) + MCE."""
    return {
        "brier": brier_score(y_prob, y_true),
        "ece_uniform": expected_calibration_error(
            y_prob, y_true, n_bins=fixed_bins, strategy="uniform"),
        "ece_adaptive": expected_calibration_error(
            y_prob, y_true, n_bins=adaptive_bins, strategy="adaptive"),
        "mce": maximum_calibration_error(
            y_prob, y_true, n_bins=fixed_bins, strategy="uniform"),
    }


# --------------------------------------------------------------------------- #
# Confusion matrix and thresholded metrics
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """Confusion matrix as explicit ``tn/fp/fn/tp`` counts.

    ``sklearn.confusion_matrix`` collapses to a 1x1 array when only one class
    is present in both truth and prediction, so the counts are accumulated
    directly against ``labels=[0, 1]``.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    matrix = sk_metrics.confusion_matrix(truth, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def safe_mcc(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """MCC that returns 0.0 (not NaN) for degenerate single-class predictions."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if len(np.unique(truth)) < 2 or len(np.unique(pred)) < 2:
        return 0.0
    return float(sk_metrics.matthews_corrcoef(truth, pred))


def binary_metrics_at_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """The full thresholded panel plus the confusion matrix.

    ``recall`` and ``sensitivity`` are the same quantity under both names --
    the reporting contract asks for "recall / sensitivity" and downstream
    tables use whichever term the surrounding text uses.

    ``f1_pathogenic`` is F1 for the positive (pathogenic) class specifically;
    ``f1_macro`` and ``f1_weighted`` are the multi-class averages, named so
    that a table can never silently mix them up.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    pred = (probs >= threshold).astype(int)

    counts = confusion_counts(truth, pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]

    # Specificity and NPV have no sklearn one-liner for the binary case and
    # are computed from the counts so that empty denominators stay NaN rather
    # than silently becoming 0.0.
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    recall = float(sk_metrics.recall_score(truth, pred, zero_division=0))

    out: Dict[str, float] = {
        "accuracy": float(sk_metrics.accuracy_score(truth, pred)),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(truth, pred)),
        "precision": float(sk_metrics.precision_score(truth, pred, zero_division=0)),
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1_pathogenic": float(sk_metrics.f1_score(truth, pred, zero_division=0)),
        "f1_macro": float(sk_metrics.f1_score(truth, pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            sk_metrics.f1_score(truth, pred, average="weighted", zero_division=0)),
        "mcc": safe_mcc(truth, pred),
    }
    out.update({k: float(v) for k, v in counts.items()})
    return out


def threshold_free_metrics(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> Dict[str, float]:
    """AUROC and AUPRC; NaN when only one class is present."""
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    if truth.size == 0 or len(np.unique(truth)) < 2:
        return _nan_panel(THRESHOLD_FREE_METRICS)
    return {
        "roc_auc": float(sk_metrics.roc_auc_score(truth, probs)),
        "pr_auc": float(sk_metrics.average_precision_score(truth, probs)),
    }


# --------------------------------------------------------------------------- #
# Metric registry -- one dispatch table for bootstrap and cohort reporting
# --------------------------------------------------------------------------- #
MetricFn = Callable[[np.ndarray, np.ndarray, float], float]


def _registry_entry(name: str) -> MetricFn:
    if name in THRESHOLD_FREE_METRICS:
        def fn(t: np.ndarray, s: np.ndarray, _threshold: float) -> float:
            return threshold_free_metrics(t, s)[name]
        return fn

    def fn(t: np.ndarray, s: np.ndarray, threshold: float) -> float:
        return binary_metrics_at_threshold(t, s, threshold=threshold)[name]
    return fn


#: Every metric name that :func:`compute_metric` and ``bootstrap_ci`` accept.
METRIC_REGISTRY: Dict[str, MetricFn] = {
    name: _registry_entry(name)
    for name in THRESHOLD_FREE_METRICS + THRESHOLDED_METRICS
}
# Legacy aliases kept so existing call sites and saved configs keep resolving.
METRIC_REGISTRY["f1"] = METRIC_REGISTRY["f1_pathogenic"]

#: Names accepted by :func:`compute_metric`, sorted for stable help text.
SUPPORTED_METRICS: Tuple[str, ...] = tuple(sorted(METRIC_REGISTRY))


def compute_metric(
    metric: Union[str, Callable[[Sequence[int], Sequence[float]], float]],
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """Evaluate one registered metric (or a caller-supplied callable)."""
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if callable(metric):
        return float(metric(truth, scores))
    try:
        fn = METRIC_REGISTRY[metric]
    except KeyError:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {', '.join(SUPPORTED_METRICS)}."
        ) from None
    return float(fn(truth, scores, threshold))


# --------------------------------------------------------------------------- #
# Threshold selection (validation data only)
# --------------------------------------------------------------------------- #
def optimal_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric: str = "mcc",
    n_grid: int = 2001,
) -> Tuple[float, float]:
    """Threshold maximising *metric* on the supplied (validation) vectors.

    Sweeps a dense grid between the observed score extremes and returns
    ``(threshold, best_value)``. Callers must pass validation vectors: tuning
    on the held-out test set is exactly the leak this project forbids.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if truth.size == 0 or len(np.unique(truth)) < 2:
        return 0.5, float("nan")
    lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5, compute_metric(metric, truth, scores, threshold=0.5)
    best_t, best_v = 0.5, -np.inf
    for t in np.linspace(lo, hi, n_grid):
        value = compute_metric(metric, truth, scores, threshold=float(t))
        if np.isfinite(value) and value > best_v:
            best_v, best_t = value, float(t)
    if not np.isfinite(best_v):
        return 0.5, float("nan")
    return best_t, float(best_v)


# --------------------------------------------------------------------------- #
# The contract report
# --------------------------------------------------------------------------- #
def evaluation_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    val_y_true: Optional[Sequence[int]] = None,
    val_y_prob: Optional[Sequence[float]] = None,
    threshold_metric: str = "mcc",
    fixed_bins: int = 15,
    adaptive_bins: int = 10,
    min_n: int = MIN_COHORT_N,
    min_per_class: int = MIN_COHORT_PER_CLASS,
) -> Dict[str, Union[float, str, bool, int]]:
    """The full reporting panel for one score vector.

    Returns a **flat** dict so a set of runs concatenates straight into a
    machine-readable results table:

    * ``n``, ``n_positive``, ``n_negative``, ``prevalence``
    * ``available`` / ``unavailable_reason``
    * ``roc_auc``, ``pr_auc``
    * ``brier``, ``ece_uniform``, ``ece_adaptive``, ``mce``
    * ``t050_*``  -- every thresholded metric + confusion matrix at 0.5
    * ``tval_*``  -- the same panel at the selected threshold
    * ``threshold``, ``threshold_source``, ``threshold_metric``

    The selected threshold comes from ``val_y_*`` when supplied. When it is
    not, the test vectors are reused and ``threshold_source`` is set to
    ``"test(fallback)"`` -- a value that must never appear in a headline
    result, but which keeps diagnostic runs usable.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)

    n_pos = int((truth == 1).sum())
    n_neg = int((truth == 0).sum())
    available, reason = cohort_availability(
        truth, min_n=min_n, min_per_class=min_per_class)

    report: Dict[str, Union[float, str, bool, int]] = {
        "n": int(truth.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(n_pos / truth.size) if truth.size else float("nan"),
        "available": available,
        "unavailable_reason": reason,
        "threshold_metric": threshold_metric,
    }

    if not available:
        report.update(_nan_panel(THRESHOLD_FREE_METRICS))
        report.update(_nan_panel(CALIBRATION_METRICS))
        for prefix in ("t050_", "tval_"):
            report.update({f"{prefix}{k}": float("nan")
                           for k in THRESHOLDED_METRICS + ("tn", "fp", "fn", "tp")})
        report["threshold"] = float("nan")
        report["threshold_source"] = "unavailable"
        return report

    if val_y_true is None or val_y_prob is None:
        thr, _ = optimal_threshold(truth, probs, metric=threshold_metric)
        source = "test(fallback)"
    else:
        thr, _ = optimal_threshold(val_y_true, val_y_prob, metric=threshold_metric)
        source = "validation"

    report.update(threshold_free_metrics(truth, probs))
    report.update(calibration_metrics(
        truth, probs, fixed_bins=fixed_bins, adaptive_bins=adaptive_bins))
    for prefix, t in (("t050_", 0.5), ("tval_", thr)):
        for key, value in binary_metrics_at_threshold(truth, probs, threshold=t).items():
            report[f"{prefix}{key}"] = value
    report["threshold"] = float(thr)
    report["threshold_source"] = source
    return report


# --------------------------------------------------------------------------- #
# Cohort / subgroup reporting
# --------------------------------------------------------------------------- #
#: Allele-frequency band edges (gnomAD joint AF). ``absent`` covers AF == 0 or
#: missing; the top band is everything at or above the last edge.
AF_BAND_EDGES: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)


def af_band_labels(af: Sequence[float], edges: Sequence[float] = AF_BAND_EDGES) -> np.ndarray:
    """Bucket allele frequencies into ordered, printable band labels.

    Missing and zero frequencies land in ``absent`` rather than the lowest
    numeric band: "never observed" and "observed once in gnomAD" are different
    evidence and must not be pooled.
    """
    values = pd.to_numeric(pd.Series(af), errors="coerce").to_numpy(dtype=float)
    out = np.empty(values.shape, dtype=object)
    out[:] = "absent"
    edges = tuple(edges)
    for i, hi in enumerate(edges):
        lo = edges[i - 1] if i else 0.0
        # Bands are half-open ``[lo, hi)`` so a frequency sitting exactly on an
        # edge belongs to the band above it. The first band opens at ``> 0``
        # instead: zero means "never observed", which is ``absent``, not rare.
        above = (values > lo) if i == 0 else (values >= lo)
        label = f"[{lo:g},{hi:g})" if i else f"(0,{hi:g})"
        out[above & (values < hi)] = label
    out[values >= edges[-1]] = f">={edges[-1]:g}"
    out[~np.isfinite(values)] = "absent"
    return out


def missingness_group_labels(
    frame: pd.DataFrame, feature_cols: Sequence[str]
) -> np.ndarray:
    """Group rows by how much of *feature_cols* is missing.

    Returns ``complete`` (nothing missing), ``partial`` (up to half missing)
    or ``sparse`` (more than half missing), so evaluation can show whether a
    model's advantage is confined to rows with full feature coverage -- a
    common source of apparent gains.
    """
    present = [c for c in feature_cols if c in frame.columns]
    if not present:
        return np.full(len(frame), "unknown", dtype=object)
    frac = frame[present].isna().mean(axis=1).to_numpy(dtype=float)
    out = np.empty(frac.shape, dtype=object)
    out[:] = "partial"
    out[frac == 0.0] = "complete"
    out[frac > 0.5] = "sparse"
    return out


def star_tier_labels(stars: Sequence[float]) -> np.ndarray:
    """ClinVar review-star tiers as printable labels (``0`` .. ``3+``)."""
    values = pd.to_numeric(pd.Series(stars), errors="coerce").to_numpy(dtype=float)
    out = np.empty(values.shape, dtype=object)
    out[:] = "unknown"
    for s in (0, 1, 2):
        out[values == s] = str(s)
    out[values >= 3] = "3+"
    return out


def cohort_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    groups: Mapping[str, Sequence],
    threshold: float,
    threshold_source: str = "validation",
    min_n: int = MIN_COHORT_N,
    min_per_class: int = MIN_COHORT_PER_CLASS,
) -> pd.DataFrame:
    """Per-subgroup metrics in long format, one row per (grouping, level).

    ``groups`` maps a grouping name (``"gene"``, ``"label_source"``,
    ``"star_tier"``, ``"af_band"``, ``"missingness"``) to a per-row level
    vector. *threshold* must have been selected on validation data; it is
    applied unchanged to every cohort so subgroups stay comparable.

    Undersized or single-class cohorts appear as rows with ``available=False``
    and NaN metrics -- they are never silently dropped, because a gene that
    cannot be scored is itself a coverage finding.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    rows = []
    for grouping, levels in groups.items():
        level_arr = pd.Series(list(levels)).astype(object).to_numpy()
        if level_arr.shape[0] != truth.shape[0]:
            raise ValueError(
                f"Grouping '{grouping}' has {level_arr.shape[0]} levels for "
                f"{truth.shape[0]} rows.")
        for level in pd.unique(level_arr):
            mask = level_arr == level
            t, p = truth[mask], probs[mask]
            n_pos = int((t == 1).sum())
            n_neg = int((t == 0).sum())
            available, reason = cohort_availability(
                t, min_n=min_n, min_per_class=min_per_class)
            row: Dict[str, Union[float, str, bool, int]] = {
                "grouping": grouping,
                "level": str(level),
                "n": int(t.size),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "available": available,
                "unavailable_reason": reason,
                "threshold": float(threshold),
                "threshold_source": threshold_source,
            }
            if available:
                row.update(threshold_free_metrics(t, p))
                row.update(calibration_metrics(t, p))
                row.update(binary_metrics_at_threshold(t, p, threshold=threshold))
            else:
                row.update(_nan_panel(THRESHOLD_FREE_METRICS))
                row.update(_nan_panel(CALIBRATION_METRICS))
                row.update(_nan_panel(THRESHOLDED_METRICS + ("tn", "fp", "fn", "tp")))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Journal table rendering
# --------------------------------------------------------------------------- #
#: Column order for the journal-ready comparison table.
JOURNAL_COLUMNS: Tuple[str, ...] = (
    "roc_auc", "pr_auc", "tval_accuracy", "tval_f1_pathogenic", "tval_f1_macro",
    "tval_balanced_accuracy", "tval_precision", "tval_recall", "tval_specificity",
    "tval_mcc", "brier", "ece_uniform",
)


def journal_table(
    reports: Mapping[str, Mapping[str, Union[float, str, bool, int]]],
    columns: Sequence[str] = JOURNAL_COLUMNS,
    n_decimals: int = 3,
) -> pd.DataFrame:
    """Assemble per-experiment reports into a journal-ready table.

    *reports* maps an experiment label to an :func:`evaluation_report` result.
    Unavailable cohorts render as ``n/a`` rather than a number. No ranking or
    "winner" is computed here -- accuracy alone must never pick a model, so
    that judgement stays with the reader and the surrounding text.
    """
    rows = []
    for label, report in reports.items():
        row: Dict[str, Union[str, float]] = {"experiment": label}
        row["n"] = report.get("n", float("nan"))
        for col in columns:
            value = report.get(col, float("nan"))
            if not report.get("available", True) or not isinstance(value, (int, float)):
                row[col] = "n/a"
            elif isinstance(value, float) and not np.isfinite(value):
                row[col] = "n/a"
            else:
                row[col] = round(float(value), n_decimals)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "MIN_COHORT_N",
    "MIN_COHORT_PER_CLASS",
    "THRESHOLD_FREE_METRICS",
    "THRESHOLDED_METRICS",
    "CALIBRATION_METRICS",
    "METRIC_REGISTRY",
    "SUPPORTED_METRICS",
    "AF_BAND_EDGES",
    "JOURNAL_COLUMNS",
    "cohort_availability",
    "calibration_bin_stats",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "calibration_metrics",
    "confusion_counts",
    "safe_mcc",
    "binary_metrics_at_threshold",
    "threshold_free_metrics",
    "compute_metric",
    "optimal_threshold",
    "evaluation_report",
    "af_band_labels",
    "missingness_group_labels",
    "star_tier_labels",
    "cohort_report",
    "journal_table",
]
