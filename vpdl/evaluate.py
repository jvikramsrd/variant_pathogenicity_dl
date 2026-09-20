"""One metric path, used by every cell in the experiment matrix.

If two cells compute their numbers differently they are not comparable, however
identical their provenance blocks look. So every arm — one source or all of
them, GBM or MLP or BiLSTM — comes through this module.

Satisfies regression landmine L6 (score orientation is asserted, not assumed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "roc_auc",
    "symmetric_agreement",
    "assert_orientation",
    "best_threshold_by_mcc",
    "bootstrap_ci",
    "evaluation_report",
    "EvaluationResult",
]


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based ROC-AUC. NaN when a class is absent.

    Implemented on ranks rather than via sklearn so the low-level guards in
    :mod:`vpdl.assemble` and :mod:`vpdl.features` can use it without pulling a
    model dependency into the data layer.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s) & np.isin(y, (0.0, 1.0))
    y, s = y[keep], s[keep]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    # Average ranks within ties, or tied scores bias the statistic.
    sorted_scores = s[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def symmetric_agreement(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Agreement ignoring sign: ``max(auc, 1 - auc)``.

    Used to catch a column that *is* the label, in either orientation. The v1
    leak (`dms_bin_median`) was the label flipped, so a one-sided check would
    have read it as maximally uninformative rather than maximally informative.
    """
    auc = roc_auc(y_true, scores)
    return float("nan") if np.isnan(auc) else max(auc, 1.0 - auc)


def assert_orientation(
    y_true: Sequence[int],
    scores: Sequence[float],
    min_auc: float = 0.5,
    name: str = "scorer",
) -> None:
    """Raise when a scorer ranks below chance — that is a sign error, not a weak model.

    v1's backbone comparison fed raw PLLR (negative for damaging) straight
    against a pathogenic=1 label and reported ROC-AUC 0.03-0.18 for two strong
    650M models, output that reads as "protein language models are useless
    here". Callers with a genuinely weak predictor may lower `min_auc`
    deliberately; the default refuses below-chance silently passing through.
    """
    auc = roc_auc(y_true, scores)
    if np.isnan(auc):
        return
    if auc < min_auc:
        raise ValueError(
            f"{name} has ROC-AUC {auc:.4f} (< {min_auc}): the score orientation "
            "is inverted. Higher must mean more pathogenic. Raw PLLR is negative "
            "for damaging variants and needs negating before use."
        )


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else float((tp * tn - fp * fn) / denominator)


def best_threshold_by_mcc(
    y_true: Sequence[int], scores: Sequence[float]
) -> tuple[float, float]:
    """Return ``(threshold, mcc)`` maximising MCC.

    MUST be selected on inner-validation predictions, never on the held-out
    fold. Scores do not transfer across genes on an absolute scale — v1 saw
    MCC-optimal thresholds span 0.057 to 0.434 across three genes — so a fixed
    0.5 is not defensible and a holdout-tuned threshold is a leak.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(s)
    if len(candidates) > 512:
        candidates = np.quantile(s, np.linspace(0, 1, 512))

    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = _mcc(y, (s >= threshold).astype(int))
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def bootstrap_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    statistic,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI. 10,000 resamples is the project's protocol.

    Standalone scripts in v1 defaulted to 2,000 while the pipeline used 10,000,
    and which one produced a given CI was not recorded. `n_bootstrap` is written
    into every run summary here so the reported interval is checkable.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")

    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        pick = rng.integers(0, n, n)
        estimates[index] = statistic(y[pick], s[pick])

    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    )


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores) & np.isin(y_true, (0, 1))
    y, s = y_true[keep], scores[keep]
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    cumulative_tp = np.cumsum(y)
    precision = cumulative_tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


@dataclass
class EvaluationResult:
    n: int
    n_positive: int
    threshold: float
    roc_auc: float
    roc_auc_ci: tuple[float, float]
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    mcc: float
    mcc_ci: tuple[float, float]
    sensitivity: float
    specificity: float
    precision: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_report(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    check_orientation: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """The full panel for one held-out fold.

    `threshold` must come from inner-validation. Passing None selects it on the
    data being scored, which is a leak and is therefore only acceptable when
    *this* data is the inner-validation set.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)

    if check_orientation:
        assert_orientation(y, s)

    if threshold is None:
        threshold, _ = best_threshold_by_mcc(y, s)

    predicted = (s >= threshold).astype(int)
    tp = float(((predicted == 1) & (y == 1)).sum())
    tn = float(((predicted == 0) & (y == 0)).sum())
    fp = float(((predicted == 1) & (y == 0)).sum())
    fn = float(((predicted == 0) & (y == 1)).sum())

    return EvaluationResult(
        n=len(y),
        n_positive=int((y == 1).sum()),
        threshold=float(threshold),
        roc_auc=roc_auc(y, s),
        roc_auc_ci=bootstrap_ci(y, s, roc_auc, n_bootstrap, seed),
        pr_auc=_average_precision(y, s),
        pr_auc_ci=bootstrap_ci(y, s, _average_precision, n_bootstrap, seed),
        mcc=_mcc(y, predicted),
        mcc_ci=bootstrap_ci(
            y, s, lambda yy, ss: _mcc(yy, (ss >= threshold).astype(int)),
            n_bootstrap, seed,
        ),
        sensitivity=tp / (tp + fn) if (tp + fn) else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        precision=tp / (tp + fp) if (tp + fp) else float("nan"),
        extra=dict(extra or {}) | {"n_bootstrap": n_bootstrap},
    )
