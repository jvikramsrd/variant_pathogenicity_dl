"""Metrics: five-class, pathogenic-vs-benign, multi-label evidence, spans.

Two views of one prediction, both always reported:

* **five-class** over P / LP / VUS / LB / B: accuracy, balanced accuracy, macro and
  weighted F1, multi-class MCC, per-class precision / recall / F1, confusion matrix,
  multi-class Brier and log-loss;
* **pathogenic vs benign** (VUS rows excluded, P/LP = 1, LB/B = 0) with the score
  ``(p_P + p_LP) / (p_P + p_LP + p_LB + p_B)``: the project's standard binary panel
  (``vpdl.evaluate.evaluation_report`` — ROC-AUC, PR-AUC, MCC with bootstrap CIs,
  sensitivity, specificity, precision, F1, Brier, ECE) plus balanced accuracy.
  The threshold must come from validation predictions.

Evidence: multi-label micro / macro precision, recall and F1 (types, ACMG codes),
single-label F1 (polarity), and span F1 where a predicted span matches a gold
span of the same label in the same document when their character IoU >= 0.5.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from vpdl.evaluate import best_threshold_by_mcc, evaluation_report
from vpdl.slm.schema import CLASSES

__all__ = ["pathogenic_score", "five_class_metrics", "binary_metrics", "multilabel_metrics",
           "single_label_f1", "span_f1", "multiclass_mcc", "validation_threshold"]

_EPS = 1e-12


def pathogenic_score(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    path, benign = probs[:, 0] + probs[:, 1], probs[:, 3] + probs[:, 4]
    return path / np.maximum(path + benign, _EPS)


def multiclass_mcc(confusion: np.ndarray) -> float:
    """Gorodkin's R_K (the multi-class MCC)."""
    c = confusion.astype(float)
    t, p, n = c.sum(1), c.sum(0), c.sum()
    correct = np.trace(c)
    denominator = np.sqrt((n ** 2 - (p ** 2).sum()) * (n ** 2 - (t ** 2).sum()))
    return float((correct * n - (t * p).sum()) / denominator) if denominator else float("nan")


def five_class_metrics(y: Sequence[int], probs: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    keep = (y >= 0) & (y < len(CLASSES))
    y, probs = y[keep], probs[keep]
    if len(y) == 0:
        return {"n": 0}
    predicted = probs.argmax(1)
    k = len(CLASSES)
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    per_class = {}
    f1s, recalls, supports = [], [], []
    for index, name in enumerate(CLASSES):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index].sum() - tp
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        support = int(confusion[index].sum())
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
            recalls.append(recall)
            supports.append(support)
    onehot = np.eye(k)[y]
    return {
        "n": int(len(y)),
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(np.nanmean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.nanmean(f1s)) if f1s else float("nan"),
        "weighted_f1": float(np.average(np.nan_to_num(f1s), weights=supports)) if supports else float("nan"),
        "mcc": multiclass_mcc(confusion),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "log_loss": float(-np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1)).mean()),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "classes": list(CLASSES),
    }


def binary_metrics(y_binary: Sequence[float], score: Sequence[float], threshold: float | None,
                   n_bootstrap: int = 2000, seed: int = 0, is_validation: bool = False) -> dict[str, Any]:
    """The standard binary panel. `threshold` None is allowed only on validation data."""
    y = np.asarray(y_binary, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep].astype(int), s[keep]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "note": "one class only: binary metrics undefined"}
    if threshold is None and not is_validation:
        raise ValueError("choose the threshold on validation predictions, not on the data scored")
    report = evaluation_report(y, s, threshold=threshold, n_bootstrap=n_bootstrap, seed=seed,
                               check_orientation=False).as_dict()
    predicted = (s >= report["threshold"]).astype(int)
    tpr = ((predicted == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = ((predicted == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
    report["balanced_accuracy"] = float((tpr + tnr) / 2)
    report["recall"] = report["sensitivity"]
    return report


def validation_threshold(y_binary: Sequence[float], score: Sequence[float]) -> float:
    y = np.asarray(y_binary, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    threshold, _ = best_threshold_by_mcc(y[keep].astype(int), s[keep])
    return float(threshold)


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str],
                       threshold: float = 0.5, mask: np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    predicted = (np.asarray(y_prob, dtype=float) >= threshold).astype(float)
    if mask is not None:
        y_true, predicted = y_true * mask, predicted * mask
    tp = (predicted * y_true).sum(0)
    fp = (predicted * (1 - y_true)).sum(0)
    fn = ((1 - predicted) * y_true).sum(0)
    per = {}
    f1s = []
    for i, label in enumerate(labels):
        denominator = 2 * tp[i] + fp[i] + fn[i]
        f1 = 2 * tp[i] / denominator if denominator else float("nan")
        per[label] = {"f1": f1, "support": int(y_true[:, i].sum())}
        if y_true[:, i].sum():
            f1s.append(f1)
    micro_p = tp.sum() / max(_EPS, tp.sum() + fp.sum())
    micro_r = tp.sum() / max(_EPS, tp.sum() + fn.sum())
    return {"micro_precision": float(micro_p), "micro_recall": float(micro_r),
            "micro_f1": float(2 * micro_p * micro_r / max(_EPS, micro_p + micro_r)),
            "macro_f1": float(np.nanmean(f1s)) if f1s else float("nan"),
            "per_label": per, "threshold": threshold}


def single_label_f1(y: Sequence[int], predicted: Sequence[int], labels: Sequence[str]) -> dict[str, Any]:
    y, predicted = np.asarray(y), np.asarray(predicted)
    keep = y >= 0
    y, predicted = y[keep], predicted[keep]
    per, f1s = {}, []
    for i, label in enumerate(labels):
        tp = int(((predicted == i) & (y == i)).sum())
        fp = int(((predicted == i) & (y != i)).sum())
        fn = int(((predicted != i) & (y == i)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        per[label] = {"f1": f1, "support": int((y == i).sum())}
        if (y == i).any():
            f1s.append(f1)
    return {"accuracy": float((y == predicted).mean()) if len(y) else float("nan"),
            "macro_f1": float(np.nanmean(f1s)) if f1s else float("nan"), "per_label": per}


def span_f1(predicted: Iterable[tuple[str, int, int, str]], gold: Iterable[tuple[str, int, int, str]],
            min_iou: float = 0.5) -> dict[str, float]:
    """Spans are (document_id, start, end, label); greedy one-to-one matching by IoU."""
    gold = list(gold)
    by_key: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for index, (doc, start, end, label) in enumerate(gold):
        by_key.setdefault((doc, label), []).append((start, end, index))
    used: set[int] = set()
    matched = 0
    predicted = list(predicted)
    for doc, start, end, label in predicted:
        best, best_iou = None, 0.0
        for g_start, g_end, index in by_key.get((doc, label), []):
            if index in used:
                continue
            overlap = max(0, min(end, g_end) - max(start, g_start))
            union = max(end, g_end) - min(start, g_start)
            iou = overlap / union if union else 0.0
            if iou > best_iou:
                best, best_iou = index, iou
        if best is not None and best_iou >= min_iou:
            used.add(best)
            matched += 1
    precision = matched / len(predicted) if predicted else float("nan")
    recall = matched / len(gold) if gold else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if predicted and gold and (precision + recall) else float("nan"))
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched,
            "predicted": len(predicted), "gold": len(gold), "min_iou": min_iou}
