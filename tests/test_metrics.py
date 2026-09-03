"""Tests for the authoritative metric engine (``src.metrics``).

These lock down the parts of the reporting contract that are easy to get
silently wrong: specificity's denominator, the difference between F1 of the
pathogenic class and the macro/weighted averages, confusion-matrix counts on
degenerate inputs, and -- most importantly -- that a threshold is never tuned
on the held-out test vectors when validation vectors were supplied.

Everything here is pure numpy; no network, no torch, no fixtures on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    MIN_COHORT_N,
    SUPPORTED_METRICS,
    af_band_labels,
    binary_metrics_at_threshold,
    cohort_availability,
    cohort_report,
    compute_metric,
    confusion_counts,
    evaluation_report,
    journal_table,
    missingness_group_labels,
    optimal_threshold,
    star_tier_labels,
    threshold_free_metrics,
)


# --------------------------------------------------------------------------- #
# Confusion matrix and thresholded metrics
# --------------------------------------------------------------------------- #
def test_confusion_counts_match_hand_computation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    #        tn tn fp fn tp tp
    assert confusion_counts(y_true, y_pred) == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


def test_confusion_counts_survive_single_class_input():
    """``sklearn`` collapses to a 1x1 matrix here; the helper must not."""
    counts = confusion_counts([0, 0, 0], [0, 0, 0])
    assert counts == {"tn": 3, "fp": 0, "fn": 0, "tp": 0}


def test_specificity_uses_the_negative_class_denominator():
    # 3 negatives (2 correct, 1 false positive) -> specificity 2/3.
    y_true = [0, 0, 0, 1, 1, 1]
    probs = [0.1, 0.2, 0.9, 0.8, 0.9, 0.2]
    m = binary_metrics_at_threshold(y_true, probs, threshold=0.5)
    assert m["specificity"] == pytest.approx(2 / 3)
    # Recall/sensitivity are the same quantity reported under both names.
    assert m["recall"] == pytest.approx(m["sensitivity"])
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_specificity_is_nan_when_there_are_no_negatives():
    m = binary_metrics_at_threshold([1, 1, 1], [0.9, 0.8, 0.7], threshold=0.5)
    assert np.isnan(m["specificity"])


def test_degenerate_cohort_emits_no_sklearn_warning():
    """A single-class cohort must not trip sklearn's confusion-matrix warning.

    The scorers (`balanced_accuracy_score` among them) rebuild a confusion
    matrix without passing `labels`, so they warn and fall back to a 1x1
    matrix. The panel is derived from counts instead, precisely to avoid that.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning becomes a failure
        m = binary_metrics_at_threshold([1, 1, 1], [0.9, 0.8, 0.7], threshold=0.5)
        binary_metrics_at_threshold([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert m["tp"] == 3 and m["tn"] == 0
    assert np.isnan(m["specificity"])
    assert np.isnan(m["balanced_accuracy"])      # undefined without negatives


def test_f1_variants_are_distinct_and_named():
    """A class-imbalanced case where the three F1s genuinely differ."""
    y_true = [0] * 8 + [1] * 2
    probs = [0.1] * 7 + [0.9] + [0.9, 0.1]
    m = binary_metrics_at_threshold(y_true, probs, threshold=0.5)
    assert {"f1_pathogenic", "f1_macro", "f1_weighted"} <= set(m)
    assert m["f1_pathogenic"] != pytest.approx(m["f1_macro"])
    assert m["f1_macro"] != pytest.approx(m["f1_weighted"])


def test_threshold_free_metrics_nan_on_single_class():
    out = threshold_free_metrics([1, 1, 1], [0.2, 0.5, 0.9])
    assert np.isnan(out["roc_auc"]) and np.isnan(out["pr_auc"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_covers_the_reporting_panel():
    required = {
        "roc_auc", "pr_auc", "accuracy", "precision", "recall", "sensitivity",
        "specificity", "f1_pathogenic", "f1_macro", "f1_weighted",
        "balanced_accuracy", "mcc",
    }
    assert required <= set(SUPPORTED_METRICS)


def test_legacy_f1_alias_matches_pathogenic_f1():
    y_true, probs = [0, 1, 0, 1], [0.1, 0.9, 0.8, 0.6]
    assert compute_metric("f1", y_true, probs) == pytest.approx(
        compute_metric("f1_pathogenic", y_true, probs))


def test_compute_metric_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metric("not_a_metric", [0, 1], [0.1, 0.9])


# --------------------------------------------------------------------------- #
# Availability guarding
# --------------------------------------------------------------------------- #
def test_availability_rejects_empty_small_and_single_class():
    assert cohort_availability([]) == (False, "empty_cohort")
    assert cohort_availability([0, 1] * 3)[0] is False           # n < MIN_COHORT_N
    assert cohort_availability([1] * (MIN_COHORT_N + 5)) == (False, "single_class")


def test_availability_rejects_tiny_minority_class():
    y = [0] * 40 + [1] * 2
    available, reason = cohort_availability(y)
    assert available is False
    assert "minority_class" in reason


def test_availability_accepts_a_healthy_cohort():
    assert cohort_availability([0] * 20 + [1] * 20) == (True, "")


# --------------------------------------------------------------------------- #
# evaluation_report -- threshold discipline
# --------------------------------------------------------------------------- #
def _separable_cohort(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    probs = np.clip(rng.normal(loc=y * 0.5 + 0.25, scale=0.15), 0.0, 1.0)
    return y, probs


def test_report_marks_validation_as_the_threshold_source():
    y, p = _separable_cohort()
    vy, vp = _separable_cohort(seed=1)
    report = evaluation_report(y, p, val_y_true=vy, val_y_prob=vp)
    assert report["threshold_source"] == "validation"
    assert report["available"] is True


def test_report_flags_the_test_fallback_threshold():
    y, p = _separable_cohort()
    report = evaluation_report(y, p)
    assert report["threshold_source"] == "test(fallback)"


def test_threshold_is_taken_from_validation_not_test():
    """The selected threshold must track the validation vectors alone."""
    y, p = _separable_cohort(seed=2)
    vy, vp = _separable_cohort(seed=3)
    # Shifting validation scores upward must move the chosen threshold up,
    # even though the test vectors are unchanged.
    low = evaluation_report(y, p, val_y_true=vy, val_y_prob=vp)["threshold"]
    high = evaluation_report(
        y, p, val_y_true=vy, val_y_prob=np.clip(vp + 0.25, 0, 1))["threshold"]
    assert high > low


def test_report_emits_both_threshold_panels_and_calibration():
    y, p = _separable_cohort()
    report = evaluation_report(y, p)
    for key in ("t050_accuracy", "t050_specificity", "t050_tn",
                "tval_accuracy", "tval_specificity", "tval_tp",
                "roc_auc", "pr_auc", "brier", "ece_uniform", "ece_adaptive"):
        assert key in report, f"missing {key}"


def test_report_on_degenerate_cohort_is_flagged_not_scored():
    report = evaluation_report([1] * 30, [0.9] * 30)
    assert report["available"] is False
    assert report["unavailable_reason"] == "single_class"
    assert report["threshold_source"] == "unavailable"
    assert np.isnan(report["roc_auc"])
    assert np.isnan(report["t050_accuracy"])


def test_optimal_threshold_on_degenerate_input_is_neutral():
    thr, value = optimal_threshold([1, 1, 1], [0.2, 0.5, 0.9])
    assert thr == 0.5 and np.isnan(value)


# --------------------------------------------------------------------------- #
# Cohort labelling helpers
# --------------------------------------------------------------------------- #
def test_af_bands_separate_absent_from_ultra_rare():
    labels = af_band_labels([np.nan, 0.0, 1e-6, 5e-4, 0.05])
    assert labels[0] == "absent"          # missing
    assert labels[1] == "absent"          # never observed
    assert labels[2] != "absent"          # observed, ultra-rare
    assert labels[4] == ">=0.01"


def test_af_band_edges_are_half_open_and_lose_nothing():
    """A frequency sitting exactly on a band edge must not fall through."""
    labels = af_band_labels([1e-5, 1e-4, 1e-3, 1e-2])
    assert "absent" not in set(labels)
    assert labels[0] == "[1e-05,0.0001)"
    assert labels[3] == ">=0.01"


def test_star_tiers_bucket_high_review_status():
    labels = star_tier_labels([0, 1, 2, 3, 4, np.nan])
    assert list(labels[:3]) == ["0", "1", "2"]
    assert labels[3] == "3+" and labels[4] == "3+"
    assert labels[5] == "unknown"


def test_missingness_groups_split_complete_from_sparse():
    import pandas as pd
    frame = pd.DataFrame({
        "a": [1.0, 1.0, np.nan],
        "b": [1.0, np.nan, np.nan],
    })
    labels = missingness_group_labels(frame, ["a", "b"])
    assert list(labels) == ["complete", "partial", "sparse"]


def test_missingness_groups_handle_absent_columns():
    import pandas as pd
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert list(missingness_group_labels(frame, ["nope"])) == ["unknown", "unknown"]


# --------------------------------------------------------------------------- #
# cohort_report / journal_table
# --------------------------------------------------------------------------- #
def test_cohort_report_retains_unscoreable_cohorts_as_rows():
    y = np.array([0] * 25 + [1] * 25 + [0, 1])
    p = np.concatenate([np.full(25, 0.2), np.full(25, 0.8), [0.3, 0.7]])
    genes = ["BRCA1"] * 50 + ["RARE"] * 2
    out = cohort_report(y, p, {"gene": genes}, threshold=0.5)
    assert set(out["level"]) == {"BRCA1", "RARE"}
    rare = out[out["level"] == "RARE"].iloc[0]
    assert bool(rare["available"]) is False
    assert np.isnan(rare["roc_auc"])
    big = out[out["level"] == "BRCA1"].iloc[0]
    assert bool(big["available"]) is True


def test_cohort_report_rejects_mismatched_group_length():
    with pytest.raises(ValueError, match="has 2 levels"):
        cohort_report([0, 1, 0], [0.1, 0.9, 0.2], {"gene": ["A", "B"]}, threshold=0.5)


def test_cohort_report_applies_one_shared_threshold():
    y = np.array([0] * 20 + [1] * 20)
    p = np.concatenate([np.full(20, 0.3), np.full(20, 0.7)])
    out = cohort_report(y, p, {"g": ["x"] * 40}, threshold=0.6)
    assert float(out.iloc[0]["threshold"]) == pytest.approx(0.6)


def test_journal_table_renders_unavailable_as_na():
    good = evaluation_report(*_separable_cohort())
    bad = evaluation_report([1] * 30, [0.9] * 30)
    table = journal_table({"model_a": good, "model_b": bad})
    assert list(table["experiment"]) == ["model_a", "model_b"]
    row_b = table[table["experiment"] == "model_b"].iloc[0]
    assert row_b["roc_auc"] == "n/a"


# --------------------------------------------------------------------------- #
# Backwards compatibility of the legacy entry points
# --------------------------------------------------------------------------- #
def test_full_report_keeps_its_historic_keys():
    """``src.train`` and several scripts read these names directly."""
    from src.calibration import full_report

    y, p = _separable_cohort()
    report = full_report(y, p)
    for key in ("roc_auc", "pr_auc", "mcc", "balanced_accuracy", "f1",
                "ece_uniform", "ece_adaptive", "mce", "brier"):
        assert key in report, f"regression: full_report lost '{key}'"


def test_bootstrap_ci_accepts_the_widened_metric_set():
    from src.eval_utils import bootstrap_ci

    y, p = _separable_cohort()
    ci = bootstrap_ci(y, p, metric="specificity", n_bootstrap=64, seed=0)
    assert set(ci) >= {"point", "lower", "upper", "n"}
    assert ci["lower"] <= ci["point"] <= ci["upper"]


def test_bootstrap_ci_still_rejects_unknown_metrics():
    from src.eval_utils import bootstrap_ci

    with pytest.raises(ValueError, match="Unknown bootstrap metric"):
        bootstrap_ci([0, 1], [0.1, 0.9], metric="nope", n_bootstrap=4)
