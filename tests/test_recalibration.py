"""Tests for post-hoc calibration of grid artifacts.

The properties worth pinning are the ones a plausible edit would break: that the
calibrator never reads held-out labels, that the threshold is re-selected on the
calibrated scale rather than carried over from the uncalibrated one, and that a
fold too small to fit is reported rather than silently scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.recalibrate_grid import (METHODS, calibrate_fold, logit,  # noqa: E402
                                      panel_for_fold, recalibrate_dir)


def _fold(n=200, seed=0, overconfident=3.0):
    """A fold whose scores are deliberately overconfident, so T > 1 is correct."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 2, n)
    # A separable-but-noisy latent score, then inflated: exactly what an
    # over-fitted head produces and what temperature scaling exists to undo.
    latent = rng.normal(loc=np.where(true == 1, 1.0, -1.0), scale=1.0)
    prob = 1.0 / (1.0 + np.exp(-latent * overconfident))
    return pd.DataFrame({"label": true, "prob": prob})


def test_logit_round_trips_within_float32_noise():
    p = np.array([1e-5, 0.01, 0.5, 0.99, 0.9999939])
    back = 1.0 / (1.0 + np.exp(-logit(p)))
    assert np.allclose(back, p, atol=1e-7)


def test_logit_clips_saturated_probabilities():
    """A stored 0.0 or 1.0 must not become an infinite logit."""
    assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))


def test_temperature_is_fitted_without_reading_holdout_labels():
    """The leak the protocol forbids, asserted rather than assumed."""
    val, ho = _fold(seed=1), _fold(seed=2)
    _, _, a = calibrate_fold(val.label, val.prob, ho.label, ho.prob, "temperature")
    # Same validation fold, holdout labels inverted: the fit must not move.
    _, _, b = calibrate_fold(val.label, val.prob, 1 - ho.label, ho.prob, "temperature")
    assert a["temperature"] == pytest.approx(b["temperature"])


def test_temperature_scaling_reduces_ece_on_overconfident_scores():
    """Severely inflated scores, and enough rows that ECE is signal not binning.

    The latent posterior here is ``sigmoid(2 * latent)``; the fold reports
    ``sigmoid(6 * latent)``, so the correct temperature is about 3. At n=200 and
    a milder inflation the ECE difference is smaller than bin noise and this
    asserts nothing.
    """
    from src.metrics import expected_calibration_error

    val = _fold(n=2000, seed=3, overconfident=6.0)
    ho = _fold(n=2000, seed=4, overconfident=6.0)
    _, cal_ho, params = calibrate_fold(
        val.label, val.prob, ho.label, ho.prob, "temperature")
    assert params["temperature"] > 1.5          # inflated scores need cooling
    before = expected_calibration_error(ho.prob, ho.label)
    after = expected_calibration_error(cal_ho, ho.label)
    assert after < before / 2


def test_threshold_is_reselected_on_the_calibrated_scale():
    """A threshold from the uncalibrated scale would fall outside the new range."""
    val, ho = _fold(seed=5), _fold(seed=6)
    row = panel_for_fold("cell", "MLH1", val, ho, "temperature")
    cal_val, _, _ = calibrate_fold(
        val.label, val.prob, ho.label, ho.prob, "temperature")
    assert row["threshold_source"] != "test(fallback)"
    assert cal_val.min() <= row["threshold"] <= cal_val.max()


def test_uncalibrated_method_passes_probabilities_through():
    val, ho = _fold(seed=7), _fold(seed=8)
    cal_val, cal_ho, params = calibrate_fold(
        val.label, val.prob, ho.label, ho.prob, "uncalibrated")
    assert params == {}
    assert np.allclose(cal_ho, ho.prob)
    assert np.allclose(cal_val, val.prob)


def test_single_class_validation_is_reported_not_scored():
    """A fold that cannot support a calibrator must say so, not fall back."""
    val = pd.DataFrame({"label": [1] * 20, "prob": np.linspace(0.6, 0.9, 20)})
    ho = _fold(seed=9)
    row = panel_for_fold("cell", "PMS2", val, ho, "temperature")
    assert row["available"] is False
    assert "positive" in row["unavailable_reason"]
    assert "roc_auc" not in row


def test_unknown_method_raises():
    val, ho = _fold(seed=10), _fold(seed=11)
    with pytest.raises(ValueError, match="unknown method"):
        calibrate_fold(val.label, val.prob, ho.label, ho.prob, "platt")


def _write_pair(d: Path, tag: str, genes=("MLH1", "MSH2")):
    val_rows, ho_rows = [], []
    for i, g in enumerate(genes):
        v, h = _fold(seed=20 + i), _fold(seed=40 + i)
        v["holdout_gene"], h["holdout_gene"] = g, g
        h["cell_slug"] = tag
        val_rows.append(v)
        ho_rows.append(h)
    pd.concat(val_rows).to_csv(d / f"esm_finetune_valpreds_{tag}.csv", index=False)
    pd.concat(ho_rows).to_csv(d / f"esm_finetune_predictions_{tag}.csv", index=False)


def test_recalibrate_dir_covers_every_cell_gene_method(tmp_path):
    _write_pair(tmp_path, "siamese_lopo_cellA")
    _write_pair(tmp_path, "siamese_lopo_cellB")
    panel = recalibrate_dir(tmp_path)
    assert len(panel) == 2 * 2 * len(METHODS)
    assert set(panel.method) == set(METHODS)
    assert set(panel.holdout_gene) == {"MLH1", "MSH2"}


def test_recalibrate_dir_refuses_a_directory_with_no_valpreds(tmp_path):
    """Pre-e919bef runs cannot be calibrated; say that instead of writing nothing."""
    with pytest.raises(FileNotFoundError, match="e919bef"):
        recalibrate_dir(tmp_path)


def test_temperature_scaling_cannot_change_ranking():
    """It is a monotone transform, so AUROC must be identical, not merely close.

    This is the cheapest guard against applying the calibrator to one vector and
    not the other, or fitting it per-row: either bug moves AUROC.
    """
    val, ho = _fold(seed=12), _fold(seed=13)
    plain = panel_for_fold("cell", "MSH6", val, ho, "uncalibrated")
    scaled = panel_for_fold("cell", "MSH6", val, ho, "temperature")
    assert scaled["roc_auc"] == pytest.approx(plain["roc_auc"])
    assert scaled["pr_auc"] == pytest.approx(plain["pr_auc"])
    assert scaled["threshold"] != plain["threshold"]      # but the cut moves
