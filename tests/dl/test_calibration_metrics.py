"""Metrics beyond discrimination: F1, Brier, ECE, calibration slope/intercept, calibrators."""

from __future__ import annotations

import numpy as np
import pytest


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_perfectly_calibrated_scores_have_slope_one_and_small_ece():
    from vpdl.dl.calibration import calibration_slope_intercept, expected_calibration_error

    y, p = _calibrated()
    fit = calibration_slope_intercept(y, p)
    assert fit["slope"] == pytest.approx(1.0, abs=0.12)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.12)
    assert expected_calibration_error(y, p) < 0.04


def test_overconfident_scores_have_slope_below_one():
    from vpdl.dl.calibration import calibration_slope_intercept

    y, p = _calibrated()
    logit = np.log(p / (1 - p))
    overconfident = 1 / (1 + np.exp(-3 * logit))
    assert calibration_slope_intercept(y, overconfident)["slope"] < 0.5


def test_brier_and_ece_refuse_non_probabilities():
    from vpdl.dl.calibration import brier_score, expected_calibration_error

    assert np.isnan(brier_score([0, 1], [-3.0, 2.0]))
    assert np.isnan(expected_calibration_error([0, 1], [-3.0, 2.0]))
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0


@pytest.mark.parametrize("name", ["platt", "temperature", "isotonic"])
def test_calibrators_reduce_ece_of_overconfident_scores(name):
    from vpdl.dl.calibration import CALIBRATORS, expected_calibration_error

    y, p = _calibrated(seed=1)
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-3 * logit))
    calibrator = CALIBRATORS[name]().fit(y[:2000], over[:2000])
    fixed = calibrator.transform(over[2000:])
    assert expected_calibration_error(y[2000:], fixed) < expected_calibration_error(
        y[2000:], over[2000:])
    if name != "isotonic":                     # strictly monotone maps keep the ranking
        assert np.all(np.diff(fixed[np.argsort(over[2000:])]) >= -1e-12)


def test_temperature_scaler_recovers_the_temperature():
    from vpdl.dl.calibration import TemperatureScaler

    y, p = _calibrated(n=20000, seed=2)
    logit = np.log(p / (1 - p))
    sharpened = 1 / (1 + np.exp(-2.0 * logit))
    assert TemperatureScaler().fit(y, sharpened).temperature == pytest.approx(2.0, rel=0.1)


def test_evaluation_report_carries_f1_brier_ece():
    from vpdl.evaluate import evaluation_report

    y, p = _calibrated(n=300, seed=3)
    report = evaluation_report(y, p, threshold=0.5, n_bootstrap=20)
    tp = ((p >= 0.5) & (y == 1)).sum()
    precision, recall = tp / (p >= 0.5).sum(), tp / (y == 1).sum()
    assert report.f1 == pytest.approx(2 * precision * recall / (precision + recall))
    assert 0 < report.brier < 0.25 and 0 <= report.ece < 0.2
