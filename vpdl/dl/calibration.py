"""Calibration: reliability, ECE, Brier, calibration slope/intercept, recalibrators.

Pure numpy (plus optional matplotlib for the diagram), so it runs wherever the
metrics do. Two rules carried over from the rest of the protocol:

* **Calibrators are fitted on inner-validation predictions only**
  (``valpreds_*.csv``), never on the held-out gene. Fitting on the test fold
  would make every calibration metric below a training-set number.
* **Report calibration per held-out gene.** Scores do not transfer across genes
  on an absolute scale (v1 saw MCC-optimal thresholds from 0.057 to 0.434 over
  three genes); a pooled reliability curve can look fine while every gene is
  miscalibrated in a different direction.

Class-weighted training (``pos_weight`` in the neural arms) deliberately shifts
predicted probabilities away from the base rate, so raw Brier/ECE for those
arms measure that choice as much as the model. Compare calibrated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_bins",
    "calibration_slope_intercept",
    "PlattScaler",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "CALIBRATORS",
    "calibration_report",
    "plot_reliability",
]

_EPS = 1e-7


def _clean(y: Sequence[float], p: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(p) & np.isin(y, (0.0, 1.0))
    return y[keep], p[keep]


def _is_probability(p: np.ndarray) -> bool:
    return p.size > 0 and bool(np.all((p >= 0.0) & (p <= 1.0)))


def brier_score(y: Sequence[float], p: Sequence[float]) -> float:
    """Mean squared error of probabilities. NaN if scores are not in [0, 1]."""
    y, p = _clean(y, p)
    if not _is_probability(p):
        return float("nan")
    return float(np.mean((p - y) ** 2))


def reliability_bins(y: Sequence[float], p: Sequence[float], n_bins: int = 10,
                     strategy: str = "uniform") -> pd.DataFrame:
    """Per-bin count, mean prediction and observed positive fraction.

    ``uniform`` bins are equal-width on [0, 1]; ``quantile`` bins hold equal
    counts, which is kinder to the sparse tails of a small fold.
    """
    y, p = _clean(y, p)
    if strategy == "quantile" and len(p):
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        in_bin = index == b
        n = int(in_bin.sum())
        rows.append({"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]), "n": n,
                     "mean_predicted": float(p[in_bin].mean()) if n else float("nan"),
                     "fraction_positive": float(y[in_bin].mean()) if n else float("nan")})
    return pd.DataFrame(rows)


def expected_calibration_error(y: Sequence[float], p: Sequence[float], n_bins: int = 10,
                               strategy: str = "uniform") -> float:
    """Count-weighted mean |observed - predicted| over bins."""
    y_, p_ = _clean(y, p)
    if not _is_probability(p_):
        return float("nan")
    bins = reliability_bins(y_, p_, n_bins, strategy)
    filled = bins[bins["n"] > 0]
    gaps = (filled["fraction_positive"] - filled["mean_predicted"]).abs()
    return float((gaps * filled["n"]).sum() / filled["n"].sum())


def maximum_calibration_error(y: Sequence[float], p: Sequence[float], n_bins: int = 10,
                              strategy: str = "uniform") -> float:
    y_, p_ = _clean(y, p)
    if not _is_probability(p_):
        return float("nan")
    bins = reliability_bins(y_, p_, n_bins, strategy)
    filled = bins[bins["n"] > 0]
    return float((filled["fraction_positive"] - filled["mean_predicted"]).abs().max())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logistic_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                  fit_slope: bool = True, iterations: int = 100,
                  ridge: float = 1e-8) -> tuple[float, float]:
    """Newton-Raphson for logit P(y) = a + b*x (+ offset). Returns (a, b)."""
    design = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    offset = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        eta = design @ beta + offset
        mu = _sigmoid(eta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * w[:, None]) + ridge * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if fit_slope else (float(beta[0]), 1.0)


def calibration_slope_intercept(y: Sequence[float], p: Sequence[float]) -> dict[str, float]:
    """Logistic recalibration of logit(p): slope 1 and intercept 0 are perfect.

    ``slope`` < 1 means predictions are too extreme (overfit); > 1 too timid.
    ``intercept_in_the_large`` is the intercept with the slope fixed at 1 —
    whether predictions are too high (negative) or too low (positive) overall.
    """
    y, p = _clean(y, p)
    if not _is_probability(p) or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "intercept_in_the_large": float("nan")}
    x = _logit(p)
    intercept, slope = _logistic_fit(x, y)
    in_the_large, _ = _logistic_fit(x, y, offset=x, fit_slope=False)
    return {"slope": slope, "intercept": intercept,
            "intercept_in_the_large": in_the_large}


@dataclass
class PlattScaler:
    """sigmoid(a + b * logit(p)). Fitted on inner validation only."""

    a: float = 0.0
    b: float = 1.0
    on_logit: bool = True

    def fit(self, y: Sequence[float], s: Sequence[float]) -> "PlattScaler":
        y, s = _clean(y, s)
        self.on_logit = _is_probability(s)
        x = _logit(s) if self.on_logit else s
        self.a, self.b = _logistic_fit(x, y)
        return self

    def transform(self, s: Sequence[float]) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        x = _logit(s) if self.on_logit else s
        return _sigmoid(self.a + self.b * x)


@dataclass
class TemperatureScaler:
    """sigmoid(logit(p) / T): one parameter, ranking-preserving, hard to overfit."""

    temperature: float = 1.0

    def fit(self, y: Sequence[float], s: Sequence[float]) -> "TemperatureScaler":
        y, s = _clean(y, s)
        x = _logit(s)
        # Golden-section search on log T for the validation NLL.
        def nll(log_t: float) -> float:
            q = np.clip(_sigmoid(x / np.exp(log_t)), _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

        lo, hi = np.log(0.05), np.log(20.0)
        ratio = (np.sqrt(5) - 1) / 2
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        for _ in range(80):
            if nll(c) < nll(d):
                hi = d
            else:
                lo = c
            c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        self.temperature = float(np.exp((lo + hi) / 2))
        return self

    def transform(self, s: Sequence[float]) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(s, dtype=float)) / self.temperature)


@dataclass
class IsotonicCalibrator:
    """Pool-adjacent-violators, monotone non-decreasing; flat beyond the fitted range."""

    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, y: Sequence[float], s: Sequence[float]) -> "IsotonicCalibrator":
        y, s = _clean(y, s)
        order = np.argsort(s, kind="mergesort")
        xs, ys = s[order], y[order]
        blocks: list[list[float]] = []            # [sum_y, count, x_max]
        for xv, yv in zip(xs, ys):
            blocks.append([yv, 1.0, xv])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                s2, n2, x2 = blocks.pop()
                blocks[-1][0] += s2
                blocks[-1][1] += n2
                blocks[-1][2] = x2
        self.thresholds = np.array([b[2] for b in blocks])
        self.values = np.array([b[0] / b[1] for b in blocks])
        return self

    def transform(self, s: Sequence[float]) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if not len(self.thresholds):
            return np.full_like(s, np.nan)
        index = np.clip(np.searchsorted(self.thresholds, s, side="left"), 0,
                        len(self.values) - 1)
        return self.values[index]


CALIBRATORS = {"platt": PlattScaler, "temperature": TemperatureScaler,
               "isotonic": IsotonicCalibrator}


def calibration_report(y: Sequence[float], p: Sequence[float], n_bins: int = 10
                       ) -> dict[str, float]:
    y_, p_ = _clean(y, p)
    return {"n": int(len(y_)), "prevalence": float(y_.mean()) if len(y_) else float("nan"),
            "brier": brier_score(y_, p_),
            "ece": expected_calibration_error(y_, p_, n_bins),
            "ece_quantile": expected_calibration_error(y_, p_, n_bins, "quantile"),
            "mce": maximum_calibration_error(y_, p_, n_bins),
            **calibration_slope_intercept(y_, p_)}


def plot_reliability(bins_by_label: dict[str, pd.DataFrame], path: Path | str,
                     title: str = "Reliability") -> Path | None:
    """Reliability diagram, one line per label (e.g. per gene). Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for label, bins in bins_by_label.items():
        filled = bins[bins["n"] > 0]
        axis.plot(filled["mean_predicted"], filled["fraction_positive"], marker="o",
                  label=f"{label} (n={int(filled['n'].sum())})")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability",
             ylabel="observed pathogenic fraction", title=title)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = Path(path)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
