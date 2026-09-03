"""Post-hoc calibrators and reliability plotting.

Metric *definitions* live in :mod:`src.metrics`, which is torch-free and is
the single authority for how a model is scored. This module re-exports the
calibration primitives (ECE / MCE / Brier / bin statistics) for backwards
compatibility and owns everything that needs ``torch`` or ``matplotlib``.

Discrimination metrics
    :func:`discrimination_metrics` and :func:`full_report` return the full
    reporting panel -- ROC-AUC, PR-AUC, accuracy, precision, recall,
    specificity, NPV, F1 (pathogenic / macro / weighted), balanced accuracy,
    MCC and the ``tn/fp/fn/tp`` confusion matrix.

Calibration metrics
    Expected Calibration Error (ECE) with both fixed-width ("uniform") and
    equal-mass ("adaptive") binning, Maximum Calibration Error (MCE), and the
    Brier score ``BS = 1/N * sum_i (p_i - y_i)^2``.

Post-hoc calibrators
    * :class:`TemperatureScaling` -- learns ``T > 0`` by minimizing NLL on the
      validation logits with L-BFGS: ``p_hat = sigmoid(z / T)``.
    * :class:`IsotonicCalibrator` -- monotonic non-parametric recalibration via
      :class:`sklearn.isotonic.IsotonicRegression`.

Visualization
    Reliability diagrams comparing uncalibrated vs calibrated model vs the
    PLLR zero-shot baseline.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CLI runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

# ``src.metrics`` is the authoritative, torch-free definition of every metric.
# The names below are re-exported so existing importers of ``src.calibration``
# keep working unchanged.
from .metrics import (  # noqa: E402
    binary_metrics_at_threshold,
    brier_score,
    calibration_bin_stats,
    calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    threshold_free_metrics,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Discrimination metrics + combined report
# --------------------------------------------------------------------------- #
def discrimination_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Discrimination + thresholded panel at *threshold*.

    Delegates to :mod:`src.metrics`, which owns the definitions. The historic
    keys (``roc_auc``, ``pr_auc``, ``mcc``, ``balanced_accuracy``, ``f1``) are
    preserved; the return value is now a superset that also carries accuracy,
    precision, recall/sensitivity, specificity, NPV, macro/weighted F1 and the
    ``tn/fp/fn/tp`` confusion matrix.
    """
    out: Dict[str, float] = dict(threshold_free_metrics(y_true, y_prob))
    out.update(binary_metrics_at_threshold(y_true, y_prob, threshold=threshold))
    # ``f1`` historically meant F1 of the positive (pathogenic) class.
    out["f1"] = out["f1_pathogenic"]
    return out


def full_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    fixed_bins: int = 15,
    adaptive_bins: int = 10,
) -> Dict[str, float]:
    """Combined discrimination + calibration report for one score vector."""
    report = discrimination_metrics(y_true, y_prob, threshold=threshold)
    report.update(calibration_metrics(
        y_true, y_prob, fixed_bins=fixed_bins, adaptive_bins=adaptive_bins))
    return report


# --------------------------------------------------------------------------- #
# Post-hoc calibrators
# --------------------------------------------------------------------------- #
class TemperatureScaling(nn.Module):
    """Learns a single scalar temperature ``T > 0`` on validation logits.

    The parameter is stored in log-space (``T = exp(log_T)``) so positivity is
    guaranteed throughout optimization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200,
            verbose: bool = False) -> "TemperatureScaling":
        """Minimize NLL on (logits, labels) with full-batch L-BFGS."""
        logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1).detach().cpu()
        labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1).detach().cpu()
        self.log_temperature.data.fill_(0.0)  # reset T to 1 before fitting

        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.forward(logits), labels)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)  # LBFGS returns the final loss tensor
        if verbose:
            logger.info("Temperature scaling: T=%.4f (final NLL=%.6f)",
                        self.temperature, float(final_loss.detach()))
        return self

    def predict_proba(self, logits: torch.Tensor) -> np.ndarray:
        """Calibrated probabilities from raw logits."""
        with torch.no_grad():
            scaled = self.forward(torch.as_tensor(logits, dtype=torch.float32).reshape(-1))
            return torch.sigmoid(scaled).cpu().numpy()

    def state_dict_for_export(self) -> Dict[str, float]:
        return {"temperature": self.temperature}


class IsotonicCalibrator:
    """Monotonic non-parametric calibration wrapping sklearn's IsotonicRegression."""

    def __init__(self) -> None:
        self._iso: Optional[IsotonicRegression] = None

    def fit(self, y_prob: Sequence[float], y_true: Sequence[int]) -> "IsotonicCalibrator":
        probs = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(probs, truth)
        return self

    def transform(self, y_prob: Sequence[float]) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform().")
        probs = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        return np.clip(self._iso.predict(probs), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_reliability_diagrams(
    series: Mapping[str, Sequence[float]],
    y_true: Sequence[int],
    out_path: str,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> str:
    """Grouped-bar reliability diagram comparing several score vectors.

    Parameters
    ----------
    series:
        Mapping from display name to probability vector, e.g.
        ``{"Uncalibrated": p1, "Temperature": p2, "PLLR baseline": p3}``.
    out_path:
        Destination PNG path.

    Returns the written path.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    truth = list(y_true)
    names = list(series.keys())
    palette = sns.color_palette("colorblind", len(names))

    n_bins_used = min(
        calibration_bin_stats(probs, truth, n_bins=n_bins)["n_bins"]
        for probs in series.values()
    )
    bin_width = 1.0 / n_bins_used
    bar_width = bin_width * 0.8 / max(1, len(names))

    for i, (name, probs) in enumerate(series.items()):
        stats = calibration_bin_stats(probs, truth, n_bins=n_bins, strategy="uniform")
        ece = expected_calibration_error(probs, truth, n_bins=n_bins, strategy="uniform")
        # Map bin indices back into probability space [0, 1].
        centers = (np.arange(stats["n_bins"]) + 0.5) * (1.0 / stats["n_bins"])
        offsets = (i - (len(names) - 1) / 2) * bar_width
        ax.bar(centers + offsets, stats["accuracy"], width=bar_width * 0.92,
               color=palette[i], alpha=0.85,
               label=f"{name} (ECE={ece:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.4, label="Perfectly calibrated")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Predicted probability (mean confidence per bin)")
    ax.set_ylabel("Observed pathogenic fraction")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved reliability diagram -> %s", out_path)
    return out_path


__all__ = [
    "calibration_bin_stats",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "calibration_metrics",
    "binary_metrics_at_threshold",
    "threshold_free_metrics",
    "discrimination_metrics",
    "full_report",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "plot_reliability_diagrams",
    "expit",
]
