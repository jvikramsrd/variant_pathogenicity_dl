"""Cross-validate each assay on ProteinGym's folds and compare with published numbers.

The comparison across all 217 assays — our per-assay Spearman against the one
ProteinGym publishes — is the "correlate with the metrics already available"
check. A pipeline that reproduces the published number assay by assay has
earned the right to be believed when it reports something new.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from vpdl.proteingym.data import FOLD_SCHEMES, Assay, iter_assays
from vpdl.proteingym.ohe import OneHotRidge

logger = logging.getLogger(__name__)

__all__ = ["PUBLISHED_COLUMN", "pooled_spearman", "cross_validate", "reproduce"]

# Our model name -> the column ProteinGym publishes it under.
PUBLISHED_COLUMN = {"ohe": "One-Hot Encodings"}


def pooled_spearman(y_true, y_pred) -> float:
    """ONE Spearman over every out-of-fold prediction, as ProteinGym computes it.

    Not the mean of per-fold Spearmans. The two differ — a model can rank well
    inside each fold while its folds sit at different offsets — and only the
    pooled version reproduces the published number
    (``proteingym/merge_supervised.py``, line 112).
    """
    from scipy.stats import spearmanr

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    if keep.sum() < 3 or np.ptp(p[keep]) == 0:
        return float("nan")
    return float(spearmanr(y[keep], p[keep])[0])


def cross_validate(assay: Assay, scheme: str, alpha: float = 1.0) -> np.ndarray:
    """Out-of-fold predictions: every row predicted once, by a model that never saw it."""
    if scheme not in FOLD_SCHEMES:
        raise ValueError(f"Unknown fold scheme {scheme!r}; use one of {FOLD_SCHEMES}.")

    frame = assay.frame
    width = int(frame["position"].max())
    folds = frame[scheme].to_numpy()
    predictions = np.full(len(frame), np.nan)

    for fold in np.unique(folds):
        test = folds == fold
        model = OneHotRidge(width, alpha=alpha).fit(frame.loc[~test])
        predictions[test] = model.predict(frame.loc[test])

    return predictions


def reproduce(
    folds_zip: Path | str,
    published_csv: Path | str,
    scheme: str = "fold_random_5",
    model: str = "ohe",
    alpha: float = 1.0,
    out_dir: Path | str = "runs/pg",
    only: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Every assay: our pooled Spearman beside ProteinGym's published one."""
    if model not in PUBLISHED_COLUMN:
        raise ValueError(f"No published column known for model {model!r}.")

    published_table = pd.read_csv(published_csv)
    column = PUBLISHED_COLUMN[model]
    if column not in published_table.columns:
        raise ValueError(f"{published_csv} has no '{column}' column.")
    published = published_table.set_index("DMS_id")[column]

    started = time.time()
    rows: list[dict] = []
    for index, assay in enumerate(iter_assays(folds_zip, only), start=1):
        oof = cross_validate(assay, scheme, alpha=alpha)
        ours = pooled_spearman(assay.frame["DMS_score"], oof)
        reference = published.get(assay.dms_id, np.nan)
        rows.append({
            "DMS_id": assay.dms_id,
            "n": len(assay.frame),
            "skipped": assay.skipped,
            "ours": round(ours, 4) if np.isfinite(ours) else np.nan,
            "published": reference,
            "diff": round(ours - reference, 4)
                    if np.isfinite(ours) and np.isfinite(reference) else np.nan,
        })
        if index % 20 == 0:
            logger.info("%d assays done (%.0fs)", index, time.time() - started)

    table = pd.DataFrame(rows)
    both = table.dropna(subset=["ours", "published"])
    diff = both["diff"].abs()
    summary = {
        "model": model,
        "published_column": column,
        "scheme": scheme,
        "alpha": alpha,
        "assays": int(len(table)),
        "assays_compared": int(len(both)),
        "mean_ours": round(float(both["ours"].mean()), 4),
        "mean_published": round(float(both["published"].mean()), 4),
        "pearson_ours_vs_published": round(float(both["ours"].corr(both["published"])), 4),
        "spearman_ours_vs_published": round(
            float(both["ours"].corr(both["published"], method="spearman")), 4),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "median_diff": round(float(both["diff"].median()), 4),
        "within_0.02": round(float((diff <= 0.02).mean()), 3),
        "within_0.05": round(float((diff <= 0.05).mean()), 3),
        "runtime_s": round(time.time() - started, 1),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"reproduce_{model}_{scheme}"
    table.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    return table, summary
