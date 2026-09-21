"""Paired comparison of arms scored on identical held-out variants.

Every arm is scored on the same ClinVar variants (see
:class:`vpdl.experiment.CellConfig`), so "does arm A beat arm B" is a PAIRED
question: resample variants once, and compute both arms' AUC on that same
resample. Test-set sampling noise shared by both arms then cancels, which is
what comparing two separate confidence intervals cannot do — two overlapping
CIs can still hide a difference that is consistent on every resample.

This is the paper's headline table: the difference between pooled and
single-source training, with a confidence interval on the difference itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from vpdl.evaluate import assert_orientation, roc_auc
from vpdl.splits import variant_keys

logger = logging.getLogger(__name__)

__all__ = [
    "parse_cell",
    "load_predictions",
    "feature_predictions",
    "paired_delta",
    "paired_table",
]

# A held-out fold smaller than this is reported but not read as evidence
# (PMS2, at n = 21 with 4 benign variants). Matches CellResult.summary().
SCOREABLE_MIN_N = 50


def parse_cell(cell: str) -> tuple[str, str, int]:
    """``train-clinvar__cap-pg_dms300__gbm__seed42`` -> (arm, model, seed).

    The arm can itself contain ``__`` (ablations, caps), so this splits from
    the right: the slug is always ``{arm}__{model}__seed{seed}``.
    """
    head, _, seed = cell.rpartition("__seed")
    arm, _, model = head.rpartition("__")
    return arm, model, int(seed)


def load_predictions(runs_dir: Path | str) -> pd.DataFrame:
    """Every held-out prediction under `runs_dir`, tagged with arm/model/seed."""
    frames = []
    for path in sorted(Path(runs_dir).glob("predictions_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        arm, model, seed = parse_cell(str(frame["cell"].iloc[0]))
        frames.append(frame.assign(arm=arm, model=model, seed=seed))
    if not frames:
        raise ValueError(f"No predictions_*.csv under {runs_dir}.")
    return pd.concat(frames, ignore_index=True)


def feature_predictions(
    table: pd.DataFrame,
    feature: str,
    eval_source: str = "clinvar",
) -> pd.DataFrame:
    """A zero-training arm: rank the evaluation variants by one raw feature.

    The floor every trained arm has to clear. If pooling scores below simply
    sorting by AlphaMissense, it is not merely unhelpful — it is destroying
    signal that was already in the inputs.
    """
    column = f"label__{eval_source}"
    if column not in table.columns or feature not in table.columns:
        raise ValueError(f"Table needs '{column}' and '{feature}'.")
    rows = table[table[column].notna()]
    out = pd.DataFrame({
        "gene": rows["gene"].to_numpy(),
        "variant_key": variant_keys(rows),
        "label": rows[column].astype(int).to_numpy(),
        "score": pd.to_numeric(rows[feature], errors="coerce").to_numpy(),
    })
    # Only pathogenicity-oriented features make sense here; an inverted one
    # (allele frequency, say) would report a meaningless below-chance AUC.
    assert_orientation(out["label"], out["score"], name=feature)
    return out.assign(arm=f"feature:{feature}", model="none", seed=0)


def _mean_auc(y: np.ndarray, scores: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        values = [roc_auc(y, scores[:, j]) for j in range(scores.shape[1])]
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def paired_delta(
    y: Sequence[int],
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(delta, ci_low, ci_high)`` for AUC(a) − AUC(b) on the SAME variants.

    `a` and `b` are ``(n_variants, n_seeds)`` score matrices; each arm's AUC
    is its mean over seeds, matching how every other table reports arms.
    Each bootstrap resample draws variants once and scores both arms on it.
    """
    y = np.asarray(y)
    a = np.asarray(a, dtype=float).reshape(len(y), -1)
    b = np.asarray(b, dtype=float).reshape(len(y), -1)

    point = _mean_auc(y, a) - _mean_auc(y, b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        pick = rng.integers(0, n, n)
        deltas[index] = _mean_auc(y[pick], a[pick]) - _mean_auc(y[pick], b[pick])
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return point, float("nan"), float("nan")
    return (point, float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def paired_mean_delta(
    folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean over genes of the per-gene paired ΔAUC, with a stratified CI.

    Each resample draws variants WITHIN each gene, so every gene keeps its own
    size and class balance, and the statistic stays "the average per-gene
    effect" rather than letting the largest gene dominate. `folds` holds one
    ``(y, a, b)`` per gene, in the shapes :func:`paired_delta` takes.
    """
    def one(y, a, b, pick):
        return _mean_auc(y[pick], a[pick]) - _mean_auc(y[pick], b[pick])

    folds = [(np.asarray(y), np.asarray(a, float).reshape(len(y), -1),
              np.asarray(b, float).reshape(len(y), -1)) for y, a, b in folds]
    point = float(np.mean([one(y, a, b, slice(None)) for y, a, b in folds]))

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        with np.errstate(all="ignore"):
            means[index] = np.nanmean([
                one(y, a, b, rng.integers(0, len(y), len(y))) for y, a, b in folds
            ])
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _score_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index="variant_key", columns="seed",
                             values="score", aggfunc="first")


def paired_table(
    predictions: pd.DataFrame,
    reference: tuple[str, str],
    n_bootstrap: int = 10_000,
    extra_arms: Iterable[pd.DataFrame] = (),
) -> pd.DataFrame:
    """Every (arm, model) against one reference arm, gene by gene.

    Refuses rather than proceeds when two arms scored different variants for
    a gene: that means they came from different tables or evaluation sources,
    and a "paired" difference between different test sets is meaningless.
    """
    predictions = pd.concat([predictions, *extra_arms], ignore_index=True)
    ref_arm, ref_model = reference
    ref = predictions[(predictions["arm"] == ref_arm)
                      & (predictions["model"] == ref_model)]
    if ref.empty:
        known = sorted({f"{a}:{m}" for a, m in
                        zip(predictions["arm"], predictions["model"])})
        raise ValueError(f"No predictions for reference {ref_arm}:{ref_model}. "
                         f"Available: {known}")

    rows: list[dict] = []
    for (arm, model), group in predictions.groupby(["arm", "model"]):
        if (arm, model) == (ref_arm, ref_model):
            continue
        informative: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for gene, arm_rows in group.groupby("gene"):
            ref_rows = ref[ref["gene"] == gene]
            if ref_rows.empty:
                continue

            a = _score_matrix(arm_rows)
            b = _score_matrix(ref_rows)
            if set(a.index) != set(b.index):
                raise ValueError(
                    f"{arm}:{model} and {ref_arm}:{ref_model} scored different "
                    f"{gene} variants ({len(a)} vs {len(b)}, "
                    f"{len(set(a.index) ^ set(b.index))} differ). Not a paired "
                    "comparison — were they built from different tables?"
                )
            b = b.loc[a.index]
            labels = (arm_rows.drop_duplicates("variant_key")
                      .set_index("variant_key").loc[a.index, "label"])
            ref_labels = (ref_rows.drop_duplicates("variant_key")
                          .set_index("variant_key").loc[a.index, "label"])
            if not labels.equals(ref_labels):
                raise ValueError(f"{gene}: arms disagree on ground-truth labels; "
                                 "they were scored against different sources.")

            y = labels.to_numpy(dtype=int)
            delta, low, high = paired_delta(y, a.to_numpy(), b.to_numpy(),
                                            n_bootstrap=n_bootstrap)
            scoreable = len(y) >= SCOREABLE_MIN_N
            rows.append({
                "gene": gene,
                "arm": arm,
                "model": model,
                "n": len(y),
                "n_benign": int((y == 0).sum()),
                "scoreable": scoreable,
                "seeds": a.shape[1],
                "auc": round(_mean_auc(y, a.to_numpy()), 4),
                "auc_reference": round(_mean_auc(y, b.to_numpy()), 4),
                "delta": round(delta, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "ci_excludes_zero": bool(np.isfinite(low) and (low > 0 or high < 0)),
            })
            # A gene where the arm and the reference agree on EVERY resample is
            # a design constant, not a measurement — e.g. MSH2 for any DMS arm,
            # since holding MSH2 out removes every DMS label. Averaging it in
            # would pull the headline toward zero by construction.
            structurally_identical = delta == 0 and low == 0 and high == 0
            if scoreable and not structurally_identical:
                informative.append((gene, y, a.to_numpy(), b.to_numpy()))

        if len(informative) >= 2:
            delta, low, high = paired_mean_delta(
                [(y, a, b) for _, y, a, b in informative], n_bootstrap=n_bootstrap)
            rows.append({
                "gene": "mean:" + "+".join(g for g, *_ in informative),
                "arm": arm,
                "model": model,
                "n": sum(len(y) for _, y, *_ in informative),
                "n_benign": sum(int((y == 0).sum()) for _, y, *_ in informative),
                "scoreable": True,
                "seeds": max(a.shape[1] for *_, a, _ in informative),
                "auc": round(float(np.mean([_mean_auc(y, a) for _, y, a, _ in informative])), 4),
                "auc_reference": round(float(np.mean([_mean_auc(y, b) for _, y, _, b in informative])), 4),
                "delta": round(delta, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "ci_excludes_zero": bool(np.isfinite(low) and (low > 0 or high < 0)),
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["_summary"] = frame["gene"].str.startswith("mean:")
    return (frame
            .sort_values(["_summary", "scoreable", "gene", "arm"],
                         ascending=[False, False, True, True])
            .drop(columns="_summary")
            .reset_index(drop=True))
