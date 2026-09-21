"""Combined vs individual across ProteinGym's assays — the study's core comparison.

Two arms, scored identically (one Spearman per assay over pooled out-of-fold
predictions, on ProteinGym's own folds):

* **individual** — one model per assay, trained only on that assay's training folds;
* **combined** — one model trained on the training folds of every assay at once.

Both arms use the same features: ProteinGym's precomputed zero-shot scores (95
unsupervised models, none trained on DMS labels). Unlike one-hot features, these
mean the same thing for every protein, so they can be pooled. Features are
standardised within each assay from training rows only; the combined arm's
targets are standardised within each assay too, because DMS scores from
different experiments are on unrelated scales.

**Sibling assays are excluded from the combined arm, and this is not optional.**
217 assays cover only 186 proteins: 24 proteins were assayed more than once,
affecting 55 assays. A variant tested in one assay can then appear in a sibling
assay's training rows with an identical feature vector and a correlated score,
and a combined model could look the answer up instead of learning anything —
inflating pooling on a quarter of the benchmark. For each assay, the combined arm
trains on every other protein's assays plus its own training folds, never its
siblings.
"""

from __future__ import annotations

import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from vpdl.proteingym.bench import pooled_spearman
from vpdl.proteingym.data import FOLD_SCHEMES, parse_assay

logger = logging.getLogger(__name__)

__all__ = [
    "REFERENCE_MODEL",
    "normalize_model_name",
    "protein_map",
    "ScoreTable",
    "load_scores",
    "check_reading",
    "standardize_within_assay",
    "targets_within_assay",
    "combined_training_masks",
    "run_comparison",
]

NON_FEATURE_COLUMNS = frozenset({"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"})

# A single, pre-specified zero-shot model reported beside the trained arms as the
# no-training floor. Chosen before seeing results; picking the per-assay best
# would select on the test data.
REFERENCE_MODEL = "TranceptEVE_L"


def normalize_model_name(name: str) -> str:
    """``'ESM-1v (ensemble)'`` and ``'ESM1v_ensemble'`` -> ``'esm1vensemble'``.

    The published tables and the score files name the same models differently.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def protein_map(dms_ids: Iterable[str], reference_csv: Path | str | None) -> tuple[dict, str]:
    """``{DMS_id: UniProt_ID}``, from ProteinGym's reference file when available.

    Falls back to the first two underscore-separated tokens of the assay ID
    (``BLAT_ECOLX_Deng_2012`` -> ``BLAT_ECOLX``), and says so, because a wrong
    grouping would let sibling assays leak into each other's combined model.
    """
    if reference_csv is not None and Path(reference_csv).exists():
        reference = pd.read_csv(reference_csv)
        mapping = dict(zip(reference["DMS_id"], reference["UniProt_ID"]))
        missing = [d for d in dms_ids if d not in mapping]
        if not missing:
            return mapping, "reference"
        logger.warning("%d assays missing from %s; using ID heuristic for them.",
                       len(missing), reference_csv)
        for dms_id in missing:
            mapping[dms_id] = "_".join(dms_id.split("_")[:2])
        return mapping, "reference+heuristic"
    logger.warning("No ProteinGym reference file — grouping sibling assays by an "
                   "ID heuristic. Download reference_files/DMS_substitutions.csv.")
    return {d: "_".join(d.split("_")[:2]) for d in dms_ids}, "heuristic"


@dataclass
class ScoreTable:
    frame: pd.DataFrame                  # single substitutions: folds + score columns
    score_columns: list[str]
    zero_shot_all_rows: pd.DataFrame     # DMS_id x model: raw Spearman on EVERY row
    unscored_singles: dict[str, int] = field(default_factory=dict)
    score_mismatches: dict[str, int] = field(default_factory=dict)


def load_scores(
    scores_zip: Path | str,
    folds_zip: Path | str,
    only: Iterable[str] | None = None,
) -> ScoreTable:
    """Join each assay's zero-shot scores onto its single substitutions and folds.

    Also computes every model's raw Spearman on ALL rows of the score file — the
    same basis as ProteinGym's zero-shot leaderboard — for :func:`check_reading`.
    """
    wanted = set(only) if only else None
    frames: list[pd.DataFrame] = []
    zero_shot_rows: list[dict] = []
    unscored: dict[str, int] = {}
    mismatches: dict[str, int] = {}
    score_columns: list[str] = []

    with zipfile.ZipFile(scores_zip) as scores, zipfile.ZipFile(folds_zip) as folds:
        fold_files = {Path(n).stem: n for n in folds.namelist() if n.endswith(".csv")}
        for name in sorted(n for n in scores.namelist() if n.endswith(".csv")):
            dms_id = Path(name).stem
            if wanted is not None and dms_id not in wanted:
                continue
            if dms_id not in fold_files:
                logger.warning("%s has scores but no fold file; skipped.", dms_id)
                continue

            with scores.open(name) as handle:
                # Sequences are the bulk of these files and are not needed here.
                full = pd.read_csv(handle, usecols=lambda c: c != "mutated_sequence")
            columns = [c for c in full.columns if c not in NON_FEATURE_COLUMNS]
            for column in columns:
                if column not in score_columns:
                    score_columns.append(column)
            full[columns] = full[columns].apply(pd.to_numeric, errors="coerce")

            zero_shot_rows.append({"DMS_id": dms_id} | {
                column: pooled_spearman(full["DMS_score"], full[column])
                for column in columns
            })

            with folds.open(fold_files[dms_id]) as handle:
                assay = parse_assay(dms_id, pd.read_csv(handle))
            merged = assay.frame.merge(
                full[["mutant", "DMS_score", *columns]]
                .rename(columns={"DMS_score": "_score_in_scores_file"})
                .drop_duplicates("mutant"),
                on="mutant", how="left",
            )
            unscored[dms_id] = int(merged[columns].isna().all(axis=1).sum())
            # Both files carry the label; if the join were wrong they would disagree.
            mismatches[dms_id] = int(
                ((merged["DMS_score"] - merged["_score_in_scores_file"]).abs() > 1e-6).sum())
            merged.insert(0, "DMS_id", dms_id)
            keep = ["DMS_id", "mutant", "position", "DMS_score",
                    *[s for s in FOLD_SCHEMES if s in merged.columns], *columns]
            frames.append(merged[keep])
            if len(frames) % 25 == 0:
                logger.info("loaded %d assays", len(frames))

    if not frames:
        raise ValueError("No assays loaded.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    return ScoreTable(frame=frame, score_columns=score_columns,
                      zero_shot_all_rows=pd.DataFrame(zero_shot_rows),
                      unscored_singles=unscored, score_mismatches=mismatches)


def check_reading(zero_shot_all_rows: pd.DataFrame, published_csv: Path | str) -> pd.DataFrame:
    """Our raw per-assay Spearman for each model beside ProteinGym's published one.

    Near-exact agreement proves the right columns were read with the right
    orientation before any of them is used as a feature.
    """
    published = pd.read_csv(published_csv)
    id_column = "DMS ID" if "DMS ID" in published.columns else "DMS_id"
    published = published.set_index(id_column)
    by_name = {normalize_model_name(c): c for c in published.columns}
    ours = zero_shot_all_rows.set_index("DMS_id")

    rows = []
    for column in ours.columns:
        match = by_name.get(normalize_model_name(column))
        if match is None:
            continue
        pair = pd.concat(
            [ours[column], pd.to_numeric(published[match], errors="coerce")],
            axis=1, join="inner",
        ).dropna()
        if len(pair) < 3:
            continue
        difference = (pair.iloc[:, 0] - pair.iloc[:, 1]).abs()
        rows.append({
            "model": column,
            "published_as": match,
            "assays": len(pair),
            "pearson": round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 4),
            "mean_abs_diff": round(float(difference.mean()), 4),
            "within_0.01": round(float((difference <= 0.01).mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)


def standardize_within_assay(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """z-score each feature within each assay, using TRAINING rows' statistics only."""
    grouped = train.groupby("DMS_id")[columns]
    means, stds = grouped.mean(), grouped.std()

    def apply(frame: pd.DataFrame) -> np.ndarray:
        mu = means.reindex(frame["DMS_id"]).to_numpy(dtype=float)
        sd = stds.reindex(frame["DMS_id"]).to_numpy(dtype=float)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        values = (frame[columns].to_numpy(dtype=float) - mu) / sd
        # Missing after imputation means the model never scored this assay;
        # 0 is that assay's training mean after standardisation.
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return apply(train), apply(test)


def targets_within_assay(train: pd.DataFrame) -> np.ndarray:
    """z-score DMS_score within each assay (training rows only).

    DMS scores from different experiments share no unit; without this the
    combined model would mostly learn which assay a row came from.
    """
    grouped = train.groupby("DMS_id")["DMS_score"]
    mu = grouped.transform("mean")
    sd = grouped.transform("std").replace(0, 1.0).fillna(1.0)
    return ((train["DMS_score"] - mu) / sd).to_numpy(dtype=float)


def combined_training_masks(
    frame: pd.DataFrame,
    scheme: str,
    proteins: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(test_rows, train_rows)`` boolean masks; every row is tested once.

    For fold k, the test rows are fold k of every assay, and no fold-k row of any
    assay is ever trained on. Assays whose protein has no sibling share one model
    trained on all fold != k rows. Each assay WITH siblings gets its own model,
    trained on the same rows minus its siblings'.
    """
    folds = frame[scheme].to_numpy()
    assay_ids = frame["DMS_id"].to_numpy()
    proteins = np.asarray(proteins)
    assays_per_protein = pd.Series(assay_ids).groupby(proteins).nunique()
    has_siblings = pd.Series(proteins).map(assays_per_protein).to_numpy() > 1

    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        shared = test & ~has_siblings
        if shared.any():
            yield shared, train
        for assay in np.unique(assay_ids[test & has_siblings]):
            own_test = test & (assay_ids == assay)
            protein = proteins[assay_ids == assay][0]
            siblings = (proteins == protein) & (assay_ids != assay)
            yield own_test, train & ~siblings


def _fit_predict(model: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    if model == "ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_train, y_train).predict(X_test)

    if model == "gbm":
        from xgboost import XGBRegressor
        # Early-stopped on an inner split. The MMR experiments' GBM grew every
        # tree with no early stopping and was the weakest arm for it.
        inner = np.random.default_rng(seed).random(len(y_train)) < 0.1
        stop = inner.sum() >= 20 and (~inner).sum() >= 20
        regressor = XGBRegressor(
            n_estimators=2000 if stop else 300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=seed, n_jobs=-1,
            early_stopping_rounds=50 if stop else None,
        )
        if stop:
            regressor.fit(X_train[~inner], y_train[~inner],
                          eval_set=[(X_train[inner], y_train[inner])], verbose=False)
        else:
            regressor.fit(X_train, y_train)
        return regressor.predict(X_test)

    raise ValueError(f"Unknown model {model!r}; use ridge or gbm.")


def _individual(frame, features, scheme, model, seed) -> np.ndarray:
    predictions = np.full(len(frame), np.nan)
    for _, positions in frame.groupby("DMS_id", sort=False).indices.items():
        assay = frame.iloc[positions]
        folds = assay[scheme].to_numpy()
        for fold in np.unique(folds):
            test = folds == fold
            train_rows = assay.iloc[np.flatnonzero(~test)]
            test_rows = assay.iloc[np.flatnonzero(test)]
            X_train, X_test = standardize_within_assay(train_rows, test_rows, features)
            predictions[positions[test]] = _fit_predict(
                model, X_train, train_rows["DMS_score"].to_numpy(dtype=float),
                X_test, seed)
    return predictions


def _combined(frame, features, scheme, model, proteins, seed) -> np.ndarray:
    data = frame[["DMS_id", "DMS_score", *features]]
    folds = frame[scheme].to_numpy()
    predictions = np.full(len(frame), np.nan)
    # Standardisation is per assay, so dropping a sibling assay changes no other
    # assay's statistics: one standardised matrix per fold serves every model
    # fitted for that fold.
    prepared: dict = {}
    for index, (test, train) in enumerate(
            combined_training_masks(frame, scheme, proteins), start=1):
        fold = folds[test][0]
        if fold not in prepared:
            prepared.clear()
            fold_train = data.loc[folds != fold]
            _, X = standardize_within_assay(fold_train, data, features)
            y = np.full(len(data), np.nan)
            y[folds != fold] = targets_within_assay(fold_train)
            prepared[fold] = (X, y)
        X, y = prepared[fold]
        predictions[test] = _fit_predict(model, X[train], y[train], X[test], seed)
        if index % 10 == 0:
            logger.info("combined: %d models fitted (%s)", index, model)
    return predictions


def run_comparison(
    scores_zip: Path | str,
    folds_zip: Path | str,
    reference_csv: Path | str | None = None,
    published_zero_shot_csv: Path | str | None = None,
    scheme: str = "fold_random_5",
    model: str = "ridge",
    min_coverage: float = 0.9,
    seed: int = 0,
    out_dir: Path | str = "runs/pg",
    only: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """Individual vs combined, per assay. Returns (per-assay table, summary, reading check)."""
    if scheme not in FOLD_SCHEMES:
        raise ValueError(f"Unknown fold scheme {scheme!r}; use one of {FOLD_SCHEMES}.")
    started = time.time()
    table = load_scores(scores_zip, folds_zip, only)
    frame = table.frame

    bad_joins = {k: v for k, v in table.score_mismatches.items() if v}
    if bad_joins:
        raise ValueError(f"DMS_score differs between the fold and score files for "
                         f"{len(bad_joins)} assays (e.g. {next(iter(bad_joins.items()))}); "
                         "the join is wrong. Refusing to train on mismatched labels.")

    reading = (check_reading(table.zero_shot_all_rows, published_zero_shot_csv)
               if published_zero_shot_csv and Path(published_zero_shot_csv).exists()
               else None)

    # The no-training floor, on these same singles, BEFORE any imputation.
    reference = frame[REFERENCE_MODEL].to_numpy(dtype=float) \
        if REFERENCE_MODEL in frame.columns else np.full(len(frame), np.nan)

    coverage = frame[table.score_columns].notna().mean()
    features = [c for c in table.score_columns if coverage[c] >= min_coverage]
    if not features:
        raise ValueError(f"No score column reaches {min_coverage:.0%} coverage.")
    # Per-assay median imputation uses feature values only, never labels.
    frame[features] = frame[features].fillna(
        frame.groupby("DMS_id")[features].transform("median"))

    mapping, protein_source = protein_map(frame["DMS_id"].unique(), reference_csv)
    proteins = frame["DMS_id"].map(mapping).to_numpy()
    assays_per_protein = pd.Series(mapping).loc[frame["DMS_id"].unique()].value_counts()
    logger.info("%d assays, %d rows, %d features (coverage >= %.0f%%), proteins via %s",
                frame["DMS_id"].nunique(), len(frame), len(features),
                100 * min_coverage, protein_source)

    individual = _individual(frame, features, scheme, model, seed)
    logger.info("individual arm done (%.0fs)", time.time() - started)
    combined = _combined(frame, features, scheme, model, proteins, seed)
    logger.info("combined arm done (%.0fs)", time.time() - started)

    labels = frame["DMS_score"].to_numpy(dtype=float)
    rows = []
    for dms_id, positions in frame.groupby("DMS_id", sort=False).indices.items():
        y = labels[positions]
        ind = pooled_spearman(y, individual[positions])
        com = pooled_spearman(y, combined[positions])
        rows.append({
            "DMS_id": dms_id,
            "protein": mapping[dms_id],
            "has_siblings": bool(assays_per_protein[mapping[dms_id]] > 1),
            "n": len(positions),
            "individual": round(ind, 4),
            "combined": round(com, 4),
            "delta": round(com - ind, 4),
            f"zero_shot_{REFERENCE_MODEL}": round(
                pooled_spearman(y, reference[positions]), 4),
        })
    result = pd.DataFrame(rows)

    valid = result.dropna(subset=["individual", "combined"])
    summary = _summarise(valid, model, scheme, features, protein_source, started)
    summary["singles_without_scores"] = int(sum(table.unscored_singles.values()))
    if reading is not None and len(reading):
        summary["reading_check"] = {
            "models_matched": int(len(reading)),
            "median_pearson_vs_published": float(reading["pearson"].median()),
            "median_mean_abs_diff": float(reading["mean_abs_diff"].median()),
            "worst_model": reading.iloc[-1]["model"],
            "worst_mean_abs_diff": float(reading.iloc[-1]["mean_abs_diff"]),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"combined_{model}_{scheme}"
    result.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    if reading is not None:
        reading.to_csv(out / "zero_shot_reading_check.csv", index=False)
    return result, summary, reading


def _summarise(valid: pd.DataFrame, model, scheme, features, protein_source, started) -> dict:
    from scipy.stats import wilcoxon

    delta = valid["delta"]
    sizes = pd.cut(valid["n"], [0, 1000, 5000, np.inf],
                   labels=["<1k variants", "1k-5k", ">5k"])
    by_size = valid.groupby(sizes, observed=True)["delta"].agg(["mean", "count"])

    try:
        p_value = float(wilcoxon(delta[delta != 0]).pvalue)
    except ValueError:
        p_value = None

    return {
        "model": model,
        "scheme": scheme,
        "assays": int(len(valid)),
        "features": len(features),
        "protein_grouping": protein_source,
        "assays_with_siblings": int(valid["has_siblings"].sum()),
        "mean_individual": round(float(valid["individual"].mean()), 4),
        "mean_combined": round(float(valid["combined"].mean()), 4),
        "mean_delta": round(float(delta.mean()), 4),
        "median_delta": round(float(delta.median()), 4),
        "combined_better_in": f"{int((delta > 0).sum())}/{len(delta)} assays",
        "wilcoxon_p": p_value,
        # Negative = pooling helps small assays more than large ones.
        "spearman_delta_vs_size": (
            round(float(delta.corr(valid["n"], method="spearman")), 4)
            if len(valid) >= 3 and valid["n"].nunique() > 1 else None),
        "mean_delta_by_size": {str(k): {"mean": round(float(v["mean"]), 4),
                                        "assays": int(v["count"])}
                               for k, v in by_size.iterrows()},
        f"mean_zero_shot_{REFERENCE_MODEL}": round(
            float(valid[f"zero_shot_{REFERENCE_MODEL}"].mean()), 4),
        "runtime_s": round(time.time() - started, 1),
    }
