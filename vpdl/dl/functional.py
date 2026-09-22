"""Functional-assay data — an independent VALIDATION axis, never supervision.

CIMRA (cell-free MMR assay, calibrated OddsPath for all four genes), MaveDB
score sets and the continuous ProteinGym DMS score are held out of training by
construction: nothing here emits a ``label__*`` or ``feature_*`` column. The
functional table is joined onto predictions after training, in
:func:`functional_validation`. (MISSING_EVIDENCE item 14 / PROJECT_PLAN Phase 2:
"Keep this data held out ... used only for validation — this is what makes
later performance claims non-circular".)

A future *controlled* experiment that does train on assay data goes through the
existing label-source machinery (``--train-sources pg_dms``) as its own arm,
never through this module, so it cannot contaminate the primary benchmark
silently.

Orientation is declared, then checked. Functional scores come in both
directions (ProteinGym DMS_score: higher = fitter = tolerated; CIMRA OddsPath:
higher = pathogenic; MaveDB: per score set). The declared direction is checked
against an independent anchor (AlphaMissense) the way label orientation is, and
a contradiction stops the run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.hgvs import normalize_hgvs_p

logger = logging.getLogger(__name__)

__all__ = [
    "FUNCTIONAL_COLUMNS",
    "TAVTIGIAN_ODDSPATH_THRESHOLDS",
    "classify_oddspath_strength",
    "load_cimra",
    "load_mavedb_scores",
    "load_proteingym_continuous",
    "assert_functional_orientation",
    "functional_validation",
]

# damage_score: oriented so that HIGHER = MORE DAMAGING, whatever the assay's
# own convention. raw_score keeps the published value.
FUNCTIONAL_COLUMNS = ("uniprot_id", "position", "wt_aa", "mut_aa", "gene",
                      "assay", "raw_score", "damage_score", "functional_class")

# Tavtigian et al. 2018 OddsPath thresholds (ported from v1 src/cimra.py).
TAVTIGIAN_ODDSPATH_THRESHOLDS: dict[str, float] = {
    "PS3_very_strong": 350.0, "PS3_strong": 18.7, "PS3_moderate": 4.3,
    "PS3_supporting": 2.08, "BS3_supporting": 1.0 / 2.08,
    "BS3_moderate": 1.0 / 4.3, "BS3_strong": 1.0 / 18.7,
    "BS3_very_strong": 1.0 / 350.0,
}


def classify_oddspath_strength(value: float,
                               thresholds: Mapping[str, float] = TAVTIGIAN_ODDSPATH_THRESHOLDS
                               ) -> str:
    """OddsPath -> ACMG PS3/BS3 strength bucket, or ``indeterminate``."""
    if not np.isfinite(value):
        return "indeterminate"
    for name in ("PS3_very_strong", "PS3_strong", "PS3_moderate", "PS3_supporting"):
        if value >= thresholds[name]:
            return name
    for name in ("BS3_very_strong", "BS3_strong", "BS3_moderate", "BS3_supporting"):
        if value <= thresholds[name]:
            return name
    return "indeterminate"


def _functional_class(strength: str) -> float:
    if strength.startswith("PS3"):
        return 1.0
    if strength.startswith("BS3"):
        return 0.0
    return np.nan


def load_cimra(csv_path: Path | str, uniprot_by_gene: Mapping[str, str],
               exclude_splicing: bool = True) -> pd.DataFrame:
    """User-extracted CIMRA OddsPath CSV -> functional table.

    CIMRA has no bulk download: the values live in supplementary tables
    (Drost et al. for MLH1/MSH2/MSH6; Rayner et al. 2022 for PMS2). Required
    columns ``gene, position, wt_aa, mut_aa, cimra_oddspath``; optional
    ``mechanism`` (rows hypothesised to act through splicing are excluded —
    a cell-free assay cannot see splicing). Fail-closed: a missing file or
    column raises instead of yielding an empty table.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CIMRA file not found: {path}")
    frame = pd.read_csv(path)
    required = {"gene", "position", "wt_aa", "mut_aa", "cimra_oddspath"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required CIMRA column(s) {sorted(missing)}")

    frame = frame.copy()
    frame["gene"] = frame["gene"].astype(str).str.upper().str.strip()
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame["cimra_oddspath"] = pd.to_numeric(frame["cimra_oddspath"], errors="coerce")
    bad = frame["position"].isna() | ~(frame["cimra_oddspath"] > 0)
    if bad.any():
        logger.warning("CIMRA: %d rows with unusable position/OddsPath dropped.", int(bad.sum()))
    frame = frame[~bad]
    if exclude_splicing and "mechanism" in frame.columns:
        splice = frame["mechanism"].astype(str).str.lower().str.contains("splic", na=False)
        if splice.any():
            logger.info("CIMRA: %d splicing-mechanism rows excluded.", int(splice.sum()))
        frame = frame[~splice]

    strength = frame["cimra_oddspath"].map(lambda v: classify_oddspath_strength(float(v)))
    out = pd.DataFrame({
        "uniprot_id": frame["gene"].map(uniprot_by_gene),
        "position": frame["position"].astype(int),
        "wt_aa": frame["wt_aa"].astype(str),
        "mut_aa": frame["mut_aa"].astype(str),
        "gene": frame["gene"],
        "assay": "cimra",
        "raw_score": frame["cimra_oddspath"].astype(float),
        # log10 OddsPath: higher = more pathogenic already.
        "damage_score": np.log10(frame["cimra_oddspath"].astype(float)),
        "functional_class": strength.map(_functional_class),
        "cimra_strength": strength,
    })
    return _dedupe(out, "cimra")


def load_mavedb_scores(csv_path: Path | str, gene: str, uniprot_id: str,
                       urn: str, higher_is_damaging: bool,
                       score_column: str = "score") -> pd.DataFrame:
    """A MaveDB score-set CSV (``hgvs_pro``, ``score``) -> functional table.

    `higher_is_damaging` must be stated by the caller from the score set's own
    documentation; it is then checked by :func:`assert_functional_orientation`.
    """
    frame = pd.read_csv(csv_path)
    if "hgvs_pro" not in frame.columns or score_column not in frame.columns:
        raise ValueError(f"{csv_path}: expected hgvs_pro and {score_column} columns")
    parsed = frame["hgvs_pro"].map(normalize_hgvs_p)
    keep = parsed.notna() & pd.to_numeric(frame[score_column], errors="coerce").notna()
    frame, parsed = frame[keep], parsed[keep]
    score = pd.to_numeric(frame[score_column], errors="coerce").astype(float)
    out = pd.DataFrame({
        "uniprot_id": uniprot_id,
        "position": [p[1] for p in parsed],
        "wt_aa": [p[0] for p in parsed],
        "mut_aa": [p[2] for p in parsed],
        "gene": gene,
        "assay": f"mavedb:{urn}",
        "raw_score": score.to_numpy(),
        "damage_score": (score if higher_is_damaging else -score).to_numpy(),
        "functional_class": np.nan,
    })
    return _dedupe(out, f"mavedb:{urn}")


def load_proteingym_continuous(path: Path | str,
                               uniprot_by_gene: Mapping[str, str],
                               genes: Sequence[str] | None = None) -> pd.DataFrame:
    """Continuous ProteinGym DMS_score as validation data.

    Reuses :func:`vpdl.sources.proteingym_dms.load` (so parsing and assay
    selection are shared with the label source), then moves ``DMS_score`` OUT
    of the feature namespace: here it is a validation target, and as a feature
    it would be the label in disguise (landmine L2).
    """
    from vpdl.sources.proteingym_dms import load

    records = load(path, uniprot_by_gene=uniprot_by_gene, genes=genes,
                   include_score_feature=True)
    if "feature_dms_score" not in records.columns:
        raise ValueError(f"{path}: no DMS_score column found")
    out = pd.DataFrame({
        "uniprot_id": records["uniprot_id"], "position": records["position"],
        "wt_aa": records["wt_aa"], "mut_aa": records["mut_aa"],
        "gene": records["gene"], "assay": "pg_dms_continuous",
        "raw_score": records["feature_dms_score"].astype(float),
        # DMS_score: higher = fitter = tolerated, so damage is its negation.
        "damage_score": -records["feature_dms_score"].astype(float),
        # DMS_score_bin == 1 is the TOP fitness half (landmine L1): the
        # already-oriented label (1 = damaging) is reused, not re-derived.
        "functional_class": records["label"],
    })
    return _dedupe(out, "pg_dms_continuous")


def _dedupe(frame: pd.DataFrame, assay: str) -> pd.DataFrame:
    key = ["uniprot_id", "position", "wt_aa", "mut_aa", "assay"]
    duplicated = frame.duplicated(subset=key, keep=False)
    if duplicated.any():
        # Replicate measurements of one variant are averaged, and counted: a
        # silent keep-first would make the value depend on file order.
        logger.info("%s: %d replicate rows averaged.", assay, int(duplicated.sum()))
        numeric = frame.groupby(key, as_index=False).agg(
            {"raw_score": "mean", "damage_score": "mean", "functional_class": "mean",
             "gene": "first"})
        extra = [c for c in frame.columns if c not in numeric.columns]
        firsts = frame.drop_duplicates(subset=key)[key + extra]
        frame = numeric.merge(firsts, on=key, how="left")
        # A variant whose replicates disagree on class gets no class.
        frame.loc[~frame["functional_class"].isin([0.0, 1.0]), "functional_class"] = np.nan
    return frame.reset_index(drop=True)


def assert_functional_orientation(functional: pd.DataFrame, anchor: pd.Series,
                                  min_abs_rho: float = 0.1) -> dict[str, float]:
    """Spearman(damage_score, anchor) per assay must be positive.

    `anchor` is an independent pathogenicity prior (AlphaMissense) aligned to
    `functional`'s rows. A clearly negative correlation means the declared
    direction is wrong — the functional-data equivalent of the ProteinGym
    label inversion — and raises. Near-zero correlation is reported, not
    enforced: it may be a weak assay rather than a sign error.
    """
    from scipy.stats import spearmanr

    result: dict[str, float] = {}
    anchor = pd.to_numeric(anchor, errors="coerce").to_numpy(dtype=float)
    for assay, rows in functional.groupby("assay").groups.items():
        rows = np.asarray(list(rows))
        positions = functional.index.get_indexer(rows)
        a = anchor[positions]
        d = functional.loc[rows, "damage_score"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(d)
        if ok.sum() < 30:
            logger.warning("%s: %d rows overlap the anchor; orientation unverified.",
                           assay, int(ok.sum()))
            result[assay] = float("nan")
            continue
        rho = float(spearmanr(d[ok], a[ok])[0])
        result[assay] = rho
        if rho < -min_abs_rho:
            raise ValueError(
                f"{assay}: damage_score correlates with the independent anchor at "
                f"rho={rho:.3f}. The declared score direction is INVERTED.")
    return result


def functional_validation(predictions: pd.DataFrame, functional: pd.DataFrame,
                          n_bootstrap: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per (gene, assay): Spearman of model score vs damage, and class ROC-AUC.

    `predictions` needs ``variant_key, gene, score`` (the held-out-fold scores a
    trained cell wrote for functional variants; see
    :mod:`vpdl.dl.runner`). Scores averaged over seeds before comparison.
    """
    from scipy.stats import spearmanr

    from vpdl.evaluate import bootstrap_ci, roc_auc
    from vpdl.splits import variant_keys

    functional = functional.assign(variant_key=variant_keys(functional))
    mean_scores = predictions.groupby(["variant_key", "gene"], as_index=False)["score"].mean()
    merged = mean_scores.merge(functional.drop(columns="gene"), on="variant_key", how="inner")

    def rho(y, s):
        return float(spearmanr(y, s)[0]) if len(y) > 2 else float("nan")

    rows = []
    for (gene, assay), group in merged.groupby(["gene", "assay"]):
        y, s = group["damage_score"].to_numpy(float), group["score"].to_numpy(float)
        low, high = bootstrap_ci(y, s, rho, n_bootstrap=n_bootstrap, seed=seed)
        classes = group["functional_class"]
        labelled = classes.notna()
        auc = roc_auc(classes[labelled].to_numpy(), group.loc[labelled, "score"].to_numpy()) \
            if labelled.any() else float("nan")
        rows.append({"gene": gene, "assay": assay, "n": len(group),
                     "spearman": round(rho(y, s), 4), "ci_low": round(low, 4),
                     "ci_high": round(high, 4), "n_classified": int(labelled.sum()),
                     "class_roc_auc": round(auc, 4) if np.isfinite(auc) else auc})
    return pd.DataFrame(rows)
