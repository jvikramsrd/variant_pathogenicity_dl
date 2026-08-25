"""CIMRA functional-assay OddsPath ingestion (PROJECT_PLAN.md Phase 2).

CIMRA ("Cell-free In-vitro Mismatch Repair Assay") has been calibrated and
validated for all four MMR genes:

* MLH1/MSH2/MSH6 — Drost et al., prior CIMRA calibration papers.
* PMS2 — Rayner et al. 2022, Human Mutation
  (https://doi.org/10.1002/humu.24387), "Predictive functional assay-based
  classification of PMS2 variants in Lynch syndrome".

Unlike ClinVar/gnomAD/ProteinGym/AlphaMissense/MaveDB, **CIMRA has no bulk
download API** — the per-variant OddsPath values live in supplementary tables
of paywalled journal articles. This module therefore follows the same
fail-closed, user-supplied-CSV pattern already used for the PMS2 homology
gate in :mod:`src.mmr_dataset` (``apply_pms2_homology_gate``): it will not
fabricate or scrape data, only tidy and validate a CSV you extract from the
paper's supplementary tables yourself.

Expected input CSV schema (one row per variant)
-------------------------------------------------
========================= ========================================================
Column                    Meaning
========================= ========================================================
gene                      HGNC symbol (MLH1/MSH2/MSH6/PMS2)
position                  1-based canonical UniProt residue position
wt_aa                     wild-type one-letter amino acid
mut_aa                    variant one-letter amino acid
cimra_oddspath            OddsPath point estimate from the assay
mechanism (optional)      "missense" / "splicing" / "unknown"; rows flagged
                          "splicing" are excluded (CIMRA is a cell-free assay
                          and cannot detect a splicing mechanism -- Phase 2,
                          plan bullet 1)
========================= ========================================================

Extracting the source numbers
------------------------------
Pull the OddsPath column directly from the relevant paper's supplementary
table (Rayner et al. 2022 Table S2/S3 for PMS2; the MLH1/MSH2/MSH6 CIMRA
calibration papers for the other three genes) and save it in the schema
above. No numeric values are hard-coded here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("gene", "position", "wt_aa", "mut_aa", "cimra_oddspath")

#: Tavtigian et al. 2018 ("Modeling the ACMG/AMP variant classification
#: guidelines as a Bayesian classification framework", Genet Med) point-based
#: OddsPath thresholds at the canonical exponential base (2.083). These are
#: the field-standard thresholds cited by both EVE and CSBJ's own Bayesian
#: calibration (PROJECT_PLAN.md Phase 6 background); override via the
#: ``thresholds=`` argument if your literature notes specify different values
#: for a specific evidence category.
TAVTIGIAN_ODDSPATH_THRESHOLDS: Dict[str, float] = {
    "PS3_very_strong": 350.0,
    "PS3_strong": 18.7,
    "PS3_moderate": 4.3,
    "PS3_supporting": 2.08,
    "BS3_supporting": 1.0 / 2.08,
    "BS3_moderate": 1.0 / 4.3,
    "BS3_strong": 1.0 / 18.7,
    "BS3_very_strong": 1.0 / 350.0,
}


def classify_oddspath_strength(
    oddspath: float, thresholds: Dict[str, float] = TAVTIGIAN_ODDSPATH_THRESHOLDS
) -> str:
    """Map one OddsPath value to an ACMG evidence-strength bucket.

    Returns one of ``{PS3_very_strong, PS3_strong, PS3_moderate,
    PS3_supporting, indeterminate, BS3_supporting, BS3_moderate, BS3_strong,
    BS3_very_strong}``. NaN input maps to ``"indeterminate"``.
    """
    if not np.isfinite(oddspath):
        return "indeterminate"
    if oddspath >= thresholds["PS3_very_strong"]:
        return "PS3_very_strong"
    if oddspath >= thresholds["PS3_strong"]:
        return "PS3_strong"
    if oddspath >= thresholds["PS3_moderate"]:
        return "PS3_moderate"
    if oddspath >= thresholds["PS3_supporting"]:
        return "PS3_supporting"
    if oddspath <= thresholds["BS3_very_strong"]:
        return "BS3_very_strong"
    if oddspath <= thresholds["BS3_strong"]:
        return "BS3_strong"
    if oddspath <= thresholds["BS3_moderate"]:
        return "BS3_moderate"
    if oddspath <= thresholds["BS3_supporting"]:
        return "BS3_supporting"
    return "indeterminate"


def load_cimra_oddspath(
    csv_path: Path,
    exclude_splicing: bool = True,
    thresholds: Dict[str, float] = TAVTIGIAN_ODDSPATH_THRESHOLDS,
) -> pd.DataFrame:
    """Load, validate and tidy a user-supplied CIMRA OddsPath CSV.

    Returns columns ``[gene, position, wt_aa, mut_aa, cimra_oddspath,
    cimra_log10_oddspath, cimra_acmg_strength, mechanism]``.

    Raises :class:`FileNotFoundError` / :class:`ValueError` rather than
    silently producing an empty or partially-wrong frame -- fail-closed,
    matching :func:`src.mmr_dataset.apply_pms2_homology_gate`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CIMRA OddsPath file not found: {csv_path}. See src/cimra.py "
            "module docstring for the expected schema and where to extract "
            "the source numbers from (Rayner et al. 2022 for PMS2; the "
            "MLH1/MSH2/MSH6 CIMRA calibration papers otherwise).")
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s) {sorted(missing)}; "
            f"expected at least {REQUIRED_COLUMNS}.")

    out = df.copy()
    out["gene"] = out["gene"].astype(str).str.upper().str.strip()
    out["position"] = pd.to_numeric(out["position"], errors="coerce").astype("Int64")
    out["cimra_oddspath"] = pd.to_numeric(out["cimra_oddspath"], errors="coerce")
    bad_rows = out["position"].isna() | out["cimra_oddspath"].isna()
    if bad_rows.any():
        logger.warning("CIMRA: dropping %d rows with unparsable position/OddsPath.",
                       int(bad_rows.sum()))
        out = out.loc[~bad_rows].copy()
    non_positive = out["cimra_oddspath"] <= 0
    if non_positive.any():
        logger.warning("CIMRA: dropping %d rows with non-positive OddsPath "
                       "(cannot take log10).", int(non_positive.sum()))
        out = out.loc[~non_positive].copy()

    if "mechanism" not in out.columns:
        out["mechanism"] = "unknown"
    out["mechanism"] = out["mechanism"].fillna("unknown").astype(str).str.lower().str.strip()
    if exclude_splicing:
        is_splice = out["mechanism"].str.contains("splic", na=False)
        if is_splice.any():
            logger.info(
                "CIMRA: excluding %d variant(s) with a hypothesized splicing "
                "mechanism -- CIMRA is a cell-free assay and cannot detect "
                "splicing effects (PROJECT_PLAN.md Phase 2).",
                int(is_splice.sum()))
            out = out.loc[~is_splice].copy()

    out["cimra_log10_oddspath"] = np.log10(out["cimra_oddspath"].to_numpy())
    out["cimra_acmg_strength"] = out["cimra_oddspath"].apply(
        lambda v: classify_oddspath_strength(float(v), thresholds))

    key = ["gene", "position", "wt_aa", "mut_aa"]
    dup = out.duplicated(subset=key, keep=False)
    if dup.any():
        logger.warning("CIMRA: %d duplicate variant key(s); keeping the first "
                       "occurrence per key.", int(out.loc[dup, key].drop_duplicates().shape[0]))
        out = out.drop_duplicates(subset=key, keep="first")

    result = out[key + ["cimra_oddspath", "cimra_log10_oddspath",
                        "cimra_acmg_strength", "mechanism"]].reset_index(drop=True)
    logger.info("CIMRA: loaded %d usable OddsPath variants across %d gene(s).",
                len(result), result["gene"].nunique())
    return result


def attach_cimra_features(master: pd.DataFrame, cimra_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join CIMRA OddsPath features onto a variant table.

    Adds ``cimra_oddspath``, ``cimra_log10_oddspath``, ``cimra_acmg_strength``.
    Unmatched rows keep NaN / ``"indeterminate"`` -- CIMRA coverage is partial
    by construction (only variants someone has functionally tested), which is
    itself informative (an ``is_missing_cimra_oddspath`` indicator is added
    downstream by :func:`src.transfer.prior_matrix`, consistent with every
    other sparse prior column).
    """
    key = ["gene", "position", "wt_aa", "mut_aa"]
    cols = ["cimra_oddspath", "cimra_log10_oddspath", "cimra_acmg_strength"]
    sub = cimra_df[key + cols].drop_duplicates(subset=key)
    merged = master.merge(sub, on=key, how="left", validate="m:1")
    merged["cimra_acmg_strength"] = merged["cimra_acmg_strength"].fillna("indeterminate")
    return merged


__all__ = [
    "REQUIRED_COLUMNS",
    "TAVTIGIAN_ODDSPATH_THRESHOLDS",
    "classify_oddspath_strength",
    "load_cimra_oddspath",
    "attach_cimra_features",
]
