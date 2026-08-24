"""Phase 1 — gene-specific dataset construction for MLH1/MSH2/MSH6/PMS2.

Implements the PROJECT_PLAN.md Phase-1 contract on top of the untouched,
shared ClinVar ingestion code (:mod:`src.data_loader`):

1. **UniProt sequences** — the four MMR genes are pinned to their reviewed
   canonical accessions with expected sequence lengths, so a UniProt release
   or mis-resolution can never silently change the coordinate system:

   =======  ========  ======  =============================
   Gene     Accession  Length  Note
   =======  ========  ======  =============================
   MLH1     P40692     756 aa  MutL alpha, N-terminal piece
   MSH2     P43246     934 aa  MutS alpha
   MSH6     P52701    1360 aa  MutS alpha-2 (needs windows)
   PMS2     P54278     862 aa  PMS2CL pseudogene homology!
   =======  ========  ======  =============================

2. **ClinVar** — pulled through :func:`src.data_loader.build_multi_gene_dataset`
   (one streaming pass, star ratings retained).  This module never re-implements
   ClinVar parsing.

3. **InSiGHT/ClinGen VCEP tiering** — records reviewed by an expert panel
   (3+ stars: "reviewed by expert panel" / "practice guideline") are flagged
   ``expert_panel`` and form the highest-confidence evidence tier used for
   training/eval weighting.

4. **PMS2 exon 11–15 pseudogene gate** — standard short-read NGS calls inside
   the PMS2CL homology region are untrustworthy.  This is a *hard filter*
   applied to supervision (labels are withheld / rows optionally dropped),
   fail-closed: you must either supply a validated substitution list with an
   ``orthogonally_confirmed`` flag, an explicit protein-coordinate range for
   the homology region, or explicitly exclude PMS2.  Nothing is guessed.

5. **Circularity-safe splits** — a leave-one-MMR-gene-out manifest is written
   alongside the data (the real generalization test with only 4 genes).

6. **Balanced-label diagnostic subset** per gene (VariPred's recipe) so the
   fused model can later be checked for shortcut-learning of gene identity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical MMR reference (PROJECT_PLAN.md Phase 1, first bullet)
# --------------------------------------------------------------------------- #
MMR_GENES: Tuple[str, ...] = ("MLH1", "MSH2", "MSH6", "PMS2")

#: gene -> (uniprot accession, canonical length in aa).
MMR_UNIPROT: Dict[str, Tuple[str, int]] = {
    "MLH1": ("P40692", 756),
    "MSH2": ("P43246", 934),
    "MSH6": ("P52701", 1360),
    "PMS2": ("P54278", 862),
}

#: Star rating -> evidence-quality weight (matches extended_builder mapping).
STAR_WEIGHT: Dict[int, float] = {1: 0.50, 2: 0.75, 3: 1.00, 4: 1.00}


def resolve_mmr_panel(session=None, overwrite: bool = False) -> Dict[str, Dict[str, str]]:
    """Fetch and validate canonical sequences for the four pinned accessions.

    Returns ``{gene: {"accession": ..., "sequence": ...}}`` in the same shape
    consumed by :func:`src.extended_builder.build_extended_dataset`.

    Raises :class:`ValueError` when the live UniProt entry disagrees with the
    pinned accession or canonical length — callers must notice a coordinate
    system change rather than silently continue.
    """
    from .data_loader import fetch_uniprot_sequence

    sequences: Dict[str, Dict[str, str]] = {}
    for gene, (acc, expected_len) in MMR_UNIPROT.items():
        seq = fetch_uniprot_sequence(acc, session=session)
        if len(seq) != expected_len:
            raise ValueError(
                f"{gene}: UniProt {acc} sequence length {len(seq)} != expected "
                f"{expected_len}. The canonical reference changed; update "
                f"MMR_UNIPROT deliberately before rebuilding.")
        invalid = set(seq) - set("ACDEFGHIKLMNPQRSTVWYX")
        if invalid:
            raise ValueError(f"{gene}: unexpected residues {sorted(invalid)}.")
        sequences[gene] = {"accession": acc, "sequence": seq}
        logger.info("Panel %s -> %s (%d aa) validated.", gene, acc, len(seq))
    return sequences


def panel_frame(records: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Records dict -> panel DataFrame (gene, uniprot_id, sequence)."""
    return pd.DataFrame([
        {"gene": g, "uniprot_id": d["accession"], "sequence": d["sequence"]}
        for g, d in records.items()
    ])


# --------------------------------------------------------------------------- #
# Expert-panel (InSiGHT/ClinGen VCEP) tiering
# --------------------------------------------------------------------------- #
EXPERT_PANEL_MARKERS = ("reviewed by expert panel", "practice guideline")


def add_evidence_tiers(df: pd.DataFrame,
                       review_col: str = "review_status",
                       star_col: str = "_stars") -> pd.DataFrame:
    """Flag InSiGHT/ClinGen VCEP calls and attach evidence-tier metadata.

    Adds:
    * ``expert_panel``      — 1 for expert-panel/practice-guideline reviews.
    * ``evidence_tier``     — {expert, high(2-3 stars), moderate(1 star)}.
    * ``tier_weight``       — numeric weight; expert rows always 1.0.
    """
    out = df.copy()
    if review_col not in out.columns:
        raise ValueError(f"Column '{review_col}' required for tiering.")
    # fillna first: with pandas' arrow-backed str dtype, astype(str) is a
    # no-op that leaves NA values untouched (they surface as floats later).
    status = (out[review_col].fillna("").astype(str)
              .str.strip().str.lower())
    is_expert = status.apply(lambda s: any(m in s for m in EXPERT_PANEL_MARKERS))
    out["expert_panel"] = is_expert.astype(int)

    if star_col in out.columns:
        stars = pd.to_numeric(out[star_col], errors="coerce").fillna(0).astype(int)
    else:
        from .data_loader import stars_for_review
        stars = status.map(stars_for_review).astype(int)
        out[star_col] = stars

    out["evidence_tier"] = np.select(
        [out["expert_panel"] == 1, stars >= 2, stars >= 1],
        ["expert", "high", "moderate"],
        default="unreviewed",
    )
    out["tier_weight"] = stars.map(STAR_WEIGHT).fillna(0.5)
    out.loc[out["expert_panel"] == 1, "tier_weight"] = 1.0
    return out


# --------------------------------------------------------------------------- #
# PMS2 exon 11–15 pseudogene-homology gate (hard filter, fail-closed)
# --------------------------------------------------------------------------- #
def variant_key(gene: str, position, wt_aa: str, mut_aa: str) -> str:
    """Canonical string join key robust to int/np.int64/Int64 dtype drift."""
    return f"{str(gene).upper()}|{int(position)}|{str(wt_aa)}|{str(mut_aa)}"


def apply_pms2_homology_gate(
    master: pd.DataFrame,
    homology_csv: Optional[Path] = None,
    codon_range: Optional[Tuple[int, int]] = None,
    exclude_pms2: bool = False,
) -> pd.DataFrame:
    """Withhold supervision for unconfirmed PMS2CL-homology-region variants.

    Parameters
    ----------
    master:
        Table with at least ``gene, position, wt_aa, mut_aa`` (+ optional
        ``label``, ``label_weight``).
    homology_csv:
        CSV of PMS2 exon 11–15 substitutions with columns
        ``gene, position, wt_aa, mut_aa, orthogonally_confirmed``.  Rows listed
        here with a truthy flag keep their supervision; every other variant
        inside the region loses its label (kept for provenance/inference only).
    codon_range:
        Explicit ``(start, end)`` inclusive protein-coordinate span of the
        homology region.  Use only when you have verified the mapping yourself
        (e.g. from an Ensembl exon table); this module refuses to invent one.
    exclude_pms2:
        Drop all PMS2 rows outright (three-gene fallback until a confirmation
        table exists).

    Fail-closed contract: if none of the three options is supplied the call
    raises.  A soft default would silently trust unreliable NGS calls.
    """
    has_pms2 = (master["gene"] == "PMS2").any()
    out = master.copy()

    if exclude_pms2:
        n_before = len(out)
        out = out[out["gene"] != "PMS2"].reset_index(drop=True)
        logger.warning("PMS2 excluded entirely (%d rows dropped); three-gene "
                       "dataset built until a homology-region confirmation "
                       "table is available.", n_before - len(out))
        out["pms2_homology_excluded"] = 0
        return out

    if homology_csv is None and codon_range is None:
        if has_pms2:
            raise ValueError(
                "PMS2 requires a validated exon 11-15 homology-region input: "
                "pass --pms2_homology_csv (with orthogonally_confirmed flags), "
                "--pms2_codon_range START END, or --exclude_pms2. Refusing to "
                "trust short-read calls in the PMS2CL homology region.")
        out["pms2_homology_excluded"] = 0
        return out

    confirmed_keys = None
    if homology_csv is not None:
        conf = pd.read_csv(homology_csv,
                           usecols=["gene", "position", "wt_aa", "mut_aa",
                                    "orthogonally_confirmed"])
        key = ["gene", "position", "wt_aa", "mut_aa"]
        conf["position"] = pd.to_numeric(conf["position"], errors="coerce").astype("Int64")
        conf["_confirmed"] = (conf["orthogonally_confirmed"].astype(str)
                              .str.strip().str.lower()
                              .isin({"1", "true", "yes", "y"})).astype(int)
        if conf["_confirmed"].sum() == 0:
            raise ValueError(
                f"{homology_csv} lists no orthogonally-confirmed PMS2 variants; "
                "the gate cannot be satisfied.")
        confirmed_keys = {
            variant_key(g, p, w, m)
            for g, p, w, m in conf.loc[conf["_confirmed"] == 1, key]
            .itertuples(index=False, name=None)
        }
        logger.info("PMS2 gate: %d/%d homology-region substitutions marked "
                    "orthogonally confirmed.",
                    int(conf["_confirmed"].sum()), len(conf))

    is_pms2 = (out["gene"] == "PMS2").to_numpy()
    pos_num = pd.to_numeric(out["position"], errors="coerce")
    in_region = np.zeros(len(out), dtype=bool)
    if codon_range is not None:
        start, end = sorted(int(v) for v in codon_range)
        in_region |= (is_pms2 & pos_num.between(start, end).fillna(False).to_numpy())
    if homology_csv is not None:
        # Without independently verified exon->codon boundaries the CSV's
        # coverage is the only trust anchor, so every non-listed PMS2 variant
        # is treated as inside the (unknown) homology region.
        in_region |= is_pms2

    unsafe = np.zeros(len(out), dtype=bool)
    if homology_csv is not None:
        idx = np.flatnonzero(in_region & is_pms2)
        keys = [variant_key(g, p, w, m) for g, p, w, m in
                out.iloc[idx][["gene", "position", "wt_aa", "mut_aa"]]
                .itertuples(index=False, name=None)]
        unsafe[idx] = [k not in confirmed_keys for k in keys]
    else:
        # Codon-range-only mode: nothing inside the range can be confirmed.
        unsafe = in_region.copy()

    out["pms2_homology_excluded"] = unsafe.astype(int)
    if unsafe.any():
        logger.warning(
            "PMS2 gate: withholding labels for %d unconfirmed homology-region "
            "variants (retained for provenance/inference).", int(unsafe.sum()))
        out.loc[unsafe, "label"] = pd.NA
        if "label_weight" in out.columns:
            out.loc[unsafe, "label_weight"] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Balanced-label diagnostic subsets (VariPred recipe)
# --------------------------------------------------------------------------- #
def make_balanced_subset(labeled: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Per-gene class-balanced diagnostic frame via majority-class down-sampling.

    Keeps every minority-class row and an equal-sized random majority-class
    sample.  Used to verify the model is not shortcut-learning gene identity
    (a balanced pooled set forces per-gene discrimination rather than prior
    exploitation).
    """
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for gene, sub in labeled.groupby("gene"):
        sub = sub.reset_index(drop=True)
        pos_idx = np.flatnonzero(sub["label"].to_numpy() == 1)
        neg_idx = np.flatnonzero(sub["label"].to_numpy() == 0)
        n_keep = min(len(pos_idx), len(neg_idx))
        if n_keep == 0:
            logger.warning("%s: single-class labelled set; skipping balance.", gene)
            continue
        keep_pos = rng.choice(pos_idx, size=n_keep, replace=False) \
            if len(pos_idx) > n_keep else pos_idx
        keep_neg = rng.choice(neg_idx, size=n_keep, replace=False) \
            if len(neg_idx) > n_keep else neg_idx
        parts.append(sub.iloc[np.sort(np.concatenate([keep_pos, keep_neg]))])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Leave-one-MMR-gene-out split manifest
# --------------------------------------------------------------------------- #
def write_leave_one_gene_out_manifest(processed_dir: Path,
                                      genes: Sequence[str] = MMR_GENES) -> Path:
    """Emit the LOPO split manifest used by every downstream evaluation."""
    genes = [g.upper() for g in genes]
    manifest = {
        "strategy": "leave_one_mmr_gene_out",
        "rationale": (
            "The real generalization test given only 4 genes total "
            "(PROJECT_PLAN.md Phase 1, circularity-safe split)."),
        "splits": [
            {"holdout_gene": g,
             "train_genes": [x for x in genes if x != g]}
            for g in genes
        ],
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "leave_one_gene_out_splits.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("LOPO manifest -> %s", path)
    return path


def load_leave_one_gene_out_manifest(path: Path) -> List[Tuple[str, List[str]]]:
    payload = json.loads(Path(path).read_text())
    return [(s["holdout_gene"], s["train_genes"]) for s in payload["splits"]]


__all__ = [
    "MMR_GENES", "MMR_UNIPROT", "STAR_WEIGHT",
    "resolve_mmr_panel", "panel_frame",
    "add_evidence_tiers", "apply_pms2_homology_gate", "variant_key",
    "make_balanced_subset",
    "write_leave_one_gene_out_manifest", "load_leave_one_gene_out_manifest",
]
