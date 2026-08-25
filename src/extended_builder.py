"""Construction of the unified *extended* variant-pathogenicity dataset.

The builder stitches five independent evidence sources into one tidy,
UniProt-addressable table plus a machine-readable provenance manifest:

======================  ======================================================
Source                  Contribution
======================  ======================================================
ClinVar                 Binary supervision (P/LP=1 vs B/LB=0) + VUS holdout
ProteinGym DMS v1.3     Continuous fitness scores + assay-binarised labels
AlphaMissense 2023      Per-variant pathogenicity score/class (prior feature)
ProteinGym zero-shot    ~17 published model scores (EVE, ESM1b, GEMME, ...)
UniProt domains         ``in_domain`` structural flag
gnomAD v4 (opt-in)      Variant AF + BA1/BS1/PM2 flags, gene-level constraint
AlphaFold DB (opt-in)   Per-residue pLDDT structural-confidence feature
InterPro (opt-in)       Domain/family/superfamily calls, complements UniProt
UniProt point features  ``is_functional_site`` (active/binding site, PTM, ...)
======================  ======================================================

Join keys and numbering guarantees
----------------------------------
* Every source is normalised to ``(uniprot_id, position, wt_aa, mut_aa)``.
* AlphaMissense already uses UniProt accessions.
* The ProteinGym clinical bundle uses RefSeq NP_ accessions; NP_ -> UniProt
  mapping happens by **exact whole-protein sequence equality** against the
  panel's canonical sequences, which also guarantees identical residue
  numbering for positional joins. Near-matches are dropped and counted.
* ClinVar rows are validated against the canonical sequence (wt residue must
  match at that position) before entering the merged table.

Master binary label precedence
------------------------------
All raw source columns are preserved.  The resolved ``label`` uses:

1. ClinVar labelled assertion (>= min_stars)
2. ProteinGym clinical benchmark label
3. ProteinGym DMS binarised label -- only when exactly one assay covers the
   substitution; multi-assay conflicts keep NaN.

Continuous DMS scores are never collapsed in the master row (median across
covering assays); full per-assay detail lives in ``dms_scores_long.csv``.

Outputs (all under ``data/processed/extended/``)
------------------------------------------------
=============================  ==============================================
extended_dataset.csv           master long-format table (unique substitutions)
dms_scores_long.csv            full per-assay DMS scores
alphamissense_subset.csv       AlphaMissense rows for the panel proteins
zeroshot_scores_subset.csv     published-model scores mapped to the panel
gnomad_subset.csv              gnomAD v4 AF per variant (opt-in)
alphafold_plddt_subset.csv     AlphaFold per-residue pLDDT (opt-in)
interpro_intervals.csv         InterPro domain/family/superfamily calls (opt-in)
uniprot_domains.csv            domain intervals for the panel
uniprot_functional_sites.csv   active/binding site, PTM, disulfide (opt-in)
panel_sequences.fasta|json     canonical reference sequences
manifest.json                  provenance, checksums, counts, parameters
=============================  ==============================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import (
    ONE_TO_THREE,
    build_multi_gene_dataset,
    fetch_uniprot_accession,
    fetch_uniprot_sequence,
    make_session,
    stars_for_review,
)
from .esm_extractor import validate_and_align
from .gnomad import (
    DEFAULT_BA1_AF,
    DEFAULT_BS1_AF,
    DEFAULT_PM2_AF,
    GENE_CONSTRAINT_COLUMNS,
    GNOMAD_FEATURE_COLUMNS,
    add_frequency_flags,
    attach_gene_constraint,
    fetch_gene_gnomad_variants,
    load_or_fetch_constraint,
    validate_against_sequence as validate_gnomad_against_sequence,
)
from .external_datasets import (
    ALPHAMISSENSE_AA_URL,
    ALPHAMISSENSE_VERSION,
    PROTEINGYM_BASE_HARVARD,
    PROTEINGYM_VERSION,
    PROTEINGYM_ZENODO_RECORD,
    attach_functional_site_features,
    download_alphamissense,
    download_proteingym,
    fetch_uniprot_domains,
    fetch_uniprot_point_features,
    load_clinical_benchmark,
    load_dms_assays,
    load_zero_shot_scores,
    map_np_to_panel,
    model_col_name,
    resolve_uniprot_entry_names,
    sha256_of,
    stream_filter_alphamissense,
    ZERO_SHOT_MODEL_COLUMNS,
)
from .structure import (
    ALPHAFOLD_FEATURE_COLUMNS,
    attach_alphafold_features,
    load_panel_alphafold_features,
)
from .interpro import (
    INTERPRO_FEATURE_COLUMNS,
    attach_interpro_features,
    load_panel_interpro_intervals,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default gene panel
# --------------------------------------------------------------------------- #
#: Every gene is covered by at least two sources:
#: TP53/BRCA1/PTEN -> large ClinVar sets *and* ProteinGym DMS assays;
#: KCNQ1 -> sizeable ClinVar cardiac channel set;
#: CALM1/HRAS/RHO/BTK/NUDT15/GAA -> well-characterised human DMS assays.
DEFAULT_GENE_PANEL: Tuple[str, ...] = (
    "TP53", "BRCA1", "PTEN", "KCNQ1", "CALM1",
    "HRAS", "RHO", "BTK", "NUDT15", "GAA",
)

EXTENDED_DIRNAME = "extended"

#: Join key for every source-level merge.
MASTER_KEY = ["uniprot_id", "position", "wt_aa", "mut_aa"]
#: Columns copied into the base union table (keys + human-readable meta).
MASTER_BASE_COLS = ["gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p"]


@dataclass
class BuildStats:
    """Counters collected during the build; serialised into the manifest."""

    genes_requested: List[str] = field(default_factory=list)
    genes_resolved: Dict[str, str] = field(default_factory=dict)
    genes_without_clinvar: List[str] = field(default_factory=list)
    clinvar_labelled: int = 0
    clinvar_vus: int = 0
    clinvar_dropped_alignment: int = 0
    dms_rows_total_human: int = 0
    dms_rows_panel: int = 0
    dms_rows_dropped_wt_mismatch: int = 0
    dms_assays_panel: int = 0
    clinical_rows_total: int = 0
    clinical_np_matched: int = 0
    clinical_np_unmatched: int = 0
    alphamissense_rows_panel: int = 0
    gnomad_rows_panel: int = 0
    gnomad_genes_fetched: List[str] = field(default_factory=list)
    gnomad_genes_failed: List[str] = field(default_factory=list)
    gnomad_constraint_genes_fetched: List[str] = field(default_factory=list)
    gnomad_constraint_genes_failed: List[str] = field(default_factory=list)
    alphafold_residues_with_plddt: int = 0
    alphafold_accessions_covered: int = 0
    interpro_intervals: int = 0
    functional_site_rows: int = 0
    zeroshot_rows_joined: int = 0
    domain_intervals: int = 0
    master_rows: int = 0
    master_label_counts: Dict[str, int] = field(default_factory=dict)
    multi_source_overlaps: Dict[str, int] = field(default_factory=dict)
    clinvar_conflicting_keys: int = 0
    clinical_conflicting_keys: int = 0
    dms_conflicting_keys: int = 0


def panel_frame_from_records(records: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Build the panel frame from pre-resolved ``{gene: {accession, sequence}}``."""
    return pd.DataFrame(
        [{"gene": g, "uniprot_id": d["accession"], "sequence": d["sequence"]}
         for g, d in records.items()]
    )


# --------------------------------------------------------------------------- #
# Panel resolution (gene -> accession + canonical sequence), cached
# --------------------------------------------------------------------------- #
def resolve_panel(
    genes: Sequence[str],
    processed_dir: Path,
    session=None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Resolve gene symbols to reviewed human UniProt accessions+sequences.

    Cached as JSON at ``processed/extended/panel_sequences.json``; the cache is
    invalidated automatically when the requested gene set changes.
    """
    ext_dir = processed_dir / EXTENDED_DIRNAME
    ext_dir.mkdir(parents=True, exist_ok=True)
    cache = ext_dir / "panel_sequences.json"
    wanted_keys = {g.upper() for g in genes}
    if cache.exists() and not overwrite:
        payload = json.loads(cache.read_text())
        if set(payload.keys()) == wanted_keys:
            logger.info("Panel loaded from cache (%d genes).", len(payload))
            return pd.DataFrame(
                [{"gene": g, "uniprot_id": d["accession"], "sequence": d["sequence"]}
                 for g, d in payload.items()]
            )
        logger.info("Panel cache gene-set mismatch; re-resolving.")
    sess = session or make_session()
    records: Dict[str, Dict[str, str]] = {}
    for gene in genes:
        acc = fetch_uniprot_accession(gene, session=sess)
        seq = fetch_uniprot_sequence(acc, session=sess)
        records[gene.upper()] = {"accession": acc, "sequence": seq}
    cache.write_text(json.dumps(records, indent=2))
    return pd.DataFrame(
        [{"gene": g, "uniprot_id": d["accession"], "sequence": d["sequence"]}
         for g, d in records.items()]
    )


def _wt_ok_mask(df: pd.DataFrame, seq_of: Dict[str, str]) -> np.ndarray:
    """Vectorised wt-residue validation against canonical sequences.

    Builds one concatenated reference string with per-accession offsets so the
    residue lookup is a single array gather; scales to millions of rows.
    """
    if not len(df):
        return np.zeros(0, dtype=bool)
    accessions = [a for a, s in seq_of.items() if s]
    offsets = {a: off for a, off in
               zip(accessions, np.cumsum([0] + [len(seq_of[a]) for a in accessions]))}
    big_seq = "".join(seq_of[a] for a in accessions)

    pos = pd.to_numeric(df["position"], errors="coerce").to_numpy(dtype="float64")
    known = df["uniprot_id"].map(offsets).fillna(-1).to_numpy(dtype="float64")
    length_of = df["uniprot_id"].map({a: float(len(seq_of[a])) for a in accessions}) \
        .fillna(0.0).to_numpy(dtype="float64")
    in_bounds = (known >= 0) & (pos >= 1) & (pos <= length_of)

    idx = np.clip((known + pos - 1), 0, max(len(big_seq) - 1, 0)).astype(np.int64)
    chars = np.frombuffer(big_seq.encode("ascii"), dtype=np.uint8)
    wt_codes = df["wt_aa"].map(
        lambda w: ord(w[0]) if isinstance(w, str) and w else -1).to_numpy()
    return in_bounds & (chars[idx] == wt_codes)


def _resolve_binary_evidence(
    df: pd.DataFrame, key: List[str], label_col: str, priority_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic evidence and keys with contradictory labels.

    Duplicates with the same label are reduced deterministically.  A key with
    both binary labels is never resolved by archive/row order: it is excluded
    from that source's supervision and returned in ``conflicts``.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=key)
    labeled = df[df[label_col].notna()].copy()
    n_labels = labeled.groupby(key, dropna=False)[label_col].nunique()
    conflicts = n_labels[n_labels > 1].reset_index()[key]
    if len(conflicts):
        flagged = labeled.merge(conflicts.assign(_conflict=True), on=key, how="left")
        labeled = flagged[flagged["_conflict"].isna()].drop(columns="_conflict")
    sort_cols = key + ([priority_col] if priority_col and priority_col in labeled else [])
    ascending = [True] * len(key) + ([False] if priority_col and priority_col in labeled else [])
    return (labeled.sort_values(sort_cols, ascending=ascending)
            .drop_duplicates(subset=key, keep="first"), conflicts)


# --------------------------------------------------------------------------- #
# Per-source builders
# --------------------------------------------------------------------------- #
def build_clinvar_part(
    panel: pd.DataFrame, raw_dir: Path, processed_dir: Path,
    min_stars: int, stats: BuildStats,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-gene ClinVar extraction aligned to canonical sequences."""
    labeled, vus = build_multi_gene_dataset(
        list(panel["gene"]), raw_dir, processed_dir, min_stars=min_stars)
    seq_of = dict(zip(panel["gene"], panel["sequence"]))
    acc_of = dict(zip(panel["gene"], panel["uniprot_id"]))

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        n_before = len(df)
        for gene, sub in df.groupby("gene"):
            sub = validate_and_align(sub.copy(), seq_of[str(gene)])
            sub["uniprot_id"] = acc_of[str(gene)]
            parts.append(sub)
        stats.clinvar_dropped_alignment += n_before - sum(len(p) for p in parts)
        return pd.concat(parts, ignore_index=True)

    labeled = _align(labeled)
    vus = _align(vus)
    stats.clinvar_labelled = len(labeled)
    stats.clinvar_vus = len(vus)
    resolved = set(labeled["gene"]) | set(vus["gene"])
    stats.genes_without_clinvar = sorted(set(map(str, panel["gene"])) - resolved)
    return labeled, vus


def build_dms_part(panel: pd.DataFrame, raw_dir: Path, stats: BuildStats) -> pd.DataFrame:
    """Human DMS assays restricted to panel accessions (wt-validated).

    ProteinGym reference ``UniProt_ID`` values are entry names
    (``BRCA1_HUMAN``); they are translated to primary accessions first.
    """
    dms = load_dms_assays(raw_dir, taxon="human")
    stats.dms_rows_total_human = len(dms)
    name_map = resolve_uniprot_entry_names(
        sorted(dms["uniprot_id"].unique()), raw_dir)
    dms = dms.assign(uniprot_acc=dms["uniprot_id"].map(name_map))
    unmapped = int(dms["uniprot_acc"].isna().sum())
    if unmapped:
        logger.info("Dropping %d DMS rows without accession mapping.", unmapped)
    sub = dms[dms["uniprot_acc"].isin(set(panel["uniprot_id"]))].copy()
    sub["uniprot_id"] = sub["uniprot_acc"]
    sub["gene"] = sub["uniprot_id"].map(dict(zip(panel["uniprot_id"], panel["gene"])))
    ok = _wt_ok_mask(sub, dict(zip(panel["uniprot_id"], panel["sequence"])))
    stats.dms_rows_dropped_wt_mismatch = int((~ok).sum())
    if stats.dms_rows_dropped_wt_mismatch:
        logger.warning(
            "Dropped %d DMS mutants whose wt residue mismatches the canonical "
            "sequence (isoform numbering); kept %d.",
            stats.dms_rows_dropped_wt_mismatch, int(ok.sum()))
    sub = sub.loc[ok].reset_index(drop=True)
    stats.dms_rows_panel = len(sub)
    stats.dms_assays_panel = int(sub["dms_id"].nunique())
    return sub


def build_alphamissense_part(
    panel: pd.DataFrame, raw_dir: Path, ext_dir: Path, stats: BuildStats
) -> pd.DataFrame:
    gz = download_alphamissense(raw_dir)
    am = stream_filter_alphamissense(
        gz, ext_dir / "alphamissense_subset.csv", panel["uniprot_id"])
    ok = _wt_ok_mask(am, dict(zip(panel["uniprot_id"], panel["sequence"])))
    if int((~ok).sum()):
        logger.warning("Dropped %d AlphaMissense rows with wt mismatch.", int((~ok).sum()))
    am = am.loc[ok].reset_index(drop=True)
    stats.alphamissense_rows_panel = len(am)
    return am


def build_gnomad_part(
    panel: pd.DataFrame, raw_dir: Path, ext_dir: Path, stats: BuildStats,
    session=None, overwrite: bool = False,
    ba1_af: float = DEFAULT_BA1_AF, bs1_af: float = DEFAULT_BS1_AF,
    pm2_af: float = DEFAULT_PM2_AF,
) -> pd.DataFrame:
    """gnomAD v4 allele-frequency features for every panel gene (cached per gene).

    PROJECT_PLAN.md Phase 3 step 1 treats gnomAD AF as an explicit **input
    feature** (not just an ACMG filtering criterion) for the whole pretraining
    panel, not only the four MMR genes — MVmamba's own ablation improved every
    metric by adding AF on top of a strong structure+sequence model. One
    GraphQL call per gene (:mod:`src.gnomad`), so this is opt-in
    (``include_gnomad=True``) and can be slow for a large panel; per-gene
    results are cached under ``data/raw/gnomad/`` and reused on rebuild.
    """
    from .data_loader import make_session as _make_session

    sess = session or _make_session()
    cache_dir = raw_dir / "gnomad"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_of = dict(zip(panel["uniprot_id"], panel["sequence"]))
    frames: List[pd.DataFrame] = []
    for gene, uniprot_id in zip(panel["gene"], panel["uniprot_id"]):
        gene_up = str(gene).upper()
        cache = cache_dir / f"{gene_up}_gnomad_v4.csv"
        try:
            if cache.exists() and not overwrite:
                df = pd.read_csv(cache)
            else:
                df = fetch_gene_gnomad_variants(gene_up, session=sess)
                if len(df):
                    df = validate_gnomad_against_sequence(df, seq_of[uniprot_id])
                df.to_csv(cache, index=False)
        except Exception as exc:  # noqa: BLE001 - one gene's API failure must not abort the panel build
            logger.warning("gnomAD: %s fetch failed (%s); skipped.", gene_up, exc)
            stats.gnomad_genes_failed.append(gene_up)
            continue
        if df.empty:
            continue
        df = df.copy()
        df["uniprot_id"] = uniprot_id
        frames.append(df)
        stats.gnomad_genes_fetched.append(gene_up)

    if not frames:
        stats.gnomad_rows_panel = 0
        return pd.DataFrame(columns=["uniprot_id", "position", "wt_aa", "mut_aa",
                                     *GNOMAD_FEATURE_COLUMNS])
    gnomad_panel = add_frequency_flags(pd.concat(frames, ignore_index=True),
                                       ba1_af=ba1_af, bs1_af=bs1_af, pm2_af=pm2_af)
    gnomad_panel.to_csv(ext_dir / "gnomad_subset.csv", index=False)
    stats.gnomad_rows_panel = len(gnomad_panel)
    return gnomad_panel


def build_gnomad_constraint_part(
    panel: pd.DataFrame, raw_dir: Path, stats: BuildStats,
    session=None, overwrite: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Gene-level gnomAD constraint metrics (pLI, oe_mis, mis_z, ...) per panel gene.

    Cheap (one small GraphQL call per gene, cached); a gene-level complement
    to the variant-level AF features from :func:`build_gnomad_part`.
    """
    from .data_loader import make_session as _make_session
    sess = session or _make_session()
    out: Dict[str, Dict[str, float]] = {}
    for gene in panel["gene"]:
        gene_up = str(gene).upper()
        try:
            out[gene_up] = load_or_fetch_constraint(gene_up, raw_dir, session=sess,
                                                     overwrite=overwrite)
            stats.gnomad_constraint_genes_fetched.append(gene_up)
        except Exception as exc:  # noqa: BLE001 - one gene's failure must not abort the panel
            logger.warning("gnomAD constraint: %s fetch failed (%s); skipped.", gene_up, exc)
            stats.gnomad_constraint_genes_failed.append(gene_up)
    return out


def build_structure_part(panel: pd.DataFrame, raw_dir: Path, ext_dir: Path,
                         stats: BuildStats, session=None, overwrite: bool = False) -> pd.DataFrame:
    """Per-residue AlphaFold pLDDT structural-confidence features for the panel."""
    from .data_loader import make_session as _make_session
    sess = session or _make_session()
    af_panel = load_panel_alphafold_features(panel, raw_dir, session=sess, overwrite=overwrite)
    if len(af_panel):
        af_panel.to_csv(ext_dir / "alphafold_plddt_subset.csv", index=False)
    stats.alphafold_residues_with_plddt = len(af_panel)
    stats.alphafold_accessions_covered = int(af_panel["uniprot_id"].nunique()) if len(af_panel) else 0
    return af_panel


def build_interpro_part(panel: pd.DataFrame, raw_dir: Path, ext_dir: Path,
                        stats: BuildStats, session=None, overwrite: bool = False) -> pd.DataFrame:
    """InterPro domain/family/superfamily intervals for the panel."""
    from .data_loader import make_session as _make_session
    sess = session or _make_session()
    intervals = load_panel_interpro_intervals(panel, raw_dir, session=sess, overwrite=overwrite)
    if len(intervals):
        intervals.to_csv(ext_dir / "interpro_intervals.csv", index=False)
    stats.interpro_intervals = len(intervals)
    return intervals


def build_functional_sites_part(panel: pd.DataFrame, raw_dir: Path, ext_dir: Path,
                                stats: BuildStats, session=None) -> pd.DataFrame:
    """Single-residue UniProt functional-site annotations for the panel.

    Reuses the UniProt JSON already cached by the domain-interval fetch (see
    :func:`src.external_datasets.fetch_uniprot_point_features`), so this adds
    no extra network calls when domains were already built for the same panel.
    """
    from .data_loader import make_session as _make_session
    sess = session or _make_session()
    sites = fetch_uniprot_point_features(list(panel["uniprot_id"]), raw_dir, session=sess)
    if len(sites):
        sites.to_csv(ext_dir / "uniprot_functional_sites.csv", index=False)
    stats.functional_site_rows = len(sites)
    return sites


def build_zeroshot_and_clinical_part(
    panel: pd.DataFrame, raw_dir: Path, stats: BuildStats
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Clinical labels + published-model scores mapped onto the panel."""
    clin = load_clinical_benchmark(raw_dir)
    stats.clinical_rows_total = len(clin)
    zs = load_zero_shot_scores(raw_dir)

    mapping = map_np_to_panel(clin.attrs.get("np_sequences", {}), panel)
    stats.clinical_np_matched = len(mapping)
    stats.clinical_np_unmatched = len(clin.attrs.get("np_sequences", {})) - len(mapping)
    if mapping.empty:
        logger.warning("No RefSeq NP accessions matched panel proteins exactly.")
        return None, None

    clin = clin.merge(mapping, on="np_accession", how="inner")
    zs = zs.merge(mapping[["np_accession", "uniprot_id"]], on="np_accession",
                  how="inner")

    seq_of = dict(zip(panel["uniprot_id"], panel["sequence"]))
    ok = _wt_ok_mask(clin, seq_of)
    clin = clin.loc[ok].reset_index(drop=True)
    clin["clinical_label"] = clin["clinical_label"].astype("Int64")
    # Zero-shot rows are joined to the wt-validated clinical table by
    # (uniprot_id, mutant) downstream, so no separate wt check is needed here.
    return clin, zs


# --------------------------------------------------------------------------- #
# Master assembly
# --------------------------------------------------------------------------- #
def assemble_master(
    labeled: pd.DataFrame,
    vus: pd.DataFrame,
    dms_panel: pd.DataFrame,
    clin_panel: Optional[pd.DataFrame],
    am_panel: pd.DataFrame,
    zs_panel: Optional[pd.DataFrame],
    domains: pd.DataFrame,
    stats: BuildStats,
    gene_of_uniprot: Optional[Dict[str, str]] = None,
    gnomad_panel: Optional[pd.DataFrame] = None,
    gnomad_constraint: Optional[Dict[str, Dict[str, float]]] = None,
    alphafold_panel: Optional[pd.DataFrame] = None,
    interpro_intervals: Optional[pd.DataFrame] = None,
    functional_sites: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge all sources into the master table, one row per substitution.

    Column families in the output:

    * keys/meta      ``gene, uniprot_id, position, wt_aa, mut_aa, hgvs_p``
    * supervision    ``label`` (resolved), ``clinvar_label``, ``stars``,
                     ``clinical_label``, ``dms_bin_median``
    * DMS            ``dms_score_median``, ``n_dms_assays``, ``dms_ids``
    * AlphaMissense  ``am_pathogenicity``, ``am_class``
    * zero-shot      one ``zs_*`` column per published model (EVE, ESM1b, ...)
    * structure      ``in_domain``, ``domain_names``
    * provenance     ``sources`` (pipe-joined), ``n_sources``
    """
    key = MASTER_KEY

    def _base(df: pd.DataFrame, source: str) -> pd.DataFrame:
        # Tolerant column pick: some sources (e.g. cached AlphaMissense
        # subsets) may lack gene/hgvs_p; those are backfilled below.
        present = [c for c in MASTER_BASE_COLS if c in df.columns]
        out = df[present].reindex(columns=MASTER_BASE_COLS).copy()
        out["source"] = source
        return out

    union = pd.concat([
        _base(labeled, "clinvar"),
        _base(vus, "clinvar_vus"),
        _base(dms_panel, "dms"),
        *( [_base(clin_panel, "pg_clinical")] if clin_panel is not None else [] ),
        _base(am_panel, "alphamissense"),
    ], ignore_index=True)

    master = union.drop(columns="source").copy()

    # Backfill gene/hgvs_p for rows whose source lacked them BEFORE the
    # de-duplication below, otherwise e.g. an AlphaMissense row (no gene)
    # would survive as a twin of its ClinVar counterpart.
    if "gene" not in master.columns:
        master["gene"] = pd.NA
    master["gene"] = master["gene"].fillna(
        master["uniprot_id"].map(gene_of_uniprot))
    need_hgvs = master["hgvs_p"].isna()
    if need_hgvs.any():
        master.loc[need_hgvs, "hgvs_p"] = (
            "p." + master.loc[need_hgvs, "wt_aa"].map(ONE_TO_THREE)
            + master.loc[need_hgvs, "position"].astype(int).astype(str)
            + master.loc[need_hgvs, "mut_aa"].map(ONE_TO_THREE))

    master = master.drop_duplicates(subset=MASTER_BASE_COLS,
                                    keep="first").reset_index(drop=True)

    # --- ClinVar ---------------------------------------------------------- #
    # _stars is re-derived from review_status because the cached per-gene CSVs
    # only persist FINAL_COLUMNS.
    cv = pd.concat(
        [
            labeled.assign(_cv=labeled["label"],
                           _stars=labeled["review_status"].map(stars_for_review)),
            vus.assign(_cv=np.nan,
                       _stars=vus["review_status"].map(stars_for_review)),
        ],
        ignore_index=True,
    )
    cv_labels, cv_conflicts = _resolve_binary_evidence(cv, key, "_cv", "_stars")
    stats.clinvar_conflicting_keys = len(cv_conflicts)
    # Keep VUS review metadata for prospective scoring, but take labels only
    # from the conflict-filtered evidence table.
    cv_meta = cv.sort_values(key + ["_stars"], ascending=[True] * len(key) + [False]) \
        .drop_duplicates(subset=key, keep="first")
    cv = cv_meta.drop(columns="_cv").merge(cv_labels[key + ["_cv"]],
                                             on=key, how="left")
    cv["stars"] = cv["_stars"]
    master = master.merge(
        cv[key + ["_cv", "stars", "review_status"]].rename(columns={"_cv": "clinvar_label"}),
        on=key, how="left")
    master["clinvar_conflict"] = master[key].merge(
        cv_conflicts.assign(clinvar_conflict=1), on=key, how="left")["clinvar_conflict"] \
        .fillna(0).astype(int).to_numpy()

    # --- ProteinGym clinical benchmark ------------------------------------ #
    if clin_panel is not None and len(clin_panel):
        cp, cp_conflicts = _resolve_binary_evidence(clin_panel, key, "clinical_label", "np_accession")
        stats.clinical_conflicting_keys = len(cp_conflicts)
        master = master.merge(
            cp[key + ["clinical_label", "np_accession"]], on=key, how="left")
        master["clinical_conflict"] = master[key].merge(
            cp_conflicts.assign(clinical_conflict=1), on=key, how="left")["clinical_conflict"] \
            .fillna(0).astype(int).to_numpy()
    else:
        master["clinical_label"] = np.nan
        master["np_accession"] = pd.NA
        master["clinical_conflict"] = 0

    # --- DMS aggregation (median across assays + ids list) ---------------- #
    agg = (dms_panel.groupby(key)
           .agg(dms_score_median=("dms_score", "median"),
                dms_bin_median=("dms_bin", "median"),
                dms_bin_nunique=("dms_bin", "nunique"),
                n_dms_assays=("dms_id", "nunique"),
                dms_ids=("dms_id", lambda s: "|".join(sorted(set(s)))),
                dms_selection_types=("selection_type",
                                     lambda s: "|".join(sorted({str(x) for x in s if pd.notna(x)}))))
           .reset_index())
    stats.dms_conflicting_keys = int((agg["dms_bin_nunique"] > 1).sum())
    master = master.merge(agg, on=key, how="left")

    # --- AlphaMissense ----------------------------------------------------- #
    am_cols = am_panel[key + ["am_pathogenicity", "am_class"]].drop_duplicates(subset=key)
    master = master.merge(am_cols, on=key, how="left")

    # --- Zero-shot published-model scores ---------------------------------- #
    if zs_panel is not None and len(zs_panel):
        zs_small = zs_panel.copy()
        keep = [c for c in zs_small.columns
                if c.startswith("zs_") or c in ("uniprot_id", "mutant")]
        zs_small = zs_small[keep].drop_duplicates(subset=["uniprot_id", "mutant"])
        master["mutant"] = master["wt_aa"] + master["position"].astype(str) \
            + master["mut_aa"]
        master = master.merge(zs_small, on=["uniprot_id", "mutant"], how="left")
        stats.zeroshot_rows_joined = int(master[[c for c in master.columns
                                                 if c.startswith("zs_")]]
                                         .notna().any(axis=1).sum())
        master = master.drop(columns="mutant")
    else:
        for model in ZERO_SHOT_MODEL_COLUMNS:
            master[_model_name(model)] = np.nan

    # --- gnomAD v4 allele frequency (explicit input feature, opt-in) ------- #
    # Filled only when gnomad_panel actually covers the whole requested panel
    # (build_gnomad_part loops over every panel gene): a left-join miss is
    # then genuine "absent from gnomAD" and join_gnomad_features's PM2=1 /
    # BA1=BS1=0 fallback is correct. Genes that failed the GraphQL fetch
    # (stats.gnomad_genes_failed) are the one exception — their rows get the
    # same fallback even though gnomAD truly wasn't checked; this is logged
    # and treated as an acceptable, rare partial-failure caveat rather than a
    # blocking condition for the whole panel build.
    if gnomad_panel is not None and len(gnomad_panel):
        from .gnomad import join_gnomad_features
        master = join_gnomad_features(master, gnomad_panel)
    else:
        for c in GNOMAD_FEATURE_COLUMNS:
            master[c] = np.nan

    # --- gnomAD gene-level constraint (opt-in, broadcast per gene) --------- #
    if gnomad_constraint:
        master = attach_gene_constraint(master, gnomad_constraint)
    else:
        for c in GENE_CONSTRAINT_COLUMNS:
            master[c] = np.nan

    # --- AlphaFold per-residue pLDDT (opt-in) ------------------------------- #
    if alphafold_panel is not None and len(alphafold_panel):
        master = attach_alphafold_features(master, alphafold_panel)
    else:
        for c in ALPHAFOLD_FEATURE_COLUMNS:
            master[c] = np.nan
        master["af_plddt_bin"] = np.nan

    # --- InterPro domain/family/superfamily calls (opt-in) ------------------ #
    if interpro_intervals is not None and len(interpro_intervals):
        master = attach_interpro_features(master, interpro_intervals)
    else:
        master["in_interpro_domain"] = 0
        master["interpro_names"] = ""

    # --- UniProt point features: active/binding site, PTM, disulfide ------- #
    if functional_sites is not None and len(functional_sites):
        master = attach_functional_site_features(master, functional_sites)
    else:
        master["is_functional_site"] = 0
        master["functional_site_types"] = ""

    # --- UniProt domains --------------------------------------------------- #
    dom = domains.loc[domains["feature_type"].isin(["Domain", "Region"])] \
        if len(domains) else domains
    if len(dom) and len(master):
        positions = master[["uniprot_id", "position"]].drop_duplicates()
        pairs = positions.merge(dom, on="uniprot_id", how="inner")
        inside = ((pairs["start"] <= pairs["position"])
                  & (pairs["position"] <= pairs["end"]))
        names_by_pos = (
            pairs.loc[inside]
            .groupby(["uniprot_id", "position"])["description"]
            .agg(lambda s: "|".join(
                sorted({str(d) for d in s if pd.notna(d) and str(d)})))
        )
        joined = master[["uniprot_id", "position"]].merge(
            names_by_pos.rename("domain_names").reset_index(),
            on=["uniprot_id", "position"], how="left")
        master["in_domain"] = joined["domain_names"].notna().astype(int).to_numpy()
        master["domain_names"] = joined["domain_names"].fillna("").to_numpy()
    else:
        master["in_domain"] = 0
        master["domain_names"] = ""

    # --- Resolved label precedence ----------------------------------------- #
    # ProteinGym DMS_score_bin convention (verified empirically against
    # AlphaMissense and PG-clinical labels): bin=1 marks the TOP half of assay
    # fitness (functionally tolerated), bin=0 the bottom half (deleterious).
    # Pathogenicity supervision therefore requires the flip below.
    dms_single_ok = ((master["dms_bin_median"].notna())
                     & (master["n_dms_assays"] == 1)
                     & master["dms_bin_median"].isin([0, 1]))
    dms_pathogenic = 1 - master["dms_bin_median"]
    master["label"] = np.where(
        master["clinvar_label"].notna(), master["clinvar_label"],
        np.where(master["clinical_label"].notna(), master["clinical_label"],
                 np.where(dms_single_ok, dms_pathogenic, np.nan)),
    )
    cross_source_conflict = (master["clinvar_label"].notna()
                             & master["clinical_label"].notna()
                             & (master["clinvar_label"] != master["clinical_label"]))
    master["cross_source_conflict"] = cross_source_conflict.astype(int)
    master["label_conflict"] = (
        master["clinvar_conflict"].astype(bool)
        | master["clinical_conflict"].astype(bool)
        | cross_source_conflict
    ).astype(int)
    # Never silently train on contradictory clinical evidence. Raw source
    # columns remain available for investigation, while the resolved target is
    # deliberately withheld.
    master.loc[master["label_conflict"] == 1, "label"] = np.nan
    master["label_source"] = np.select(
        [master["clinvar_label"].notna(), master["clinical_label"].notna(), dms_single_ok],
        ["clinvar", "pg_clinical", "dms"], default="")
    # Confidence is explicit metadata, not a claim that DMS fitness is a
    # clinical ground truth. It gives downstream trainers a safe default for
    # source-aware losses and can be overridden by experiment configuration.
    star_weight = master["stars"].fillna(0).map({1: 0.50, 2: 0.75, 3: 1.0, 4: 1.0})
    master["label_weight"] = np.select(
        [master["label_source"] == "clinvar", master["label_source"] == "pg_clinical",
         master["label_source"] == "dms"],
        [star_weight.fillna(0.5), 0.75, 0.20], default=0.0).astype(float)
    master.loc[master["label_conflict"] == 1, "label_weight"] = 0.0

    # --- Provenance tags ---------------------------------------------------- #
    src_cols = {
        "clinvar": master["clinvar_label"].notna() | master["review_status"].notna(),
        "pg_clinical": master["clinical_label"].notna(),
        "dms": master["n_dms_assays"].notna() & (master["n_dms_assays"] > 0),
        "alphamissense": master["am_pathogenicity"].notna(),
        "zeroshot_models": master[[c for c in master.columns if c.startswith("zs_")]]
        .notna().any(axis=1),
        "domains": master["in_domain"] == 1,
        "gnomad": master["gnomad_af_joint"].notna(),
        "alphafold": master["af_plddt"].notna(),
        "interpro": master["in_interpro_domain"] == 1,
        "functional_site": master["is_functional_site"] == 1,
    }
    parts = []
    for name, mask in src_cols.items():
        parts.append(np.where(mask, name, ""))
    stacked = np.vstack(parts).T
    master["sources"] = ["|".join([s for s in row if s]) for row in stacked]
    master["n_sources"] = master["sources"].apply(
        lambda s: len([t for t in s.split("|") if t]))

    for name, mask in src_cols.items():
        stats.multi_source_overlaps[name] = int(mask.sum())

    order = [
        "gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p",
        "label", "label_source", "label_weight", "label_conflict", "cross_source_conflict",
        "clinvar_label", "clinvar_conflict", "stars", "review_status",
        "clinical_label", "clinical_conflict", "np_accession",
        "dms_score_median", "dms_bin_median", "dms_bin_nunique", "n_dms_assays", "dms_ids",
        "dms_selection_types",
        "am_pathogenicity", "am_class", "in_domain", "domain_names",
        *GNOMAD_FEATURE_COLUMNS, *GENE_CONSTRAINT_COLUMNS,
        *ALPHAFOLD_FEATURE_COLUMNS, "af_plddt_bin",
        "in_interpro_domain", "interpro_names",
        "is_functional_site", "functional_site_types",
        *[c for c in master.columns if c.startswith("zs_")],
        "sources", "n_sources",
    ]
    order = [c for c in order if c in master.columns] + \
        [c for c in master.columns if c not in order]
    master = master.sort_values(["gene", "position", "mut_aa"]).reset_index(drop=True)
    return master[order]


def _model_name(model: str) -> str:
    """Normalise a raw model header into a snake_case ``zs_*`` feature name."""
    return model_col_name(model)


def validate_master_for_export(master: pd.DataFrame) -> None:
    """Raise before export when a master table violates training invariants."""
    required = set(MASTER_KEY + ["gene", "hgvs_p", "label", "label_source",
                                 "label_weight", "label_conflict"])
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master table missing required columns: {sorted(missing)}")
    if master[MASTER_KEY].isna().any(axis=None):
        raise ValueError("Master table has null variant join keys.")
    if master.duplicated(subset=MASTER_KEY).any():
        raise ValueError("Master table has duplicate variant join keys.")
    if master["gene"].isna().any() or master["hgvs_p"].isna().any():
        raise ValueError("Master table has unresolved gene or HGVS metadata.")
    labels = pd.to_numeric(master["label"], errors="coerce")
    if (~labels.isna() & ~labels.isin([0, 1])).any():
        raise ValueError("Master table has non-binary resolved labels.")
    conflicts = master["label_conflict"].fillna(0).astype(int) == 1
    if (conflicts & labels.notna()).any():
        raise ValueError("Conflicting clinical evidence was not quarantined.")
    source = master["label_source"].fillna("")
    bad_source = labels.notna() & ~source.isin(["clinvar", "pg_clinical", "dms"])
    if bad_source.any():
        raise ValueError("Resolved labels are missing a recognized label source.")


def write_fasta(panel: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as fh:
        for _, row in panel.iterrows():
            fh.write(f">{row['uniprot_id']} {row['gene']}\n")
            seq = row["sequence"]
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def build_extended_dataset(
    genes: Sequence[str] = DEFAULT_GENE_PANEL,
    data_dir: Path = Path("data"),
    min_stars: int = 2,
    include_alphamissense: bool = True,
    include_zeroshot: bool = True,
    include_gnomad: bool = False,
    gnomad_ba1_af: float = DEFAULT_BA1_AF,
    gnomad_bs1_af: float = DEFAULT_BS1_AF,
    gnomad_pm2_af: float = DEFAULT_PM2_AF,
    include_structure: bool = False,
    include_interpro: bool = False,
    include_functional_sites: bool = False,
    overwrite_cache: bool = False,
    panel_records: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, object]:
    """Run the full build; returns a summary dict (also stored in manifest).

    Stages
    ------
    1. Resolve gene panel to UniProt canonical sequences (or use
       *panel_records* — ``{gene: {accession, sequence}}`` — directly).
    2. Download/parse ProteinGym bundles (cached).
    3. Multi-gene single-pass ClinVar extraction + alignment.
    4. AlphaMissense streaming filter (only if requested; heavy first run).
    5. Zero-shot clinical scores + RefSeq mapping.
    6. gnomAD v4 allele-frequency features (only if requested; one GraphQL
       call per panel gene, so this can add minutes for a large panel).
    7. gnomAD v4 gene-level constraint metrics (pLI, oe_mis, mis_z, ...;
       same opt-in flag as step 6, one small extra call per gene).
    8. AlphaFold DB per-residue pLDDT structural-confidence feature
       (only if requested; one REST call + one PDB download per accession).
    9. InterPro domain/family/superfamily calls (only if requested; one REST
       call per accession, complements UniProt's own domain annotations).
    10. UniProt domain annotations.
    11. UniProt point features -- active/binding site, PTM, disulfide bond
        (only if requested; reuses the UniProt JSON already cached by step 10,
        no extra network calls).
    12. Assemble master table, write artefacts + manifest.json.
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    ext_dir = processed_dir / EXTENDED_DIRNAME
    ext_dir.mkdir(parents=True, exist_ok=True)
    stats = BuildStats(genes_requested=[g.upper() for g in genes])

    session = make_session()

    if panel_records:
        records = {g.upper(): {"accession": v["accession"],
                               "sequence": v["sequence"]}
                   for g, v in panel_records.items()}
        cache_path = ext_dir / "panel_sequences.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if overwrite_cache or set(cached) != set(records):
            cache_path.write_text(json.dumps(records, indent=2))
        panel = panel_frame_from_records(records)
        logger.info("Panel injected from pre-resolved records (%d genes).",
                    len(panel))
    else:
        panel = resolve_panel(genes, processed_dir, session=session,
                              overwrite=overwrite_cache)
    stats.genes_resolved = dict(zip(panel["gene"].astype(str), panel["uniprot_id"]))
    write_fasta(panel, ext_dir / "panel_sequences.fasta")

    N = 11
    logger.info("[1/%d] ProteinGym bundles ...", N)
    download_proteingym(raw_dir, overwrite=overwrite_cache)

    logger.info("[2/%d] ClinVar multi-gene pass (%d genes) ...", N, len(panel))
    labeled, vus = build_clinvar_part(panel, raw_dir, processed_dir,
                                      min_stars=min_stars, stats=stats)

    logger.info("[3/%d] ProteinGym DMS assays ...", N)
    dms_panel = build_dms_part(panel, raw_dir, stats)
    dms_panel.to_csv(ext_dir / "dms_scores_long.csv", index=False)

    clin_panel = zs_panel = None
    if include_zeroshot:
        logger.info("[4/%d] Clinical benchmark + zero-shot model scores ...", N)
        clin_panel, zs_panel = build_zeroshot_and_clinical_part(panel, raw_dir, stats)
        if clin_panel is not None:
            clin_panel.to_csv(ext_dir / "pg_clinical_panel.csv", index=False)
        if zs_panel is not None:
            zs_panel.to_csv(ext_dir / "zeroshot_scores_subset.csv", index=False)

    am_panel = pd.DataFrame(columns=MASTER_BASE_COLS)
    if include_alphamissense:
        logger.info("[5/%d] AlphaMissense streaming filter ...", N)
        am_panel = build_alphamissense_part(panel, raw_dir, ext_dir, stats)

    gnomad_panel = None
    gnomad_constraint = None
    if include_gnomad:
        logger.info("[6/%d] gnomAD v4 allele frequencies (%d genes) ...", N, len(panel))
        gnomad_panel = build_gnomad_part(panel, raw_dir, ext_dir, stats,
                                         session=session, overwrite=overwrite_cache,
                                         ba1_af=gnomad_ba1_af, bs1_af=gnomad_bs1_af,
                                         pm2_af=gnomad_pm2_af)
        logger.info("[7/%d] gnomAD gene-level constraint (%d genes) ...", N, len(panel))
        gnomad_constraint = build_gnomad_constraint_part(
            panel, raw_dir, stats, session=session, overwrite=overwrite_cache)

    alphafold_panel = None
    if include_structure:
        logger.info("[8/%d] AlphaFold DB per-residue pLDDT (%d accessions) ...",
                    N, panel["uniprot_id"].nunique())
        alphafold_panel = build_structure_part(panel, raw_dir, ext_dir, stats,
                                               session=session, overwrite=overwrite_cache)

    interpro_intervals = None
    if include_interpro:
        logger.info("[9/%d] InterPro domain/family calls (%d accessions) ...",
                    N, panel["uniprot_id"].nunique())
        interpro_intervals = build_interpro_part(panel, raw_dir, ext_dir, stats,
                                                 session=session, overwrite=overwrite_cache)

    logger.info("[10/%d] UniProt domain annotations ...", N)
    domains = fetch_uniprot_domains(list(panel["uniprot_id"]), raw_dir, session=session)
    domains.to_csv(ext_dir / "uniprot_domains.csv", index=False)
    stats.domain_intervals = len(domains)

    functional_sites = None
    if include_functional_sites:
        logger.info("[11/%d] UniProt point features (active/binding site, PTM, ...) ...", N)
        functional_sites = build_functional_sites_part(panel, raw_dir, ext_dir, stats,
                                                        session=session)

    logger.info("Assembling master table ...")
    master = assemble_master(labeled, vus, dms_panel, clin_panel, am_panel,
                             zs_panel, domains, stats,
                             gene_of_uniprot=dict(zip(panel["uniprot_id"],
                                                      panel["gene"].astype(str))),
                             gnomad_panel=gnomad_panel,
                             gnomad_constraint=gnomad_constraint,
                             alphafold_panel=alphafold_panel,
                             interpro_intervals=interpro_intervals,
                             functional_sites=functional_sites)
    stats.master_rows = len(master)
    vc = master["label"].value_counts(dropna=False)
    stats.master_label_counts = {("nan" if pd.isna(k) else str(int(k))):
                                 int(v) for k, v in vc.items()}
    validate_master_for_export(master)
    master_path = ext_dir / "extended_dataset.csv"
    master.to_csv(master_path, index=False)

    pgf_files = download_proteingym(raw_dir)   # cached paths for the manifest
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "genes": list(stats.genes_requested),
            "min_stars": min_stars,
            "include_alphamissense": include_alphamissense,
            "include_zeroshot": include_zeroshot,
            "include_gnomad": include_gnomad,
            "include_structure": include_structure,
            "include_interpro": include_interpro,
            "include_functional_sites": include_functional_sites,
        },
        "sources": {
            "clinvar": {
                "file": "raw/variant_summary.txt.gz",
                "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
                       "variant_summary.txt.gz",
                "sha256": sha256_of(raw_dir / "variant_summary.txt.gz"),
            },
            "proteingym_dms": {
                "version": PROTEINGYM_VERSION,
                "urls": {"harvard_mirror": PROTEINGYM_BASE_HARVARD,
                         "zenodo": PROTEINGYM_ZENODO_RECORD},
                "files": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size}
                          for p in [pgf_files.dms_zip, pgf_files.dms_reference]},
            },
            "proteingym_clinical_zeroshot": {
                "version": PROTEINGYM_VERSION,
                "files": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size}
                          for p in [pgf_files.clinical_zip, pgf_files.zeroshot_zip]},
            },
            "alphamissense": {
                "enabled": include_alphamissense,
                "version": ALPHAMISSENSE_VERSION,
                "url": ALPHAMISSENSE_AA_URL,
                "licence": "CC BY-NC-SA 4.0",
            },
            "uniprot_domains": {"rest_url": "https://rest.uniprot.org/uniprotkb/{acc}.json"},
            "gnomad": {
                "enabled": include_gnomad,
                "api_url": "https://gnomad.broadinstitute.org/api",
                "dataset": "gnomad_r4",
                "genes_fetched": stats.gnomad_genes_fetched,
                "genes_failed": stats.gnomad_genes_failed,
                "constraint_genes_fetched": stats.gnomad_constraint_genes_fetched,
                "constraint_genes_failed": stats.gnomad_constraint_genes_failed,
            },
            "alphafold": {
                "enabled": include_structure,
                "api_url": "https://alphafold.ebi.ac.uk/api",
                "licence": "CC-BY-4.0",
                "accessions_covered": stats.alphafold_accessions_covered,
                "residues_with_plddt": stats.alphafold_residues_with_plddt,
            },
            "interpro": {
                "enabled": include_interpro,
                "api_url": "https://www.ebi.ac.uk/interpro/api",
                "licence": "CC0",
                "intervals": stats.interpro_intervals,
            },
            "uniprot_point_features": {
                "enabled": include_functional_sites,
                "rest_url": "https://rest.uniprot.org/uniprotkb/{acc}.json",
                "rows": stats.functional_site_rows,
            },
        },
        "panel": {g: {"accession": a, "length":
                      int(panel.loc[panel["gene"] == g, "sequence"].str.len().iloc[0])}
                  for g, a in stats.genes_resolved.items()},
        "artefacts": {
            p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size}
            for p in sorted(ext_dir.glob("*.csv")) + sorted(ext_dir.glob("*.fasta"))
            if p.is_file()
        },
        "stats": asdict(stats),
    }
    (ext_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Extended dataset built: %d unique substitutions | labels: %s",
        stats.master_rows, stats.master_label_counts)
    logger.info("Artifacts -> %s", ext_dir)
    return manifest
