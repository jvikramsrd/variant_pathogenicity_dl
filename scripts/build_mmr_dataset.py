#!/usr/bin/env python3
"""Build the dedicated MLH1/MSH2/MSH6/PMS2 dataset (PROJECT_PLAN.md Phase 1).

Pipeline stages
---------------
1. Pin the four canonical UniProt references (P40692/P43246/P52701/P54278)
   and validate their sequence lengths before anything else runs.
2. Reuse the untouched multi-source builder (ClinVar + ProteinGym-clinical +
   AlphaMissense + zero-shot scores + UniProt domains) restricted to the four
   genes — never genome-wide dumps.
3. Join **gnomAD v4** allele frequencies per gene (GraphQL API, cached) as
   explicit input features plus BA1/BS1/PM2 ACMG frequency flags.
4. Apply the **InSiGHT/ClinGen VCEP expert-panel tier** metadata.
5. Enforce the **PMS2 exon 11–15 pseudogene gate** (hard filter, fail-closed):
   you must provide either a validated substitution list with an
   ``orthogonally_confirmed`` flag, an explicit protein codon range, or
   exclude PMS2 outright.
6. Emit **balanced-label diagnostic subsets** per gene (VariPred recipe) and
   the **leave-one-MMR-gene-out** split manifest.

Examples
--------
    python scripts/build_mmr_dataset.py --exclude_pms2
    python scripts/build_mmr_dataset.py \
        --pms2_homology_csv data/raw/pms2/pms2_ex11_15_confirmed.csv
    # Codon range for exons 11-15, derived (not guessed) from the Ensembl
    # exon table for the MANE Select transcript -- run
    # scripts/derive_pms2_homology_range.py to reproduce it, or read the
    # derivation in src.mmr_dataset.PMS2_PSEUDOGENE_CODON_RANGE. This keeps
    # PMS2's 381 pre-homology residues instead of dropping the gene.
    python scripts/build_mmr_dataset.py --pms2_codon_range 382 862
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extended_builder import build_extended_dataset  # noqa: E402
from src.gnomad import (  # noqa: E402
    GENE_CONSTRAINT_COLUMNS,
    GNOMAD_FEATURE_COLUMNS,
    attach_gene_constraint,
    fetch_mmr_genes,
    join_gnomad_features,
    load_or_fetch_constraint,
)
from src.mmr_dataset import (  # noqa: E402
    MMR_GENES,
    add_evidence_tiers,
    apply_pms2_homology_gate,
    attach_cimra,
    attach_mavedb,
    make_balanced_subset,
    resolve_mmr_panel,
    write_leave_one_gene_out_manifest,
)

logger = logging.getLogger("build_mmr")

#: Shared raw artefacts reused from the repository's main cache so the
#: dedicated MMR directory does not re-download multi-GB sources.
SHARED_RAW_ARTIFACTS = (
    "variant_summary.txt.gz",   # ClinVar
    "proteingym",               # DMS / clinical / zero-shot bundles
    "alphamissense",            # AlphaMissense aa-substitutions
    "uniprot",                  # entry-name map / panel caches
    "uniprot_domains",          # UniProt REST JSON caches
)


def ensure_raw_links(mmr_raw: Path, root_raw: Path) -> None:
    """Symlink shared raw artefacts into ``data/mmr/raw`` when missing."""
    mmr_raw.mkdir(parents=True, exist_ok=True)
    for name in SHARED_RAW_ARTIFACTS:
        src = root_raw / name
        dst = mmr_raw / name
        if dst.exists() or not src.exists():
            continue
        dst.symlink_to(src.resolve(),
                       target_is_directory=src.is_dir())
        logger.info("Linked shared artefact: %s -> %s", dst, src)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data/mmr",
                   help="Dedicated MMR data directory (kept separate from the "
                        "broad 80-gene dataset).")
    p.add_argument("--pms2_homology_csv", type=Path, default=None,
                   help="CSV of PMS2 exon 11-15 substitutions with columns "
                        "gene, position, wt_aa, mut_aa, orthogonally_confirmed.")
    p.add_argument("--pms2_codon_range", type=int, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Explicit inclusive protein-coordinate span of the "
                        "homology region. Use 382 862 for exons 11-15 -- "
                        "derived from the Ensembl exon table for MANE Select "
                        "ENST00000265849 by "
                        "scripts/derive_pms2_homology_range.py, which "
                        "self-validates against the 862 aa pinned for P54278. "
                        "Prefer this over --exclude_pms2: it keeps the 381 "
                        "residues N-terminal to the region (21 labelled "
                        "variants, 1,118 VUS) instead of dropping the gene.")
    p.add_argument("--exclude_pms2", action="store_true",
                   help="Build a three-gene set until a confirmation input "
                        "is available.")
    p.add_argument("--min_stars", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Minimum ClinVar review stars for labelled rows.")
    p.add_argument("--skip_gnomad", action="store_true",
                   help="Skip the gnomAD v4 frequency join (offline runs).")
    p.add_argument("--cimra_csv", type=Path, default=None,
                   help="Optional CIMRA OddsPath CSV (see src/cimra.py for the "
                        "expected schema; no bulk CIMRA API exists so this must "
                        "be extracted from paper supplementary tables). "
                        "Validation-only feature, never fed into ESM-branch "
                        "training (PROJECT_PLAN.md Phase 2).")
    p.add_argument("--skip_mavedb", action="store_true",
                   help="Skip the MaveDB deep-mutational-scan join (offline runs). "
                        "MSH2 gets the Jia et al. 2021 LOF screen, MLH1 the 2025 "
                        "abundance assay; validation-only, same non-circularity "
                        "contract as CIMRA.")
    p.add_argument("--mavedb_check_for_new", action="store_true",
                   help="Also live-search MaveDB for MSH6/PMS2 score sets that "
                        "may have been published since KNOWN_MMR_SCORESETS was "
                        "last updated (logs hits for manual review, does not "
                        "auto-ingest).")
    p.add_argument("--skip_gnomad_constraint", action="store_true",
                   help="Skip gnomAD gene-level constraint metrics (pLI, oe_mis, "
                        "mis_z, ...); cheap (one extra GraphQL call per gene) so "
                        "on by default.")
    p.add_argument("--skip_structure", action="store_true",
                   help="Skip AlphaFold DB per-residue pLDDT (one REST call + one "
                        "PDB download per gene; on by default -- only 4 genes here).")
    p.add_argument("--skip_interpro", action="store_true",
                   help="Skip InterPro domain/family/superfamily calls (complements "
                        "UniProt's own domain annotations; on by default).")
    p.add_argument("--skip_functional_sites", action="store_true",
                   help="Skip UniProt point features (active/binding site, PTM, "
                        "disulfide bond; reuses the already-cached UniProt JSON, "
                        "no extra network calls; on by default).")
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    # Reuse the repo's multi-GB raw downloads instead of re-fetching.
    ensure_raw_links(args.data_dir / "raw", ROOT / "data" / "raw")

    N = 12
    # --- Stage 1: pinned reference panel -------------------------------- #
    logger.info("[1/%d] Resolving pinned MMR panel (accession+length validated) ...", N)
    records = resolve_mmr_panel(overwrite=args.overwrite_cache)

    # --- Stages 2: multi-source master table ----------------------------- #
    logger.info("[2/%d] Building multi-source master table (%s) ...",
                N, ", ".join(MMR_GENES))
    build_extended_dataset(
        genes=list(MMR_GENES),
        data_dir=args.data_dir,
        min_stars=args.min_stars,
        include_alphamissense=True,
        include_zeroshot=True,
        include_structure=not args.skip_structure,
        include_interpro=not args.skip_interpro,
        include_functional_sites=not args.skip_functional_sites,
        overwrite_cache=args.overwrite_cache,
        panel_records={g: dict(d) for g, d in records.items()},
    )
    ext_dir = args.data_dir / "processed" / "extended"
    master_path = ext_dir / "extended_dataset.csv"
    master = pd.read_csv(master_path, low_memory=False)

    # --- Stage 3: gnomAD v4 allele frequencies --------------------------- #
    if args.skip_gnomad:
        logger.warning("[3/%d] Skipping gnomAD join (--skip_gnomad).", N)
        for c in GNOMAD_FEATURE_COLUMNS:
            master[c] = np.nan if "af" in c or "ac" in c or "an" in c else 0
        master["gnomad_log10_af"] = np.nan
    else:
        logger.info("[3/%d] Fetching/joining gnomAD v4 allele frequencies ...", N)
        seq_of = {g: d["sequence"] for g, d in records.items()}
        tables = fetch_mmr_genes(args.data_dir / "raw", seq_of,
                                 overwrite=args.overwrite_cache)
        gnomad_all = pd.concat(
            [t.assign(gene=g) for g, t in tables.items()], ignore_index=True)
        master = join_gnomad_features(master, gnomad_all)
        covered = int(master["gnomad_af_joint"].notna().sum())
        logger.info("gnomAD coverage: %d/%d master rows have a joint AF.",
                    covered, len(master))

    # --- Stage 4: gnomAD gene-level constraint ---------------------------- #
    if args.skip_gnomad_constraint:
        logger.warning("[4/%d] Skipping gnomAD constraint (--skip_gnomad_constraint).", N)
        for c in GENE_CONSTRAINT_COLUMNS:
            master[c] = np.nan
    else:
        logger.info("[4/%d] Fetching gnomAD gene-level constraint metrics ...", N)
        # Fetched one gene at a time so a failure names the gene it belongs to.
        # This stays fail-closed: unlike the 80-gene panel (where
        # ``extended_builder`` records per-gene failures in the manifest and
        # carries on), the MMR panel is four genes and every one of them is
        # load-bearing, so a silent NaN column here would misreport the build.
        # ``--skip_gnomad_constraint`` is the deliberate opt-out.
        constraint_by_gene: dict = {}
        failed: list = []
        for g in MMR_GENES:
            if g not in set(master["gene"].unique()):
                continue
            try:
                constraint_by_gene[g] = load_or_fetch_constraint(
                    g, args.data_dir / "raw", overwrite=args.overwrite_cache)
            except Exception as exc:  # noqa: BLE001 - reported together below
                logger.error("gnomAD constraint: %s failed (%s).", g, exc)
                failed.append(g)
        if failed:
            raise RuntimeError(
                f"gnomAD gene-level constraint unavailable for: {', '.join(failed)}. "
                "Every successful gene is cached under "
                f"{args.data_dir / 'raw' / 'gnomad'}, so re-running the identical "
                "command resumes and only retries what is missing. If gnomAD is "
                "down for longer than you can wait, re-run with "
                "--skip_gnomad_constraint to build without these five columns "
                "-- the manifest then records their absence."
            )
        master = attach_gene_constraint(master, constraint_by_gene)
        logger.info("gnomAD constraint: %s", constraint_by_gene)

    # --- Stage 5: InSiGHT/ClinGen VCEP tiers ------------------------------ #
    logger.info("[5/%d] Adding expert-panel evidence tiers ...", N)
    master = add_evidence_tiers(master)
    n_expert = int((master["expert_panel"] == 1).sum())
    logger.info("Expert-panel (VCEP) calls flagged: %d", n_expert)

    # --- Stage 6: PMS2 pseudogene gate (fail-closed) ---------------------- #
    logger.info("[6/%d] Applying PMS2 homology gate ...", N)
    master = apply_pms2_homology_gate(
        master,
        homology_csv=args.pms2_homology_csv,
        codon_range=tuple(args.pms2_codon_range) if args.pms2_codon_range else None,
        exclude_pms2=args.exclude_pms2,
    )

    # --- Stage 7: CIMRA OddsPath (Phase 2, validation-only) ---------------- #
    n_cimra = 0
    if args.cimra_csv is not None:
        logger.info("[7/%d] Joining CIMRA OddsPath evidence from %s ...", N, args.cimra_csv)
        master = attach_cimra(master, args.cimra_csv)
        n_cimra = int(master["cimra_oddspath"].notna().sum())
        logger.info("CIMRA coverage: %d/%d master rows.", n_cimra, len(master))
    else:
        logger.info("[7/%d] No --cimra_csv supplied; skipping CIMRA join.", N)

    # --- Stage 8: MaveDB DMS (Phase 2, validation-only) --------------------- #
    n_mave = 0
    if args.skip_mavedb:
        logger.warning("[8/%d] Skipping MaveDB join (--skip_mavedb).", N)
    else:
        logger.info("[8/%d] Joining MaveDB deep-mutational-scan scores ...", N)
        master = attach_mavedb(master, args.data_dir / "raw",
                               overwrite=args.overwrite_cache,
                               check_for_new=args.mavedb_check_for_new)
        n_mave = int(master["mave_score"].notna().sum())
        logger.info("MaveDB coverage: %d/%d master rows.", n_mave, len(master))

    # --- Stages 9-11: AlphaFold / InterPro / functional sites -------------- #
    # Already joined inside build_extended_dataset (Stage 2) when the
    # corresponding --skip_* flag above was not set; report coverage here.
    for stage_n, col, label in (
        (9, "af_plddt", "AlphaFold pLDDT"),
        (10, "in_interpro_domain", "InterPro domain"),
        (11, "is_functional_site", "UniProt functional-site"),
    ):
        if col in master.columns:
            covered = int(pd.to_numeric(master[col], errors="coerce").fillna(0).astype(bool).sum())
            logger.info("[%d/%d] %s coverage: %d/%d master rows.",
                        stage_n, N, label, covered, len(master))

    # --- Stage 12: diagnostics + splits ------------------------------------ #
    logger.info("[12/%d] Writing balanced subsets + LOPO manifest ...", N)
    processed_root = args.data_dir / "processed"
    labeled = master[master["label"].notna()].copy()
    balanced = make_balanced_subset(labeled, seed=42)
    balanced_path = processed_root / "mmr_balanced_diagnostic.csv"
    balanced.to_csv(balanced_path, index=False)
    per_gene_counts = balanced.groupby(["gene", "label"]).size().unstack(fill_value=0)
    logger.info("Balanced diagnostic subset: %d rows\n%s", len(balanced),
                per_gene_counts.to_string())

    active_genes = sorted(set(labeled["gene"]) | set(master["gene"]))
    write_leave_one_gene_out_manifest(processed_root, genes=active_genes)

    master.to_csv(master_path, index=False)
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "genes": list(active_genes),
        "rows": int(len(master)),
        "labelled": int(len(labeled)),
        "labelled_per_gene": {g: int(n) for g, n in
                              labeled["gene"].value_counts().items()},
        "vus": int((master["clinvar_label"].isna()
                    & master["review_status"].notna()).sum()),
        "expert_panel_calls": n_expert,
        "balanced_subset_rows": int(len(balanced)),
        "pms2_homology_excluded_labels": int(master.get(
            "pms2_homology_excluded", pd.Series(dtype=float)).fillna(0).sum()),
        "gnomad_features_joined": not args.skip_gnomad,
        "gnomad_rows_with_af": int(master["gnomad_af_joint"].notna().sum())
            if "gnomad_af_joint" in master.columns else 0,
        "cimra_csv": str(args.cimra_csv) if args.cimra_csv else None,
        "cimra_rows_with_oddspath": n_cimra,
        "mavedb_joined": not args.skip_mavedb,
        "mavedb_rows_with_score": n_mave,
        "gnomad_constraint_joined": not args.skip_gnomad_constraint,
        "alphafold_joined": not args.skip_structure,
        "interpro_joined": not args.skip_interpro,
        "functional_sites_joined": not args.skip_functional_sites,
    }
    (processed_root / "mmr_phase1_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("MMR Phase-1 master written: %s\nSummary: %s", master_path,
                json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
