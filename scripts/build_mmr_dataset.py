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
    python scripts/build_mmr_dataset.py --pms2_codon_range 419 862
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
from src.gnomad import GNOMAD_FEATURE_COLUMNS, fetch_mmr_genes, join_gnomad_features  # noqa: E402
from src.mmr_dataset import (  # noqa: E402
    MMR_GENES,
    add_evidence_tiers,
    apply_pms2_homology_gate,
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
                        "homology region (you verify this mapping yourself).")
    p.add_argument("--exclude_pms2", action="store_true",
                   help="Build a three-gene set until a confirmation input "
                        "is available.")
    p.add_argument("--min_stars", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Minimum ClinVar review stars for labelled rows.")
    p.add_argument("--skip_gnomad", action="store_true",
                   help="Skip the gnomAD v4 frequency join (offline runs).")
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    # Reuse the repo's multi-GB raw downloads instead of re-fetching.
    ensure_raw_links(args.data_dir / "raw", ROOT / "data" / "raw")

    # --- Stage 1: pinned reference panel -------------------------------- #
    logger.info("[1/6] Resolving pinned MMR panel (accession+length validated) ...")
    records = resolve_mmr_panel(overwrite=args.overwrite_cache)

    # --- Stages 2: multi-source master table ----------------------------- #
    logger.info("[2/6] Building multi-source master table (%s) ...",
                ", ".join(MMR_GENES))
    build_extended_dataset(
        genes=list(MMR_GENES),
        data_dir=args.data_dir,
        min_stars=args.min_stars,
        include_alphamissense=True,
        include_zeroshot=True,
        overwrite_cache=args.overwrite_cache,
        panel_records={g: dict(d) for g, d in records.items()},
    )
    ext_dir = args.data_dir / "processed" / "extended"
    master_path = ext_dir / "extended_dataset.csv"
    master = pd.read_csv(master_path, low_memory=False)

    # --- Stage 3: gnomAD v4 allele frequencies --------------------------- #
    if args.skip_gnomad:
        logger.warning("[3/6] Skipping gnomAD join (--skip_gnomad).")
        for c in GNOMAD_FEATURE_COLUMNS:
            master[c] = np.nan if "af" in c or "ac" in c or "an" in c else 0
        master["gnomad_log10_af"] = np.nan
    else:
        logger.info("[3/6] Fetching/joining gnomAD v4 allele frequencies ...")
        seq_of = {g: d["sequence"] for g, d in records.items()}
        tables = fetch_mmr_genes(args.data_dir / "raw", seq_of,
                                 overwrite=args.overwrite_cache)
        gnomad_all = pd.concat(
            [t.assign(gene=g) for g, t in tables.items()], ignore_index=True)
        master = join_gnomad_features(master, gnomad_all)
        covered = int(master["gnomad_af_joint"].notna().sum())
        logger.info("gnomAD coverage: %d/%d master rows have a joint AF.",
                    covered, len(master))

    # --- Stage 4: InSiGHT/ClinGen VCEP tiers ------------------------------ #
    logger.info("[4/6] Adding expert-panel evidence tiers ...")
    master = add_evidence_tiers(master)
    n_expert = int((master["expert_panel"] == 1).sum())
    logger.info("Expert-panel (VCEP) calls flagged: %d", n_expert)

    # --- Stage 5: PMS2 pseudogene gate (fail-closed) ---------------------- #
    logger.info("[5/6] Applying PMS2 homology gate ...")
    master = apply_pms2_homology_gate(
        master,
        homology_csv=args.pms2_homology_csv,
        codon_range=tuple(args.pms2_codon_range) if args.pms2_codon_range else None,
        exclude_pms2=args.exclude_pms2,
    )

    # --- Stage 6: diagnostics + splits ------------------------------------ #
    logger.info("[6/6] Writing balanced subsets + LOPO manifest ...")
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
    }
    (processed_root / "mmr_phase1_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("MMR Phase-1 master written: %s\nSummary: %s", master_path,
                json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
