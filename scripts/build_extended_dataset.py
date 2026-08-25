#!/usr/bin/env python3
"""CLI for building the unified extended variant-pathogenicity dataset.

Examples
--------
    # default 10-gene panel, all sources
    python scripts/build_extended_dataset.py

    # custom panel, skip the heavy AlphaMissense scan
    python scripts/build_extended_dataset.py --genes TP53,BRCA1 --no-alphamissense

    # force re-download of every remote artefact
    python scripts/build_extended_dataset.py --overwrite_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extended_builder import (  # noqa: E402
    DEFAULT_GENE_PANEL,
    build_extended_dataset,
)
from src.gnomad import DEFAULT_BA1_AF, DEFAULT_BS1_AF, DEFAULT_PM2_AF  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the extended multi-source variant pathogenicity "
                    "dataset (ClinVar + ProteinGym + AlphaMissense + UniProt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--genes", type=str,
                   default=",".join(DEFAULT_GENE_PANEL),
                   help="Comma-separated HGNC gene symbols.")
    p.add_argument("--panel_file", type=Path, default=None,
                   help="JSON file {gene: {accession, sequence}} overriding "
                        "--genes (e.g. the expanded DMS-wide panel from "
                        "scripts/make_expanded_panel.py).")
    p.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data",
                   help="Project data directory (expects raw/ and processed/).")
    p.add_argument("--min_stars", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Minimum ClinVar review stars for labelled variants; 2 is the robust default.")
    p.add_argument("--no-alphamissense", action="store_true",
                   help="Skip the AlphaMissense streaming filter (~1.2 GB scan).")
    p.add_argument("--no-zeroshot", action="store_true",
                   help="Skip ProteinGym zero-shot model-score enrichment.")
    p.add_argument("--include_gnomad", action="store_true",
                   help="Fetch gnomAD v4 allele frequencies for every panel gene "
                        "(one GraphQL call per gene; opt-in, can take minutes for "
                        "a large panel) and attach them as explicit input features "
                        "(gnomad_log10_af, acmg_ba1/bs1/pm2), not just for the 4 "
                        "MMR genes -- PROJECT_PLAN.md Phase 3's 'genome data used "
                        "for pretraining' should carry the same features the MMR "
                        "fine-tuning stage sees.")
    p.add_argument("--gnomad_ba1_af", type=float, default=DEFAULT_BA1_AF,
                   help="ACMG BA1 (stand-alone benign) allele-frequency threshold.")
    p.add_argument("--gnomad_bs1_af", type=float, default=DEFAULT_BS1_AF,
                   help="ACMG BS1 (moderate benign) allele-frequency threshold.")
    p.add_argument("--gnomad_pm2_af", type=float, default=DEFAULT_PM2_AF,
                   help="ACMG PM2 (absent/rare) allele-frequency threshold.")
    p.add_argument("--include_structure", action="store_true",
                   help="Fetch AlphaFold DB per-residue pLDDT structural-confidence "
                        "feature for every panel accession (one REST call + one PDB "
                        "download per accession; opt-in).")
    p.add_argument("--include_interpro", action="store_true",
                   help="Fetch InterPro domain/family/superfamily calls (one REST "
                        "call per accession; complements UniProt's own domain "
                        "annotations; opt-in).")
    p.add_argument("--include_functional_sites", action="store_true",
                   help="Attach UniProt point features -- active/binding site, PTM, "
                        "disulfide bond (reuses the UniProt JSON already cached for "
                        "domains, no extra network calls; opt-in).")
    p.add_argument("--all_sources", action="store_true",
                   help="Shortcut for --include_gnomad --include_structure "
                        "--include_interpro --include_functional_sites.")
    p.add_argument("--overwrite_cache", action="store_true",
                   help="Re-download remote artefacts and re-resolve the panel.")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    panel_records = None
    if args.panel_file is not None:
        panel_records = json.loads(args.panel_file.read_text())
        genes = list(panel_records.keys())
        logging.info("Panel file %s: %d genes override --genes.",
                     args.panel_file, len(genes))
    manifest = build_extended_dataset(
        genes=genes,
        data_dir=args.data_dir,
        min_stars=args.min_stars,
        include_alphamissense=not args.no_alphamissense,
        include_zeroshot=not args.no_zeroshot,
        include_gnomad=args.include_gnomad or args.all_sources,
        gnomad_ba1_af=args.gnomad_ba1_af,
        gnomad_bs1_af=args.gnomad_bs1_af,
        gnomad_pm2_af=args.gnomad_pm2_af,
        include_structure=args.include_structure or args.all_sources,
        include_interpro=args.include_interpro or args.all_sources,
        include_functional_sites=args.include_functional_sites or args.all_sources,
        overwrite_cache=args.overwrite_cache,
        panel_records=panel_records,
    )
    print("\n==================== EXTENDED DATASET MANIFEST ====================")
    print(json.dumps(manifest["stats"], indent=2)[:2000])
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
