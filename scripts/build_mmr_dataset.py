#!/usr/bin/env python3
"""Build the dedicated MLH1/MSH2/MSH6/PMS2 dataset required for fine-tuning.

PMS2 pseudogene-homology filtering is fail-closed: provide a validated list of
variants in exons 11–15 keyed by canonical protein substitution, or explicitly
exclude PMS2. This avoids silently treating short-read calls in the homology
region as ground truth.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.extended_builder import build_extended_dataset  # noqa: E402

MMR = ("MLH1", "MSH2", "MSH6", "PMS2")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data/mmr")
    p.add_argument("--pms2_homology_csv", type=Path, default=None,
                   help="CSV of PMS2 exon 11–15 substitutions with gene, position, wt_aa, mut_aa, orthogonally_confirmed.")
    p.add_argument("--exclude_pms2", action="store_true",
                   help="Build a three-gene clinical fine-tuning set until a PMS2 confirmation list is available.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not args.exclude_pms2 and args.pms2_homology_csv is None:
        p.error("PMS2 requires --pms2_homology_csv or --exclude_pms2; refusing unsafe labels.")

    genes = [g for g in MMR if not (g == "PMS2" and args.exclude_pms2)]
    build_extended_dataset(genes=genes, data_dir=args.data_dir, min_stars=2,
                           include_alphamissense=True, include_zeroshot=True)
    master_path = args.data_dir / "processed/extended/extended_dataset.csv"
    master = pd.read_csv(master_path, low_memory=False)
    if args.pms2_homology_csv is not None:
        confirmed = pd.read_csv(args.pms2_homology_csv,
                                usecols=["gene", "position", "wt_aa", "mut_aa", "orthogonally_confirmed"])
        key = ["gene", "position", "wt_aa", "mut_aa"]
        confirmed["_confirmed"] = (confirmed["orthogonally_confirmed"].astype(str)
                                    .str.strip().str.lower()
                                    .isin({"1", "true", "yes", "y"}).astype(int))
        master = master.merge(confirmed.drop_duplicates(key), on=key, how="left")
        unsafe = (master["gene"] == "PMS2") & (master["_confirmed"] == 0)
        # Retain unconfirmed PMS2 rows for provenance/inference, but never
        # permit them to become training targets.
        master.loc[unsafe, "label"] = pd.NA
        master.loc[unsafe, "label_weight"] = 0.0
        master.loc[unsafe, "pms2_homology_excluded"] = 1
        master = master.drop(columns=["_confirmed", "orthogonally_confirmed"])
    else:
        master["pms2_homology_excluded"] = 0
    master.to_csv(master_path, index=False)
    logging.info("MMR master written: %s", master_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
