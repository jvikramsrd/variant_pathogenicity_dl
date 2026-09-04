#!/usr/bin/env python3
"""Re-stamp an existing build manifest against the files actually on disk.

Why this exists
---------------
``src/extended_builder.build_extended_dataset`` writes ``manifest.json`` right
after it writes ``extended_dataset.csv``. ``scripts/build_mmr_dataset.py`` then
joins gnomAD allele frequencies (stage 3) and gene constraint (stage 4) and
rewrites the CSV -- but until now it did not rewrite the manifest. Builds
produced before that fix therefore carry a manifest that describes the
*pre-join* table: ``include_gnomad: false``, ``gnomad_rows_panel: 0``, and an
``artefacts`` checksum for a file that no longer exists in that form.

This script repairs such a manifest in place without rebuilding the dataset. It
recomputes every artefact checksum from disk and, unless ``--checksums-only`` is
passed, corrects the gnomAD parameters and statistics by measuring the table
rather than trusting the recorded flags.

It never touches the dataset itself, and it never invents provenance it cannot
measure: source URLs, download checksums and per-source counts recorded at build
time are left exactly as they were.

Usage
-----
    python scripts/repair_manifest.py data/mmr/processed/extended
    python scripts/repair_manifest.py data/mmr/processed/extended --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extended_builder import refresh_manifest  # noqa: E402
from src.external_datasets import sha256_of  # noqa: E402

#: Columns whose presence and fill rate prove whether gnomAD was actually joined.
AF_COL = "gnomad_af_joint"
CONSTRAINT_COLS = ("gnomad_pli", "gnomad_oe_lof")


def measure_gnomad(csv_path: Path) -> dict:
    """Measure what the table actually contains, rather than trusting flags."""
    cols = pd.read_csv(csv_path, nrows=0).columns
    wanted = [c for c in (AF_COL, *CONSTRAINT_COLS, "gene") if c in cols]
    if AF_COL not in wanted:
        return {"include_gnomad": False, "gnomad_rows_joined": 0,
                "gnomad_constraint_rows": 0}
    df = pd.read_csv(csv_path, usecols=wanted, low_memory=False)
    joined = int(df[AF_COL].notna().sum())
    present = [c for c in CONSTRAINT_COLS if c in df.columns]
    constraint_rows = int(df[present].notna().all(axis=1).sum()) if present else 0
    genes = (sorted(df.loc[df[present].notna().all(axis=1), "gene"].astype(str).unique())
             if present and "gene" in df.columns else [])
    return {
        "include_gnomad": joined > 0,
        "include_gnomad_constraint": constraint_rows > 0,
        "gnomad_rows_joined": joined,
        "gnomad_constraint_rows": constraint_rows,
        "gnomad_constraint_genes_fetched": genes,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ext_dir", type=Path,
                   help="Directory holding manifest.json and the built artefacts.")
    p.add_argument("--checksums-only", action="store_true",
                   help="Only recompute artefact checksums; leave stats untouched.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change and write nothing.")
    args = p.parse_args(argv)

    manifest_path = args.ext_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 1
    before = json.loads(manifest_path.read_text())

    stale = []
    for name, rec in (before.get("artefacts") or {}).items():
        path = args.ext_dir / name
        if not path.exists():
            stale.append((name, "recorded but missing from disk"))
            continue
        actual = sha256_of(path)
        if actual != rec.get("sha256"):
            stale.append((name, f"{rec.get('sha256','?')[:12]}... -> {actual[:12]}..."))
    for name, why in stale:
        print(f"  stale artefact: {name}  ({why})")
    if not stale:
        print("  all recorded artefact checksums already match disk")

    updates: dict = {}
    if not args.checksums_only:
        csv_path = args.ext_dir / "extended_dataset.csv"
        if csv_path.exists():
            measured = measure_gnomad(csv_path)
            recorded = before.get("parameters", {}).get("include_gnomad")
            if recorded != measured["include_gnomad"]:
                print(f"  parameters.include_gnomad: {recorded} -> "
                      f"{measured['include_gnomad']} (measured from the table)")
            updates = {
                "parameters": {k: v for k, v in measured.items()
                               if k.startswith("include_")},
                "stats": {k: v for k, v in measured.items()
                          if not k.startswith("include_")},
            }

    if args.dry_run:
        print("  --dry-run: nothing written")
        return 0
    refresh_manifest(args.ext_dir, updates=updates)
    print(f"  manifest repaired -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
