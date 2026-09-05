"""Annotate run summaries written before ``src/provenance.py`` existed.

The 16 grid cells and the four seed-42 ablations record hyperparameters but no
dataset, feature-schema or split identity. The table that produced them is still
on disk and hashable, and their per-variant predictions are committed, so most
of the identity block can be recovered exactly rather than assumed.

What is recovered exactly:

* ``dataset_sha256``   -- hashed from the table passed as ``--mmr_csv``
* ``split_definition`` -- read from the run's own predictions CSV
* ``git.commit``       -- from ``--git_commit`` (e.g. ``git_commit.txt``)

What is *reconstructed*, and flagged as such:

* ``feature_schema``   -- from the summary's ``prior_columns`` when present;
  otherwise re-derived from the table for the recorded branch. A reconstruction
  is a claim about what the run read, not a record of it.

What cannot be recovered at all: the library versions in force at run time.
They are recorded as ``null`` rather than filled with today's, which would be a
fabrication.

Run this on the machine holding the build that produced the results.

    python scripts/backfill_provenance.py data/processed/stage2b_grid \\
        --mmr_csv data/mmr/processed/extended/extended_dataset.csv \\
        --git_commit "$(cat data/processed/stage2b_grid/git_commit.txt)" --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.provenance import (feature_schema_hash, library_versions,  # noqa: E402
                            sha256_file, split_definition_hash)
from src.transfer import drop_prior_groups, prior_columns_of  # noqa: E402


def reconstruct_feature_columns(summary: dict, master: pd.DataFrame) -> tuple[list[str], bool]:
    """Return (columns, exact). ``exact`` is False when re-derived from the table."""
    recorded = summary.get("prior_columns")
    branch = summary.get("branch", "esm+priors")
    if recorded is not None:
        priors, exact = list(recorded), True
    elif branch == "esm":
        priors, exact = [], True          # no priors is no priors; nothing to re-derive
    else:
        priors = prior_columns_of(master, drop_gene_constant=True)
        groups = summary.get("drop_prior_groups") or []
        if groups:
            priors = drop_prior_groups(priors, groups, allow_proxy_leak=True)
        priors, exact = list(priors), False
    representation = [
        f"esm::{summary.get('esm_model')}" if branch != "priors" else "esm::none",
        f"pllr::{summary.get('pllr_mode', 'off')}",
        f"fusion::{summary.get('fusion', 'concat')}",
    ]
    return priors + representation, exact


def split_assignments_from(predictions_csv: Path) -> dict[str, list[str]]:
    d = pd.read_csv(predictions_csv)
    keys = (d.uniprot_id.astype(str) + ":" + d.position.astype(str) + ":"
            + d.wt_aa.astype(str) + ">" + d.mut_aa.astype(str))
    return {g: sub.tolist() for g, sub in keys.groupby(d.holdout_gene)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_dir", type=Path)
    ap.add_argument("--mmr_csv", type=Path,
                    default=Path("data/mmr/processed/extended/extended_dataset.csv"))
    ap.add_argument("--git_commit", type=str, default=None,
                    help="Code state the runs were produced at.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate summaries that already carry provenance.")
    args = ap.parse_args()

    if not args.mmr_csv.exists():
        raise SystemExit(f"{args.mmr_csv} not found -- run this on the build machine.")
    print(f"hashing {args.mmr_csv} ...", flush=True)
    dataset_sha = sha256_file(args.mmr_csv)
    print(f"  {dataset_sha}")
    master = pd.read_csv(args.mmr_csv, low_memory=False)

    summaries = sorted(args.summary_dir.glob("esm_finetune_summary_*.json"))
    if not summaries:
        raise SystemExit(f"no summaries in {args.summary_dir}")

    written = skipped = 0
    for path in summaries:
        summary = json.loads(path.read_text())
        if "provenance" in summary and not args.overwrite:
            skipped += 1
            continue
        preds = args.summary_dir / f"esm_finetune_predictions_{path.name[len('esm_finetune_summary_'):-len('.json')]}.csv"
        if not preds.exists():
            print(f"  SKIP {path.name}: no predictions CSV, split not recoverable")
            skipped += 1
            continue
        columns, exact = reconstruct_feature_columns(summary, master)
        summary["provenance"] = {
            "dataset_path": str(args.mmr_csv),
            "dataset_sha256": dataset_sha,
            "feature_schema": feature_schema_hash(columns),
            "n_features": len(columns),
            "split_definition": split_definition_hash(split_assignments_from(preds)),
            "git": {"commit": args.git_commit, "dirty": None},
            # Not recorded at run time and not inventable after the fact.
            "libraries": None,
            "backfilled": True,
            "feature_schema_exact": exact,
            "backfill_note": (
                "Added by scripts/backfill_provenance.py. Dataset hash and split "
                "definition are measured; the feature schema is "
                + ("read from the run's own prior_columns"
                   if exact else "RECONSTRUCTED from the table for the recorded branch")
                + "; library versions were not recorded and are null."
            ),
        }
        print(f"  {'would write' if args.dry_run else 'writing'} {path.name}"
              f"  schema={summary['provenance']['feature_schema']}"
              f" ({summary['provenance']['n_features']} cols, "
              f"{'exact' if exact else 'reconstructed'})"
              f"  split={summary['provenance']['split_definition']}")
        if not args.dry_run:
            path.write_text(json.dumps(summary, indent=2) + "\n")
        written += 1

    print(f"\n{written} summaries {'would be ' if args.dry_run else ''}annotated, "
          f"{skipped} skipped.")
    if not args.dry_run and written:
        print("Current environment (for reference only, NOT written into the "
              f"backfilled records): {library_versions()}")


if __name__ == "__main__":
    main()
