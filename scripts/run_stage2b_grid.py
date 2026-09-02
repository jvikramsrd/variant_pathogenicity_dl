#!/usr/bin/env python3
"""Run the Stage-2b ablation grid, resumably, one cell at a time.

Answers the question ``docs/PAPER_DRAFT.md`` §6.12 poses but cannot currently
settle: that section varies freeze depth alone and compares the result to a
frozen *priors* probe, while Stage 2b reads no prior features -- so the gap it
measures confounds freeze depth with feature set. The ``branch`` axis here
holds the feature set constant.

Tiers are ordered by scientific value, so an interrupted weekend degrades
gracefully; tier 1 alone settles the headline. Completed cells are skipped on
restart, which matters because a multi-day run on a shared desktop will be
interrupted.

Example
-------
    python scripts/run_stage2b_grid.py --tiers 1 2 3 4 5 \
        --esm_model facebook/esm2_t33_650M_UR50D --mode siamese --eval lopo \
        --batch_size 1 --grad_accum 8 --gradient_checkpointing
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finetune_grid import TIERS, GridCell, cells_for, output_tag  # noqa: E402

logger = logging.getLogger("run_stage2b_grid")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=["1"], choices=sorted(TIERS))
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/stage2b_grid")
    p.add_argument("--mode", choices=("wt_site", "siamese"), default="siamese")
    p.add_argument("--eval", dest="eval_mode", choices=("lopo", "holdout"),
                   default="lopo")
    p.add_argument("--holdout_gene", type=str, default=None,
                   help="Required with --eval holdout.")
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mmr_csv", type=Path, default=None)
    p.add_argument("--panel_json", type=Path, default=None)
    p.add_argument("--af_labels_active", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the plan and the per-cell command lines, run nothing.")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells that already have complete artefacts.")
    return p.parse_args(argv)


def cell_tag(cell: GridCell, mode: str, eval_mode: str,
             holdout_gene: str | None = None) -> str:
    return output_tag(mode, eval_mode, cell.slug(), holdout_gene)


def cell_is_complete(out_dir, cell: GridCell, mode: str, eval_mode: str,
                     holdout_gene: str | None = None) -> bool:
    """True only when every artefact of this cell exists.

    All three are required: a run that died between writing the summary and
    writing the predictions must re-run, not be silently skipped.
    """
    tag = cell_tag(cell, mode, eval_mode, holdout_gene)
    return all((Path(out_dir) / f"esm_finetune_{kind}_{tag}.{ext}").exists()
               for kind, ext in (("summary", "json"), ("results", "csv"),
                                 ("predictions", "csv")))


def build_cell_argv(args: argparse.Namespace, cell: GridCell) -> list:
    """The `finetune_esm_mmr.py` command line that runs *cell*."""
    argv = [
        sys.executable, str(ROOT / "scripts" / "finetune_esm_mmr.py"),
        "--mode", args.mode,
        "--eval", args.eval_mode,
        "--esm_model", args.esm_model,
        "--branch", cell.branch,
        "--fusion", cell.fusion,
        "--n_unfrozen_layers", str(cell.n_unfrozen_layers),
        "--seed", str(cell.seed),
        "--cell_slug", cell.slug(),
        "--out_dir", str(args.out_dir),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--n_bootstrap", str(args.n_bootstrap),
        "--no-save_checkpoints",
    ]
    if cell.pllr_mode == "off":
        argv.append("--no-use_pllr")
    else:
        argv += ["--pllr_mode", cell.pllr_mode]
    if args.eval_mode == "holdout" and args.holdout_gene:
        argv += ["--holdout_gene", args.holdout_gene]
    if args.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if args.af_labels_active:
        argv.append("--af_labels_active")
    if args.mmr_csv:
        argv += ["--mmr_csv", str(args.mmr_csv)]
    if args.panel_json:
        argv += ["--panel_json", str(args.panel_json)]
    return argv


def aggregate(out_dir) -> pd.DataFrame:
    """Every cell's results CSV, concatenated."""
    frames = [pd.read_csv(p) for p in sorted(Path(out_dir).glob(
        "esm_finetune_results_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    if args.eval_mode == "holdout" and not args.holdout_gene:
        raise SystemExit("--holdout_gene is required with --eval holdout.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = cells_for(args.tiers)

    pending = [c for c in cells
               if args.force or not cell_is_complete(args.out_dir, c, args.mode,
                                                     args.eval_mode,
                                                     args.holdout_gene)]
    logger.info("Grid: %d cells in tiers %s | %d already complete | %d to run",
                len(cells), ",".join(args.tiers), len(cells) - len(pending),
                len(pending))
    for c in pending:
        logger.info("  [tier %s] %s", c.tier, c.slug())
    if args.dry_run:
        for c in pending:
            print(" ".join(build_cell_argv(args, c)))
        return 0

    t0 = time.time()
    completed, failed = [], []
    for i, cell in enumerate(pending, start=1):
        logger.info("=== cell %d/%d: %s (tier %s) ===", i, len(pending),
                    cell.slug(), cell.tier)
        cell_t0 = time.time()
        result = subprocess.run(build_cell_argv(args, cell), cwd=str(ROOT))
        elapsed = (time.time() - cell_t0) / 60
        if result.returncode == 0:
            completed.append(cell.slug())
            logger.info("cell done in %.1f min | %.1f h elapsed of grid",
                        elapsed, (time.time() - t0) / 3600)
        else:
            # Keep going: one cell's OOM should not cost the remaining tiers.
            failed.append(cell.slug())
            logger.error("cell FAILED (exit %d) after %.1f min -- continuing",
                         result.returncode, elapsed)

    combined = aggregate(args.out_dir)
    combined_path = args.out_dir / "stage2b_grid_results.csv"
    combined.to_csv(combined_path, index=False)
    (args.out_dir / "stage2b_grid_manifest.json").write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tiers": args.tiers,
        "cells": [c.to_dict() for c in cells],
        "completed": completed, "failed": failed,
        "esm_model": args.esm_model, "mode": args.mode, "eval": args.eval_mode,
        "runtime_h": round((time.time() - t0) / 3600, 2),
    }, indent=2))

    print(f"\nGrid complete: {len(completed)} ok, {len(failed)} failed, "
          f"{(time.time() - t0) / 3600:.1f} h")
    if failed:
        print("FAILED cells (re-run the same command to retry them):")
        for slug in failed:
            print(f"  {slug}")
    print(f"Combined results -> {combined_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
