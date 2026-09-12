#!/usr/bin/env python3
"""Score a CSV of variants against a saved stage-2 ``arch="priors"`` checkpoint.

Not executed in this environment. This is a portable CLI over
``src/inference.py`` -- run it on the machine that has the trained checkpoint,
pandas, and PyTorch installed. See ``docs/INFERENCE.md`` for exact commands
and the input/output schema this expects.

Example
-------
    python scripts/predict.py \\
        --checkpoint data/mmr/processed/transfer/priors_MLH1_holdout_head.pt \\
        --input_csv my_variants.csv \\
        --output_csv predictions.csv \\
        --device cpu

``my_variants.csv`` must contain the identifying columns ``uniprot_id,
position, wt_aa, mut_aa`` plus every feature column the checkpoint's schema
names (see the error message from a failed run, or ``--print_schema``, for
the exact list) -- column order does not matter, extra columns are ignored.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gene_aliases import UnknownGeneError  # noqa: E402
from src.inference import (  # noqa: E402
    InferenceInputError,
    UnsupportedArchitectureError,
    load_inference_model,
    predict,
)
from src.transfer import TransferHeadFormatError  # noqa: E402

logger = logging.getLogger("predict")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                  help="Path to a stage-2 transfer-head checkpoint "
                       "(src.transfer.save_transfer_head output, arch=priors).")
    p.add_argument("--input_csv", type=Path, required=True,
                  help="CSV with uniprot_id, position, wt_aa, mut_aa plus "
                       "the checkpoint's declared feature columns.")
    p.add_argument("--output_csv", type=Path, required=True,
                  help="Where to write predictions. Parent dir must exist "
                       "(this CLI does not create arbitrary output trees).")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                  help="'auto' picks cuda if available, else cpu.")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=None,
                  help="Override the checkpoint's stored MCC-optimal "
                       "threshold. Default: use the checkpoint's own value, "
                       "falling back to 0.5 if it has none.")
    p.add_argument("--known_genes", type=str, default=None,
                  help="Comma-separated HGNC symbols/accessions to restrict "
                       "scoring to (e.g. 'MLH1,MSH2'). Default: accept any "
                       "gene resolvable via src.gene_aliases.")
    p.add_argument("--print_schema", action="store_true",
                  help="Print the checkpoint's required input schema and "
                       "exit without scoring anything.")
    p.add_argument("--dry_run", action="store_true",
                  help="Validate the checkpoint and input schema, print the "
                       "row count that would be scored, and exit without "
                       "running the model or writing output.")
    return p.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is visible "
                             "to this PyTorch install.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    device = resolve_device(args.device)
    try:
        loaded = load_inference_model(args.checkpoint, device=device)
    except (TransferHeadFormatError, UnsupportedArchitectureError) as exc:
        logger.error("%s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("Checkpoint not found: %s", exc)
        return 1

    if args.print_schema:
        print(json.dumps({
            "checkpoint": str(args.checkpoint),
            "arch": loaded.config.get("arch"),
            "device": str(loaded.device),
            "required_identifying_columns": ["uniprot_id", "position", "wt_aa", "mut_aa"],
            "required_feature_columns": list(loaded.feature_columns),
            "threshold": loaded.threshold,
        }, indent=2))
        return 0

    if not args.input_csv.exists():
        logger.error("Input CSV not found: %s", args.input_csv)
        return 1
    df = pd.read_csv(args.input_csv)

    known_genes = (args.known_genes.split(",") if args.known_genes else None)

    if args.dry_run:
        try:
            from src.inference import align_genes, validate_input_frame
            validated = validate_input_frame(df, loaded)
            align_genes(validated, known_genes=known_genes)
        except (InferenceInputError, UnknownGeneError) as exc:
            logger.error("Input would fail validation: %s", exc)
            return 1
        print(json.dumps({
            "dry_run": True,
            "checkpoint": str(args.checkpoint),
            "input_csv": str(args.input_csv),
            "rows_that_would_be_scored": int(len(df)),
            "device": str(loaded.device),
        }, indent=2))
        return 0

    try:
        result = predict(loaded, df, known_genes=known_genes,
                         batch_size=args.batch_size, threshold=args.threshold)
    except (InferenceInputError, UnknownGeneError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.output_csv.parent.exists():
        logger.error("Output directory does not exist: %s", args.output_csv.parent)
        return 1
    result.predictions.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d prediction(s) -> %s (checkpoint=%s, threshold=%.4f)",
               len(result.predictions), args.output_csv, result.checkpoint_path,
               result.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
