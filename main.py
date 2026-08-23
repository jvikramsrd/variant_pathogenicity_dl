#!/usr/bin/env python3
"""End-to-end CLI runner for the variant pathogenicity pipeline.

Stages
------
1. Resolve gene(s) -> UniProt canonical sequences (cached CSVs for ClinVar).
2. Extract ESM-2 features + PLLR zero-shot scores (disk-cached npz), one
   protein at a time; multi-gene runs concatenate per-protein feature blocks.
3. Optionally append external prior features from the *extended dataset*
   (AlphaMissense, published-model zero-shot scores such as EVE/ESM1b/GEMME,
   ProteinGym DMS aggregates, UniProt domain flags).
4. K-fold leakage-grouped cross-validation of the residual MLP head with
   focal / weighted-BCE losses, AdamW + cosine schedule and early stopping.
   Groups are ``(protein, position)`` pairs so multi-gene pooling never leaks.
5. Calibration benchmarking (temperature scaling, isotonic regression) against
   the raw model and the PLLR baseline; reliability diagrams.
6. Prospective inference on held-out ClinVar VUS variants.

Examples
--------
    python main.py --gene TP53 --esm_model facebook/esm2_t33_650M_UR50D
    python main.py --gene PTEN --debug                       # quick smoke run
    python main.py --gene TP53,BRCA1,PTEN --extra_features all
    python scripts/build_extended_dataset.py                 # build priors first
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable no matter where the script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    build_multi_gene_dataset,
    fetch_uniprot_accession,
    fetch_uniprot_sequence,
    make_session,
)
from src.esm_extractor import extract_features_cached, get_device  # noqa: E402
from src.train import TrainConfig, run_pipeline, set_global_seed  # noqa: E402

logger = logging.getLogger("main")

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
DEBUG_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"

EXTENDED_DIRNAME = "extended"

#: External prior columns selectable through ``--extra_features``.
EXTRA_FEATURE_SETS: dict[str, list[str]] = {
    "zeroshot": ["zs_*"],
    "alphamissense": ["am_pathogenicity"],
    "dms": ["dms_score_median", "dms_bin_median", "n_dms_assays"],
    "structure": ["in_domain"],
}
EXTRA_FEATURE_SETS["all"] = sorted(
    {pat for pats in EXTRA_FEATURE_SETS.values() for pat in pats}
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gene-specific variant pathogenicity pipeline "
                    "(ClinVar -> ESM-2 -> MLP -> calibrated evaluation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    core = parser.add_argument_group("core")
    core.add_argument("--gene", type=str, required=True,
                      help="HGNC gene symbol(s); comma-separated list pools "
                           "proteins into one training run, e.g. TP53,BRCA1,PTEN.")
    core.add_argument("--uniprot", type=str, default=None,
                      help="Explicit UniProt accession; skips symbol lookup "
                           "(single-gene runs only).")
    core.add_argument("--esm_model", type=str, default=DEFAULT_ESM_MODEL,
                      help=f"HuggingFace ESM-2 checkpoint. Use '{DEBUG_ESM_MODEL}' for quick runs.")
    core.add_argument("--batch_size", type=int, default=32,
                      help="Mini-batch size (feature extraction and training).")
    core.add_argument("--epochs", type=int, default=30, help="Max training epochs per fold.")
    core.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    core.add_argument("--k_folds", type=int, default=5, help="Cross-validation folds.")

    model = parser.add_argument_group("model / loss")
    model.add_argument("--hidden_dim", type=int, default=512, help="MLP hidden width.")
    model.add_argument("--dropout", type=float, default=0.3, help="MLP dropout probability.")
    model.add_argument("--loss_type", choices=["focal", "wbce"], default="focal",
                       help="Training criterion.")
    model.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma.")
    model.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha.")
    model.add_argument("--patience", type=int, default=6,
                       help="Early-stopping patience on val ROC-AUC.")
    model.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    model.add_argument("--seed", type=int, default=42, help="Global random seed.")
    model.add_argument("--min_stars", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Minimum ClinVar review stars for labelled variants.")

    ext = parser.add_argument_group("external priors (extended dataset)")
    ext.add_argument("--extra_features", type=str, default="none",
                     choices=["none"] + list(EXTRA_FEATURE_SETS),
                     help="Append prior features from data/processed/extended/"
                          "extended_dataset.csv to the ESM embeddings. Requires "
                          "'python scripts/build_extended_dataset.py' first.")

    io = parser.add_argument_group("io")
    io.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data",
                    help="Project data directory (expects raw/ and processed/).")
    io.add_argument("--overwrite_cache", action="store_true",
                    help="Re-download ClinVar data and recompute ESM-2 features.")
    io.add_argument("--debug", action="store_true",
                    help="Quick-debug mode: 2 epochs, patience 2, verbose logging.")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # Keep third-party libraries quiet even in debug mode (HTTP traces drown
    # out everything otherwise); only project loggers honour --debug.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("main", "src", "__main__"):
        logging.getLogger(name).setLevel(level)


def apply_debug_overrides(args: argparse.Namespace) -> None:
    """Trim the run budget when --debug is passed."""
    args.epochs = min(args.epochs, 2)
    args.patience = min(args.patience, 2)
    if args.esm_model == DEFAULT_ESM_MODEL:
        logger.info("Debug mode: consider --esm_model %s for a much faster run.", DEBUG_ESM_MODEL)


def load_extra_features(
    meta: pd.DataFrame,
    data_dir: Path,
    choice: str,
) -> pd.DataFrame:
    """Join selected external prior columns onto *meta* (label-independent).

    Missing values are filled with each column's median over the joined frame
    and any still-missing value with 0.0. These priors are produced by external
    models/experiments, never from our own labels, so filling before splitting
    does not leak supervision information.
    """
    ext_csv = data_dir / "processed" / EXTENDED_DIRNAME / "extended_dataset.csv"
    if not ext_csv.exists():
        raise FileNotFoundError(
            f"{ext_csv} not found - build it first with "
            "`python scripts/build_extended_dataset.py`.")
    ext = pd.read_csv(ext_csv, low_memory=False)

    patterns = EXTRA_FEATURE_SETS[choice]
    prefix_cols = sorted({c for pat in patterns if pat.endswith("*")
                          for c in ext.columns if c.startswith(pat[:-1])})
    exact_cols = [pat for pat in patterns
                  if not pat.endswith("*") and pat in ext.columns]
    cols = prefix_cols + [c for c in exact_cols if c not in prefix_cols]
    if not cols:
        raise ValueError(f"No matching prior columns for --extra_features {choice}.")
    logger.info("External priors (%s): %d columns %s", choice, len(cols), cols)

    key = ["gene", "position", "wt_aa", "mut_aa"]
    merged = meta.merge(ext[key + cols].drop_duplicates(subset=key),
                        on=key, how="left", validate="m:1")
    for c in cols:
        col = merged[c]
        if pd.api.types.is_numeric_dtype(col):
            merged[c] = col.fillna(col.median()).fillna(0.0)
        else:
            merged[c] = 0.0
    return merged[cols].astype(np.float32)


def main() -> int:
    args = parse_args()
    configure_logging(args.debug)
    t_start = time.time()

    if args.debug:
        apply_debug_overrides(args)

    set_global_seed(args.seed)
    device = get_device()
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        dropout=args.dropout, loss_type=args.loss_type,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        patience=args.patience, k_folds=args.k_folds, seed=args.seed,
    )
    logger.info("TrainConfig: %s", cfg.to_dict())

    raw_dir = args.data_dir / "raw"
    processed_dir = args.data_dir / "processed"

    genes = [g.strip().upper() for g in args.gene.split(",") if g.strip()]
    if not genes:
        raise SystemExit("--gene must contain at least one symbol.")
    if len(genes) > 1 and args.uniprot:
        raise SystemExit("--uniprot applies only to single-gene runs.")

    # ------------------------------------------------------------------ #
    # Stage 1 — sequences + ClinVar datasets (one pass for all genes)
    # ------------------------------------------------------------------ #
    session = make_session()
    sequences: dict[str, str] = {}
    accessions: dict[str, str] = {}
    if len(genes) == 1 and args.uniprot:
        accessions[genes[0]] = args.uniprot
        sequences[genes[0]] = fetch_uniprot_sequence(args.uniprot, session=session)
    else:
        for gene in genes:
            accessions[gene] = fetch_uniprot_accession(gene, session=session)
            sequences[gene] = fetch_uniprot_sequence(accessions[gene], session=session)

    labeled_meta_all, vus_meta_all = build_multi_gene_dataset(
        genes, raw_dir, processed_dir, min_stars=args.min_stars,
        overwrite=args.overwrite_cache,
    )

    # ------------------------------------------------------------------ #
    # Stage 2 — validate against each sequence, then ESM-2 features per protein
    # ------------------------------------------------------------------ #
    from src.esm_extractor import validate_and_align  # local import keeps stages clear

    feature_blocks: list[np.ndarray] = []
    meta_frames: list[pd.DataFrame] = []
    vus_feature_blocks: list[np.ndarray] = []
    vus_meta_frames: list[pd.DataFrame] = []

    extras_cache: dict[str, pd.DataFrame] = {}

    for gene in [g for g in genes if g in set(labeled_meta_all["gene"]) | set(vus_meta_all["gene"])]:
        sequence = sequences[gene]
        lab = validate_and_align(
            labeled_meta_all[labeled_meta_all["gene"] == gene].copy(), sequence)
        vus = validate_and_align(
            vus_meta_all[vus_meta_all["gene"] == gene].copy(), sequence)
        full_meta = pd.concat([lab, vus], ignore_index=True)
        logger.info("[%s] extracting ESM-2 features for %d variants ...",
                    gene, len(full_meta))

        extra_df = None
        if args.extra_features != "none":
            if gene not in extras_cache:
                extras_cache[gene] = load_extra_features(full_meta, args.data_dir,
                                                         args.extra_features)
            extra_df = extras_cache[gene]

        feats, meta_out = extract_features_cached(
            full_meta, sequence, gene=gene, model_name=args.esm_model,
            processed_dir=processed_dir, batch_size=args.batch_size,
            device=device, overwrite=args.overwrite_cache,
            extra_features=extra_df,
        )

        is_labeled = meta_out["label"].notna().to_numpy()
        feature_blocks.append(feats[is_labeled])
        meta_frames.append(meta_out.loc[is_labeled].reset_index(drop=True))
        if (~is_labeled).any():
            vus_feature_blocks.append(feats[~is_labeled])
            vus_meta_frames.append(meta_out.loc[~is_labeled].reset_index(drop=True))
        n_pos = int((meta_out.loc[is_labeled, "label"] == 1).sum())
        n_neg = int((meta_out.loc[is_labeled, "label"] == 0).sum())
        logger.info("[%s] labelled after alignment: %d variants (%d P/LP, %d B/LB)",
                    gene, int(is_labeled.sum()), n_pos, n_neg)

    labeled_features = np.vstack(feature_blocks) if feature_blocks else np.zeros((0, 1), dtype=np.float32)
    labeled_meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    vus_features = (np.vstack(vus_feature_blocks) if vus_feature_blocks
                    and len({f.shape[1] for f in vus_feature_blocks}) == 1 else None)
    vus_meta_out = pd.concat(vus_meta_frames, ignore_index=True) if vus_meta_frames else None

    n_pos = int((labeled_meta["label"] == 1).sum())
    n_neg = int((labeled_meta["label"] == 0).sum())
    logger.info("Pooled labelled set: %d variants (%d P/LP, %d B/LB) across %d proteins",
                len(labeled_meta), n_pos, n_neg, labeled_meta["gene"].nunique())

    # ------------------------------------------------------------------ #
    # Stage 3-5 — train, benchmark calibration, infer VUS
    # ------------------------------------------------------------------ #
    paths = run_pipeline(
        labeled_features=labeled_features,
        labeled_meta=labeled_meta,
        vus_features=vus_features,
        vus_meta=vus_meta_out,
        gene="_".join(genes),
        cfg=cfg,
        device=device,
        out_dir=processed_dir,
    )

    print("\n=========================== ARTIFACTS ================================")
    for key, path in sorted(paths.items()):
        print(f"  {key:<22} {path}")
    print(f"  {'total_runtime':<22} {time.time() - t_start:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
