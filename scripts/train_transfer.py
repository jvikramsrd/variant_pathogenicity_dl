#!/usr/bin/env python3
"""Two-stage transfer learning for the Lynch/MMR project.

Stage 1 pretrains on the broad 80-protein dataset. Stage 2 fine-tunes only on
clinical MMR labels. Functional-assay labels are deliberately excluded from
stage 2, preserving them for independent evaluation.

The script supports two scientifically distinct configurations:

* ``practical``: pretraining may include MMR proteins; use only for an adapted
  MMR model, never as an unseen-gene claim.
* ``leave_gene_out``: exclude every MMR gene from pretraining and exclude one
  requested MMR gene from fine-tuning. This is the transfer estimate for that
  unseen gene.

No command is run automatically; callers must explicitly invoke a stage.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import make_position_group_folds  # noqa: E402
from src.esm_extractor import get_device  # noqa: E402
from src.train import TrainConfig, _train_single_fold, predict_logits, set_global_seed  # noqa: E402

logger = logging.getLogger("train_transfer")
MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")
PRIOR_COLS = ("am_pathogenicity", "in_domain")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("stage", choices=("pretrain", "finetune"))
    p.add_argument("--train_csv", type=Path,
                   default=PROJECT_ROOT / "data/processed/extended/extended_dataset_train.csv")
    p.add_argument("--out_dir", type=Path,
                   default=PROJECT_ROOT / "data/processed/transfer")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Required for finetune; produced by pretrain.")
    p.add_argument("--mode", choices=("practical", "leave_gene_out"), default="leave_gene_out")
    p.add_argument("--holdout_gene", choices=MMR_GENES, default=None,
                   help="Required in leave_gene_out mode for a final held-out MMR gene.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--finetune_lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--clinical_weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Leakage-safe prior matrix shared by both transfer stages.

    DMS-derived columns are intentionally absent: their bins are supervision
    for most rows and would leak the target. Missingness flags are retained.
    """
    cols = [c for c in df.columns if c in PRIOR_COLS or c.startswith("zs_")]
    if not cols:
        raise ValueError("No non-DMS prior columns are available.")
    raw = df[cols].apply(pd.to_numeric, errors="coerce")
    missing = raw.isna().astype(np.float32)
    missing.columns = [f"is_missing_{c}" for c in cols]
    values = raw.fillna(raw.median()).fillna(0.0)
    return pd.concat([values, missing], axis=1).to_numpy(np.float32), cols + list(missing.columns)


def _clinical_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["clinvar_label"].notna() | df["clinical_label"].notna()).to_numpy()


def select_rows(df: pd.DataFrame, stage: str, mode: str, holdout_gene: str | None) -> pd.DataFrame:
    """Apply the plan's anti-circularity policy before any fitting occurs."""
    clinical = _clinical_mask(df)
    if stage == "pretrain":
        out = df.copy()
        if mode == "leave_gene_out":
            out = out[~out["gene"].isin(MMR_GENES)]
        return out.reset_index(drop=True)

    # Fine-tuning is clinical-only. DMS/CIMRA never become stage-2 labels.
    out = df[df["gene"].isin(MMR_GENES) & clinical].copy()
    if mode == "leave_gene_out":
        if holdout_gene is None:
            raise ValueError("--holdout_gene is required in leave_gene_out mode.")
        out = out[out["gene"] != holdout_gene]
    return out.reset_index(drop=True)


def fit_stage(df: pd.DataFrame, args: argparse.Namespace, init_state=None) -> Path:
    X, columns = feature_matrix(df)
    y = df["label"].astype(int).to_numpy()
    groups = (df["uniprot_id"].astype(str) + ":" + df["position"].astype(str)).to_numpy()
    # Inner validation uses residue-disjoint groups and is never later exported
    # as a performance estimate.
    train_idx, val_idx = make_position_group_folds(df["position"].to_numpy(), y,
                                                   k_folds=5, seed=args.seed,
                                                   groups=groups)[0]
    scaler = StandardScaler().fit(X[train_idx])
    quality_series = (df["label_weight"] if "label_weight" in df.columns
                      else pd.Series(1.0, index=df.index))
    source_weight = pd.to_numeric(quality_series, errors="coerce") \
        .fillna(1.0).to_numpy(np.float32)
    clinical_weight = np.where(_clinical_mask(df), args.clinical_weight, 1.0).astype(np.float32)
    weights = source_weight * clinical_weight
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience,
                      lr=args.finetune_lr if args.stage == "finetune" else args.lr,
                      batch_size=args.batch_size, hidden_dim=args.hidden_dim,
                      n_blocks=args.n_blocks, dropout=args.dropout,
                      loss_type="bce", seed=args.seed)
    model, best_epoch = _train_single_fold(
        scaler.transform(X[train_idx]).astype(np.float32), y[train_idx],
        scaler.transform(X[val_idx]).astype(np.float32), y[val_idx], cfg,
        get_device(), args.seed, sample_weights=weights[train_idx],
        monitor_mask=_clinical_mask(df)[val_idx], init_state=init_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.stage}_{args.mode}" + (f"_holdout-{args.holdout_gene}" if args.holdout_gene else "")
    path = args.out_dir / f"{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_, "feature_columns": columns,
                "config": cfg.to_dict(), "best_epoch": best_epoch,
                "genes": sorted(df["gene"].unique()), "n_rows": len(df)}, path)
    logger.info("Saved %s checkpoint: %s", args.stage, path)
    return path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    set_global_seed(args.seed)
    df = pd.read_csv(args.train_csv, low_memory=False)
    df = df[pd.to_numeric(df["label"], errors="coerce").isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    selected = select_rows(df, args.stage, args.mode, args.holdout_gene)
    if selected.empty or selected["label"].nunique() < 2:
        raise ValueError("Selected training partition has insufficient binary labels.")
    init_state = None
    if args.stage == "finetune":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for finetune.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # A different feature order makes transfer learning invalid even if
        # dimensions happen to match, so reject it explicitly.
        _, current_columns = feature_matrix(selected)
        if checkpoint.get("feature_columns") != current_columns:
            raise ValueError("Checkpoint feature schema does not match fine-tuning data.")
        init_state = checkpoint["model_state_dict"]
    fit_stage(selected, args, init_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
