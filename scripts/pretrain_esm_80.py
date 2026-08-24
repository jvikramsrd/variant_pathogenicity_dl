#!/usr/bin/env python3
"""Stage 1 — pretrain the pathogenicity head on ALL 80 panel-gene embeddings.

Consumes the audited broad multi-source table (default: the 80-protein panel)
and fits a classification head on frozen ESM-2 embedding features stacked with
the published-model prior scores.  The resulting checkpoint is consumed by
``scripts/run_mmr_transfer.py finetune`` for MMR fine-tuning and leave-one-
gene-out evaluation.

Modes
-----
* ``practical``       : pretraining may include MMR proteins present in the
  panel; produces an adapted model, never an unseen-gene claim.
* ``leave_gene_out``  : every MMR gene is excluded from pretraining so no MMR
  variant contaminates the general representation (transfer estimate).

Examples
--------
    # CPU-friendly pretraining on priors only
    python scripts/pretrain_esm_80.py --features priors

    # Full ESM-embedding pretraining across all 80 genes (GPU advised)
    python scripts/pretrain_esm_80.py --features esm+priors \
        --esm_model facebook/esm2_t33_650M_UR50D

No stage runs automatically; invoke explicitly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import make_position_group_folds  # noqa: E402
from src.esm_extractor import get_device  # noqa: E402
from src.train import set_global_seed  # noqa: E402
from src.transfer import (  # noqa: E402
    assemble_features,
    build_model,
    fit_head,
    predict_logits,
    save_checkpoint,
    select_stage_rows,
    stage_sample_weights,
)

logger = logging.getLogger("pretrain_esm_80")

DEFAULT_TRAIN_CSV = ROOT / "data/processed/extended/extended_dataset_train.csv"
DEFAULT_PANEL_JSON = ROOT / "data/raw/uniprot/expanded_panel.json"
DEFAULT_OUT_DIR = ROOT / "data/processed/transfer"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features", choices=["priors", "esm+priors"],
                   default="priors",
                   help="'esm+priors' uses every gene's frozen ESM embeddings "
                        "as pretraining signal (the plan's requirement); "
                        "'priors' is the CPU-only fallback.")
    p.add_argument("--mode", choices=("practical", "leave_gene_out"),
                   default="leave_gene_out")
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--panel_json", type=Path, default=DEFAULT_PANEL_JSON)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--checkpoint_name", type=str, default=None,
                   help="Override output checkpoint filename.")
    p.add_argument("--genes", type=str, default=None,
                   help="Comma-separated gene subset (smoke tests only).")
    p.add_argument("--esm_model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="ESM-2 checkpoint for esm+priors mode. Use "
                        "facebook/esm2_t33_650M_UR50D on GPU.")
    p.add_argument("--extract_batch_size", type=int, default=8)
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--clinical_weight", type=float, default=5.0)
    t.add_argument("--dms_weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    set_global_seed(args.seed)
    device = get_device()

    df = pd.read_csv(args.train_csv, low_memory=False)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df[df["label"].notna()].reset_index(drop=True)
    if args.genes:
        keep = {g.strip().upper() for g in args.genes.split(",") if g.strip()}
        df = df[df["gene"].isin(keep)].reset_index(drop=True)

    selected = select_stage_rows(df, stage="pretrain", mode=args.mode,
                                 holdout_gene=None)
    if selected.empty or selected["label"].nunique() < 2:
        raise ValueError("Pretraining partition has insufficient binary labels.")
    n_pos = int((selected["label"] == 1).sum())
    n_neg = int((selected["label"] == 0).sum())
    logger.info("Pretraining rows: %d (%d P/LP, %d B/LB) across %d genes "
                "[mode=%s]", len(selected), n_pos, n_neg,
                selected["gene"].nunique(), args.mode)

    if args.features == "esm+priors":
        panel = json.loads(Path(args.panel_json).read_text())
        sequence_by_gene = {g: d["sequence"] for g, d in panel.items()}
    else:
        sequence_by_gene = {}

    bundle = assemble_features(
        selected, sequence_by_gene, args.esm_model,
        processed_dir=ROOT / "data" / "processed", device=device,
        features_mode=args.features, batch_size=args.extract_batch_size,
        overwrite_cache=args.overwrite_cache)
    X_full = bundle.stack() if bundle.X_esm is not None else bundle.X_prior
    meta = bundle.meta
    y = meta["label"].astype(int).to_numpy()
    logger.info("Feature matrix: %s (ESM block: %s | prior cols: %d)",
                X_full.shape,
                None if bundle.X_esm is None else bundle.X_esm.shape[1],
                len(bundle.prior_cols))

    groups = (meta["uniprot_id"].astype(str) + ":"
              + meta["position"].astype(str)).to_numpy()
    positions = meta["position"].to_numpy()
    train_idx, val_idx = make_position_group_folds(
        positions, y, k_folds=5, seed=args.seed, groups=groups)[0]

    scaler = StandardScaler().fit(X_full[train_idx])
    X_tr = scaler.transform(X_full[train_idx]).astype(np.float32)
    X_va = scaler.transform(X_full[val_idx]).astype(np.float32)
    weights_all = stage_sample_weights(meta, args.clinical_weight, args.dms_weight)

    model = build_model("esm", dims=[X_full.shape[1]],
                        hidden_dim=args.hidden_dim, dropout=args.dropout)
    model, best_epoch = fit_head(
        model, [X_tr], y[train_idx], [X_va], y[val_idx],
        sample_weights=weights_all[train_idx],
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        weight_decay=1e-2, batch_size=args.batch_size, device=device,
        seed=args.seed)

    from sklearn.metrics import roc_auc_score
    from src.calibration import expit
    val_logits = predict_logits(model, [X_va], device)
    val_auc = roc_auc_score(y[val_idx], expit(val_logits))
    logger.info("Inner-validation ROC-AUC: %.4f (best epoch %d)", val_auc, best_epoch)

    name = args.checkpoint_name or (
        f"pretrain_{args.mode}_{args.features.replace('+', '_')}"
        + ("_" + args.esm_model.split("/")[-1] if args.features == "esm+priors" else "")
        + ".pt")
    ckpt_path = save_checkpoint(
        args.out_dir / name, model, scaler.mean_, scaler.scale_,
        feature_columns=bundle.prior_cols,
        cfg_dict={
            **{k: getattr(args, k) for k in (
                "features", "mode", "esm_model", "epochs", "patience", "lr",
                "batch_size", "hidden_dim", "dropout", "clinical_weight",
                "seed")},
            "input_dim": int(X_full.shape[1]),
            "esm_dim": 0 if bundle.X_esm is None else int(bundle.X_esm.shape[1]),
            "arch": "branch_full",
        },
        extra={
            "stage": "pretrain",
            "genes": sorted(meta["gene"].unique()),
            "n_rows": int(len(meta)),
            "inner_val_roc_auc": float(val_auc),
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    summary = {
        "checkpoint": str(ckpt_path),
        "rows": int(len(meta)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_genes": int(meta["gene"].nunique()),
        "feature_dim": int(X_full.shape[1]),
        "esm_dim": 0 if bundle.X_esm is None else int(bundle.X_esm.shape[1]),
        "prior_cols": len(bundle.prior_cols),
        "inner_val_roc_auc": float(val_auc),
        "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / f"{Path(name).stem}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
