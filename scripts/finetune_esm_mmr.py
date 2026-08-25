#!/usr/bin/env python3
"""Full ESM-2 fine-tuning on MMR data with leave-one-gene-out evaluation.

Unlike ``scripts/run_mmr_transfer.py`` (frozen ESM embeddings + MLP head,
VariPred's linear-probe recipe), this script back-propagates into the ESM-2
backbone itself using :mod:`src.esm_finetune` -- ProPath's Siamese/PLLR-style
recipe (``--mode siamese``) or a CSBJ-style single-pass token classifier
(``--mode wt_site``). See PROJECT_PLAN.md Phase 3 step 4: benchmark all four
fine-tuning strategies on our own data before committing to one.

Examples
--------
    # CSBJ-style, last 4 layers unfrozen, small checkpoint for a quick check
    python scripts/finetune_esm_mmr.py --mode wt_site --n_unfrozen_layers 4 \
        --esm_model facebook/esm2_t12_35M_UR50D --eval holdout --holdout_gene MSH2

    # ProPath recipe, full backbone fine-tune, GPU
    python scripts/finetune_esm_mmr.py --mode siamese --n_unfrozen_layers -1 \
        --esm_model facebook/esm2_t33_650M_UR50D --eval lopo \
        --backbone_lr 1e-5 --batch_size 8 --epochs 10 --gradient_checkpointing
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import make_position_group_folds  # noqa: E402
from src.esm_extractor import get_device  # noqa: E402
from src.esm_finetune import (  # noqa: E402
    ESMFineTuneClassifier,
    build_examples,
    fit_esm_finetune,
    predict_proba,
)
from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc  # noqa: E402
from src.transfer import load_checkpoint, save_checkpoint  # noqa: E402

logger = logging.getLogger("finetune_esm_mmr")

DEFAULT_MMR_CSV = ROOT / "data/mmr/processed/extended/extended_dataset.csv"
DEFAULT_PANEL_JSON = ROOT / "data/mmr/processed/extended/panel_sequences.json"
MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("wt_site", "siamese"), default="wt_site",
                   help="'wt_site'=CSBJ-style single WT forward pass; "
                        "'siamese'=ProPath-style WT+VT forward passes.")
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--n_unfrozen_layers", type=int, default=-1,
                   help="-1 = full fine-tune; 0 = frozen backbone (ablation "
                        "floor); N>0 = unfreeze the last N transformer layers.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mmr_csv", type=Path, default=DEFAULT_MMR_CSV)
    p.add_argument("--panel_json", type=Path, default=DEFAULT_PANEL_JSON)
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/esm_finetune")
    p.add_argument("--eval", choices=("lopo", "holdout"), default="lopo")
    p.add_argument("--holdout_gene", choices=MMR_GENES, default=None)
    p.add_argument("--max_residues", type=int, default=1022)
    t = p.add_argument_group("training (ProPath defaults)")
    t.add_argument("--backbone_lr", type=float, default=1e-5)
    t.add_argument("--head_lr", type=float, default=3e-4)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--weight_decay", type=float, default=1e-2)
    t.add_argument("--clinical_weight", type=float, default=5.0)
    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_checkpoints", action="store_true",
                   help="Persist the fine-tuned model per split (large: full "
                        "backbone state dict per gene).")
    return p.parse_args(argv)


def prepare_split(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clinical-label MMR partitions (fine-tune genes, held-out gene).

    Same contract as ``scripts/run_mmr_transfer.py::prepare_split`` -- PMS2
    homology-gated rows and DMS-only labels are excluded so both scripts stay
    directly comparable.
    """
    df = df[df["gene"].isin(MMR_GENES)].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()].copy()
    if "pms2_homology_excluded" in df.columns:
        keep = pd.to_numeric(df["pms2_homology_excluded"], errors="coerce").fillna(0) != 1
        df = df.loc[keep]
    if "label_source" in df.columns:
        df = df.loc[df["label_source"].isin(["clinvar", "pg_clinical"])]
    ft = df[df["gene"] != holdout].reset_index(drop=True)
    ho = df[df["gene"] == holdout].reset_index(drop=True)
    return ft, ho


def sample_weights_for(meta: pd.DataFrame, clinical_weight: float) -> np.ndarray:
    if "label_source" in meta.columns:
        is_clinical = meta["label_source"].isin(["clinvar", "pg_clinical"]).to_numpy()
    else:
        is_clinical = np.ones(len(meta), dtype=bool)
    w = np.where(is_clinical, clinical_weight, 1.0).astype(np.float32)
    if "label_weight" in meta.columns:
        quality = pd.to_numeric(meta["label_weight"], errors="coerce").fillna(1.0).to_numpy(np.float32)
        w = w * quality
    return w


def run_one_split(args: argparse.Namespace, master: pd.DataFrame,
                  sequence_by_gene: dict, device: torch.device, holdout: str) -> dict:
    logger.info("=== ESM fine-tune leave-one-gene-out: holdout=%s (mode=%s) ===",
                holdout, args.mode)
    ft_df, ho_df = prepare_split(master, holdout)
    if ft_df.empty or ho_df.empty:
        raise ValueError(f"Split {holdout}: empty partition "
                         f"(fine-tune={len(ft_df)}, holdout={len(ho_df)}).")
    ft_df = ft_df.assign(label_weight=sample_weights_for(ft_df, args.clinical_weight))

    positions = ft_df["position"].to_numpy()
    groups = (ft_df["uniprot_id"].astype(str) + ":" + ft_df["position"].astype(str)).to_numpy()
    labels = ft_df["label"].astype(int).to_numpy()
    tr_local, va_local = make_position_group_folds(
        positions, labels, k_folds=5, seed=args.seed, groups=groups)[0]

    model = ESMFineTuneClassifier(
        model_name=args.esm_model, mode=args.mode,
        n_unfrozen_layers=args.n_unfrozen_layers, hidden_dim=args.hidden_dim,
        dropout=args.dropout, gradient_checkpointing=args.gradient_checkpointing)

    train_ex = build_examples(ft_df.iloc[tr_local], sequence_by_gene, args.mode,
                              max_residues=args.max_residues)
    val_ex = build_examples(ft_df.iloc[va_local], sequence_by_gene, args.mode,
                            max_residues=args.max_residues)
    ho_ex = build_examples(ho_df, sequence_by_gene, args.mode, max_residues=args.max_residues)
    logger.info("Fine-tune train=%d val=%d holdout=%d", len(train_ex), len(val_ex), len(ho_ex))

    t0 = time.time()
    model, best_epoch, best_val_auc = fit_esm_finetune(
        model, train_ex, val_ex, device,
        backbone_lr=args.backbone_lr, head_lr=args.head_lr, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        weight_decay=args.weight_decay, seed=args.seed)

    val_probs = predict_proba(model, val_ex, device, batch_size=max(16, args.batch_size))
    ho_probs = predict_proba(model, ho_ex, device, batch_size=max(16, args.batch_size))
    val_labels = np.array([e.label for e in val_ex])
    ho_labels = np.array([e.label for e in ho_ex])

    thr, _ = optimal_threshold_by_mcc(val_labels, val_probs)
    rep: dict = {"threshold": float(thr)}
    for name in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(ho_labels, ho_probs, metric=name, n_bootstrap=args.n_bootstrap, seed=args.seed)
        rep[name], rep[f"{name}_ci_low"], rep[f"{name}_ci_high"] = ci["point"], ci["lower"], ci["upper"]
    mcc_ci = bootstrap_ci(ho_labels, ho_probs, metric="mcc", threshold=thr,
                          n_bootstrap=args.n_bootstrap, seed=args.seed)
    rep["mcc"], rep["mcc_ci_low"], rep["mcc_ci_high"] = mcc_ci["point"], mcc_ci["lower"], mcc_ci["upper"]

    row = {
        "holdout_gene": holdout, "mode": args.mode, "esm_model": args.esm_model,
        "n_unfrozen_layers": args.n_unfrozen_layers,
        "n_finetune": len(train_ex), "n_inner_val": len(val_ex), "n_holdout": len(ho_ex),
        "best_epoch": best_epoch, "inner_val_roc_auc": best_val_auc,
        "runtime_s": round(time.time() - t0, 1), **rep,
    }
    logger.info("[holdout=%s] ROC-AUC %.3f [%.3f-%.3f] | PR-AUC %.3f | MCC %.3f @thr %.3f",
                holdout, rep["roc_auc"], rep["roc_auc_ci_low"], rep["roc_auc_ci_high"],
                rep["pr_auc"], rep["mcc"], rep["threshold"])

    if args.save_checkpoints:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.out_dir / f"esm_finetune_{args.mode}_holdout_{holdout}.pt"
        save_checkpoint(ckpt_path, model, None, None, feature_columns=[],
                        cfg_dict={"mode": args.mode, "esm_model": args.esm_model,
                                  "n_unfrozen_layers": args.n_unfrozen_layers})
        row["checkpoint"] = str(ckpt_path)
    return row


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
    device = get_device()

    if not args.mmr_csv.exists():
        raise SystemExit(f"{args.mmr_csv} not found -- run scripts/build_mmr_dataset.py first.")
    master = pd.read_csv(args.mmr_csv, low_memory=False)
    panel = json.loads(Path(args.panel_json).read_text())
    sequence_by_gene = {g.upper(): d["sequence"] for g, d in panel.items()}

    splits = [args.holdout_gene] if args.eval == "holdout" else list(MMR_GENES)
    if args.eval == "holdout" and args.holdout_gene is None:
        raise SystemExit("--holdout_gene required with --eval holdout.")

    rows = [run_one_split(args, master, sequence_by_gene, device, g)
            for g in splits if g in set(master["gene"].unique())]

    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{'lopo' if args.eval == 'lopo' else 'holdout_' + args.holdout_gene}"
    results_path = args.out_dir / f"esm_finetune_results_{tag}.csv"
    results.to_csv(results_path, index=False)
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "esm_model": args.esm_model,
        "n_unfrozen_layers": args.n_unfrozen_layers,
        "splits_evaluated": splits, "results_csv": str(results_path),
        "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / f"esm_finetune_summary_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n================ ESM FINE-TUNE LEAVE-ONE-GENE-OUT RESULTS ============")
    cols = ["holdout_gene", "mode", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "pr_auc", "mcc", "threshold", "n_holdout", "best_epoch"]
    print(results[[c for c in cols if c in results.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    print(f"Results -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
