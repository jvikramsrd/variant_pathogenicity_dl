#!/usr/bin/env python3
"""Leave-one-protein-out CV over the broad pretraining panel.

docs/TRAINING_NOTES.md flagged this as the next step after the priors-only
80-gene run: "measures how well the model transfers to proteins it never
saw (the clinically relevant question for rare disease genes)". The existing
position-grouped K-fold CV (:func:`src.dataset.make_position_group_folds`)
only guarantees *residue*-disjoint folds; a protein can still appear on both
sides of a fold. This script instead holds out entire proteins, one at a
time, training on every other panel gene -- the honest per-protein
generalisation estimate, same spirit as ``scripts/run_mmr_transfer.py``'s
leave-one-MMR-gene-out but for the full panel.

Expensive by construction (one full fit per held-out gene); use ``--genes``
to restrict to a manageable subset for a first pass.

Example
-------
    # Cheap CPU pass with priors-only features on 5 genes
    python scripts/eval_leave_one_protein_out.py --features priors \
        --genes TP53,BRCA1,PTEN,KCNQ1,CALM1

    # Full esm+priors run (GPU advised)
    python scripts/eval_leave_one_protein_out.py --features esm+priors \
        --esm_model facebook/esm2_t33_650M_UR50D
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
from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc  # noqa: E402
from src.train import set_global_seed  # noqa: E402
from src.transfer import (  # noqa: E402
    assemble_features,
    build_model,
    fit_head,
    predict_logits,
    stage_sample_weights,
)

logger = logging.getLogger("eval_leave_one_protein_out")

DEFAULT_TRAIN_CSV = ROOT / "data/processed/extended/extended_dataset_train.csv"
DEFAULT_PANEL_JSON = ROOT / "data/raw/uniprot/expanded_panel.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--panel_json", type=Path, default=DEFAULT_PANEL_JSON)
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/lopo_broad_panel")
    p.add_argument("--features", choices=["priors", "esm+priors"], default="priors")
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--extract_batch_size", type=int, default=8)
    p.add_argument("--genes", type=str, default=None,
                   help="Comma-separated gene subset; default = every gene "
                        "with >= --min_variants labelled rows.")
    p.add_argument("--min_variants", type=int, default=30,
                   help="Skip a gene as a held-out target if it has fewer "
                        "labelled rows than this (too noisy an estimate).")
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--clinical_weight", type=float, default=5.0)
    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
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
    set_global_seed(args.seed)
    device = get_device()

    df = pd.read_csv(args.train_csv, low_memory=False)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df[df["label"].notna()].reset_index(drop=True)

    if args.genes:
        target_genes = [g.strip().upper() for g in args.genes.split(",") if g.strip()]
    else:
        counts = df["gene"].value_counts()
        target_genes = sorted(counts[counts >= args.min_variants].index.tolist())
    logger.info("Leave-one-protein-out over %d gene(s): %s", len(target_genes), target_genes)

    sequence_by_gene = {}
    if args.features == "esm+priors":
        panel = json.loads(Path(args.panel_json).read_text())
        sequence_by_gene = {g: d["sequence"] for g, d in panel.items()}

    rows = []
    for held_out in target_genes:
        t_gene = time.time()
        train_pool = df[df["gene"] != held_out].reset_index(drop=True)
        held_pool = df[df["gene"] == held_out].reset_index(drop=True)
        if held_pool["label"].nunique() < 2:
            logger.warning("%s: single-class holdout set, skipping.", held_out)
            continue
        logger.info("=== held out: %s (train pool %d rows, %d proteins) ===",
                    held_out, len(train_pool), train_pool["gene"].nunique())

        pool = pd.concat([train_pool, held_pool], ignore_index=True)
        bundle = assemble_features(
            pool, sequence_by_gene, args.esm_model, processed_dir=ROOT / "data" / "processed",
            device=device, features_mode=args.features, batch_size=args.extract_batch_size,
            overwrite_cache=args.overwrite_cache)
        X = bundle.stack() if bundle.X_esm is not None else bundle.X_prior
        meta = bundle.meta
        y = meta["label"].astype(int).to_numpy()
        is_held = (meta["gene"] == held_out).to_numpy()
        tr_pool_idx = np.flatnonzero(~is_held)
        held_idx = np.flatnonzero(is_held)
        if len(np.unique(y[held_idx])) < 2:
            logger.warning("%s: single-class after feature-extraction alignment, skipping.", held_out)
            continue

        positions = meta["position"].to_numpy()
        groups = (meta["uniprot_id"].astype(str) + ":" + meta["position"].astype(str)).to_numpy() \
            if "uniprot_id" in meta.columns else positions
        tr_local, va_local = make_position_group_folds(
            positions[tr_pool_idx], y[tr_pool_idx], k_folds=5, seed=args.seed,
            groups=groups[tr_pool_idx])[0]
        tr_idx, va_idx = tr_pool_idx[tr_local], tr_pool_idx[va_local]

        scaler = StandardScaler().fit(X[tr_idx])
        X_tr = scaler.transform(X[tr_idx]).astype(np.float32)
        X_va = scaler.transform(X[va_idx]).astype(np.float32)
        X_ho = scaler.transform(X[held_idx]).astype(np.float32)
        weights = stage_sample_weights(meta, args.clinical_weight)

        model = build_model("esm", dims=[X.shape[1]], hidden_dim=args.hidden_dim,
                            dropout=args.dropout)
        model, best_epoch = fit_head(
            model, [X_tr], y[tr_idx], [X_va], y[va_idx], sample_weights=weights[tr_idx],
            lr=args.lr, epochs=args.epochs, patience=args.patience, weight_decay=1e-2,
            batch_size=args.batch_size, device=device, seed=args.seed)

        from scipy.special import expit
        val_probs = expit(predict_logits(model, [X_va], device))
        ho_probs = expit(predict_logits(model, [X_ho], device))
        thr, _ = optimal_threshold_by_mcc(y[va_idx], val_probs)
        rep: dict = {"threshold": float(thr)}
        for name in ("roc_auc", "pr_auc"):
            ci = bootstrap_ci(y[held_idx], ho_probs, metric=name,
                              n_bootstrap=args.n_bootstrap, seed=args.seed)
            rep[name], rep[f"{name}_ci_low"], rep[f"{name}_ci_high"] = ci["point"], ci["lower"], ci["upper"]
        mcc_ci = bootstrap_ci(y[held_idx], ho_probs, metric="mcc", threshold=thr,
                              n_bootstrap=args.n_bootstrap, seed=args.seed)
        rep["mcc"], rep["mcc_ci_low"], rep["mcc_ci_high"] = mcc_ci["point"], mcc_ci["lower"], mcc_ci["upper"]

        row = {
            "held_out_gene": held_out, "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)), "n_held_out": int(len(held_idx)),
            "best_epoch": int(best_epoch), "runtime_s": round(time.time() - t_gene, 1),
            **rep,
        }
        rows.append(row)
        logger.info("[held_out=%s] ROC-AUC %.3f [%.3f-%.3f] | MCC %.3f (%.1fs)",
                    held_out, rep["roc_auc"], rep["roc_auc_ci_low"], rep["roc_auc_ci_high"],
                    rep["mcc"], row["runtime_s"])

    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"lopo_{args.features.replace('+', '_')}_results.csv"
    results.to_csv(out_path, index=False)
    (args.out_dir / f"lopo_{args.features.replace('+', '_')}_summary.json").write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": args.features, "esm_model": args.esm_model if args.features == "esm+priors" else None,
        "genes_evaluated": target_genes, "results_csv": str(out_path),
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2))

    print("\n============= LEAVE-ONE-PROTEIN-OUT RESULTS (broad panel) ============")
    if len(results):
        cols = ["held_out_gene", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                "pr_auc", "mcc", "n_held_out"]
        print(results[[c for c in cols if c in results.columns]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nMean ROC-AUC across held-out proteins: {results['roc_auc'].mean():.4f}")
    print("=" * 74)
    print(f"Results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
