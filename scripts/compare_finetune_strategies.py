#!/usr/bin/env python3
"""Benchmark all four PROJECT_PLAN.md Phase 3 step 4 fine-tuning strategies.

    A. mvmamba_recipe        -- frozen WT/VT global+local pooled ESM features
                                (src.mvmamba_features) + linear-probe head.
    B. varipred_linear_probe -- frozen WT/VT site + delta ESM features
                                (src.esm_extractor) + linear-probe head.
    C. propath_siamese       -- WT+VT forward passes through an UNFROZEN ESM-2
                                backbone (src.esm_finetune, mode='siamese');
                                LR 1e-5/batch 8/10 epochs per the plan.
    D. csbj_token_classifier -- single WT forward pass through an UNFROZEN
                                ESM-2 backbone, classify from the mutated-
                                position token (src.esm_finetune, mode='wt_site').

All four run on the identical leave-one-gene-out split (single holdout gene
per invocation) so their MCC / ROC-AUC / PR-AUC are directly comparable --
"all four are independently validated but for different data regimes; don't
commit without comparing on our own MMR data" (Phase 3 step 4).

Example
-------
    python scripts/compare_finetune_strategies.py --holdout_gene MSH2 \
        --esm_model facebook/esm2_t12_35M_UR50D
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
from src.esm_extractor import extract_features_cached, get_device  # noqa: E402
from src.esm_finetune import (  # noqa: E402
    ESMFineTuneClassifier,
    build_examples,
    fit_esm_finetune,
    predict_proba as ft_predict_proba,
)
from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc  # noqa: E402
from src.mvmamba_features import extract_mvmamba_cached  # noqa: E402
from src.transfer import BranchHead, fit_head  # noqa: E402
from scripts.finetune_esm_mmr import prepare_split, sample_weights_for  # noqa: E402

logger = logging.getLogger("compare_finetune_strategies")

MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")
STRATEGIES = ("mvmamba_recipe", "varipred_linear_probe",
             "propath_siamese", "csbj_token_classifier")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mmr_csv", type=Path,
                   default=ROOT / "data/mmr/processed/extended/extended_dataset.csv")
    p.add_argument("--panel_json", type=Path,
                   default=ROOT / "data/mmr/processed/extended/panel_sequences.json")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/finetune_comparison")
    p.add_argument("--feature_cache_dir", type=Path,
                   default=ROOT / "data/mmr/processed/esm_features")
    p.add_argument("--holdout_gene", choices=MMR_GENES, required=True)
    p.add_argument("--esm_model", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--strategies", type=str, default=",".join(STRATEGIES),
                   help="Comma-separated subset of: " + ", ".join(STRATEGIES))
    p.add_argument("--n_unfrozen_layers", type=int, default=-1,
                   help="Applies to propath_siamese / csbj_token_classifier.")
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=3e-4)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_batch_size", type=int, default=8)
    p.add_argument("--probe_epochs", type=int, default=60)
    p.add_argument("--probe_lr", type=float, default=3e-5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--clinical_weight", type=float, default=5.0)
    p.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args(argv)


def evaluate(y_true, probs, thr_val_y, thr_val_p, n_bootstrap, seed) -> dict:
    thr, _ = optimal_threshold_by_mcc(thr_val_y, thr_val_p)
    rep: dict = {"threshold": float(thr)}
    for name in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(y_true, probs, metric=name, n_bootstrap=n_bootstrap, seed=seed)
        rep[name], rep[f"{name}_ci_low"], rep[f"{name}_ci_high"] = ci["point"], ci["lower"], ci["upper"]
    mcc_ci = bootstrap_ci(y_true, probs, metric="mcc", threshold=thr, n_bootstrap=n_bootstrap, seed=seed)
    rep["mcc"], rep["mcc_ci_low"], rep["mcc_ci_high"] = mcc_ci["point"], mcc_ci["lower"], mcc_ci["upper"]
    return rep


def run_frozen_probe(args, extractor_fn, tag, ft_df, ho_df, sequence_by_gene, device) -> dict:
    """Shared driver for strategies A (mvmamba) and B (varipred) -- both are a
    frozen-feature extractor feeding an identical BranchHead linear probe, so
    only the feature-extraction call differs."""
    pool = pd.concat([ft_df, ho_df], ignore_index=True)
    blocks, metas = [], []
    for gene, sub in pool.groupby("gene"):
        seq = sequence_by_gene.get(gene)
        if not seq:
            continue
        feats, meta = extractor_fn(
            sub.reset_index(drop=True), seq, gene=gene, model_name=args.esm_model,
            processed_dir=args.feature_cache_dir / tag, batch_size=8,
            overwrite=args.overwrite_cache)
        blocks.append(feats.astype(np.float32))
        metas.append(meta)
    X = np.vstack(blocks)
    meta = pd.concat(metas, ignore_index=True)
    y = meta["label"].astype(int).to_numpy()
    is_holdout = (meta["gene"] == args.holdout_gene).to_numpy()
    ft_idx_all = np.flatnonzero(~is_holdout)
    ho_idx = np.flatnonzero(is_holdout)

    positions = meta["position"].to_numpy()
    groups = (meta["uniprot_id"].astype(str) + ":" + meta["position"].astype(str)).to_numpy()
    tr_local, va_local = make_position_group_folds(
        positions[ft_idx_all], y[ft_idx_all], k_folds=5, seed=args.seed,
        groups=groups[ft_idx_all])[0]
    tr_idx, va_idx = ft_idx_all[tr_local], ft_idx_all[va_local]

    scaler = StandardScaler().fit(X[tr_idx])
    X_tr = scaler.transform(X[tr_idx]).astype(np.float32)
    X_va = scaler.transform(X[va_idx]).astype(np.float32)
    X_ho = scaler.transform(X[ho_idx]).astype(np.float32)

    weights = sample_weights_for(meta, args.clinical_weight)
    model = BranchHead(X.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    model, best_epoch = fit_head(
        model, [X_tr], y[tr_idx], [X_va], y[va_idx], sample_weights=weights[tr_idx],
        lr=args.probe_lr, epochs=args.probe_epochs, patience=10, weight_decay=1e-2,
        batch_size=128, device=device, seed=args.seed)

    from src.transfer import predict_logits
    from scipy.special import expit
    val_probs = expit(predict_logits(model, [X_va], device))
    ho_probs = expit(predict_logits(model, [X_ho], device))
    rep = evaluate(y[ho_idx], ho_probs, y[va_idx], val_probs, args.n_bootstrap, args.seed)
    return {"best_epoch": best_epoch, "n_finetune": len(tr_idx), **rep}


def run_esm_finetune(args, mode, ft_df, ho_df, sequence_by_gene, device) -> dict:
    positions = ft_df["position"].to_numpy()
    groups = (ft_df["uniprot_id"].astype(str) + ":" + ft_df["position"].astype(str)).to_numpy()
    labels = ft_df["label"].astype(int).to_numpy()
    tr_local, va_local = make_position_group_folds(
        positions, labels, k_folds=5, seed=args.seed, groups=groups)[0]

    model = ESMFineTuneClassifier(
        model_name=args.esm_model, mode=mode, n_unfrozen_layers=args.n_unfrozen_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout)
    train_ex = build_examples(ft_df.iloc[tr_local], sequence_by_gene, mode)
    val_ex = build_examples(ft_df.iloc[va_local], sequence_by_gene, mode)
    ho_ex = build_examples(ho_df, sequence_by_gene, mode)

    model, best_epoch, _ = fit_esm_finetune(
        model, train_ex, val_ex, device, backbone_lr=args.backbone_lr,
        head_lr=args.head_lr, epochs=args.finetune_epochs, patience=3,
        batch_size=args.finetune_batch_size, seed=args.seed)
    val_probs = ft_predict_proba(model, val_ex, device)
    ho_probs = ft_predict_proba(model, ho_ex, device)
    val_labels = np.array([e.label for e in val_ex])
    ho_labels = np.array([e.label for e in ho_ex])
    rep = evaluate(ho_labels, ho_probs, val_labels, val_probs, args.n_bootstrap, args.seed)
    return {"best_epoch": best_epoch, "n_finetune": len(train_ex), **rep}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    device = get_device()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"Unknown strategies {sorted(unknown)}; choose from {STRATEGIES}.")

    master = pd.read_csv(args.mmr_csv, low_memory=False)
    panel = json.loads(Path(args.panel_json).read_text())
    sequence_by_gene = {g.upper(): d["sequence"] for g, d in panel.items()}
    ft_df, ho_df = prepare_split(master, args.holdout_gene)
    ft_df = ft_df.assign(label_weight=sample_weights_for(ft_df, args.clinical_weight))
    logger.info("holdout=%s | fine-tune rows=%d | holdout rows=%d",
                args.holdout_gene, len(ft_df), len(ho_df))

    rows = []
    for strat in strategies:
        t0 = time.time()
        logger.info("=== strategy: %s ===", strat)
        if strat == "mvmamba_recipe":
            rep = run_frozen_probe(args, extract_mvmamba_cached, "mvmamba",
                                   ft_df, ho_df, sequence_by_gene, device)
        elif strat == "varipred_linear_probe":
            rep = run_frozen_probe(args, extract_features_cached, "varipred",
                                   ft_df, ho_df, sequence_by_gene, device)
        elif strat == "propath_siamese":
            rep = run_esm_finetune(args, "siamese", ft_df, ho_df, sequence_by_gene, device)
        elif strat == "csbj_token_classifier":
            rep = run_esm_finetune(args, "wt_site", ft_df, ho_df, sequence_by_gene, device)
        else:
            continue
        rep["strategy"] = strat
        rep["runtime_s"] = round(time.time() - t0, 1)
        rows.append(rep)
        logger.info("[%s] ROC-AUC %.3f [%.3f-%.3f] | MCC %.3f (%.1fs)",
                    strat, rep["roc_auc"], rep["roc_auc_ci_low"], rep["roc_auc_ci_high"],
                    rep["mcc"], rep["runtime_s"])

    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"strategy_comparison_{args.holdout_gene}.csv"
    results.to_csv(out_path, index=False)
    (args.out_dir / f"strategy_comparison_{args.holdout_gene}_meta.json").write_text(json.dumps({
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_gene": args.holdout_gene, "esm_model": args.esm_model,
        "strategies": strategies,
    }, indent=2))

    print("\n============= FINE-TUNING STRATEGY COMPARISON (holdout=%s) ==========="
          % args.holdout_gene)
    cols = ["strategy", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "pr_auc", "mcc", "threshold", "n_finetune", "runtime_s"]
    print(results[[c for c in cols if c in results.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    print(f"Results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
