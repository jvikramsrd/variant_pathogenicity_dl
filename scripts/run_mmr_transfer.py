#!/usr/bin/env python3
"""Stage 2 — fine-tune the pretrained head on MMR data + leave-one-gene-out eval.

Consumes the Phase-1 dedicated MMR table built by ``scripts/build_mmr_dataset.py``
and optionally the Stage-1 checkpoint produced by ``scripts/pretrain_esm_80.py``
(pretraining over ALL 80 panel genes' ESM embeddings).

Evaluation contract (PROJECT_PLAN.md Phases 3/5/6)
--------------------------------------------------
* **Leave-one-MMR-gene-out**: fine-tune without one gene, report metrics only
  on the held-out gene — the honest generalization number for a 4-gene project.
* MCC primary with **bootstrap CIs** (10,000 iterations by default), ROC-AUC /
  PR-AUC secondary; the decision threshold is tuned by MCC on an inner
  validation slice drawn from the fine-tuning genes only — never assumed 0.5,
  never tuned on the held-out gene.
* Mandatory ablation comparison on identical splits: ESM branch-only,
  prior branch-only, concat fusion, GateWave gated fusion ("never assume
  fusion wins").

Examples
--------
    python scripts/run_mmr_transfer.py \
        --checkpoint data/processed/transfer/pretrain_leave_gene_out_esm_priors.pt \
        --eval lopo --features esm+priors

    python scripts/run_mmr_transfer.py --eval holdout --holdout_gene MSH2 --scratch
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
    GENE_CONSTANT_PRIOR_COLS,
    FeatureBundle,
    add_within_gene_rank_features,
    assemble_features,
    prior_impute_values,
    rankable_score_columns,
    build_model,
    fit_head,
    load_checkpoint,
    predict_logits,
    stage_sample_weights,
)

logger = logging.getLogger("run_mmr_transfer")

DEFAULT_MMR_CSV = ROOT / "data/mmr/processed/extended/extended_dataset.csv"
MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")

#: Deterministic per-architecture seed offsets (never use hash()).
ARCH_SEED_OFFSET = {"priors": 11, "esm": 23, "concat": 37, "gatewave": 53}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--eval", choices=("lopo", "holdout"), default="lopo",
                   help="'lopo' runs every leave-one-gene-out split; "
                        "'holdout' evaluates a single --holdout_gene.")
    p.add_argument("--holdout_gene", choices=MMR_GENES, default=None)
    p.add_argument(
        "--rank_normalize", choices=("off", "add", "replace"), default="off",
        help="Replace or augment the raw published-score columns with "
             "within-gene percentile ranks plus a skip-NaN consensus mean. "
             "'replace' drops the raw scores and keeps only ranks (the "
             "cleanest test); 'add' keeps both. Each predictor's score "
             "distribution differs per gene, so raw scales do not transfer "
             "across a leave-one-gene-out split -- ranks do. Ranks are "
             "computed over every variant of each gene, labelled and VUS "
             "alike, and never touch the label column. Changing this changes "
             "the feature schema, so a warm-start checkpoint must have been "
             "pretrained with the same setting (or use --scratch).")
    p.add_argument(
        "--gene_constant_priors", choices=("auto", "drop", "keep"),
        default="auto",
        help="What to do with the gnomAD gene-level constraint columns (pLI, "
             "oe_lof, oe_mis, mis_z, syn_z). They are constant across every "
             "variant of a gene, so under leave-one-gene-out they are not "
             "features but a 5-dimensional gene identifier: the head "
             "memorises each training gene's base rate, then meets an unseen "
             "vector on the held-out gene. Measured cost of keeping them: "
             "MLH1 collapses to ROC-AUC 0.500 / MCC 0.000 in every seed "
             "tried; dropping them gives 0.965 +/- 0.007. 'auto' therefore "
             "drops them for --eval lopo and keeps them for a single "
             "--eval holdout; 'keep' restores the old behaviour.")
    p.add_argument("--scratch", action="store_true",
                   help="Train without loading any pretrained checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Stage-1 pretraining checkpoint (required unless "
                        "--scratch).")
    p.add_argument("--mmr_csv", type=Path, default=DEFAULT_MMR_CSV)
    p.add_argument("--panel_json", type=Path,
                   default=ROOT / "data/mmr/processed/extended/panel_sequences.json")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "data/processed/mmr_transfer")
    p.add_argument("--feature_cache_dir", type=Path,
                   default=ROOT / "data/mmr/processed/esm_features",
                   help="Separate cache namespace for MMR ESM extraction so "
                        "broad-panel caches are never clobbered.")
    p.add_argument("--features", choices=["priors", "esm+priors"],
                   default="esm+priors",
                   help="Must match the checkpoint's feature mode.")
    p.add_argument("--esm_model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="Must match the checkpoint's ESM backbone.")
    p.add_argument("--extract_batch_size", type=int, default=8)
    t = p.add_argument_group("fine-tuning")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--finetune_lr", type=float, default=3e-5)
    t.add_argument("--scratch_lr", type=float, default=3e-4)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--clinical_weight", type=float, default=5.0)
    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstrap", type=int, default=10_000,
                   help="Plan-mandated bootstrap iterations (lower for smoke runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite_cache", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Split preparation
# --------------------------------------------------------------------------- #
def prepare_split(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clinical-label MMR partitions: (fine-tune genes, held-out gene).

    PMS2 rows inside the pseudogene-homology region were already label-stripped
    upstream; they are dropped here entirely to keep partitions clean.
    """
    df = df[df["gene"].isin(MMR_GENES)].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()].copy()
    if "pms2_homology_excluded" in df.columns:
        keep = pd.to_numeric(df["pms2_homology_excluded"], errors="coerce") \
            .fillna(0) != 1
        df = df.loc[keep]
    if "label_source" in df.columns:
        clinical = df["label_source"].isin(["clinvar", "pg_clinical"]).to_numpy()
        df = df.loc[clinical]
    ft = df[df["gene"] != holdout].reset_index(drop=True)
    ho = df[df["gene"] == holdout].reset_index(drop=True)
    return ft, ho


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray,
                         thr_val_y: np.ndarray, thr_val_p: np.ndarray,
                         n_bootstrap: int, seed: int) -> dict:
    """Tune the threshold on inner-validation, then CI-report on held-out."""
    thr, _ = optimal_threshold_by_mcc(thr_val_y, thr_val_p)
    rep: dict = {"threshold": float(thr)}
    for name in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(y_true, probs, metric=name,
                          n_bootstrap=n_bootstrap, seed=seed)
        rep[name] = ci["point"]
        rep[f"{name}_ci_low"] = ci["lower"]
        rep[f"{name}_ci_high"] = ci["upper"]
    mcc_ci = bootstrap_ci(y_true, probs, metric="mcc", threshold=thr,
                          n_bootstrap=n_bootstrap, seed=seed)
    rep["mcc"] = mcc_ci["point"]
    rep["mcc_ci_low"] = mcc_ci["lower"]
    rep["mcc_ci_high"] = mcc_ci["upper"]
    return rep


def scale_views(mats: list[np.ndarray], tr_idx: np.ndarray,
                va_idx: np.ndarray, ho_idx: np.ndarray):
    """Per-view StandardScaler fitted strictly on the fine-tune train slice."""
    scaled_tr, scaled_va, scaled_ho = [], [], []
    for m in mats:
        sc = StandardScaler().fit(m[tr_idx])
        scaled_tr.append(sc.transform(m[tr_idx]).astype(np.float32))
        scaled_va.append(sc.transform(m[va_idx]).astype(np.float32))
        scaled_ho.append(sc.transform(m[ho_idx]).astype(np.float32))
    return scaled_tr, scaled_va, scaled_ho


# --------------------------------------------------------------------------- #
# One LOPO split
# --------------------------------------------------------------------------- #
def run_one_split(args: argparse.Namespace, master: pd.DataFrame,
                  sequence_by_gene: dict, device: torch.device,
                  holdout: str, ckpt: dict | None) -> list[dict]:
    logger.info("=== Leave-one-gene-out split: holdout=%s ===", holdout)
    ft_df, ho_df = prepare_split(master, holdout)
    if ft_df.empty or ho_df.empty:
        raise ValueError(f"Split {holdout}: empty partition "
                         f"(fine-tune={len(ft_df)}, holdout={len(ho_df)}); "
                         "check Phase-1 labels.")
    logger.info("Fine-tune rows %d %s | holdout %s: %d rows",
                len(ft_df), dict(ft_df["gene"].value_counts()), holdout, len(ho_df))

    pool = pd.concat([ft_df, ho_df], ignore_index=True)
    # When warm-starting, force the pretrained checkpoint's exact prior-column
    # order so weight transfer stays valid (missing columns fail loudly here).
    fixed_prior_cols = (ckpt or {}).get("feature_columns")
    # Pin the pretraining stage's imputation constants as well as its column
    # order. Checkpoints written before this was persisted simply have no
    # entry, and fall back to medians computed here.
    fixed_impute = (ckpt or {}).get("prior_impute_values")
    if not fixed_impute:
        if ckpt is not None:
            logger.warning(
                "Checkpoint has no stored prior-imputation values (written by "
                "an older pretrain run); deriving them from the fine-tune "
                "partition instead. Re-run stage 1 to remove this source of "
                "pretrain/fine-tune feature drift.")
        # Derive fill values from the fine-tune partition ONLY. `pool` below
        # deliberately contains the holdout gene so its features can be built
        # in the same call, but letting the holdout influence the imputation
        # constants would leak it into training -- and because most zs_*
        # columns are ~99% missing, that constant is the feature for almost
        # every row.
        fixed_impute = prior_impute_values(ft_df)
    bundle: FeatureBundle = assemble_features(
        pool, sequence_by_gene, args.esm_model,
        processed_dir=args.feature_cache_dir, device=device,
        features_mode=args.features, batch_size=args.extract_batch_size,
        overwrite_cache=args.overwrite_cache,
        fixed_prior_columns=fixed_prior_cols,
        fixed_impute_values=fixed_impute)
    meta = bundle.meta
    y_all = meta["label"].astype(int).to_numpy()
    is_holdout = (meta["gene"] == holdout).to_numpy()
    X_esm, X_prior = bundle.X_esm, bundle.X_prior
    d_esm = 0 if X_esm is None else X_esm.shape[1]

    # --- Warm-start schema validation ------------------------------------- #
    init_state = None
    if ckpt is not None:
        cfg = ckpt.get("config", {})
        if int(cfg.get("esm_dim", -1)) != d_esm:
            raise ValueError(
                f"Checkpoint esm_dim {cfg.get('esm_dim')} != current {d_esm}; "
                "backbone/feature mode changed since pretraining.")
        if list(ckpt.get("feature_columns") or []) != bundle.prior_cols:
            missing = set(ckpt.get("feature_columns") or []) - set(bundle.prior_cols)
            raise ValueError(
                f"Checkpoint prior-column schema mismatch (missing {sorted(missing)[:5]}); "
                "rebuild the MMR table or re-pretrain.")
        init_state = ckpt["model_state_dict"]

    groups = (meta["uniprot_id"].astype(str) + ":"
              + meta["position"].astype(str)).to_numpy()
    positions = meta["position"].to_numpy()
    ft_idx_all = np.flatnonzero(~is_holdout)
    inner_tr_local, inner_va_local = make_position_group_folds(
        positions[ft_idx_all], y_all[ft_idx_all], k_folds=5, seed=args.seed,
        groups=groups[ft_idx_all])[0]
    tr_idx = ft_idx_all[inner_tr_local]
    va_idx = ft_idx_all[inner_va_local]
    ho_idx = np.flatnonzero(is_holdout)
    weights_all = stage_sample_weights(meta, args.clinical_weight, 1.0)

    # --- Architecture benchmark -------------------------------------------- #
    # ('row_name', matrices_per_branch, kind, warm_start?)
    X_full = None
    if X_esm is not None:
        X_full = np.concatenate([X_esm, X_prior], axis=1).astype(np.float32)
    specs: list[tuple[str, list[np.ndarray], str, bool]] = []
    if X_esm is not None:
        specs += [
            ("esm_only", [X_esm], "esm", False),
            ("priors_only", [X_prior], "priors", False),
            ("fused_pretrained", [X_full], "esm", init_state is not None),
            ("concat_fusion", [X_esm, X_prior], "concat", False),
            ("gatewave_fusion", [X_esm, X_prior], "gatewave", False),
        ]
    else:
        specs += [("priors_only", [X_prior], "priors", init_state is not None)]

    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for name, mats, kind, warm in specs:
        t_arch = time.time()
        this_init = None
        if warm and init_state is not None:
            this_init = init_state
        elif warm and init_state is None:
            logger.info("%s: no checkpoint available; training from scratch.", name)

        dims = [m.shape[1] for m in mats]
        model = build_model(kind, dims=dims, hidden_dim=args.hidden_dim,
                            dropout=args.dropout)
        if this_init is not None:
            try:
                model.load_state_dict(this_init)
            except RuntimeError as exc:
                raise ValueError(
                    f"Checkpoint incompatible with '{name}' architecture: {exc}"
                ) from exc

        scaled_tr, scaled_va, scaled_ho = scale_views(mats, tr_idx, va_idx, ho_idx)
        model, best_epoch = fit_head(
            model, scaled_tr, y_all[tr_idx], scaled_va, y_all[va_idx],
            sample_weights=weights_all[tr_idx],
            lr=args.finetune_lr if this_init is not None else args.scratch_lr,
            epochs=args.epochs, patience=args.patience, weight_decay=1e-2,
            batch_size=args.batch_size, device=device,
            seed=args.seed + ARCH_SEED_OFFSET[kind])

        val_probs = expit_np(predict_logits(model, scaled_va, device))
        ho_probs = expit_np(predict_logits(model, scaled_ho, device))
        rep = evaluate_predictions(y_all[ho_idx], ho_probs,
                                   y_all[va_idx], val_probs,
                                   args.n_bootstrap, args.seed)
        row = {
            "holdout_gene": holdout,
            "arch": name,
            "warm_started": bool(this_init is not None),
            "lr": args.finetune_lr if this_init is not None else args.scratch_lr,
            "n_finetune": int(len(tr_idx)),
            "n_inner_val": int(len(va_idx)),
            "n_holdout": int(len(ho_idx)),
            "best_epoch": int(best_epoch),
            "runtime_s": round(time.time() - t_arch, 1),
            **rep,
        }
        rows.append(row)
        # Persist the per-variant probabilities, not just the aggregates.
        # Calibration (PROJECT_PLAN.md Phase 6) needs the raw held-out scores
        # per variant, and no downstream stage can reconstruct them from a
        # metrics row. Both partitions are kept and tagged: the held-out gene
        # is what gets calibrated and reported, while the inner-validation
        # rows are the only label-bearing scores a calibrator may be *fitted*
        # on without touching the evaluation set.
        for split_name, idx, probs in (("holdout", ho_idx, ho_probs),
                                       ("inner_val", va_idx, val_probs)):
            block = meta.iloc[idx][
                ["gene", "position", "wt_aa", "mut_aa"]].copy()
            block["label"] = y_all[idx]
            block["prob"] = probs
            block["split"] = split_name
            block["holdout_gene"] = holdout
            block["arch"] = name
            prediction_rows.append(block)
        logger.info("[holdout=%s | %-17s] ROC-AUC %.3f [%.3f-%.3f] | PR-AUC %.3f "
                    "| MCC %.3f @thr %.3f",
                    holdout, name, rep["roc_auc"], rep["roc_auc_ci_low"],
                    rep["roc_auc_ci_high"], rep["pr_auc"], rep["mcc"],
                    rep["threshold"])
    return rows, prediction_rows


def expit_np(x: np.ndarray) -> np.ndarray:
    from scipy.special import expit
    return expit(x)


# --------------------------------------------------------------------------- #
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

    if not args.scratch:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --scratch is set.")
        ckpt = load_checkpoint(args.checkpoint)
        cfg = ckpt.get("config", {})
        mode_ckpt = cfg.get("features")
        if mode_ckpt and mode_ckpt != args.features:
            raise SystemExit(
                f"--features {args.features} != checkpoint mode '{mode_ckpt}'.")
        if cfg.get("esm_model") and cfg["esm_model"] != args.esm_model:
            raise SystemExit(
                f"--esm_model {args.esm_model} != checkpoint backbone "
                f"'{cfg['esm_model']}'. Extraction caches would be inconsistent.")
    else:
        ckpt = None
        logger.info("Scratch training requested; no checkpoint loaded.")

    master = pd.read_csv(args.mmr_csv, low_memory=False)
    # Feature-representation transforms run on the FULL table -- every variant
    # of every gene, VUS included -- before any split. That is what makes the
    # per-gene rank reference population identical at training and inference
    # time; ranking only the labelled rows would shift the scale between them.
    if args.rank_normalize != "off":
        raw_score_cols = rankable_score_columns(master)
        master = add_within_gene_rank_features(master, raw_score_cols)
        if args.rank_normalize == "replace":
            master = master.drop(columns=raw_score_cols)
            logger.info("rank_normalize=replace: dropped %d raw score columns.",
                        len(raw_score_cols))
    drop_gene_constant = (args.gene_constant_priors == "drop"
                          or (args.gene_constant_priors == "auto"
                              and args.eval == "lopo"))
    if drop_gene_constant:
        drop = [c for c in GENE_CONSTANT_PRIOR_COLS if c in master.columns]
        if drop:
            master = master.drop(columns=drop)
            logger.info("Dropped %d gene-constant constraint columns (%s): "
                        "they encode gene identity, not variant evidence, "
                        "under a cross-gene split.", len(drop), drop)
    panel_path = Path(args.panel_json)
    sequence_by_gene: dict[str, str] = {}
    if args.features == "esm+priors":
        if not panel_path.exists():
            raise SystemExit(
                f"{panel_path} not found — run scripts/build_mmr_dataset.py first.")
        panel = json.loads(panel_path.read_text())
        sequence_by_gene = {g.upper(): d["sequence"] for g, d in panel.items()}

    if args.eval == "holdout":
        if args.holdout_gene is None:
            raise SystemExit("--holdout_gene required with --eval holdout.")
        splits = [args.holdout_gene]
    else:
        splits = list(MMR_GENES)

    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    evaluated: list[str] = []
    skipped: list[str] = []
    for holdout in splits:
        if holdout not in set(master["gene"].unique()):
            logger.warning("Gene %s absent from the Phase-1 table; skipping.",
                           holdout)
            skipped.append(holdout)
            continue
        evaluated.append(holdout)
        split_rows, split_preds = run_one_split(
            args, master, sequence_by_gene, device, holdout, ckpt)
        all_rows.extend(split_rows)
        all_predictions.extend(split_preds)
    if skipped:
        logger.warning("Evaluated %d/%d requested splits; %s had no rows in "
                       "the Phase-1 table (PMS2 is absent whenever the dataset "
                       "was built with --exclude_pms2).",
                       len(evaluated), len(splits), skipped)

    results = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "lopo" if args.eval == "lopo" else f"holdout_{args.holdout_gene}"
    results_path = args.out_dir / f"mmr_transfer_results_{tag}.csv"
    results.to_csv(results_path, index=False)
    predictions_path = args.out_dir / f"mmr_transfer_predictions_{tag}.csv"
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            predictions_path, index=False)
        logger.info("Per-variant predictions -> %s", predictions_path)
    else:
        predictions_path = None

    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "scratch": bool(args.scratch),
        "features": args.features,
        "rank_normalize": args.rank_normalize,
        "gene_constant_priors": args.gene_constant_priors,
        "gene_constant_priors_dropped": bool(drop_gene_constant),
        "esm_model": args.esm_model,
        "n_bootstrap": args.n_bootstrap,
        # What actually produced a row, not what was asked for. Reporting the
        # requested list here claimed PMS2 had been evaluated on every
        # --exclude_pms2 build, where it is skipped above. Mirrors the
        # scripts/finetune_esm_mmr.py summary schema.
        "splits_requested": splits,
        "splits_evaluated": evaluated,
        "splits_skipped_no_rows": skipped,
        "results_csv": str(results_path),
        "predictions_csv": str(predictions_path) if all_predictions else None,
        "runtime_s": round(time.time() - t0, 1),
    }
    (args.out_dir / f"mmr_transfer_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2))

    print("\n================ LEAVE-ONE-GENE-OUT RESULTS =========================")
    cols = ["holdout_gene", "arch", "warm_started", "roc_auc", "roc_auc_ci_low",
            "roc_auc_ci_high", "pr_auc", "mcc", "threshold", "n_holdout"]
    print(results[[c for c in cols if c in results.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    print(f"Results -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
