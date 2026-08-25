#!/usr/bin/env python3
"""Train the pathogenicity MLP head on the *extended* multi-source dataset.

Unlike ``main.py`` (ClinVar supervision only), this trainer consumes the full
audited master table ``data/processed/extended/extended_dataset_train.csv``
(ClinVar + ProteinGym-clinical + single-assay DMS labels) and optionally the
AlphaMissense / zero-shot / DMS / domain prior columns as extra features.

Feature modes
-------------
priors      External priors only (AlphaMissense, DMS aggregates, zs_* models,
            in_domain).  Trains in minutes on any CPU.
esm+priors  ESM-2 embeddings + PLLR stacked with the same priors.  This is the
            full model; use a CUDA GPU for anything beyond a few proteins
            (~1 forward pass per unique mutant sequence).

Evaluation contract
-------------------
* StratifiedGroupKFold on ``uniprot_id:position`` groups -> no residue leaks.
* StandardScaler fitted per training fold only.
* Pooled OOF metrics are reported twice: over ALL labelled rows, and over the
  clinical-only slice (ClinVar or PG-clinical sourced labels), since DMS-bin
  labels measure assay fitness rather than clinical consequence.
* Baseline comparison: AlphaMissense, inverted DMS score, published zs_ models.

Examples
--------
    python scripts/train_extended.py                          # priors, CPU OK
    python scripts/train_extended.py --features esm+priors \
        --esm_model facebook/esm2_t33_650M_UR50D              # full model, GPU
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.esm_extractor import extract_features_cached, get_device  # noqa: E402
from src.train import TrainConfig, cross_validate, evaluate_and_export  # noqa: E402
from src.train import predict_vus, set_global_seed  # noqa: E402

logger = logging.getLogger("train_extended")

EXT_DIR = PROJECT_ROOT / "data" / "processed" / "extended"

PRIOR_COLS = [
    "am_pathogenicity",
    "dms_score_median", "dms_bin_median", "n_dms_assays",
    "in_domain",
]
ZS_PREFIX = "zs_"
#: Columns copied into OOF / VUS outputs for traceability.
META_COLS = ["gene", "uniprot_id", "position", "wt_aa", "mut_aa", "hgvs_p"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-validated residual-MLP training on the extended "
                    "variant-pathogenicity dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features", choices=["priors", "esm+priors"],
                   default="priors",
                   help="'priors' trains on external scores only (fast CPU); "
                        "'esm+priors' adds ESM-2 embeddings + PLLR.")
    p.add_argument("--train_csv", type=Path,
                   default=EXT_DIR / "extended_dataset_train.csv",
                   help="Labelled master table (audit-clean).")
    p.add_argument("--full_csv", type=Path,
                   default=EXT_DIR / "extended_dataset.csv",
                   help="Full master table; mined for ClinVar VUS rows when "
                        "--score_vus.")
    p.add_argument("--esm_model", type=str,
                   default="facebook/esm2_t12_35M_UR50D",
                   help="ESM-2 checkpoint for --features esm+priors. Use "
                        "facebook/esm2_t33_650M_UR50D on an NVIDIA GPU.")
    p.add_argument("--extract_batch_size", type=int, default=8,
                   help="Mini-batch size for ESM-2 forward passes (keep small "
                        "on GPUs; independent of the training batch size).")
    p.add_argument("--genes", type=str, default=None,
                   help="Optional comma-separated subset of gene symbols.")
    p.add_argument("--no-score-vus", action="store_true",
                   help="Skip prospective scoring of ClinVar VUS variants.")
    p.add_argument("--no_dms_features", action="store_true",
                   help="Exclude DMS-derived features (dms_bin_median equals "
                        "the flipped label on single-assay rows -> circular). "
                        "Use for leakage-clean evaluation.")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight_decay", type=float, default=1e-2)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--n_blocks", type=int, default=1,
                   help="Number of residual MLP blocks.")
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--loss_type", choices=["bce", "focal", "wbce"], default="bce",
                   help="Use BCE by default; focal/WBCE are alternatives to tune.")
    t.add_argument("--focal_gamma", type=float, default=2.0)
    t.add_argument("--focal_alpha", type=float, default=0.5)
    t.add_argument("--clinical_weight", type=float, default=5.0,
                   help="Relative loss weight for ClinVar/PG-clinical labels.")
    t.add_argument("--dms_weight", type=float, default=1.0,
                   help="Relative loss weight for DMS-only labels.")
    t.add_argument("--k_folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)

    p.add_argument("--overwrite_cache", action="store_true",
                   help="Recompute cached ESM-2 feature blocks.")
    return p.parse_args(argv)


def build_prior_matrix(df: pd.DataFrame, drop_dms: bool = False) -> np.ndarray:
    """Numeric prior matrix with median imputation (external info only).

    ``drop_dms`` removes DMS-derived columns (dms_score_median,
    dms_bin_median, n_dms_assays).  Use it when evaluating generalisation:
    on single-assay rows ``dms_bin_median`` equals the flipped label by
    construction, so keeping it makes the all-labels metric circular.
    """
    cols = [c for c in df.columns
            if c in PRIOR_COLS or c.startswith(ZS_PREFIX)]
    if drop_dms:
        dropped = [c for c in cols if c.startswith("dms_")
                   or c == "n_dms_assays"]
        cols = [c for c in cols if c not in dropped]
        logger.info("Dropping %d DMS-derived features to avoid label "
                    "circularity: %s", len(dropped), dropped)
    mat = df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    # dms_score_median: invert so that larger = more pathogenic-like.
    if "dms_score_median" in mat:
        med = mat["dms_score_median"]
        mat["dms_score_median"] = -med.where(med.notna())
    # Missing published scores are highly structured (most zs_* values cover
    # only a small subset of proteins).  Preserve that information instead of
    # making an absent score indistinguishable from a median-valued score.
    missing = mat.isna().astype(np.float64)
    missing.columns = [f"is_missing_{c}" for c in mat.columns]
    med = mat.median(skipna=True)
    mat = pd.concat([mat.fillna(med).fillna(0.0), missing], axis=1)
    logger.info("Prior features: %d values + %d missingness indicators (%s)", len(cols), len(cols),
                ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else ""))
    return mat.to_numpy(dtype=np.float32)


def esm_features_for_panel(
    df: pd.DataFrame, panel: dict, args, device, processed_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-protein ESM-2 + PLLR extraction, cached; returns aligned matrix."""
    blocks: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []
    genes = sorted(df["gene"].unique())
    for gi, gene in enumerate(genes, 1):
        sub = df[df["gene"] == gene].sort_values(["position", "mut_aa"])
        seq = panel[gene]["sequence"]
        extras = pd.DataFrame(build_prior_matrix(sub), index=sub.index)
        try:
            feats, meta = extract_features_cached(
                sub.reset_index(drop=True), seq, gene=gene,
                model_name=args.esm_model, processed_dir=processed_dir,
                batch_size=args.extract_batch_size, device=device,
                overwrite=args.overwrite_cache, extra_features=extras)
        except RuntimeError as exc:
            logger.warning("[%s] cache mismatch (%s); recomputing.", gene, exc)
            feats, meta = extract_features_cached(
                sub.reset_index(drop=True), seq, gene=gene,
                model_name=args.esm_model, processed_dir=processed_dir,
                batch_size=args.extract_batch_size, device=device,
                overwrite=True, extra_features=extras)
        logger.info("[%d/%d] %s: %d variant feature vectors (d=%d)",
                    gi, len(genes), gene, len(feats), feats.shape[1])
        blocks.append(feats.astype(np.float32))
        metas.append(meta)
    return np.vstack(blocks), pd.concat(metas, ignore_index=True)


def slice_report(oof: pd.DataFrame, prob_cols: dict, mask: np.ndarray,
                 label: str) -> pd.DataFrame:
    """Pooled metric table restricted to a boolean slice of OOF rows."""
    from src.calibration import full_report
    rows = []
    y = oof.loc[mask, "label"].to_numpy()
    for name, col in prob_cols.items():
        rep = full_report(y, oof.loc[mask, col].to_numpy())
        rows.append({"slice": label, "score": name,
                     "n": int(mask.sum()), **rep})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    set_global_seed(args.seed)
    device = get_device()

    train = pd.read_csv(args.train_csv, low_memory=False)
    train["label"] = pd.to_numeric(train["label"], errors="coerce").astype("Int64")
    train = train[train["label"].notna()].reset_index(drop=True)
    if args.genes:
        keep = {g.strip().upper() for g in args.genes.split(",") if g.strip()}
        train = train[train["gene"].isin(keep)].reset_index(drop=True)
    n_pos = int((train["label"] == 1).sum())
    n_neg = int((train["label"] == 0).sum())
    logger.info("Training rows: %d (%d P/LP, %d B/LB) across %d proteins",
                len(train), n_pos, n_neg, train["gene"].nunique())

    clinical_mask = (train["clinvar_label"].notna()
                     | train["clinical_label"].notna()).to_numpy()
    clinical_keys = set(map(
        tuple, train.loc[clinical_mask, ["gene", "position", "wt_aa", "mut_aa"]]
        .itertuples(index=False, name=None)))

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        dropout=args.dropout, n_blocks=args.n_blocks, loss_type=args.loss_type,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        patience=args.patience, k_folds=args.k_folds, seed=args.seed,
    )

    out_dir = PROJECT_ROOT / "data" / "processed" / "extended_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "ext80" if not args.genes else "ext" + str(len(train["gene"].unique()))

    vus_feats = vus_meta = None
    if args.features == "esm+priors":
        import json
        panel = json.loads(
            (PROJECT_ROOT / "data" / "raw" / "uniprot" /
             "expanded_panel.json").read_text())
        # Keep raw priors in the extraction pool. Previously only metadata
        # columns were retained here, so ``esm+priors`` silently appended zero
        # prior features despite its name.
        prior_input_cols = [c for c in train.columns
                            if c in PRIOR_COLS or c.startswith(ZS_PREFIX)]
        quality_cols = [c for c in ("label_source", "label_weight") if c in train.columns]
        pool_cols = META_COLS + ["label"] + quality_cols + prior_input_cols
        pool = train[pool_cols].copy()
        if not args.no_score_vus:
            full = pd.read_csv(args.full_csv, low_memory=False)
            vus_rows = full[full["label"].isna()
                            & full["review_status"].notna()
                            & (full["gene"].isin(panel.keys()))]
            logger.info("Adding %d ClinVar VUS variants to extraction pool "
                        "(for prospective scoring).", len(vus_rows))
            pool = pd.concat([pool, vus_rows.reindex(columns=pool_cols)],
                             ignore_index=True)
            pool = pool.drop_duplicates(
                subset=["gene", "position", "wt_aa", "mut_aa"])
        all_feats, all_meta = esm_features_for_panel(pool, panel, args,
                                                     device,
                                                     PROJECT_ROOT / "data" / "processed")
        key_cols = ["gene", "position", "wt_aa", "mut_aa"]
        train_keys = set(map(tuple, train[key_cols].itertuples(index=False,
                                                               name=None)))
        is_train = np.array([
            (g, int(p), w, m) in train_keys
            for g, p, w, m in zip(all_meta["gene"], all_meta["position"],
                                  all_meta["wt_aa"], all_meta["mut_aa"])])
        features = all_feats[is_train]
        meta = all_meta.loc[is_train].reset_index(drop=True)
        if (~is_train).any():
            vus_feats = all_feats[~is_train]
            vus_meta = all_meta.loc[~is_train].reset_index(drop=True)
        pllr = meta["pllr"].to_numpy(dtype=np.float64)
        baseline_name = "pllr"
    else:
        features = build_prior_matrix(train, drop_dms=args.no_dms_features)
        meta = train.copy()
        # No PLLR without ESM: use logit(AlphaMissense) as the calibration
        # baseline so reliability diagrams still have a reference curve.
        am = meta["am_pathogenicity"].fillna(
            meta["am_pathogenicity"].median()).clip(1e-6, 1 - 1e-6)
        pllr = np.log(am / (1 - am)).to_numpy()
        baseline_name = "alphamissense_logit"
        meta["pllr"] = pllr

    group_keys = (meta["uniprot_id"].astype(str) + ":"
                  + meta["position"].astype(str)).to_numpy()
    labels = meta["label"].astype(int).to_numpy()
    positions = meta["position"].to_numpy()
    aligned_clinical_mask = np.array([
        (g, int(p), w, m) in clinical_keys
        for g, p, w, m in zip(meta["gene"], meta["position"],
                              meta["wt_aa"], meta["mut_aa"])
    ])
    sample_weights = np.where(aligned_clinical_mask,
                              args.clinical_weight, args.dms_weight).astype(np.float32)
    if "label_weight" in meta.columns:
        quality = pd.to_numeric(meta["label_weight"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        # Source weights express evidence quality; clinical/DMS weights express
        # the experiment's target preference. Both should influence the loss.
        sample_weights *= quality
    logger.info("Source-aware loss weights: clinical=%.2f (%d rows), DMS-only=%.2f (%d rows)",
                args.clinical_weight, int(aligned_clinical_mask.sum()), args.dms_weight,
                int((~aligned_clinical_mask).sum()))

    # Baseline columns ride through CV metadata so they land inside the OOF
    # frame (validation rows only) for the slice comparison below.
    ride_cols = [c for c in META_COLS + ["pllr", "am_pathogenicity",
                                         "dms_score_median"]
                 if c in meta.columns]
    logger.info("Starting %d-fold group-disjoint CV (%s mode, d=%d) ...",
                cfg.k_folds, args.features, features.shape[1])
    oof, metric_rows, artifacts = cross_validate(
        features, labels, positions, pllr,
        meta_extra=meta[ride_cols],
        cfg=cfg, device=device, groups=group_keys,
        sample_weights=sample_weights, clinical_mask=aligned_clinical_mask)

    paths = evaluate_and_export(oof, metric_rows, out_dir, tag, cfg)

    # ---- Slice + baseline report ----------------------------------------- #
    oof_is_clinical = np.array([
        (g, int(p), w, m) in clinical_keys
        for g, p, w, m in zip(oof["gene"], oof["position"],
                              oof["wt_aa"], oof["mut_aa"])])
    prob_cols = {
        "MLP_uncalibrated": "prob_uncalibrated",
        "MLP_temperature": "prob_temperature",
        "MLP_isotonic": "prob_isotonic",
        baseline_name: "prob_pllr",
        "AlphaMissense": "prob_am",
        "-DMS_score": "prob_negdms",
    }
    am = oof["am_pathogenicity"].fillna(0.5).clip(1e-6, 1 - 1e-6)
    oof["prob_am"] = am.to_numpy()
    dms = pd.to_numeric(oof.get("dms_score_median"), errors="coerce")
    neg_dms = -dms.fillna(dms.median())
    rng = float(neg_dms.max() - neg_dms.min()) or 1.0
    oof["prob_negdms"] = ((neg_dms - neg_dms.min()) / rng).to_numpy()

    slices = [
        slice_report(oof, prob_cols, np.ones(len(oof), dtype=bool), "all_labels"),
        slice_report(oof, prob_cols, oof_is_clinical, "clinical_only"),
    ]
    slice_df = pd.concat(slices, ignore_index=True)
    slice_path = out_dir / f"{tag}_slice_baselines.csv"
    slice_df.to_csv(slice_path, index=False)
    paths["slice_baselines"] = str(slice_path)

    show = ["slice", "score", "n", "roc_auc", "pr_auc", "mcc", "brier", "ece_uniform"]
    print("\n============ POOLED SLICE + BASELINE COMPARISON ======================")
    print(slice_df[[c for c in show if c in slice_df.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)

    oof_path = out_dir / f"{tag}_oof_with_baselines.csv"
    oof.to_csv(oof_path, index=False)
    paths["oof_with_baselines"] = str(oof_path)

    if vus_feats is not None and vus_meta is not None and len(vus_meta):
        vus_pred = predict_vus(vus_feats, vus_meta, artifacts, cfg, device)
        vus_path = out_dir / f"{tag}_vus_predictions.csv"
        vus_pred.to_csv(vus_path, index=False)
        paths["vus_predictions"] = str(vus_path)
        logger.info("VUS predictions -> %s (%d rows)",
                    vus_path, len(vus_pred))

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    for art in artifacts:
        _torch.save({
            "model_state_dict": art["model"].state_dict(),
            "input_dim": features.shape[1],
            "scaler_mean": art["scaler"].mean_,
            "scaler_scale": art["scaler"].scale_,
            "temperature": art["temperature"].state_dict_for_export()["temperature"],
            "best_epoch": art["best_epoch"],
            "config": cfg.to_dict(),
        }, ckpt_dir / f"{tag}_fold{art['fold']}.pt")
    paths["checkpoints_dir"] = str(ckpt_dir)

    print("\n=========================== ARTIFACTS ================================")
    for k, v in sorted(paths.items()):
        print(f"  {k:<24} {v}")
    print(f"  {'total_runtime':<24} {time.time() - t0:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
