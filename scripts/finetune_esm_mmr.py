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

    # ProPath recipe, full backbone fine-tune, 16 GB consumer GPU.
    # A full siamese fine-tune of esm2_t33_650M runs two 1024-token forward
    # passes through 33 layers per example and keeps every activation for the
    # backward pass; at --batch_size 8 in fp32 that is ~41 GB of activations
    # on top of ~10 GB of weights/gradients/AdamW state. The micro-batch,
    # accumulation, AMP and checkpointing flags below are what make it fit --
    # the effective batch is still ProPath's 8.
    python scripts/finetune_esm_mmr.py --mode siamese --n_unfrozen_layers -1 \
        --esm_model facebook/esm2_t33_650M_UR50D --eval lopo \
        --backbone_lr 1e-5 --batch_size 1 --grad_accum 8 --epochs 10 \
        --gradient_checkpointing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
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
    save_finetuned,
)
from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc  # noqa: E402
from src.finetune_grid import GridCell, output_tag  # noqa: E402
from src.transfer import (  # noqa: E402
    assert_af_quarantine,
    prior_columns_of,
    prior_impute_values,
    prior_matrix,
)

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
    p.add_argument("--use_pllr", action=argparse.BooleanOptionalAction, default=True,
                   help="Feed the zero-shot PLLR term "
                        "log P(mut|X) - log P(wt|X), read off the same "
                        "wild-type forward pass, into the classification head. "
                        "On its own that score reaches ROC-AUC 0.834 pooled "
                        "across these genes with no training at all, so "
                        "including it lets the head learn a residual instead "
                        "of rediscovering it from a few hundred labels. "
                        "Use --no-use_pllr for the ablation.")
    p.add_argument("--branch", choices=("esm", "esm+priors"), default="esm",
                   help="'esm+priors' fuses the leakage-safe prior columns "
                        "into the fine-tune head. The frozen Stage-2 probe "
                        "reads 27 such columns; without this flag Stage 2b "
                        "reads none, so the two are not comparable.")
    p.add_argument("--fusion", choices=("concat", "gatewave"), default="concat")
    p.add_argument("--pllr_mode", choices=("residual", "concat"), default="residual",
                   help="How the zero-shot term enters the head. 'residual' "
                        "makes the untrained model the zero-shot predictor, "
                        "so training learns a correction on top of it; "
                        "'concat' is the historical extra-input-dimension "
                        "behaviour. Ignored when --no-use_pllr is passed.")
    p.add_argument("--af_labels_active", action="store_true",
                   help="Declare that allele-frequency-derived labels are in "
                        "the training pool. Forces the AF-derived feature "
                        "columns out of the feature set -- labelling and "
                        "featuring on frequency at once makes acmg_bs1 equal "
                        "the label by construction.")
    p.add_argument("--cell_slug", type=str, default=None,
                   help="Tag for this cell's output files. Defaults to a slug "
                        "derived from the configuration.")
    t = p.add_argument_group("training (ProPath defaults)")
    t.add_argument("--backbone_lr", type=float, default=1e-5)
    t.add_argument("--head_lr", type=float, default=3e-4)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--batch_size", type=int, default=8,
                   help="Micro-batch that must fit in VRAM. On a 16 GB card a "
                        "full siamese fine-tune of esm2_t33_650M needs 1-2; "
                        "raise --grad_accum to keep the effective batch at 8.")
    t.add_argument("--grad_accum", type=int, default=1,
                   help="Accumulate this many micro-batches per optimizer "
                        "step. Effective batch = batch_size * grad_accum, so "
                        "--batch_size 2 --grad_accum 4 trains at ProPath's "
                        "batch of 8 using a quarter of the activation memory.")
    t.add_argument("--no_amp", dest="amp", action="store_false",
                   help="Disable mixed-precision autocast (bf16 where "
                        "supported, else fp16). AMP roughly halves activation "
                        "memory and is on by default on CUDA.")
    t.set_defaults(amp=True)
    t.add_argument("--hidden_dim", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--weight_decay", type=float, default=1e-2)
    t.add_argument("--clinical_weight", type=float, default=5.0)
    e = p.add_argument_group("evaluation")
    e.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Persist the fine-tuned model per split so the "
                        "backbone weights survive the run (default: on). "
                        "fit_esm_finetune restores the best epoch in memory "
                        "only -- without this the fine-tuned backbone is gone "
                        "on exit and just the metrics CSV remains. A "
                        "full-unfreeze esm2_t33_650M state dict is ~2.6 GiB "
                        "per gene (~10 GiB for a 4-gene leave-one-gene-out "
                        "sweep); pass --no-save_checkpoints to skip it.")
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


@dataclass
class PriorInputs:
    """Standardised prior matrices for the three partitions of one split."""

    train: np.ndarray
    val: np.ndarray
    holdout: np.ndarray
    columns: list
    impute_values: dict
    mean: np.ndarray
    scale: np.ndarray


def build_prior_inputs(ft_df: pd.DataFrame, ho_df: pd.DataFrame,
                       tr_idx, va_idx, *, drop_gene_constant: bool,
                       af_labels_active: bool) -> PriorInputs:
    """Prior features for the fine-tune, inner-val and holdout partitions.

    Imputation medians and standardisation constants are fitted on the
    **fine-tune training rows only** and applied unchanged to inner-val and
    holdout: fitting them per partition would centre each split differently
    and hand the head shifted inputs (the same defect ``docs/PAPER.md``
    Finding 5 records for the Stage-1/Stage-2 imputation constants).

    *drop_gene_constant* must be true for leave-one-gene-out. The five gnomAD
    gene-level columns hold one value per gene, so across genes they are a
    5-dimensional gene identifier rather than evidence -- ``RUNLOG.md``
    2026-08-28 records MLH1 collapsing to ROC-AUC 0.500 in every seed with
    them kept.
    """
    columns = prior_columns_of(ft_df, drop_gene_constant=drop_gene_constant)
    assert_af_quarantine(columns, af_labels_active)
    if not columns:
        raise ValueError("no prior columns found in the fine-tune table")

    train_rows = ft_df.iloc[list(tr_idx)]
    impute = prior_impute_values(train_rows, columns)
    x_tr, full_cols = prior_matrix(train_rows, columns=columns,
                                   impute_values=impute)
    # Pin the *returned* order (values then derived missingness flags) for the
    # other two partitions: a different order would silently permute the
    # head's inputs at evaluation time.
    x_va, _ = prior_matrix(ft_df.iloc[list(va_idx)], columns=full_cols,
                           impute_values=impute)
    x_ho, _ = prior_matrix(ho_df, columns=full_cols, impute_values=impute)

    mean = x_tr.mean(axis=0)
    scale = x_tr.std(axis=0)
    scale[scale < 1e-8] = 1.0          # constant column -> leave it centred

    def to_std(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / scale).astype(np.float32)

    return PriorInputs(train=to_std(x_tr), val=to_std(x_va), holdout=to_std(x_ho),
                       columns=list(full_cols), impute_values=dict(impute),
                       mean=mean, scale=scale)


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

    priors = None
    if args.branch == "esm+priors":
        priors = build_prior_inputs(
            ft_df, ho_df, tr_local, va_local,
            drop_gene_constant=(args.eval == "lopo"),
            af_labels_active=args.af_labels_active)
        logger.info("Prior branch: %d columns (gene-constant dropped=%s)",
                    len(priors.columns), args.eval == "lopo")

    pllr_mode = args.pllr_mode if args.use_pllr else "off"
    model = ESMFineTuneClassifier(
        model_name=args.esm_model, mode=args.mode,
        n_unfrozen_layers=args.n_unfrozen_layers, hidden_dim=args.hidden_dim,
        dropout=args.dropout, gradient_checkpointing=args.gradient_checkpointing,
        pllr_mode=pllr_mode,
        n_prior_features=0 if priors is None else priors.train.shape[1],
        fusion=args.fusion)

    # The tokenizer is required so each example carries its wild-type and
    # substituted residue vocabulary ids; without them the PLLR term would
    # silently read the logit of token 0.
    tok = model.tokenizer
    train_ex = build_examples(ft_df.iloc[tr_local], sequence_by_gene, args.mode,
                              max_residues=args.max_residues, tokenizer=tok,
                              prior_matrix=None if priors is None else priors.train)
    val_ex = build_examples(ft_df.iloc[va_local], sequence_by_gene, args.mode,
                            max_residues=args.max_residues, tokenizer=tok,
                            prior_matrix=None if priors is None else priors.val)
    ho_ex = build_examples(ho_df, sequence_by_gene, args.mode,
                           max_residues=args.max_residues, tokenizer=tok,
                           prior_matrix=None if priors is None else priors.holdout)
    logger.info("Fine-tune train=%d val=%d holdout=%d", len(train_ex), len(val_ex), len(ho_ex))

    t0 = time.time()
    model, best_epoch, best_val_auc = fit_esm_finetune(
        model, train_ex, val_ex, device,
        backbone_lr=args.backbone_lr, head_lr=args.head_lr, epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size,
        weight_decay=args.weight_decay, seed=args.seed,
        grad_accum_steps=args.grad_accum, amp=args.amp)

    # Eval reuses the training micro-batch rather than a larger one: the
    # optimizer state is still resident here, so a bigger eval batch would
    # push peak VRAM above what training itself needed.
    val_probs = predict_proba(model, val_ex, device, batch_size=args.batch_size, amp=args.amp)
    ho_probs = predict_proba(model, ho_ex, device, batch_size=args.batch_size, amp=args.amp)
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

    # Rows failing wt-validation never became examples, so recover the keys of
    # the ones that did rather than assuming ho_df and ho_probs line up.
    kept = ho_df.iloc[[e.row_index for e in ho_ex]].reset_index(drop=True)
    key_cols = [c for c in ("gene", "uniprot_id", "position", "wt_aa", "mut_aa")
                if c in kept.columns]
    predictions = kept[key_cols].copy()
    predictions.insert(0, "holdout_gene", holdout)
    predictions["label"] = ho_labels
    predictions["prob"] = ho_probs
    predictions["threshold"] = float(thr)
    predictions["seed"] = args.seed
    predictions["cell_slug"] = args.cell_slug
    predictions["branch"] = args.branch
    predictions["n_unfrozen_layers"] = args.n_unfrozen_layers
    predictions["pllr_mode"] = pllr_mode

    # The inner-validation probabilities are computed above to select `thr` and
    # were previously discarded, which made every post-hoc calibration step
    # impossible: temperature scaling, isotonic recalibration and re-selecting a
    # threshold for a seed-ensembled score all need a held-out-from-training
    # split that is *not* the holdout gene. Fitting any of those on the holdout
    # is the leak this project forbids, so without these rows the only way to
    # recalibrate was to retrain. They are small (one fold of the fine-tune
    # genes) and cost nothing to keep.
    val_meta = ft_df.iloc[va_local].reset_index(drop=True)
    val_kept = val_meta.iloc[[e.row_index for e in val_ex]].reset_index(drop=True)
    val_predictions = val_kept[[c for c in key_cols if c in val_kept.columns]].copy()
    val_predictions.insert(0, "holdout_gene", holdout)
    val_predictions["label"] = val_labels
    val_predictions["prob"] = val_probs
    val_predictions["threshold"] = float(thr)
    val_predictions["seed"] = args.seed
    val_predictions["cell_slug"] = args.cell_slug
    val_predictions["branch"] = args.branch
    val_predictions["n_unfrozen_layers"] = args.n_unfrozen_layers
    val_predictions["pllr_mode"] = pllr_mode

    row = {
        "holdout_gene": holdout, "mode": args.mode, "esm_model": args.esm_model,
        "n_unfrozen_layers": args.n_unfrozen_layers,
        "use_pllr": bool(args.use_pllr),
        "branch": args.branch, "fusion": args.fusion, "pllr_mode": pllr_mode,
        "cell_slug": args.cell_slug,
        "n_prior_features": 0 if priors is None else priors.train.shape[1],
        "seed": args.seed,
        "_predictions": predictions,
        "_val_predictions": val_predictions,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum, "amp": args.amp,
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
        save_finetuned(
            ckpt_path, model,
            threshold=float(thr), max_residues=args.max_residues,
            metrics={k: row[k] for k in
                     ("roc_auc", "pr_auc", "mcc", "best_epoch",
                      "inner_val_roc_auc", "n_holdout") if k in row},
            extra={"holdout_gene": holdout, "seed": args.seed,
                   "built_at_utc": datetime.now(timezone.utc).isoformat()})
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
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        logger.info("CUDA device: %s (%.1f GiB)",
                    torch.cuda.get_device_name(0), total_gb)
        logger.info("Memory plan: micro-batch %d x %d accum = effective batch "
                    "%d | amp=%s | gradient_checkpointing=%s | max_residues=%d",
                    args.batch_size, args.grad_accum,
                    args.batch_size * args.grad_accum, args.amp,
                    args.gradient_checkpointing, args.max_residues)
    device = get_device()

    if args.cell_slug is None:
        args.cell_slug = GridCell(
            branch=args.branch, n_unfrozen_layers=args.n_unfrozen_layers,
            pllr_mode=(args.pllr_mode if args.use_pllr else "off"),
            seed=args.seed, fusion=args.fusion).slug()

    if not args.mmr_csv.exists():
        raise SystemExit(f"{args.mmr_csv} not found -- run scripts/build_mmr_dataset.py first.")
    master = pd.read_csv(args.mmr_csv, low_memory=False)
    panel = json.loads(Path(args.panel_json).read_text())
    sequence_by_gene = {g.upper(): d["sequence"] for g, d in panel.items()}

    splits = [args.holdout_gene] if args.eval == "holdout" else list(MMR_GENES)
    if args.eval == "holdout" and args.holdout_gene is None:
        raise SystemExit("--holdout_gene required with --eval holdout.")

    # One split at a time, releasing the previous split's backbone and AdamW
    # state before building the next: the caching allocator would otherwise
    # carry that fragmentation across all four leave-one-gene-out models.
    rows = []
    evaluated, skipped = [], []
    available = set(master["gene"].unique())
    for g in splits:
        if g not in available:
            # Most often PMS2: run_mmr_pipeline.py passes --exclude_pms2 to the
            # dataset builder by default (PMS2CL pseudogene homology makes
            # short-read clinical calls unreliable), so the gene never reaches
            # this table. Say so -- a silently 3-row "leave-one-gene-out"
            # result is indistinguishable from a crashed fourth split.
            skipped.append(g)
            continue
        rows.append(run_one_split(args, master, sequence_by_gene, device, g))
        evaluated.append(g)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if skipped:
        logger.warning("No rows in %s for %s -- excluded from this "
                       "leave-one-gene-out run (%d/%d genes evaluated). For "
                       "PMS2 this is run_mmr_pipeline.py's --exclude_pms2 "
                       "default; pass --pms2_homology_csv or "
                       "--pms2_codon_range to include it.",
                       args.mmr_csv.name, ", ".join(skipped),
                       len(evaluated), len(splits))

    predictions = (pd.concat([r.pop("_predictions") for r in rows],
                             ignore_index=True) if rows else pd.DataFrame())
    val_predictions = (pd.concat([r.pop("_val_predictions") for r in rows],
                                 ignore_index=True) if rows else pd.DataFrame())
    results = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Every filename carries the cell slug: a grid sweep writes a dozen of
    # these into one directory, and an untagged name would make each run
    # silently overwrite the last.
    tag = output_tag(args.mode, args.eval, args.cell_slug,
                     args.holdout_gene)
    results_path = args.out_dir / f"esm_finetune_results_{tag}.csv"
    predictions_path = args.out_dir / f"esm_finetune_predictions_{tag}.csv"
    val_predictions_path = args.out_dir / f"esm_finetune_valpreds_{tag}.csv"
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    val_predictions.to_csv(val_predictions_path, index=False)
    checkpoints = [r["checkpoint"] for r in rows if r.get("checkpoint")]
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell_slug, "branch": args.branch, "fusion": args.fusion,
        "pllr_mode": args.pllr_mode if args.use_pllr else "off",
        "seed": args.seed, "af_labels_active": bool(args.af_labels_active),
        "mode": args.mode, "esm_model": args.esm_model,
        "n_unfrozen_layers": args.n_unfrozen_layers,
        "splits_requested": splits, "splits_evaluated": evaluated,
        "splits_skipped_no_rows": skipped, "results_csv": str(results_path),
        "predictions_csv": str(predictions_path),
        "val_predictions_csv": str(val_predictions_path),
        "checkpoints": checkpoints,
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
    if checkpoints:
        print(f"Fine-tuned models -> {len(checkpoints)} checkpoint(s) in {args.out_dir}/")
        for c in checkpoints:
            print(f"  {c}")
    elif not args.save_checkpoints:
        print("Fine-tuned models -> not saved (--no-save_checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
