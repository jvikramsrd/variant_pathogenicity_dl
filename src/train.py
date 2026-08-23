"""Training loop, 5-fold cross-validation, calibration benchmarking and VUS inference.

Pipeline per outer fold
-----------------------
1. Position-grouped stratified outer split (no residue appears in both sides).
2. The outer-training partition is further split by residue group into model,
   early-stopping, and calibration partitions.  The outer validation partition
   is never used for model selection or calibration.
3. ``StandardScaler`` fitted on the model-training partition only.
4. Residual MLP trained with AdamW + CosineAnnealingLR; early stopping on the
   inner validation ROC-AUC with best-weight restoration.
5. Post-hoc calibrators are fitted on the independent inner calibration
   partition and evaluated only on outer validation predictions.
6. All fold predictions and metrics are pooled into summary CSVs; reliability
   diagrams compare uncalibrated / temperature / isotonic / PLLR-baseline.

Finally the fold ensemble scores the held-out ClinVar VUS set and exports risk
predictions with calibrated confidence percentiles.

This costs some fitting data per fold, but makes reported calibration and
threshold metrics valid out-of-fold estimates.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .calibration import (
    IsotonicCalibrator,
    TemperatureScaling,
    expit,
    full_report,
    plot_reliability_diagrams,
)
from .dataset import VariantDataset, make_position_group_folds
from .loss import build_loss
from .model import VariantPathogenicityMLP

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyper-parameters for cross-validation training."""

    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    hidden_dim: int = 512
    dropout: float = 0.3
    n_blocks: int = 1
    loss_type: str = "bce"            # {"bce", "focal", "wbce"}
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    patience: int = 8                 # early stopping on inner val ROC-AUC
    k_folds: int = 5
    seed: int = 42
    threshold: float = 0.5            # decision threshold for MCC / F1 / bal-acc
    fixed_bins: int = 15              # ECE fixed-width bin count
    adaptive_bins: int = 10           # ECE equal-mass bin count

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def set_global_seed(seed: int) -> None:
    """Seed python / numpy / torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Core train / predict helpers
# --------------------------------------------------------------------------- #
def predict_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device,
                   batch_size: int = 1024) -> np.ndarray:
    """Batched no-grad logit computation for a feature matrix."""
    model.eval()
    outputs: List[np.ndarray] = []
    tensor_X = torch.as_tensor(X, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(tensor_X), batch_size):
            chunk = tensor_X[start:start + batch_size].to(device)
            outputs.append(model(chunk).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def _train_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    fold_seed: int,
    sample_weights: Optional[np.ndarray] = None,
    monitor_mask: Optional[np.ndarray] = None,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[VariantPathogenicityMLP, int]:
    """Train one fold; restore best weights by inner validation ROC-AUC."""
    set_global_seed(fold_seed)

    model = VariantPathogenicityMLP(
        input_dim=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        n_blocks=cfg.n_blocks,
    ).to(device)
    if init_state is not None:
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Pretrained checkpoint is incompatible with this model: "
                f"missing={missing}, unexpected={unexpected}")

    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = max(1, int(np.sum(y_train == 0)))
    criterion = build_loss(cfg.loss_type, cfg.focal_gamma, cfg.focal_alpha,
                           pos_weight=n_neg / n_pos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_loader = torch.utils.data.DataLoader(
        VariantDataset(X_train, y_train, sample_weights), batch_size=cfg.batch_size, shuffle=True,
        drop_last=False,
    )
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = np.asarray(y_val, dtype=np.float32)

    best_auc, best_state, best_epoch, patience_left = -np.inf, None, -1, cfg.patience
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb, wb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(xb)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val_t).cpu().numpy()
        try:
            active = np.asarray(monitor_mask, dtype=bool) if monitor_mask is not None else np.ones(len(y_val_np), dtype=bool)
            # Do not try to monitor on a one-class clinical subset.
            if np.unique(y_val_np[active]).size < 2:
                active = np.ones(len(y_val_np), dtype=bool)
            val_auc = float(roc_auc_score(y_val_np[active], expit(val_logits)[active]))
        except ValueError:
            val_auc = float("nan")  # single-class validation fold
        if not np.isfinite(val_auc):
            val_auc = -np.inf

        logger.debug("fold seed %d epoch %d: loss=%.4f val_auc=%.4f",
                     fold_seed, epoch + 1, running / max(1, len(train_loader.dataset)), val_auc)

        if val_auc > best_auc:
            best_auc, best_epoch, patience_left = val_auc, epoch + 1, cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("  early stopping at epoch %d (best AUC %.4f @ epoch %d)",
                            epoch + 1, best_auc, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_epoch, 0)


# --------------------------------------------------------------------------- #
# Cross-validation driver
# --------------------------------------------------------------------------- #
SCORE_VARIANTS: Tuple[str, ...] = ("uncalibrated", "temperature", "isotonic", "pllr")


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    pllr: np.ndarray,
    meta_extra: Optional[pd.DataFrame] = None,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    groups: Optional[Sequence[str]] = None,
    sample_weights: Optional[np.ndarray] = None,
    clinical_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]]]:
    """Run position-grouped K-fold CV; return OOF predictions + metrics + artifacts.

    Parameters
    ----------
    groups:
        Optional explicit leakage-group keys (e.g. ``"P04637:273"``) overriding
        raw *positions*; required when pooling multiple proteins so that equal
        position numbers in different sequences do not collapse into one group.

    Returns
    -------
    oof_df:
        One row per labelled variant with fold id, metadata columns, raw logit
        and uncalibrated / temperature-calibrated / isotonic-calibrated /
        PLLR-baseline probabilities.
    metric_rows:
        Per-(fold, score variant) discrimination + calibration reports.
    artifacts:
        Per-fold dicts holding the trained model, scaler and calibrators (used
        later for VUS ensembling).
    """
    cfg = cfg or TrainConfig()
    device = device or torch.device("cpu")
    labels = np.asarray(labels)
    positions = np.asarray(positions)
    group_keys = np.asarray(groups) if groups is not None else positions
    sample_weights = (np.ones(len(labels), dtype=np.float32) if sample_weights is None
                      else np.asarray(sample_weights, dtype=np.float32))
    clinical_mask = (np.zeros(len(labels), dtype=bool) if clinical_mask is None
                     else np.asarray(clinical_mask, dtype=bool))

    folds = make_position_group_folds(positions, labels, cfg.k_folds, cfg.seed,
                                      groups=groups)
    oof_frames: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
    artifacts: List[Dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        # Reserve two disjoint inner group folds: one for early stopping and
        # one for calibration.  The outer validation fold remains untouched.
        inner_groups = group_keys[train_idx]
        inner_splits = make_position_group_folds(
            positions[train_idx], labels[train_idx], k_folds=5,
            seed=cfg.seed + 10_000 + fold_idx, groups=inner_groups)
        early_local = inner_splits[0][1]
        calibration_local = inner_splits[1][1]
        fit_mask = np.ones(len(train_idx), dtype=bool)
        fit_mask[early_local] = False
        fit_mask[calibration_local] = False
        fit_local = np.flatnonzero(fit_mask)
        fit_idx = train_idx[fit_local]
        early_idx = train_idx[early_local]
        calibration_idx = train_idx[calibration_local]

        scaler = StandardScaler().fit(features[fit_idx])
        X_tr = scaler.transform(features[fit_idx]).astype(np.float32)
        X_early = scaler.transform(features[early_idx]).astype(np.float32)
        X_va = scaler.transform(features[val_idx]).astype(np.float32)

        model, best_epoch = _train_single_fold(
            X_tr, labels[fit_idx], X_early, labels[early_idx], cfg, device,
            fold_seed=cfg.seed + fold_idx, sample_weights=sample_weights[fit_idx],
            monitor_mask=clinical_mask[early_idx],
        )

        val_logits = predict_logits(model, X_va, device)
        probs_uncal = expit(val_logits)
        calibration_logits = predict_logits(
            model, scaler.transform(features[calibration_idx]).astype(np.float32), device)
        calibration_selector = clinical_mask[calibration_idx]
        if (calibration_selector.sum() < 20
                or np.unique(labels[calibration_idx][calibration_selector]).size < 2):
            calibration_selector = np.ones(len(calibration_idx), dtype=bool)
        cal_logits = calibration_logits[calibration_selector]
        cal_labels = labels[calibration_idx][calibration_selector]
        temp = TemperatureScaling().fit(torch.tensor(cal_logits), cal_labels)
        probs_temp = temp.predict_proba(torch.tensor(val_logits))
        iso = IsotonicCalibrator().fit(expit(cal_logits), cal_labels)
        probs_iso = iso.transform(probs_uncal)
        probs_pllr = expit(np.asarray(pllr)[val_idx])

        frame = pd.DataFrame({
            "fold": fold_idx,
            "position": positions[val_idx],
            "label": labels[val_idx],
            "pllr": np.asarray(pllr)[val_idx],
            "logit": val_logits,
            "prob_uncalibrated": probs_uncal,
            "prob_temperature": probs_temp,
            "prob_isotonic": probs_iso,
            "prob_pllr": probs_pllr,
        })
        if meta_extra is not None:
            for col in meta_extra.columns:
                if col not in frame.columns:
                    frame[col] = meta_extra.iloc[val_idx][col].to_numpy()
        oof_frames.append(frame)

        prob_map = {
            "uncalibrated": probs_uncal,
            "temperature": probs_temp,
            "isotonic": probs_iso,
            "pllr": probs_pllr,
        }
        for name in SCORE_VARIANTS:
            report = full_report(
                labels[val_idx], prob_map[name],
                threshold=cfg.threshold, fixed_bins=cfg.fixed_bins,
                adaptive_bins=cfg.adaptive_bins,
            )
            metric_rows.append({"fold": fold_idx, "score_variant": name, **report})

        artifacts.append({
            "fold": fold_idx,
            "model": model,
            "scaler": scaler,
            "temperature": temp,
            "isotonic": iso,
            "best_epoch": best_epoch,
            "n_fit": int(len(fit_idx)),
            "n_early_stop": int(len(early_idx)),
            "n_calibration": int(len(calibration_idx)),
            "val_roc_auc": float(
                next(r["roc_auc"] for r in metric_rows[-len(SCORE_VARIANTS):]
                     if r["score_variant"] == "uncalibrated")
            ),
        })
        logger.info(
            "fold %d/%d done in %.1fs | best_epoch=%d | val ROC-AUC=%.4f | T=%.3f",
            fold_idx + 1, len(folds), time.time() - t0, best_epoch,
            artifacts[-1]["val_roc_auc"], temp.temperature,
        )

    oof_df = pd.concat(oof_frames, ignore_index=True)
    return oof_df, metric_rows, artifacts


# --------------------------------------------------------------------------- #
# Evaluation & export
# --------------------------------------------------------------------------- #
def _format_metric_table(metric_rows: List[Dict[str, object]]) -> str:
    """Pretty 'mean ± std' table across folds for every score variant."""
    df = pd.DataFrame(metric_rows)
    value_cols = ["roc_auc", "pr_auc", "mcc", "balanced_accuracy", "f1",
                  "ece_uniform", "ece_adaptive", "mce", "brier"]
    value_cols = [c for c in value_cols if c in df.columns]
    rows = []
    for name, grp in df.groupby("score_variant"):
        row: Dict[str, str] = {"score_variant": str(name)}
        for col in value_cols:
            mean = pd.to_numeric(grp[col], errors="coerce").mean()
            std = pd.to_numeric(grp[col], errors="coerce").std(ddof=1)
            row[col] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("score_variant")
    return table.to_string()


def evaluate_and_export(
    oof_df: pd.DataFrame,
    metric_rows: List[Dict[str, object]],
    out_dir: Path,
    gene: str,
    cfg: TrainConfig,
) -> Dict[str, str]:
    """Write prediction/metric CSVs, plot reliability diagrams, print tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    pred_path = out_dir / f"{gene}_oof_predictions.csv"
    oof_df.to_csv(pred_path, index=False)
    paths["oof_predictions"] = str(pred_path)

    fold_metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / f"{gene}_fold_metrics.csv"
    fold_metrics.to_csv(metrics_path, index=False)
    paths["fold_metrics"] = str(metrics_path)

    # Pooled (out-of-fold aggregate) comparison across all variants.
    pooled_rows: List[Dict[str, object]] = []
    prob_col = {
        "uncalibrated": "prob_uncalibrated",
        "temperature": "prob_temperature",
        "isotonic": "prob_isotonic",
        "pllr": "prob_pllr",
    }
    for name in SCORE_VARIANTS:
        report = full_report(
            oof_df["label"].to_numpy(), oof_df[prob_col[name]].to_numpy(),
            threshold=cfg.threshold, fixed_bins=cfg.fixed_bins,
            adaptive_bins=cfg.adaptive_bins,
        )
        pooled_rows.append({"fold": "pooled", "score_variant": name, **report})
    pooled_path = out_dir / f"{gene}_pooled_metrics.csv"
    pd.DataFrame(pooled_rows).to_csv(pooled_path, index=False)
    paths["pooled_metrics"] = str(pooled_path)

    fig_path = out_dir / f"{gene}_reliability_diagram.png"
    plot_reliability_diagrams(
        series={
            "Uncalibrated MLP": oof_df["prob_uncalibrated"],
            "Temperature scaling": oof_df["prob_temperature"],
            "Isotonic regression": oof_df["prob_isotonic"],
            "PLLR baseline": oof_df["prob_pllr"],
        },
        y_true=oof_df["label"].to_numpy(),
        out_path=str(fig_path),
        n_bins=min(cfg.fixed_bins, 10),
        title=f"{gene.upper()} — reliability of pathogenicity scores",
    )
    paths["reliability_diagram"] = str(fig_path)

    print("\n================ FOLD METRICS (mean ± std over folds) ================")
    print(_format_metric_table(metric_rows))
    print("\n===================== POOLED OOF COMPARISON =========================")
    pooled_display = pd.DataFrame(pooled_rows).drop(columns=["fold"]).set_index("score_variant")
    print(pooled_display.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 74)
    return paths


# --------------------------------------------------------------------------- #
# VUS inference
# --------------------------------------------------------------------------- #
def predict_vus(
    vus_features: np.ndarray,
    vus_meta: pd.DataFrame,
    artifacts: List[Dict[str, object]],
    cfg: TrainConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Ensemble-score the held-out VUS set with calibrated confidence percentiles.

    Each fold's scaler/model/temperature pipeline produces one calibrated
    probability; we report their mean, the inter-fold standard deviation, the
    percentile within the VUS cohort and a coarse risk tier.
    """
    fold_probs: List[np.ndarray] = []
    for art in artifacts:
        scaled = art["scaler"].transform(vus_features).astype(np.float32)
        logits = predict_logits(art["model"], scaled, device, cfg.batch_size)
        fold_probs.append(art["temperature"].predict_proba(torch.tensor(logits)))
    stack = np.vstack(fold_probs)
    mean_prob = stack.mean(axis=0)
    std_prob = stack.std(axis=0)

    percentile = rankdata(mean_prob, method="average") / len(mean_prob) * 100.0
    tier = np.where(percentile >= 90.0, "High",
                    np.where(percentile >= 70.0, "Moderate", "Low"))

    out = vus_meta.reset_index(drop=True).copy()
    out["mean_calibrated_prob"] = mean_prob
    out["fold_std"] = std_prob
    out["confidence_percentile"] = percentile
    out["risk_tier"] = tier
    return out


# --------------------------------------------------------------------------- #
# End-to-end runner used by main.py
# --------------------------------------------------------------------------- #
def run_pipeline(
    labeled_features: np.ndarray,
    labeled_meta: pd.DataFrame,
    vus_features: Optional[np.ndarray],
    vus_meta: Optional[pd.DataFrame],
    gene: str,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, str]:
    """Execute CV training, evaluation/export and optional VUS inference."""
    labels = labeled_meta["label"].astype(int).to_numpy()
    positions = labeled_meta["position"].to_numpy()
    pllr = labeled_meta["pllr"].to_numpy()

    extra_cols = [c for c in ["gene", "uniprot_id", "wt_aa", "mut_aa", "hgvs_p", "review_status"]
                  if c in labeled_meta.columns]

    # Leakage groups: one key per (protein, residue) so multi-gene pooling
    # never lets equal position numbers in different proteins mix folds.
    if "uniprot_id" in labeled_meta.columns:
        group_keys = (
            labeled_meta["uniprot_id"].astype(str) + ":"
            + labeled_meta["position"].astype(str)
        ).to_numpy()
    else:
        group_keys = None

    logger.info("Starting %d-fold position-grouped cross-validation ...", cfg.k_folds)
    oof_df, metric_rows, artifacts = cross_validate(
        labeled_features, labels, positions, pllr,
        meta_extra=labeled_meta[extra_cols],
        cfg=cfg, device=device, groups=group_keys,
    )
    paths = evaluate_and_export(oof_df, metric_rows, out_dir, gene, cfg)

    # Persist fold artifacts for reproducible re-inference.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for art in artifacts:
        ckpt_path = ckpt_dir / f"{gene}_fold{art['fold']}.pt"
        torch.save({
            "model_state_dict": art["model"].state_dict(),
            "input_dim": labeled_features.shape[1],
            "scaler_mean": art["scaler"].mean_,
            "scaler_scale": art["scaler"].scale_,
            "temperature": art["temperature"].state_dict_for_export()["temperature"],
            "best_epoch": art["best_epoch"],
            "config": cfg.to_dict(),
        }, ckpt_path)
    paths["checkpoints_dir"] = str(ckpt_dir)

    if vus_features is not None and vus_meta is not None and len(vus_meta):
        logger.info("Scoring %d held-out VUS variants ...", len(vus_meta))
        vus_pred = predict_vus(vus_features, vus_meta, artifacts, cfg, device)
        vus_path = out_dir / f"{gene}_vus_predictions.csv"
        vus_pred.to_csv(vus_path, index=False)
        paths["vus_predictions"] = str(vus_path)
        top = vus_pred.sort_values("confidence_percentile", ascending=False).head(15)
        cols = [c for c in ["hgvs_p", "position", "mut_aa", "mean_calibrated_prob",
                            "confidence_percentile", "risk_tier", "fold_std"] if c in top.columns]
        print("\n=============== TOP-15 HIGHEST-RISK VUS PREDICTIONS ==================")
        print(top[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("=" * 74)
    else:
        logger.info("No VUS variants available; skipping prospective inference.")

    return paths
