#!/usr/bin/env python3
"""Overfitting / leakage diagnostic for the extended-dataset MLP head.

Configs
-------
no_dms   : AlphaMissense + zs_* published models + in_domain  (DMS features removed)
am_only  : AlphaMissense single feature

For each config: group-disjoint CV (uniprot:position), pooled OOF metrics over
all labels and the clinical-only slice, plus per-fold TRAIN vs VAL ROC-AUC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration import expit, full_report
from src.dataset import make_position_group_folds
from src.train import TrainConfig, cross_validate, predict_logits, set_global_seed
from src.esm_extractor import get_device

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "extended" / "extended_dataset_train.csv"


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    m = df[cols].apply(pd.to_numeric, errors="coerce")
    return m.fillna(m.median()).fillna(0.0).to_numpy(np.float32)


def main() -> int:
    set_global_seed(42)
    device = get_device()
    df = pd.read_csv(TRAIN_CSV, low_memory=False)
    df["label"] = df["label"].astype(int)
    clinical_mask = (df["clinvar_label"].notna() | df["clinical_label"].notna()).to_numpy()

    zs = [c for c in df.columns if c.startswith("zs_")]
    configs = {
        "no_dms": ["am_pathogenicity", "in_domain"] + zs,
        "am_only": ["am_pathogenicity"],
    }
    cfg = TrainConfig(epochs=15, patience=4, k_folds=3, batch_size=256)

    for name, cols in configs.items():
        feats = feature_matrix(df, cols)
        groups = (df["uniprot_id"].astype(str) + ":" + df["position"].astype(str)).to_numpy()
        oof, metric_rows, artifacts = cross_validate(
            feats, df["label"].to_numpy(), df["position"].to_numpy(),
            np.zeros(len(df)), meta_extra=None, cfg=cfg, device=device,
            groups=groups)

        # Pooled OOF: score every row with its own fold's model
        # (each row is out-of-fold exactly once).
        folds = make_position_group_folds(
            df["position"].to_numpy(), df["label"].to_numpy(),
            cfg.k_folds, cfg.seed, groups=groups)
        gaps = []
        for art, (tr_idx, va_idx) in zip(artifacts, folds):
            X_tr = art["scaler"].transform(feats[tr_idx]).astype(np.float32)
            tr_auc = roc_auc_score(df["label"].to_numpy()[tr_idx],
                                   expit(predict_logits(art["model"], X_tr, device)))
            gaps.append(tr_auc - art["val_roc_auc"])
            print(f"[{name}] fold {art['fold']}: TRAIN AUC {tr_auc:.4f} "
                  f"| VAL AUC {art['val_roc_auc']:.4f} | gap {tr_auc - art['val_roc_auc']:+.4f}")
        print(f"[{name}] mean train-val AUC gap: {np.mean(gaps):+.4f}")

        oof_prob = np.full(len(df), np.nan)
        for art, (tr_idx, va_idx) in zip(artifacts, folds):
            X_va = art["scaler"].transform(feats[va_idx]).astype(np.float32)
            oof_prob[va_idx] = expit(predict_logits(art["model"], X_va, device))
        assert not np.isnan(oof_prob).any()

        am = df["am_pathogenicity"].fillna(0.5).clip(1e-6, 1 - 1e-6).to_numpy()
        for label, mask in [("all_labels", np.ones(len(df), bool)),
                            ("clinical_only", clinical_mask)]:
            r_mlp = full_report(df["label"].to_numpy()[mask], oof_prob[mask])
            r_am = full_report(df["label"].to_numpy()[mask], am[mask])
            print(f"[{name}] {label}: MLP ROC-AUC {r_mlp['roc_auc']:.4f} "
                  f"MCC {r_mlp['mcc']:.4f} | AM ROC-AUC {r_am['roc_auc']:.4f} "
                  f"MCC {r_am['mcc']:.4f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
