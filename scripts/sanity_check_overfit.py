#!/usr/bin/env python3
"""Tiny synthetic-data sanity check: can the training loop overfit a trivial,
perfectly-separable dataset?

Not run in this environment (no working Python install with torch/pandas/
sklearn exists on this dev box; see docs/RUNLOG.md). This is a portable
implementation-correctness smoke test for the other PC -- it says nothing
about real-data generalization, only about whether the model/optimizer/loss
wiring in :func:`src.transfer.build_model` / :func:`src.transfer.fit_head`
can drive train loss down and train ROC-AUC up at all. If it can't reach
near-1.0 AUC on data that is perfectly separable by construction, something
in that wiring (a wrong loss sign, a frozen parameter, a broadcasting bug,
tensors on the wrong device) is broken -- fix that before trusting *any*
result produced by the same code path on real data.

Generates and uses ONLY an in-memory numpy array (``numpy.random.default_rng``);
never reads or writes anything under ``data/``.

Usage
-----
    python scripts/sanity_check_overfit.py
    python scripts/sanity_check_overfit.py --seed 7 --epochs 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.calibration import expit  # noqa: E402
from src.transfer import build_model, fit_head, predict_logits  # noqa: E402


def make_synthetic(n: int, d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """A linearly-separable-by-construction binary classification set.

    Label = 1 iff the projection onto a fixed random direction exceeds the
    training set's own median -- exactly balanced, and separable by a linear
    head with zero label noise, so nothing about the *data* should prevent
    perfect training-set fit.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal(d).astype(np.float32)
    projection = X @ w_true
    y = (projection > np.median(projection)).astype(np.float32)
    return X, y


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", choices=("priors", "esm"), default="priors",
                  help="Both resolve to the same single-view BranchHead in "
                       "src.transfer.build_model; kept as a choice only for "
                       "parity with that function's API.")
    p.add_argument("--n", type=int, default=200, help="Synthetic sample count.")
    p.add_argument("--dims", type=int, default=8, help="Synthetic feature count.")
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_train_auc", type=float, default=0.98,
                  help="Fail if training-set ROC-AUC falls below this.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device("cpu")  # a synthetic sanity check never needs a GPU
    X, y = make_synthetic(args.n, args.dims, args.seed)

    model = build_model(args.arch, dims=[args.dims], hidden_dim=args.hidden_dim,
                        dropout=0.0)
    # Train and "validation" are deliberately the SAME synthetic array: this
    # checks whether the loop can overfit at all, not whether it generalizes
    # -- a held-out split belongs in a real-data test, not here.
    model, best_epoch = fit_head(
        model, [X], y, [X], y, sample_weights=None,
        lr=1e-2, epochs=args.epochs, patience=args.epochs, weight_decay=0.0,
        batch_size=max(1, args.n // 4), device=device, seed=args.seed)

    logits = predict_logits(model, [X], device)
    if not np.isfinite(logits).all():
        print("FAIL: model produced non-finite logits (NaN/Inf) on synthetic data.")
        return 1
    proba = expit(logits)
    auc = float(roc_auc_score(y, proba))
    print(f"best_epoch={best_epoch} train_roc_auc={auc:.4f}")
    if auc < args.min_train_auc:
        print(f"FAIL: train ROC-AUC {auc:.4f} < required {args.min_train_auc} "
             "on a perfectly-separable synthetic dataset -- the "
             "model/optimizer/loss wiring is likely broken.")
        return 1
    print("PASS: the training loop can overfit trivial synthetic data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
