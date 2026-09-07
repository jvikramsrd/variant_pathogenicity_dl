"""Post-hoc calibration of the Stage-2b grid, fitted only on inner validation.

``MISSING_EVIDENCE.md`` item 5: every reported arm carries ECE 0.19-0.21 and none
is calibrated. ``src/calibration.py`` has had ``TemperatureScaling`` and
``IsotonicCalibrator`` throughout; what was missing was a leak-free split to fit
them on. The reported runs computed inner-validation probabilities to select a
threshold and then discarded them, and the only other split is the held-out gene
-- fitting there is the leak the protocol exists to forbid. Commit ``e919bef``
made ``finetune_esm_mmr.py`` keep them as ``esm_finetune_valpreds_{tag}.csv``, so
this driver has something honest to fit.

Per cell and per held-out gene:

1. fit the calibrator on that fold's inner-validation rows only;
2. re-select the decision threshold on **calibrated** validation scores;
3. score the held-out gene and report the full panel.

Step 2 is not optional. A threshold chosen on the uncalibrated scale does not
survive a monotone transform of the scores: applying it after calibration moves
every probability under a cut that was fitted to a different scale, which can
leave accuracy worse while ECE improves. That is the calibration equivalent of
correcting the units and leaving the number.

**One fold, one calibrator.** Each leave-one-gene-out fold is a different model
with its own inner-validation set. A calibrator pooled across folds would be
fitted partly on models it is not applied to, which is a subtler version of the
same leak.

**The artifacts store probabilities, not logits.** Temperature scaling needs
logits, so it fits on ``log(p / (1 - p))`` recovered from the stored
probability. The stored values are float32, so a recovered logit is good to
about +/-0.02 at the extremes -- immaterial for a one-parameter fit, but it is a
recovery rather than a record, and the paper should say so rather than imply the
run emitted logits.

**Isotonic is reported but should be read with suspicion here.** It is fitted
and then evaluated on the same inner-validation rows to select the threshold,
so its validation calibration is optimistic by construction; with folds this
small (inner validation is roughly 95 variants) a non-parametric fit has room to
memorise. Temperature scaling has one parameter and does not.

    python scripts/recalibrate_grid.py --grid_dir data/processed/stage2b_grid
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import IsotonicCalibrator, TemperatureScaling  # noqa: E402
from src.metrics import evaluation_report  # noqa: E402

logger = logging.getLogger(__name__)

#: Probabilities are clipped before the logit so a saturated score cannot become
#: an infinite one. 1e-6 sits below the smallest probability observed in the
#: committed artifacts (3.7e-06), so it clips nothing that was actually written.
_EPS = 1e-6

METHODS = ("uncalibrated", "temperature", "isotonic")


def logit(prob: Sequence[float]) -> np.ndarray:
    """Recover logits from stored probabilities, clipped away from 0 and 1."""
    p = np.clip(np.asarray(prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _fittable(val_true: np.ndarray, min_per_class: int = 2) -> Tuple[bool, str]:
    """Whether a fold's inner validation can support a calibrator at all."""
    if val_true.size == 0:
        return False, "no inner-validation rows"
    pos, neg = int((val_true == 1).sum()), int((val_true == 0).sum())
    if pos < min_per_class or neg < min_per_class:
        return False, f"inner validation has {pos} positive / {neg} negative"
    return True, ""


def calibrate_fold(
    val_true: Sequence[int],
    val_prob: Sequence[float],
    ho_true: Sequence[int],
    ho_prob: Sequence[float],
    method: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """Calibrated ``(val_prob, ho_prob)`` for one fold, plus fitted parameters.

    Returns ``(None, None, {...})`` when the fold cannot support the method.
    The calibrator sees ``val_*`` only: ``ho_true`` is passed for nothing but
    shape, and is deliberately unused, so that a future edit that reaches for it
    is visible as a change to this signature.
    """
    # Copies, not views: a pandas-backed array arrives read-only and torch
    # warns on every fit when it is handed one.
    v_true = np.array(val_true, dtype=np.int64)
    v_prob = np.array(val_prob, dtype=np.float64)
    h_prob = np.array(ho_prob, dtype=np.float64)

    if method == "uncalibrated":
        return v_prob, h_prob, {}

    ok, reason = _fittable(v_true)
    if not ok:
        return None, None, {"skip_reason": reason}

    if method == "temperature":
        scaler = TemperatureScaling().fit(logit(v_prob), v_true)
        return (scaler.predict_proba(logit(v_prob)).astype(np.float64),
                scaler.predict_proba(logit(h_prob)).astype(np.float64),
                {"temperature": scaler.temperature})

    if method == "isotonic":
        iso = IsotonicCalibrator().fit(v_prob, v_true)
        return iso.transform(v_prob), iso.transform(h_prob), {}

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def panel_for_fold(
    cell_slug: str,
    holdout_gene: str,
    val: pd.DataFrame,
    ho: pd.DataFrame,
    method: str,
) -> Dict[str, object]:
    """One output row: the full reporting panel for *method* on this fold."""
    row: Dict[str, object] = {
        "cell_slug": cell_slug,
        "holdout_gene": holdout_gene,
        "method": method,
        "n_inner_val": int(len(val)),
        "n_inner_val_positive": int((val["label"].to_numpy() == 1).sum()),
    }

    cal_val, cal_ho, params = calibrate_fold(
        val["label"].to_numpy(), val["prob"].to_numpy(),
        ho["label"].to_numpy(), ho["prob"].to_numpy(), method)

    if cal_val is None:
        row["available"] = False
        row["unavailable_reason"] = params.get("skip_reason", "calibrator not fitted")
        return row

    row.update(params)
    # The threshold is selected inside evaluation_report from the validation
    # vectors handed to it -- so passing the *calibrated* validation scores is
    # what re-selects it on the calibrated scale.
    row.update(evaluation_report(
        y_true=ho["label"].to_numpy(),
        y_prob=cal_ho,
        val_y_true=val["label"].to_numpy(),
        val_y_prob=cal_val,
    ))
    return row


def tag_of(path: Path, prefix: str) -> str:
    """``esm_finetune_valpreds_siamese_lopo_x.csv`` -> ``siamese_lopo_x``."""
    return path.name[len(prefix):-len(".csv")]


def iter_folds(grid_dir: Path) -> Iterator[Tuple[str, str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(cell_slug, holdout_gene, inner_val, holdout)`` for every fold.

    Shared by the metrics panel and the per-variant export so the two cannot
    end up walking different sets of cells.
    """
    valpred_files = sorted(grid_dir.glob("esm_finetune_valpreds_*.csv"))
    if not valpred_files:
        raise FileNotFoundError(
            f"no esm_finetune_valpreds_*.csv in {grid_dir}. Runs predating commit "
            "e919bef did not write inner-validation predictions; those cells "
            "cannot be calibrated without re-running them.")

    for vp in valpred_files:
        tag = tag_of(vp, "esm_finetune_valpreds_")
        pred = grid_dir / f"esm_finetune_predictions_{tag}.csv"
        if not pred.exists():
            logger.warning("%s: no matching predictions CSV, skipping", tag)
            continue

        val_all, ho_all = pd.read_csv(vp), pd.read_csv(pred)
        for gene in sorted(ho_all["holdout_gene"].unique()):
            val = val_all[val_all["holdout_gene"] == gene]
            ho = ho_all[ho_all["holdout_gene"] == gene]
            if val.empty:
                logger.warning("%s/%s: no inner-validation rows", tag, gene)
                continue
            slug = str(ho["cell_slug"].iloc[0]) if "cell_slug" in ho else tag
            yield slug, gene, val, ho


def recalibrate_dir(
    grid_dir: Path, methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Every fold of every cell that has inner-validation predictions."""
    rows: List[Dict[str, object]] = []
    for slug, gene, val, ho in iter_folds(grid_dir):
        for method in methods:
            rows.append(panel_for_fold(slug, gene, val, ho, method))
    return pd.DataFrame(rows)


#: Columns that identify a variant, carried into the per-variant export so a
#: calibrated probability can be joined back to the master table.
_ID_COLUMNS = ["gene", "position", "wt_aa", "mut_aa", "label"]


def calibrated_probabilities(
    grid_dir: Path, methods: Sequence[str] = METHODS,
) -> pd.DataFrame:
    """Long table of every held-out probability under every method.

    Figure 4 needs the probability vectors themselves, not the summary panel:
    a reliability diagram is a statement about the distribution of scores, and
    recomputing calibration inside the figure script would let the picture
    drift from the table it illustrates. Only held-out rows are exported --
    inner validation is what the calibrator was fitted on, so its reliability
    is not a measurement.
    """
    frames: List[pd.DataFrame] = []
    for slug, gene, val, ho in iter_folds(grid_dir):
        for method in methods:
            cal_val, cal_ho, params = calibrate_fold(
                val["label"].to_numpy(), val["prob"].to_numpy(),
                ho["label"].to_numpy(), ho["prob"].to_numpy(), method)
            if cal_ho is None:
                continue
            frame = ho[[c for c in _ID_COLUMNS if c in ho.columns]].copy()
            frame["cell_slug"] = slug
            frame["holdout_gene"] = gene
            frame["method"] = method
            frame["prob"] = cal_ho
            frame["temperature"] = params.get("temperature", np.nan)
            frames.append(frame)

    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=_ID_COLUMNS + ["cell_slug", "holdout_gene",
                                                     "method", "prob"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid_dir", type=Path,
                   default=Path("data/processed/stage2b_grid"))
    p.add_argument("--out_csv", type=Path, default=None,
                   help="Default: <grid_dir>/calibration_panel.csv")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--out_probs", type=Path, default=None,
                   help="Per-variant calibrated held-out probabilities. "
                        "Default: <grid_dir>/calibration_probs.csv")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would be calibrated, write nothing.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S")

    panel = recalibrate_dir(args.grid_dir, args.methods)
    if panel.empty:
        logger.error("nothing calibrated")
        return 1

    scored = panel[panel.get("available", False) == True]  # noqa: E712
    logger.info("%d rows over %d cells and %d genes",
                len(panel), panel.cell_slug.nunique(), panel.holdout_gene.nunique())

    if not scored.empty and "ece_uniform" in scored:
        summary = scored.groupby("method")["ece_uniform"].mean().sort_values()
        for method, ece in summary.items():
            logger.info("  mean ECE  %-13s %.4f", method, ece)

    out = args.out_csv or (args.grid_dir / "calibration_panel.csv")
    out_probs = args.out_probs or (args.grid_dir / "calibration_probs.csv")
    if args.dry_run:
        logger.info("dry run: would write %s and %s", out, out_probs)
        return 0

    panel.to_csv(out, index=False)
    logger.info("wrote %s", out)

    probs = calibrated_probabilities(args.grid_dir, args.methods)
    probs.to_csv(out_probs, index=False)
    logger.info("wrote %s (%d rows)", out_probs, len(probs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
