"""Error analysis for the Stage-2b grid: what the model gets wrong, and whether
the mistakes share anything.

``MISSING_EVIDENCE.md`` item 9: §3.7 of the manuscript is a placeholder because
no script has ever joined a prediction back to the variant it was made about.
This one exports every held-out error, attaches the covariates the master table
carries, and reports which of them separate errors from correct calls.

Three decisions shape what comes out, and each is a claim about what an "error"
is worth reporting:

**Errors are taken per seed and intersected, not pooled.** A variant that one
seed of an arm gets wrong and two get right is a draw of the initialisation,
not a property of the variant -- and after the 2026-09-06 seeding fix
(``MISSING_EVIDENCE.md`` item 12) we know initialisation alone moves MLH1 by
0.021 AUROC. ``consensus`` counts how many seeds erred; ``--min_seeds`` selects
how many must agree before a row is called an error. For a single-seed arm the
distinction collapses and the column reads 1, which is the honest report of what
one draw can support.

**The threshold is the one that fold selected**, read from the predictions CSV,
applied per seed. Re-deriving a threshold here would score the model against a
cut it was never evaluated at.

**Model inputs and independent covariates are reported apart.** 27 of the
columns below are prior features the model reads (``prior_columns`` in the run
summary), so an association between one of them and the errors restates the
model's own weighting -- it cannot be evidence *about* the model. The columns
that are not inputs -- ClinVar review status, star rating, label source,
evidence tier, cross-source conflict -- are the ones where an association says
something the model did not already assume. The output marks every covariate
with ``is_model_input`` so a reader cannot mistake one for the other.

    python scripts/error_analysis.py
    python scripts/error_analysis.py --arm esmpri_concat_full_pllr-residual
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

GRID_DIR = ROOT / "data/processed/stage2b_grid"
MASTER_CSV = ROOT / "data/mmr/processed/extended/extended_dataset.csv"

#: The arm Figures 3 and 6 report, and the one §3.7 should describe.
DEFAULT_ARM = "esmpri_concat_frozen_pllr-residual"

#: (gene, position, wt_aa, mut_aa) identifies a variant across every artifact.
VARIANT_KEY = ["gene", "position", "wt_aa", "mut_aa"]

#: Covariates worth testing, and how each is compared. "binary" and "nominal"
#: are rate comparisons; "continuous" compares distributions.
COVARIATES: Dict[str, str] = {
    "am_pathogenicity": "continuous",
    "af_plddt": "continuous",
    "in_domain": "binary",
    "in_interpro_domain": "binary",
    "is_functional_site": "binary",
    "af_disordered": "binary",
    "gnomad_log10_af": "continuous",
    "zs_revel": "continuous",
    "zs_cadd": "continuous",
    "zs_gemme": "continuous",
    "stars": "continuous",
    "review_status": "nominal",
    "label_source": "nominal",
    "evidence_tier": "nominal",
    "cross_source_conflict": "binary",
    "label_conflict": "binary",
    "n_sources": "continuous",
}


def arm_of(slug: str) -> str:
    """``..._seed42`` -> ``...``. Matches ``make_figures.arm_of``."""
    return re.sub(r"_seed\d+$", "", str(slug))


def prior_columns(grid_dir: Path) -> List[str]:
    """The feature columns the model reads, from any run summary that has them.

    Read rather than hardcoded: the grid varies the feature set by arm, and a
    stale list here would mislabel a model input as an independent covariate --
    the one error this whole distinction exists to prevent.
    """
    for path in sorted(grid_dir.glob("esm_finetune_summary_*.json")):
        cols = json.loads(path.read_text()).get("prior_columns")
        if cols:
            return list(cols)
    logger.warning("no run summary carries prior_columns; "
                   "every covariate will be reported as not-an-input")
    return []


def load_arm(grid_dir: Path, arm: str) -> pd.DataFrame:
    """Held-out predictions for every seed of *arm*, one row per (variant, seed)."""
    frames = []
    for path in sorted(glob.glob(str(grid_dir / "esm_finetune_predictions_*.csv"))):
        d = pd.read_csv(path)
        if "cell_slug" not in d.columns or d.empty:
            continue
        if arm_of(d["cell_slug"].iloc[0]) != arm:
            continue
        frames.append(d)
    if not frames:
        raise FileNotFoundError(
            f"no predictions CSV in {grid_dir} belongs to arm {arm!r}. "
            f"Available: {sorted({arm_of(pd.read_csv(f, nrows=1).cell_slug.iloc[0]) for f in glob.glob(str(grid_dir / 'esm_finetune_predictions_*.csv')) if 'cell_slug' in pd.read_csv(f, nrows=1).columns})}")
    return pd.concat(frames, ignore_index=True)


def classify(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-variant error consensus over the seeds of one arm.

    Each seed is scored at the threshold that seed's fold selected, then the
    seeds are intersected. ``n_seeds`` is carried through so a consensus of 1
    out of 1 is never read as a consensus of 1 out of 3.
    """
    d = preds.copy()
    d["pred"] = (d["prob"] >= d["threshold"]).astype(int)
    d["wrong"] = (d["pred"] != d["label"]).astype(int)
    # Signed distance past the cut: how far the score is into the decision it
    # made. Large and wrong is a confident error.
    d["margin"] = (d["prob"] - d["threshold"]).abs()

    g = d.groupby(VARIANT_KEY + ["holdout_gene", "label"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        n_wrong=("wrong", "sum"),
        mean_prob=("prob", "mean"),
        sd_prob=("prob", "std"),
        mean_margin=("margin", "mean"),
        mean_threshold=("threshold", "mean"),
    )
    g["sd_prob"] = g["sd_prob"].fillna(0.0)
    g["consensus"] = g["n_wrong"]
    g["error_type"] = np.where(
        g["n_wrong"] == 0, "correct",
        np.where(g["label"] == 1, "false_negative", "false_positive"))
    return g


def attach_master(errors: pd.DataFrame, master_csv: Path) -> pd.DataFrame:
    """Join the covariates back on. Fails loudly on an incomplete join."""
    m = pd.read_csv(master_csv, low_memory=False)
    keep = VARIANT_KEY + [c for c in COVARIATES if c in m.columns]
    m = m[keep].drop_duplicates(VARIANT_KEY)

    out = errors.merge(m, on=VARIANT_KEY, how="left", validate="many_to_one")
    missing = out[[c for c in COVARIATES if c in out.columns]].isna().all(axis=1).sum()
    if missing:
        raise ValueError(
            f"{missing} of {len(out)} predicted variants did not join to "
            f"{master_csv.name}. The predictions and the master table are not "
            "describing the same build; do not interpret this output.")
    return out


def _compare_continuous(err: pd.Series, ok: pd.Series) -> Dict[str, object]:
    err, ok = err.dropna(), ok.dropna()
    if len(err) < 3 or len(ok) < 3:
        return {"test": "mannwhitneyu", "p_value": np.nan,
                "note": f"too few non-null ({len(err)} err / {len(ok)} correct)"}
    u = stats.mannwhitneyu(err, ok, alternative="two-sided")
    return {
        "test": "mannwhitneyu",
        "errors_median": float(err.median()),
        "correct_median": float(ok.median()),
        "effect": float(err.median() - ok.median()),
        "p_value": float(u.pvalue),
    }


def _compare_binary(err: pd.Series, ok: pd.Series) -> Dict[str, object]:
    def rate(s: pd.Series) -> tuple:
        s = s.dropna()
        truthy = s.astype(bool).sum()
        return int(truthy), int(len(s))

    e_yes, e_n = rate(err)
    o_yes, o_n = rate(ok)
    if min(e_n, o_n) < 3:
        return {"test": "fisher", "p_value": np.nan, "note": "too few non-null"}
    table = [[e_yes, e_n - e_yes], [o_yes, o_n - o_yes]]
    p = stats.fisher_exact(table).pvalue
    return {
        "test": "fisher",
        "errors_rate": e_yes / e_n if e_n else np.nan,
        "correct_rate": o_yes / o_n if o_n else np.nan,
        "effect": (e_yes / e_n - o_yes / o_n) if e_n and o_n else np.nan,
        "p_value": float(p),
    }


def _compare_nominal(err: pd.Series, ok: pd.Series) -> Dict[str, object]:
    err, ok = err.dropna().astype(str), ok.dropna().astype(str)
    levels = sorted(set(err) | set(ok))
    if len(levels) < 2 or min(len(err), len(ok)) < 3:
        return {"test": "chi2", "p_value": np.nan,
                "note": f"{len(levels)} level(s)"}
    table = np.array([[int((err == lv).sum()) for lv in levels],
                      [int((ok == lv).sum()) for lv in levels]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return {"test": "chi2", "p_value": np.nan, "note": "one level after pruning"}
    chi2, p, _, _ = stats.chi2_contingency(table)
    top = err.value_counts(normalize=True)
    return {
        "test": "chi2",
        "errors_mode": top.index[0],
        "errors_mode_frac": float(top.iloc[0]),
        "p_value": float(p),
    }


def covariate_report(joined: pd.DataFrame, inputs: Sequence[str]) -> pd.DataFrame:
    """Per covariate: errors vs correct, with the model-input flag attached.

    p-values are raw. 17 covariates are tested, so a Benjamini-Hochberg column
    is added and is the one to read; an uncorrected 0.03 among 17 tests is not
    a finding.
    """
    is_error = joined["error_type"] != "correct"
    rows = []
    for col, kind in COVARIATES.items():
        if col not in joined.columns:
            continue
        err, ok = joined.loc[is_error, col], joined.loc[~is_error, col]
        cmp = {"binary": _compare_binary,
               "nominal": _compare_nominal}.get(kind, _compare_continuous)(err, ok)
        rows.append({
            "covariate": col,
            "kind": kind,
            "is_model_input": col in inputs,
            "n_errors_nonnull": int(err.notna().sum()),
            "n_correct_nonnull": int(ok.notna().sum()),
            **cmp,
        })
    out = pd.DataFrame(rows)
    if out.empty or out["p_value"].notna().sum() == 0:
        out["p_value_bh"] = np.nan
        return out

    ok_mask = out["p_value"].notna()
    p = out.loc[ok_mask, "p_value"].to_numpy()
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate(
        (p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out.loc[ok_mask, "p_value_bh"] = np.minimum(adj, 1.0)
    return out.sort_values("p_value_bh")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid_dir", type=Path, default=GRID_DIR)
    ap.add_argument("--master_csv", type=Path, default=MASTER_CSV)
    ap.add_argument("--arm", default=DEFAULT_ARM,
                    help=f"Cell slug without the seed suffix (default: {DEFAULT_ARM})")
    ap.add_argument("--min_seeds", type=int, default=None,
                    help="Seeds that must agree before a row counts as an error. "
                         "Default: all available seeds of the arm.")
    ap.add_argument("--top_n", type=int, default=25,
                    help="Confident errors to list per direction in the log.")
    ap.add_argument("--out_prefix", type=Path, default=None,
                    help="Default: <grid_dir>/error_analysis_<arm>")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S")

    preds = load_arm(args.grid_dir, args.arm)
    n_seeds = preds["seed"].nunique()
    min_seeds = args.min_seeds or n_seeds
    logger.info("arm %s: %d seeds, %d prediction rows", args.arm, n_seeds, len(preds))

    per_variant = classify(preds)
    # Demote errors that not enough seeds agree on -- they are draws, not errors.
    weak = (per_variant["n_wrong"] > 0) & (per_variant["n_wrong"] < min_seeds)
    if weak.any():
        logger.info("%d variants wrong in some but not %d seeds -> not counted "
                    "as errors", int(weak.sum()), min_seeds)
        per_variant.loc[weak, "error_type"] = "correct"

    joined = attach_master(per_variant, args.master_csv)
    inputs = prior_columns(args.grid_dir)

    n_err = int((joined.error_type != "correct").sum())
    logger.info("%d held-out variants, %d consensus errors (%.1f%%)",
                len(joined), n_err, 100 * n_err / max(len(joined), 1))
    logger.info("  by type: %s", joined.error_type.value_counts().to_dict())
    logger.info("  by gene:\n%s",
                pd.crosstab(joined.holdout_gene, joined.error_type).to_string())

    report = covariate_report(joined, inputs)
    indep = report[~report.is_model_input]
    logger.info("covariates that are NOT model inputs, by corrected p:\n%s",
                indep[["covariate", "test", "p_value", "p_value_bh"]]
                .to_string(index=False))

    prefix = args.out_prefix or (args.grid_dir / f"error_analysis_{args.arm}")
    errors = joined[joined.error_type != "correct"].sort_values(
        ["error_type", "mean_margin"], ascending=[True, False])
    errors.to_csv(f"{prefix}_errors.csv", index=False)
    report.to_csv(f"{prefix}_covariates.csv", index=False)
    logger.info("wrote %s_errors.csv (%d rows) and %s_covariates.csv",
                prefix, len(errors), prefix)

    for kind in ("false_positive", "false_negative"):
        top = errors[errors.error_type == kind].head(args.top_n)
        if top.empty:
            continue
        logger.info("most confident %ss:\n%s", kind,
                    top[["gene", "position", "wt_aa", "mut_aa", "mean_prob",
                         "mean_threshold", "mean_margin", "review_status"]]
                    .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
