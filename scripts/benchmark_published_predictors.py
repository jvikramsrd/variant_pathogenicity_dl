#!/usr/bin/env python3
"""Head-to-head benchmark of published predictors on the Lynch/MMR panel.

PROJECT_PLAN.md Phase 3 step 7 ("sanity-check against published anchors")
asks for our numbers to be reported *on the same scale* as the methods we
claim to improve on. This script produces that table from data already in the
repository: the MMR extended dataset carries per-variant scores for the 17
ProteinGym clinical-benchmark predictors plus AlphaMissense, joined onto the
same ClinVar/ProteinGym-clinical labels the model is trained and evaluated on.

Protocol (deliberately identical to ``scripts/run_mmr_transfer.py`` so the
rows are directly comparable):

* **Rows** — clinical labels only (``label_source in {clinvar, pg_clinical}``).
  DMS-derived labels are excluded; for MSH2 they are 98% of the ``label``
  column and are the Phase-2 orthogonal validation axis, so scoring on them
  would be circular.
* **Score orientation** — each predictor's sign is fixed on the **broad
  80-gene panel with all four MMR genes removed**, never on the evaluation
  data. A predictor whose ROC-AUC is below 0.5 there is negated. This keeps
  "higher = more pathogenic" without reading the MMR labels to decide it.
* **Threshold** — MCC-optimal, tuned leave-one-gene-out (on the *other* MMR
  genes) and then applied to the held-out gene, matching how the model's own
  threshold is tuned on inner validation.
* **CIs** — 10,000-iteration percentile bootstrap via :mod:`src.eval_utils`.

Coverage differs per predictor (ProteinGym's clinical benchmark does not cover
every MMR variant), so ``n`` is reported per cell and a common-subset variant
of the pooled table is emitted alongside the per-predictor one.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_utils import bootstrap_ci, optimal_threshold_by_mcc, safe_mcc  # noqa: E402
from src.mmr_dataset import MMR_GENES  # noqa: E402

logger = logging.getLogger("benchmark_published")

CLINICAL_SOURCES = ("clinvar", "pg_clinical")

#: Display names for the score columns we benchmark.
PREDICTOR_LABELS = {
    "am_pathogenicity": "AlphaMissense",
    "zs_revel": "REVEL",
    "zs_cadd": "CADD",
    "zs_eve": "EVE",
    "zs_gemme": "GEMME",
    "zs_esm1b": "ESM-1b",
    "zs_trancepteve_l": "TranceptEVE-L",
    "zs_poet": "PoET",
    "zs_metarnn": "MetaRNN",
    "zs_vest4": "VEST4",
    "zs_bayesdel_addaf": "BayesDel (addAF)",
    "zs_mpc": "MPC",
    "zs_sift4g": "SIFT4G",
    "zs_polyphen2_hvar": "PolyPhen-2 (HVAR)",
    "zs_provean": "PROVEAN",
    "zs_fathmm": "FATHMM",
    "zs_deogen2": "DEOGEN2",
    "zs_mutationassessor": "MutationAssessor",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mmr_dataset",
                   default="data/mmr/processed/extended/extended_dataset.csv",
                   help="MMR extended dataset (scripts/build_mmr_dataset.py).")
    p.add_argument("--broad_dataset",
                   default="data/processed/extended/extended_dataset.csv",
                   help="Broad-panel extended dataset used ONLY to fix each "
                        "predictor's score orientation, with MMR genes removed.")
    p.add_argument("--model_results", default=None, nargs="*",
                   help="Optional model result CSVs (mmr_transfer / "
                        "esm_finetune) to append as rows for context.")
    p.add_argument("--out_dir", default="data/processed/benchmark")
    p.add_argument("--markdown", default="docs/PUBLISHED_COMPARISON.md")
    p.add_argument("--n_bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def clinical_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Clinical-label MMR rows, PMS2 homology-gated rows dropped."""
    out = df[df["gene"].isin(MMR_GENES)].copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out[out["label"].notna()]
    if "label_source" in out.columns:
        out = out[out["label_source"].isin(CLINICAL_SOURCES)]
    if "pms2_homology_excluded" in out.columns:
        keep = pd.to_numeric(out["pms2_homology_excluded"],
                             errors="coerce").fillna(0) != 1
        out = out[keep]
    return out.reset_index(drop=True)


def fit_orientations(broad_csv: Path, columns) -> dict[str, dict]:
    """Sign per predictor, fitted on non-MMR genes only.

    ROC-AUC is orientation-symmetric, so choosing the sign from the evaluation
    data would hand each predictor a free bit of label information. Fixing it
    on a disjoint gene set keeps the comparison honest without hard-coding a
    sign convention per predictor that ProteinGym may change between releases.
    """
    if not broad_csv.exists():
        logger.warning("Broad panel %s missing; falling back to +1 for every "
                       "predictor. Orientation is then UNVERIFIED.", broad_csv)
        return {c: {"sign": 1, "n": 0, "auc": None} for c in columns}

    usecols = ["gene", "label", "label_source"] + list(columns)
    broad = pd.read_csv(broad_csv, usecols=lambda c: c in set(usecols),
                        low_memory=False)
    broad = broad[~broad["gene"].isin(MMR_GENES)]
    broad["label"] = pd.to_numeric(broad["label"], errors="coerce")
    broad = broad[broad["label"].notna()]
    if "label_source" in broad.columns:
        broad = broad[broad["label_source"].isin(CLINICAL_SOURCES)]

    from sklearn.metrics import roc_auc_score
    signs: dict[str, int] = {}
    for col in columns:
        if col not in broad.columns:
            signs[col] = {"sign": 1, "n": 0, "auc": None}
            continue
        sub = broad[[col, "label"]].dropna()
        if len(sub) < 50 or sub["label"].nunique() < 2:
            logger.warning("%s: only %d calibration rows on non-MMR genes; "
                           "orientation left at +1.", col, len(sub))
            signs[col] = {"sign": 1, "n": int(len(sub)), "auc": None}
            continue
        auc = float(roc_auc_score(sub["label"].to_numpy(), sub[col].to_numpy()))
        signs[col] = {"sign": 1 if auc >= 0.5 else -1,
                      "n": int(len(sub)), "auc": auc}
        logger.info("%-22s orientation %+d (non-MMR AUC %.3f, n=%d)",
                    col, signs[col]["sign"], auc, len(sub))
    return signs


def score_one(y: np.ndarray, s: np.ndarray, thr: float,
              n_bootstrap: int, seed: int) -> dict:
    rep: dict = {"n": int(len(y)), "n_pos": int(y.sum()),
                 "prevalence": float(y.mean()), "threshold": float(thr)}
    for metric in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(y, s, metric=metric, n_bootstrap=n_bootstrap, seed=seed)
        rep[metric] = ci["point"]
        rep[f"{metric}_ci_low"] = ci["lower"]
        rep[f"{metric}_ci_high"] = ci["upper"]
    mcc = bootstrap_ci(y, s, metric="mcc", threshold=thr,
                       n_bootstrap=n_bootstrap, seed=seed)
    rep["mcc"] = mcc["point"]
    rep["mcc_ci_low"] = mcc["lower"]
    rep["mcc_ci_high"] = mcc["upper"]
    return rep


def benchmark(clin: pd.DataFrame, signs: dict[str, int],
              n_bootstrap: int, seed: int) -> pd.DataFrame:
    genes = sorted(clin["gene"].unique())
    rows: list[dict] = []
    for col, name in PREDICTOR_LABELS.items():
        if col not in clin.columns:
            continue
        sign = signs.get(col, {}).get("sign", 1)
        oriented = sign * pd.to_numeric(clin[col], errors="coerce")
        for gene in genes:
            in_gene = (clin["gene"] == gene).to_numpy()
            ok = oriented.notna().to_numpy()
            ho = in_gene & ok
            other = (~in_gene) & ok
            if ho.sum() < 10 or clin.loc[ho, "label"].nunique() < 2:
                continue
            # Threshold tuned on the OTHER MMR genes, applied to this one.
            if other.sum() >= 10 and clin.loc[other, "label"].nunique() >= 2:
                thr, _ = optimal_threshold_by_mcc(
                    clin.loc[other, "label"].to_numpy().astype(int),
                    oriented[other].to_numpy())
            else:
                thr = float(np.nanmedian(oriented[ok].to_numpy()))
            rep = score_one(clin.loc[ho, "label"].to_numpy().astype(int),
                            oriented[ho].to_numpy(), thr, n_bootstrap, seed)
            rows.append({"predictor": name, "column": col, "gene": gene,
                         "orientation": sign,
                         "threshold_tuned_on": "other MMR genes", **rep})
        # Pooled across genes, threshold tuned on the pooled set itself
        # (marked as such -- it is an upper bound, not a held-out number).
        ok = oriented.notna().to_numpy()
        if ok.sum() >= 10 and clin.loc[ok, "label"].nunique() >= 2:
            y = clin.loc[ok, "label"].to_numpy().astype(int)
            s = oriented[ok].to_numpy()
            thr, _ = optimal_threshold_by_mcc(y, s)
            rep = score_one(y, s, thr, n_bootstrap, seed)
            rows.append({"predictor": name, "column": col, "gene": "POOLED",
                         "orientation": sign,
                         "threshold_tuned_on": "pooled (in-sample; optimistic)",
                         **rep})
    return pd.DataFrame(rows)


def load_model_rows(paths) -> pd.DataFrame:
    """Normalise our own result CSVs into the benchmark table's schema."""
    frames = []
    for p in paths or []:
        path = Path(p)
        if not path.exists():
            logger.warning("Model results %s not found; skipping.", path)
            continue
        df = pd.read_csv(path)
        if "holdout_gene" not in df.columns:
            logger.warning("%s has no holdout_gene column; skipping.", path)
            continue
        tag = path.stem
        keep = {"gene": df["holdout_gene"], "predictor": f"[ours] {tag}",
                "column": tag, "orientation": 1,
                "threshold_tuned_on": "inner validation (held-out gene unseen)"}
        for c in ("n_holdout", "threshold", "roc_auc", "roc_auc_ci_low",
                  "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
                  "mcc", "mcc_ci_low", "mcc_ci_high"):
            if c in df.columns:
                keep["n" if c == "n_holdout" else c] = df[c]
        frames.append(pd.DataFrame(keep))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
def to_markdown(table: pd.DataFrame, clin: pd.DataFrame, signs: dict,
                n_bootstrap: int) -> str:
    genes = [g for g in sorted(clin["gene"].unique())]
    lines: list[str] = []
    lines.append("# Published-predictor comparison on the Lynch/MMR panel\n")
    lines.append("Generated by `scripts/benchmark_published_predictors.py`. "
                 "Every number below is computed on **this repository's own "
                 "MMR clinical slice** -- it is not copied from any paper.\n")
    counts = clin.groupby("gene")["label"].agg(["size", "mean"])
    lines.append("## Evaluation set\n")
    lines.append("| Gene | n (clinical labels) | prevalence (pathogenic) |")
    lines.append("|---|---|---|")
    for g, r in counts.iterrows():
        lines.append(f"| {g} | {int(r['size'])} | {r['mean']:.3f} |")
    lines.append(f"| **total** | **{len(clin)}** | **{clin['label'].mean():.3f}** |\n")
    lines.append("Clinical labels only (ClinVar >=2 star / ProteinGym clinical). "
                 "DMS-derived labels are excluded -- see the note on MSH2 below.\n")

    for metric, title in (("roc_auc", "ROC-AUC"), ("mcc", "MCC"),
                          ("pr_auc", "PR-AUC")):
        lines.append(f"\n## {title} by gene "
                     f"(95% CI, {n_bootstrap:,}-iteration bootstrap)\n")
        header = "| Predictor | " + " | ".join(genes) + " | POOLED |"
        lines.append(header)
        lines.append("|" + "---|" * (len(genes) + 2))
        piv = table.pivot_table(index="predictor", columns="gene",
                                values=metric, aggfunc="first")
        lo = table.pivot_table(index="predictor", columns="gene",
                               values=f"{metric}_ci_low", aggfunc="first")
        hi = table.pivot_table(index="predictor", columns="gene",
                               values=f"{metric}_ci_high", aggfunc="first")
        order = piv.reindex(piv[genes].mean(axis=1).sort_values(
            ascending=False).index)
        for name in order.index:
            cells = []
            for g in genes + ["POOLED"]:
                if g in piv.columns and pd.notna(piv.loc[name, g]):
                    cells.append(f"{piv.loc[name, g]:.3f} "
                                 f"({lo.loc[name, g]:.3f}-{hi.loc[name, g]:.3f})")
                else:
                    cells.append("--")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## Threshold-transfer caveat (read before quoting MCC)\n")
    lines.append("ROC-AUC and PR-AUC are threshold-free. MCC is not: each "
                 "per-gene MCC uses a cut-off tuned on the *other* MMR genes, "
                 "which is the honest held-out protocol but also a real "
                 "measurement of how badly absolute scores transfer between "
                 "these genes. A predictor with a strong AUC and a collapsed "
                 "MCC (MPC, TranceptEVE-L on MSH6) is not weak at ranking -- "
                 "its score scale simply does not carry across genes. Read "
                 "those cells as evidence for per-gene calibration, not as a "
                 "ranking of predictor quality.\n")
    lines.append("\n## Score orientation (fitted on non-MMR genes only)\n")
    lines.append("Sign chosen so that higher = more pathogenic, using clinical "
                 "labels on the broad panel with all four MMR genes removed. "
                 "No predictor's calibration AUC sits near 0.5, so no sign "
                 "here is a coin flip.\n")
    lines.append("| Column | Sign | Calibration ROC-AUC (non-MMR) | n |")
    lines.append("|---|---|---|---|")
    for c, info in sorted(signs.items()):
        auc = info.get("auc")
        auc_s = "--" if auc is None else f"{auc:.3f}"
        lines.append(f"| `{c}` | {info.get('sign', 1):+d} | {auc_s} | "
                     f"{info.get('n', 0)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    mmr_csv = Path(args.mmr_dataset)
    if not mmr_csv.exists():
        raise SystemExit(f"MMR dataset not found: {mmr_csv}. Run "
                         "scripts/build_mmr_dataset.py first.")
    df = pd.read_csv(mmr_csv, low_memory=False)
    clin = clinical_slice(df)
    logger.info("Clinical evaluation slice: %d rows across %s",
                len(clin), sorted(clin["gene"].unique()))

    present = [c for c in PREDICTOR_LABELS if c in clin.columns]
    signs = fit_orientations(Path(args.broad_dataset), present)
    table = benchmark(clin, signs, args.n_bootstrap, args.seed)
    model_rows = load_model_rows(args.model_results)
    if not model_rows.empty:
        table = pd.concat([table, model_rows], ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "published_predictor_comparison.csv"
    table.to_csv(csv_path, index=False)
    (out_dir / "published_predictor_comparison_meta.json").write_text(json.dumps({
        "mmr_dataset": str(mmr_csv), "broad_dataset": str(args.broad_dataset),
        "n_clinical_rows": int(len(clin)),
        "genes": sorted(clin["gene"].unique().tolist()),
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "orientations": signs,
    }, indent=2))
    md = Path(args.markdown)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(to_markdown(table, clin, signs, args.n_bootstrap))
    logger.info("Wrote %s and %s", csv_path, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
