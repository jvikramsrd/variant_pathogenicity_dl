"""Failure analysis — the checks to run on every result before believing it.

Each function takes run artefacts (``predictions_*.csv``, ``valpreds_*.csv``,
the canonical table) and returns a small DataFrame; :func:`failure_report`
renders them together. None of them needs a GPU.

    gene_shortcut          do the features identify the gene? (nearest-centroid
                           accuracy, residue-grouped) + per-gene base-rate drift
    population_shortcut    does the model just rank by allele frequency? AUC
                           among variants ABSENT from gnomAD vs observed
    overfitting            inner-validation vs held-out AUC gap per gene
    msh6_context           MSH6 performance inside vs beyond the first 1,022 residues
    class_imbalance        prevalence per gene = the PR-AUC of a random ranker
    calibration_by_gene    ECE / Brier / slope per held-out gene
    pretraining_regression Pk arms whose paired delta vs P0 is negative with a
                           CI excluding zero

Duplicate variants and sequence leakage are the leakage report's job
(vpdl.dl.leakage); their counts are pulled in here so one page has everything.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from vpdl.dl.calibration import calibration_report
from vpdl.evaluate import roc_auc
from vpdl.splits import group_keys

__all__ = ["gene_shortcut", "population_shortcut", "overfitting", "msh6_context",
           "class_imbalance", "calibration_by_gene", "pretraining_regression",
           "failure_report"]


def gene_shortcut(table: pd.DataFrame, columns: Sequence[str], n_folds: int = 5,
                  label_column: str = "label__clinvar") -> pd.DataFrame:
    """How well do the features alone predict WHICH GENE a labelled variant is in?

    Near-perfect accuracy is not a leak under leave-one-gene-out (the held-out
    gene is unseen), but it means the features carry a gene signature the
    model can fit to per-gene base rates — the shortcut this checks for.
    """
    rows = table[table[label_column].notna()] if label_column in table else table
    X = rows[list(columns)].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    genes = rows["gene"].to_numpy()
    groups = group_keys(rows)
    from vpdl.dl.splits import _hash_fold
    fold = np.array([_hash_fold(g, n_folds, 0) for g in groups])   # stable across processes
    correct = 0
    for k in range(n_folds):
        train, test = fold != k, fold == k
        if not test.any():
            continue
        labels = sorted(set(genes[train]))
        centroids = np.stack([X[train & (genes == g)].mean(0) for g in labels])
        nearest = np.argmin(((X[test][:, None] - centroids[None]) ** 2).sum(-1), axis=1)
        correct += int((np.array(labels)[nearest] == genes[test]).sum())
    chance = rows["gene"].value_counts(normalize=True).max()
    return pd.DataFrame([{"check": "gene_from_features", "accuracy": correct / max(1, len(rows)),
                          "majority_baseline": float(chance), "n": len(rows)}])


def population_shortcut(predictions: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Model AUC among gnomAD-absent vs gnomAD-observed variants, and AF-alone AUC."""
    from vpdl.splits import variant_keys

    info = table.assign(variant_key=variant_keys(table))[
        ["variant_key", "feature_gnomad_observed", "feature_gnomad_log10_af"]]
    merged = predictions.merge(info, on="variant_key", how="left")
    out = []
    for gene, rows in merged.groupby("gene"):
        entry = {"gene": gene, "n": len(rows),
                 "auc_all": roc_auc(rows["label"], rows["score"]),
                 "auc_af_alone": roc_auc(rows["label"], -rows["feature_gnomad_log10_af"])}
        for flag, name in ((1.0, "observed"), (0.0, "absent")):
            subset = rows[rows["feature_gnomad_observed"] == flag]
            entry[f"n_{name}"] = len(subset)
            entry[f"auc_{name}"] = roc_auc(subset["label"], subset["score"])
        out.append(entry)
    return pd.DataFrame(out)


def overfitting(predictions: pd.DataFrame, valpreds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, test in predictions.groupby("gene"):
        val = valpreds[valpreds["gene"] == gene]
        test_auc, val_auc = roc_auc(test["label"], test["score"]), roc_auc(val["label"],
                                                                          val["score"])
        rows.append({"gene": gene, "inner_val_auc": val_auc, "heldout_auc": test_auc,
                     "gap": val_auc - test_auc})
    return pd.DataFrame(rows)


def msh6_context(predictions: pd.DataFrame, window: int = 1022) -> pd.DataFrame:
    rows = predictions[predictions["gene"] == "MSH6"].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["position"] = rows["variant_key"].str.split(":").str[1].astype(int)
    out = []
    for name, subset in (("within_first_1022", rows[rows["position"] <= window]),
                         ("beyond_1022", rows[rows["position"] > window])):
        out.append({"region": name, "n": len(subset),
                    "n_benign": int((subset["label"] == 0).sum()),
                    "auc": roc_auc(subset["label"], subset["score"])})
    return pd.DataFrame(out)


def class_imbalance(predictions: pd.DataFrame) -> pd.DataFrame:
    return (predictions.groupby("gene")["label"]
            .agg(n="size", pathogenic="sum").assign(
                prevalence=lambda f: f["pathogenic"] / f["n"],
                random_pr_auc=lambda f: f["pathogenic"] / f["n"]).reset_index())


def calibration_by_gene(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"gene": gene, **calibration_report(rows["label"], rows["score"])}
                         for gene, rows in predictions.groupby("gene")])


def pretraining_regression(paired: pd.DataFrame, reference_arm_contains: str = "P0"
                           ) -> pd.DataFrame:
    """Rows of a ``vpdl paired`` table (reference = the P0 arm) that regress."""
    if paired.empty:
        return paired
    flagged = paired[(paired["ci_high"] < 0)]
    return flagged.assign(note=f"significantly worse than {reference_arm_contains}")


def failure_report(sections: dict[str, pd.DataFrame]) -> str:
    lines = ["# Failure analysis", ""]
    for name, frame in sections.items():
        lines += [f"## {name}", ""]
        if frame.empty:
            lines.append("_(nothing to report)_")
        else:
            try:
                lines.append(frame.to_markdown(index=False))
            except ImportError:                  # tabulate not installed
                lines += ["```", frame.to_string(index=False), "```"]
        lines.append("")
    return "\n".join(lines)
