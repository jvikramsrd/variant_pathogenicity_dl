"""VUS as a first-class task: rank, enrich, calibrate — against outcomes that actually happened.

A VUS cannot be scored against its own label (it has none). It can be scored
against what later happened to it: take an older ClinVar release, keep its
VUS, and look them up in a newer release. Those reclassified to P/LP are
positives, to LB/B negatives; those still VUS are not scored.
(:func:`reclassification_outcomes`). ClinVar archives monthly releases, so
this needs no new data source — only an older ``variant_summary`` (TBD — RUN
ON DGX SPARK; not downloaded here).

Scores:
    priority        p(P) + p(LP)  — how pathogenic-leaning a VUS is
    benign_priority p(LB) + p(B)
    resolution      max of the two — how far from "uncertain" the model thinks it is

Metrics: ROC-AUC and average precision of `priority` for reclassified-to-
pathogenic vs to-benign; precision and enrichment (over the base rate) among
the top-k; calibration of `priority` on the reclassified set; agreement with
ClinVar's own VUS sub-tiers (VUS-high / -mid / -low) where present.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from vpdl.evaluate import roc_auc
from vpdl.slm.labels import normalize_classification

__all__ = ["vus_scores", "reclassification_outcomes", "ranking_metrics", "subtier_agreement"]


def vus_scores(probs: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    priority = probs[:, 0] + probs[:, 1]
    benign = probs[:, 3] + probs[:, 4]
    return pd.DataFrame({"priority": priority, "benign_priority": benign,
                         "p_vus": probs[:, 2], "resolution": np.maximum(priority, benign)})


def reclassification_outcomes(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Variants that were VUS in `old`; outcome 1 = now P/LP, 0 = now LB/B, NaN = still unresolved.

    Both frames need ``variant_id`` and ``clinvar_classification`` (the variants table
    built from each release).
    """
    before = old.assign(_info=old["clinvar_classification"].map(normalize_classification))
    was_vus = before["_info"].map(lambda i: i.label5 == "vus")
    after = new.set_index("variant_id")["clinvar_classification"].map(normalize_classification)
    frame = before.loc[was_vus, ["variant_id"]].copy()
    frame["new_classification"] = frame["variant_id"].map(
        new.set_index("variant_id")["clinvar_classification"])
    frame["outcome"] = frame["variant_id"].map(lambda v: after[v].binary if v in after.index else None)
    frame["outcome"] = pd.to_numeric(frame["outcome"], errors="coerce")
    return frame.reset_index(drop=True)


def ranking_metrics(priority: Sequence[float], outcome: Sequence[float],
                    ks: Sequence[int | float] = (10, 50, 100, 0.05, 0.1)) -> dict[str, Any]:
    priority = np.asarray(priority, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    keep = np.isfinite(priority) & np.isfinite(outcome)
    priority, outcome = priority[keep], outcome[keep].astype(int)
    if len(outcome) == 0 or len(np.unique(outcome)) < 2:
        return {"n": int(len(outcome)), "note": "needs both outcomes"}
    from vpdl.evaluate import _average_precision
    base = float(outcome.mean())
    order = np.argsort(-priority, kind="stable")
    top = {}
    for k in ks:
        count = int(round(k * len(order))) if isinstance(k, float) else int(k)
        count = max(1, min(count, len(order)))
        precision = float(outcome[order[:count]].mean())
        top[str(k)] = {"k": count, "precision": precision,
                       "enrichment": precision / base if base else float("nan")}
    return {"n": int(len(outcome)), "base_rate": base, "roc_auc": roc_auc(outcome, priority),
            "average_precision": float(_average_precision(outcome, priority)), "top_k": top}


def subtier_agreement(priority: Sequence[float], modifiers: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Spearman between priority and ClinVar's VUS-low < VUS-mid < VUS-high, where given."""
    from scipy.stats import spearmanr
    order = {"vus_low": 0, "vus_mid": 1, "vus_high": 2}
    values, tiers = [], []
    for score, mods in zip(priority, modifiers):
        tier = next((order[m] for m in mods if m in order), None)
        if tier is not None and np.isfinite(score):
            values.append(score)
            tiers.append(tier)
    if len(set(tiers)) < 2:
        return {"n": len(tiers), "note": "fewer than two VUS sub-tiers present"}
    rho, p = spearmanr(values, tiers)
    return {"n": len(tiers), "spearman": float(rho), "p_value": float(p)}
