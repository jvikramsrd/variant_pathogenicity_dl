"""Leakage checks, the report they produce, and the gate training passes through.

Two severities matter. **Critical** findings make any result uninterpretable
and stop training (:func:`leakage_gate`, called by ``run_cell`` before the
first model is built). **Warnings** are properties of the design that a reader
must be told about — e.g. leave-one-gene-out keeps the held-out gene's paralog
in training — and go into the report, not into an exception.

Checks, and where each defect class was paid for:

    exact_duplicates        one variant, two rows (assembly guarantees one)
    hgvs_duplicates         one protein change under two identities
    protein_duplicates      two accessions, one sequence (would straddle folds)
    split_isolation         a variant or residue on both sides of a fold (L3)
    homology_straddle       aligned paralog residues on both sides of a fold
    sequence_similarity     held-out protein vs training proteins; homolog twins
    functional_overlap      validation-only assay data reaching training (item 14)
    feature_leakage         a feature reproducing the label (L2)
    derived_feature_leakage gene-constant features (L15), label-derived columns,
                            functional values as features, circular priors
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.splits import GENE_DISJOINT_SCHEMES, Fold
from vpdl.evaluate import symmetric_agreement
from vpdl.splits import group_keys, variant_keys

logger = logging.getLogger(__name__)

__all__ = ["LeakageFinding", "LeakageReport", "LeakageError", "FUNCTIONAL_VALUE_COLUMNS",
           "run_checks", "leakage_gate"]

# Validation-only columns written by vpdl.dl.canonical. None may be a feature.
FUNCTIONAL_VALUE_COLUMNS = ("cimra_value", "cimra_strength", "dms_score", "mavedb_score")

# Features trained on or calibrated against clinical labels. Not a leak — a
# circularity a reader must be told about when evaluating on ClinVar.
CIRCULAR_PRIORS = {
    "alphamissense": "AlphaMissense thresholds were calibrated on ClinVar; it was "
                     "trained on population frequency and structure, not ClinVar labels.",
}


class LeakageError(ValueError):
    """Critical leakage: training must not proceed."""


@dataclass
class LeakageFinding:
    check: str
    severity: str                 # critical | warning | info
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageReport:
    scheme: str
    findings: list[LeakageFinding] = field(default_factory=list)

    @property
    def critical(self) -> list[LeakageFinding]:
        return [f for f in self.findings if f.severity == "critical"]

    def add(self, *findings: LeakageFinding) -> None:
        self.findings.extend(findings)

    def assert_clean(self) -> None:
        if self.critical:
            raise LeakageError(
                "Critical leakage detected — refusing to train:\n  "
                + "\n  ".join(f"[{f.check}] {f.message}" for f in self.critical))

    def as_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "n_critical": len(self.critical),
                "findings": [asdict(f) for f in self.findings]}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [f"### Split scheme: `{self.scheme}`", "",
                 f"Critical findings: **{len(self.critical)}**", "",
                 "| check | severity | count | finding |", "|---|---|---|---|"]
        order = {"critical": 0, "warning": 1, "info": 2}
        for f in sorted(self.findings, key=lambda f: (order.get(f.severity, 3), f.check)):
            lines.append(f"| {f.check} | {f.severity} | {f.count} | "
                         f"{f.message.replace('|', '/')} |")
        return "\n".join(lines) + "\n"


# -- individual checks ------------------------------------------------------

def check_exact_duplicates(table: pd.DataFrame) -> LeakageFinding:
    keys = pd.Series(variant_keys(table))
    dup = int(keys.duplicated().sum())
    return LeakageFinding("exact_duplicates", "critical" if dup else "info",
                          f"{dup} duplicate variant rows" if dup else "no duplicate variant rows",
                          dup, {"examples": keys[keys.duplicated()].head(5).tolist()})


def check_hgvs_duplicates(table: pd.DataFrame) -> list[LeakageFinding]:
    findings = []
    if "hgvs_p" in table.columns:
        per_change = table.groupby(["uniprot_id", "hgvs_p"])["variant_id"].nunique() \
            if "variant_id" in table.columns else pd.Series(dtype=int)
        clash = int((per_change > 1).sum())
        findings.append(LeakageFinding(
            "hgvs_duplicates", "critical" if clash else "info",
            f"{clash} protein changes under more than one variant identity" if clash
            else "every normalised HGVS p. maps to exactly one variant id", clash))
    if "clinvar_n_records" in table.columns:
        multi = int((pd.to_numeric(table["clinvar_n_records"], errors="coerce") > 1).sum())
        findings.append(LeakageFinding(
            "hgvs_duplicates", "info",
            f"{multi} protein changes are backed by several ClinVar records "
            "(resolved order-independently in the canonical table)", multi))
    return findings


def check_protein_duplicates(sequences: Mapping[str, str]) -> LeakageFinding:
    by_sequence: dict[str, list[str]] = {}
    for accession, sequence in sequences.items():
        by_sequence.setdefault(sequence, []).append(accession)
    clashes = [sorted(ids) for ids in by_sequence.values() if len(ids) > 1]
    return LeakageFinding("protein_duplicates", "critical" if clashes else "info",
                          f"identical sequences under several accessions: {clashes}"
                          if clashes else "all protein sequences are distinct",
                          len(clashes))


def check_split_isolation(work: pd.DataFrame, folds: Sequence[Fold], scheme: str
                          ) -> list[LeakageFinding]:
    keys = variant_keys(work)
    groups = group_keys(work)
    clusters = work["cluster_id"].to_numpy() if "cluster_id" in work.columns else None
    findings = []
    shared_variants = shared_groups = 0
    straddle: dict[str, int] = {}
    for fold in folds:
        tr, te = fold.train_positions, fold.test_positions
        shared_variants += len(set(keys[tr]) & set(keys[te]))
        shared_groups += len(set(groups[tr]) & set(groups[te]))
        if clusters is not None:
            straddle[fold.name] = len(set(clusters[tr]) & set(clusters[te]))
    findings.append(LeakageFinding(
        "split_isolation", "critical" if shared_variants else "info",
        f"{shared_variants} variant(s) on both sides of a fold" if shared_variants
        else "no variant appears on both sides of any fold", shared_variants))
    findings.append(LeakageFinding(
        "split_isolation", "critical" if shared_groups else "info",
        f"{shared_groups} (protein, residue) group(s) straddle a fold" if shared_groups
        else "no (protein, residue) group straddles any fold", shared_groups))
    if clusters is not None:
        total = sum(straddle.values())
        # Expected under plain leave-one-gene-out: the paralog stays in training.
        severity = ("warning" if scheme == "logo" else
                    "critical" if total and scheme in ("logo_purged", "family") else "info")
        findings.append(LeakageFinding(
            "homology_straddle", severity if total else "info",
            f"{total} homology cluster(s) have aligned paralog residues on both sides"
            + (" — expected under leave-one-gene-out; use logo_purged or family to remove"
               if scheme == "logo" and total else ""),
            total, {"per_fold": straddle}))
    return findings


def check_sequence_similarity(work: pd.DataFrame, folds: Sequence[Fold],
                              sequences: Mapping[str, str],
                              alignments: Mapping | None = None,
                              warn_identity: float = 0.2) -> list[LeakageFinding]:
    from vpdl.dl.homology import homologous_twins, pairwise_identity

    present = {acc: sequences[acc] for acc in pd.unique(work["uniprot_id"]) if acc in sequences}
    alignments = alignments or pairwise_identity(present)
    findings = []
    identity = {f"{a}-{b}": round(al.identity_shorter, 3) for (a, b), al in alignments.items()}
    findings.append(LeakageFinding("sequence_similarity", "info",
                                   "pairwise identity (identical / shorter length)",
                                   len(identity), {"identity": identity}))
    key_cols = ["uniprot_id", "position", "wt_aa", "mut_aa"]
    for fold in folds:
        test = work.iloc[fold.test_positions]
        train = work.iloc[fold.train_positions]
        near = {pair: v for pair, v in identity.items()
                if any(a in pair for a in set(test["uniprot_id"]))
                and any(b in pair for b in set(train["uniprot_id"]))
                and v >= warn_identity}
        twins = homologous_twins(list(test[key_cols].itertuples(index=False, name=None)),
                                 list(train[key_cols].itertuples(index=False, name=None)),
                                 alignments)
        severity = "warning" if near or twins["exact_twins"] else "info"
        findings.append(LeakageFinding(
            "sequence_similarity", severity,
            f"fold {fold.name}: training proteins >= {warn_identity:.0%} identical to the "
            f"held-out protein: {near or 'none'}; {twins['position_twins']} test variants have "
            f"a labelled training variant at the aligned paralog residue "
            f"({twins['exact_twins']} with the same substitution)",
            twins["exact_twins"], {"fold": fold.name, **twins, "near_identity": near}))
    return findings


def check_functional_overlap(work: pd.DataFrame, folds: Sequence[Fold], scheme: str,
                             feature_columns: Sequence[str],
                             validating_functional: bool,
                             functional_keys: Iterable[str] = ()) -> list[LeakageFinding]:
    findings = []
    used = [c for c in feature_columns
            if any(f in c for f in ("dms_score", "cimra", "mavedb"))]
    findings.append(LeakageFinding(
        "functional_overlap", "critical" if used else "info",
        f"validation-only functional values used as features: {used}" if used
        else "no functional-assay value is a feature", len(used)))
    if not validating_functional:
        return findings
    if scheme not in GENE_DISJOINT_SCHEMES:
        findings.append(LeakageFinding(
            "functional_overlap", "critical",
            f"functional validation under `{scheme}` would score assay variants whose "
            "gene was trained on — only gene-disjoint schemes give independent validation", 1))
        return findings
    keys = variant_keys(work)
    functional_keys = set(functional_keys)
    trained = sum(len(set(keys[f.train_positions]) & functional_keys
                      & set(keys[work["gene"].isin(f.test_genes).to_numpy()])) for f in folds)
    findings.append(LeakageFinding(
        "functional_overlap", "critical" if trained else "info",
        f"{trained} functional-validation variants were trained on in their own fold"
        if trained else "functional-validation variants never reach their fold's training set",
        trained))
    return findings


def check_feature_leakage(work: pd.DataFrame, feature_columns: Sequence[str],
                          label_column: str = "_train", warn: float = 0.95,
                          abort: float = 0.99) -> list[LeakageFinding]:
    labelled = work[work[label_column].notna()]
    y = labelled[label_column].to_numpy()
    findings = []
    for column in feature_columns:
        values = pd.to_numeric(labelled[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        agreement = symmetric_agreement(y, values)
        if not np.isfinite(agreement) or agreement < warn:
            continue
        findings.append(LeakageFinding(
            "feature_leakage", "critical" if agreement >= abort else "warning",
            f"{column} agrees with the training label at {agreement:.4f}"
            + (" — a label proxy" if agreement >= abort else " — strong; confirm it is not derived"),
            1, {"column": column, "agreement": round(float(agreement), 4)}))
    if not findings:
        findings.append(LeakageFinding("feature_leakage", "info",
                                       f"no feature agrees with the label at >= {warn}", 0))
    return findings


def check_derived_feature_leakage(table: pd.DataFrame, feature_columns: Sequence[str],
                                  train_sources: Sequence[str], scheme: str,
                                  eval_source: str = "clinvar") -> list[LeakageFinding]:
    from vpdl.features import drop_gene_constant

    findings = []
    if scheme in GENE_DISJOINT_SCHEMES:
        kept = set(drop_gene_constant(table, list(feature_columns)))
        constant = [c for c in feature_columns if c in table.columns and c not in kept]
        findings.append(LeakageFinding(
            "derived_feature_leakage", "info",
            f"gene-constant features removed before training (they encode gene identity): "
            f"{constant or 'none'}", len(constant)))
    derived = [c for c in feature_columns
               for source in train_sources
               if source == "pg_dms" and "dms" in c]
    findings.append(LeakageFinding(
        "derived_feature_leakage", "critical" if derived else "info",
        f"features derived from a training label source: {derived}" if derived
        else "no feature is derived from a training label source", len(derived)))
    for name, why in CIRCULAR_PRIORS.items():
        hits = [c for c in feature_columns if name in c]
        if hits and eval_source.startswith("clinvar"):
            findings.append(LeakageFinding("derived_feature_leakage", "warning",
                                           f"{hits}: {why}", len(hits)))
    return findings


# -- orchestration ------------------------------------------------------------

def run_checks(work: pd.DataFrame, folds: Sequence[Fold], scheme: str,
               feature_columns: Sequence[str], train_sources: Sequence[str],
               eval_source: str = "clinvar",
               sequences: Mapping[str, str] | None = None,
               validating_functional: bool = False,
               functional_keys: Iterable[str] = (),
               full: bool = True) -> LeakageReport:
    """All checks. `full=False` runs only the cheap ones that can be critical.

    The label-proxy check (landmine L2: a feature that IS the label) is cheap — one
    agreement score per column — and can be critical, so it runs in both modes.
    """
    report = LeakageReport(scheme)
    report.add(check_exact_duplicates(work), *check_hgvs_duplicates(work))
    report.add(*check_split_isolation(work, folds, scheme))
    report.add(*check_functional_overlap(work, folds, scheme, feature_columns,
                                         validating_functional, functional_keys))
    report.add(*check_derived_feature_leakage(work, feature_columns, train_sources,
                                              scheme, eval_source))
    report.add(*check_feature_leakage(work, feature_columns))
    if full:
        if sequences:
            report.add(check_protein_duplicates(sequences))
            report.add(*check_sequence_similarity(work, folds, sequences))
    return report


def leakage_gate(work: pd.DataFrame, folds: Sequence[Fold], scheme: str,
                 feature_columns: Sequence[str], train_sources: Sequence[str],
                 eval_source: str = "clinvar", validating_functional: bool = False,
                 functional_keys: Iterable[str] = ()) -> LeakageReport:
    """The cheap critical checks, run by every cell before training. Raises."""
    report = run_checks(work, folds, scheme, feature_columns, train_sources, eval_source,
                        validating_functional=validating_functional,
                        functional_keys=functional_keys, full=False)
    for finding in report.findings:
        if finding.severity == "warning":
            logger.warning("leakage [%s] %s", finding.check, finding.message)
    report.assert_clean()
    return report
