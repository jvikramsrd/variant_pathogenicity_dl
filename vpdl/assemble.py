"""Compose 1..N sources into one training table.

Selecting a single source is the normal case here, not a degenerate one: the
paper's question is whether pooling sources beats training on each alone, so
`assemble(["clinvar"])` and `assemble(ALL)` must be equally first-class and
produce tables that differ only in what went into them.

Satisfies regression landmines L1 (label orientation is checked against an
independent anchor) and L9 (a gene that vanishes fails the build).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.evaluate import roc_auc
from vpdl.sources.base import RECORD_COLUMNS, validate_frame

logger = logging.getLogger(__name__)

__all__ = [
    "LABEL_PRECEDENCE",
    "VARIANT_KEY",
    "label_sources_in",
    "resolve_labels",
    "label_anchor_agreement",
    "assert_genes_present",
    "assert_label_orientation",
    "AssemblyReport",
    "assemble",
]

VARIANT_KEY: tuple[str, ...] = ("uniprot_id", "position", "wt_aa", "mut_aa")

# Lower wins. Clinical assertions outrank benchmark labels, which outrank a
# single assay's binarised fitness.
LABEL_PRECEDENCE: dict[str, int] = {
    "clinvar": 1,
    "pg_clinical": 2,
    "mavedb": 3,
    "pg_dms": 4,
}


def label_anchor_agreement(
    labels: Sequence[int],
    prior: Sequence[float],
) -> float:
    """Agreement between a label set and an independent pathogenicity prior.

    This is the check that caught v1's ProteinGym inversion. An inverted label
    set does not merely look weak against an independent prior — it lands
    *below* chance, which is a signature no amount of label noise produces.
    """
    return roc_auc(labels, prior)


def assert_label_orientation(
    labels: Sequence[int],
    prior: Sequence[float],
    source: str,
    min_agreement: float = 0.5,
) -> float:
    """Raise when a source's labels disagree with an independent prior below chance.

    ProteinGym's ``DMS_score_bin == 1`` marks the TOP fitness half — tolerated,
    therefore benign. v1 mapped it straight to pathogenic and inverted roughly
    185,000 labels. Nothing failed; the model trained happily on backwards
    supervision until the merged labels were anchored against AlphaMissense.
    """
    agreement = label_anchor_agreement(labels, prior)
    if np.isnan(agreement):
        logger.warning("No anchor overlap for %s — orientation unverified.", source)
        return agreement
    if agreement < min_agreement:
        raise ValueError(
            f"Source '{source}' labels agree with the independent prior at "
            f"{agreement:.4f}, below chance. This is the signature of an "
            "INVERTED label mapping, not of weak labels. Check the source's "
            "score/bin polarity before proceeding."
        )
    return agreement


def assert_genes_present(df: pd.DataFrame, expected: Sequence[str]) -> None:
    """Fail the build when a gene the panel asked for is not in the table.

    On 2026-09-13 a rebuild dropped PMS2 entirely; the grid logged a warning
    that nobody read and 16 cells trained on a three-gene panel believing it was
    four. A missing gene is a failed build, not a log line.
    """
    present = set(df["gene"].dropna().unique())
    absent = [gene for gene in expected if gene not in present]
    if absent:
        raise ValueError(
            f"Expected genes absent from the assembled table: {absent}. "
            f"Present: {sorted(present)}. A panel gene with zero rows silently "
            "reduces the evaluation and must not pass."
        )


@dataclass
class AssemblyReport:
    sources: list[str] = field(default_factory=list)
    rows_per_source: dict[str, int] = field(default_factory=dict)
    dropped_per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)
    orientation_checks: dict[str, float] = field(default_factory=dict)
    conflicts_quarantined: int = 0
    feature_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sources": self.sources,
            "rows_per_source": self.rows_per_source,
            "dropped_per_source": self.dropped_per_source,
            "label_counts": self.label_counts,
            "orientation_checks": self.orientation_checks,
            "conflicts_quarantined": self.conflicts_quarantined,
            "feature_columns": self.feature_columns,
        }


_LABEL_FIELDS = ("label", "label_source", "evidence_tier")


def _merge_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per variant: labels by precedence, every other column coalesced.

    The two halves are resolved differently on purpose.

    * **Labels** come from exactly one source — the highest-precedence one that
      labels the variant — and variants whose sources contradict each other are
      quarantined (label withheld, row kept).
    * **Everything else** is coalesced: each column takes its first non-null
      value across ALL sources for that variant.

    The first version of this function applied precedence to whole rows with
    ``drop_duplicates(keep="first")``. For every variant ClinVar labelled, the
    ClinVar row won and the AlphaMissense and gnomAD rows — which carry the
    features — were discarded, so every labelled variant reached training with
    all-NaN features. Found by review on 2026-09-21 before any run; regression
    landmine L16.
    """
    key = list(VARIANT_KEY)
    frame = frame.copy()
    frame["_precedence"] = (
        frame["label_source"].map(LABEL_PRECEDENCE).fillna(99).astype(int)
    )
    frame = frame.sort_values("_precedence", kind="mergesort")

    labelled = frame[frame["label"].notna()]
    distinct = labelled.groupby(key)["label"].nunique()
    conflicted = distinct[distinct > 1].index

    winners = (
        labelled.drop_duplicates(subset=key, keep="first")
        .set_index(key)[list(_LABEL_FIELDS)]
    )
    if len(conflicted):
        logger.warning(
            "%d variants carry contradictory labels across sources; their labels "
            "are withheld and the rows kept unlabelled.", len(conflicted)
        )
        # Index-aligned rather than matching tuples by value, which was
        # dtype-fragile across a groupby MultiIndex (np.int64 vs int).
        winners.loc[conflicted, "label"] = np.nan
        winners.loc[conflicted, "label_source"] = "conflict_quarantined"

    other = [c for c in frame.columns
             if c not in key and c not in _LABEL_FIELDS and c != "_precedence"]
    coalesced = frame.groupby(key, sort=False)[other].first()

    table = coalesced.join(winners, how="left")
    table["label_source"] = table["label_source"].fillna("unlabelled")

    # Each labelling source's own label, kept alongside the resolved one. This is
    # what lets one pooled table serve every arm of the experiment: an arm picks
    # which sources it TRAINS on (resolve_labels), and every arm is scored on the
    # same held-out clinical labels (label__clinvar). Without these columns the
    # arms would need separate tables, and separate tables mean separate test sets.
    for name in labelled["label_source"].dropna().unique():
        own = (
            labelled[labelled["label_source"] == name]
            .drop_duplicates(subset=key, keep="first")
            .set_index(key)["label"]
            .rename(f"label__{name}")
        )
        table = table.join(own, how="left")

    return table.reset_index(), int(len(conflicted))


def label_sources_in(table: pd.DataFrame) -> list[str]:
    """Labelling sources present in an assembled table, in precedence order."""
    found = [c[len("label__"):] for c in table.columns if c.startswith("label__")]
    return sorted(found, key=lambda name: LABEL_PRECEDENCE.get(name, 99))


def resolve_labels(table: pd.DataFrame, sources: Sequence[str]) -> pd.Series:
    """Training labels from a chosen subset of label sources.

    Same rules as assembly — precedence decides, contradictions are withheld —
    but applied only across `sources`. A variant ClinVar calls pathogenic and a
    DMS assay calls benign is a conflict when training on both, and simply a
    ClinVar label when training on ClinVar alone.
    """
    missing = [s for s in sources if f"label__{s}" not in table.columns]
    if missing:
        raise ValueError(
            f"No labels from {missing} in this table; it carries labels from "
            f"{label_sources_in(table)}. Rebuild with those sources included."
        )
    ordered = sorted(sources, key=lambda name: LABEL_PRECEDENCE.get(name, 99))
    stacked = table[[f"label__{name}" for name in ordered]]
    resolved = stacked.bfill(axis=1).iloc[:, 0]
    conflicted = stacked.nunique(axis=1) > 1
    if conflicted.any():
        logger.info("Training on %s: %d contradictory variants withheld.",
                    "+".join(ordered), int(conflicted.sum()))
    return resolved.mask(conflicted)


def assemble(
    frames: Mapping[str, pd.DataFrame],
    sequences: Mapping[str, str] | None = None,
    expected_genes: Sequence[str] | None = None,
    anchor_column: str | None = "feature_alphamissense_score",
    out_path: Path | str | None = None,
) -> tuple[pd.DataFrame, AssemblyReport]:
    """Merge per-source frames into one table.

    `frames` maps source name to its loaded records. Passing one entry is the
    single-source arm of the experiment; passing all of them is the pooled arm.
    """
    report = AssemblyReport(sources=sorted(frames))
    if not frames:
        raise ValueError("assemble() needs at least one source.")

    validated: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        missing = set(RECORD_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Source '{name}' does not emit the record schema; "
                f"missing {sorted(missing)}."
            )
        kept, dropped = validate_frame(frame, sequences=dict(sequences or {}))
        validated[name] = kept
        report.rows_per_source[name] = len(kept)
        report.dropped_per_source[name] = dropped

    combined = pd.concat(validated.values(), ignore_index=True, sort=False)
    table, conflicts = _merge_sources(combined)
    report.conflicts_quarantined = conflicts

    # Absence from gnomAD is evidence (PM2), not missingness. Applied here, not
    # left to the caller, because forgetting it is silent: median imputation
    # would quietly turn every unobserved variant into a typical observed one.
    if "gnomad" in frames:
        from vpdl.sources.gnomad import fill_unobserved
        table = fill_unobserved(table)

    # Orientation is checked AFTER the merge, deliberately. Before it, a
    # ClinVar label and the AlphaMissense anchor for the same variant sit on
    # different rows, so the check found zero overlapping rows and silently
    # never ran — the one guard that caught v1's 185,000-label inversion.
    if anchor_column and anchor_column in table.columns:
        anchor = pd.to_numeric(table[anchor_column], errors="coerce")
        for name in validated:
            rows = table["label_source"].eq(name) & table["label"].notna()
            overlap = int((rows & anchor.notna()).sum())
            if overlap >= 30:
                report.orientation_checks[name] = assert_label_orientation(
                    table.loc[rows & anchor.notna(), "label"].to_numpy(),
                    anchor[rows & anchor.notna()].to_numpy(),
                    source=name,
                )
            elif rows.any():
                logger.warning(
                    "Source '%s': only %d labelled rows overlap the anchor; "
                    "orientation UNVERIFIED (needs >= 30).", name, overlap,
                )
    else:
        logger.warning(
            "No anchor column (%s) available — label orientation is UNVERIFIED "
            "for this assembly. This is the check that caught the ProteinGym "
            "inversion; include alphamissense in every build.", anchor_column
        )

    if expected_genes:
        assert_genes_present(table, expected_genes)

    labels = table["label"].dropna()
    report.label_counts = {
        "benign": int((labels == 0).sum()),
        "pathogenic": int((labels == 1).sum()),
        "unlabelled": int(table["label"].isna().sum()),
    }
    report.feature_columns = sorted(
        column for column in table.columns if column.startswith("feature_")
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)

    return table.reset_index(drop=True), report
