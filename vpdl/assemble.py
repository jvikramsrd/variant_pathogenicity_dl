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


def _resolve_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse duplicate variant rows by label precedence, quarantining conflicts."""
    frame = frame.copy()
    frame["_precedence"] = (
        frame["label_source"].map(LABEL_PRECEDENCE).fillna(99).astype(int)
    )
    frame = frame.sort_values(["_precedence"], kind="mergesort")

    labelled = frame[frame["label"].notna()]
    conflicts = (
        labelled.groupby(list(VARIANT_KEY))["label"].nunique(dropna=True) > 1
    )
    conflicted_keys = set(conflicts[conflicts].index)

    if conflicted_keys:
        logger.warning(
            "%d variants carry contradictory labels across sources; their labels "
            "are withheld and the rows kept unlabelled.", len(conflicted_keys)
        )
        key_tuples = list(zip(*[frame[column] for column in VARIANT_KEY]))
        is_conflicted = np.array([key in conflicted_keys for key in key_tuples])
        frame.loc[is_conflicted, "label"] = np.nan
        frame.loc[is_conflicted, "label_source"] = "conflict_quarantined"

    resolved = frame.drop_duplicates(subset=list(VARIANT_KEY), keep="first")
    return resolved.drop(columns="_precedence"), len(conflicted_keys)


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

    # Orientation check before precedence collapses anything: each labelling
    # source is tested against an independent prior on its own rows.
    if anchor_column and anchor_column in combined.columns:
        anchor = pd.to_numeric(combined[anchor_column], errors="coerce")
        for name in validated:
            rows = combined["label_source"].eq(name) & combined["label"].notna()
            if rows.sum() >= 30 and anchor[rows].notna().sum() >= 30:
                report.orientation_checks[name] = assert_label_orientation(
                    combined.loc[rows, "label"].to_numpy(),
                    anchor[rows].to_numpy(),
                    source=name,
                )
    else:
        logger.warning(
            "No anchor column (%s) available — label orientation is UNVERIFIED "
            "for this assembly. This is the check that caught the ProteinGym "
            "inversion; prefer including a prior source.", anchor_column
        )

    table, conflicts = _resolve_labels(combined)
    report.conflicts_quarantined = conflicts

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
