"""ProteinGym deep-mutational-scanning assays.

Satisfies regression landmine L1 — the label orientation that cost v1 roughly
185,000 inverted labels.

READ THIS BEFORE CHANGING ``to_label``:

ProteinGym's ``DMS_score_bin`` is 1 for the **top fitness half**. High fitness
means the protein still works, which means the variant is **tolerated**, which
means **benign**, which is label **0** in this project's convention
(1 = pathogenic). v1 mapped ``DMS_score_bin == 1`` straight to pathogenic. The
pipeline did not fail; it trained on backwards supervision and produced
plausible numbers for weeks. It was caught only by anchoring the merged labels
against AlphaMissense, which is why :func:`vpdl.assemble.assert_label_orientation`
runs on every labelling source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.sources.base import SourceCapabilities

logger = logging.getLogger(__name__)

__all__ = ["PROTEINGYM_VERSION", "provides", "to_label", "load"]

PROTEINGYM_VERSION = "v1.3"


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="pg_dms",
        supplies_labels=True,
        label_precedence=4,
        feature_columns=("feature_dms_score",),
        licence="MIT",
        notes=(
            "Assay-binarised labels. On the MMR panel the entire DMS pool comes "
            "from ONE MSH2 assay, so under leave-one-gene-out it contributes "
            "nothing when MSH2 is held out and single-gene signal otherwise — "
            "any label-source comparison using it is confounded with gene "
            "identity and must say so."
        ),
    )


def to_label(dms_score_bin: int | float | None) -> float:
    """``DMS_score_bin`` to this project's label convention.

    bin == 1  ->  top fitness half  ->  tolerated  ->  benign  ->  0
    bin == 0  ->  bottom half        ->  damaging  ->  pathogenic -> 1
    """
    if dms_score_bin is None or (
        isinstance(dms_score_bin, float) and np.isnan(dms_score_bin)
    ):
        return np.nan
    value = int(dms_score_bin)
    if value not in (0, 1):
        return np.nan
    return 0.0 if value == 1 else 1.0


def load(
    path: Path | str,
    uniprot_by_gene: Mapping[str, str],
    genes: Sequence[str] | None = None,
    include_score_feature: bool = True,
) -> pd.DataFrame:
    """Load a ProteinGym substitutions table into the record schema.

    Expects the standard ProteinGym columns ``mutant`` (e.g. ``A123V``),
    ``DMS_score`` and ``DMS_score_bin``, plus a gene or UniProt identifier.
    """
    frame = pd.read_csv(path)

    if "mutant" not in frame.columns:
        raise ValueError(
            f"{path} lacks a 'mutant' column; this does not look like a "
            "ProteinGym substitutions table."
        )

    parsed = frame["mutant"].astype(str).str.extract(
        r"^([A-Z])(\d+)([A-Z])$"
    )
    parsed.columns = ["wt_aa", "position", "mut_aa"]
    frame = pd.concat([frame, parsed], axis=1)
    frame = frame[frame["position"].notna()].copy()
    frame["position"] = frame["position"].astype(int)

    gene_column = next(
        (name for name in ("gene", "gene_name", "UniProt_ID", "DMS_id")
         if name in frame.columns),
        None,
    )
    if gene_column is None:
        raise ValueError(f"{path} has no gene identifier column.")
    frame["gene"] = frame[gene_column].astype(str).str.split("_").str[0].str.upper()

    if genes:
        frame = frame[frame["gene"].isin(set(genes))]

    records = pd.DataFrame({
        "uniprot_id": frame["gene"].map(uniprot_by_gene),
        "position": frame["position"],
        "wt_aa": frame["wt_aa"],
        "mut_aa": frame["mut_aa"],
        "gene": frame["gene"],
        "label": frame.get("DMS_score_bin", pd.Series(index=frame.index)).map(to_label),
        "label_source": "pg_dms",
        "evidence_tier": f"proteingym_{PROTEINGYM_VERSION}_dms",
    })

    if include_score_feature and "DMS_score" in frame.columns:
        records["feature_dms_score"] = pd.to_numeric(
            frame["DMS_score"], errors="coerce"
        )

    logger.info(
        "ProteinGym DMS: %d rows, %d labelled (%d benign / %d pathogenic)",
        len(records), int(records["label"].notna().sum()),
        int((records["label"] == 0).sum()), int((records["label"] == 1).sum()),
    )
    return records.reset_index(drop=True)
