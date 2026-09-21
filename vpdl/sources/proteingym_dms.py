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


def gene_from_assay_name(name: str) -> str:
    """``MSH2_HUMAN_Jia_2020.csv`` -> ``MSH2``.

    ProteinGym per-assay CSVs carry no gene column: the assay's identity lives
    in its filename, and the reference file's ``UniProt_ID`` is an *entry name*
    (``MSH2_HUMAN``), not an accession. Both reduce to the first token.
    """
    return Path(name).stem.split("_")[0].upper()


def _parse_assay(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "mutant" not in frame.columns:
        raise ValueError(
            f"{source_name} lacks a 'mutant' column; this does not look like a "
            "ProteinGym substitutions table."
        )
    # Multi-mutant entries ("A12V:G45D") fail this pattern and are dropped:
    # a double mutant is not a single-variant observation.
    parsed = frame["mutant"].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
    parsed.columns = ["wt_aa", "position", "mut_aa"]
    frame = pd.concat([frame.reset_index(drop=True), parsed], axis=1)
    frame = frame[frame["position"].notna()].copy()
    frame["position"] = frame["position"].astype(int)
    return frame


def load(
    path: Path | str,
    uniprot_by_gene: Mapping[str, str],
    genes: Sequence[str] | None = None,
    include_score_feature: bool = False,
) -> pd.DataFrame:
    """Load ProteinGym DMS data into the record schema.

    Accepts either the release archive ``DMS_ProteinGym_substitutions.zip``
    (every assay, filtered to `genes` while reading) or a single per-assay CSV.
    `genes` defaults to the panel in `uniprot_by_gene`, so pointing this at the
    full archive does not pull in 200+ unrelated assays.

    `include_score_feature` defaults to **False**. When this source supplies
    labels, ``DMS_score`` is the continuous value the label was binarised from,
    so feeding it back in as a feature is the label in disguise — v1's
    ``dms_bin_median`` leak, which produced a fake 0.9987 AUC. The
    ``assert_no_label_proxy`` guard would refuse it anyway; defaulting it off
    means a normal build does not trip that guard.
    """
    import zipfile

    path = Path(path)
    wanted = set(genes) if genes else set(uniprot_by_gene)
    assays: list[pd.DataFrame] = []

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".csv"):
                    continue
                gene = gene_from_assay_name(member)
                if gene not in wanted:
                    continue
                with archive.open(member) as handle:
                    frame = _parse_assay(pd.read_csv(handle), member)
                frame["gene"] = gene
                frame["_assay"] = Path(member).stem
                assays.append(frame)
    else:
        frame = _parse_assay(pd.read_csv(path), str(path))
        column = next((c for c in ("gene", "gene_name") if c in frame.columns), None)
        frame["gene"] = (
            frame[column].astype(str).str.upper() if column
            else gene_from_assay_name(path.name)
        )
        frame["_assay"] = path.stem
        assays.append(frame[frame["gene"].isin(wanted)])

    if not assays:
        logger.warning("ProteinGym: no assays found for %s in %s", sorted(wanted), path)
        return pd.DataFrame(columns=["uniprot_id", "position", "wt_aa", "mut_aa",
                                     "gene", "label", "label_source",
                                     "evidence_tier"])

    frame = pd.concat(assays, ignore_index=True)
    logger.info("ProteinGym: %d assay(s) matched: %s",
                frame["_assay"].nunique(), sorted(frame["_assay"].unique()))

    records = pd.DataFrame({
        "uniprot_id": frame["gene"].map(uniprot_by_gene),
        "position": frame["position"],
        "wt_aa": frame["wt_aa"],
        "mut_aa": frame["mut_aa"],
        "gene": frame["gene"],
        "label": frame.get("DMS_score_bin", pd.Series(np.nan, index=frame.index)).map(to_label),
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
