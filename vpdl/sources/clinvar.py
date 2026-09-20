"""ClinVar clinical assertions — the primary label source.

Satisfies regression landmine L9 (the PMS2 homology gate is fail-closed, and
gating withholds supervision without deleting the gene).
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.sources.base import VALID_AA, SourceCapabilities

logger = logging.getLogger(__name__)

__all__ = [
    "CLINVAR_URL",
    "PMS2_HOMOLOGY_CODONS",
    "provides",
    "parse_hgvs_p",
    "significance_to_label",
    "apply_homology_gate",
    "load",
]

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# PMS2CL pseudogene homology region in protein coordinates, derived from the
# exon 11-15 span. Short-read clinical calls inside it are not trustworthy
# without orthogonal confirmation (long-range PCR or cDNA).
PMS2_HOMOLOGY_CODONS: tuple[int, int] = (382, 862)

_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}

_HGVS_P = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")

_PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
_BENIGN = {"benign", "likely benign", "benign/likely benign"}


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="clinvar",
        supplies_labels=True,
        label_precedence=1,
        licence="public domain (NCBI)",
        notes="Only assertions at >=2 review stars are eligible for supervision.",
    )


def parse_hgvs_p(name: str) -> tuple[str, int, str] | None:
    """Extract ``(wt_aa, position, mut_aa)`` from an HGVS protein change.

    Returns None for anything that is not a clean single-residue missense —
    synonymous, nonsense, frameshift and indel notations all fall out here
    rather than being coerced into a substitution.
    """
    if not isinstance(name, str):
        return None
    match = _HGVS_P.search(name)
    if not match:
        return None
    wt, position, mut = match.groups()
    wt_aa, mut_aa = _THREE_TO_ONE.get(wt), _THREE_TO_ONE.get(mut)
    if wt_aa is None or mut_aa is None or wt_aa == mut_aa:
        return None
    if wt_aa not in VALID_AA or mut_aa not in VALID_AA:
        return None
    return wt_aa, int(position), mut_aa


def significance_to_label(significance: str) -> float:
    """Map a clinical significance string to 1 / 0 / NaN.

    Conflicting and uncertain assertions become NaN deliberately: they are kept
    as rows (they are the VUS this project exists to resolve) but must never
    supply supervision.
    """
    if not isinstance(significance, str):
        return np.nan
    normalised = significance.strip().lower()
    if normalised in _PATHOGENIC:
        return 1.0
    if normalised in _BENIGN:
        return 0.0
    return np.nan


def apply_homology_gate(
    df: pd.DataFrame,
    policy: Mapping | None = None,
) -> pd.DataFrame:
    """Withhold supervision inside the PMS2CL homology region. Fail-closed.

    `policy` must be one of:

    * ``{"codon_range": (start, end)}`` — withhold labels inside the span.
    * ``{"confirmed": <iterable of (position, wt, mut)>}`` — withhold inside the
      default span except for orthogonally confirmed variants.
    * ``{"exclude_gene": True}`` — drop PMS2 outright. Use only knowingly.

    Passing ``None`` with PMS2 rows present raises. A soft default would
    silently trust untrustworthy short-read calls, and the opposite error is
    just as expensive: on 2026-09-13 a build passed the exclude-gene option and
    produced a silently three-gene table that sixteen grid cells then trained
    on. Neither behaviour is something to arrive at by default.
    """
    out = df.copy()
    has_pms2 = (out["gene"] == "PMS2").any()

    if "homology_excluded" not in out.columns:
        out["homology_excluded"] = 0

    if not has_pms2:
        return out

    if policy is None:
        raise ValueError(
            "PMS2 rows present but no homology policy given. Pass "
            "{'codon_range': (382, 862)}, {'confirmed': [...]}, or "
            "{'exclude_gene': True}. Refusing to trust short-read calls in the "
            "PMS2CL homology region by default."
        )

    if policy.get("exclude_gene"):
        dropped = int(has_pms2 and (out["gene"] == "PMS2").sum())
        logger.warning(
            "PMS2 excluded entirely (%d rows). The resulting table has THREE "
            "genes; any leave-one-gene-out evaluation over it is a three-fold "
            "design and must be reported as such.", dropped
        )
        return out[out["gene"] != "PMS2"].reset_index(drop=True)

    start, end = tuple(policy.get("codon_range", PMS2_HOMOLOGY_CODONS))
    in_region = (
        (out["gene"] == "PMS2")
        & (pd.to_numeric(out["position"], errors="coerce") >= start)
        & (pd.to_numeric(out["position"], errors="coerce") <= end)
    )

    confirmed = policy.get("confirmed")
    if confirmed:
        keys = {tuple(item) for item in confirmed}
        is_confirmed = out.apply(
            lambda row: (row["position"], row.get("wt_aa"), row.get("mut_aa")) in keys,
            axis=1,
        )
        in_region &= ~is_confirmed

    out.loc[in_region, "homology_excluded"] = 1
    if "label" in out.columns:
        out.loc[in_region, "label"] = np.nan

    logger.info(
        "PMS2 homology gate: %d of %d PMS2 rows withheld from supervision "
        "(codons %d-%d); the gene itself is retained.",
        int(in_region.sum()), int((out["gene"] == "PMS2").sum()), start, end,
    )
    return out


def load(
    path: Path | str,
    genes: Sequence[str],
    uniprot_by_gene: Mapping[str, str],
    min_stars: int = 2,
    pms2_policy: Mapping | None = None,
) -> pd.DataFrame:
    """Stream variant_summary.txt.gz once and emit the record schema.

    One pass for all requested genes. v1 streamed the whole ~4 GB decompressed
    file once *per gene* (CODE_REVIEW D1).
    """
    wanted = set(genes)
    rows: list[dict] = []
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            gene = fields[index.get("GeneSymbol", 0)]
            if gene not in wanted:
                continue
            stars = _review_stars(fields[index.get("ReviewStatus", 0)])
            if stars < min_stars:
                continue
            parsed = parse_hgvs_p(fields[index.get("Name", 0)])
            if parsed is None:
                continue
            wt_aa, position, mut_aa = parsed
            rows.append({
                "uniprot_id": uniprot_by_gene.get(gene),
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "gene": gene,
                "label": significance_to_label(
                    fields[index.get("ClinicalSignificance", 0)]
                ),
                "label_source": "clinvar",
                "evidence_tier": fields[index.get("ReviewStatus", 0)],
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        logger.warning("ClinVar yielded no rows for %s", sorted(wanted))
        return frame

    frame = frame.drop_duplicates(
        subset=["uniprot_id", "position", "wt_aa", "mut_aa"], keep="first"
    )
    return apply_homology_gate(frame, policy=pms2_policy).reset_index(drop=True)


_STAR_MAP = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
}


def _review_stars(review_status: str) -> int:
    return _STAR_MAP.get((review_status or "").strip().lower(), 0)
