"""Genomic-context features from the transcript's exon structure.

Protein coordinates alone cannot say that a missense change sits on an exon
junction — where a "missense" variant may really act through splicing, the
mechanism a cell-free functional assay cannot see and a protein model cannot
represent. The transcript's exon table can, for every substitution alike:

    feature_genomic_exon_rel        coding-exon index scaled to [0, 1]
    feature_genomic_to_junction     codons to the nearest exon junction
    feature_genomic_junction_codon  1 when the codon spans a junction
    feature_genomic_cds_rel         codon / protein length

Nucleotide-level features (transition/transversion, codon position) exist only
for variants with a ClinVar record, so their missingness would tell a model
which rows are clinical — a label-source shortcut in pooled arms. They are
off by default (``include_nucleotide=False``).

Exon coordinates come from Ensembl (MANE Select transcripts, cached JSON); the
walk from exons to codons is v1's ``scripts/derive_pms2_homology_range.py``,
self-validated against the pinned protein length. The same table re-derives
the PMS2CL homology codon range, so the constant in the ClinVar source is
checked rather than trusted.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from vpdl.dl.hgvs import parse_coding_snv

logger = logging.getLogger(__name__)

__all__ = ["MANE_ENSEMBL", "GENOMIC_COLUMNS", "fetch_transcript", "exon_codon_table",
           "homology_codon_range", "genomic_features"]

# Ensembl IDs of the MANE Select transcripts pinned in vpdl.dl.hgvs.MANE_TRANSCRIPTS.
MANE_ENSEMBL: dict[str, str] = {
    "MLH1": "ENST00000231790",
    "MSH2": "ENST00000233146",
    "MSH6": "ENST00000234420",
    "PMS2": "ENST00000265849",
}
GENOMIC_COLUMNS = ("feature_genomic_exon_rel", "feature_genomic_to_junction",
                   "feature_genomic_junction_codon", "feature_genomic_cds_rel")
ENSEMBL_REST = "https://rest.ensembl.org"


def fetch_transcript(transcript_id: str, cache_dir: Path | str, timeout: int = 60) -> dict:
    """Ensembl ``lookup/id/<id>?expand=1``, cached as JSON."""
    cache = Path(cache_dir) / f"{transcript_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    request = urllib.request.Request(f"{ENSEMBL_REST}/lookup/id/{transcript_id}?expand=1",
                                     headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def exon_codon_table(transcript: Mapping, protein_length: int) -> pd.DataFrame:
    """Per-exon c. and codon intervals in transcript order (ported from v1).

    Raises if the CDS does not encode exactly `protein_length` residues plus a
    stop codon — a moved transcript or reference must stop the build.
    """
    strand = int(transcript["strand"])
    cds_lo = int(transcript["Translation"]["start"])
    cds_hi = int(transcript["Translation"]["end"])
    exons = sorted(transcript["Exon"], key=lambda e: int(e["start"]), reverse=strand == -1)
    rows, consumed = [], 0
    for number, exon in enumerate(exons, start=1):
        lo, hi = max(int(exon["start"]), cds_lo), min(int(exon["end"]), cds_hi)
        coding = max(0, hi - lo + 1)
        if not coding:
            continue
        c_start, c_end = consumed + 1, consumed + coding
        consumed += coding
        rows.append({"exon": number, "c_start": c_start, "c_end": c_end,
                     "codon_start": (c_start + 2) // 3, "codon_end": (c_end + 2) // 3})
    if consumed // 3 - 1 != protein_length:
        raise ValueError(f"{transcript.get('id')}: CDS of {consumed} bp implies "
                         f"{consumed // 3 - 1} aa, pinned protein has {protein_length}")
    return pd.DataFrame(rows)


def homology_codon_range(table: pd.DataFrame, protein_length: int,
                         exons: tuple[int, int] = (11, 15)) -> tuple[int, int]:
    """Protein codons covered by `exons` (PMS2CL: 11-15), stop codon excluded."""
    by_exon = table.set_index("exon")
    first, last = exons
    return int(by_exon.loc[first, "codon_start"]), min(int(by_exon.loc[last, "codon_end"]),
                                                       protein_length)


def genomic_features(variants: pd.DataFrame, exon_tables: Mapping[str, pd.DataFrame],
                     protein_lengths: Mapping[str, int],
                     include_nucleotide: bool = False) -> pd.DataFrame:
    """Feature columns for `variants` (needs ``gene, position``; ``hgvs_c`` optional).

    Genes without an exon table get NaN — missing, reported, never guessed.
    """
    out = pd.DataFrame(np.nan, index=variants.index, columns=list(GENOMIC_COLUMNS))
    for gene, rows in variants.groupby("gene"):
        table = exon_tables.get(gene)
        if table is None:
            logger.warning("no exon table for %s: genomic features left missing", gene)
            continue
        length = protein_lengths[gene]
        codons = rows["position"].astype(int).to_numpy()
        starts, ends = table["codon_start"].to_numpy(), table["codon_end"].to_numpy()
        # A junction lies between exon k's last codon and exon k+1's first; a
        # codon split across it has codon_end(k) == codon_start(k+1).
        junction_codons = set(ends[:-1][ends[:-1] == starts[1:]])
        boundaries = np.concatenate([ends[:-1], starts[1:]])
        exon_index = np.searchsorted(ends, codons, side="left")
        exon_index = np.clip(exon_index, 0, len(table) - 1)
        out.loc[rows.index, "feature_genomic_exon_rel"] = (
            exon_index / max(1, len(table) - 1))
        out.loc[rows.index, "feature_genomic_to_junction"] = (
            np.abs(codons[:, None] - boundaries[None, :]).min(axis=1)
            if len(boundaries) else np.nan)
        out.loc[rows.index, "feature_genomic_junction_codon"] = [
            float(c in junction_codons) for c in codons]
        out.loc[rows.index, "feature_genomic_cds_rel"] = codons / length
    if include_nucleotide and "hgvs_c" in variants.columns:
        snvs = variants["hgvs_c"].map(lambda v: parse_coding_snv(
            v.split(";")[0] if isinstance(v, str) else None))
        out["feature_genomic_transition"] = snvs.map(
            lambda s: float(s.is_transition) if s else np.nan)
        for k in (1, 2, 3):
            out[f"feature_genomic_codon_pos{k}"] = snvs.map(
                lambda s, k=k: float(s.codon_position == k) if s else np.nan)
    return out
