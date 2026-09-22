"""HGVS and variant-identity normalisation.

One variant has many spellings: ``p.Arg123His``, ``p.(Arg123His)``, ``p.R123H``,
``R123H``, and the ClinVar ``Name`` field
``NM_000249.4(MLH1):c.367C>T (p.Arg123His)``. Everything downstream keys on one
identity, :func:`variant_id`, which is deliberately the same string as
:func:`vpdl.splits.variant_keys` so predictions, features and the DL output
join without a translation table.

Nothing here guesses. A notation that is not a clean single-residue missense
returns ``None``; a c./p. pair that disagree about the codon is reported as
inconsistent rather than repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "THREE_TO_ONE",
    "ONE_TO_THREE",
    "MANE_TRANSCRIPTS",
    "variant_id",
    "parse_variant_id",
    "normalize_hgvs_p",
    "format_hgvs_p",
    "ClinVarName",
    "parse_clinvar_name",
    "CodingSNV",
    "parse_coding_snv",
    "codon_of",
    "transcript_status",
    "hgvs_consistency",
]

THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
ONE_TO_THREE: dict[str, str] = {one: three for three, one in THREE_TO_ONE.items()}

# MANE Select transcripts for the panel (RefSeq accession.version). The
# canonical builder reports every ClinVar record whose transcript differs, and
# counts which transcript the records actually use, so a wrong pin is visible
# on the first build rather than assumed correct. Protein coordinates are
# validated against UniProt regardless.
MANE_TRANSCRIPTS: dict[str, str] = {
    "MLH1": "NM_000249.4",
    "MSH2": "NM_000251.3",
    "MSH6": "NM_000179.3",
    "PMS2": "NM_000535.7",
}

_P_THREE = re.compile(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")
_P_ONE = re.compile(r"^(?:p\.)?\(?([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])\)?$")
_NAME = re.compile(
    r"^(?P<transcript>[NX]M_\d+(?:\.\d+)?)"
    r"(?:\((?P<gene>[^)]+)\))?"
    r":(?P<hgvs_c>c\.[^\s(]+)"
    r"(?:\s*\((?P<hgvs_p>p\.[^)]*\)?)\))?"
)
_C_SNV = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$")


def variant_id(uniprot_id: str, position: int, wt_aa: str, mut_aa: str) -> str:
    """``P40692:123:R>H`` — identical to :func:`vpdl.splits.variant_keys`."""
    return f"{uniprot_id}:{int(position)}:{wt_aa}>{mut_aa}"


def parse_variant_id(value: str) -> tuple[str, int, str, str]:
    """Inverse of :func:`variant_id`. Raises on anything else."""
    try:
        accession, position, change = value.split(":")
        wt_aa, mut_aa = change.split(">")
        return accession, int(position), wt_aa, mut_aa
    except ValueError as error:
        raise ValueError(f"not a variant id: {value!r}") from error


def normalize_hgvs_p(value: object) -> tuple[str, int, str] | None:
    """``(wt, position, mut)`` for a clean single-residue missense, else None.

    Accepts three-letter (``p.Arg123His``, ``p.(Arg123His)``) and one-letter
    (``p.R123H``, ``R123H``) forms. Synonymous (``=``), nonsense (``Ter``/``*``),
    frameshift, indels, extensions and uncertain positions all return None —
    they are not substitutions this pipeline can score.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = _P_THREE.match(text)
    if match:
        wt3, position, mut3 = match.groups()
        wt, mut = THREE_TO_ONE.get(wt3), THREE_TO_ONE.get(mut3)
    else:
        match = _P_ONE.match(text)
        if not match:
            return None
        wt, position, mut = match.groups()
    if wt is None or mut is None or wt == mut or int(position) < 1:
        return None
    return wt, int(position), mut


def format_hgvs_p(wt_aa: str, position: int, mut_aa: str) -> str:
    """Normalised three-letter form, ``p.Arg123His``."""
    return f"p.{ONE_TO_THREE[wt_aa]}{int(position)}{ONE_TO_THREE[mut_aa]}"


@dataclass(frozen=True)
class ClinVarName:
    transcript: str | None
    gene: str | None
    hgvs_c: str | None
    hgvs_p: str | None


def parse_clinvar_name(name: object) -> ClinVarName:
    """Split ClinVar's ``Name`` field into transcript, gene, c. and p. parts.

    ``NM_000249.4(MLH1):c.367C>T (p.Arg123His)`` ->
    ``ClinVarName("NM_000249.4", "MLH1", "c.367C>T", "p.Arg123His")``.
    Unparseable names return all-None fields rather than raising: many ClinVar
    records are not transcript-level HGVS at all (cytogenetic, genomic-only).
    """
    if not isinstance(name, str):
        return ClinVarName(None, None, None, None)
    match = _NAME.match(name.strip())
    if not match:
        return ClinVarName(None, None, None, None)
    hgvs_p = match.group("hgvs_p")
    if hgvs_p and hgvs_p.count("(") < hgvs_p.count(")"):
        hgvs_p = hgvs_p.rstrip(")")
    return ClinVarName(match.group("transcript"), match.group("gene"),
                       match.group("hgvs_c"), hgvs_p)


@dataclass(frozen=True)
class CodingSNV:
    cds_position: int
    ref: str
    alt: str

    @property
    def codon(self) -> int:
        return codon_of(self.cds_position)

    @property
    def codon_position(self) -> int:
        """1, 2 or 3 — which base of the codon changed."""
        return (self.cds_position - 1) % 3 + 1

    @property
    def is_transition(self) -> bool:
        return {self.ref, self.alt} in ({"A", "G"}, {"C", "T"})


def parse_coding_snv(hgvs_c: object) -> CodingSNV | None:
    """Exonic coding SNV ``c.367C>T`` only; intronic offsets (``c.367+1G>A``),
    UTR (``c.-12``, ``c.*5``), indels and delins return None."""
    if not isinstance(hgvs_c, str):
        return None
    match = _C_SNV.match(hgvs_c.strip())
    if not match:
        return None
    position, ref, alt = match.groups()
    return CodingSNV(int(position), ref, alt)


def codon_of(cds_position: int) -> int:
    """1-based codon containing a 1-based CDS nucleotide position."""
    if cds_position < 1:
        raise ValueError(f"CDS position must be >= 1, got {cds_position}")
    return (int(cds_position) - 1) // 3 + 1


def transcript_status(gene: str, transcript: str | None,
                      mane: Mapping[str, str] = MANE_TRANSCRIPTS) -> str:
    """``mane`` | ``mane_version_mismatch`` | ``non_mane`` | ``absent`` | ``unpinned``."""
    if not transcript:
        return "absent"
    pinned = mane.get(gene)
    if pinned is None:
        return "unpinned"
    if transcript == pinned:
        return "mane"
    if transcript.split(".")[0] == pinned.split(".")[0]:
        return "mane_version_mismatch"
    return "non_mane"


def hgvs_consistency(hgvs_c: str | None, protein_position: int) -> str:
    """Do a record's c. and p. agree on the codon?

    ``consistent`` | ``inconsistent`` | ``not_checkable`` (no exonic SNV to
    check — indels, delins, or no c. at all). An inconsistent pair is a
    transcript or annotation problem and is flagged, never corrected.
    """
    snv = parse_coding_snv(hgvs_c)
    if snv is None:
        return "not_checkable"
    return "consistent" if snv.codon == int(protein_position) else "inconsistent"
