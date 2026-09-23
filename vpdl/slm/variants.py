"""Variant identity and a coarse consequence for EVERY variant type, not just missense.

The DL branch keys on protein substitutions (``UniProt:pos:wt>mut``) because
its models read protein sequence. The SLM reads clinical text about any
variant — splice, UTR, frameshift, copy number — so it keys on ClinVar's
VariationID and derives a consequence class from the HGVS name.

The consequence is derived from notation alone:

    p.Gly67Arg           missense           p.Arg12Ter / p.R12*   nonsense
    p.Gly67=             synonymous         p.Lys5fs              frameshift
    p.Met1? / p.Met1Val  start_loss         p.Ter757Glnext*?      stop_loss
    p.Lys12del           inframe_indel      c.123+1G>A            splice_site (+/-1, 2)
    c.123+5G>A           splice_region      c.123+60G>A           intronic
    c.-20C>T             utr5_or_upstream   c.*40A>G              utr3_or_downstream
    copy number / >1 kb  cnv / structural   anything else         other

It is a notation-level class, deliberately conservative: an exonic variant
next to a splice site is "missense" here even if it disrupts splicing, and a
far-upstream c.-2000 is not called "promoter". A transcript-aware annotation
(VEP on the DGX) would refine this; until then the class says only what the
name says. ``consequence_source`` records that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["CONSEQUENCES", "ParsedName", "parse_name", "consequence", "DLJoin",
           "MMR_GENES", "CONSEQUENCE_SOURCE"]

CONSEQUENCE_SOURCE = "hgvs-notation/v1"

CONSEQUENCES = ("missense", "nonsense", "synonymous", "frameshift", "inframe_indel",
                "start_loss", "stop_loss", "splice_site", "splice_region", "intronic",
                "utr5_or_upstream", "utr3_or_downstream", "noncoding_transcript",
                "cnv", "structural", "other")

MMR_GENES = ("MLH1", "MSH2", "MSH6", "PMS2")

_NAME = re.compile(
    r"^(?P<transcript>[A-Z]{2}_\d+(?:\.\d+)?)"
    r"(?:\((?P<gene>[^)]+)\))?"
    r":(?P<change>[cngm]\.[^\s(]+)"
    r"(?:\s*\((?P<protein>p\.[^)]*\)?)\))?")
_PROTEIN_ONLY = re.compile(r"\((?P<protein>p\.[^)]*\)?)\)")
_AA3 = r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Sec|Pyl|Xaa)"
_P_MISSENSE = re.compile(rf"^p\.\(?({_AA3})(\d+)({_AA3})\)?$")
_P_NONSENSE = re.compile(rf"^p\.\(?({_AA3})(\d+)(?:Ter|\*|X)\)?$")
_P_SYNONYMOUS = re.compile(rf"^p\.\(?({_AA3})(\d+)=\)?$|^p\.\(?=\)?$")
_P_FRAMESHIFT = re.compile(r"fs")
_P_START = re.compile(r"^p\.\(?Met1(?:[?]|[A-Z][a-z]{2}|\?|ext)")
_P_STOP_LOSS = re.compile(r"^p\.\(?(?:Ter|\*)\d+[A-Za-z]*ext")
_P_INFRAME = re.compile(r"(?:del|dup|ins)")
_C_OFFSET = re.compile(r"^[cn]\.(?P<pos>[-*]?\d+)(?P<offset>[+-]\d+)?")

# ClinVar Type values that are not sequence-level changes.
_CNV_TYPES = {"copy number loss", "copy number gain", "deletion (cnv)"}
_STRUCTURAL_TYPES = {"inversion", "translocation", "complex", "fusion", "tandem duplication",
                     "variation"}


@dataclass(frozen=True)
class ParsedName:
    transcript: str | None
    gene: str | None
    change: str | None          # c./n./g./m. part
    protein: str | None         # p. part, as written


def parse_name(name: object) -> ParsedName:
    """``NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)`` -> its parts. Never guesses."""
    if not isinstance(name, str):
        return ParsedName(None, None, None, None)
    match = _NAME.match(name.strip())
    if match:
        return ParsedName(match.group("transcript"), match.group("gene"),
                          match.group("change"), match.group("protein"))
    protein = _PROTEIN_ONLY.search(name)
    return ParsedName(None, None, None, protein.group("protein") if protein else None)


def _coding_consequence(change: str) -> str | None:
    match = _C_OFFSET.match(change)
    if not match:
        return None
    position, offset = match.group("pos"), match.group("offset")
    if offset:
        distance = abs(int(offset))
        if position.startswith("-"):
            return "utr5_or_upstream"
        if position.startswith("*"):
            return "utr3_or_downstream"
        if distance <= 2:
            return "splice_site"
        if distance <= 8:
            return "splice_region"
        return "intronic"
    if position.startswith("-"):
        return "utr5_or_upstream"
    if position.startswith("*"):
        return "utr3_or_downstream"
    return None


def consequence(variant_type: object, name: object, length: int | None = None) -> str:
    """Coarse consequence class; see the module docstring for the rules."""
    kind = str(variant_type or "").strip().lower()
    if kind in _CNV_TYPES:
        return "cnv"
    if kind in _STRUCTURAL_TYPES:
        return "structural"
    if length is not None and length > 1000:
        return "cnv" if kind in ("deletion", "duplication") else "structural"
    parsed = parse_name(name)
    protein = (parsed.protein or "").strip()
    change = parsed.change or ""
    if change.startswith("n."):
        coding = _coding_consequence(change)
        return coding if coding in ("splice_site", "splice_region") else "noncoding_transcript"
    if protein:
        if _P_START.match(protein):
            return "start_loss"
        if _P_STOP_LOSS.match(protein):
            return "stop_loss"
        if _P_FRAMESHIFT.search(protein):
            return "frameshift"
        if _P_NONSENSE.match(protein):
            return "nonsense"
        if _P_SYNONYMOUS.match(protein):
            # Exonic but may still act on splicing; the name cannot say.
            return "synonymous"
        if _P_MISSENSE.match(protein):
            return "missense"
        if _P_INFRAME.search(protein):
            return "inframe_indel"
    if change.startswith("c."):
        coding = _coding_consequence(change)
        if coding:
            return coding
        # An exonic c. indel without a p. name: frameshift or in-frame cannot
        # be told apart from the name alone, so it stays "other".
    return "other"


@dataclass
class DLJoin:
    """VariationID -> the DL branch's protein variant id, read from ITS canonical table.

    The DL canonical table (``vpdl-dl canonical``) records every ClinVar
    VariationID behind each protein substitution (``clinvar_variation_ids``),
    after validating the residue against UniProt. Reading that mapping — rather
    than re-deriving UniProt coordinates here — means the two branches join on
    the identity the DL branch already checked, and the DL branch is not touched.
    """

    mapping: dict[int, str]
    source: str | None = None

    @classmethod
    def from_canonical(cls, path: Path | str) -> "DLJoin":
        import pandas as pd
        frame = pd.read_csv(path, usecols=["variant_id", "clinvar_variation_ids"],
                            dtype=str, low_memory=False)
        mapping: dict[int, str] = {}
        conflicts = 0
        for variant_id, ids in frame.dropna().itertuples(index=False):
            for value in str(ids).split(";"):
                value = value.strip()
                if not value:
                    continue
                key = int(float(value))
                if key in mapping and mapping[key] != variant_id:
                    conflicts += 1
                    continue
                mapping[key] = variant_id
        if conflicts:
            import logging
            logging.getLogger(__name__).warning(
                "%d VariationIDs map to more than one DL protein variant; first kept.",
                conflicts)
        return cls(mapping, str(path))

    @classmethod
    def empty(cls) -> "DLJoin":
        return cls({}, None)

    def get(self, variation_id: int) -> str | None:
        return self.mapping.get(int(variation_id))

    def __len__(self) -> int:
        return len(self.mapping)

    def as_dict(self) -> Mapping[int, str]:
        return self.mapping
