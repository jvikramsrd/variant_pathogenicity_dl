"""ACMG/AMP evidence criteria: what they are, how they are written, when they leak.

The criteria (Richards et al. 2015, Genet Med 17:405) are 28 codes, each with a
direction (pathogenic P*, benign B*) and a default strength that ClinGen's
Sequence Variant Interpretation group allows to be modified
(``PM2_Supporting``, ``PVS1_Strong``, ``PS3_Moderate``...).

Three uses here:

1. **Parsing** codes out of text (a lab narrative listing "PM2_P, PP3" or a
   ClinGen expert-panel summary). A bare "PS1" or "PM2" can mean other things
   ("PS1" is also a name for presenilin-1; "PM2.5" is air pollution), so a
   mention counts only with a strength suffix, a second code nearby, or
   criteria language in the sentence.
2. **Masking** codes out of model inputs — whether that is required depends on
   the task (:data:`TASK_POLICIES`): predicting PS3 from text that says "PS3"
   is not evidence extraction; interpreting a list of met codes is a
   legitimate, different task.
3. **Applicability**: gene/disease-specific specifications switch codes off
   (:class:`GeneSpec`). The model's ACMG head masks inapplicable codes rather
   than learning to predict codes an expert panel does not use for that gene.
   Specifications here are recorded WITH their source and ``verified=False``
   until checked against the ClinGen CSpec registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["CRITERIA", "Criterion", "CODES", "STRENGTHS", "CodeMention", "find_codes",
           "normalize_code", "mask_codes", "GeneSpec", "GENE_SPECS", "applicable_codes",
           "TaskPolicy", "TASK_POLICIES", "task_policy"]

STRENGTHS = ("Supporting", "Moderate", "Strong", "VeryStrong", "StandAlone")


@dataclass(frozen=True)
class Criterion:
    code: str
    direction: str             # pathogenic | benign
    default_strength: str      # one of STRENGTHS
    category: str              # Richards 2015 Table 5 column
    evidence_type: str         # vpdl.slm.text.evidence.EVIDENCE_TYPES
    summary: str


def _c(code, direction, strength, category, evidence_type, summary):
    return Criterion(code, direction, strength, category, evidence_type, summary)


# Categories follow the columns of Richards et al. 2015 Table 5; summaries are
# paraphrases for readers of this code, not the guideline's wording.
CRITERIA: dict[str, Criterion] = {c.code: c for c in [
    _c("PVS1", "pathogenic", "VeryStrong", "computational_predictive", "mechanism",
       "null variant in a gene where loss of function causes disease"),
    _c("PS1", "pathogenic", "Strong", "computational_predictive", "same_residue",
       "same amino-acid change as an established pathogenic variant"),
    _c("PS2", "pathogenic", "Strong", "de_novo", "de_novo", "de novo, parentage confirmed"),
    _c("PS3", "pathogenic", "Strong", "functional", "functional",
       "well-established functional study shows a damaging effect"),
    _c("PS4", "pathogenic", "Strong", "population", "case",
       "prevalence in affected individuals significantly above controls"),
    _c("PM1", "pathogenic", "Moderate", "functional", "domain",
       "mutational hot spot or critical functional domain"),
    _c("PM2", "pathogenic", "Moderate", "population", "population",
       "absent (or extremely rare) in population databases"),
    _c("PM3", "pathogenic", "Moderate", "allelic", "allelic",
       "recessive disorder: in trans with a pathogenic variant"),
    _c("PM4", "pathogenic", "Moderate", "computational_predictive", "mechanism",
       "protein length change (in-frame indel, stop-loss)"),
    _c("PM5", "pathogenic", "Moderate", "computational_predictive", "same_residue",
       "different missense change at a residue with an established pathogenic missense"),
    _c("PM6", "pathogenic", "Moderate", "de_novo", "de_novo", "assumed de novo"),
    _c("PP1", "pathogenic", "Supporting", "segregation", "segregation",
       "co-segregation with disease in affected family members"),
    _c("PP2", "pathogenic", "Supporting", "functional", "mechanism",
       "missense in a gene with few benign missense variants"),
    _c("PP3", "pathogenic", "Supporting", "computational_predictive", "computational",
       "computational evidence of a deleterious effect"),
    _c("PP4", "pathogenic", "Supporting", "other", "phenotype",
       "phenotype or family history highly specific for the gene's disease"),
    _c("PP5", "pathogenic", "Supporting", "other_database", "laboratory",
       "reputable source reports pathogenic (ClinGen SVI: do not use)"),
    _c("BA1", "benign", "StandAlone", "population", "population",
       "allele frequency above the stand-alone threshold"),
    _c("BS1", "benign", "Strong", "population", "population",
       "allele frequency greater than expected for the disorder"),
    _c("BS2", "benign", "Strong", "population", "population",
       "observed in healthy adults where full penetrance is expected early"),
    _c("BS3", "benign", "Strong", "functional", "functional",
       "well-established functional study shows no damaging effect"),
    _c("BS4", "benign", "Strong", "segregation", "segregation",
       "lack of segregation in affected family members"),
    _c("BP1", "benign", "Supporting", "functional", "mechanism",
       "missense in a gene where only truncating variants cause disease"),
    _c("BP2", "benign", "Supporting", "allelic", "allelic",
       "in trans with a pathogenic variant (dominant) or in cis (any)"),
    _c("BP3", "benign", "Supporting", "computational_predictive", "mechanism",
       "in-frame indel in a repetitive region without known function"),
    _c("BP4", "benign", "Supporting", "computational_predictive", "computational",
       "computational evidence of no impact"),
    _c("BP5", "benign", "Supporting", "other", "case",
       "found in a case with an alternate molecular basis for disease"),
    _c("BP6", "benign", "Supporting", "other_database", "laboratory",
       "reputable source reports benign (ClinGen SVI: do not use)"),
    _c("BP7", "benign", "Supporting", "computational_predictive", "splicing",
       "synonymous with no predicted splice impact"),
]}
CODES = tuple(CRITERIA)

_SUFFIX_LONG = {
    "supporting": "Supporting", "moderate": "Moderate", "strong": "Strong",
    "verystrong": "VeryStrong", "very strong": "VeryStrong", "very-strong": "VeryStrong",
    "very_strong": "VeryStrong", "standalone": "StandAlone", "stand-alone": "StandAlone",
    "stand alone": "StandAlone", "stand_alone": "StandAlone",
}
_SUFFIX_SHORT = {"p": "Supporting", "sup": "Supporting", "supp": "Supporting",
                 "m": "Moderate", "mod": "Moderate", "s": "Strong", "str": "Strong",
                 "vs": "VeryStrong", "vstr": "VeryStrong", "sa": "StandAlone"}

_CODE = r"(?P<code>PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])"
# Codes are matched case-sensitively ("bp1" is not BP1); strength words in any case.
_MENTION = re.compile(
    r"(?<![A-Za-z0-9])" + _CODE +
    r"(?:"
    r"[_\-](?P<short>(?i:very[ _-]?strong|stand[ _-]?alone|supporting|moderate|strong|"
    r"vstr|supp|sup|mod|str|vs|sa|p|m|s))"
    r"|[ ]?\(\s*(?P<paren>(?i:very[ _-]?strong|stand[ _-]?alone|supporting|moderate|strong))\s*\)"
    r"|[ ](?P<spaced>(?i:very strong|stand-alone|supporting|moderate|strong))"
    r")?(?![A-Za-z0-9])")
_CRITERIA_CONTEXT = re.compile(
    r"acmg|amp\b|criteri|evidence code|applied|clingen|guideline|met\b|codes?\b", re.I)
_NEGATION = re.compile(r"not\s+met|not\s+applied|not\s+applicable|were\s+not\s+used|"
                       r"was\s+not\s+used|did\s+not\s+meet|does\s+not\s+meet|unmet", re.I)
_CLAUSE_END = re.compile(r"[.;|]|\bmet\s*:", re.I)


@dataclass(frozen=True)
class CodeMention:
    code: str                  # PM2
    strength: str | None       # Supporting (explicit suffix) or None
    normalized: str            # PM2_Supporting, or PM2 when no suffix
    start: int
    end: int
    met: bool = True

    @property
    def criterion(self) -> Criterion:
        return CRITERIA[self.code]

    @property
    def effective_strength(self) -> str:
        return self.strength or self.criterion.default_strength


def normalize_code(code: str, suffix: str | None) -> tuple[str, str | None]:
    code = code.upper()
    if not suffix:
        return code, None
    key = suffix.strip().lower()
    strength = _SUFFIX_LONG.get(key) or _SUFFIX_SHORT.get(key) or _SUFFIX_LONG.get(
        key.replace(" ", "").replace("-", "").replace("_", ""))
    if strength is None:
        return code, None
    if strength == CRITERIA[code].default_strength:
        return code, strength
    return f"{code}_{strength}", strength


def find_codes(text: str, require_context: bool = True) -> list[CodeMention]:
    """Code mentions in `text`, with explicit strength and met / not-met status."""
    raw = []
    for match in _MENTION.finditer(text or ""):
        code = match.group("code").upper()
        if code not in CRITERIA:
            continue
        suffix = match.group("short") or match.group("paren") or match.group("spaced")
        normalized, strength = normalize_code(code, suffix)
        raw.append((match, code, suffix, normalized, strength))
    if not raw:
        return []
    if require_context:
        distinct = {code for _, code, *_ in raw}
        contextual = bool(_CRITERIA_CONTEXT.search(text))
        raw = [r for r in raw if r[2] or len(distinct) >= 2 or contextual]

    negations = [m.end() for m in _NEGATION.finditer(text)]
    mentions = []
    for match, code, _suffix, normalized, strength in raw:
        met = True
        for start in negations:
            if start <= match.start():
                end = _CLAUSE_END.search(text, start)
                if end is None or end.start() >= match.start():
                    met = False
        mentions.append(CodeMention(code, strength, normalized, match.start(), match.end(), met))
    return mentions


_LEFTOVER = re.compile(r"(?:\s*[,/;&]\s*(?:and\s+)?)+(?=[\s,/;&)\].]|$)")


def mask_codes(text: str, mentions: Iterable[CodeMention] | None = None,
               replacement: str = "") -> tuple[str, int]:
    """Remove code mentions; returns (text, how many). List punctuation is tidied."""
    mentions = sorted(mentions if mentions is not None else find_codes(text, False),
                      key=lambda m: m.start)
    if not mentions:
        return text, 0
    pieces, cursor = [], 0
    for mention in mentions:
        pieces.append(text[cursor:mention.start])
        pieces.append(replacement)
        cursor = mention.end
    pieces.append(text[cursor:])
    masked = "".join(pieces)
    masked = re.sub(r"\(\s*[,;/\s]*\)", "", masked)          # "( , )" left by a list
    masked = re.sub(r"(?<=[:(])\s*[,;/]\s*", " ", masked)
    masked = re.sub(r"\s*[,;/]\s*(?=[,;/.)])", "", masked)
    masked = re.sub(r"[ \t]{2,}", " ", masked)
    return masked.strip(), len(mentions)


# -- gene / disease specifications ------------------------------------------

@dataclass(frozen=True)
class GeneSpec:
    name: str
    genes: tuple[str, ...]
    not_applicable: frozenset[str]
    source: str
    verified: bool = False
    notes: str = ""


GENE_SPECS: tuple[GeneSpec, ...] = (
    GeneSpec(
        name="ClinGen SVI general",
        genes=("*",),
        not_applicable=frozenset({"PP5", "BP6"}),
        source="ClinGen SVI recommendation (Biesecker & Harrison, Genet Med 2018) that "
               "PP5/BP6 no longer be used",
        verified=False,
        notes="Codes still appear in older submissions; parsed, never predicted."),
    GeneSpec(
        name="InSiGHT MMR (Lynch syndrome)",
        genes=("MLH1", "MSH2", "MSH6", "PMS2"),
        not_applicable=frozenset({"PM1", "PP2", "BP1"}),
        source="PROJECT_PLAN.md Phase 4 note citing Plazzer et al., medRxiv "
               "2024.05.13.24307108 (InSiGHT ACMG/AMP specification)",
        verified=False,
        notes="Confirm against the ClinGen CSpec registry entry for the InSiGHT VCEP "
              "before any gene-specific result is reported."),
)


def applicable_codes(gene: str | None, specs: Iterable[GeneSpec] = GENE_SPECS) -> tuple[str, ...]:
    """Codes a model may predict for `gene`, in :data:`CODES` order."""
    off: set[str] = set()
    for spec in specs:
        if "*" in spec.genes or (gene and gene in spec.genes):
            off |= spec.not_applicable
    return tuple(code for code in CODES if code not in off)


# -- per-task leakage policy (docs/slm/GENOMIC_SLM_DATA_LEAKAGE_REPORT.md) -----

@dataclass(frozen=True)
class TaskPolicy:
    task: str
    target: str
    mask_conclusions: bool = True
    mask_external_classifications: bool = True
    mask_acmg_codes: bool = True
    forbidden_features: frozenset[str] = field(default_factory=lambda: frozenset(
        {"review_status", "stars", "number_submitters", "submitter"}))
    notes: str = ""


TASK_POLICIES: dict[str, TaskPolicy] = {p.task: p for p in [
    TaskPolicy("classify", "five-class classification of the document's variant",
               notes="Codes are masked: under the ACMG combining rules the met codes "
                     "nearly determine the class, so leaving them in would test rule "
                     "arithmetic, not evidence reading."),
    TaskPolicy("classify_from_codes", "five-class classification given the met codes",
               mask_acmg_codes=False,
               notes="An interpretation task (apply combining rules to stated evidence). "
                     "Reported separately; never compared with 'classify'."),
    TaskPolicy("acmg_codes", "which ACMG codes the submitter applied",
               notes="Target codes must not be in the input."),
    TaskPolicy("evidence_type", "evidence types of one sentence",
               notes="Codes inside the sentence are masked; conclusion sentences are "
                     "not units of this task."),
    TaskPolicy("evidence_polarity", "direction of one sentence's evidence"),
    TaskPolicy("vus_priority", "rank VUS by pathogenic-leaning probability"),
    TaskPolicy("explanation", "evidence-cited explanation"),
    TaskPolicy("pretraining", "masked / causal language modelling",
               mask_acmg_codes=False,
               notes="ClinVar narratives are excluded from pretraining by default; if "
                     "included, only for TRAINING-role variants and conclusion-masked."),
]}


def task_policy(task: str) -> TaskPolicy:
    if task not in TASK_POLICIES:
        raise KeyError(f"no leakage policy for task {task!r}; define one in "
                       "vpdl.slm.text.acmg.TASK_POLICIES before building its examples")
    return TASK_POLICIES[task]
