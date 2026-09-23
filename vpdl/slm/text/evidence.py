"""Evidence units: a document cut into sentences, each labelled by transparent rules.

Every sentence of a narrative becomes one unit with:

    sentence_role      description | evidence | conclusion | external_classification |
                       code_list | boilerplate | other
    evidence_types     population, functional, segregation, ... (below); a sentence
                       may carry several ("absent from gnomAD and predicted damaging")
    evidence_polarity  pathogenic | benign | neutral | mixed
    acmg_codes         codes the sentence states as applied, parsed (vpdl.slm.text.acmg)

**These labels are weak supervision, not ground truth.** They come from
keyword rules (provenance ``rule:evidence-lexicon/v1``) and their precision is
unknown until a clinician-annotated sample is scored against them — no such
sample exists in this repository (docs/slm/GENOMIC_SLM_DATASET.md, "Data gaps").
Until then they are used to (a) measure what the corpus contains, (b) mark
conclusion sentences for masking, and (c) give auxiliary training targets whose
benefit is an ablation (EXP-012), never an assumption.

A type is only assigned when a rule for it fires; a sentence that no rule
recognises gets no type, rather than a guessed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from vpdl.slm.text.acmg import CRITERIA, find_codes
from vpdl.slm.text.conclusion import classify_sentence
from vpdl.slm.text.sentences import split_sentences

__all__ = ["EVIDENCE_TYPES", "POLARITIES", "SENTENCE_ROLES", "EvidenceUnit",
           "extract_units", "RULE_PROVENANCE", "PMID"]

RULE_PROVENANCE = "rule:evidence-lexicon/v1"

EVIDENCE_TYPES = ("population", "functional", "segregation", "de_novo", "phenotype", "case",
                  "allelic", "computational", "conservation", "splicing", "domain", "mechanism",
                  "same_residue", "disease_association", "literature", "clinical_history",
                  "laboratory", "expert_curation")
POLARITIES = ("pathogenic", "benign", "neutral", "mixed")
SENTENCE_ROLES = ("description", "evidence", "conclusion", "external_classification",
                  "code_list", "boilerplate", "other")

PMID = re.compile(r"(?:PMID|PubMed(?:\s+ID)?)\s*[:#]?\s*(\d{5,9})", re.I)


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.I)


# Type cues. Kept deliberately literal so a reviewer can read what fires.
_TYPE_RULES: dict[str, re.Pattern] = {
    "population": _rx(r"\bgnomad\b", r"\bexac\b", r"1000\s*genomes", r"\besp\b", r"\btopmed\b",
                      r"population\s+(?:database|frequency|data|controls?)",
                      r"allele\s+(?:frequency|count)", r"minor\s+allele", r"\bmaf\b",
                      r"general\s+population", r"control\s+(?:population|individuals|chromosomes)",
                      r"absent\s+from\s+(?:\w+\s+){0,3}(?:databases?|controls|population)"),
    "functional": _rx(r"functional\s+(?:assay|stud|analys|evidence|data|test|characteri)",
                      r"\bin\s+vitro\b", r"\bin\s+vivo\b", r"\bassays?\b", r"yeast", r"complementation",
                      r"(?:mismatch\s+repair|mmr|enzym\w*|catalytic|kinase|transcriptional)\s+activity",
                      r"protein\s+(?:expression|stability|level|abundance|function)",
                      r"knock-?(?:in|out|down)", r"cell[- ]based", r"saturation\s+(?:genome\s+editing|mutagenesis)",
                      r"deep\s+mutational\s+scan", r"multiplex(?:ed)?\s+assay",
                      r"experimental\s+(?:studies|evidence|data)"),
    "segregation": _rx(r"segregat", r"co-?segregat", r"\blod\s+score", r"pedigree",
                       r"affected\s+(?:family\s+members|relatives|siblings)"),
    "de_novo": _rx(r"\bde\s+novo\b", r"confirmed\s+(?:paternity|maternity|parentage)"),
    "phenotype": _rx(r"phenotype", r"clinical\s+features", r"microsatellite\s+instab", r"\bmsi(?:-h)?\b",
                     r"immunohistochem", r"\bihc\b", r"loss\s+of\s+\w+\s+(?:protein\s+)?(?:expression|staining)",
                     r"tumou?r", r"\bcancer\b", r"carcinoma", r"polyp", r"consistent\s+with\s+(?:a\s+)?diagnosis"),
    "case": _rx(r"(?:observed|reported|identified|detected|found|seen)\s+in\s+(?:at\s+least\s+)?"
                r"(?:an?|one|two|three|four|five|several|multiple|\d+)\s+(?:\w+\s+){0,3}?"
                r"(?:individuals?|patients?|probands?|families|family|cases?|subjects?|carriers?)",
                r"case[- ]control", r"odds\s+ratio", r"case\s+report", r"affected\s+individuals?",
                r"alternate\s+molecular\s+basis"),
    "allelic": _rx(r"\bin\s+trans\b", r"\bin\s+cis\b", r"compound\s+heterozyg", r"biallelic",
                   r"homozyg", r"second\s+(?:pathogenic\s+)?(?:variant|allele)"),
    "computational": _rx(r"in\s+silico", r"computational", r"prediction\s+(?:tools?|algorithms?|programs?)",
                         r"predict(?:ed|s|ion)?\s+(?:to\s+be\s+)?(?:deleterious|damaging|tolerated|benign|"
                         r"disruptive|neutral)", r"\bsift\b", r"polyphen", r"\brevel\b", r"\bcadd\b",
                         r"alphamissense", r"mutationtaster", r"bayesdel", r"\bmetasvm\b", r"algorithms?"),
    "conservation": _rx(r"conserv", r"phylop", r"\bgerp\b", r"across\s+species", r"evolutionar"),
    "splicing": _rx(r"splic", r"spliceai", r"\bdonor\b", r"\bacceptor\b", r"exon\s+skipping",
                    r"cryptic", r"maxent", r"intron\s+retention", r"minigene", r"\brna\s+(?:analysis|studies|stud)",
                    r"aberrant\s+transcript"),
    "domain": _rx(r"\bdomain\b", r"hot\s*spot", r"functional\s+region", r"\bmotif\b",
                  r"critical\s+(?:region|residue)", r"active\s+site", r"binding\s+(?:site|interface)"),
    "mechanism": _rx(r"loss[- ]of[- ]function", r"null\s+(?:variant|allele)", r"nonsense[- ]mediated",
                     r"\bnmd\b", r"frameshift", r"premature\s+(?:translational\s+)?(?:stop|termination)",
                     r"truncat", r"haploinsufficien", r"dominant[- ]negative", r"gain[- ]of[- ]function",
                     r"disease\s+mechanism", r"start\s+codon", r"initiator\s+codon", r"in-frame"),
    "same_residue": _rx(r"(?:same|this)\s+(?:amino\s+acid\s+)?(?:residue|codon|position)",
                        r"(?:other|different)\s+(?:\w+\s+){0,2}?(?:changes?|variants?|substitutions?)\s+at\s+"
                        r"(?:this|the\s+same)"),
    "disease_association": _rx(r"(?:associated|associat\w+)\s+with\s+(?:\w+\s+){0,4}?(?:syndrome|disease|"
                               r"cancer|disorder)", r"causes?\s+(?:\w+\s+){0,3}?(?:syndrome|disease|disorder)",
                               r"autosomal\s+(?:dominant|recessive)", r"x-linked",
                               r"gene\s+(?:is|has\s+been)\s+(?:associated|implicated)"),
    "literature": _rx(r"\bpmid", r"et\s+al\b", r"\bliterature\b", r"publications?", r"published",
                      r"\breported\s+in\s+(?:the\s+)?(?:literature|studies)"),
    "clinical_history": _rx(r"family\s+history", r"personal\s+history", r"age\s+(?:of|at)\s+(?:onset|diagnosis)",
                            r"diagnosed\s+(?:at|with)"),
    "laboratory": _rx(r"(?:this|our)\s+(?:laboratory|lab)\b", r"internal\s+(?:data|database|observations?)",
                      r"in\s+our\s+(?:cohort|patients?|database)"),
    "expert_curation": _rx(r"expert\s+panel", r"\bvcep\b", r"\binsight\b", r"\bclingen\b", r"\benigma\b"),
}

# Direction cues. Order matters only for readability; all are checked.
_PATHOGENIC_CUES = _rx(
    r"absent\s+from", r"not\s+(?:been\s+)?(?:observed|found|reported|detected)\s+in\s+(?:\w+\s+){0,3}?"
    r"(?:gnomad|exac|population|controls|databases?)",
    r"(?:co-?)?segregat\w*\s+with\s+(?:disease|the\s+(?:disease|phenotype|condition))",
    r"\bde\s+novo\b", r"loss[- ]of[- ]function", r"deleterious", r"damaging", r"disrupt",
    r"abolish", r"(?:reduced|decreased|impaired|diminished|deficient|defective|abnormal|aberrant)\s+"
    r"(?:\w+\s+){0,2}?(?:activity|function|expression|splicing|repair|stability|binding|protein)",
    r"affects?\s+(?:\w+\s+)?(?:protein\s+)?function", r"null\s+(?:variant|allele)",
    r"premature\s+(?:translational\s+)?stop", r"nonsense[- ]mediated\s+decay", r"exon\s+skipping",
    r"loss\s+of\s+\w+\s+(?:protein\s+)?(?:expression|staining)", r"microsatellite\s+instab(?:le|ility)[- ]high|msi-h",
    r"in\s+trans\s+with\s+a\s+(?:known\s+)?(?:likely\s+)?pathogenic",
    r"(?:highly|strongly|well)\s+conserved", r"significantly\s+(?:more\s+)?(?:frequent|enriched)\s+in\s+(?:cases|patients|affected)")
_BENIGN_CUES = _rx(
    r"(?:high|elevated|higher\s+than\s+expected)\s+(?:allele\s+)?frequency", r"too\s+common",
    r"(?:observed|found|present)\s+in\s+(?:\w+\s+){0,3}?(?:homozyg\w+|healthy|unaffected)\s+(?:state|individuals|controls|adults)?",
    r"does\s+not\s+(?:co-?)?segregate", r"did\s+not\s+(?:co-?)?segregate", r"lack\s+of\s+segregation",
    r"no\s+(?:significant\s+)?(?:functional\s+)?(?:effect|impact|difference)",
    r"does\s+not\s+(?:affect|impact|disrupt|alter|impair)", r"did\s+not\s+(?:affect|impact|disrupt|alter|impair)",
    r"(?:normal|wild[- ]type[- ]like|proficient|intact)\s+(?:\w+\s+){0,2}?(?:function|activity|splicing|expression|repair)",
    r"\btolerated\b", r"predict\w*\s+(?:to\s+be\s+)?(?:benign|neutral|tolerated)",
    r"not\s+(?:predicted|expected)\s+to\s+(?:\w+\s+)?(?:affect|impact|disrupt)",
    r"in\s+cis\s+with\s+a\s+(?:known\s+)?pathogenic", r"co-?occur\w*\s+with\s+a\s+(?:known\s+)?pathogenic",
    r"alternate\s+molecular\s+basis", r"poorly\s+conserved|not\s+conserved|weakly\s+conserved",
    r"allele\s+frequency\s+of\s+\d+(?:\.\d+)?\s*%")

_DESCRIPTION = _rx(r"^(?:this|the)\s+(?:sequence\s+)?(?:change|variant|alteration)\s+(?:replaces|creates|"
                   r"deletes|duplicates|inserts|results\s+in|causes|is\s+located|occurs|affects\s+(?:codon|"
                   r"nucleotide|the\s+(?:canonical|coding))|is\s+a\s+(?:single|missense|nonsense|synonymous|"
                   r"frameshift))",
                   r"^the\s+\S+\s+variant\s+(?:\(.*?\)\s+)?(?:is|was)\s+located",
                   r"which\s+is\s+(?:basic|acidic|neutral|polar|non-polar|nonpolar|hydrophobic|aromatic)")
_BOILERPLATE = _rx(r"^clinvar\s+contains\s+an\s+entry", r"variation\s+id\s*:",
                   r"^for\s+these\s+reasons", r"^this\s+(?:report|result|interpretation)\s+(?:is|was|should)",
                   r"genetic\s+counsel", r"^(?:please|note:)", r"^this\s+variant\s+has\s+not\s+been\s+reported\s+"
                   r"in\s+the\s+literature\s+in\s+individuals\s+affected\s+with")


@dataclass(frozen=True)
class EvidenceUnit:
    index: int
    start: int
    end: int
    text: str
    sentence_role: str
    evidence_types: tuple[str, ...]
    evidence_polarity: str
    acmg_codes: tuple[str, ...]
    acmg_codes_not_met: tuple[str, ...]
    evidence_strength: str | None
    pmids: tuple[str, ...]
    confidence: float
    conclusion_category: str | None = None
    cues: dict[str, int] = field(default_factory=dict)


def _polarity(text: str, codes) -> tuple[str, int, int]:
    pathogenic = len(_PATHOGENIC_CUES.findall(text))
    benign = len(_BENIGN_CUES.findall(text))
    for mention in codes:
        if mention.met:
            if CRITERIA[mention.code].direction == "pathogenic":
                pathogenic += 1
            else:
                benign += 1
    if pathogenic and benign:
        return "mixed", pathogenic, benign
    if pathogenic:
        return "pathogenic", pathogenic, benign
    if benign:
        return "benign", pathogenic, benign
    return "neutral", 0, 0


def extract_units(text: str) -> list[EvidenceUnit]:
    """One :class:`EvidenceUnit` per sentence of `text`, offsets into `text`."""
    units = []
    all_codes = find_codes(text or "")
    for index, sentence in enumerate(split_sentences(text or "")):
        codes = [c for c in all_codes if sentence.start <= c.start < sentence.end]
        conclusion = classify_sentence(sentence.text)
        types = [name for name, rule in _TYPE_RULES.items() if rule.search(sentence.text)]
        text_types = list(types)          # what the words say, before the codes add theirs
        for mention in codes:
            coded_type = CRITERIA[mention.code].evidence_type
            if mention.met and coded_type not in types:
                types.append(coded_type)
        polarity, n_path, n_benign = _polarity(sentence.text, codes)

        if conclusion == "external_classification":
            role = "external_classification"
        elif conclusion is not None:
            role = "conclusion"
        elif codes and not text_types and len(re.findall(
                r"[A-Za-z]{3,}", _strip_codes(sentence.text, codes, sentence.start))) <= 10:
            # "The following criteria were applied: PM2_P, PP3." — only codes; with
            # the codes masked nothing is left to read.
            role = "code_list"
        elif _BOILERPLATE.search(sentence.text):
            role = "boilerplate"
        elif _DESCRIPTION.search(sentence.text) and not (n_path or n_benign):
            role = "description"
        elif types:
            role = "evidence"
        else:
            role = "other"

        strengths = [c.effective_strength for c in codes if c.met]
        cue_total = n_path + n_benign + len(types)
        units.append(EvidenceUnit(
            index=index, start=sentence.start, end=sentence.end, text=sentence.text,
            sentence_role=role,
            evidence_types=tuple(t for t in EVIDENCE_TYPES if t in types) if role not in (
                "conclusion", "external_classification", "boilerplate") else (),
            evidence_polarity=polarity if role in ("evidence", "code_list") else "neutral",
            acmg_codes=tuple(dict.fromkeys(c.normalized for c in codes if c.met)),
            acmg_codes_not_met=tuple(dict.fromkeys(c.normalized for c in codes if not c.met)),
            evidence_strength=strengths[0] if len(set(strengths)) == 1 else None,
            pmids=tuple(dict.fromkeys(PMID.findall(sentence.text))),
            confidence=round(min(1.0, cue_total / 4.0), 3),
            conclusion_category=conclusion,
            cues={"pathogenic": n_path, "benign": n_benign, "types": len(types)},
        ))
    return units


def _strip_codes(text: str, codes: Iterable, offset: int = 0) -> str:
    """`text` without the code mentions; `offset` is where `text` starts in the document."""
    pieces, cursor = [], 0
    for code in sorted(codes, key=lambda c: c.start):
        start, end = code.start - offset, code.end - offset
        pieces.append(text[cursor:max(cursor, start)])
        cursor = max(cursor, end)
    pieces.append(text[cursor:])
    return "".join(pieces)
