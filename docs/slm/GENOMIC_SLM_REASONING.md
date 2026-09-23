# Clinical reasoning: evidence in, grounded explanation out

Code: `vpdl/slm/text/evidence.py`, `vpdl/slm/evaluation/explain.py`,
`vpdl/slm/retrieval.py`. Status: **implemented and unit-tested; no model has
been trained to do any of it.**

## 1. The chain

```
variant + its documents
   → sentences (with character offsets)
   → evidence units: role, types, polarity, ACMG codes
   → classification probabilities over P / LP / VUS / LB / B
   → uncertainty, and the option to abstain
   → a structured, cited explanation
```

The question the model is trained to answer is "what evidence supports or
opposes this interpretation?", not "what label belongs to this variant?" —
which is why the verdict sentences are removed from its input before it ever
sees them (GENOMIC_SLM_DATA_LEAKAGE_REPORT.md).

## 2. Evidence types

`population, functional, segregation, de_novo, phenotype, case, allelic,
computational, conservation, splicing, domain, mechanism, same_residue,
disease_association, literature, clinical_history, laboratory, expert_curation`,
with polarity `pathogenic | benign | neutral | mixed`.

Assigned by transparent keyword rules (`rule:evidence-lexicon/v1`) — weak
supervision whose own accuracy is **not measured**, because no annotated sample
exists (DATA GAP). A sentence no rule recognises gets **no** type rather than a
guessed one.

## 3. Conflicting and insufficient evidence

Disagreement is kept, not resolved: each submission keeps its own
classification; a sentence with cues in both directions is `mixed`; a variant
whose submitters disagree keeps ClinVar's `conflicting` category (and is never
given a five-class target). "No evidence of this type was found" is recorded
as `missing_evidence` and is explicitly **not** treated as evidence of absence.

## 4. ACMG / ClinGen

The 28 criteria are encoded with direction, default strength, the Richards 2015
category and an evidence type. Strength modifiers (`PM2_Supporting`,
`PVS1_Strong`…) are parsed, and negation ("not met") is read. Gene- and
disease-specific specifications switch codes off — the general ClinGen SVI
advice against PP5/BP6, and the InSiGHT MMR specification excluding PM1/PP2/BP1
— both recorded with their source and `verified=False` until checked against
the ClinGen CSpec registry. The model's ACMG head is masked to the codes a
gene's specification allows, so it cannot learn to predict a criterion an
expert panel does not use there. Codes are never invented: the grounding check
(below) flags any code in an explanation that no cited evidence states and the
model did not predict.

## 5. The explanation

A structured object, not free prose:

```
classification        one of the five classes, or "abstain"
probabilities         five calibrated numbers
key_evidence          the units used, verbatim, each with an id [E#] and its source
evidence_polarity     counts by direction
acmg_context          codes stated by the source, and codes the model predicted (marked)
uncertainty           entropy, abstention and why
explanation           sentences, each ending in the ids it rests on
missing_evidence      evidence types not found in the text
sources               document, submitter, date for every cited unit
disclaimer            RESEARCH USE ONLY …
```

The default explainer is deterministic: it states what the cited unit's words
say and attaches the model's own numbers in a sentence marked `[MODEL]`. It is
grounded by construction, and there is no free-running chain of thought. A
generative explainer is a later experiment (EXP-016) and would face the same
check.

## 6. The grounding check

`check_grounding(explanation, evidence)` scores **any** explanation — this
one's, a teacher's, a future generative head's. A sentence is unsupported when
it:

* cites nothing, or cites an id it was not given;
* states a number that no cited unit contains;
* names an ACMG code that no cited unit states and the model did not predict;
* claims an evidence type (segregation, functional, population, de novo,
  allelic, splicing) that no cited unit shows;
* states the opposite direction to every unit it cites.

Reported: unsupported-claim rate, invalid citations, number mismatches, code
hallucinations, invented evidence types, contradiction rate, evidence coverage.
The smoke run's explanation scores 0 unsupported claims and 1.0 coverage — on
synthetic data, which says the check and the explainer agree, not that any
model is good.

## 7. Retrieval

`EvidenceIndex` retrieves evidence units with full provenance (source,
document, variant, gene, disease, publication, timestamp, evidence type, text,
score, role). Two rules are enforced in code: an index built for training
**refuses** documents whose role is validation, test or independent, and
`as_of` drops anything later than a given date for temporal experiments. The
search is the knowledge base's BM25, imported unchanged, because genetics
queries are full of identifiers.

## 8. What the model must never do

Invent patient history, phenotypes, functional experiments, frequencies,
literature findings, ACMG codes, ClinGen rules, segregation or mechanisms. When
evidence is unavailable the output says so (`missing_evidence`), and when
confidence or evidence is too low it abstains with a reason. Every answer
carries the research-use disclaimer, and none of this is wired into
`vpdl kb-ask`, whose own rule is that the model never classifies a variant.
