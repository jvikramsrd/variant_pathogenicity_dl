# SLM clinical NLP audit — local code audit, 2026-09-24

Scope: `vpdl/slm/clinvar_text.py`, `vpdl/slm/text/**`, `vpdl/slm/erepo.py`,
`vpdl/slm/retrieval.py`, `vpdl/slm/build/{records,examples,pretrain_corpus}.py`,
`vpdl/slm/{pubmed,corpus,tokenizer}.py`, `vpdl/slm/modeling/data.py`.
**No clinical text was acquired.** ClinVar narratives (`submission_summary.txt.gz`) are
still not on any machine; everything here ran on synthetic sentences.

## The headline finding: verdicts without a verb reached the model input (CRITICAL, fixed)

Every conclusion pattern required a subject + verb ("this variant **is classified as**…").
Common lab-report endings have no verb, so both the masker and the audit's tripwire
missed them. On a synthetic narrative ending
`… absent from gnomAD. … Pathogenic (PVS1, PM2_Supporting, PP3)` the `classify` task's
model input ended with the word **"Pathogenic"** — the label it is asked to predict — and
`vpdl-slm leakage` would have reported 0 critical findings. The same sentence became an
evidence unit (`code_list`) whose masked text was the answer word.

**Fix** (`vpdl/slm/text/conclusion.py`): a new `label_statement` category and extended
patterns — a label after a separator (`:`, `->`, `=>`, `→`) at the end of a clause; a
label followed by an ACMG code list; heading words `assertion/result/call/acmg`; short
forms `LP/LB/P/B/VUS/VOUS/US` matched case-sensitively and only where a label is expected;
a trailing parenthetical such as `(VUS-high)`. A label after a separator is **kept** when
the words before it name evidence (`REVEL: benign`, `Functional assay: benign`), and other
variants' classifications stay kept. The tripwire (`residual_assertions`) now also counts
verb-less labels. Because masking, evidence-unit roles and the leakage audit all go
through `classify_sentence`, one fix covers all three.

Masker results after the fix (all asserted in `tests/slm/test_genomic_audit.py`):

| Input | Result |
|---|---|
| `Pathogenic (PVS1, PM2_Supporting, PP3)` | masked (heading) |
| `Likely benign: BS1, BP4.` | masked (label_statement) |
| `Assertion: Likely Pathogenic` | masked |
| `Result: Likely Pathogenic (PS3, PM2).` | masked |
| `Criteria met: PS3, PM2 -> Likely Pathogenic` | masked |
| `ACMG: LP`, `Final call: VUS`, `LP.` | masked |
| `This variant is classified as LP.` | masked (classification_statement) |
| `This variant is a variant of uncertain significance (VUS-high).` | masked |
| `REVEL: benign.`, `SIFT: tolerated; PolyPhen: benign`, `Functional assay: benign.` | kept |
| `Segregation: 3 affected carriers (PP1).`, `c.123A>G -> p.Arg41Gly` | kept |
| `Other pathogenic variants at this codon have been reported.` | kept |
| `It was observed in trans with a pathogenic variant (PMID: 12345).` | kept |
| `B cells from the patient showed reduced expression.` | kept |

The previously passing pattern tests (`tests/slm/test_genomic_text.py`) still pass.

## Other findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| D-6 | MEDIUM (deferred) | `vpdl/slm/clinvar_text.py` | ClinVar TSVs are split with `str.split("\t")`; an embedded tab in free text would shift columns (malformed rows are counted, shifted ones would not be). | DEFERRED — check against the real file on the DGX |
| D-7 | LOW (deferred) | `clinvar_text.py:86-91` | `newline=""` means a lone `\r` inside free text still breaks a line. | DEFERRED |
| D-8 | LOW | `modeling/backbone.py`, `modeling/data.py` | Truncation keeps the start (HF default); the end of a narrative — where verdicts sit — is what gets cut. Not a leak now that masking precedes tokenisation. | Documented |
| E-1 | LOW | `vpdl/slm/labels.py:46-49` | `normalize_classification(NaN)` → `out_of_scope` instead of `missing`; unreachable today (callers pass strings). | OPEN |

## Verified correct

- Sentence splitting protects `et al.`, `vs.`, `Fig.`, HGVS `p.`/`c.` dots and decimals, and
  offsets satisfy `text[start:end] == sentence.text`.
- Canonical verdicts ("is classified as", "we classify … as", "meets ACMG criteria for",
  "reclassified from X to Y", "insufficient evidence to determine") are masked; other
  laboratories' verdicts are their own category (ablation switch `keep_external`).
- The label is always the document's own classification; review status, stars, submitter
  count, submitter and the classification text are blocked from model inputs
  (`NEVER_INPUT`, per-task `forbidden_features`).
- The training-time retrieval index refuses validation/test documents.
- Narratives in the pretraining corpus are conclusion-masked and restricted to training
  documents.
- The PubMed reader drops retractions and carries no per-variant verdict.

## Risks for the data phase

The real false-negative rate of the masker is unmeasured: new lab templates will exist.
Before training, run `vpdl-slm leakage` on the real examples and read the tripwire rate
(`conclusion_leakage` warning) — a jump means a template is getting through. A sample of
real narratives should be hand-checked against the masker.
