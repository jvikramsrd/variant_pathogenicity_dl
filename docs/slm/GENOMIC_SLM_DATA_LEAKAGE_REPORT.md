# Leakage report — what can leak, what is checked, what fails a build

Code: `vpdl/slm/build/leakage.py` (checks), `vpdl/slm/text/conclusion.py`
(masking), `vpdl/slm/text/acmg.py` (per-task policy).
Command: `vpdl-slm leakage --records … --examples … --scheme … --out …`
(exit code **3** on a critical finding).

**Status: the audit has run only on synthetic data** (`vpdl-slm smoke`:
0 critical findings). On real narratives it is **TBD — RUN ON DGX SPARK**; the
runbook makes it a gate before any training command.

## 1. The rule this branch is built around

Clinical text usually ends by stating its own verdict. A model shown that
sentence learns to read verdicts. But deleting the words "pathogenic",
"benign" and "VUS" would destroy real evidence — "in trans with a **pathogenic**
variant" (PM3), "**benign** in silico predictions" (BP4), "**pathogenic**
variants in MLH1 cause Lynch syndrome" (background).

So the unit is the **sentence**, and only sentences matching a conclusion
pattern are removed:

| Category | Example (invented) |
|---|---|
| `classification_statement` | "Therefore, this variant is classified as pathogenic." |
| `criteria_statement` | "This variant meets ACMG criteria for likely pathogenic." |
| `insufficient_evidence` | "In summary, the available evidence is currently insufficient to determine the role of this variant in disease." |
| `external_classification` | "This has been classified as pathogenic by other laboratories." |
| `heading` | "Classification: Pathogenic" |

Everything else is kept byte for byte, and each removal is recorded with its
character span and category. `residual_assertions()` then re-scans the masked
text with a deliberately broader pattern; the audit reports that rate as an
upper bound, so a new laboratory template that slips through shows up as a
number rather than as luck.

## 2. ACMG codes: masked or not, per task

Under the ACMG combining rules the met codes nearly determine the class, so a
"predict the class from the evidence" task with the codes left in would be
testing rule arithmetic:

| Task | conclusions | others' verdicts | ACMG codes |
|---|---|---|---|
| `classify` | masked | masked | **masked** |
| `classify_from_codes` (interpreting stated evidence) | masked | masked | **kept — it is the input** |
| `acmg_codes` (predict the criteria) | masked | masked | **masked — they are the target** |
| `evidence_type`, `evidence_polarity` | unit excluded if it is a conclusion | — | masked inside the unit |
| `pretraining` | masked if narratives are included at all | — | kept |

`classify` and `classify_from_codes` are different tasks and are never
compared with each other.

## 3. The fifteen checks

| # | Check | Critical when |
|---|---|---|
| 1 | `label_leakage` | an answer-describing column is an input (`review_status`, `stars`, `number_submitters`, `submitter`, any `NEVER_INPUT` column) |
| 2 | `conclusion_leakage` | any model input still contains a conclusion sentence (plus a reported residual-assertion rate) |
| 3 | `template_leakage` | a laboratory template family crosses train/evaluation under `text` or `laboratory` |
| 4 | `exact_duplicates` | identical input text crosses train/evaluation (same schemes) |
| 5 | `near_duplicates` | near-identical text crosses (same schemes); a warning with its rate elsewhere |
| 6 | `variant_duplicates` | a variant group is on both sides — critical under every scheme except `random`, where it is reported as expected |
| 7 | `literature_leakage` | (strict mode) an evaluation variant's cited publication is also cited for a training variant; otherwise a warning with the rate |
| 8 | `gene_shortcut` | never critical — reports the ROC-AUC obtainable from training per-gene label rates alone, the bar a model must beat to claim it reads evidence |
| 9 | `disease_shortcut` | the same for diseases |
| 10 | `functional_leakage` | an independent functional variant trains, an assay value is a feature, or a training input cites a holdout publication |
| 11 | `temporal_leakage` | a temporal training set contains post-cutoff or undated documents |
| 12 | `transcript_leakage` | one genomic change appears under two VariationIDs on opposite sides |
| 13 | `teacher_contamination` | teacher output exists for a non-training example |
| 14 | `benchmark_contamination` | an evaluation narrative is in the pretraining corpus (critical); a cited publication is (warning, critical in strict mode) |
| 15 | `retrieval_contamination` | a training-time retrieval index holds an evaluation document |

Checks that cannot run say so instead of passing quietly: without
`var_citations` the literature check reports "NOT checked", and without a
pretraining manifest the contamination check does the same.

## 4. Second-order leakage this branch takes seriously

* **Through the DL branch.** A `DLRepresentation` score is out-of-fold for its
  own gene, but the DL model saw other genes' labels. `DLInput.fold_check`
  reports, for every joined variant, whether its DL record's fold matches its
  gene; anything else is named `in_fold_or_other`, not accepted silently.
* **Through pretraining.** The corpus is built once and reused, so evaluation
  is reserved *before* it is built and removed from it (DATASET.md §5).
* **Through the teacher.** Prompts are generated only for training examples,
  contain the masked input (never the label), and every output records its
  model, version and prompt hash.
* **Through retrieval.** A training-time index refuses documents whose role is
  validation, test or independent — an exception at build time, not a filter
  at query time.
* **Through feature fitting.** Categorical vocabularies and numeric scalers are
  fitted on training rows only; unknown values map to `<unk>`.
* **Through calibration.** Calibrators refuse rows marked test.

## 5. Gene and disease shortcuts are measured, not assumed away

Per-gene pathogenic rates differ enormously across ClinVar. Under any split
that does not hold genes out, a model can score well by learning the gene. The
audit therefore fits the trivial "training label rate for this gene" predictor
and reports its test ROC-AUC. If a neural arm does not beat that number, it has
not shown it reads evidence — and the gene-identity feature is off by default
for exactly this reason.

## 6. What a clean run looks like

```
### Split scheme: `variant`
Critical findings: 0
| check | severity | count | finding |
| conclusion_leakage | info | 0 | no conclusion sentence in any model input |
| variant_duplicates | info | 0 | no variant group crosses train/evaluation |
| gene_shortcut | info | 0 | training label rate per gene alone gives test ROC-AUC 0.73 … |
```

`leakage_gate()` raises `LeakageError` on any critical finding, and the CLI
exits 3, so a pipeline cannot walk past it.
