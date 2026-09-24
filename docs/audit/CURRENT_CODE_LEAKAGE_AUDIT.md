# Current-code leakage audit — local code audit, 2026-09-24

Scope: both branches' leakage machinery as code — `vpdl/dl/{leakage,splits,canonical,homology,functional}.py`,
`vpdl/dl/pretrain/corpus.py`, `vpdl/experiment.py`; `vpdl/slm/build/{leakage,splits,roles,features,pretrain_corpus,clusters}.py`,
`vpdl/slm/text/{conclusion,dedup}.py`, `vpdl/slm/{retrieval,teacher,catalog}.py`.
No data was downloaded; risks that need real data are listed separately at the end.

## Matrix

| Class | Prevention in code | Detected by a gate? | After this audit |
|---|---|---|---|
| Label leakage (review status, stars, submitters) | `FORBIDDEN` / `NEVER_INPUT` | yes | unchanged, correct |
| Feature that IS the label (DL, landmine L2) | `check_feature_leakage` | only in the standalone full report | **now in the per-cell gate too** (M-1) |
| Conclusion leakage in text | conclusion masker + tripwire | **missed verb-less verdicts** | **fixed** (see SLM_CLINICAL_NLP_AUDIT.md) |
| Duplicate variant identities | variant groups (VariationID ∪ genomic change ∪ gene+protein change) | yes | correct |
| Train/test straddle | `check_split_isolation`, group-wise assignment | yes (planted leak caught) | correct |
| Near-duplicate / template text | cluster join + crossing check | **silently passed without clusters** | **critical for text/laboratory schemes** (M-4) |
| Paralog / homology (MLH1–PMS2, MSH2–MSH6) | `logo_purged`, `family`, `homology_straddle` | yes (warning under plain logo, by design) | correct |
| Literature (PMIDs of evaluation variants in pretraining) | exclusions + `benchmark_contamination` | yes when wired; **lenient by default** in the gate | pretrain-corpus now **requires** exclusions (M-5); strictness default OPEN (M-6) |
| Functional data as feature and label | DL: not emitted as features; SLM: `FUNCTIONAL_COLUMNS` | yes | correct |
| Teacher | train-only prompts, re-audited | yes | correct |
| Retrieval | role filter at build, audit check | yes (module not yet wired) | excludes by exact variant id, not variant group (M-8, OPEN) |
| DL features inside the SLM | none | **no** | fold report computed and recorded (I-4); `dl_score` feature unguarded (M-7, OPEN) |
| Benchmark contamination of pretrained PLMs / teacher | catalogue note | not enforceable | documented |
| Temporal | per-submission `DateLastEvaluated`; "novel" mode conservative | yes | documented |

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| D-1/D-2 | **CRITICAL** | `vpdl/slm/text/conclusion.py` | Verb-less verdicts ("Pathogenic (PVS1, PM2)", "Assertion: LP", "… -> Likely Pathogenic") passed both the masker and the tripwire, so the label reached the `classify` input and the leakage audit would report it clean. | **FIXED** |
| M-1 | HIGH | `vpdl/dl/leakage.py` `run_checks(full=False)` | The per-cell gate skipped the label-proxy check. A planted feature equal to the label passed the gate (caught only by `vpdl-dl leakage`). **The DGX's full leakage reports ran this check and found no feature ≥ 0.95 agreement, so existing DL results are not affected.** | **FIXED** — the check runs in both modes |
| M-3 | HIGH | `vpdl/slm/modeling/finetune.py` | The SLM's 15-check gate runs only as `vpdl-slm leakage`; nothing in `finetune`/`pretrain` enforced it. | **PARTLY FIXED** — `finetune` now runs `input_gate` (answer-describing features, conclusion sentences) and refuses on a critical finding. The split/duplicate/literature checks still need `vpdl-slm leakage` first (runbook order). |
| M-4 | HIGH | `vpdl/slm/build/leakage.py` `_check_crossing` | Without `vpdl-slm dedup` clusters, near-duplicate and template checks returned "warning, 0" — a planted near-duplicate across a text-isolating split passed. | **FIXED** — critical under `text` / `laboratory` (the schemes that promise isolation) |
| M-5 | HIGH | `vpdl/slm/cli.py` `pretrain-corpus` | Without `--exclusions` the corpus was built with evaluation narratives and cited PMIDs, exit 0. | **FIXED** — refuses unless `--no-exclusions` is passed explicitly |
| M-6 | MEDIUM | `vpdl/slm/cli.py` `leakage` | Literature overlap is a warning unless `--strict-literature`, while exclusions default to strict. | OPEN — decide the default before the data phase |
| M-7 | MEDIUM | `vpdl/slm/build/features.py` | `dl_score` has no fold-aware resolver or check. | OPEN (dormant: always NaN today) |
| M-8 | LOW | `vpdl/slm/retrieval.py` | Exclusion by literal variant id, not variant group. | OPEN (module unwired) |
| — | LOW | `vpdl/dl/leakage.py:39,43` | `FUNCTIONAL_VALUE_COLUMNS` is declared but the check hard-codes its own list. | OPEN |

## Risks that need the data phase

1. The masker's real false-negative rate on real ClinVar narratives (new templates).
2. Literature overlap between evaluation variants and PubMed pretraining text at real scale.
3. The multi-allelic gnomAD feature issue (DL_DATA_PIPELINE_AUDIT.md, B-1).
4. Near-duplicate rates across laboratories on real text.
5. What the teacher (qwen3:32b) already knows about evaluation variants.
6. The DL fold report once a real DL export is joined to SLM examples.
