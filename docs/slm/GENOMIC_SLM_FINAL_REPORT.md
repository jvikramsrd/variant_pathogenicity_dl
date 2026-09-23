# Final report — the LLM / SLM branch as a broad genomic model

Date: 2026-09-23. Machine: the Windows development PC (CPU only).
**No training on real data, no download, and no genomics result is reported
here.** Everything expensive is marked `NOT RUN — REQUIRES DGX SPARK`.

## Required statements

> The LLM/SLM branch was developed as a broad genomic model. MMR/Lynch
> Syndrome is treated as a downstream specialisation and evaluation domain.
> The DL branch was not redesigned.

> The SLM was designed to use a multi-source genomic knowledge ecosystem with
> clinical text as a core reasoning modality.

> Clinical text was audited separately for quantity, diversity, provenance,
> leakage, evidence coverage, and reasoning usefulness.

> Full training and large-scale experiments are only considered validated when
> actually run on the DGX Spark.

One qualification, stated plainly because the report would otherwise mislead:
the clinical-text audit is **designed, implemented and tested, and has been run
only on synthetic data**. ClinVar's narratives live in
`submission_summary.txt.gz`, which has never been downloaded in this project
(audit finding 1), so the corpus's size, diversity and evidence coverage are
`TBD — RUN ON DGX SPARK`. What has been measured is the variant layer of the
release that *is* here: 4,558,681 variants, 35,013 genes, 22,111 conditions —
of which 55% of variants name no specific condition at all.

## 1–3. Files inspected, modified, created

**Inspected** (read-only): the whole repository — `vpdl/` (kb, slm, dl, models,
sources, proteingym, cli, evaluate, provenance, device, splits), `src/` and
`scripts/` (v1 archive), all four test suites, `docs/` (kb, slm, dl, v2,
RUNLOG, DATASETS, HOLDOUT_PROTOCOL, MISSING_EVIDENCE), `PROJECT_PLAN.md`,
`README.md`, `pyproject.toml`, `.gitignore`, and the data trees under `data/`.

**Modified** (four files, additively):
`vpdl/slm/__init__.py` (docstring only), `pyproject.toml` (a `genomic-slm`
extra and the `vpdl-slm` entry point), `.gitignore` (new artefact paths),
`docs/slm/PLAN.md` and `docs/slm/RUNBOOK.md` (a pointer box each).
**Not modified:** `vpdl/dl/**`, `vpdl/kb/**`, `vpdl/cli.py`, `vpdl/sources/**`,
`tests/dl`, `tests/kb`, `docs/dl`, `docs/kb`, `src/`, `scripts/`.

**Created:** 47 new Python modules under `vpdl/slm/` (top level: schema,
labels, variants, clinvar_text, erepo, catalog, retrieval, teacher, baselines,
interface, experiments, hardware, config, cli, smoke, `__main__`; plus `text/`,
`build/`, `modeling/`, `evaluation/`), 4 test files
(`tests/slm/test_genomic_*.py`, **159 tests**), 22 configs
(`configs/slm/*.toml`) and 14 documents (`docs/slm/LLM_CODEBASE_AUDIT.md`,
`GENOMIC_SLM_*.md`, this report). About 10,700 lines of Python, roughly a third
of it tests.

## 4–6. Existing code reused, preserved, replaced

**Reused by import, unchanged:** `vpdl.dl.trainer` (the supervised loop),
`vpdl.dl.plm.peft` (LoRA/adapters for encoders), `vpdl.dl.calibration`
(binary calibration), `vpdl.dl.interface` (`ReasoningRepresentation` — the SLM
output subclasses it), `vpdl.dl.leakage` (report classes), `vpdl.dl.tracking`
(registry), `vpdl.dl.hardware`, `vpdl.slm.train` (checkpoint discipline,
windows, schedule), `vpdl.slm.corpus` (document iteration, hash split),
`vpdl.kb.search` (BM25), `vpdl.kb.ollama` (loopback client),
`vpdl.sources.clinvar._review_stars`, `vpdl.evaluate.evaluation_report`,
`vpdl.provenance`.

**Preserved:** every existing pipeline. The from-scratch SLM, the knowledge
base, the DL branch and the v1 archive all run exactly as before; the new
commands live behind `vpdl-slm`.

**Replaced:** nothing.

## 7–10. Checkpoints, datasets, sources

**Existing checkpoints:** none for this branch (`runs/slm/` is empty here; the
DGX holds a trial run). **Datasets on this machine:** ClinVar
`variant_summary` (measured), AlphaMissense, ProteinGym v1.3, MaveDB MSH2 LOF
(17,746 rows) and MLH1 abundance (5,056 rows), per-gene gnomAD, UniProt,
InterPro, AlphaFold, v1 MMR tables. **Newly incorporated:** none — no download
was made. Readers were written and tested for ClinVar `submission_summary` and
`var_citations` (column names checked against NCBI's README) and for the
ClinGen Evidence Repository (column names **unverified**, and the reader says
so rather than guessing).

## 11. Clinical text corpus statistics

`TBD — RUN ON DGX SPARK` (`vpdl-slm stats`). The measurement code produces
every count the plan asks for — documents, variants, genes, diseases,
submitters, expert-panel records, lengths, tokens, per-evidence-type coverage,
MMR share, and breakdowns by gene, disease, variant type, laboratory, year,
review status and tier. On synthetic data it runs end to end in the smoke test.

## 12–14. Knowledge base, data roles, leakage controls

A machine-readable catalogue of 28 sources, each with roles, licence/terms,
access method, status, scope, leakage risk and hash. Roles are enforced:
independent functional data cannot also train; a training-time retrieval index
refuses evaluation documents; reserved evaluation variants (and, in strict
mode, their cited publications) are removed from the pretraining corpus.
Leakage is fifteen checks with per-scheme severities and a gate that exits 3.
The conclusion masker removes only verdict sentences and keeps evidence that
happens to use the words "pathogenic" or "benign"; a broader tripwire reports
the residual rate.

## 15–16. Dataset construction and splits

Four tables plus ACMG labels, built by one streaming pass per file with a
manifest of input hashes and skip counts. Nine split schemes, all but `random`
keeping a variant group (VariationID ∪ genomic change ∪ gene+protein change)
wholly on one side; a split that cannot isolate anything raises instead of
pretending.

## 17–18. Architecture and the size decision

Backbone (biomedical encoder, our from-scratch decoder, or a tiny test model) →
pooled text + structured fields + optional DL embedding (masked) → fused
representation → class head (5), ACMG head (28, gene-masked), evidence heads
per sentence. Size is decided by `vpdl-slm sizing` from measured corpus tokens,
supervised example count and device memory, with the assumptions printed and
every figure labelled an estimate until `pretrain --benchmark` measures it.
No size is hard-coded.

## 19–23. Pretraining, reasoning, evidence, ACMG, VUS

Continued pretraining (MLM or CLM by backbone) on PubMed + knowledge-base
passages (+ optionally conclusion-masked training narratives), with evaluation
material excluded; step-based, resumable, interrupt-safe.
Evidence extraction is rule-based weak supervision with its provenance
recorded and its accuracy openly unmeasured. ACMG criteria are a table with
directions, strengths and gene specifications (both marked unverified); codes
are masked or kept per task policy. VUS is a class, ranked and scored against
later reclassification, with sub-tier agreement as a free check.

## 24–27. Calibration, uncertainty, grounding, teacher

Multi-class temperature and vector scaling fitted on validation only (test rows
are refused), reported pooled and per gene/disease. Uncertainty is entropy,
margin, mutual information and evidence sufficiency, scored by how well it
predicts error, with abstention that states its reason. Explanations are
structured and cited; the grounding checker flags uncited claims, invented
numbers, hallucinated codes, invented evidence types and contradictions.
The teacher is optional, prompts only training examples, never sees labels,
records model/version/prompt hash, and refuses a model whose terms for
training on outputs have not been checked.

## 28–30. Baselines, ablations, DL interface

Majority, per-gene rate, TF-IDF+LR, TF-IDF+SVM, structured-only, three
biomedical encoders, a ClinVar-BERT-style arm, then the SLM and its ablations —
all through the same metrics and calibration code. The DL interface reads the
DL branch's own export, takes dimensions from it, masks missing records and
reports fold compatibility. **Fusion is not implemented** and is out of scope.

## 31–32. DGX readiness and hardware detected

Readiness: a runbook of eleven phases where every command exists; `--dry-run`
on both expensive pipelines; `--benchmark` for throughput; resumable
checkpoints; a leakage gate before training; a registry line per run.

Detected here (`vpdl-slm hardware`): Windows x86-64 development PC, **no CUDA**,
CPU-only torch 2.14, transformers 5.17, tokenizers 0.23, pyarrow 25;
scikit-learn imports but its linear models are blocked by an Application
Control policy. `running_on_intended_target: false`. Intended target: NVIDIA
DGX Spark (GB10, aarch64, unified memory, bf16), previously measured at 90.1
TFLOPS peak and 30.7 TFLOPS sustained on the from-scratch small model.

## 33–35. Tests run, dry-runs run, experiments NOT run

**Tests:** `pytest tests/slm tests/kb tests/dl tests/regression` → **402
passed, 3 skipped** (tests/slm alone: 181 — 22 existing, 159 new). The three
skips are CUDA-only.
**Dry runs:** `vpdl-slm finetune --dry-run` (builds model, one forward and
backward, checkpoint round-trip identical, no training), `vpdl-slm pretrain
--dry-run` and `--benchmark 3`, and `vpdl-slm smoke` — the whole pipeline on
synthetic data, `ok: true` in 42 s. Every runbook phase was also driven through
the **CLI** on synthetic data: `hardware, catalog, experiments, sizing,
build-records, stats, dedup, splits, roles, examples ×2, leakage, baselines,
pretrain-corpus, pretrain-pack, pretrain --dry-run, pretrain --benchmark,
finetune --dry-run, finetune` all exit 0, a config naming missing data exits 2,
and an input with a verdict left in exits 3.
**Not run:** every experiment EXP-001 … EXP-025; any download; any real
narrative; any pretrained backbone; any timing or accuracy claim.

## 36. Exact DGX commands

See GENOMIC_SLM_DGX_RUNBOOK.md — phases 0–11, each command copy-pasteable.

## 37–39. Limitations, data gaps, remaining decisions

**Limitations.** Evidence-type and polarity labels are keyword rules of
unmeasured accuracy. Consequence classes are notation-level, not
transcript-aware. The ACMG gene specifications are taken from project notes and
marked unverified. Literature leakage is controlled at variant granularity, so
a paper describing a test variant that ClinVar does not cite for it can still
be in pretraining. The knowledge base's evaluation set is a development set and
cannot serve as a clean test.

**Data gaps.** ClinVar narratives and citations (not downloaded) · ClinGen
ERepo (not downloaded, columns unverified) · genome-wide gnomAD (so `log10_af`
is missing outside panel genes) · CIMRA (absent) · an annotated evidence sample
(so evidence extraction is unmeasurable) · an archived ClinVar release (so VUS
reclassification cannot be scored) · VEP annotations.

**Remaining decisions.** Which backbone (EXP-004…006 decides) · whether to
include narratives in pretraining · the temporal cutoff · how many seeds beyond
three · whether the from-scratch decoder or a biomedical encoder carries the
final model · whether a generative explainer is worth the grounding risk ·
whether MedGemma's terms permit training on its outputs (unchecked; qwen3:32b
is the default teacher because its licence is clear).

## 40. Exact next steps

1. `pip install -e ".[genomic-slm]" && pytest tests/slm -q` on the DGX (expect 180).
2. `vpdl-slm hardware --smoke` and `vpdl-slm smoke` — confirm the stack.
3. Download `submission_summary.txt.gz`, `var_citations.txt` and a matching
   `variant_summary.txt.gz` (runbook phase 2).
4. `vpdl-slm build-records --limit-documents 50000` first, then the full build.
5. `vpdl-slm stats` — **this is the answer to "is there enough clinical text?"**
   Everything after it depends on that number.
6. `dedup → splits → roles → examples → leakage`. Do not train until the audit
   is clean.
7. Baselines, then `pretrain --dry-run`, `--benchmark`, then the real run.

## Status of every major component

| Component | Status |
|---|---|
| Source catalogue, roles, licences | IMPLEMENTED · TESTED · verified against the providers' pages (2026-09-21/22) |
| ClinVar readers (variant_summary) | IMPLEMENTED · TESTED · **VERIFIED on the real 4.5M-variant file** |
| ClinVar readers (submission_summary, var_citations) | IMPLEMENTED · TESTED on synthetic files in ClinVar's layout · **NOT VERIFIED on real data** |
| ClinGen ERepo reader | IMPLEMENTED · column names **UNVERIFIED** (fails loudly) |
| Conclusion masking, ACMG parsing, evidence units | IMPLEMENTED · TESTED · rule accuracy **NOT MEASURED** (DATA GAP) |
| Dedup, splits, roles, examples | IMPLEMENTED · TESTED |
| Leakage audit (15 checks) + gate | IMPLEMENTED · TESTED · **run only on synthetic data** |
| Structured features, encoder | IMPLEMENTED · TESTED · `log10_af` is a DATA GAP |
| Model, heads, losses, PEFT | IMPLEMENTED · TESTED on tiny random models |
| Continued pretraining | IMPLEMENTED · TESTED (2 steps, tiny model) · **REQUIRES DGX SPARK** |
| Fine-tuning, prediction, registry | IMPLEMENTED · TESTED (1 epoch, tiny model) · **REQUIRES DGX SPARK** |
| Calibration, uncertainty, VUS | IMPLEMENTED · TESTED · VUS reclassification **needs an archived release** |
| Explanations + grounding check | IMPLEMENTED · TESTED · deterministic explainer only; generative explainer **PROPOSED** |
| Retrieval | IMPLEMENTED · TESTED · BM25 only; embeddings PROPOSED |
| Teacher distillation | IMPLEMENTED · TESTED (parsing, filtering, refusals) · **NOT RUN**, no teacher output exists |
| Baselines | IMPLEMENTED · TESTED · **NOT RUN on real data** |
| DL ↔ SLM interface | IMPLEMENTED · TESTED · DL export does not exist yet (DL branch untrained) |
| Experiment matrix EXP-001…025 | DEFINED · **NOT RUN — REQUIRES DGX SPARK** |
| Edge deployment | **PROPOSED · NOT TESTED** |
| Final multimodal DL+SLM fusion | **NOT IMPLEMENTED — out of scope for this phase, by instruction** |
