# Codebase audit — LLM / SLM branch

Written 2026-09-22/23, before any code in this branch was changed. Read-only
inspection, the existing test suites, and one measured pass over the ClinVar
file already on this machine. **No data was downloaded and nothing was
trained.** Numbers below are measured or absent; where a number is not
measured it says so.

## 1. Scope map

| Area | Paths | Status for this work |
|---|---|---|
| **LLM/SLM — in scope** | `vpdl/slm/` (new modules), `tests/slm/test_genomic_*.py`, `docs/slm/GENOMIC_SLM_*.md`, `configs/slm/` | Created here. |
| **LLM/SLM — existing, preserved** | `vpdl/slm/{pubmed,corpus,tokenizer,pack,model,train}.py`, `tests/slm/test_slm.py`, `docs/slm/{PLAN,RUNBOOK}.md` | **Not modified** except `vpdl/slm/__init__.py`'s docstring. The from-scratch pretraining pipeline still runs exactly as the runbook says. Its 22 tests still pass. |
| **Knowledge base — preserved** | `vpdl/kb/`, `tests/kb/`, `docs/kb/` | **Not modified.** Imported read-only (`kb.search.BM25`, `kb.ollama.LocalOllama`). Its 47 tests still pass. |
| **DL branch — DO NOT TOUCH** | `vpdl/dl/`, `tests/dl/`, `docs/dl/`, `configs/dl/` | **Not modified, not retrained, not re-evaluated.** Imported read-only (trainer, calibration, PEFT, output contract, leakage report classes, tracking, hardware). Its 107 tests still pass. |
| **v2 shared layer** | `vpdl/sources/`, `vpdl/evaluate.py`, `vpdl/provenance.py`, `vpdl/device.py`, `vpdl/splits.py` | Imported, not modified (`sources.clinvar._review_stars`, `evaluate.evaluation_report`, `provenance._git_state`). |
| **Shared entry points** | `vpdl/cli.py`, `pyproject.toml`, `.gitignore` | `vpdl/cli.py` **not modified** — the new commands live in `vpdl/slm/cli.py` behind `vpdl-slm`, exactly as the DL branch did with `vpdl-dl`. `pyproject.toml` / `.gitignore`: additive lines only. |
| **v1 archive** | `src/`, `scripts/`, `tests/test_*.py`, `run_*.py`, `main.py`, `PROJECT_PLAN.md` | Frozen. Read for the earlier LLM plan (below). |

## 2. What exists

### 2.1 Code

* **From-scratch SLM** (`vpdl/slm`, 2026-09-22): PubMed reader (drops retractions
  and expressions of concern), corpus builder (hash-based 1% validation split),
  32k byte-level BPE, uint16 token packing, Llama-style random-weight model
  (small ~110M / medium ~340M), and a resumable step-based training loop with
  rotating + timed checkpoints, save-on-SIGINT, milestones and a NaN guard.
  Benchmarked on the DGX: 34.7k tokens/s with `torch.compile`, 30.7 TFLOPS.
* **Knowledge base** (`vpdl/kb`): GeneReviews → passages, BM25 + embedding
  search, loopback-only Ollama client with a GPU-placement check, ClinVar
  variant lookup (never through the model), citation check that withholds a
  whole answer, and an evaluation harness. Default reader `medgemma:27b`.
* **No genomic reasoning code existed**: no ClinVar narrative reader, no
  conclusion masking, no ACMG parsing, no evidence extraction, no
  classification head, no calibration or VUS handling in this branch.

### 2.2 The earlier plan, never implemented

`PROJECT_PLAN.md` Phase 4 describes a ClinVar-BERT-style branch: ClinVar
free-text summaries + PubMed case reports, MinHash/Jaccard(0.95) dedup grouped
by lab × gene, a sentence classifier that strips conclusion sentences,
class balancing, BioBERT-base, and a MedGemma-27B teacher for ACMG-grounded
rationales. **None of it exists in code.** Its ideas are carried into this
work (conclusion masking, template dedup, ACMG grounding, optional teacher)
with the pipeline written to this project's standards.

### 2.3 Data actually on this machine (measured 2026-09-22, `vpdl-slm inventory`)

| What | Measured |
|---|---|
| `data/raw/variant_summary.txt.gz` | 442,477,016 bytes, sha256 `7e5f0c79…`; **4,558,681 variants**, **35,013 gene symbols**, 22,111 distinct conditions |
| Classifications | five-class 4,027,357 · conflicting 166,121 · paired terms 107,712 · out of scope 4,725 · missing 252,766 |
| Classes | VUS 2,372,680 · likely benign 1,098,869 · pathogenic 215,659 · benign 214,051 · likely pathogenic 126,098 |
| Consequence (from HGVS) | missense 2,510,403 (55%) · synonymous 762,471 · intronic 499,277 · splice region 164,900 · frameshift 150,167 · nonsense 96,941 · splice site 96,258 · CNV 75,793 · 3′UTR 70,453 · in-frame indel 43,147 · 5′UTR 39,607 · other 38,172 · non-coding 4,375 · start-loss 3,637 · structural 2,864 · stop-loss 216 |
| Review status | 1 star 3,467,376 · 2 stars 669,231 · 0 stars 399,621 · **3 stars (expert panel) 22,390** · 4 stars (practice guideline) 63 |
| Dates | 2025 1,368,249 · 2024 840,384 · 2023 602,568 · 2022 459,559 · undated 287,323 · 2026 271,708 · older ~700k |
| MMR genes (MLH1/MSH2/MSH6/PMS2) | **31,725 variants (0.70% of ClinVar)**; P 6,098 · LP 1,065 · VUS 11,846 · LB 4,582 · B 797 |
| Conditions | **2,520,821 variants (55%) carry no specific condition** — only "not provided", "not specified" or a laboratory umbrella term |
| MaveDB | MSH2 LOF (HAP1) `urn:mavedb:00000050-a-1`, 17,746 rows, CC0, PMID 33357406; MLH1 abundance `urn:mavedb:00001218-a-1`, 5,056 rows, CC0, DOI 10.1101/2024.07.28.605491 |
| Other local | AlphaMissense (1.2 GB), ProteinGym v1.3, gnomAD per-gene files (panel genes only), UniProt, InterPro, AlphaFold, v1 MMR tables |
| **Not on this machine** | `submission_summary.txt.gz` (**the narratives**), `var_citations.txt`, ClinGen ERepo, PubMed baseline (downloading on the DGX), `data/kb` (built on the DGX), `data/slm` corpus, CIMRA, genome-wide gnomAD |

### 2.4 Checkpoints

No LLM/SLM checkpoint exists in this repository. `runs/slm/` is empty here (the
DGX holds the trial run). The two `.pt` files under `data/processed/transfer/`
are v1 DL artefacts.

### 2.5 Tests before this work

`tests/slm` 22 passed, `tests/kb` 47 passed, `tests/dl` 107 passed (176 total,
this PC, CPU). `tests/regression` was not re-run here; the DL audit recorded
62 passed / 1 skipped / 4 failed, the four being this PC's Application Control
policy blocking scikit-learn's `_loss` DLL.

## 3. Assessment

### Works, and is kept
* The from-scratch pretraining pipeline and its checkpoint discipline — reused
  directly for continued pretraining (`TokenWindows`, the learning-rate
  schedule, checkpoint listing/rotation, the Python-headers check).
* The knowledge base's safety design (retrieve → cite → refuse, loopback-only,
  variant facts looked up and never generated). Untouched.
* The DL branch's contracts: the frozen output schema with its named
  `ReasoningRepresentation`, the resumable trainer, calibration, LoRA, the
  leakage report format, the run registry.

### Findings of this audit

1. **The LLM branch's primary text does not exist here, and never did.**
   ClinVar's narratives live in `submission_summary.txt.gz` (`Description` =
   "an optional free text description of the basis of the interpretation",
   NCBI README, checked 2026-09-22) — not in `variant_summary`, which is what
   this project has always downloaded. Every earlier plan that said "ClinVar
   free-text summaries" was planning against a file nobody had fetched. The
   reader is written and tested against synthetic files in ClinVar's layout;
   the corpus itself is **TBD — RUN ON DGX SPARK**.
2. **ClinVar now carries VUS sub-tiers.** The local release holds `VUS-high`
   (48) and `VUS-mid` (42) at variant level — a native ranking signal for the
   VUS task that no code in this project handled. It is now parsed as VUS with
   the tier kept as a modifier, and used as a check on VUS ranking.
3. **Most variants have no specific condition.** 2,520,821 of 4,558,681 (55%) name only catch-all or umbrella terms, and 3,439,736 condition mentions are of that kind. Three laboratory umbrella terms dominate:
   "Inborn genetic diseases" (362,633 variants), "Hereditary
   cancer-predisposing syndrome" (202,567), "Cardiovascular phenotype"
   (93,976), on top of "not provided" (1,431,065) and "not specified"
   (1,308,925). A naive disease-holdout split would put most of a laboratory's
   submissions in one "disease"; these names are now excluded from disease
   grouping.
4. **Documentation contradiction.** `docs/kb/SOURCES.md` says "The small model
   is **not trained on these facts**", while `vpdl/slm/corpus.py` (and
   `docs/slm/PLAN.md`) put GeneReviews passages into the pretraining corpus.
   Both are defensible — pretraining teaches language, the knowledge base
   answers from documents — but the two documents disagree. Recorded here and
   in GENOMIC_SLM_PRETRAINING.md; neither file was changed.
5. **The knowledge base's rule 4 and a classifier.** `docs/kb/DESIGN.md` rule 4:
   "The model never classifies a variant." The genomic SLM does classify. The
   two do not meet: the classifier is a research model evaluated offline, and
   is **not** wired into `vpdl kb-ask`. If that ever changes it is a decision
   with clinical consequences, not a refactor.
6. **The knowledge base's evaluation set is a development set** (RUNLOG
   2026-09-22: the prompt and citation parser were changed after reading its
   answers). It cannot serve as a clean test for anything, including a
   comparison against this branch.
7. **The v1 MMR tables come from a different ClinVar release** (manifest sha256
   `186ad283…` vs the local file's `7e5f0c79…`). They are preserved as v1
   inputs, not reused as labels here.
8. **Population frequency is missing for broad variants.** gnomAD is present
   only for panel genes, so `log10_af` is NaN for nearly all of ClinVar until a
   genome-wide join runs on the DGX. It is carried as a feature with a
   presence mask, never imputed. DATA GAP.
9. **No CIMRA file exists here**, so one of the two independent functional
   validation sets is unavailable. DATA GAP.
10. **scikit-learn's linear models cannot be imported on this PC** (Application
    Control blocks `sklearn._loss`), so the TF-IDF baselines fit their convex
    objectives with SciPy L-BFGS here; `--backend sklearn` runs the reference
    implementation on the DGX.

## 4. Reuse plan (what this branch does with what exists)

| Existing component | Decision |
|---|---|
| `vpdl/slm/train.py` helpers (`TokenWindows`, `learning_rate`, checkpoint listing, `same_run`, `missing_python_headers`) | **Imported unchanged** by `vpdl/slm/modeling/continued.py`. |
| `vpdl/slm/corpus.py` (`iter_documents`, `iter_texts`, `in_validation`) | **Imported unchanged** by the pretraining-corpus builder. |
| `vpdl/slm/{pubmed,tokenizer,pack,model}.py` | Untouched; the from-scratch arm stays exactly as benchmarked. |
| `vpdl/dl/trainer.py` | **Reused as the supervised training loop** (resume, precision, early stopping). |
| `vpdl/dl/plm/peft.py` | **Reused** for encoders; a decoder equivalent added beside it. |
| `vpdl/dl/calibration.py` | **Reused** for the binary view; multi-class scaling added. |
| `vpdl/dl/interface.py` | **Reused**: `SLMRepresentation` subclasses its `ReasoningRepresentation`. |
| `vpdl/dl/leakage.py` report classes | **Reused** so both branches' reports read alike. |
| `vpdl/dl/tracking.py`, `vpdl/dl/hardware.py`, `vpdl/device.py`, `vpdl/provenance.py` | **Reused** for the registry, hardware report and git/hash provenance. |
| `vpdl/kb/search.py` (BM25, RRF), `vpdl/kb/ollama.py` | **Imported** by the evidence retriever and the teacher client. |
| `vpdl/sources/clinvar.py` `_review_stars`, `vpdl/evaluate.py` | **Imported** for star ratings and the standard binary panel. |
| `PROJECT_PLAN.md` Phase 4 recipe | **Implemented as designed** (conclusion stripping, lab-template dedup, ACMG grounding, optional teacher) rather than as written (no BioBERT-only commitment; the backbone is an experiment). |

## 5. After the work — validation actually executed on this PC

| Check | Result |
|---|---|
| `pytest tests/slm` (22 existing + 166 new) | **188 passed** |
| `pytest tests/kb` | 47 passed (unchanged) |
| `pytest tests/dl` | 107 passed (unchanged) |
| `vpdl-slm smoke` — records → stats → clusters → split → roles → examples → leakage → pretraining corpus → packing → continued pretraining (2 steps) → fine-tuning (1 epoch) → baselines → export → explanation | **ok: true, 42 s** on CPU with synthetic data and random weights |
| `vpdl-slm inventory` over the real ClinVar file | 4,558,681 variants measured (section 2.3) |
| `git status` on `vpdl/kb`, `vpdl/cli.py`, `vpdl/sources`, `tests/kb`, `docs/kb` | no changes |
| `git status` on `vpdl/dl`, `tests/dl`, `docs/dl` | **changed by a concurrent session, not by this work**: `vpdl/dl/report.py` (new), a `vpdl-dl results` command in `vpdl/dl/cli.py`, artefact archiving, and the matching docs/tests, all written 07:22–07:25 on 2026-09-23 while this branch was being built. Nothing in this branch writes to those paths; `tests/dl` still passes (107). |

Not executed here: any download, any real narrative, any pretraining or
fine-tuning on real data, any model with pretrained weights, any timing claim.
Those are **NOT RUN — REQUIRES DGX SPARK**.
