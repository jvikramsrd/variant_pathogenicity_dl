# Experiment plan

The matrix below is generated from `vpdl/slm/experiments.py`
(`vpdl-slm experiments --markdown`), so the plan, the commands and this
document cannot drift apart. **Nothing in it has been run.**

## How a result becomes reportable here

1. **Three seeds** (42, 43, 44) per arm. A difference smaller than the seed
   spread is not a result — this project has paid for that lesson
   (docs/RUNLOG.md, 2026-09-06: two arms "beat" a comparator on seed 42 and
   fell back inside noise over three seeds).
2. **The leakage audit passes** for that split before training
   (`vpdl-slm leakage`, exit 3 on a critical finding).
3. **Thresholds and calibrators come from validation**, never from the data
   being scored.
4. **The trivial baselines are reported beside it**: majority, the per-gene
   label rate (reported by the audit as `gene_shortcut`), TF-IDF.
5. **Provenance matches** before two runs are compared: dataset hash, split
   hash, preprocessing version, git commit — all recorded per run in
   `runs/slm_genomic/registry.jsonl`.
6. Anything not measured is written `TBD — RUN ON DGX SPARK`, not estimated.

## Primary metrics

Five-class: macro F1, balanced accuracy, multi-class MCC, per-class F1.
Pathogenic vs benign (VUS excluded): MCC, PR-AUC, ROC-AUC, F1 with bootstrap
CIs, sensitivity, specificity, precision.
Calibration: Brier, top-label and class-wise ECE, reliability, per gene and per
disease, before and after calibration.
Evidence: multi-label precision/recall/F1 for types and ACMG codes, polarity
F1, span F1.
Explanation: unsupported-claim rate, citation validity, evidence coverage,
contradiction rate.
VUS: ranking AUROC and enrichment against later reclassification, calibration,
abstention.
Efficiency: parameters, trainable parameters, memory, training time, latency.

All experiments: **NOT RUN — REQUIRES DGX SPARK**. Seeds: 42, 43, 44 per arm.

| id | experiment | hypothesis | split | model | tasks | status |
|---|---|---|---|---|---|---|
| `EXP-001` | Majority | — | variant | majority | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-002` | TF-IDF + logistic regression | H2 | variant | tfidf_lr | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-003` | TF-IDF + linear SVM | H2 | variant | tfidf_svm | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-004` | BioBERT | H1 | variant | hf:dmis-lab/biobert-base-cased-v1.2 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-005` | PubMedBERT (BiomedBERT) | H1 | variant | hf:microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-006` | BioClinicalBERT | H1 | variant | hf:emilyalsentzer/Bio_ClinicalBERT | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-007` | ClinVar-BERT-style | H2 | text | hf:dmis-lab/biobert-base-cased-v1.2 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-008` | Existing project LLM | — | variant | medgemma:27b via vpdl kb-ask | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-009` | Custom SLM (no genomic pretraining) | H1 | variant | best of EXP-004..006 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-010` | + broad genomic continued pretraining | H1 | variant | EXP-009 backbone after `vpdl-slm pretrain` | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-011` | + structured features | H3 | variant | EXP-010 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-012` | + evidence extraction | H4,H5 | variant | EXP-011 | classify, evidence | NOT RUN — REQUIRES DGX SPARK |
| `EXP-013` | + ACMG / ClinGen context | H4 | variant | EXP-012 | classify, evidence, acmg | NOT RUN — REQUIRES DGX SPARK |
| `EXP-014` | Multi-task | H5 | variant | EXP-013 | classify, evidence, acmg | NOT RUN — REQUIRES DGX SPARK |
| `EXP-015` | + VUS objective | H6 | variant | EXP-014 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-016` | + explanation supervision | H4 | variant | EXP-015 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-017` | + teacher distillation | H5 | variant | EXP-014 + qwen3:32b labels | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-018` | + DL representation | H7 | variant | EXP-014 + DLRepresentation | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-019` | + MMR adapter | H8 | mmr | EXP-014 + LoRA adapter | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-020` | Final SLM | — | variant | decided by EXP-009..019 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-021` | Unseen genes | H1 | gene | EXP-020 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-022` | Disease holdout | H1 | disease | EXP-020 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-023` | Temporal holdout | H1 | temporal | EXP-020 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-024` | Independent functional validation | H7 | functional | EXP-020 | classify | NOT RUN — REQUIRES DGX SPARK |
| `EXP-025` | MMR / Lynch evaluation | H8 | mmr | EXP-020 (zero-shot) vs EXP-019 (adapter) | classify | NOT RUN — REQUIRES DGX SPARK |

### Commands

- `EXP-001`: `vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet --kinds majority --out runs/slm_genomic/EXP-001`
- `EXP-002`: `vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet --kinds tfidf_lr --out runs/slm_genomic/EXP-002`
- `EXP-003`: `vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet --kinds tfidf_svm --out runs/slm_genomic/EXP-003`
- `EXP-004`: `vpdl-slm finetune --config configs/slm/exp004_biobert.toml`
- `EXP-005`: `vpdl-slm finetune --config configs/slm/exp005_pubmedbert.toml`
- `EXP-006`: `vpdl-slm finetune --config configs/slm/exp006_bioclinicalbert.toml`
- `EXP-007`: `vpdl-slm finetune --config configs/slm/exp007_clinvarbert_style.toml`  
  The published ClinVar-BERT weights are not available; this reproduces its recipe (sentence filtering, lab-template dedup) on our data, and says so.
- `EXP-008`: `vpdl kb-eval --models medgemma:27b`  
  Different task by design: the KB reads passages and refuses; it never classifies a variant (docs/kb/DESIGN.md rule 4). Reported as context, not as a classification arm.
- `EXP-009`: `vpdl-slm finetune --config configs/slm/exp009_custom_slm.toml`
- `EXP-010`: `vpdl-slm pretrain --config configs/slm/pretrain_dgx.toml && vpdl-slm finetune --config configs/slm/exp010_genomic_pretrained.toml`
- `EXP-011`: `vpdl-slm finetune --config configs/slm/exp011_structured.toml`
- `EXP-012`: `vpdl-slm finetune --config configs/slm/exp012_evidence.toml`  
  Evidence targets are rule-derived weak labels; their own accuracy is unmeasured until an annotated sample exists (DATA GAP).
- `EXP-013`: `vpdl-slm finetune --config configs/slm/exp013_acmg.toml`
- `EXP-014`: `vpdl-slm finetune --config configs/slm/exp014_multitask.toml`
- `EXP-015`: `vpdl-slm finetune --config configs/slm/exp015_vus.toml`  
  Scored by later reclassification (vpdl-slm evaluate --vus-old/--vus-new), which needs an archived ClinVar release.
- `EXP-016`: `vpdl-slm finetune --config configs/slm/exp016_explanation.toml`  
  The deterministic explainer needs no training; this arm exists for a generative explainer and is PROPOSED — no generative head is implemented.
- `EXP-017`: `vpdl-slm teacher --examples ... && vpdl-slm finetune --config configs/slm/exp017_teacher.toml`
- `EXP-018`: `vpdl-slm finetune --config configs/slm/exp018_dl.toml`  
  Needs `vpdl-dl export`; the DL branch has not been trained yet.
- `EXP-019`: `vpdl-slm finetune --config configs/slm/exp019_mmr_adapter.toml`
- `EXP-020`: `vpdl-slm finetune --config configs/slm/exp020_final.toml`
- `EXP-021`: `vpdl-slm finetune --config configs/slm/exp021_gene_holdout.toml`
- `EXP-022`: `vpdl-slm finetune --config configs/slm/exp022_disease_holdout.toml`
- `EXP-023`: `vpdl-slm finetune --config configs/slm/exp023_temporal.toml`
- `EXP-024`: `vpdl-slm finetune --config configs/slm/exp024_functional.toml`  
  MSH2 (MaveDB 00000050-a-1) and MLH1 abundance (00001218-a-1); Spearman and class AUROC against the assay, not against ClinVar.
- `EXP-025`: `vpdl-slm finetune --config configs/slm/exp025_mmr.toml`

### Clinical-text ablations

| arm | what the model reads | experiment |
|---|---|---|
| A | structured features only | EXP-001/baselines structured_lr |
| B | clinical text only | EXP-010 |
| C | clinical text + structured features | EXP-011 |
| D | clinical text + literature (pretraining with PubMed) | EXP-010 vs a corpus without PubMed |
| E | clinical text + evidence extraction | EXP-012 |
| F | clinical text + ACMG/ClinGen context | EXP-013 |
| G | clinical text + VUS objective | EXP-015 |
| H | clinical text + DL representation | EXP-018 |
| I | full system | EXP-020 |

### Hypotheses

- **H1**: Broad genomic continued pretraining improves transfer to unseen genes.
- **H2**: Clinical text improves evidence-grounded reasoning over structured features alone.
- **H3**: Structured genomic features improve classification over text alone.
- **H4**: Evidence extraction improves explanation grounding.
- **H5**: Multi-task learning improves evidence-aware classification.
- **H6**: Explicit VUS training improves uncertainty quality.
- **H7**: DL representations add complementary biological information.
- **H8**: MMR adaptation improves MMR performance without destroying broad transfer.
