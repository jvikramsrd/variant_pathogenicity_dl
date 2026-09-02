# Technology Stack

What this project is built with, and **why each choice was made**. Grounded in
`requirements.txt` / `requirements-cuda.txt`, the `src/` and `scripts/` code,
and `PROJECT_PLAN.md`. Companion to `docs/PROJECT_DOCUMENTATION.md` (what was
built) and `README.md` §Architecture (data flow).

---

## 1. At a glance

| Layer | Technology | Role in this project |
|---|---|---|
| Language / runtime | **Python 3.13+** (3.14 on the dev box) | single language, whole pipeline |
| Deep learning | **PyTorch ≥ 2.1** (`torch`, `torchvision`) | all model training, ESM inference, fine-tuning |
| Protein language models | **ESM-2** (`facebook/esm2_t33_650M_UR50D`), **ESM-1b** | variant embeddings + zero-shot PLLR scores |
| NLP / LLM branch | **Hugging Face `transformers` ≥ 4.40** | loads ESM checkpoints; BioBERT for the clinical-text branch |
| Classical ML | **scikit-learn ≥ 1.3** | MLP-adjacent utilities, `StratifiedGroupKFold`, `IsotonicRegression`, metrics |
| Scientific computing | **NumPy ≥ 1.24**, **SciPy ≥ 1.11**, **pandas ≥ 2.0** | arrays, bootstrap statistics, the whole tabular data layer |
| Bioinformatics | **BioPython ≥ 1.83**, **MMseqs2** (external binary) | PDB parsing; sequence-cluster-disjoint splits |
| HTTP / ingestion | **`requests` ≥ 2.31**, **`urllib3` ≥ 2.0**, **`tqdm`** | resumable downloads, GraphQL/REST API calls, progress bars |
| Visualization | **matplotlib ≥ 3.7**, **seaborn ≥ 0.13** | reliability diagrams, calibration plots |
| Testing | **pytest** | 60+ unit tests over parsing / splitting / API adapters / fine-tuning logic |
| Compute | **CUDA 12.x GPU** (fine-tuning); CPU-only for data + frozen-embedding stages | see §9 |

No cloud services, no external LLM API in the deployed critical path, no
orchestration framework — it is a single-command Python pipeline.

---

## 2. Language & runtime

- **Python 3.13+.** `PROJECT_PLAN.md` Phase 0 notes 3.14 is very new for PyTorch
  wheels; the dev box runs 3.14.7, the recommended target is 3.13.
- **`venv`** for isolation (not conda) — readiness is gated on a post-install
  stamp file so a half-finished `pip install` is never silently adopted.
- Cross-platform: `run_mmr_pipeline.{sh,ps1}` and `run_pipeline.{sh,ps1}`
  wrappers forward every argument to the Python entrypoints, so the same
  pipeline runs on Linux, macOS and Windows.

---

## 3. Deep learning — PyTorch

Plain PyTorch (`torch.nn`), no Lightning / Keras / fast.ai wrapper.

- **Model head** (`src/model.py`): a residual MLP —
  `Linear(d→h) → N×(Linear → LayerNorm → GELU → Dropout + residual) → Linear(h→1)`.
- **Fusion heads** (`src/fusion.py`): `BranchHead` (single-branch baseline),
  `ConcatFusionHead` (the plan's default: BatchNorm → Linear → ReLU → Dropout →
  Linear), `GateWaveFusionHead` (MVmamba's gated design: sigmoid branch gate +
  softmax per-feature gating + GLU + residual).
- **Losses** (`src/loss.py`): `FocalLoss`, `WeightedBCE` with a **fixed**
  `pos_weight` from the fitting partition (not recomputed per mini-batch — that
  produced noisy gradients on small final batches).
- **Optimisation**: `AdamW` + `CosineAnnealingLR`, early stopping with
  best-weight restoration.
- **Device selection** (`src/esm_extractor.py:get_device`): CUDA → MPS → CPU,
  auto-detected.
- **Mixed precision for fine-tuning**: `torch.autocast` (bf16 where supported,
  else fp16 + `GradScaler`), **gradient checkpointing**, and **gradient
  accumulation** — the three knobs that make the 650M Siamese fine-tune fit in
  ~10 GiB VRAM instead of ~51 GiB fp32.

---

## 4. Protein language models — ESM

| Checkpoint | Params | Use |
|---|---|---|
| `facebook/esm2_t33_650M_UR50D` | 650 M | **default** — embeddings + PLLR, all fine-tuning recipes |
| `facebook/esm1b_t33_650M_UR50S` | 650 M | backbone comparison (ESM-1b beat ESM-2 for clinical pathogenicity in two prior studies) |
| `facebook/esm2_t6_8M` … `t36_3B` | 8 M – 3 B | size-sweep; the 8M/35M checkpoints are used for CPU smoke tests |

- Loaded via `transformers.AutoModelForMaskedLM` / `AutoTokenizer` — the
  **masked-LM head is required** for true masked-marginal PLLR scoring
  (`log P(mut | context) − log P(wt | context)`).
- Imported **lazily** inside the extractor/fine-tuner classes so `import src`
  stays cheap on machines without `transformers`.
- Long sequences (MSH6, 1360 aa > ESM's ~1022 window): overlapping sliding
  windows, hidden states averaged across covering windows, log-probs averaged
  in log space (`src/esm_extractor.py`); VariPred's asymmetric-window recipe
  for the fine-tuning path.
- **MVmamba-style features** (`src/mvmamba_features.py`): global (mean-pooled)
  + local (mutation ±3 residues) embeddings for both WT and variant sequences.

---

## 5. LLM clinical-reasoning branch (Phase 4)

- **Base encoder: BioBERT** (via `transformers`), `max_length` 512 — the
  ClinVar-BERT recipe.
- **Self-distillation teacher: MedGemma-27B-text-it**, self-hosted, used
  offline at training time only to generate ACMG-grounded chain-of-thought
  rationales (RareDAI pattern). No API dependency in the deployed model.
- Same `transformers` stack as the ESM branch; the branch plugs into the exact
  `src/fusion.py` interface the prior-feature branch already uses.

*(Phase 4 is in progress — the encoder choice and pipeline are fixed, the
implementation is not yet in `src/`.)*

---

## 6. Classical ML & scientific computing

- **scikit-learn**: `StratifiedGroupKFold` (group = `"{uniprot_id}:{position}"`,
  asserted at runtime — no same-residue leakage across folds),
  `IsotonicRegression`, ROC-AUC / PR-AUC / MCC / Brier metrics.
- **SciPy**: 10,000-iteration bootstrap confidence intervals, binomial power
  test for the calibration go/no-go gate.
- **pandas / NumPy**: the entire multi-source merge, the master variant table
  (~1.16 M rows), feature matrices, caching to CSV/Parquet.

---

## 7. Bioinformatics tooling

| Tool | Type | Use |
|---|---|---|
| **BioPython** | Python pkg | `Bio.PDB.PDBParser` for AlphaFold structure files; HGVS-p parsing helpers |
| **MMseqs2** | external binary (not a pip dep) | `mmseqs easy-cluster` at 20% identity / 20% coverage → sequence-cluster-disjoint train/val split (`scripts/build_cluster_split.py`). Install via bioconda / prebuilt binary |
| **AlphaFold DB** | REST | per-residue pLDDT structural-confidence feature (`src/structure.py`) |
| **FoldX / DSSP** | optional external | full structural ΔΔG pipeline — **not required**; the code degrades to pLDDT-only when absent |

---

## 8. External data sources & APIs

All ingestion is through `requests` with atomic `.part` staging, exponential
backoff, ZIP-integrity checks, and a **SHA-256 provenance manifest**
(`data/processed/extended/manifest.json`).

| Host | Source | What it provides |
|---|---|---|
| `ftp.ncbi.nlm.nih.gov`, `www.ncbi.nlm.nih.gov` | **ClinVar** `variant_summary`, **E-utils** | clinical labels (≥ 2-star), VUS holdout |
| `marks.hms.harvard.edu` | **ProteinGym v1.3** | DMS substitutions, clinical benchmark, 17 zero-shot model scores (EVE, ESM-1b, …) |
| `storage.googleapis.com` | **AlphaMissense** | per-variant pathogenicity prior (full-proteome scan) |
| `rest.uniprot.org` | **UniProt** | canonical sequences, domain annotations, point features |
| `gnomad.broadinstitute.org` | **gnomAD v4 GraphQL API** | allele frequency (PM2/BA1/BS1 flags) + gene-level constraint (pLI, oe_mis, mis_z) |
| `alphafold.ebi.ac.uk` | **AlphaFold DB** | pLDDT per-residue confidence |
| `www.ebi.ac.uk` | **InterPro** | domain / family / superfamily calls |
| `api.mavedb.org` | **MaveDB REST** | MSH2 DMS (Jia 2021), MLH1 abundance assay — held out for validation |
| `rest.ensembl.org` | **Ensembl REST** | genomic coordinate mapping for the PMS2 pseudogene-homology range |
| `zenodo.org` | archived supplementary data | gene-specific calibration priors / thresholds (Chen et al. 2026) |

**CIMRA OddsPath** (MMR functional assay) has no bulk API — supplied as a CSV
extracted from paper supplements (`src/cimra.py`).

---

## 9. Compute & performance

- **GPU (fine-tuning):** NVIDIA CUDA 12.x. Install `requirements-cuda.txt`
  (`--extra-index-url https://download.pytorch.org/whl/cu126`). ESM-2 extraction
  is ~20× faster on GPU; fp16 autocast on the forward pass.
- **VRAM preflight** (`estimate_finetune_vram_gib`): a closed-form estimate
  refuses an impossible fine-tune config *before* the multi-hour download
  stages rather than OOM-ing in the backward pass. 650M Siamese full unfreeze ≈
  51 GiB fp32 → ≈ 10 GiB with AMP + accumulation + checkpointing.
- **CPU-only path:** the pipeline checks for a CUDA/MPS device before the
  downloads and stops at the frozen-embedding stages rather than starting a
  multi-day CPU fine-tune (override with `--allow_cpu_finetune` + a tiny
  checkpoint).
- **Dev box:** Linux, 12 cores, ~14 GB RAM, no GPU — used for data build,
  audits, and frozen-embedding linear probes only. Stage-2b numbers came from a
  separate Windows CUDA machine.

---

## 10. Model architecture components

| Component | Where | Notes |
|---|---|---|
| Feature vector | `src/esm_extractor.py` | `z = [h_wt ‖ h_mut ‖ Δh ‖ |Δh| ‖ PLLR]` (+ ~22 optional prior columns, each with an `is_missing_*` indicator) |
| Residual MLP head | `src/model.py` | 256 hidden, dropout ~0.15 |
| Concat / GateWave fusion | `src/fusion.py` | frozen-branch projections → shared dim → shallow head |
| ESM-2 Siamese fine-tune | `src/esm_finetune.py` | ProPath recipe: separate WT/VT passes, pool at mutated residue |
| ESM-2 token-classifier fine-tune | `src/esm_finetune.py` | CSBJ recipe |
| Two-stage transfer | `src/transfer.py` | Stage 1 pretrain head on 80-gene panel → Stage 2 warm-start on MMR labels → Stage 2b backbone gradient fine-tune |
| Calibration | `src/calibration.py` | temperature scaling + isotonic regression, nested (never fitted on the fold it scores); ECE / MCE / Brier + reliability diagrams |
| Evaluation | `src/eval_utils.py` | MCC-optimal threshold tuned on an inner slice; 10,000-iteration bootstrap CIs |

---

## 11. Engineering practices

- **Provenance:** every remote artefact → URL + version + byte count + date +
  SHA-256 in `manifest.json`, regenerated each build.
- **Independent audit:** `scripts/audit_extended_dataset.py` re-derives the
  merge from a separate implementation and runs 12 integrity checks.
- **Leakage instrumentation:** `scripts/diagnose_overfitting.py` (target-leakage
  detector), balanced-label diagnostic subsets, schema-drift abort (checkpoints
  store prior-column order + ESM block dim).
- **Reproducibility:** one command (`run_mmr_pipeline.py`) does
  download → clean → train; `--dry_run` prints every stage's command first.
- **Tests:** `pytest`, ~60+ unit tests (`tests/test_*.py`) covering HGVS
  parsing, group-disjoint splitting, the gnomAD/MaveDB/CIMRA adapters, and the
  ESM fine-tuning logic. No CI workflow committed yet.

---

## 12. Deliberately not used

| Not used | Why |
|---|---|
| conda | `venv` + pip is enough; conda only needed for the optional MMseqs2 binary |
| PyTorch Lightning / Keras | the training loops are small and explicit; wrappers would hide the leakage-control logic |
| LoRA / PEFT / bitsandbytes / DeepSpeed | full fine-tune fits in ~10 GiB with plain AMP + checkpointing + accumulation |
| External LLM API (OpenAI, etc.) in the critical path | GPT-4 shows run-to-run nondeterminism even at temperature 0; the deployed model is a frozen local encoder |
| dbNSFP | ~30 GB and redundant with the ProteinGym `zs_*` score columns |
| Vector DB / RAG infra | the LLM branch is a fine-tuned classifier, not retrieval-augmented (a bounded literature-mining sub-module is optional and separate) |
| Cloud / Kubernetes / Airflow | single-machine, single-command pipeline |

---

*Compiled 2026-08-28 from `requirements*.txt`, `src/`, `scripts/`,
`PROJECT_PLAN.md`, `docs/DATASETS.md`, `docs/PAPER_DRAFT.md`.*
