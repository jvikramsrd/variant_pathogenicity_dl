# Variant Pathogenicity DL

Gene- and panel-level missense-variant pathogenicity prediction:
**ClinVar supervision → ESM-2 embeddings + PLLR → residual MLP → calibrated
evaluation**, now extended with a **multi-source dataset** that unifies
ClinVar, ProteinGym (DMS + clinical benchmark + 17 published model scores),
AlphaMissense and UniProt domain annotations into one leakage-audited table.

---

## Quickstart

### Lynch-syndrome MMR project (recommended entrypoint)

One command runs the whole **download -> clean/process -> train** pipeline
for the MLH1/MSH2/MSH6/PMS2 project end to end (PROJECT_PLAN.md Phases 0-3,
stopping at fine-tuning by design -- no calibration/LLM/fusion stages here).
By default this includes **true ESM-2 backbone gradient fine-tuning**
(`src/esm_finetune.py`, ProPath's Siamese recipe) as the final training
stage -- the actual deep-learning training, not just a linear probe over
frozen embeddings. That stage is expensive and GPU strongly advised.

```bash
python run_mmr_pipeline.py --dry_run      # preview every stage's command first

# Full model on a GPU box: broad-panel structural/domain sources + ESM-2
# embeddings + true backbone fine-tuning (all on by default)
python run_mmr_pipeline.py --features esm+priors \
    --esm_model facebook/esm2_t33_650M_UR50D --all_sources

# CPU-friendly / quick smoke test: priors-only features, skip the backbone
# fine-tuning stage (stops at the frozen-embedding pretrain/fine-tune stages)
python run_mmr_pipeline.py --no-full_finetune

# Re-run training only, reusing already-downloaded/cleaned data
python run_mmr_pipeline.py --skip_build
```

No GPU? The pipeline checks for a CUDA/MPS device **before** the downloads and
the frozen-embedding stages and stops there rather than starting a multi-day
CPU fine-tune. Either pass `--no-full_finetune`, or force it with
`--allow_cpu_finetune` alongside a tiny checkpoint and a shallow unfreeze:

```bash
python run_mmr_pipeline.py --allow_cpu_finetune \
    --esm_model facebook/esm2_t12_35M_UR50D --n_unfrozen_layers 2 \
    --eval holdout --holdout_gene MLH1
```


Cross-platform wrappers: `bash run_mmr_pipeline.sh [args...]` (macOS/Linux),
`.\run_mmr_pipeline.ps1 [args...]` (Windows) -- both just forward every
argument to `run_mmr_pipeline.py`. See `python run_mmr_pipeline.py --help`
for the full flag list (feature mode, PMS2 handling, CIMRA CSV, pretrain
mode, leave-one-gene-out vs single holdout, full fine-tuning options).

This is a sibling to `run_pipeline.py` below (the original, still-unchanged
single-/multi-gene extended-dataset workflow) -- use whichever matches what
you're trying to reproduce.

### Original single-/multi-gene workflow

One command reproduces EVERYTHING (tests -> dataset -> audit -> training).

```bash
# macOS/Linux
bash run_pipeline.sh
FEATURES="esm+priors" bash run_pipeline.sh
SKIP_BUILD=1 bash run_pipeline.sh
```

```powershell
# Windows PowerShell
.\run_pipeline.ps1
.\run_pipeline.ps1 -Features "esm+priors"
.\run_pipeline.ps1 -SkipBuild
```

The shared runner can also be called directly on any platform:

```bash
python run_pipeline.py --features priors
python run_pipeline.py --features esm+priors --skip-build
```

Or stage by stage:

```bash
python -m venv .venv
# Activate it, then:
python -m pip install -r requirements.txt

# 1. Build the extended multi-source dataset (default 10-gene panel, or expand it)
python scripts/make_expanded_panel.py                 # panel = all human ProteinGym DMS proteins
python scripts/build_extended_dataset.py \
    --panel_file data/raw/uniprot/expanded_panel.json # build the full master table
python scripts/audit_extended_dataset.py              # 12-point audit + train-CSV emission

# Use --min_stars 1 only when intentionally including lower-confidence ClinVar labels.

# 2. Train on one gene (original workflow)
python main.py --gene TP53 --esm_model facebook/esm2_t33_650M_UR50D

# 3. Pool several proteins and inject every external prior as a feature
python main.py --gene TP53,BRCA1,PTEN --extra_features all

# 4. Train the MLP head on the full audited extended dataset
python scripts/train_extended.py                                  # priors-only (CPU OK)
python scripts/train_extended.py --features esm+priors \
    --esm_model facebook/esm2_t33_650M_UR50D                      # full model (GPU)

# Recommended clinical-target configuration (nested calibration; DMS features excluded)
python scripts/train_extended.py --no_dms_features --clinical_weight 5

# Two-stage transfer learning for the MMR/Lynch project
#   Stage 1: pretrain the head on ALL 80 panel genes' ESM embeddings
#   Stage 2: fine-tune on dedicated MLH1/MSH2/MSH6(/PMS2) clinical data,
#            evaluate leave-one-gene-out with bootstrap CIs + fusion ablations
python scripts/build_mmr_dataset.py --exclude_pms2     # Phase-1 4-gene dataset
    # (+ gnomAD v4 AF/constraint, AlphaFold pLDDT, InterPro, UniProt point
    #  features, all on by default for the small 4-gene panel -- use the
    #  --skip_* flags to disable any of them for an offline run)
python scripts/pretrain_esm_80.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D
python scripts/run_mmr_transfer.py \
    --checkpoint data/processed/transfer/pretrain_leave_gene_out_esm_priors.pt \
    --features esm+priors --eval lopo

# 5. Orthogonal functional-assay data (Phase 2; validation-only, never fed
#    into ESM-branch training) -- MaveDB has a live API (MSH2 Jia et al. 2021
#    LOF screen + MLH1 2025 abundance assay); CIMRA has no bulk API, so supply
#    a CSV extracted from the paper supplements (see src/cimra.py docstring).
python scripts/build_mmr_dataset.py --exclude_pms2 \
    --cimra_csv data/raw/cimra/cimra_oddspath.csv    # optional

# 6. Broad-panel gnomAD + AlphaFold + InterPro + UniProt point features (the
#    *pretraining* data gets the same feature set the MMR fine-tuning stage
#    does, not just the 4 MMR genes) -- all opt-in, one extra REST/GraphQL
#    call per gene per source:
python scripts/build_extended_dataset.py --panel_file data/raw/uniprot/expanded_panel.json \
    --all_sources   # = --include_gnomad --include_structure --include_interpro --include_functional_sites

# 7. Full ESM-2 fine-tuning (backbone gradients, not a frozen linear probe) --
#    ProPath's Siamese/PLLR recipe or a CSBJ-style token classifier:
python scripts/finetune_esm_mmr.py --mode siamese --n_unfrozen_layers -1 \
    --esm_model facebook/esm2_t33_650M_UR50D --eval lopo \
    --backbone_lr 1e-5 --batch_size 8 --epochs 10 --gradient_checkpointing

# 8. Benchmark all four Phase-3 fine-tuning strategies on identical MMR splits
#    (mvmamba recipe / VariPred linear probe / ProPath siamese / CSBJ token
#    classifier) before committing to one:
python scripts/compare_finetune_strategies.py --holdout_gene MSH2 \
    --esm_model facebook/esm2_t33_650M_UR50D

# 9. ESM-1b vs ESM2-650M masked-marginal zero-shot -- don't assume ESM2 wins:
python scripts/compare_backbones.py

# 10. Leave-one-protein-out CV over the broad pretraining panel (the honest
#     transfer-to-a-new-gene estimate, not just residue-disjoint folds):
python scripts/eval_leave_one_protein_out.py --features priors

# 11. Sequence-cluster-disjoint split (MMseqs2, 20% id / 20% cov) so
#     near-duplicate/paralogous proteins never straddle train/val:
python scripts/build_cluster_split.py \
    --panel_json data/raw/uniprot/expanded_panel.json \
    --out_dir data/processed/extended

# 12. Unit tests for parsing / splitting / MMR-gnomAD / MaveDB / CIMRA / ESM
#     fine-tuning logic
python tests/test_datasets.py && python tests/test_mmr_modules.py \
    && python tests/test_new_data_sources.py
```

## NVIDIA GPU support

ESM-2 extraction is ~20x faster on a CUDA GPU and is required in practice for
`--features esm+priors` over large panels. On an NVIDIA machine install:

```bash
pip install -r requirements-cuda.txt        # CUDA 12.x PyTorch build
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

The code auto-detects CUDA (`get_device()`: CUDA → MPS → CPU) and runs ESM-2
forward passes under fp16 autocast on GPUs. Keep `--extract_batch_size` small
(default 8); raise the training `--batch_size` freely.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
 data acquisition   │ ClinVar variant_summary.txt.gz  (labels)     │
                    │ ProteinGym v1.3 DMS substitutions (fitness)  │
                    │ ProteinGym v1.3 clinical benchmark (labels)  │
                    │ ProteinGym zero-shot scores (EVE/ESM1b/...)  │
                    │ AlphaMissense aa-substitutions   (priors)    │
                    │ UniProt sequences + domains      (reference) │
                    └───────────────────┬──────────────────────────┘
                                        ▼
        src/extended_builder.py   unified master table (110k+ rows)
        scripts/build_extended_dataset.py     + manifest.json (SHA-256 provenance)
                                        ▼
        src/data_loader.py        per-gene labelled / VUS CSVs (cached)
                                        ▼
        src/esm_extractor.py      ESM-2 last-layer embeddings per variant
                                  z = [h_wt ‖ h_mut ‖ Δh ‖ |Δh| ‖ PLLR]
                                  (+ optional 22 external prior columns)
                                        ▼
        src/train.py              K-fold CV, groups = protein:residue
                                  focal / weighted-BCE · AdamW · cosine ·
                                  early stop on val ROC-AUC
                                        ▼
        src/calibration.py        temperature scaling & isotonic regression
                                  vs PLLR baseline · reliability diagrams
                                        ▼
                                  OOF metrics CSVs · fold checkpoints ·
                                  VUS risk predictions
```

## Repository layout

| Path | Role |
|------|------|
| `run_mmr_pipeline.py` (+ `.sh`/`.ps1`) | **recommended entrypoint**: download -> clean -> train for the Lynch-syndrome MMR project |
| `main.py` | end-to-end training CLI (single or comma-separated genes) |
| `scripts/build_extended_dataset.py` | extended-dataset build CLI |
| `src/data_loader.py` | ClinVar streaming, HGVS-p parsing, star filtering |
| `src/external_datasets.py` | ProteinGym / AlphaMissense / UniProt downloaders & parsers |
| `src/extended_builder.py` | multi-source merge, label precedence, manifest |
| `src/gnomad.py` | gnomAD v4 GraphQL adapter: variant AF + BA1/BS1/PM2 flags + gene-level constraint (pLI, oe_mis, mis_z) |
| `src/mavedb.py` | MaveDB REST client (MSH2 Jia 2021 LOF screen, MLH1 abundance assay) |
| `src/cimra.py` | CIMRA OddsPath CSV loader + Tavtigian ACMG-strength classification |
| `src/structure.py` | AlphaFold DB per-residue pLDDT structural-confidence feature |
| `src/interpro.py` | InterPro domain/family/superfamily calls (complements UniProt domains) |
| `src/mmr_dataset.py` | pinned MMR references, VCEP tiers, PMS2 pseudogene gate, functional-assay attach |
| `src/mvmamba_features.py` | MVmamba WT/VT global/local features + masked-marginal scorers |
| `src/esm_finetune.py` | full ESM-2 fine-tuning (ProPath siamese / CSBJ token-classifier) |
| `src/fusion.py` | concat + GateWave fusion heads (Phase-5 start) |
| `src/eval_utils.py` | bootstrap CIs (10k) + MCC-optimal threshold tuning |
| `src/transfer.py` | 80-gene pretraining / MMR fine-tuning primitives |
| `src/esm_extractor.py` | ESM-2 embedding + PLLR extraction (sliding windows) |
| `src/dataset.py` | group-disjoint stratified CV splits |
| `src/model.py` | residual MLP head |
| `src/loss.py` | focal loss, weighted BCE |
| `src/calibration.py` | ECE/MCE/Brier, temperature & isotonic calibrators, plots |
| `scripts/finetune_esm_mmr.py` | full backbone fine-tuning + leave-one-gene-out eval CLI |
| `scripts/compare_finetune_strategies.py` | benchmarks all 4 Phase-3 fine-tuning strategies |
| `scripts/compare_backbones.py` | ESM-1b vs ESM2-650M masked-marginal comparison |
| `scripts/eval_leave_one_protein_out.py` | leave-one-protein-out CV over the broad panel |
| `scripts/build_cluster_split.py` | MMseqs2 sequence-cluster-disjoint split |
| `tests/test_datasets.py`, `tests/test_mmr_modules.py`, `tests/test_new_data_sources.py` | unit tests |
| `docs/PROJECT_DOCUMENTATION.md` | **start here** -- what was built, why, and what it improves on |
| `docs/CODE_GUIDE.md` | **how it works** -- module mechanics, the data merge step by step, better alternatives |
| `docs/DATA_TO_MODEL.md` | dataset schema + how the data is fed to the model (CSV row -> tensor), and what is verified |
| `docs/DATASETS.md` | **full dataset documentation** (licences, schemas, rules) |
| `docs/CODE_REVIEW.md` | issues found & fixed + complete changelog |
| `data/raw/`, `data/processed/` | cached artefacts (never edit by hand) |

## Key design guarantees

* **No spatial leakage** — all variants at one `(protein, residue)` stay in a
  single CV fold (`StratifiedGroupKFold`, asserted at runtime).
* **Numbering safety** — every external row is validated against the UniProt
  canonical sequence (wt-residue check) before joining; RefSeq NP_ accessions
  are matched by *exact whole-protein sequence equality*.
* **Provenance** — every remote artefact is checksummed into
  `data/processed/extended/manifest.json` with URL, version, size and date.
* **Label precedence** — ClinVar > ProteinGym-clinical > single-assay DMS bin;
  raw columns are always preserved alongside.

**Start here:** `docs/PROJECT_DOCUMENTATION.md` is the single entry point --
what was built, why each design decision was made, and what it improves on
relative to both the original code and the published methods it draws from.
Then `docs/CODE_GUIDE.md` for the mechanics: how each module works, the merge
walked through step by step, and the better alternative for every significant
design decision.

See `docs/DATASETS.md` for the exhaustive data reference and
`docs/CODE_REVIEW.md` for what changed relative to the original pipeline.
See `docs/FINE_TUNING_FINDINGS.md` for the current performance findings,
implemented training safeguards, and the recommended evaluation sequence.
See `docs/DATA_PIPELINE_HARDENING.md` for the ingestion, merge, provenance,
and conflict-quarantine guarantees.
See `docs/MMR_TRANSFER_WORKFLOW.md` for the Lynch-syndrome MMR workflow:
pinned MLH1/MSH2/MSH6/PMS2 references, the PMS2 pseudogene gate, gnomAD v4
frequency features (BA1/BS1/PM2), 80-gene ESM pretraining, leave-one-gene-out
evaluation, and MVmamba-style WT/VT + masked-marginal feature extraction.
