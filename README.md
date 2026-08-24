# Variant Pathogenicity DL

Gene- and panel-level missense-variant pathogenicity prediction:
**ClinVar supervision → ESM-2 embeddings + PLLR → residual MLP → calibrated
evaluation**, now extended with a **multi-source dataset** that unifies
ClinVar, ProteinGym (DMS + clinical benchmark + 17 published model scores),
AlphaMissense and UniProt domain annotations into one leakage-audited table.

---

## Quickstart

```bash
# One command reproduces EVERYTHING (tests -> dataset -> audit -> training).
bash run_pipeline.sh                       # priors-only model, CPU OK
FEATURES="esm+priors" bash run_pipeline.sh # full ESM-2 model, NVIDIA GPU advised
SKIP_BUILD=1 bash run_pipeline.sh          # reuse an existing dataset build

# Or stage by stage:
python -m venv .venv && .venv/bin/pip install -r requirements.txt

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
python scripts/build_mmr_dataset.py --exclude_pms2     # Phase-1 4-gene dataset (+gnomAD v4)
python scripts/pretrain_esm_80.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D
python scripts/run_mmr_transfer.py \
    --checkpoint data/processed/transfer/pretrain_leave_gene_out_esm_priors.pt \
    --features esm+priors --eval lopo

# 5. Unit tests for parsing / splitting / MMR-gnomAD logic
python tests/test_datasets.py && python tests/test_mmr_modules.py
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
| `main.py` | end-to-end training CLI (single or comma-separated genes) |
| `scripts/build_extended_dataset.py` | extended-dataset build CLI |
| `src/data_loader.py` | ClinVar streaming, HGVS-p parsing, star filtering |
| `src/external_datasets.py` | ProteinGym / AlphaMissense / UniProt downloaders & parsers |
| `src/extended_builder.py` | multi-source merge, label precedence, manifest |
| `src/gnomad.py` | gnomAD v4 GraphQL adapter + BA1/BS1/PM2 frequency flags |
| `src/mmr_dataset.py` | pinned MMR references, VCEP tiers, PMS2 pseudogene gate |
| `src/mvmamba_features.py` | MVmamba WT/VT global/local features + masked-marginal scorers |
| `src/fusion.py` | concat + GateWave fusion heads (Phase-5 start) |
| `src/eval_utils.py` | bootstrap CIs (10k) + MCC-optimal threshold tuning |
| `src/transfer.py` | 80-gene pretraining / MMR fine-tuning primitives |
| `src/esm_extractor.py` | ESM-2 embedding + PLLR extraction (sliding windows) |
| `src/dataset.py` | group-disjoint stratified CV splits |
| `src/model.py` | residual MLP head |
| `src/loss.py` | focal loss, weighted BCE |
| `src/calibration.py` | ECE/MCE/Brier, temperature & isotonic calibrators, plots |
| `tests/test_datasets.py` | unit tests |
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
