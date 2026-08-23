# Training Notes — Extended Dataset & Deep-Learning Stage

*Date: 2026-08-23 · machine: CPU-only dev box · target: NVIDIA GPU box for ESM runs*

---

## 1. What was done

### 1.1 Data expansion (10 → 80 proteins)

| | Before | After |
|---|---|---|
| Gene panel | 10 curated genes | **80 genes** = every human protein with a ProteinGym v1.3 DMS assay |
| Master table | 110,124 × 40 | **1,156,625 × 40** unique substitutions |
| Labelled rows | 14,131 | **190,494** (82,149 P/LP · 108,345 B/LB) |
| ClinVar supervision | 2,067 | 5,466 labelled + 25k VUS |
| AlphaMissense coverage | 10 proteomes | 80 proteomes (~1.16M scored variants) |

New tooling:
- `scripts/make_expanded_panel.py` — batched UniProt REST resolution of all
  human DMS entry names → HGNC symbols + canonical sequences (cached JSON).
- `scripts/build_extended_dataset.py --panel_file ...` — inject a pre-resolved
  panel instead of per-gene lookups.
- `scripts/audit_extended_dataset.py` — independent 12-check integrity audit
  (`audit_report.json`): key uniqueness, wt-residue validation against
  canonical sequences, label-precedence re-derivation, provenance-token
  coherence, cross-source conflicts, feature-coverage report.
- Vectorised `_wt_ok_mask` + interval-join domain flags (the row-wise versions
  would have taken hours at 1M+ rows).

Merge guarantees kept: every source normalised to
`(uniprot_id, position, wt_aa, mut_aa)`; 50,304 isoform-mismatched DMS rows and
152 AlphaMissense rows *dropped*, not merged; strict label precedence
ClinVar > PG-clinical > single-assay DMS. Audit: **12/12 passed**; the 1,919
variants covered by both ClinVar and PG-clinical agree 100%.

### 1.2 Bug found & fixed during DL bring-up ⚠️

The first training run exposed a **label-orientation inversion**: ProteinGym's
`DMS_score_bin=1` marks the *top half of assay fitness* (= tolerated), but the
builder had mapped it straight to `label=1` ("pathogenic"), inverting ~185k
DMS-sourced labels. Two independent anchors confirmed:

- P(AlphaMissense ≥ 0.9 | bin=1) = 0.217 vs (bin=0) = 0.506 → bin=1 looks benign
- PG-clinical rows: mean DMS score −0.8 for pathogenic vs +1.9 for benign

Fixed in `src/extended_builder.py` (DMS-derived label is now `1 − dms_bin`),
mirrored in the auditor, dataset rebuilt and re-audited. CV metrics were
unchanged (AUC is flip-invariant when the head re-learns the mapping), but the
*semantics and any prospective risk ranking were wrong before this fix.*

### 1.3 Deep-learning stage

- New trainer `scripts/train_extended.py` consuming the audited train CSV.
  Two modes:
  - **priors** — external scores only (AlphaMissense, inverted DMS median,
    n_assays, domain flag, 17 zs_* published models): trains anywhere.
  - **esm+priors** — adds ESM-2 embeddings `[h_wt ‖ h_mut ‖ Δh ‖ |Δh|]` +
    PLLR per variant (fp16 autocast on CUDA).
- Same leakage contract as the original pipeline: StratifiedGroupKFold on
  `uniprot_id:position`, per-fold StandardScaler, focal loss, AdamW + cosine,
  early stop on val ROC-AUC, temperature/isotonic calibration.
- Honest evaluation slices: pooled OOF over **all** labels vs **clinical_only**
  slice (ClinVar ∪ PG-clinical sourced), because DMS bins measure assay
  fitness, not clinical consequence.

### 1.4 NVIDIA support added

- `requirements-cuda.txt` (CUDA 12.x PyTorch wheel index) — one-command install.
- README "NVIDIA GPU support" section + verification snippet.
- fp16 autocast in `src/esm_extractor.py::_embed_spans` (≈2–3× throughput),
  TF32 matmul enabled in the trainer, device auto-detection already present
  (CUDA → MPS → CPU). `--extract_batch_size` decoupled from training batch.

Run on your GPU box:

```bash
pip install -r requirements-cuda.txt
python scripts/train_extended.py --features esm+priors \
    --esm_model facebook/esm2_t33_650M_UR50D
```

---

## 2. Results

### 2.1 Overfitting / leakage audit ⚠️ (read before quoting any number)

The first run reported ROC-AUC **0.9987** on all labels. Diagnosis
(`scripts/diagnose_overfitting.py`):

1. **Target leakage found**: on single-assay rows (97.3% of labelled data) the
   feature `dms_bin_median` equals `1 − label` by *construction* — the answer
   was literally in the feature matrix. The 0.9987 was circularity, not
   learning.
2. **No classic overfitting**: with the leaky features removed, train-vs-val
   AUC gap per fold ≈ **0.000** (−0.001 … +0.011). The head generalises; folds
   early-stop within 2–16 epochs.
3. `-DMS_score` as a baseline is likewise circular on the DMS slice (the label
   is a median split of that same score); only its clinical-slice value is
   interpretable.
4. Remaining caveats: AlphaMissense itself was trained on ClinVar (possible
   temporal contamination of the clinical slice), and isotonic calibration is
   fitted on each fold's validation split (mild optimism, documented in
   `src/train.py`).

`train_extended.py --no_dms_features` reproduces the clean configuration.

### 2.2 Leakage-clean numbers (priors mode, 80 genes, 3-fold group-disjoint CV)

| Slice | Model | ROC-AUC | MCC |
|---|---|---|---|
| all labels (190k) | MLP + isotonic | 0.716 | 0.32 |
| all labels | AlphaMissense | 0.709 | 0.30 |
| clinical only | **MLP + isotonic** | **0.963** | **0.78** |
| clinical only | AlphaMissense | 0.945 | 0.75 |

Reading:
- All-labels ≈ 0.71 is the honest number once circularity is removed: the task
  "predict assay fitness bins without seeing the assay" is intrinsically hard.
- **The clinically meaningful headline is 0.963 ROC-AUC / 0.78 MCC** on
  ClinVar ∪ PG-clinical labels — +1.8 pt AUC over AlphaMissense alone by
  combining AM + 17 published zero-shot scores + domain flags (no DMS, no ESM).
- Temperature scaling cut ECE from 23% → 22% here and isotonic to 11%; fixed-
  threshold metrics (MCC@0.5) are sensitive to score shift — trust AUC and the
  calibrated variants.

Artifacts under `data/processed/extended_train/`: `ext80_oof_predictions.csv`,
`ext80_pooled_metrics.csv`, `ext80_slice_baselines.csv`,
`ext80_reliability_diagram.png`, fold checkpoints in `checkpoints/`.

---

## 3. What we can achieve next

1. **Full model on your GPU box** (`--features esm+priors`,
   t33_650M checkpoint). Expect the biggest jump exactly where it matters: the
   clinical slice, because PLLR/embeddings add protein-context information no
   prior score contains. Run with `--no_dms_features` for leakage-clean
   evaluation. Rough budget: ~1M mutant forward passes ≈ hours on a mid-range
   GPU, cached afterwards.
2. **Per-gene leave-one-protein-out CV** — measures how well the model
   transfers to proteins it never saw (the clinically relevant question for
   rare disease genes). One CLI flag away with the existing group machinery.
3. **Label-source-aware losses / two-stage curriculum** — pretrain on the
   185k DMS labels, fine-tune on the 5.5k clinical labels, so the noisy-but-
   plentiful signal doesn't drown the clean-but-small one.
4. **Prospective VUS triage at scale** — once ESM features are cached, score
   all ~966k unlabelled substitutions (including the 25k ClinVar VUS) into
   calibrated risk tiers for prioritisation.
5. **Publishable benchmark story** — leakage-audited multi-source table +
   baseline suite (AM, EVE/ESM1b/GEMME zero-shots, PLLR, our head) on identical
   splits is directly comparable to ProteinGym-style leaderboards.
6. **Scale-out options** — full ClinVar mining beyond 80 genes (the streaming
   extractor already handles arbitrary gene lists), or swap in newer PLMs
   (ESM-3-class / AM replacement models) behind the same extractor interface.
