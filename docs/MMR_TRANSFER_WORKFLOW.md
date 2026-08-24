# MMR transfer-learning workflow (PROJECT_PLAN.md Phases 0–3, DL + pretraining)

Implementation status: **Phases 0–3 data + deep-learning stages are
implemented and smoke-tested end-to-end.** The ClinVar ingestion code
(`src/data_loader.py`) is untouched — every stage composes it from above.

## Phase 1 — dedicated MMR dataset

`scripts/build_mmr_dataset.py` builds the four-gene table in a separate
`data/mmr/` directory:

1. Pins the canonical references (MLH1 P40692/756aa, MSH2 P43246/934aa,
   MSH6 P52701/1360aa, PMS2 P54278/862aa) and hard-fails on length drift.
2. Reuses the shared multi-source builder (ClinVar + ProteinGym-clinical +
   AlphaMissense + published zero-shot scores + UniProt domains); raw
   artefacts are symlinked from the main cache instead of re-downloaded.
3. Joins **gnomAD v4** per-gene missense AFs via the official GraphQL API
   (`src/gnomad.py`, cached under `data/raw/gnomad/`), emitting explicit input
   features (`gnomad_af_joint`, `gnomad_log10_af`, ...) plus BA1/BS1/PM2 ACMG
   frequency flags.
4. Flags InSiGHT/ClinGen VCEP expert-panel calls (`expert_panel`,
   `evidence_tier`, `tier_weight`).
5. Enforces the **PMS2 exon 11–15 pseudogene gate** fail-closed:
   `--pms2_homology_csv` (with `orthogonally_confirmed` flags), an explicitly
   verified `--pms2_codon_range START END`, or `--exclude_pms2`. Without one of
   these the build refuses to run; unconfirmed homology-region labels are
   withheld, never trusted.
6. Emits balanced-label diagnostic subsets (`mmr_balanced_diagnostic.csv`)
   and the leave-one-MMR-gene-out manifest
   (`leave_one_gene_out_splits.json`).

```bash
python scripts/build_mmr_dataset.py --exclude_pms2          # safe default today
python scripts/build_mmr_dataset.py --pms2_codon_range 419 862   # when you have verified the span
```

## Stage 1 — pretraining over ALL 80 panel-gene ESM embeddings

`scripts/pretrain_esm_80.py` fits the classification head on frozen ESM-2
embeddings (+ leakage-safe priors, never DMS-derived columns) across the whole
80-protein broad table:

```bash
python scripts/pretrain_esm_80.py --features esm+priors \
    --esm_model facebook/esm2_t33_650M_UR50D        # GPU advised for the full run
python scripts/pretrain_esm_80.py --features priors # CPU fallback (minutes)
```

* `--mode leave_gene_out` excludes every MMR gene from pretraining (transfer
  estimate); `--mode practical` allows them (adapted model, no unseen-gene
  claim).
* The checkpoint stores its exact prior-column order and ESM block dimension.

## Stage 2 — MMR fine-tuning + leave-one-gene-out evaluation

`scripts/run_mmr_transfer.py` warm-starts from the Stage-1 checkpoint on
clinical-label MMR rows only (ClinVar / PG-clinical sources; DMS-only labels
and functional-assay values are excluded by construction), then evaluates:

```bash
python scripts/run_mmr_transfer.py \
    --checkpoint data/processed/transfer/pretrain_leave_gene_out_esm_priors.pt \
    --features esm+priors --eval lopo
```

Per held-out gene it reports ROC-AUC / PR-AUC / MCC with percentile bootstrap
CIs (10,000 iterations by default) at a decision threshold tuned by MCC on an
inner validation slice drawn only from fine-tuning genes.  The mandatory
ablation battery runs on identical splits: ESM branch-only, prior branch-only,
pretrained-fused head, concat fusion, GateWave gated fusion.

Checkpoint schema mismatches (ESM backbone change or prior-column drift)
abort the run rather than silently invalidating transfer.

## Phase 3 extras

* MVmamba-style WT/VT global/local features with the ±3-residue optimum
  window (sweepable) live in `src/mvmamba_features.py`
  (`extract_mvmamba_cached`), including mutation-centred windows for MSH6's
  1360-aa chain.
* True masked-marginal zero-shot baselines for both mandated backbones
  (ESM-1b `facebook/esm1b_t33_650M_UR50S`, ESM2-650M) via
  `MaskedMarginalScorer.score` — compare before committing to ESM-2.

```python
from src.mvmamba_features import MaskedMarginalScorer
scores = MaskedMarginalScorer("facebook/esm1b_t33_650M_UR50S").score(variants_df, sequence)
```

## Still open (later plan phases)

* Phase 2 CIMRA OddsPath encoding and the MSH2 DMS held-out evaluation axis.
* Phase 4 LLM clinical-text branch (BioBERT + ClinVar-BERT preprocessing).
* Phase 6 gene-specific calibration (F1/PredictMD reuse + domain-aggregate
  fallback).
