# MMR transfer-learning workflow (PROJECT_PLAN.md Phases 0–3, DL + pretraining)

Implementation status: **Phases 0–3 data + deep-learning stages are
implemented and smoke-tested end-to-end.** The ClinVar ingestion code
(`src/data_loader.py`) is untouched — every stage composes it from above.

**One-command entrypoint**: `run_mmr_pipeline.py` (+ `.sh`/`.ps1` wrappers at
the repo root) chains every stage documented in this file --
download/clean the broad panel, download/clean the MMR-specific dataset,
Stage-1 pretrain, Stage-2 fine-tune, Stage-2b true ESM-2 backbone gradient
fine-tuning -- in one command, stopping there by design. Stage-2b (the actual
DL training stage, not a frozen-embedding linear probe) runs **by default**;
pass `--no-full_finetune` to stop at the frozen-embedding stages instead
(faster, CPU-friendly). Because that default is expensive, the pipeline probes
for a CUDA/MPS device **before** the downloads and the frozen-embedding stages
and stops there if none is visible -- pass `--allow_cpu_finetune` to run it on
CPU anyway. Run `python run_mmr_pipeline.py --dry_run` first to
preview the exact command sequence for your flags before committing to a
multi-hour/GPU run. Everything below can also still be run stage-by-stage by
hand for finer control.

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

## Stage 2 — MMR fine-tuning (frozen embeddings) + leave-one-gene-out evaluation

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

## Phase 2 — orthogonal functional-assay data

Both are joined as **validation-only** columns
(`src.mmr_dataset.FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS`) -- never fed into
`src.transfer.TRANSFER_PRIOR_COLS`, so the ESM-branch performance claims stay
non-circular against this independent evidence, exactly like the existing
ProteinGym DMS exclusion.

* **MaveDB** (`src/mavedb.py`, live REST API, no manual download needed):
  MSH2 gets the Jia et al. 2021 loss-of-function screen
  (`urn:mavedb:00000050-a-1`, 17,746 variants) -- per the plan, "it
  outperforms every computational predictor for MSH2 classification"; MLH1
  gets a 2025 cellular-abundance assay (`urn:mavedb:00001218-a-1`, a
  *different* evidence class, tagged `mave_assay_type="abundance"`). MSH6/PMS2
  have no MaveDB score set as of this writing --
  `load_mmr_mavedb_features(..., check_for_new=True)` re-runs the live search
  so a future submission isn't silently missed.
  ```bash
  python scripts/build_mmr_dataset.py --exclude_pms2   # MaveDB joined by default
  python scripts/build_mmr_dataset.py --skip_mavedb    # offline runs
  ```
* **CIMRA OddsPath** (`src/cimra.py`): no bulk API exists (paywalled
  supplementary tables), so supply a CSV yourself -- see the module docstring
  for the exact schema and source papers (Rayner et al. 2022 for PMS2). Rows
  with a hypothesized splicing mechanism are excluded automatically (CIMRA is
  cell-free and cannot detect splicing).
  ```bash
  python scripts/build_mmr_dataset.py --cimra_csv data/raw/cimra/oddspath.csv
  ```

## Phase 3 extras (this change) -- fine-tuning strategy benchmark + backbone choice

All four fine-tuning strategies PROJECT_PLAN.md Phase 3 step 4 asks to
compare on our own data now exist in this codebase:

| Strategy | Implementation |
|---|---|
| MVmamba recipe (frozen WT/VT global+local pooled) | `src/mvmamba_features.py` (pre-existing) |
| VariPred frozen linear probe | `src/esm_extractor.py` + `src/fusion.BranchHead` (pre-existing) |
| ProPath Siamese/PLLR (backbone gradients) | `src/esm_finetune.py`, `mode="siamese"` |
| CSBJ per-residue token classifier (backbone gradients) | `src/esm_finetune.py`, `mode="wt_site"` |

```bash
# All four, identical leave-one-gene-out split, directly comparable metrics:
python scripts/compare_finetune_strategies.py --holdout_gene MSH2 \
    --esm_model facebook/esm2_t33_650M_UR50D

# Just the full fine-tune, LOPO across all four MMR genes:
python scripts/finetune_esm_mmr.py --mode siamese --n_unfrozen_layers -1 \
    --esm_model facebook/esm2_t33_650M_UR50D --eval lopo \
    --backbone_lr 1e-5 --batch_size 8 --epochs 10 --gradient_checkpointing

# ESM-1b vs ESM2-650M masked-marginal zero-shot ("don't assume ESM2 wins"):
python scripts/compare_backbones.py
```

`n_unfrozen_layers` controls how much of the backbone gets gradient updates:
`-1` = full fine-tune, `0` = frozen (ablation floor), `N>0` = last N
transformer layers only (cheaper middle ground). `--gradient_checkpointing`
trades compute for memory on the 650M checkpoint.

## Circularity-safe sequence clustering (Phase 1)

`scripts/build_cluster_split.py` clusters every panel protein at the plan's
20% identity / 20% coverage threshold via MMseqs2 (external binary, not a
Python dependency -- see the script docstring for install instructions) and
assigns whole clusters to train/val, so paralogs or near-duplicate isoforms
never straddle a split.

## Leave-one-protein-out over the broad panel (Phase 3)

`scripts/eval_leave_one_protein_out.py` generalizes the MMR-specific
leave-one-gene-out evaluation to any subset of the 80-gene pretraining panel
-- "the clinically relevant question for rare disease genes" per
`docs/TRAINING_NOTES.md`. Expensive by construction (one full fit per
held-out gene); start with `--genes` restricted to a handful.

## Broad-panel gnomAD + structural/domain sources (this change)

gnomAD v4 allele-frequency features were previously joined onto the MMR table
but never reached `src.transfer.TRANSFER_PRIOR_COLS`, so the prior branch
silently ignored them; that's fixed. They were also only ever fetched for the
4 MMR genes -- `scripts/build_extended_dataset.py --include_gnomad` now
extends the same fetch (one GraphQL call per gene, opt-in) to the whole
pretraining panel, so "the genome data used for pretraining" carries the same
explicit AF input feature the MMR fine-tuning stage does (PROJECT_PLAN.md
Phase 3 step 1 / MVmamba's own AF ablation, AUC 0.895->0.901).

Three more sources were added the same way, each wired all the way through
to `TRANSFER_PRIOR_COLS` (a feature joined onto the table but not in that
tuple is invisible to the model -- verified by
`tests/test_new_data_sources.py::test_new_prior_columns_are_wired_into_transfer_prior_cols`):

| Source | Feature | Flag (broad panel) | Flag (MMR, on by default) |
|---|---|---|---|
| gnomAD constraint (`src/gnomad.py`) | `gnomad_pli`, `gnomad_oe_lof`, `gnomad_oe_mis`, `gnomad_mis_z`, `gnomad_syn_z` (gene-level, broadcast) | `--include_gnomad` | `--skip_gnomad_constraint` |
| AlphaFold DB (`src/structure.py`) | `af_plddt`, `af_disordered` (per-residue structural confidence) | `--include_structure` | `--skip_structure` |
| InterPro (`src/interpro.py`) | `in_interpro_domain` (complements UniProt's own domain calls) | `--include_interpro` | `--skip_interpro` |
| UniProt point features (`src/external_datasets.py`) | `is_functional_site` (active/binding site, PTM, disulfide) | `--include_functional_sites` | `--skip_functional_sites` |

`--all_sources` on `scripts/build_extended_dataset.py` turns on all four at
once. See `docs/DATASETS.md` §16-19 for full provenance/licence/API details.

## Still open (later plan phases)

* Phase 4 LLM clinical-text branch (BioBERT + ClinVar-BERT preprocessing).
* Phase 5 fusion of the LLM branch into GateWave (the ESM/priors fusion heads
  already exist and are benchmarked by `scripts/run_mmr_transfer.py`).
* Phase 6 gene-specific calibration (F1/PredictMD reuse + domain-aggregate
  fallback).
