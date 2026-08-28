# Running Log

Append one entry per build / training run. Newest at the top.
Format: date · what ran (command) · outcome · artifacts.

---

## 2026-08-28 (evening, CPU box) — closing the gap to the published bar

**Result: mean ROC-AUC 0.9229 -> 0.9445, mean MCC 0.4626 -> 0.6894** over the
three MMR genes with meaningful n. MSH2 now exceeds the best published single
predictor. The change is a one-flag configuration fix, not a new model.

### What was actually wrong

The broad panel was built with `include_gnomad: false`, so the stage-1
checkpoint's prior schema has **19** columns. Stage 2 pins its feature order to
the checkpoint's (correctly — weight transfer requires it), which silently
discarded the **8** richer features the MMR table does have:

    gnomad_log10_af  acmg_ba1  acmg_bs1  acmg_pm2
    af_plddt  af_disordered  in_interpro_domain  is_functional_site

`gnomad_log10_af` is the one PROJECT_PLAN.md Phase 3 step 1 explicitly requires
as an *input* feature, citing MVmamba's own 0.895->0.901 ablation. It was being
thrown away by the warm start.

### Measured (leave-one-gene-out, priors mode, 10,000-iteration CIs, seed 42)

| holdout | warm-start (19 feats) | scratch (27 feats) | delta | best published |
|---|---|---|---|---|
| MLH1 | 0.9425 | **0.9641** | +0.0217 | 0.984 (BayesDel) |
| MSH2 | 0.8527 | **0.9007** | +0.0480 | 0.896 (TranceptEVE-L) **beaten** |
| MSH6 | 0.9734 | 0.9686 | -0.0048 | 0.991 (MetaRNN) |
| PMS2 | 0.9559 | 1.0000 | +0.0441 | 0.956 (AlphaMissense) |

MCC gains are larger and uniform: +0.262 / +0.212 / +0.206 / +0.103. MCC is the
plan's primary metric.

Against PROJECT_PLAN.md Phase 3 step 7's MVmamba anchor (AUC 0.901 / AUPR 0.848
/ MCC 0.656): now 0.9445 / 0.9389 / 0.6894 — ahead on all three. **This is not
a like-for-like claim**: MVmamba reports on 18,731 variants across many genes,
this is 662 clinical variants across three. Report both denominators.

### New guard: `--gene_constant_priors {auto,drop,keep}`

The five gnomAD *gene-level* constraint columns (pLI, oe_lof, oe_mis, mis_z,
syn_z) hold one value per gene, so under leave-one-gene-out they are a
5-dimensional gene identifier, not evidence. Scratch-trained with them in,
**MLH1 collapses to ROC-AUC 0.500 / MCC 0.000 in every seed tried** (n=3);
dropped, 0.965 +/- 0.007. `auto` drops them for `--eval lopo`, keeps them for a
single `--eval holdout`, and `run_mmr_pipeline.py` resolves it once and passes
the same literal to both stages so warm-start schemas still match.

This currently only bites in `--scratch` mode, because the checkpoint schema has
no gnomAD columns at all. **It becomes live the moment the broad panel is
rebuilt with `--all_sources`** — which is exactly what the README recommends for
the full GPU run. Rebuild both stages together.

### Rank normalisation: implemented, not recommended

`--rank_normalize {off,add,replace}` converts each published score to a
within-gene percentile (+ a skip-NaN consensus mean). Motivation was strong: a
*zero-training* rank-mean beats the trained head on every gene, and on MSH2
beats the best published predictor (0.916 vs 0.896). But as head input it did
not reproduce that: `replace` gave no gain over dropping the gene-constant
columns, and `add` helped MSH6 (0.971 -> 0.985) while destabilising MLH1's
threshold (MCC 0.345 +/- 0.375). Left off by default; the flag and the finding
are worth keeping — a plain rank-mean consensus is a strong, honest baseline
the learned head still has to justify itself against.

### Caveats

- Four configurations were compared and the holdout was read each time. The two
  that shipped are defect fixes with an independent rationale (a discarded
  feature schema; a gene-identifier feature), not hyperparameters tuned on the
  test set — but seed-average and confirm via inner validation before writing up.
- Scratch beats warm-start here only because the checkpoint is
  feature-impoverished *and* 98.9% DMS-labelled. The right fix is to rebuild the
  broad panel with `--all_sources` and re-pretrain, so pretraining and
  fine-tuning share the 27-feature schema. Then re-test warm-start vs scratch.
- MLH1 and MSH6 remain ~0.02 ROC-AUC below the best single published predictor.
- PMS2's 1.000 is n=21 with 4 benign. Not a result.
- Artifacts: `data/processed/mmr_transfer_scratch/`.

## 2026-08-28 (later, CPU box) — audit, PMS2 inclusion, published-predictor benchmark

- **PMS2 is now trainable instead of dropped.** `scripts/derive_pms2_homology_range.py`
  derives the PMS2CL homology span from the Ensembl exon table for MANE Select
  `ENST00000265849`: exons 11-15 = c.1145-2589 = **protein codons 382-862**. The
  derivation self-validates (2,589 bp CDS -> 862 aa, matching the pinned P54278).
  Recorded as `src.mmr_dataset.PMS2_PSEUDOGENE_CODON_RANGE`; the gate itself stays
  fail-closed, the range must still be passed explicitly.
  - `scripts/build_mmr_dataset.py --min_stars 2 --pms2_codon_range 382 862`
    -> 74,328 rows; PMS2 contributes **21 clinical labels (17 P / 4 B), all at
    codon <=301**, plus 1,118 in-region-free VUS. 9,139 PMS2 rows inside the
    region keep their data and lose their labels, as intended.
  - Four-gene leave-one-gene-out, frozen priors probe, warm-started from
    `mmr_pipeline_pretrain.pt`, 10,000-iteration CIs:

    | holdout | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n |
    |---|---|---|---|---|---|---|
    | MLH1 | 0.9425 | 0.8822-0.9872 | 0.9787 | 0.4871 | 0.8123 | 208 |
    | MSH2 | 0.8527 | 0.8106-0.8903 | 0.7549 | 0.3986 | 0.4346 | 335 |
    | MSH6 | 0.9734 | 0.9477-0.9910 | 0.9740 | 0.5020 | 0.7753 | 119 |
    | PMS2 | 0.9559 | 0.8382-1.0000 | 0.9903 | 0.5830 | 0.3172 |  21 |

  - **Do not quote the PMS2 row as a performance result.** n=21 with 4 benign;
    the CI reaches 1.0. It demonstrates the gene runs end to end, nothing more.
  - The other three genes' numbers shifted slightly from the earlier three-gene
    run because PMS2 now contributes to their fine-tuning pool.

- **Published-predictor benchmark** — new `scripts/benchmark_published_predictors.py`
  scores all 17 ProteinGym clinical-benchmark predictors + AlphaMissense on this
  repo's own MMR clinical slice. Score orientation is fitted on the broad panel
  with the MMR genes removed, never on the evaluation data. Writes
  `docs/PUBLISHED_COMPARISON.md` + `data/processed/benchmark/`.
  - **Not yet run to completion on committed data.** A provisional pass
    (2,000 bootstrap iterations, three genes, before PMS2 was added) produced
    the numbers below; the definitive 10,000-iteration four-gene run is still
    owed and no benchmark artefact is checked in. Treat these as indicative.
  - Provisional: **the model did not clear the published bar on any Lynch
    gene.** Per-gene ROC-AUC, ours vs best published: MLH1 0.940 vs 0.984
    (BayesDel), MSH2 0.853 vs 0.896 (TranceptEVE-L), MSH6 0.973 vs 0.991
    (MetaRNN). Same ordering on MCC. CIs overlapped in every individual
    comparison, but the gap was consistent across all three genes and all
    three metrics.
  - The unsupervised methods (GEMME, EVE, TranceptEVE-L, PoET) also beat it,
    so the gap cannot be explained away as ClinVar circularity in the
    baselines' favour. GEMME was the strongest single comparator.
  - **PMS2 has zero ProteinGym zero-shot coverage** — AlphaMissense is its only
    published comparator, and its prior feature vector is correspondingly thin.
  - This is PROJECT_PLAN.md Phase 3 step 7's bar, and it is currently unmet.

- **Fixes.** MVmamba feature cache rejected itself permanently once any variant
  failed alignment (compared post-alignment `meta` against the raw table);
  `run_mmr_transfer.py`'s summary JSON reported requested rather than evaluated
  splits, so it claimed PMS2 was evaluated on every `--exclude_pms2` build. Two
  regression tests added for the PMS2 range (82 passing).

## 2026-08-28

- **Stage 2b: first successful ESM-2 backbone fine-tune** — `run_mmr_pipeline.py`
  (siamese / ProPath recipe, leave-one-gene-out), on the Windows CUDA box
  - **First Stage-2b run ever to complete.** The 2026-08-27 attempt OOM'd in the
    backward pass; the AMP + grad-accum + checkpointing knobs and the VRAM
    preflight added that day are what made it fit.
  - Leave-one-**gene**-out over the MMR panel — these are *unseen-gene* numbers,
    not residue-disjoint ones:

    | holdout | mode | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n | best_epoch |
    |---|---|---|---|---|---|---|---|---|
    | MLH1 | siamese | 0.8993 | 0.8446–0.9445 | 0.9757 | 0.5275 | 0.4344 | 208 | 1 |
    | MSH2 | siamese | 0.8776 | 0.8350–0.9162 | 0.7931 | 0.2972 | 0.0567 | 335 | 3 |
    | MSH6 | siamese | 0.9073 | 0.8508–0.9536 | 0.9195 | 0.6519 | 0.2674 | 119 | 1 |

  - Mean ROC-AUC **0.895** (n-weighted 0.890) over 662 held-out clinical rows.
    PMS2 absent, as expected — the fail-closed pseudogene gate excludes it by default.
  - Artifacts: `data/processed/esm_finetune/esm_finetune_results_siamese_lopo.csv`

- **Cautions on the above — do not write these into the paper unqualified**
  - `best_epoch` is 1, 3, 1. The run early-stops almost immediately, so it is not
    yet established that backbone gradients bought anything over the frozen
    probe. **Run `--n_unfrozen_layers 0` as the ablation floor before claiming
    the fine-tune is what produced these numbers.**
  - MCC-optimal thresholds span 0.0567–0.4344 across three genes. Scores do not
    transfer on an absolute scale; MSH2 reaches MCC 0.297 despite ROC-AUC 0.878.
    Any deployed threshold must be per-gene or recalibrated.
  - PR-AUC moves opposite to ROC-AUC between MLH1 (0.976 vs 0.899) and MSH2
    (0.793 vs 0.878) → per-gene base rates differ sharply. Report prevalence
    per gene alongside these.
  - Single seed. Holdouts are small (119–335) and all three CIs overlap
    heavily, so no gene is significantly different from another. CIs are
    10,000-iteration (`run_mmr_pipeline.py` defaults `--n_bootstrap` to 10,000
    and passes it through to stage 2b), matching the paper's protocol section.
    Note the *standalone* scripts default to 2,000 — pass it explicitly there.

## 2026-08-23

- **Leakage-clean full training** — `scripts/train_extended.py --no_dms_features`
  - 80 genes, 190,494 labelled rows, group-disjoint CV, DMS-derived features removed
  - Train-vs-val AUC gap ≈ 0.000 → no classic overfitting
  - Clinical-only slice: **ROC-AUC 0.963 / MCC 0.78** (AM baseline: 0.945 / 0.75)
  - All-labels: 0.716 (honest number; earlier 0.9987 was circular via `dms_bin_median`)
  - Artifacts: `data/processed/extended_train/ext80_*`, checkpoints

- **Overfitting diagnostic added** — `scripts/diagnose_overfitting.py`
  - Proved `dms_bin_median` == flipped label on 97.3% of rows (target leakage)

- **Label-inversion bug fixed** in `src/extended_builder.py`
  - ProteinGym `DMS_score_bin=1` = top fitness half (tolerated); builder had
    mapped it straight to "pathogenic" → ~185k labels were inverted
  - Verified vs AlphaMissense anchor + PG-clinical anchor; dataset rebuilt, re-audited 12/12

- **Expanded dataset build** — `make_expanded_panel.py` + `build_extended_dataset.py --panel_file`
  - Panel 10 → 80 genes; master table 110,124 → 1,156,625 rows
  - 50,304 isoform-mismatched DMS + 152 AlphaMissense rows dropped by wt-validation
  - Audit: 12/12 passed (`audit_report.json`); ClinVar↔PG-clinical agree on all 1,919 overlaps
  - Train CSV: `extended_dataset_train.csv` (190,494 rows: 82,149 P/LP · 108,345 B/LB)

## TODO / next runs

- [x] GPU box: install `requirements-cuda.txt`, verify `torch.cuda.is_available()` — done 2026-08-28
- [ ] Definitive benchmark: `python scripts/benchmark_published_predictors.py
      --n_bootstrap 10000 --model_results data/processed/mmr_transfer/mmr_transfer_results_lopo.csv`
      (~90 min CPU; run after rebuilding the MMR table with `--pms2_codon_range 382 862`)
- [ ] Stage 2b ablation floor: rerun siamese LOGO with `--n_unfrozen_layers 0`
- [ ] Stage 2b seed averaging (>=3 seeds)
- [ ] Remaining recipes on the same splits: `wt_site`, VariPred probe, MVmamba pooled
- [ ] Full model: `train_extended.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D --no_dms_features`
- [ ] Leave-one-**protein**-out CV on the broad 80-gene panel (still owed; distinct from the MMR gene-out above)
- [ ] Score VUS risk tiers from ESM-mode run (`ext80_vus_predictions.csv`)
