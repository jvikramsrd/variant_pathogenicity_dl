# Missing evidence for `docs/MANUSCRIPT.md`

Every placeholder in the manuscript, what it needs, and the exact command or artifact
that would replace it. Ordered by how much each blocks a claim the paper wants to make.

Dataset identity for all current model results: `extended_dataset.csv` SHA-256
`79B683993967ABD4CE758EEAD0FFD2652CDF6B1DA5E90CBBDC392F5CA0E15010` (re-pinned
2026-09-15, superseding `78EB5D60860C…`; see item 14). Anything re-run for
comparison must read that table, or the comparison is invalid.

---

## 14. Dataset re-pinned 2026-09-15 — MaveDB added as a source, evidence chain reopened

**What changed.** The GPU box rebuilt `data/mmr/processed/extended/extended_dataset.csv`
on 2026-09-13 (`built_at_utc` in the manifest), producing a new dataset SHA-256
(`79b68399…`, was `78eb5d60860c…`) via `build_mmr_dataset.py`. Three things moved at
once, confirmed from the manifest diff:

1. **ClinVar re-downloaded** — a fresher upstream snapshot (`0946d166…`, was
   `186ad283…`), moving one label: `clinvar_labelled` 464→463, `clinvar_vus` 9589→9590.
2. **`sources.gnomad.enabled` corrected `false`→`true`**, with `genes_fetched` now
   populated. `gnomad_rows_panel` is unchanged at 6492, so this looks like the same
   long-standing manifest-flag bug item 2 fixed for the broad panel, now also fixed here
   — not new gnomAD data.
3. **MaveDB is now a populated source**: `"mavedb": {"enabled": true, "rows_with_score":
   17014}`, where before there was no MaveDB block at all. `extended_dataset.csv` shrank
   ~21% (48.5 MB → 38.4 MB) — a materially different table, not a metadata-only rewrite.

**Decision: MaveDB stays validation-only.** [PROJECT_PLAN.md](PROJECT_PLAN.md) Phase 2
designed MaveDB as held-out-from-training specifically *"to make later performance
claims non-circular."* Wiring it into training would reverse a deliberate, reasoned
design call for a large methodological change (label binarisation, loss weighting) with
no worked design behind it yet. Practically, it is also what the code already does:
`prepare_split()` in `scripts/finetune_esm_mmr.py` and `run_mmr_transfer.py` filters
`label_source` to `{clinvar, pg_clinical}`, so MaveDB rows are present in the rebuilt
table but excluded from every LOPO training/eval split unchanged. **No code change was
needed or made.** If MaveDB-in-training is wanted later, treat it as its own ablation
(comparable in spirit to item 8's label-source axis) with an explicit design, not a side
effect of a dataset rebuild.

**The 28 cells on the branch are NOT one comparable set.** Checked
`provenance.dataset_sha256` in every `esm_finetune_summary_siamese_lopo_*.json` under
`data/processed/stage2b_grid/`:

| dataset_sha256 | cells | which |
|---|---|---|
| `79b68399…` (new) | 16 | the main branch×freeze×PLLR grid |
| `78eb5d60860c…` (old) | 12 | all four `ablate_*` feature-family ablations, all 3 seeds |

The 16-cell grid was retrained against the new table; **the 12 ablation cells were not**
— they are exactly what was on the branch before. Pooling all 28 (Table 4, Figure 5)
would fail `assert_comparable()`'s dataset check, correctly, if run.

**What this reopens.** Every item below resolved against `78eb5d60860c…` needs
re-verification or a re-run against `79b68399…` before its numbers can be cited again:

| Item | Status now |
|---|---|
| 1 — priors-only baseline | **Stale.** Re-run needed on `79b68399…`. |
| 3 — feature-family ablations 4-7 | **Stale artifacts, and the ones on the branch are on the OLD dataset anyway** — re-run all 12 ablation cells against `79b68399…` first. |
| 9 — error analysis | **Stale** — reads the grid+ablation predictions, which are now a mixed-dataset set; re-run after item 3. |
| 10 — figures | **Stale** — same mixed-dataset problem; `make_figures.py --strict_provenance` should currently refuse to pool 3 and 5/6 across the 16 vs 12 split if pointed at both. |
| 5 — calibration | Was already open; `recalibrate_grid.py` exists but has not been run against either dataset. |

**PMS2 has zero eligible clinical variants in the rebuilt table.** All 16 new-dataset
cells' summaries record `splits_skipped_no_rows: ["PMS2"]` — confirmed across every one
of them, not an isolated cell. MLH1/MSH2/MSH6 holdout counts are unchanged at their old
values (208/335/119), so this is not a proportional ClinVar-snapshot effect; PMS2 went
from 21 usable holdout variants to exactly 0. The raw ClinVar count only moved by 1
record overall, which cannot explain losing all 21 PMS2 labels — the far more likely
cause is a regression in the PMS2 homology-gate or coordinate mapping introduced by
whatever else changed in `build_mmr_dataset.py` for this build (possibly interacting
with the MaveDB integration or the gnomAD flag fix touching shared code paths). **This
needs to be root-caused before `79b68399…` is trusted as a baseline for anything** — if
it re-runs the 12 ablations against a build that has silently lost an entire gene's
supervision, that work has to be thrown away a second time. Check on the GPU box: does
`prepare_split(df, holdout="PMS2")` on the new `extended_dataset.csv` return zero rows
even before the LOPO wrapper touches it, and does `pms2_homology_excluded` cover 100% of
PMS2 rows now versus a partial exon 11-15 gate before.

**Next concrete action on the GPU box, in order:** (1) root-cause the PMS2 zero-rows
regression above — do not proceed past this step until it is understood and either fixed
or confirmed to be a real, explainable data change; (2) re-run the 12 ablation cells
(`ablate_domains`/`structure`/`prior_scores`/`gnomad_and_scores`, seeds 42/43/44) with
the same command as before against the now-current `extended_dataset.csv`, confirm their
summaries record `dataset_sha256` starting `79b68399`, then re-run error analysis and
figures. Manuscript Table 1 counts were updated to the new manifest's numbers in the
same pass as this entry; Tables 3/4 and the abstract's headline numbers were **not**
touched and still describe the old-dataset run — do not cite them until item 1 and item 3
are re-closed.

---

## 1. Priors-only baseline on the current dataset build — STALE, see item 14

**Reopened 2026-09-15.** The dataset was re-pinned to `79b68399…` (item 14); this
resolution below was computed against the superseded `78eb5d60860c…` build. Re-run
before citing.

**Resolved 2026-09-05 (against the superseded dataset).** Re-run on the build machine against the pinned table
(`78EB5D60…4B4430D`); the summary is now dated `2026-09-05T10:29:27Z` with
`gene_constant_priors_dropped: true` and `n_bootstrap: 10000`. Mean AUROC 0.9507 over the
three scoreable genes (MLH1 0.9651, MSH2 0.9078, MSH6 0.9791, PMS2 1.000), above every cell
in the grid. Table 3, Table 4 row 1, §3.3, §4 and §5 now state the comparison.

**The original problem statement was wrong, and this is worth recording.** The claim was
that the 2026-09-02 run predated the gnomAD join and so read a different feature set. It did
not. `prior_columns_of(df, drop_gene_constant=True)` returns the same 27 columns on both
builds — including `gnomad_log10_af`, `acmg_ba1`, `acmg_bs1` and `acmg_pm2` — over the same
clinical cohort, so the priors-only arm saw an identical feature matrix either side of the
rebuild. The re-run reproduced every metric to the last recorded digit: AUROC, AUPRC, MCC and
all bootstrap bounds on all four genes, with only `built_at_utc` and `runtime_s` changing.
The arms were comparable all along; the withheld comparison cost the paper its headline for
no reason. **Bit-identical metrics across a supposed dataset change are evidence about the
data, not a coincidence to wave through** — the check that would have caught this in advance
is item 7's provenance record.

**Table 3 is complete.** The re-run's per-variant predictions were committed (`bf9bd1c`),
and the threshold-dependent columns were computed from them with
`src.metrics.evaluation_report`, selecting the threshold on the `inner_val` rows exactly as
the grid cells do. They reproduce the run's own AUROC, MCC and per-gene thresholds to six
decimal places, and are appended to `full_metric_panel_by_gene.csv` under the cell slug
`priors_only`. The baseline leads ten of eleven columns; only recall is lower.

---

## 2. Manifest does not describe the table it accompanies — RESOLVED

**Resolved 2026-09-04.** Code fixed on `main` (`34c5505`); `scripts/repair_manifest.py` has
been run on both panels on the build machine. The MMR manifest now records
`extended_dataset.csv` at `78eb5d60860cc08eadb6faa0c0d7fbd22adbe6f277bb95e97a3a680264b4430d`,
matching both the file on disk and the independent pre-run capture in
`data/processed/stage2b_grid/dataset_sha256.txt`, and `parameters.include_gnomad` was
corrected `false → true` by measuring the table. The broad panel reported
"all recorded artefact checksums already match disk" — expected, since it is built by
`extended_builder` alone with no post-hoc modification, which confirms the defect was
specific to the MMR two-phase path.

**Still to do:** commit the two repaired `manifest.json` files (see below). The original
problem description is kept for the record.

---

### Original finding

**Manuscript location.** §2.1 (discrepancy note), Table 1.

**Problem.** `data/mmr/processed/extended/manifest.json` records
`parameters.include_gnomad: false`, `sources.gnomad.enabled: false` and
`stats.gnomad_rows_panel: 0`, but the table carries a joint allele frequency for 6,492
variants and gene constraint for all 74,328. `scripts/build_mmr_dataset.py` joins gnomAD in
its stages 3–4, after `src/extended_builder.py` has written the manifest, and never rewrote
it.

**The checksum was stale too**, which is worse than the flags. `build_mmr_dataset.py` reads
the CSV back, adds the gnomAD columns and rewrites the file, so the recorded digest is the
pre-join one. Measured on the local copy of the build:

| | manifest records | actually on disk |
|---|---|---|
| `extended_dataset.csv` SHA-256 | `63fd5288…` | `c2731292…` |
| bytes | 37,862,375 | 48,450,632 |

A manifest whose checksum does not match the file it names invites a reader to trust it.

**Status.** Fixed in code on `main` (commit `34c5505`): `extended_builder.refresh_manifest`
re-stamps artefact checksums from disk and merges corrections one level deep, keeping
`built_at_utc` and adding `refreshed_at_utc` so the two-phase build stays visible;
`build_mmr_dataset.py` records what stages 3–4 did and calls it after the final write. Four
tests in `tests/test_merge.py`.

**Outstanding.** The manifests already on disk are still stale. Repair them in place — this
does not rebuild or modify any dataset:

```bash
python scripts/repair_manifest.py data/mmr/processed/extended --dry-run   # inspect first
python scripts/repair_manifest.py data/mmr/processed/extended
python scripts/repair_manifest.py data/processed/extended
```

Run this on the machine holding the build that produced the Stage-2b results, then re-record
the corrected checksum in the manuscript's §6 and re-commit the manifest. Until then, the
manuscript's §2.1 discrepancy note must stay.

---

## 3. Feature-group ablations 4–7 — STALE, see item 14

**Reopened 2026-09-15.** These 12 cells' summaries still record `dataset_sha256`
starting `78eb5d60860c…` — the superseded build. They were not part of the 2026-09-13
retrain (only the 16-cell main grid was). Re-run against `79b68399…` before citing.

**Resolved 2026-09-05 (against the superseded dataset).** All four ran on the build machine against the pinned table, at the
grid cell `esmpri_concat_frozen_pllr-residual_seed42` with only the feature set moving. Each
summary records the resolved `prior_columns`, the groups dropped, and `allow_proxy_leak:
false`. Results are in Table 4 rows 4–7 and discussed in §3.6 and §4.

| Row | Cell slug | Prior cols | Seeds | AUROC (3 scoreable) | Δ vs comparator |
|---|---|---|---|---|---|
| 4 | `ablate_structure` | 25 | 3 | 0.9354 ± 0.0074 | +0.0101 |
| 6 | `ablate_domains` | 24 | 3 | 0.9353 ± 0.0068 | +0.0100 |
| 7 | `ablate_prior_scores` | 9 | 3 | 0.8938 ± 0.0075 | −0.0315 |
| 5 | `ablate_gnomad_and_scores` | 5 | 3 | 0.8727 ± 0.0050 | −0.0526 |

Comparator: `esmpri_concat_frozen_pllr-residual`, 0.9253 ± 0.0097 over the same three seeds.
Seed-42 slugs carry no seed suffix; seeds 43 and 44 are suffixed, because `output_tag()` names
files from the cell slug alone and a reused slug would have overwritten the seed-42 artifacts.

**Reading.** Each arm and the comparator are three-seed means, so the scale is the standard
error of their difference, 0.007. Rows 4 and 6 move the mean by +0.010 — about one and a half
standard errors, upward — so neither structural nor domain features do detectable work. Row 7
costs 0.032 and row 5 costs 0.053, four to eight times that scale. Allele frequency's
increment over the external scores (row 7 versus row 5) is 0.0211 against a standard error of
0.0052.

**Why the seeds were worth 1.5 GPU-hours.** At seed 42 alone rows 4 and 6 appeared to *beat*
the comparator by 0.007–0.009, because seed 42 is the highest of the comparator's three draws.
Averaging moved both back inside noise. The single-seed table would have supported a reading —
"removing structural features helps" — that three seeds do not.

**Closed alongside.** `esm_finetune_summary_*.json` now records `n_bootstrap`, so the
manuscript's 10,000-resample claim is checkable against the artifact. The eight seed-43/44
runs carry the field; the four seed-42 ablations and the 16 grid cells predate it and do not.

---

## 4. No models trained on the broad 79-gene panel

**Manuscript location.** §3.1 (final sentence).

**Problem.** `data/processed/extended/` holds a validated 1,152,863-row × 68-column table that
passes all 11 audit checks, with 189,006 labelled variants. No model has been trained on it,
so the paper reports it as a dataset artifact only. This is the obvious route to a cohort
large enough to support the accuracy claims the MMR panel cannot.

**Command.** `python scripts/train_extended.py` (see `--help`; use a group-based split, not
random). **Note.** 185,600 of the 189,006 labels are single-assay DMS; only 2,148 are
clinical. Any headline figure from this panel is a DMS-proxy result unless the evaluation is
restricted to clinical labels.

---

## 5. No calibration applied to any reported model

**Manuscript location.** §2.5, §3.4, §4.

**Problem.** ECE ≈ 0.19–0.20 for every arm. `src/calibration.py` implements
`TemperatureScaling` and `IsotonicCalibrator`, but neither was applied, because
`scripts/finetune_esm_mmr.py` computed inner-validation probabilities to select the threshold
and then discarded them — leaving no leak-free split on which to fit a calibrator. Fitting on
the held-out gene is the leak the protocol forbids.

**Status.** Fixed in code (commit `e919bef`): the script now writes
`esm_finetune_valpreds_{tag}.csv`. The fix **postdates the reported runs**, so any calibrated
result requires re-running at least one tier.

**Command.** Re-run the three-seed tier-1 arm (approximately 3 h, not the full 17.84 h):
```bash
python scripts/run_stage2b_grid.py --tiers 1 --esm_model facebook/esm2_t33_650M_UR50D \
  --mode siamese --eval lopo --batch_size 1 --grad_accum 8 --gradient_checkpointing \
  --epochs 10 --n_bootstrap 10000 --out_dir data/processed/stage2b_grid
```
Then fit temperature scaling on the valpreds, re-select the threshold on calibrated
validation scores, and recompute the panel. **No recalibration script exists yet.**

---

## 6. No random-split diagnostic

**Manuscript location.** §2.6, §4 (the paragraph on why held-out-gene evaluation matters).

**Problem.** The manuscript argues that random splits inflate performance, but reports no
number for how much on this data, so the argument rests on reasoning alone.

**Required.** A random-split run over the same table with everything else held constant. No
random-split evaluation mode currently exists in `scripts/run_mmr_transfer.py` or
`scripts/finetune_esm_mmr.py`; `--eval` accepts only `lopo` and `holdout`. **A code change is
needed**, not just a run.

---

## 7. Run artifacts carry no dataset, feature-schema or split identity — RESOLVED

**Manuscript location.** §2.5 (tracking), and implicitly every comparison in the paper.

**Problem.** `esm_finetune_summary_*.json` recorded cell slug, branch, fusion, PLLR mode, seed,
model name, freeze depth, split names, checkpoints and runtime — but no dataset hash, feature
schema, split hash, git commit or library versions. Two runs on different dataset builds were
therefore indistinguishable from their artifacts. Item 1 is exactly this failure in practice:
the priors-only baseline was withheld from the paper on a *suspicion* about which table it had
read, and nothing in the artifact could settle it. Re-running settled it — the suspicion was
wrong — but that is an expensive way to answer a question a hash answers for free.

**Mechanism — done 2026-09-05.** `src/provenance.py` provides `provenance_record()`, written
into every new run summary: full dataset SHA-256, an order-independent feature-schema hash, a
split hash taken over the held-out *keys* (not counts — two splits of equal size holding
different variants are different splits), git commit with a dirty flag, and versions of
`torch`, `transformers`, `scikit-learn`, `numpy`, `pandas`.

`assert_comparable()` gates aggregation and is wired into `scripts/make_figures.py`, which
pools cells into one picture and is therefore where an incomparable pool would do the most
damage. Two gates, deliberately different:

- **`COMPARABILITY_KEYS`** (dataset, split) — required to pool anything at all.
- **`REPLICATE_KEYS`** (+ feature schema) — required among seeds of one arm.

Feature schema is *not* in the pooling gate: an ablation arm differs from its comparator in
exactly that field, and a gate that forbade it would forbid the ablation table. A run with no
provenance block is reported as `<missing>` rather than treated as matching, because a
pre-provenance artifact is precisely the case that needs flagging. 18 tests.

**Resolved 2026-09-07 — no backfill was needed.** The post-seed-fix re-runs (`b9a47c3`,
`472d20b`) wrote provenance natively, and the 12 pre-provenance ablation summaries were
superseded by the re-run rather than annotated. All 28 summaries now carry a block, and they
agree: one `dataset_sha256` (`78eb5d60860c…`) and one `split_definition` (`6f87ecd06877ff66`)
across every cell, so `COMPARABILITY_KEYS` passes for the whole set.
`python scripts/make_figures.py --strict_provenance` runs clean — 28 cells, 16 arms.

Two commits appear across the set, both carrying the seeding fix: the 16 grid cells ran at
`aec4333`, the 12 ablations at `a3acf30` (a descendant). Every summary records `dirty: true`;
what was uncommitted on the CUDA box at run time is not on the record, and that is the one
identity claim these artifacts still cannot make.

**Superseded — the backfill route, kept for the record.** The 28 pre-fix summaries predated the block.
`scripts/backfill_provenance.py` annotates them from artifacts that still exist: it measures
the dataset hash and reads each split from the run's own predictions CSV, reconstructs the
feature schema (flagged `feature_schema_exact: false` where reconstructed rather than read
from a recorded `prior_columns`), and records library versions as `null` rather than filling
in today's, which would be a fabrication.

```powershell
.\.venv\Scripts\python.exe scripts\backfill_provenance.py data\processed\stage2b_grid --git_commit c84aa26 --dry-run
.\.venv\Scripts\python.exe scripts\backfill_provenance.py data\processed\stage2b_grid --git_commit c84aa26
```

Then `python scripts/make_figures.py --strict_provenance` must pass on the dev box, which is
the check that the whole set is comparable. Note the git commit differs by artifact age: the
16 grid cells are `c84aa26`, the ablations `a4ed0bb`; run the script once per group if that
distinction is worth preserving in the record.

---

## 8. Label-source sensitivity ablation (item 9) not run, and confounded by design

**Manuscript location.** §3.6 Table 4 row 9.

**Problem.** The pre-registered ablation compares ClinVar-only, ClinVar + ProteinGym clinical,
and DMS-proxy training. The DMS pool is 16,420 variants **all from one assay on *MSH2***. Under
leave-one-gene-out it contributes nothing when *MSH2* is held out, and single-gene,
single-assay signal for the other three folds. The comparison is therefore confounded with
gene identity and cannot be interpreted as a label-source effect on this panel.

**Required.** Either run it on the broad 79-gene panel (item 4), where DMS labels span many
genes, or run it here and report it explicitly as a single-gene result. `prepare_split` in
`scripts/finetune_esm_mmr.py` currently hard-filters `label_source` to
`{clinvar, pg_clinical}`; a flag is needed to vary this.

---

## 9. No error analysis — STALE, see item 14

**Reopened 2026-09-15.** Reads the grid+ablation predictions, which are now a
mixed-dataset set (16 cells on `79b68399…`, 12 on `78eb5d60860c…`). Re-run after item 3.

**Manuscript location.** §3.7 (was a placeholder).

**Done 2026-09-07 (against the superseded dataset).** `scripts/error_analysis.py` intersects the errors across the seeds of one
arm, scores each seed at the threshold its own fold selected, joins the master table back on,
and tests every covariate — reporting model inputs and independent covariates **apart**, since
an association with a feature the model reads restates the model's own weighting rather than
telling us anything about it.

    python scripts/error_analysis.py
    python scripts/error_analysis.py --arm esmpri_concat_full_pllr-residual

**What it found, and it is a label-quality result rather than a model result.**

| arm | consensus errors / 683 | FP | FN | seed-unstable |
|---|---|---|---|---|
| `esmpri_concat_frozen_pllr-residual` | 85 (12.4%) | 56 | 29 | 193 |
| `esmpri_concat_full_pllr-residual` | 97 (14.2%) | 73 | 24 | 113 |

- **Every false positive of the frozen arm is *MSH2*** (56/56); the full fine-tune adds 5
  elsewhere out of 73. *MLH1* and *MSH6* produce none.
- The two covariates separating errors from correct calls are `label_source` and
  `evidence_tier` (BH-corrected p = 1.0e-4 frozen, 1e-6 full), and **neither is a model
  input**. 45 of the 56 frozen-arm false positives are ProteinGym-clinical benign calls; 44
  carry `evidence_tier == unreviewed`.
- This explains the *MSH2* MCC anomaly first flagged on 2026-08-28 (ROC-AUC 0.878 against MCC
  0.297). *MSH2* contributes 144 of the 180 PG-clinical benign labels in the whole panel —
  *MLH1* contributes 6 — so *MSH2*'s benign class is largely unreviewed calls that the model
  scores pathogenic at >0.999. See Figure 2(c).
- 193 of 683 variants flip between the three seeds of one arm without being consensus errors.
  That is the item-12 initialisation spread measured per variant, and it belongs next to any
  per-variant claim.

**What this does not license.** The association is with label provenance, not with biology. No
claim that these variants are misannotated belongs in the paper without going back to the
ClinVar submissions themselves. Artifacts:
`data/processed/stage2b_grid/error_analysis_<arm>_{errors,covariates}.csv`.

---

## 10. Figures — 3 of 6 GENERATED, now STALE, see item 14

**Reopened 2026-09-15.** Same mixed-dataset problem as items 3 and 9 — regenerate after
the ablations are re-run against `79b68399…`.

**Done 2026-09-05 (against the superseded dataset).** `scripts/make_figures.py` generates Figures 3, 5 and 6 into
`docs/figures/` as 300-dpi PNG and vector PDF, reading the committed results CSVs directly —
no intermediate spreadsheet, so a figure cannot drift from the table it illustrates. It globs
the cell artifacts, so re-running it after further seeds land refreshes all three unedited.
Series carry hue *and* marker shape; the palette validates at all pairs for colour-vision
deficiency (worst CVD dE 9.2, normal-vision dE 24.0) and every series is direct-labelled,
which is what the below-3:1 contrast of the third hue requires.

| Fig | Caption | Status |
|---|---|---|
| 1 | Dataset assembly and evaluation pipeline. | **Outstanding** — schematic; draw, do not plot. Output `docs/figures/fig1_pipeline.svg` |
| 2 | Dataset composition, source overlap and label provenance. | **Outstanding** — needs an UpSet or stacked-bar plot; no script exists |
| 3 | Main model comparison under LOPO with bootstrap CIs. | Generated — `fig3_main.png` / `.pdf` |
| 4 | Reliability diagram and confusion matrix. | **Outstanding** — `src.calibration.plot_reliability_diagrams` exists but no driver calls it on grid outputs; blocked behind item 5 anyway, since the reported runs have no valpreds |
| 5 | Ablation results across all cells. | Generated — `fig5_ablation.png` / `.pdf` |
| 6 | Per-gene performance with cohort sizes. | Generated — `fig6_pergene.png` / `.pdf` |

Figure 4 is the one worth sequencing deliberately: it needs inner-validation predictions,
which the reported grid cells do not have (item 5), so it should follow the tier-1 re-run
rather than being attempted against the current artifacts.

---

## 12. MLH1 is unseeded in every grid cell run so far — RESOLVED

**Found 2026-09-06 by the reproducibility check the previous item asked for.** Two runs of
one cell at one commit (`data/processed/repro_check/`, `repro_a` / `repro_b`) disagree on
MLH1 — ROC-AUC 0.9485 vs 0.9276, every one of the 208 held-out and 95 inner-validation
probabilities different, max difference 0.94 — while MSH2, MSH6 and PMS2 come back
bit-identical. `run_one_split` constructed the model, and so initialised the head, before
anything seeded the RNG, so the first split of each process drew its head from process-start
entropy. `MMR_GENES` is ordered `(MLH1, MSH2, MSH6, PMS2)`; the first split is always MLH1.

**Fixed** in `scripts/finetune_esm_mmr.py` (and the same defect in
`scripts/compare_finetune_strategies.py`) by seeding per split before construction, with a
regression test that fails on the old code. See `docs/RUNLOG.md` 2026-09-06.

**What the paper must not claim until a re-run.** Initialisation alone moves MLH1 by 0.021
AUROC. The three-seed arms show 0.019-0.049 spread on MLH1, so most of that is initialisation,
not seed. **No MLH1 comparison below ~0.02 AUROC is interpretable**, which includes any Table 4
or Table 6 row whose MLH1 margin is that small. MSH2/MSH6/PMS2 as measurements are unaffected.

**Re-run completed 2026-09-07 — all 28 cells, nothing mixed.** The 16 grid cells came back in
`b9a47c3` and the 12 feature-family ablations in `472d20b`. As predicted, the fix moved all
four genes rather than MLH1 alone: 58 of 64 grid rows changed, mean |dAUROC| 0.011 (MLH1),
0.006 (MSH2), 0.014 (MSH6), 0.022 (PMS2). Every number in the paper now comes from one
initialisation regime, and `--strict_provenance` confirms one dataset and one split across
the set.

**The pre-registered reading survives the re-run**, which is the result that mattered. At
`pllr=residual`, seed 42, mean AUROC over the scoreable genes:

| branch | frozen | last2 | full |
|---|---|---|---|
| `esm+priors` | **0.945** | 0.936 | 0.941 |
| `esm` only | 0.904 | 0.932 | 0.927 |

Backbone gradients buy nothing once the priors are present; they only recover ground the
esm-only branch was missing. The 6.5-point gap §6.12 was built around was the feature set,
not the freeze depth.

**What the caveat becomes.** The MLH1 warning is retired for these artifacts — they are
replayable. It still binds anything quoted from a pre-2026-09-07 run, and the seed spread it
exposed is real and unchanged: see item 9 for the 193-of-683 per-variant flip rate.

---

## 13. The ablation artifacts were deleted by a push, and nearly went unnoticed

**Recorded 2026-09-07, resolved the same day.** The grid re-run push (`b9a47c3`) staged 48
deletions alongside its additions — every `ablate_*` results, predictions, summary and
valpreds file, the artifacts behind Table 4 and the ablation series of Figure 5. The CUDA
box's working tree simply did not have them, and a broad `git add` recorded that absence as
an intended removal.

Nothing failed. `make_figures.py` regenerated Figure 5 without complaint, legend still
advertising a "feature-family ablation" series with no points in it.

**Two guards added.** `figure5` now prints a WARNING naming the missing glob when no ablation
arm is present, and drops the legend entry rather than captioning an empty series. The push
procedure in `docs/GPU_RUN_SHEET_2026-09-07.md` uses path-scoped `git add` and requires
reading `git status` for `deleted:` lines before committing.

**The general lesson, which the provenance work had not covered.** `src/provenance.py` gates
whether artifacts that *are* present may be pooled. It says nothing about artifacts that have
gone missing, because a figure built from a subset is still internally consistent. Absence
needs its own check.

---

## 11. Smaller items

- **21-row label discrepancy (§3.2).** Manifest `master_label_counts` totals 17,124; the
  `label_source` tally totals 17,103. The difference matches the 21 homology-gated *PMS2*
  rows, but this must be confirmed, not assumed.
- **ClinVar archive date (Table 1).** The manifest stores the URL and SHA-256 but no download
  date or release version. Record the release used; ClinVar changes weekly. The clinical
  counts (208/335/119/21 = 683) were identical to the previous build, which suggests the
  snapshot did not refresh between builds — worth confirming.
- **ClinVar raw record count (Table 1).** Not recorded in the manifest; add it to the source
  block.
- **MaveDB and CIMRA (Table 1).** Adapters exist and are tested, but the build manifest has no
  source block for either, indicating they were not enabled. Confirm and state this explicitly
  rather than omitting the rows.
- **The 27 prior column names (§2.4).** ~~Reconstructed from source rather than recorded.~~
  Resolved 2026-09-04: `prior_columns` is written into every `esm_finetune_summary_*.json`.
- **PLLR-mode and fusion axes (§3.6).** Run at a single seed each. The three-seed axis shows
  MCC SD up to 0.056, so these single-seed differences are not interpretable. Either add seeds
  or continue to report them without interpretation.
- **References (§8).** No citation in the repository carries author lists, titles, years or
  DOIs. Every entry needs completion from the primary literature; none should be guessed.
- **Ethics/funding/conflicts/contributions (§7).** Standard placeholders, to be completed by
  the authors.
- **Checkpoint distribution (§6).** Grid cells ran with `--no-save_checkpoints`, so no backbone
  weights exist. Decide whether the small baseline heads are distributable.
