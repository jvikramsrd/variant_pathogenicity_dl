# Missing evidence for `docs/MANUSCRIPT.md`

Every placeholder in the manuscript, what it needs, and the exact command or artifact
that would replace it. Ordered by how much each blocks a claim the paper wants to make.

Dataset identity for all current model results: `extended_dataset.csv` SHA-256
`78EB5D60860CC08EADB6FAA0C0D7FBD22ADBE6F277BB95E97A3A680264B4430D`, code `c84aa26`.
Anything re-run for comparison must read that table, or the comparison is invalid.

---

## 1. Priors-only baseline on the current dataset build — RESOLVED

**Resolved 2026-09-05.** Re-run on the build machine against the pinned table
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

## 3. Feature-group ablations 4–7 — RESOLVED

**Resolved 2026-09-05.** All four ran on the build machine against the pinned table, at the
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

## 7. Run artifacts carry no dataset, feature-schema or split identity

**Manuscript location.** §2.5 (tracking), and implicitly every comparison in the paper.

**Problem.** `esm_finetune_summary_*.json` records cell slug, branch, fusion, PLLR mode, seed,
model name, freeze depth, split names, checkpoints and runtime — but no dataset-manifest hash,
feature-schema hash, split hash, git commit, or library versions. Two runs on different
dataset builds are therefore indistinguishable from their artifacts. Item 1 above is exactly
this failure occurring in practice.

The current grid's identity survives only in two loose text files
(`dataset_sha256.txt`, `git_commit.txt`) captured manually after the fact.

**Required.** A `src/provenance.py` returning dataset SHA-256, sorted-feature-schema hash,
split-definition hash, git commit and `torch`/`transformers`/`scikit-learn`/`numpy` versions;
wired into the per-cell summary and the grid manifest; plus a comparability check that refuses
to aggregate cells whose dataset or schema hashes disagree. A backfill script should annotate
the existing 16 summaries, since the table that produced them is still on disk and hashable.

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

## 9. No error analysis

**Manuscript location.** §3.7 (section is a placeholder).

**Required.** Export high-confidence false positives and false negatives from
`esm_finetune_predictions_*.csv`, join back to the master table for domain membership, pLDDT,
allele frequency, ClinVar review status and label source, and inspect for pattern. No script
does this. Until it exists, no biological interpretation of individual errors belongs in the
manuscript.

---

## 10. Figures — 3 of 6 GENERATED

**Done 2026-09-05.** `scripts/make_figures.py` generates Figures 3, 5 and 6 into
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
