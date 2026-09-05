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

**Outstanding, minor.** `mmr_transfer_predictions_lopo.csv` from the re-run is gitignored and
has not been committed, so the threshold-dependent columns of Table 3 (accuracy, F1, balanced
accuracy, precision, recall, specificity, Brier, ECE) are still `[TODO]` for that row. The
file carries both `holdout` and `inner_val` rows, so the panel can be computed leak-free with
`src.metrics.evaluation_report` once it is pushed:

```powershell
git add -f data\processed\mmr_transfer_scratch\mmr_transfer_predictions_lopo.csv
```

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

| Row | Cell slug | Prior cols | AUROC (3 scoreable) | Δ vs comparator 0.9341 |
|---|---|---|---|---|
| 4 | `ablate_structure` | 25 | 0.9411 | +0.0070 |
| 6 | `ablate_domains` | 24 | 0.9429 | +0.0088 |
| 7 | `ablate_prior_scores` | 9 | 0.8937 | −0.0404 |
| 5 | `ablate_gnomad_and_scores` | 5 | 0.8704 | −0.0637 |

**Reading.** The three-seed comparator has AUROC SD 0.0097 over the scoreable genes, so a
difference between two single-seed runs is expected to scatter by about 0.014. Rows 4 and 6
are inside that and point upward: structural and domain features contribute nothing
detectable. The 18 external prior-score columns carry essentially the whole prior
contribution; allele frequency adds a further 0.023 on top (row 5 versus row 7), above the
noise floor but resting on two single-seed runs.

**Residual weaknesses, stated rather than fixed.** Each ablation is a single seed against a
three-seed baseline whose seed-42 draw is the highest of its three (0.9341 / 0.9149 /
0.9270), so deltas measured from it are slightly generous; against the three-seed mean of
0.9253 they are +0.016, +0.018, −0.032 and −0.055. Adding seeds 43 and 44 to each ablation
(~11 min per run, 8 runs) would let rows 4–7 be reported as mean ± SD like the rest of the
paper. Row 5 bounds gnomAD and the external scores jointly by design — the proxy guard
forbids removing gnomAD alone — so it is not a gnomAD-only effect and the table says so.

**Also outstanding.** `esm_finetune_summary_*.json` does not record `n_bootstrap`, although
the bootstrap ran and the CIs are in the results CSVs. The manuscript's protocol section
claims 10,000 resamples; that claim is not checkable against the artifact until the field is
added.

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

## 10. All six figures missing

No `.png`, `.pdf` or `.svg` exists anywhere in the repository.

| Fig | Caption | Input artifact | How to generate | Output |
|---|---|---|---|---|
| 1 | Dataset assembly and evaluation pipeline: sources → wild-type validation → variant keys → evidence columns → conflict screening → master table → features → LOPO splits → models → evaluation. | `manifest.json`, `audit_report.json` | Schematic — draw, do not plot | `docs/figures/fig1_pipeline.svg` |
| 2 | Dataset composition, source overlap and label provenance for the 74,328-variant panel. | `manifest.json → stats.multi_source_overlaps`, measured `label_source` counts | **No script exists.** Needs an UpSet or stacked-bar plot | `docs/figures/fig2_composition.png` |
| 3 | Main model comparison under leave-one-gene-out, with bootstrap CIs. | `stage2b_grid_results.csv` | **No script exists** | `docs/figures/fig3_main.png` |
| 4 | Reliability diagram and confusion matrix at 0.5 and at the validation-selected threshold. | `esm_finetune_predictions_*.csv` | `src.calibration.plot_reliability_diagrams` exists but **no driver calls it on grid outputs** | `docs/figures/fig4_calibration.png` |
| 5 | Ablation results across the 16 grid cells. | `journal_table_stage2b.csv` | **No script exists** | `docs/figures/fig5_ablation.png` |
| 6 | Per-gene performance with cohort sizes, showing *PMS2* as unscoreable. | `full_metric_panel_by_gene.csv` | **No script exists** | `docs/figures/fig6_pergene.png` |

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
