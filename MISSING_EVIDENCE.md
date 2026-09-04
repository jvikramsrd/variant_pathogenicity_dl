# Missing evidence for `docs/MANUSCRIPT.md`

Every placeholder in the manuscript, what it needs, and the exact command or artifact
that would replace it. Ordered by how much each blocks a claim the paper wants to make.

Dataset identity for all current model results: `extended_dataset.csv` SHA-256
`78EB5D60860CC08EADB6FAA0C0D7FBD22ADBE6F277BB95E97A3A680264B4430D`, code `c84aa26`.
Anything re-run for comparison must read that table, or the comparison is invalid.

---

## 1. Priors-only baseline on the current dataset build — BLOCKS THE HEADLINE CLAIM

**Manuscript location.** §3.3 Table 3 (last row), §3.6 Table 4 item 1, §5.

**Problem.** A curated-priors-only head was run under the same LOPO protocol and reached
mean AUROC 0.951 over the three scoreable genes — higher than any ESM-2 cell in the grid.
That run (`data/processed/mmr_transfer_scratch/mmr_transfer_summary_lopo.json`) is dated
`2026-09-02T15:22:43Z` and used the **previous** dataset build, before gnomAD features were
joined. The grid used the 2026-09-03 build. Comparing across dataset versions is exactly
what the protocol forbids, so the single most interesting comparison in the paper — protein
language model versus curated features — currently cannot be made.

**Command.**
```bash
python scripts/run_mmr_transfer.py --scratch --features priors --eval lopo \
  --n_bootstrap 10000 --out_dir data/processed/mmr_transfer_scratch
```
**Runtime.** 98 s on the previous build. **Output.** `mmr_transfer_{results,summary,predictions}_lopo.*`.
**Check.** `gene_constant_priors_dropped: true` and `n_bootstrap: 10000` in the summary.

---

## 2. Manifest does not describe the table it accompanies

**Manuscript location.** §2.1 (discrepancy note), Table 1.

**Problem.** `data/mmr/processed/extended/manifest.json` records
`parameters.include_gnomad: false`, `sources.gnomad.enabled: false` and
`stats.gnomad_rows_panel: 0`, but the table carries a joint allele frequency for 6,492
variants and gene constraint for all 74,328. `scripts/build_mmr_dataset.py` joins gnomAD in
its stages 3–4, after `src/extended_builder.py` has written the manifest, and never rewrites
it. The data are correct; the provenance record is wrong, which is a reproducibility defect
in a paper whose contribution is provenance.

**Fix.** Code change in `scripts/build_mmr_dataset.py`: update and rewrite the manifest after
stages 3 and 4, recording `gnomad_rows_panel`, `gnomad_genes_fetched`,
`gnomad_constraint_genes_fetched` and the effective `include_gnomad`. Then either rebuild or
patch the existing manifest in place, and note which was done.

---

## 3. Feature-group ablations 4–7 not run

**Manuscript location.** §3.6 Table 4 rows 4–7; §4 (the paragraph on what priors contribute).

**Problem.** The grid varies branch, freeze depth, PLLR mode and fusion. It never removes an
individual feature family, so the paper cannot say which of the 27 prior columns carry the
+0.031 AUROC.

**Required.** A source/feature inclusion switch at the prior-block level. `src/transfer.py`
defines the prior column groups (`TRANSFER_PRIOR_COLS`, `GENE_CONSTANT_PRIOR_COLS`,
`AF_DERIVED_PRIOR_COLS`), but `scripts/finetune_esm_mmr.py` exposes no flag to drop a named
group. **A CLI flag must be added before these ablations can run** — for example
`--drop_prior_groups structure,gnomad,domains,prior_scores`.

**Design requirement, not optional.** Each ablation must remove derived proxies as well as the
named group. In particular ablation 5 ("without gnomAD") is invalid while AlphaMissense
remains in the feature set, because AlphaMissense was trained on population data and carries
that signal. State in the table which proxies were removed with each group.

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
- **The 27 prior column names (§2.4).** Reconstructed from source rather than recorded. Emit
  the resolved list into `esm_finetune_summary_*.json`.
- **PLLR-mode and fusion axes (§3.6).** Run at a single seed each. The three-seed axis shows
  MCC SD up to 0.056, so these single-seed differences are not interpretable. Either add seeds
  or continue to report them without interpretation.
- **References (§8).** No citation in the repository carries author lists, titles, years or
  DOIs. Every entry needs completion from the primary literature; none should be guessed.
- **Ethics/funding/conflicts/contributions (§7).** Standard placeholders, to be completed by
  the authors.
- **Checkpoint distribution (§6).** Grid cells ran with `--no-save_checkpoints`, so no backbone
  weights exist. Decide whether the small baseline heads are distributable.
