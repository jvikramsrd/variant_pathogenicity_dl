# Run sheet — 2026-09-07, closing the paper's outstanding data

Follow-up to the stage-2b grid re-run (`b9a47c3`). One GPU job, three CPU jobs,
one push. Total: **~2.2 GPU-hours + ~15 CPU-minutes.**

Everything below runs on the Windows CUDA box from the repo root, on branch
`results/stage2b-grid`. Linux GPU box: substitute `.venv/bin/python`, forward
slashes and `\` line continuations.

---

## 0. Get the code

```powershell
git pull
```

You should land on the commit that adds `scripts/error_analysis.py`, the
per-variant export in `scripts/recalibrate_grid.py`, and Figures 2 and 4 in
`scripts/make_figures.py`.

---

## 1. GPU — re-run the 12 feature-family ablation cells (~2.2 h)

**Why this is needed.** The re-run push deleted all 12 ablation cells from the
branch — 48 files. They are recoverable from `aec4333`, but they were computed
*before* the per-split seeding fix, and `MISSING_EVIDENCE.md` item 12 forbids a
table that mixes pre- and post-fix cells: they are different initialisation
regimes. Table 4 and the ablation series of Figure 5 currently have no artifacts
behind them.

All 12 are frozen probes (`--n_unfrozen_layers 0`), ~165 s per held-out gene, so
this is cheap. Parameters below are read off the deleted summaries at `aec4333`,
so the arms are identical to the ones Table 4 was written against — only the
seeding differs, which is the point.

```powershell
$groups = [ordered]@{
  "ablate_domains"           = @("domains")
  "ablate_structure"         = @("structure")
  "ablate_prior_scores"      = @("prior_scores")
  "ablate_gnomad_and_scores" = @("gnomad", "prior_scores")
}

foreach ($seed in 42, 43, 44) {
  foreach ($cell in $groups.Keys) {
    Write-Host "=== $cell seed $seed ===" -ForegroundColor Cyan
    .\.venv\Scripts\python.exe scripts\finetune_esm_mmr.py `
      --mode siamese --eval lopo `
      --esm_model facebook/esm2_t33_650M_UR50D `
      --branch esm+priors --fusion concat --pllr_mode residual `
      --n_unfrozen_layers 0 `
      --drop_prior_groups $groups[$cell] `
      --seed $seed `
      --cell_slug "$($cell)_seed$seed" `
      --batch_size 1 --grad_accum 8 `
      --epochs 10 --patience 3 --n_bootstrap 10000 `
      --out_dir data\processed\stage2b_grid
  }
}
```

**Two deliberate differences from the deleted runs.**

- `--cell_slug` carries `_seed42` for every seed. The old seed-42 cells were
  written without a seed suffix while 43 and 44 carried one, so the set did not
  sort or glob uniformly. `arm_of()` strips `_seed\d+$`, so the arm name is
  unchanged and Figure 5 groups them exactly as before.
- `--n_bootstrap 10000` is passed explicitly. The deleted summaries recorded
  `n_bootstrap: null`, so what they actually used is not on the record.

**Expected on completion:** 48 files back in `data\processed\stage2b_grid\` —
12 each of `esm_finetune_{results,predictions,summary,valpreds}_siamese_lopo_ablate_*`.
Each summary should carry a `provenance` block whose `dataset_sha256` starts
`78eb5d60860c`, matching the 16 grid cells. If it does not, stop: the ablations
were computed on a different table than the grid, and nothing can be pooled.

---

## 2. CPU — calibration panel (~10 min)

Closes `MISSING_EVIDENCE.md` item 5. Every reported arm carries ECE 0.19–0.21
and none has been calibrated; the re-run finally wrote inner-validation
predictions for all 16 cells, so there is now a leak-free split to fit on.

Run this **after** step 1 so the ablation cells are included.

```powershell
.\.venv\Scripts\python.exe scripts\recalibrate_grid.py --grid_dir data\processed\stage2b_grid
```

Writes `calibration_panel.csv` (metrics per cell × gene × method) and
`calibration_probs.csv` (per-variant calibrated held-out probabilities, which
Figure 4 reads). The run is slow for its size — roughly 3 s per fold-method,
almost all of it the torch optimiser inside `TemperatureScaling.fit` — so about
10 minutes for 28 cells. It prints mean ECE per method at the end; expect
`temperature` and `isotonic` well below `uncalibrated`.

> **Read isotonic with suspicion.** It is fitted and then threshold-selected on
> the same ~95-variant inner-validation set, so its validation calibration is
> optimistic by construction. Temperature scaling has one parameter and does not
> have that problem. The script's docstring says so; the paper should too.

---

## 3. CPU — error analysis (seconds)

Closes item 9. §3.7 is a placeholder because nothing had ever joined a
prediction back to the variant it was made about.

```powershell
.\.venv\Scripts\python.exe scripts\error_analysis.py
```

Verified on the dev box against the current artifacts. What it found there, for
the headline arm `esmpri_concat_frozen_pllr-residual` over 3 seeds:

| | |
|---|---|
| consensus errors | 85 of 683 held-out variants (12.4%) |
| false positives | 56 — **every one of them MSH2** |
| false negatives | 29, spread across MLH1/MSH2/MSH6 |
| wrong in some but not all seeds | 193 — not counted as errors |

The two covariates that separate errors from correct calls are `label_source`
and `evidence_tier` (BH-corrected p = 1.0e-4), and **neither is a model input**.
45 of the 56 false positives are ProteinGym-clinical benign calls, 44 of them
`evidence_tier == unreviewed`. Re-run it on the GPU box so the numbers in the
paper come from the same machine as the rest; they should reproduce exactly,
since this reads committed artifacts and trains nothing.

Add `--arm esmpri_concat_full_pllr-residual` for the full fine-tune arm if §3.7
should compare the two.

---

## 4. CPU — figures (seconds)

```powershell
.\.venv\Scripts\python.exe scripts\make_figures.py --strict_provenance
```

`--strict_provenance` now **passes** — all 16 grid summaries carry native
provenance blocks, so the backfill that item 7 asked for is not needed. After
step 1 it should report 28 cells comparable rather than 16.

Produces Figures 2–6 into `docs\figures\`:

- **Figure 2 (new)** — dataset composition. Verified rendering on the dev box.
- **Figure 4 (new)** — reliability diagram + confusion matrix. **This one has
  never rendered**: it needs `calibration_probs.csv` from step 2, which did not
  finish locally. If it throws, send me the traceback rather than working around
  it. `make_figures.py` skips it with a printed note when its inputs are absent,
  so a failure here will not cost you Figures 2, 3, 5 and 6.
- **Figure 5** now prints a `WARNING` if no ablation arms are present. After
  step 1 that warning should be gone. If you see it, step 1 did not land.

Figure 1 is still outstanding and is a schematic — drawn, not plotted.

---

## 5. Push it back

```powershell
git add data\processed\stage2b_grid\esm_finetune_results_*.csv `
        data\processed\stage2b_grid\esm_finetune_summary_*.json
git add -f data\processed\stage2b_grid\esm_finetune_predictions_*.csv `
           data\processed\stage2b_grid\esm_finetune_valpreds_*.csv `
           data\processed\stage2b_grid\calibration_panel.csv `
           data\processed\stage2b_grid\calibration_probs.csv `
           data\processed\stage2b_grid\error_analysis_*.csv
git add docs\figures\

git status     # expect 48 ablation files restored, nothing else deleted
git commit -m "results: ablation arms re-run post-seed-fix, calibration and error analysis"
git push
```

**Check `git status` before committing.** The last push deleted 48 files without
anyone noticing until they were already gone. If this one shows deletions,
something in `data\processed\stage2b_grid\` is missing on your box that exists on
the branch — do not commit until you know which.

---

## What this leaves open

| Item | Needs |
|---|---|
| 4 — no model on the broad 79-gene panel | a GPU run, not yet scoped |
| 6 — random-split diagnostic | a code change: `--eval` accepts only `lopo` and `holdout` |
| 8 — label-source ablation | a flag to vary the `label_source` filter in `prepare_split`, plus a run; confounded on this panel regardless, since the DMS pool is one MSH2 assay |
| 10 — Figure 1 | a schematic to draw |
| 12 — the `git.dirty: true` flag | all 16 grid summaries record an uncommitted working tree at run time; run `git status` on the GPU box and tell me what was modified |
