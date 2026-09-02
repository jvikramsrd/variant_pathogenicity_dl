# GPU run playbook — Stage-2b ablation grid

Operator's document for the CUDA box (~15 GiB card). The dev box is CPU-only,
so everything here runs on the *other* machine and the artefacts come back as
files.

Design rationale: `docs/superpowers/specs/2026-09-02-accuracy-ablation-paper-design.md`.
Grid definition in code: `src/finetune_grid.py`.

---

## 0. Why this grid, in one paragraph

`docs/PAPER_DRAFT.md` §6.12 proposes a freeze-depth × PLLR ablation and a
decision rule: if the frozen floor matches the full fine-tune, backbone
gradients bought nothing. That comparison could not have settled the question,
because Stage 2b read **none** of the 27 prior columns the frozen probe reads —
so the measured 6.5-point gap (0.880 vs 0.945) confounded freeze depth with
feature set. This grid adds a **branch** axis (`esm` vs `esm+priors`) so freeze
depth is measured with the feature set held constant.

---

## 1. One-time setup

```bash
git clone <repo> && cd variant_pathogenicity_dl
git checkout feat/stage2b-fair-comparison

python -m venv .venv
.venv/bin/pip install -r requirements-cuda.txt

# Must print True. If it prints False the grid will run on CPU and take weeks.
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Build the MMR table **with PMS2 included** — the codon range is derived and
self-validating (`scripts/derive_pms2_homology_range.py`), but the gate is
fail-closed, so it must be passed explicitly:

```bash
.venv/bin/python scripts/build_mmr_dataset.py --min_stars 2 --pms2_codon_range 382 862
```

Expect ~74,328 rows, of which 683 carry a clinical label
(MLH1 208 / MSH2 335 / MSH6 119 / PMS2 21).

Confirm the tests pass on this machine before spending GPU hours:

```bash
.venv/bin/python -m pytest tests/ -q      # expect 146 passed
```

---

## 2. Run the grid

One command. It is **resumable** — re-running skips cells that already have all
three artefacts, so an interrupted run is continued by repeating the identical
command, not restarted.

```bash
.venv/bin/python scripts/run_stage2b_grid.py \
  --tiers 1 2 3 4 5 \
  --esm_model facebook/esm2_t33_650M_UR50D \
  --mode siamese --eval lopo \
  --batch_size 1 --grad_accum 8 --gradient_checkpointing \
  --epochs 10 --n_bootstrap 10000 \
  --out_dir data/processed/stage2b_grid
```

`--batch_size 1 --grad_accum 8 --gradient_checkpointing` is what makes a 650M
siamese full fine-tune fit ~10 GiB on a 15 GiB card while keeping ProPath's
effective batch of 8. Below ~11 GiB a full fine-tune is impossible at all: the
static AdamW state alone is ~9.7 GiB.

Add `--dry_run` first to print the plan and every per-cell command line without
running anything.

### Expected wall-clock

| Tier | Cells | What it answers | Est. |
|---|---|---|---|
| 1 | 6 | **The headline.** Does ESM add anything on top of published priors, on unseen genes? 3 seeds × {full FT, frozen floor}, both `esm+priors` | ~15.75 h |
| 2 | 2 | Branch attribution: how much of any gain is the priors branch vs the backbone? | ~5.25 h |
| 3 | 5 | The PLLR axis, measured at the frozen floor where the backbone cannot relearn the term. Includes the GateWave-fusion cell | ~1 h |
| 4 | 2 | Freeze-depth middle ground (`--n_unfrozen_layers 2`) | ~4 h |
| 5 | 1 | PLLR off at full fine-tune | ~5 h |
| | **16** | | **~31 h** |

Tiers are ordered by scientific value, so an interruption degrades gracefully.
**Tier 1 alone settles the paper's headline claim.** If time runs short, stop
after any tier boundary rather than part-way through tier 1's seeds.

Frozen (`n_unfrozen_layers 0`) cells are cheap because the backbone output is
constant across epochs and is encoded once — that is why every floor cell can
carry three seeds.

---

## 3. If a cell fails

The driver logs the failure and continues to the next cell; one OOM does not
cost the remaining tiers. At the end it prints the failed slugs. To retry, run
the identical command — completed cells are skipped.

If a full-unfreeze cell OOMs repeatedly, the knobs in order of preference:

1. `--grad_accum 16` with `--batch_size 1` (halves nothing, but shortens the
   optimizer's live state peak in some drivers)
2. Drop the cell's `--epochs` — early stopping usually beats the cap anyway
3. Skip tier 5 and the full-unfreeze cell of tier 2; they are the least
   load-bearing of the full-unfreeze cells

Do **not** raise `--batch_size` above 1 for a 650M full unfreeze on 15 GiB.

---

## 4. Bring these files back

The whole output directory:

```
data/processed/stage2b_grid/
├── esm_finetune_results_siamese_lopo_<cell>.csv       # per-gene metrics, 1 per cell
├── esm_finetune_predictions_siamese_lopo_<cell>.csv   # per-VARIANT probabilities
├── esm_finetune_summary_siamese_lopo_<cell>.json      # config + provenance
├── stage2b_grid_results.csv                           # all cells, concatenated
└── stage2b_grid_manifest.json                         # what ran, what failed, timings
```

**The predictions CSVs are the important ones.** Metrics alone cannot be
seed-ensembled, recalibrated, or analysed per-gene after the fact; the previous
Stage-2b runs returned only metrics, which is why Run 1 and Run 2 could not be
compared at the variant level.

Also worth capturing, since §6.10 flags it as unverified provenance: the exact
`nvidia-smi` output and the `requirements-cuda.txt` freeze from this box.

---

## 5. Pre-registered reading of the results

Fixed **before** the numbers land, and not revised afterwards.

- If the **frozen floor** (`n_unfrozen_layers 0`) mean sits inside the 95% CI of
  the **full fine-tune** at the same branch, backbone gradients bought nothing
  measurable at this label budget (277–530 training rows per split). Report the
  frozen probe as the model and the fine-tune as a negative result.
- If **`esm+priors` does not beat `priors`-only** (the frozen Stage-2 probe,
  mean ROC-AUC 0.945 / MCC 0.687 on MLH1/MSH2/MSH6), then ESM adds nothing on
  top of published priors for unseen-gene MMR prediction. That is a publishable
  negative result and the current best guess.
- If **`n_unfrozen_layers 2` matches `-1`** at a fraction of the compute,
  recommend the partial unfreeze and quote the VRAM saving.
- The **PLLR-on minus PLLR-off gap at the frozen floor** is the clean
  measurement of what the 2026-08-28 PLLR fix is worth. At full unfreeze the
  same gap is confounded with the backbone relearning the term.
- Holdouts are small (119–335; PMS2 21). **If the seed sd exceeds the
  between-cell differences, the ablation is underpowered — say so and do not
  rank cells.**
- PMS2's 21 held-out rows stay out of every headline mean.

---

## 6. What is *not* in this grid, and why

**Data expansion** (gnomAD-AF benign labels, 1-star ClinVar, the `--all_sources`
broad-panel rebuild) deliberately runs *after* the grid, as a single before/after
delta on the winning cell. Changing the data, the model, and the training recipe
at once and reporting the aggregate is precisely the practice this paper
criticises.

When that run happens, `--af_labels_active` is **mandatory** if AF-derived labels
are in the pool. It forces the four AF-derived feature columns
(`gnomad_log10_af`, `acmg_ba1`, `acmg_bs1`, `acmg_pm2`) out of the feature set.
Without it, `acmg_bs1 == label` by construction on every minted row — the same
target leakage as Finding 2, inside the paper that reports Finding 2. The guard
is enforced in code (`src.transfer.assert_af_quarantine`) and will refuse to run.

**CPU-only experiments** (leave-one-protein-out on the 80-gene panel, seed
averaging, the temporal ClinVar holdout) run on the dev box and consume none of
this budget.
