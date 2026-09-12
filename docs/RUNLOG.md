# Running Log

Append one entry per build / training run. Newest at the top.
Format: date · what ran (command) · outcome · artifacts.

---

## 2026-09-12 (read-only dev box, round 2) — fresh-context review; one checkpoint-loading gap closed, inference pipeline added

No training, inference, dataset processing, or benchmark was run (same
environment as the entry below: no working Python install with
torch/pandas/pytest here). Scope was a second fresh-context static review of
this branch, followed by new implementation work for two gaps the review
found still open: an ECC-plugin-driven audit/hardening pass per the
project's standing "inspect -> map -> static review -> implement -> static
test/review -> document -> fresh-context review" loop.

**Found.** `src/esm_finetune.py::save_finetuned` has stamped a `format` tag
(`FINETUNE_CHECKPOINT_FORMAT`) on every checkpoint since it was introduced,
but `load_finetuned_model` never checked it back on load — the identical
defect class already fixed for `src.transfer.load_transfer_head`
(`TransferHeadFormatError`), just not ported to this sibling function. A
foreign or future-format `.pt` file would have been accepted silently and
failed later, deep inside model construction, with an error naming neither
the checkpoint nor its format.

**Fixed.** `src/esm_finetune.py` — added `FinetuneCheckpointFormatError` and
a format check in `load_finetuned_model`. Deliberately *not* a byte-for-byte
mirror of `TransferHeadFormatError`: real legacy checkpoints predating this
field already exist and are migrated elsewhere in the same function (the
`use_pllr` -> `pllr_mode` handling just below), so a missing tag is accepted
(logged as a warning) and only a *present*, *mismatched* tag is rejected.
Rejecting on missing would have silently broken loading of every
already-produced checkpoint that predates this session. Two new tests added
to `tests/test_new_data_sources.py`: a foreign-format rejection, and an
explicit no-format-tag-still-loads regression (so a future edit can't
reintroduce the stricter, breaking check by accident). Syntax-checked with
`python3 -m py_compile`; not run, for the same reason nothing else on this
branch has been.

**Reviewed, no defect found.** `scripts/run_mmr_transfer.py`'s own load path
for the stage-1 pretrain checkpoint (`ckpt.get("config")`'s `esm_dim` /
`feature_columns` check, lines 269-278) already rejects an incompatible
checkpoint via a targeted schema comparison rather than a version-string
tag — adequate for what it guards, and not the same gap as the two format-tag
cases above; left unchanged.

**Added (new capability, not a defect fix).** No inference entry point
existed anywhere in this codebase — every model-scoring codepath was fused
into a train-and-evaluate script. Added `src/inference.py` +
`scripts/predict.py`, scoped deliberately to `arch="priors"` stage-2
transfer-head checkpoints only (a single named-column feature view; scoring
an `esm`/`concat`/`gatewave` checkpoint needs an ESM-2 backbone forward pass,
which this pass does not implement — `UnsupportedArchitectureError` names
the gap rather than guessing at it). Reuses `load_transfer_head`'s existing
format versioning and stored per-view scaler statistics; never fits
anything on the inference input. `tests/test_inference.py` (16 tests,
synthetic-only: hand-built `torch.nn.Module` stub, no real checkpoint) covers
malformed/reordered/missing columns, unknown/unseen genes, an unsupported
architecture, a propagated format error, NaN features, single-row-vs-batch
consistency, an explicit threshold override, and CPU-only device selection
with no CUDA visible. See `docs/INFERENCE.md`. Syntax-checked, not run — no
real `.pt` checkpoint has been loaded through this path in any environment
yet; see that doc's "Remaining risk".

**Added (portable smoke test, not run here).**
`scripts/sanity_check_overfit.py` — a tiny synthetic (numpy-generated,
perfectly-separable) overfit check for `src.transfer.build_model` /
`fit_head`, for the other PC to run before trusting any real-data result
from the same training code path.

---

## 2026-09-12 (read-only dev box) — static review of the uncommitted audit branch; one doc/deliverable gap closed

No training, inference, dataset processing, or benchmark was run (no working
Python environment with pytest/pandas exists on this machine; confirmed by
import failure in both the system interpreter and `.venv`). Scope was a
fresh-context static review of the already-uncommitted changes on this
branch (`docs/PIPELINE_MAP.md`, `docs/HOLDOUT_PROTOCOL.md`,
`src/data_validation.py`, `src/gene_aliases.py`, `src/split_manifest.py`,
and the provenance/figure/seeding diffs), plus the `git diff` against `HEAD`.

**Found:** `docs/HOLDOUT_PROTOCOL.md` (written earlier this branch) claims
`tests/test_split_manifest.py` exists, covers `src/split_manifest.py` and
`src/gene_aliases.py` with synthetic data, and has been syntax-checked but
not run. The file did not exist (confirmed via `ls tests/` and `git status`).

**Fixed.** Added `tests/test_split_manifest.py` — 15 tests, synthetic data
only (a 2-line temp CSV stands in for a dataset file so `sha256_file` has
real bytes to hash; real MMR gene symbols/accessions from `CANONICAL_GENE_IDS`
so `resolve_gene_id` exercises the actual alias table). Covers: symbol/
accession/case resolution, unknown-gene rejection, same-spelling and
cross-spelling disjointness violations, manifest round-trip through JSON,
manifest-version rejection, and dataset-checksum verification (match and
mismatch). Syntax-checked with `python3 -m py_compile`; not run, for the
same reason nothing else on this branch has been (no pytest/pandas here).

**Reviewed, no defect found.** The seeding fixes in
`scripts/eval_leave_one_protein_out.py` and `scripts/run_mmr_transfer.py`,
the SSRF-style hardening in `src/interpro.py`/`src/structure.py`
(host-prefix checks, page caps), the `download_file` `max_bytes` cap, the
`refresh_manifest` label-count recomputation, and the `figure2`/`figure4`
additions in `scripts/make_figures.py` were cross-checked against their
declared column/schema producers (`src/metrics.py::evaluation_report`,
`scripts/recalibrate_grid.py`) and found internally consistent.
`prepare_split` in `scripts/finetune_esm_mmr.py` was checked against
`assert_disjoint_gene_sets`: not wired in, and correctly so — its two
partitions are a boolean mask and its exact complement over one column, so
they are disjoint by construction and an alias-collision check adds no
protection there; it remains valuable for a future caller that builds
partitions from independently-specified gene lists.

---

## 2026-09-07 (CUDA box + CPU dev box) — post-seed-fix re-run completed, 28 cells

Two pushes closed the re-run item 12 left open, and the paper's model numbers now
all come from one initialisation regime.

- **16 grid cells** (`b9a47c3`, run at `aec4333`) — the full freeze-depth x PLLR x
  branch grid, re-run with the per-split seeding fix.
- **12 feature-family ablation cells** (`472d20b`, run at `a3acf30`, a descendant
  of the fix) — re-run after the first push deleted the originals from the branch;
  see `MISSING_EVIDENCE.md` item 13.

**Provenance holds across the whole set.** All 28 summaries carry a native block,
one `dataset_sha256` (`78eb5d60860c...`) and one `split_definition`
(`6f87ecd06877ff66`). `make_figures.py --strict_provenance` passes. No backfill was
needed, so item 7 closes. Caveat: every summary records `git dirty: true`; what was
uncommitted on the CUDA box at run time is not recoverable from the artifacts.

**The fix moved all four genes, as item 12 predicted** — 58 of 64 grid rows changed.
Mean |dAUROC| vs the pre-fix run: MLH1 0.011, MSH2 0.006, MSH6 0.014, PMS2 0.022.
Largest single shift 0.066 (PMS2, `esm_frozen_pllr-concat`), which is what a
21-variant holdout with 4 negatives does.

**The pre-registered reading survives.** Mean AUROC over the scoreable genes,
`pllr=residual`, seed 42:

| branch | frozen | last2 | full |
|---|---|---|---|
| `esm+priors` | **0.9449** | 0.9358 | 0.9409 |
| `esm` only | 0.9042 | 0.9321 | 0.9270 |

Backbone gradients buy nothing once the priors are there. The branch axis added on
2026-09-02 was the right correction: the 0.880-vs-0.945 gap §6.12 was built around
was the feature set, not the freeze depth.

**Error analysis (item 9, new).** `scripts/error_analysis.py`, run on both headline
arms. 85 consensus errors of 683 for the frozen arm, 97 for the full fine-tune. Every
false positive of the frozen arm is *MSH2*. The covariates that separate errors from
correct calls are `label_source` and `evidence_tier` (BH p = 1.0e-4), **neither a
model input**: 44 of 56 false positives are `evidence_tier == unreviewed`
ProteinGym-clinical benign calls. That is a label-quality finding and it explains the
*MSH2* MCC anomaly logged on 2026-08-28. 193 of 683 variants flip between seeds of one
arm — the item-12 spread, measured per variant.

**Figures.** `scripts/make_figures.py` now produces Figures 2-6. Figure 2 (dataset
composition) is new; panel (c) shows why *MSH2* behaves differently — it carries 144
of the panel's 180 PG-clinical benign labels against *MLH1*'s 6.

---

## 2026-09-06 (CPU dev box) — reproducibility check FAILED, root-caused, fixed

The 2026-09-05 check on the CUDA box (`10fdc98`) ran the same cell twice at one
commit — `repro_a` / `repro_b`, frozen backbone, `esm+priors`, seed 42. It does
not reproduce, and the way it fails names the cause.

| holdout | order | predictions differing | max abs prob delta | ROC-AUC a / b |
|---|---|---|---|---|
| MLH1 | 1st | 208/208 holdout + 95/95 inner-val | 0.942 | 0.9485 / 0.9276 |
| MSH2 | 2nd | 0/335 | 0.0 | bit-identical |
| MSH6 | 3rd | 0/119 | 0.0 | bit-identical |
| PMS2 | 4th | 0/21 | 0.0 | bit-identical |

**Root cause.** `run_one_split` built `ESMFineTuneClassifier` — which
initialises the head — before anything seeded the RNG; the only `set_seed` was
inside `fit_esm_finetune`, several statements later. `main()` never seeded
either, so the *first* split of a process drew its head from the entropy
PyTorch seeds its default generator with at import. Every later split inherited
a state that split one's `set_seed(42)` had already made deterministic, which is
exactly why three of four folds matched to the last bit. `MMR_GENES` is ordered
`(MLH1, MSH2, MSH6, PMS2)`, so the unseeded split has always been MLH1.

**Folds 2-4 matching was contingent, not safe.** They inherit the RNG state
*after* split one's training, so their initialisation depends on how many draws
split one consumed — which depends on the epoch it early-stops at. Both runs
here stopped MLH1 at epoch 3. Had they stopped at different epochs, all four
folds would have moved.

**Fix.** `set_seed(args.seed)` at the top of `run_one_split`, before the model is
constructed. Per split rather than once in `main()` on purpose: it also makes a
single gene re-run with `--eval holdout` reproduce what the full sweep computed
for it, which chained-from-fold-one seeding would not.
`scripts/compare_finetune_strategies.py` had the same defect in
`run_frozen_probe` and `run_esm_finetune` — there the unseeded model is whichever
strategy runs first, so a head-to-head comparison depended on loop order — and is
fixed the same way. No other driver is affected: `run_mmr_transfer.py`,
`train_extended.py`, `eval_leave_one_protein_out.py`, `pretrain_esm_80.py` and
`main.py` all call `set_global_seed` at the top of `main()`.

**Tests:** 1 new in `tests/test_finetune_grid.py`, which aborts `run_one_split` at
model construction and asserts the head's first RNG draw is the same under two
different ambient seeds. It fails on the old code in 4 s on CPU. Suite 214 -> 215,
all passing.

**What this does to the results already in hand.** All 28 grid cells have an
unseeded MLH1. The measured spread from initialisation alone is 0.021 AUROC on
MLH1, against a seed-to-seed spread of 0.019-0.049 on the same gene in the
three-seed arms — so most of what was read as seed variance on MLH1 is
uncontrolled initialisation, and **no MLH1 difference below about 0.02 AUROC
separates two cells.** MSH2/MSH6/PMS2 are unaffected as *measurements*. The
numbers stay valid draws; they are not replayable, and the fix changes every
fold's initialisation, so restoring replayability means re-running.

---

## 2026-09-04 (CPU dev box) — feature-family ablation mechanism; no training run

Code and documentation only. **No model was trained and no number in the paper
changed.**

- **Feature-family ablation** (`MISSING_EVIDENCE.md` item 3, Table 4 rows 4–7).
  `src/transfer.py` names four families in `PRIOR_FEATURE_GROUPS` — `structure`,
  `gnomad`, `domains`, `prior_scores` — and `scripts/finetune_esm_mmr.py`
  exposes `--drop_prior_groups`. The groups partition `TRANSFER_PRIOR_COLS`
  exactly; a prior column added later without a group now fails a test rather
  than being silently unablatable.
- **The proxy guard is the point.** `--drop_prior_groups gnomad` alone *raises*,
  naming the AlphaMissense/zero-shot columns that survive: those models were
  trained on population data, so a "without gnomAD" arm that keeps them has
  removed the legible copy of the signal and nothing else. `--allow_proxy_leak`
  overrides it for the comparison where the proxy is the subject, and is
  recorded in the summary JSON.
- **Provenance.** Each run's summary JSON now records `drop_prior_groups`,
  `allow_proxy_leak` and the resolved `prior_columns` list — "27 prior columns"
  in the paper becomes checkable against the artifact instead of asserted.
- **Tests:** 6 new in `tests/test_mmr_modules.py`; suite 190 → 196, all passing.
- **Manuscript bibliography.** `docs/manuscript/references.bib` was rewritten
  from the verified list in `docs/PAPER.md` §11 plus `docs/DATASETS.md`, and the
  eight now-dangling `\cite` keys in `main.tex` (`acmg`, `clinvar`,
  `proteingym`, `alphamissense`, `gnomad`, `alphafold`, `mavedb`, `esm2`) were
  repointed at the new author-year keys; four citations were added where the
  text names a method the bibliography already covers (PLLR, temperature
  scaling, isotonic regression, circularity). Every `\cite` key now resolves.
  **Not verified by a build** — no TeX toolchain on this box; `latexmk -pdf
  main.tex` still has to run somewhere before submission. Author lists are
  still truncated to `and others`, DOIs are still absent, and `uniprot`,
  `interpro` and `mane` are still `TODO` placeholders.

**Still outstanding:** the four ablation runs themselves. Commands are in
`MISSING_EVIDENCE.md` item 3, pinned to the `esmpri_concat_frozen_pllr-residual_seed42`
grid cell so only the feature set moves.

---

## 2026-09-02 (CPU dev box) — Stage-2b ablation grid: plumbing smoke run

End-to-end check of the new grid path before it leaves for the CUDA box. **8M
parameters, two epochs, one held-out gene — a plumbing test, not a result.**
None of these numbers is comparable to a 650M run and none belongs in the paper.

- **Command:**

  ```bash
  .venv/bin/python scripts/run_stage2b_grid.py --tiers 3 \
    --esm_model facebook/esm2_t6_8M_UR50D --mode siamese \
    --eval holdout --holdout_gene MSH2 \
    --mmr_csv data/mmr/processed/extended/extended_dataset.csv \
    --panel_json data/mmr/processed/extended/panel_sequences.json \
    --epochs 2 --n_bootstrap 200 --out_dir <scratch>/grid_smoke
  ```

- **5/5 tier-3 cells completed, 0 failed.** Split sizes 277 fine-tune / 71
  inner-val / 335 holdout — the clinical-only partition, with PMS2's 21 rows in
  the fine-tune pool.
- Every cell wrote all three artefacts (results CSV, per-variant predictions
  CSV, summary JSON), plus the combined `stage2b_grid_results.csv` and
  `stage2b_grid_manifest.json`.
- **The predictions are seed-ensemblable**, which is the point of the change:
  all five cells return the same 335 variants keyed by
  `gene:position:wt_aa:mut_aa`, each row carrying `label`, `prob`, `threshold`,
  `seed`, `cell_slug`, `branch`, `n_unfrozen_layers`, `pllr_mode`. Runs 1 and 2
  of Stage 2b returned metrics only and cannot be compared variant-by-variant.
- **Resumability verified:** re-running the identical command logs
  `5 already complete | 0 to run` and executes nothing.

**Two things the run itself turned up:**

1. **Resume was broken under `--eval holdout`** — fixed in this commit.
   `finetune_esm_mmr.py` writes `..._siamese_holdout_MSH2_<slug>.csv`, while the
   driver's completeness check looked for `..._siamese_holdout_<slug>.csv`, so no
   holdout cell was ever recognised as complete and a resume silently re-ran the
   whole tier. Both sides now derive the name from
   `src.finetune_grid.output_tag`, with a regression test that creates files
   using the *script's* function and asserts the *driver* finds them. `--eval
   lopo` — what the GPU run uses — was unaffected: the two constructions
   happened to agree there.
2. **Ignore the wall-clock in that manifest.** One cell reports 172.8 min against
   ~2.8 min for its four siblings. The box suspended (s2idle) at 09:39:17 and
   resumed at 12:29:07 mid-cell; `journalctl` confirms it. Profiling the three
   fusion/PLLR configurations directly gives 0.43–0.46 s per example with no
   difference between them, so a tier-3 cell here really costs ~3 min.

Grid driver and `docs/GPU_RUN_PLAYBOOK.md` are ready for the CUDA box; 149 tests
pass.

## 2026-08-30 (Windows CUDA box) — Stage 2b re-run: PLLR fix + PMS2 included

Second Stage-2b run to complete ("Run 2" in `docs/PAPER_DRAFT.md` §6.10).
Incorporates the four ESM-branch corrections from 2026-08-28 (night) — chiefly
the PLLR term `log P(mut|X) − log P(wt|X)` now being structurally computable by
the fine-tuned model — and includes **PMS2 for the first time**: the fail-closed
pseudogene gate was opened with a verified `--pms2_codon_range`, so the 21
held-out PMS2 rows are homology-checked.

- **Command:** `run_mmr_pipeline.py`, siamese / ProPath recipe, `--eval lopo`.
  Pipeline defaults imply full unfreeze (`--n_unfrozen_layers -1`), `--use_pllr`,
  seed 42, 10,000-iteration CIs.
- **Not captured in the pasted console output:** the micro-batch / grad-accum /
  gradient-checkpointing values, the summary JSON, and the exact PMS2 codon
  range. Read `data/processed/esm_finetune/esm_finetune_summary_siamese_lopo.json`
  on the CUDA box and reconcile before quoting a config in the paper.

| holdout | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n | best_epoch |
|---|---|---|---|---|---|---|---|
| MLH1 | 0.9029 | 0.8506–0.9474 | 0.9764 | 0.5275 | 0.9316 | 208 | 5 |
| MSH2 | 0.8503 | 0.8057–0.8913 | 0.7669 | 0.4575 | 0.8963 | 335 | 2 |
| MSH6 | 0.8874 | 0.8200–0.9450 | 0.9121 | 0.6444 | 0.5131 | 119 | 1 |
| PMS2 | 0.9559 | 0.8382–1.0000 | 0.9903 | 0.6912 | 0.0678 | 21  | 3 |

Mean ROC-AUC **0.880** over MLH1/MSH2/MSH6 (n-weighted 0.873); 0.899 with PMS2.
Mean MCC **0.543** (three genes). Artifacts:
`data/processed/esm_finetune/esm_finetune_results_siamese_lopo.csv` (that box).

**Change vs the 2026-08-28 run** (PLLR structurally uncomputable, PMS2 absent,
MLH1 0.8993 / MSH2 0.8776 / MSH6 0.9073, mean 0.895 / MCC 0.492):

- ROC-AUC: MLH1 +0.004, MSH2 −0.027, MSH6 −0.020 → mean **−0.015**.
- MCC (the plan's primary metric): MSH2 0.297 → **0.458** (+0.161); MSH6
  0.652 → 0.644 (−0.008); MLH1 0.528 unchanged → mean **+0.051**, essentially
  all from MSH2.
- Thresholds tightened: MSH2's MCC-optimal threshold moved 0.057 → 0.896, so
  MLH1/MSH2 now agree within 0.04 (MSH6 0.51, PMS2 0.068 remain outliers).
- `best_epoch` 1/3/1 → 5/2/1/3: MLH1 now trains a meaningful number of epochs.

Net: the fix trades a fraction of a ranking point for a better operating point.

**Still owed (unchanged):** `--n_unfrozen_layers 0` ablation floor,
`--no-use_pllr` ablation, ≥3-seed averaging — now specified as a single grid in
`docs/PAPER_DRAFT.md` §6.12 (Table 8). Until the floor runs, these numbers
cannot be attributed to backbone fine-tuning: the frozen priors-probe (Stage 2,
2026-08-28: MLH1 0.9425 / MSH2 0.8527 / MSH6 0.9734, mean 0.923; from-scratch
27-feat variant mean 0.945) is still **ahead of Stage 2b by ~4–6 ROC-AUC
points**.

## 2026-08-28 (night, CPU box) — fixing the ESM branch

The ESM branch was the weakest part of the project (siamese LOGO mean ROC-AUC
0.895 against the priors probe's 0.944, early-stopping at epoch 1/3/1). Four
defects found, all in the branch itself rather than in the data.

### 1. The fine-tuned model could not see PLLR *(root cause)*

`ESMFineTuneClassifier` was built on `AutoModel` — the encoder alone, no
masked-LM head — so `log P(mut|X) - log P(wt|X)` was structurally
uncomputable. That term is the zero-shot ESM score, which on this panel
reaches **ROC-AUC 0.834 pooled with no training at all**. A 650M-parameter
backbone on 662 labels was being asked to rediscover it from scratch, and
early-stopped before it could.

Now loads `AutoModelForMaskedLM`, keeps `.esm` + `.lm_head`, and reads PLLR off
the **same** wild-type forward pass that already produces `site_wt` (Meier et
al. 2021 — one shared context, so no extra compute). Verified identical to
`src/esm_extractor.py`'s existing PLLR to **2.4e-06**. `--no-use_pllr` runs the
ablation. The LM head follows the backbone's freeze state.

### 2. Unnormalised head input

`feat` went straight into `nn.Linear`; the LayerNorm sat *after* it. Raw ESM
hidden states have large position-dependent norms, and in siamese mode half the
concatenated vector (`site_wt`, `site_mut`) is near-duplicate while the
informative part — their difference at one substituted residue — is small. A
`LayerNorm` now precedes the head. PLLR is scaled by a fixed constant (10.0,
a registered buffer, not a batch statistic) so it stays identical between
training and single-variant inference.

### 3. MVmamba's variant-type local window was the *wild-type* window

For chains over the positional capacity the branch sliced the centred window
out of `sequence` (wild type) and assigned it to `l_vt`. So `l_vt == l_wt`
exactly, and `l_vt - l_wt` / `|l_vt - l_wt|` were **identically zero** — two of
eight feature blocks dead, and precisely the windowed WT/VT contrast the
MVmamba recipe is built around. On this panel only **MSH6** (1360 aa) exceeds
the limit, so one gene silently had a different feature space from the other
three, inside a leave-one-gene-out design.

### 4. `compare_backbones.py` reported 1 - AUC

Raw PLLR is *negative* for damaging variants, but it was compared directly
against a pathogenic=1 label. Every cell came out below 0.5 (ROC-AUC
0.03-0.18 for two strong 650M models) — output that reads as "the protein
language model is useless on our data". Added
`MaskedMarginalScorer.pathogenicity_score()` (negated, higher = more
pathogenic) and switched the script to it. **Any earlier reading of that
script's output should be discarded.**

### Also fixed

The MVmamba extractor materialised the full `[N, L, d]` hidden tensor:
**25.2 GiB** for MSH6's 3,886 VUS at 1360 aa x 1280 dims — an immediate OOM on
a 14 GiB box, on exactly the VUS-scoring task the pipeline exists to perform.
Both outputs wanted from that pass are reductions, so it now embeds in chunks
(`vt_chunk_size`, default 64) and reduces on arrival: peak ~0.4 GiB. Verified
bit-identical to the old implementation on the short-sequence path at chunk
sizes 1/3/64; the long-sequence path differs only by defect 3 above.

### Verification

- In-model vs extractor PLLR: max abs diff 2.4e-06 across 6 variants.
- Chunked vs original MVmamba features: max diff 0.00e+00 (short path).
- End-to-end `finetune_esm_mmr.py` on CPU (esm2_t6_8M, frozen backbone,
  holdout MSH6): validation AUC now **rises across epochs** — 0.8813, 0.8829,
  0.8895 — instead of peaking at epoch 1. Holdout ROC-AUC 0.847. That is an
  8M-parameter smoke test, **not** a result to compare against the 650M runs.
- 86 tests pass under pytest; 37/37 in `test_mmr_modules.py` standalone. Three
  regression tests added (PLLR orientation, WT/VT local contrast, chunking).

### Still owed on the GPU box

- ~~Re-run stage 2b with PLLR on 650M~~ — done 2026-08-30 (see the entry at the
  top of this log). ROC-AUC mean −0.015, MCC mean +0.051 vs the pre-fix run.
- `--no-use_pllr` ablation, `--n_unfrozen_layers 0` ablation floor, and seed
  averaging are now a single grid — see the top-of-log TODO and
  `docs/PAPER_DRAFT.md` §6.12.

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
- [ ] Stage 2b ablation grid on the CUDA box — freeze depth × PLLR × **branch**
      (`esm` vs `esm+priors`), 16 cells in 5 tiers, ~31 h. One resumable command;
      see `docs/GPU_RUN_PLAYBOOK.md`. Every filename is cell-tagged, so the old
      "move the results CSV aside between cells" step is gone. Subsumes the old
      "ablation floor" and "seed averaging" line items. `PAPER_DRAFT.md` §6.12
      and `PAPER.md` §7.2/Table 6 now describe *this* grid; the earlier
      freeze-depth-only design they carried could not separate freeze depth from
      feature set.
- [ ] Remaining recipes on the same splits: `wt_site`, VariPred probe, MVmamba pooled
- [ ] Full model: `train_extended.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D --no_dms_features`
- [ ] Leave-one-**protein**-out CV on the broad 80-gene panel (still owed; distinct from the MMR gene-out above)
- [ ] Score VUS risk tiers from ESM-mode run (`ext80_vus_predictions.csv`)
