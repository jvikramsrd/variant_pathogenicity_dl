# Design — Stage-2b fair comparison, attribution ablation, and paper close-out

**Date:** 2026-09-02
**Status:** design, awaiting review
**Target:** journal submission (Bioinformatics / Datasets & Benchmarks track)
**Hardware constraint:** dev box is CPU-only; a separate Windows box has a ~15 GiB
CUDA card with a ~40–50 h weekend budget. Results come back to this box as files.

---

## 1. The problem this design solves

Stage 2b (ESM-2 650M siamese backbone fine-tune) reports mean unseen-gene ROC-AUC
**0.880** over MLH1/MSH2/MSH6. The frozen priors probe reports **0.945** on the
identical leave-one-gene-out splits. `docs/PAPER_DRAFT.md` §6.12 proposes to
resolve that gap with a freeze-depth × PLLR ablation and, if the frozen floor
matches the full fine-tune, to conclude that "backbone gradients bought nothing."

**That conclusion would not survive review, because the two models do not see the
same features.**

`ESMFineTuneClassifier` (`src/esm_finetune.py:161`) builds its head over
`hidden_size * 4 + (1 if use_pllr)`. There is no prior-feature input anywhere in
the module. The priors probe reads 27 columns — AlphaMissense, 17 ProteinGym
zero-shot scores, gnomAD constraint, AlphaFold pLDDT, InterPro domains. The
measured 6.5-point gap therefore confounds **freeze depth** with **feature set**,
and freeze depth is the only axis §6.12 varies.

Four further defects in the same training path plausibly account for the
`best_epoch = 1–5` early stopping that §6.10 flags as unexplained:

| # | Defect | Location | Contrast |
|---|---|---|---|
| D1 | No LR warmup or decay — constant 1e-5 AdamW on a 650M backbone | `src/esm_finetune.py:454` | `src/train.py:140` uses `CosineAnnealingLR` |
| D2 | No `pos_weight`; per-gene prevalence runs 40–82% positive | `src/esm_finetune.py:475` | `src/train.py:138`, `src/transfer.py:389` both compute it |
| D3 | PLLR enters as 1 of 5,121 input dims with a random init weight, so the untrained model discards a 0.834-AUC signal | `src/esm_finetune.py:254` | — |
| D4 | No per-variant predictions are written — only 4 rows of metrics | `scripts/finetune_esm_mmr.py:310` | `run_mmr_transfer.py` writes a predictions CSV |

D4 is the one that bites this specific workflow hardest: a 45-hour run on the
other box currently returns a metrics table and nothing else, so no post-hoc
seed ensembling, calibration analysis, or per-gene threshold study is possible
from the returned artefacts.

---

## 2. Scope

**In scope**

1. A priors branch and the D1–D4 fixes in the Stage-2b path (§4).
2. A resumable, cell-tagged grid driver for the weekend GPU run (§5).
3. CPU-only experiments that close paper gaps without consuming GPU budget (§6).
4. Data expansion under a leakage quarantine, measured as an isolated delta (§7).
5. Paper and supporting-doc close-out, plus a result-ingest path (§8).

**Out of scope**

- Multitask continuous-DMS heads (`PAPER_DRAFT.md` §8 limitation 3).
- The LLM clinical-text branch that `src/fusion.py` anticipates.
- Any change to `src/extended_builder.py`'s merge semantics or the 12-check audit.

---

## 3. Two constraints that govern every decision below

### 3.1 The AF-label / AF-feature quarantine

`src/gnomad.py:312-315` derives `acmg_ba1 = AF > 0.05`, `acmg_bs1 = AF > bs1_af`,
`acmg_pm2 = AF < pm2_af`. `src/transfer.py:58` places all four of
`gnomad_log10_af, acmg_ba1, acmg_bs1, acmg_pm2` in `TRANSFER_PRIOR_COLS`.

Minting benign labels from allele frequency while feeding allele frequency as a
feature makes `acmg_bs1 == label` **by construction** on every minted row. That is
Finding 2 (`dms_bin_median == 1 − label` on 97.3% of rows) reproduced inside the
paper that reports Finding 2.

**Rule.** Whenever AF-derived labels are in the training pool, the four AF-derived
columns are removed from the feature set, and the rule is enforced in code — not
by convention — with a test that fails if both are enabled at once.

### 3.2 The evaluation set never changes

Every number in §6.10, §6.12 and the published-predictor benchmark is computed on
ClinVar ≥2★ ∪ ProteinGym-clinical held-out rows. Data expansion (§7) adds weaker
labels to the **fine-tune partition only**. `prepare_split` currently applies one
filter to both partitions; it must take separate train and eval label policies.

Two supporting observations:

- The MMR table holds **16,420 MSH2 DMS labels** against 683 clinical ones. Those
  DMS rows are almost certainly the Jia et al. MSH2 LOF screen that
  `PAPER_DRAFT.md` §5.8 declares withheld from training by construction. Verify
  the assay identity before any change touches the DMS filter; if it matches,
  the existing `label_source.isin(["clinvar", "pg_clinical"])` filter is
  load-bearing for the external-validation claim and must be commented as such.
- The broad train CSV holds 2,368 `clinvar` + 1,034 `pg_clinical` = **3,402**
  clinical rows, against the 5,152 and 5,466 quoted in the paper. This is the
  open `[RECOUNT]` in `PAPER_DRAFT.md` §5.2. Resolve it with a script that counts
  every candidate definition on one table and prints them side by side.

---

## 4. Sub-project 1 — Fair-comparison Stage 2b

### 4.1 Priors branch

`FineTuneExample` gains `priors: Optional[np.ndarray]`. `build_examples` takes an
optional aligned prior matrix and carries each row's vector onto its example;
`make_collate_fn` stacks them into `[B, P]`.

`ESMFineTuneClassifier.__init__` gains `n_prior_features: int = 0` and
`fusion: str = "concat"`. When `n_prior_features > 0` the ESM feature block and
the prior vector are fused through the **existing** heads in `src/fusion.py`
(`ConcatFusionHead`, `GateWaveFusionHead`) rather than a new implementation, so
Stage-2b architectures stay directly comparable to Stage-2's `concat_fusion` and
`gatewave_fusion` cells.

**Blocking detail.** `ConcatFusionHead` uses `nn.BatchNorm1d` (`src/fusion.py:70`).
The GPU config that fits a 650M full fine-tune on a 15 GiB card is
`--batch_size 1 --grad_accum 8`, and `BatchNorm1d` raises at batch size 1 in
training mode. Add a `norm: str = "batch"` parameter to `ConcatFusionHead`
defaulting to the current behaviour, and have Stage 2b pass `norm="layer"`.
Existing Stage-2 results stay reproducible; the fine-tune path gets a norm that
works at micro-batch 1.

**Prior preprocessing discipline**, mirroring `src/transfer.py`:

- Column set from `prior_columns_of(df, drop_gene_constant=True)` for every
  leave-one-gene-out run. Non-negotiable: `RUNLOG.md` 2026-08-28 records MLH1
  collapsing to ROC-AUC 0.500 in every seed when the five gene-constant columns
  are kept under LOGO.
- Impute values and standardisation constants fitted on the **fine-tune partition
  only**, applied unchanged to inner-val and holdout, and persisted into the
  checkpoint via `save_finetuned`.
- `is_missing_*` indicators preserved.
- Under §3.1, the AF columns are dropped when AF labels are active.

### 4.2 PLLR as a residual base (D3)

```
logit = pllr_gain * (pllr / pllr_scale) + head_out
```

`pllr_gain` is a learnable scalar initialised to **−1.0** (raw PLLR is negative
for damaging variants; pathogenic is the positive class), and the final `out`
layer is zero-initialised. The untrained model is then exactly the zero-shot ESM
predictor — ROC-AUC ≈ 0.834 on this panel — and training learns a residual on top
of a working predictor instead of rediscovering it from a few hundred labels.
`out.weight` leaves zero on the first optimizer step, so nothing is frozen out.

A `--pllr_mode {residual,concat,off}` flag retains the current concatenation
behaviour as an ablation cell; `off` is today's `--no-use_pllr`.

### 4.3 Warmup + cosine schedule (D1)

`fit_esm_finetune` gains `warmup_frac: float = 0.1` and a `LambdaLR` stepping per
**optimizer step**, not per epoch — with 10 epochs and few batches, per-epoch
granularity is too coarse to matter. Total steps =
`epochs * ceil(n_batches / grad_accum)`. Early stopping may truncate the schedule;
that is expected and standard.

### 4.4 `pos_weight` (D2)

Compute `n_neg / n_pos` on the fine-tune partition and fold it into the
per-example weight for positives, matching `src/transfer.py:412`'s
`torch.where(y > 0.5, pos_weight, 1.0)` pattern exactly.

### 4.5 Frozen-backbone feature cache

When `n_unfrozen_layers == 0` the ESM feature block is constant across epochs.
Extract once, cache keyed by `(model_name, mode, max_residues, variant-key hash)`,
and train the head on cached tensors.

This is what makes the ablation floor affordable: a floor cell drops from roughly
an hour to about fifteen minutes, so every floor cell can carry three seeds
instead of one. The cache key must include the variant keys, not the row count —
`PAPER_DRAFT.md` §6.5 records a prior bug where a row-count-only cache returned
another variant's embeddings silently.

### 4.6 Per-variant predictions and cell-tagged outputs (D4)

`finetune_esm_mmr.py` writes, per run:

- `predictions_<cell>.csv` — one row per held-out variant: keys, gene, label,
  probability, seed, and every config field defining the cell.
- `results_<cell>.csv` — the existing metrics table, cell-tagged.
- `summary_<cell>.json` — config, environment, timings, git SHA.

`<cell>` is a deterministic slug over `(mode, branch, n_unfrozen, pllr_mode, seed,
esm_model, eval)`. This removes the manual "move each CSV aside before the next
run" step that `RUNLOG.md` currently mandates across a dozen cells — an
error-prone instruction to follow at 3 a.m. on a weekend run.

### 4.7 Verification on this CPU box

Every change above is verified before the code leaves for the GPU box, using
`facebook/esm2_t6_8M_UR50D` on a holdout gene: shape and dtype assertions on the
fused forward pass at micro-batch 1; PLLR residual reproducing the zero-shot
score at initialisation to within tolerance; schedule producing the expected LR
trace; `pos_weight` matching `src/transfer.py`'s value on the same partition; the
frozen cache proven bit-identical to the uncached path; and a test asserting that
AF labels plus AF features raises. These are smoke tests for correctness, **not**
results to compare against 650M runs.

---

## 5. Sub-project 2 — The grid driver

`scripts/run_stage2b_grid.py`:

- Takes a tier name or an explicit cell list; runs cells sequentially.
- **Resumable** — skips any cell whose `summary_<cell>.json` exists and validates.
  A 45-hour Windows run will be interrupted; requiring a clean restart would lose
  the weekend.
- Writes a run manifest (torch/transformers versions, GPU name and VRAM, git SHA,
  dataset SHA-256) and an aggregated `stage2b_grid_results.csv`.
- Prints a running wall-clock projection against the tier's budget.

### 5.1 The grid

Cost model, LOGO over 4 genes, 10 epochs, 650M siamese, 15 GiB card: full
unfreeze ≈ 5 h/cell (`PAPER_DRAFT.md` §6.12's ~16 h for three such cells);
`nuf=2` ≈ 2 h; `nuf=0` with §4.5's cache ≈ 0.25 h.

| Tier | Cells | Seeds | Cost |
|---|---|---|---|
| 1 — headline fair fight | `esm+priors × {nuf=−1, nuf=0} × pllr=residual` | 42,43,44 | 15.75 h |
| 2 — branch attribution | `esm × {nuf=−1, nuf=0} × pllr=residual` | 42 | 5.25 h |
| 3 — PLLR axis at the floor | `{esm+priors, esm} × nuf=0 × pllr={off, concat}` | 42 | 1 h |
| 4 — freeze-depth middle | `{esm+priors, esm} × nuf=2 × pllr=residual` | 42 | 4 h |
| 5 — PLLR at full FT | `esm+priors × nuf=−1 × pllr=off` | 42 | 5 h |
| **Subtotal** | | | **31 h** |
| 6 — data-expansion delta (§7) | winning Tier-1 cell, expanded training pool | 42 | 5 h |
| **Total** | | | **36 h**, ~9 h buffer |

The priors-only comparator row is not in this grid — it runs on CPU here (§6) and
costs no GPU time.

### 5.2 What the grid answers

- **Tier 1 vs the priors probe** — does ESM add anything on top of published
  priors, on unseen genes? This is the paper's central model claim, and it is
  currently unmeasured in either direction.
- **Tier 1 vs Tier 2** — how much of any gain is the priors branch versus the
  backbone? The comparison §6.12 currently claims to make, made properly.
- **`nuf=−1` vs `nuf=0` within a branch** — the actual freeze-depth question,
  with feature set held constant.
- **Tier 3** — what the 2026-08-28 PLLR fix is worth, measured where it is
  cleanly measurable (at `nuf=0`, where the backbone cannot relearn the term).

The §6.12 pre-registered reading rules carry over unchanged, restated per branch.
**They are pre-registered: the decision rule is fixed in the spec before the
numbers land, and it is not revised afterwards.** If seed sd exceeds between-cell
differences, the ablation is reported as underpowered and cells are not ranked.

---

## 6. Sub-project 3 — CPU experiments, run here, in parallel

None of these consume GPU budget. The priors probe fits a gene in ~12 s.

1. **Leave-one-protein-out on the broad 80-gene panel** —
   `eval_leave_one_protein_out.py --features priors --n_bootstrap 10000`.
   `PAPER_DRAFT.md` §6.7 calls this "the single most important missing
   experiment"; it is ~20 minutes of CPU. Fills Table 4. Note the script defaults
   to `--n_bootstrap 2000`; pass 10,000 to match the paper's protocol.
2. **Seed averaging (≥3 seeds)** on §6.6 and on the priors-probe comparator.
   `PAPER_DRAFT.md` §10 item 2 calls its absence "a common desk-reject."
3. **Frozen fusion battery** — `run_mmr_transfer.py --eval lopo --features
   esm+priors` over all five architectures. Extraction covers only labelled rows
   (683 for MMR), so this is feasible on CPU; fall back to a smaller checkpoint
   if 650M extraction is too slow, and label the result by checkpoint.
4. **Temporal ClinVar holdout** — `variant_summary.txt.gz` carries
   `LastEvaluated`. This is the only answer to the AlphaMissense-contamination
   objection (`PAPER_DRAFT.md` §8 limitation 2, §10 item 5).
5. **The `[RECOUNT]`** of §3.2 — one script, every candidate definition, one table.
6. **MMseqs2 cluster-disjoint split** — `mmseqs` is **not installed on this box**
   (verified). Either install it or record the third rung of the split ladder as
   unavailable; do not leave Fig 6 pending without saying which.

---

## 7. Sub-project 4 — Data expansion, measured as an isolated delta

Three approved changes: gnomAD-AF-derived benign labels, ClinVar 1-star with
star-scaled `label_weight`, and a broad-panel `--all_sources` rebuild plus
re-pretrain (which fixes the known 19-vs-27 schema mismatch that currently makes
warm-starting useless).

**Sequencing is the methodological point.** The grid in §5 runs on the *current*
dataset, so its cells stay attributable and comparable to Run 2. Data expansion is
then measured as **one before/after delta on the winning configuration** (Tier 6).
Changing the data, the model, and the training recipe simultaneously and reporting
the aggregate is precisely the practice this paper criticises.

Implementation:

- `--min_stars 1` already exists; `label_weight` scaling by stars already exists.
  The change is a with/without sensitivity analysis, because 1-star single-submitter
  assertions frequently cite PP3/BP4 computational evidence, which makes
  AlphaMissense and the `zs_*` columns partially circular with the label. Report both.
- A new `--af_benign_labels` flag on the MMR builder mints `label=0`,
  `label_source="gnomad_af"`, low `label_weight`, and **sets a flag in the output
  manifest that forces the §3.1 feature quarantine downstream**.
- `prepare_split` takes separate train and eval label policies (§3.2).

---

## 8. Sub-project 5 — Paper, docs, and the ingest path

### 8.1 New and changed docs

- **`docs/GPU_RUN_PLAYBOOK.md`** (new) — the exact command sequence for the other
  box, in order, with per-tier wall-clock expectations, VRAM preflight
  invocation, resume instructions after an interruption, and an explicit list of
  which files to bring back. This is the artefact the workflow actually turns on.
- **`docs/PAPER.md`** — the manuscript. Gains a sixth finding (the AF-label
  circularity near-miss of §3.1, caught in our own expansion plan), a §6.12
  rewritten around the branch axis, and the resolved `[RECOUNT]`.
- **`docs/PAPER_DRAFT.md`** — §6.12 grid and reading rules replaced by §5 above;
  §6.9/§6.10 "still owed" lists reconciled.
- **`docs/RUNLOG.md`** — an entry per CPU run as it completes, per its existing
  newest-at-top format.
- **`docs/TECH_STACK.md`, `PROJECT_PLAN.md`, `README.md`** — reconciled with the
  new flags and the new script.
- **`docs/PP1_Review1_slide_content.md`** — refreshed against final numbers.

### 8.2 `scripts/ingest_gpu_results.py` (new)

Point it at the returned artefact directory; it validates the manifest against the
expected grid, reports which cells are missing, computes seed means and standard
deviations, builds the seed-ensemble rows from the per-variant predictions of §4.6,
and emits paper-ready markdown for Tables 7 and 8 plus the Fig 7 forest plot.

This is what makes "I'll give you the data once I run it on another PC" a
mechanical step rather than a transcription exercise. It also enforces the §5.2
pre-registered reading: the script prints the decision-rule verdict from the
numbers, before any prose is written about them.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| The priors branch wins and ESM adds nothing even when fused | That is a publishable negative result and the §5.2 rules pre-commit to reporting it. It is also the current best guess. |
| 45 h proves optimistic; Tiers 4–6 do not finish | Tiers are ordered by scientific value. Tier 1 alone (15.75 h) settles the headline. The driver is resumable. |
| BatchNorm/micro-batch-1 interaction has further surprises at 650M | Verified at 8M on CPU first (§4.7); the VRAM preflight already gates the GPU run. |
| 1-star ClinVar imports predictor-derived labels | Reported as a with/without sensitivity analysis, never silently merged. |
| MSH2 DMS rows are the withheld Jia et al. assay | Verify assay identity before touching the DMS filter (§3.2). |
| `mmseqs` unavailable | Install, or state the third rung as unavailable rather than pending. |

---

## 10. Open questions for review

1. **Fusion default for Stage 2b** — `concat` or `gatewave`? Concat is the plan's
   default and the safer baseline; GateWave is the MVmamba-derived design the
   project already implements. Proposal: concat as the Tier-1 default, GateWave
   added as a cheap floor-only cell in Tier 3.
2. **Is `mmseqs2` worth installing** on this box for the third split rung, or is
   the two-rung ladder acceptable for submission?
3. **ProteinGym MSH2 assay identity** — does anyone already know whether it is the
   Jia et al. screen, or should the spec's verification step stand?
