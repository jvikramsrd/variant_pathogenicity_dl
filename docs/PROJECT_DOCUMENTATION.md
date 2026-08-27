# Project Documentation — Variant Pathogenicity DL

**What this project is, what has been built, why each piece exists, and what it
improves on relative to the code and the published methods it started from.**

*Status of this document: written 2026-08-26, revised 2026-08-27. Every claim
about code is checked against the tree at that date; every performance number is
quoted from `docs/TRAINING_NOTES.md` / `docs/RUNLOG.md` and is a **past** run's
result — see §8 for what is and is not currently reproducible on this machine.*

*For the mechanics — how each module works, the merge walked through step by
step, and the alternative design for every significant decision — see
`docs/CODE_GUIDE.md`. This file is the "what and why"; that one is the "how".*

---

## Table of contents

1. [The problem and why it is hard](#1-the-problem-and-why-it-is-hard)
2. [System overview](#2-system-overview)
3. [The data layer — what was built and why](#3-the-data-layer--what-was-built-and-why)
4. [The learning layer — what was built and why](#4-the-learning-layer--what-was-built-and-why)
5. [The evaluation layer — what was built and why](#5-the-evaluation-layer--what-was-built-and-why)
6. [The MMR / Lynch-syndrome project](#6-the-mmr--lynch-syndrome-project)
7. [Improvements over the existing work](#7-improvements-over-the-existing-work)
8. [Results, and how to read them honestly](#8-results-and-how-to-read-them-honestly)
9. [Reproduction](#9-reproduction)
10. [Known limitations and open work](#10-known-limitations-and-open-work)
11. [Where everything lives](#11-where-everything-lives)

---

## 1. The problem and why it is hard

A **missense variant** changes one amino acid in a protein. Clinical genetics
needs to know whether that change is *pathogenic* (causes disease) or *benign*.
Most observed missense variants are neither: they are **Variants of Uncertain
Significance (VUS)** — the single largest bottleneck in returning actionable
results to patients.

Three properties of this problem shape every design decision in this repository:

| Property | Consequence for the design |
|---|---|
| **Clinical labels are scarce and unevenly distributed.** ClinVar has millions of rows but only a few thousand two-star-or-better missense assertions for any given gene panel. | Supervision has to be pooled across proteins and supplemented; a single-gene model has too few labels to train anything deep. |
| **The plentiful labels are proxies, not ground truth.** Deep mutational scanning (DMS) measures *assay fitness*, not clinical consequence. Published predictor scores were themselves trained on ClinVar. | Loss weighting, label provenance tracking, and separate *clinical-only* evaluation slices are mandatory, or the model silently optimises the wrong target and the reported metric is circular. |
| **Leakage is the default outcome, not the exception.** Variants at the same residue, paralogous proteins, an assay-derived feature that encodes the label, a calibrator fitted on the fold it scores — each of these inflates metrics without any learning. | Leakage control is a first-class subsystem here (group-disjoint splits, cluster-disjoint splits, nested calibration, target-leakage diagnostics, a 12-check audit), not an afterthought. |

The project therefore has two goals that are equally weighted: build a model
that predicts pathogenicity well, and build an evidence pipeline whose numbers
can be trusted. Much of the work described below is the second goal.

---

## 2. System overview

The repository contains **two sibling pipelines** that share the same `src/`
primitives.

### 2.1 Broad-panel pipeline (`run_pipeline.py`, `main.py`)

The original workflow, extended: build a leakage-audited multi-source table over
a panel of proteins, then train an MLP head over ESM-2 embeddings and external
priors with grouped cross-validation and post-hoc calibration.

```
ClinVar · ProteinGym DMS · ProteinGym clinical · 17 published zero-shot scores
· AlphaMissense · UniProt domains (+ optional gnomAD / AlphaFold / InterPro)
        │
        ▼  src/external_datasets.py, src/extended_builder.py
   master table, keyed (uniprot_id, position, wt_aa, mut_aa) + manifest.json
        │
        ▼  scripts/audit_extended_dataset.py    (12-check integrity audit)
   extended_dataset_train.csv
        │
        ▼  src/esm_extractor.py     z = [h_wt ‖ h_mut ‖ Δh ‖ |Δh| ‖ PLLR] (+ priors)
        ▼  src/train.py             StratifiedGroupKFold on uniprot:position
        ▼  src/calibration.py       temperature / isotonic vs PLLR baseline
   OOF metrics · fold checkpoints · calibrated VUS risk predictions
```

### 2.2 MMR / Lynch-syndrome pipeline (`run_mmr_pipeline.py`) — recommended entrypoint

A two-stage transfer-learning workflow targeting the four mismatch-repair genes
**MLH1 / MSH2 / MSH6 / PMS2**, with a third, genuinely deep training stage:

| Stage | Script | What it does |
|---|---|---|
| 1 | `scripts/build_extended_dataset.py` | download + clean the broad 80-gene pretraining panel |
| 2 | `scripts/build_mmr_dataset.py` | download + clean the dedicated 4-gene MMR table (pinned references, gnomAD, VCEP tiers, PMS2 gate, MaveDB/CIMRA validation columns) |
| 3 | `scripts/pretrain_esm_80.py` | **Stage 1** — fit the head on frozen ESM-2 embeddings across all 80 panel genes (MMR genes excluded in `leave_gene_out` mode) |
| 4 | `scripts/run_mmr_transfer.py` | **Stage 2** — warm-start fine-tune on MMR clinical labels; leave-one-gene-out eval with bootstrap CIs and the fusion/branch ablation battery |
| 5 | `scripts/finetune_esm_mmr.py` | **Stage 2b** — true ESM-2 **backbone gradient** fine-tuning (ProPath Siamese or CSBJ token-classifier recipe) |

Stages 1–2 of training are *linear probes over frozen embeddings*; stage 2b is
the actual deep-learning training where gradients flow into the transformer.
This distinction is deliberate and is now surfaced in the CLI, the banners and
the docs, because conflating the two is the single easiest way to overstate what
the model has learned. Stage 2b runs **by default**; `--no-full_finetune` stops
at the frozen-embedding stages for a CPU-friendly smoke run.

Because that default can turn an afternoon into a week, the pipeline probes for
a CUDA/MPS device **before** the downloads and the frozen-embedding stages and
stops with an explanation if there is none — `--allow_cpu_finetune` overrides.
Stage banners are numbered against the stages your flags will actually run, so
`Stage 3/4` means three of four real stages are done.

---

## 3. The data layer — what was built and why

### 3.1 Sources adopted, and the reason for each

| Source | What it contributes | Why it was adopted |
|---|---|---|
| **ClinVar** `variant_summary` | primary clinical supervision + the VUS set to score | the only large, curated, review-status-graded clinical label set |
| **ProteinGym v1.3 DMS** (217 assays) | ~2.5M experimentally measured mutant fitness values | the only supervision that scales to hundreds of thousands of rows; keyed directly by UniProt accession |
| **ProteinGym v1.3 clinical benchmark** | 63K independently curated clinical labels | an *independent* clinical label source to cross-check ClinVar against |
| **ProteinGym zero-shot scores** | 17 published model scores (EVE, ESM-1b, GEMME, …) | strong priors for free, and the baseline suite any headline claim must beat |
| **AlphaMissense** | full-proteome pathogenicity prior (~1.16M rows for the panel) | the current state-of-the-art general prior; both a feature and the baseline to beat |
| **UniProt** sequences + domains + point features | canonical reference sequences, domain flags, active/binding sites, PTMs | numbering safety anchor *and* cheap structural context |
| **gnomAD v4** (opt-in) | variant allele frequency + BA1/BS1/PM2 ACMG flags; gene-level constraint (pLI, o/e, z-scores) | population frequency is the single strongest *benign* evidence in ACMG practice and is absent from every sequence-only model |
| **AlphaFold DB** (opt-in) | per-residue pLDDT, disorder flag | cheap structural confidence without any structure pipeline |
| **InterPro** (opt-in) | domain/family/superfamily calls | complements UniProt's own domain annotation |
| **MaveDB** (MMR only) | MSH2 Jia 2021 LOF screen (17,746 variants); MLH1 2025 abundance assay | orthogonal functional evidence — held out as **validation only** |
| **CIMRA OddsPath** (MMR only, user-supplied CSV) | calibrated ACMG evidence strengths | orthogonal, clinically calibrated evidence — **validation only** |

Sources considered and **rejected**, with reasons, are recorded in
`docs/RUN_REPORT.md` §2 and `docs/DATASETS.md` §11 (dbNSFP: ~30 GB and redundant
with the `zs_*` columns; EVE repo: ships MSAs only, scores arrive via ProteinGym).
Recording the rejections matters as much as the adoptions — it is what stops the
same evaluation being redone every few months.

### 3.2 Merge discipline

Every source is normalised to one key, `(uniprot_id, position, wt_aa, mut_aa)`,
and **validated against the UniProt canonical sequence before joining**. Rows
whose wild-type residue disagrees with the canonical sequence are *dropped, not
merged* — 50,304 isoform-mismatched DMS rows and 152 AlphaMissense rows were
discarded on the 80-gene build. RefSeq `NP_` accessions are matched by **exact
whole-protein sequence equality**, never by identifier string.

*Why:* silent isoform/numbering mismatch is the most common way a multi-source
variant table becomes quietly wrong. A dropped row is recoverable; a wrongly
joined row corrupts training and is invisible downstream.

### 3.3 Label policy and conflict quarantine

- **Precedence:** ClinVar > ProteinGym-clinical > single-assay DMS bin. Raw
  columns from every source are always preserved alongside the resolved label.
- **Default confidence floor:** ClinVar assertions with **≥2 review stars**.
  `--min_stars 1` is available but is an explicit, documented choice.
- **Every label carries `label_source` and `label_weight`** (ClinVar weighted by
  review stars, PG-clinical 0.75, DMS 0.20), combined at train time with the
  experiment's clinical/DMS weights.
- **Cross-source disagreement is quarantined, not resolved by fiat:** the raw
  evidence stays in the master CSV, but `label` is left blank and
  `label_conflict=1`, so a conflicted variant cannot enter the training CSV.
- **The builder refuses to export** a table with null or duplicate keys,
  unresolved metadata, non-binary labels, unquarantined conflicts, or labels
  without a recognised source.

*Why:* heterogeneous public evidence cannot establish clinical truth. The honest
engineering response is to make disagreement explicit and unusable for training,
rather than to pick a winner and lose the fact that there was a dispute.

### 3.4 Provenance

Every remote artefact is checksummed (SHA-256) into
`data/processed/extended/manifest.json` with URL, version, byte count and date;
generated artefacts are checksummed too. Downloads stage into a `.part` file and
are promoted atomically only after size and checksum validation, so a server
that ignores an HTTP Range request can no longer corrupt a resumed download.
ProteinGym ZIPs are integrity-checked before parsing.

### 3.5 The independent audit

`scripts/audit_extended_dataset.py` re-derives the merge from scratch and runs a
**12-check integrity audit** (`audit_report.json`): key uniqueness, wt-residue
validation against canonical sequences, label-precedence re-derivation,
provenance-token coherence, cross-source conflicts, and a feature-coverage
report. The 80-gene build passes **12/12**, and the 1,919 variants covered by
both ClinVar and ProteinGym-clinical agree **100%**.

*Why:* the audit is deliberately a *separate* implementation from the builder.
A builder that validates itself only proves it is self-consistent.

---

## 4. The learning layer — what was built and why

### 4.1 Feature construction

For a variant `(i, wt → mut)` in sequence `X`, `src/esm_extractor.py` builds

```
z = [ h_wt ‖ h_mut ‖ (h_mut − h_wt) ‖ |h_mut − h_wt| ‖ PLLR_i ]
```

where `h_wt` / `h_mut` are last-layer ESM-2 hidden states at residue `i` for the
wild-type and in-silico-mutated sequences, and

```
PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)
```

is the pseudo-log-likelihood ratio zero-shot score. Both log-probabilities are
conditioned on the *same* masked context, so both are read from a **single**
forward pass over the wild-type sequence (Meier et al. 2021). Sequences longer
than the 1,022-residue positional capacity are handled with overlapping sliding
windows, hidden states averaged across covering windows and log-probabilities
averaged in log space. All features are cached to `data/processed/`.

Optionally the ~22 external prior columns are appended, each accompanied by an
`is_missing_*` indicator (see §7.2). The leakage-safe prior set is defined once,
in `src.transfer.TRANSFER_PRIOR_COLS` — DMS-derived columns and the
functional-assay validation-only columns are deliberately absent.

### 4.2 Models

| Component | Where | Notes |
|---|---|---|
| Residual MLP head | `src/model.py` | `Linear(d→h)` → N × (`Linear → LayerNorm → GELU → Dropout` + residual) → `Linear(h→1)` |
| Branch / concat / GateWave fusion heads | `src/fusion.py` | frozen-branch projections to a shared dim; GateWave = sigmoid branch gate + softmax per-feature gating + GLU + residual, adapted from MVmamba |
| ESM-2 fine-tune classifier | `src/esm_finetune.py` | `mode="siamese"` (ProPath: separate WT/VT passes, pool at the mutated residue, `[h_wt ‖ h_vt ‖ Δ ‖ |Δ|]` head) and `mode="wt_site"` (CSBJ per-residue token classifier); `n_unfrozen_layers` = −1 full / 0 frozen / N last-N; gradient checkpointing available |
| MVmamba WT/VT features | `src/mvmamba_features.py` | global + local pooled features with a ±3-residue window, mutation-centred for long chains such as MSH6's 1,360 aa |

*Why four fine-tuning strategies rather than one:* the frozen linear probe is one
recipe among four in the literature, not automatically the best. All four
(MVmamba frozen pooled, VariPred linear probe, ProPath Siamese, CSBJ token
classifier) are implemented and benchmarked against each other on **identical**
MMR splits by `scripts/compare_finetune_strategies.py` before any is committed
to. The same scepticism is applied to the backbone itself:
`scripts/compare_backbones.py` compares ESM-1b against ESM2-650M with true
masked-marginal zero-shot scoring, on the explicit principle *don't assume ESM2
wins*.

### 4.3 Losses and optimisation

Ordinary BCE is the default. Weighted BCE uses a **fixed** class ratio computed
from the fitting partition (not recomputed per mini-batch). Focal loss remains
available for comparison with a neutral default `alpha=0.5`. AdamW + cosine
annealing; early stopping with best-weight restoration.

---

## 5. The evaluation layer — what was built and why

| Control | Implementation | What it prevents |
|---|---|---|
| **Residue-group disjointness** | `StratifiedGroupKFold` on `"{uniprot_id}:{position}"`, asserted at runtime | different substitutions at the same residue landing on both sides of a split |
| **Protein-level holdout** | `scripts/eval_leave_one_protein_out.py`; leave-one-gene-out for MMR | reporting a *new-residue-in-known-protein* number as if it were a *new-gene* number |
| **Sequence-cluster disjointness** | `scripts/build_cluster_split.py` (MMseqs2, 20% identity / 20% coverage) | paralogues and near-duplicate isoforms straddling a split |
| **Nested calibration** | outer validation fold untouched; outer-training group-split into model-fit / early-stop / calibration partitions | calibrators fitted on the fold they score (the cause of the impossible "isotonic ECE = 0") |
| **Clinical-only slices** | pooled OOF metrics reported for *all labels* and for the *ClinVar ∪ PG-clinical* slice separately | DMS assay-fitness performance being quoted as clinical performance |
| **Target-leakage diagnostics** | `scripts/diagnose_overfitting.py` | features that encode the label by construction (see §7.2) |
| **Bootstrap CIs + tuned thresholds** | `src/eval_utils.py`, 10,000-iteration percentile CIs; MCC-optimal threshold tuned on an inner slice only | point estimates on a few-thousand-row clinical set read as if precise |
| **Mandatory ablations** | `scripts/run_mmr_transfer.py`: ESM-only, priors-only, pretrained fused, concat fusion, GateWave — identical splits | a fusion architecture credited for gains that come from one branch alone |
| **Schema-drift abort** | checkpoints store exact prior-column order + ESM block dim; mismatch aborts | a transfer run that silently invalidates itself when the feature schema changed |

---

## 6. The MMR / Lynch-syndrome project

Lynch syndrome is caused by germline defects in the mismatch-repair genes, and
its VUS burden is exactly the clinical bottleneck this project targets. The
four-gene build adds domain-specific safeguards that a generic pipeline has no
reason to have:

1. **Pinned canonical references with hard-fail length checks** — MLH1 P40692 /
   756 aa, MSH2 P43246 / 934 aa, MSH6 P52701 / 1360 aa, PMS2 P54278 / 862 aa.
   A silent reference update cannot shift the numbering underneath a build.
2. **The PMS2 exon 11–15 pseudogene gate, fail-closed.** Short-read NGS calls
   inside the PMS2CL homology region are untrustworthy. The build **refuses to
   run** unless given one of: a homology CSV with `orthogonally_confirmed` flags,
   an explicitly verified `--pms2_codon_range`, or `--exclude_pms2`. Unconfirmed
   homology-region labels are withheld, never guessed. This is the clearest
   example of the project's general stance: *no supervision is better than
   supervision you cannot defend.*
3. **InSiGHT / ClinGen VCEP expert-panel tiering** (`expert_panel`,
   `evidence_tier`, `tier_weight`) so the highest-quality assertions can be
   weighted as such.
4. **Orthogonal functional evidence held out by construction.** MaveDB and CIMRA
   columns live in `FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS` and are never added to
   `TRANSFER_PRIOR_COLS`, so ESM-branch performance claims stay non-circular
   against that independent evidence — the same discipline already applied to
   ProteinGym DMS.
5. **Leave-one-MMR-gene-out evaluation** with bootstrap CIs, because the
   clinically relevant question for a rare-disease gene is *how does this
   transfer to a gene it has never seen*, not *how does it interpolate within a
   gene it knows*.

---

## 7. Improvements over the existing work

There are three distinct baselines this project improves on. Keeping them apart
matters, because "we improved X" means something different in each case.

### 7.1 Against the pre-existing codebase in this repository

Found by review before the data layer was extended (full detail:
`docs/CODE_REVIEW.md`).

**Bugs fixed**

| # | Location | Issue | Fix |
|---|---|---|---|
| B1 | `esm_extractor.validate_and_align` | `np.char.str_len(np.asarray(df["mut_aa"], dtype="U1")) == 1` truncates every value to ≤1 char — the mutant-residue check was **vacuous** | explicit per-row `len(m)==1 and m in VALID_AA` |
| B2 | `esm_extractor` | missing `VALID_AA` import after B1 | imported from `data_loader` (single source of truth) |
| B3 | `main.py` stage-2 cache | features cached under `{gene}_{model}_features.npz` regardless of appended prior columns — adding priors silently reused an incompatible cache | cache tag now includes `+extras{n}` |

**Design weaknesses addressed**

| # | Location | Weakness | Improvement |
|---|---|---|---|
| D1 | `data_loader._stream_gene_variants` | one full ~4 GB-decompressed ClinVar pass **per gene** — O(genes × minutes) | `_stream_genes_variants` serves all requested genes in **one** pass; single-gene API preserved as a wrapper |
| D2 | `dataset.make_position_group_folds` | grouping key was the raw `position`, so pooling proteins collapsed equal positions of *different* proteins into one leakage group | optional explicit `groups`; `train.run_pipeline` passes `"{uniprot}:{position}"` automatically |
| D3 | `loss.WeightedBCELoss` | `pos_weight` recomputed per batch → noisy gradients on small final batches | fixed ratio from the fitting partition |
| D4 | `esm_extractor._embed_spans` | dead if/else branch accumulating identically in both arms | collapsed |
| D5 | `esm_extractor.extract_features_cached` | GPU memory freed only after npz compression | extractor deleted before compression |

Behaviours that were reviewed and found **correct** (per-fold temperature
scaling as an intentional documented benchmark caveat; the single-forward-pass
PLLR trick; the `StratifiedGroupKFold` partition assertion; the logging
hierarchy) were left unchanged and recorded as such — a review that only lists
findings gives no information about what was actually checked.

**Capability added on top:** the multi-source data layer
(`src/external_datasets.py`, `src/extended_builder.py`), the audit, the MMR
project, transfer learning, fusion heads, real backbone fine-tuning, the
evaluation utilities, cross-platform one-command runners, and the test suite
(60 tests, all passing as of 2026-08-26).

### 7.2 Against this project's own earlier versions — the four findings that changed the numbers

These are the corrections that materially changed what the model was learning or
what the reported metric meant. They are the substance of the project's
scientific hygiene claim.

**(a) Label inversion — ~185k labels were backwards.**
ProteinGym's `DMS_score_bin=1` marks the *top half of assay fitness* (i.e.
**tolerated**), but the builder mapped it straight to `label=1` ("pathogenic").
Two independent anchors confirmed the inversion:
`P(AlphaMissense ≥ 0.9 | bin=1) = 0.217` vs `0.506` for `bin=0`; and on
PG-clinical rows, mean DMS score −0.8 for pathogenic vs +1.9 for benign. Fixed in
`src/extended_builder.py` (DMS label is now `1 − dms_bin`), mirrored in the
auditor, dataset rebuilt and re-audited. **Cross-validated AUC did not change**
(AUC is flip-invariant once the head re-learns the mapping) — which is precisely
why this class of bug survives ordinary metric-watching. The *semantics* and any
prospective risk ranking were wrong until it was fixed.

**(b) Target leakage — the 0.9987 AUC was circular.**
The first run reported ROC-AUC **0.9987** on all labels.
`scripts/diagnose_overfitting.py` showed that on single-assay rows (97.3% of
labelled data) the feature `dms_bin_median` equals `1 − label` **by
construction** — the answer was in the feature matrix. With the leaky features
removed the honest all-labels number is **0.716**. Separately, the same
diagnostic confirmed there was **no classic overfitting**: train-vs-val AUC gap
per fold ≈ 0.000 (−0.001 … +0.011). `--no_dms_features` reproduces the clean
configuration.

**(c) In-fold calibration — calibration metrics were optimistic.**
Temperature and isotonic calibrators were fitted on each outer validation fold
and then scored on that same fold. An isotonic ECE of exactly zero was the
tell. Calibration is now **nested**: the outer validation fold is untouched, and
both calibrators are fitted on a dedicated calibration partition carved by
residue group out of the outer-training data. This costs fitting data per fold
and buys valid out-of-fold calibration, Brier, ECE and threshold metrics.

**(d) The objective was optimising the wrong target.**
Only 5,152 of 190,494 labelled rows (**2.7%**) carry a clinical label; ~188k are
DMS aggregates. A loss treating rows equally optimises assay fitness, not
pathogenicity. Fixed with a configurable `--clinical_weight` (default 5.0) and
early stopping monitored on **clinical-only** ROC-AUC whenever both classes are
present. The prior default `focal_alpha=0.25` was also wrong for this data —
positives are 43% of all labels and ~52% of clinical labels — so it was
down-weighting the positive class; the neutral 0.5 is now the default and plain
BCE is the default loss.

**Three further silent-failure fixes of the same family:**

- **Missingness was indistinguishable from a median prediction.** Most `zs_*`
  columns are absent for ~99% of rows; median imputation without an indicator
  made "not scored" look like a real median-valued prediction. Every numerical
  prior now carries an `is_missing_*` flag.
- **`esm+priors` was not actually appending priors.** The extraction pool
  retained only metadata columns, so the prior matrix had **zero** columns and
  the advertised external priors never reached the model. The pool now carries
  raw prior columns for labelled and VUS rows alike (the changed cache width
  intentionally forces a new feature-cache key).
- **gnomAD features were joined but invisible.** AF columns were merged onto the
  MMR table but never added to `src.transfer.TRANSFER_PRIOR_COLS`, so the prior
  branch ignored them. Fixed, and the class of bug was closed permanently: a
  regression test,
  `tests/test_new_data_sources.py::test_new_prior_columns_are_wired_into_transfer_prior_cols`,
  now fails if a new source is joined without being wired through.

**(e) Three silent-failure classes found in the 2026-08-27 review.**
These were found by reading the code rather than by a metric moving, which is
the point: none of them would have shown up as a bad number.

- **The feature cache validated only its row count.** `extract_features_cached`
  compared `len(meta)` against `len(df)` and nothing else, while every caller
  relies on the returned features being *positionally aligned* with the request
  and reads prior columns straight off the cached metadata. A variant table
  rebuilt with the same row count in a different order silently returned another
  variant's embeddings — no error, no metric change, wrong features. Cache hits
  are now validated by variant key.
- **Prior-imputation constants were recomputed at every stage.** Stage 1
  pretrained the head on the 80-gene panel's medians; stage 2 warm-started it
  onto the four MMR genes' medians. Because `zs_*` columns are missing for ~99%
  of rows, the imputed constant *is* the feature for almost every variant, so
  warm-starting was carrying weights onto a differently-centred input. The
  constants are now persisted in the checkpoint alongside the column order and
  reused, with a warning when an older checkpoint lacks them.
- **A half-built virtualenv counted as ready.** `ensure_environment` gated on the
  interpreter existing. `python -m venv` finishes in a second while the
  dependency install takes minutes, so any interrupted install left an
  environment that every later run silently adopted and then failed inside a
  stage with an ImportError. Readiness is now gated on a stamp file written only
  after a successful install, which also forces a reinstall when switching
  between the CPU and CUDA requirement sets.

The first two are the same failure family as the rest of §7.2: a value that is
wrong but well-formed, in a place no metric looks. Each now has a regression
test (64 unit tests, all passing as of 2026-08-27).

All build-time failures encountered while bringing the pipeline up — the
entry-name-vs-accession mismatch that silently produced a 0-row DMS panel, the
`THREE_TO_ONE`/`ONE_TO_THREE` inversion that NaN'd every `hgvs_p`, the
dedupe-before-backfill ordering that left 29,327 duplicate rows, and nine others
— are recorded with root cause and fix in `docs/RUN_REPORT.md` §5.

### 7.3 Against the published external work

| Prior work | What this project does differently |
|---|---|
| **AlphaMissense** (single general prior) | used as *both* a feature and the baseline to beat. On the leakage-clean clinical slice the fused head reaches **0.963 ROC-AUC / 0.78 MCC** vs AlphaMissense's **0.945 / 0.75** — a **+1.8 pt AUC** gain from combining AM with 17 published zero-shot scores and domain flags, **without DMS features and without ESM**. |
| **ProteinGym zero-shot leaderboard models** (EVE, ESM-1b, GEMME, …) | all 17 are ingested as features *and* kept as baselines on identical splits, so any claimed gain is measured against the actual published state of the art rather than against a weak internal baseline. |
| **VariPred** (frozen ESM linear probe) | implemented as one of four benchmarked strategies rather than assumed; its asymmetric mutation-centred window recipe is reused for long chains. |
| **ProPath** (Siamese WT/VT backbone fine-tuning) | implemented (`mode="siamese"`, backbone LR 1e-5 / batch 8 / 10 epochs) and benchmarked head-to-head against the other three recipes on identical MMR splits. |
| **CSBJ per-residue token classifier** | implemented (`mode="wt_site"`) and benchmarked in the same comparison. |
| **MVmamba** (WT/VT global+local pooled features, GateWave fusion, AF ablation) | frozen feature recipe reimplemented; the GateWave gated-fusion head adapted to two modality vectors and benchmarked against plain concat fusion; MVmamba's own allele-frequency ablation (AUC 0.895→0.901) motivated wiring gnomAD AF into the **broad pretraining panel**, not just the MMR genes. |
| **Typical single-source variant-effect papers** | this pipeline unifies six-plus sources under one validated key with SHA-256 provenance, explicit label precedence, quarantined cross-source conflicts, and an independent 12-check audit — the merge itself is a reviewable artefact, not an unlogged preprocessing step. |
| **Typical residue-level CV reporting** | protein-level (leave-one-protein-out / leave-one-gene-out) and MMseqs2 cluster-disjoint splits are provided as first-class evaluations, because residue-grouped CV answers a strictly easier question than deployment to a new disease gene. |
| **Functional-assay usage in the literature** | MaveDB and CIMRA evidence is deliberately **withheld from training** and used only for validation, so agreement with orthogonal experimental evidence is a real check rather than a restatement of a training input. |

---

## 8. Results, and how to read them honestly

**Leakage-clean, priors-only mode, 80 genes, 3-fold group-disjoint CV**
(from `docs/TRAINING_NOTES.md` §2.2 / `docs/RUNLOG.md`, run 2026-08-23):

| Slice | Model | ROC-AUC | MCC |
|---|---|---|---|
| all labels (190k) | MLP + isotonic | 0.716 | 0.32 |
| all labels | AlphaMissense | 0.709 | 0.30 |
| **clinical only** | **MLP + isotonic** | **0.963** | **0.78** |
| clinical only | AlphaMissense | 0.945 | 0.75 |

How to read these:

- **0.716 on all labels is the honest number.** The task "predict assay-fitness
  bins without seeing the assay" is intrinsically hard; the earlier 0.9987 was
  circularity (§7.2b), not learning.
- **0.963 / 0.78 on the clinical slice is the meaningful headline**, and it is a
  *new-residue-in-known-protein* estimate — the outer folds group by
  `uniprot_id:position`, not by protein. It is **not** an unseen-gene number.
  The protein-level evaluation exists (`eval_leave_one_protein_out.py`) but its
  result is not yet recorded here.
- **Calibration is imperfect and stated as such:** temperature scaling moved ECE
  23% → 22%; isotonic reached 11%. Fixed-threshold metrics (MCC@0.5) are
  sensitive to score shift — trust AUC and the calibrated variants.
- **AlphaMissense was itself trained on ClinVar**, so temporal contamination of
  the clinical slice is possible. `-DMS_score` as a baseline is likewise
  circular on the DMS slice; only its clinical-slice value is interpretable.
- These numbers predate the changes in §7.2(c)–(d) being evaluated. The nested
  calibration, clinical weighting, loss defaults and capacity changes are
  **starting values, not claimed improvements, until re-measured.**

### Verification status on this machine, 2026-08-27

| Item | Status |
|---|---|
| Unit tests | **64 passed** (`python -m pytest tests/ -q`), 4.8 s |
| `data/processed/`, `data/mmr/` | **absent** — the artefacts behind the table above are not on disk and are gitignored by design |
| `data/raw/` | partial: ProteinGym bundles, UniProt panel JSONs, MaveDB score sets present; ClinVar `variant_summary.txt.gz` and AlphaMissense **not** present |
| Reproducing the numbers | requires a full rebuild (§9): ~1.27 GB of downloads plus a ~7-minute one-off AlphaMissense scan of 216M lines |
| `PROJECT_PLAN.md` | referenced throughout the docs and module docstrings but **not present in the repository** — the phase numbering below is otherwise unanchored |

---

## 9. Reproduction

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt        # or requirements-cuda.txt on NVIDIA

# Preview every stage's exact command before committing to a multi-hour run
python run_mmr_pipeline.py --dry_run

# CPU-friendly smoke run: priors-only features, no backbone fine-tuning
python run_mmr_pipeline.py --no-full_finetune

# Full run on a GPU box
python run_mmr_pipeline.py --features esm+priors \
    --esm_model facebook/esm2_t33_650M_UR50D --all_sources

# Stage 2b on CPU anyway (smoke test only -- tiny checkpoint, shallow
# unfreeze, one gene). Without --allow_cpu_finetune the pipeline stops
# before stage 2b when no CUDA/MPS device is visible.
python run_mmr_pipeline.py --allow_cpu_finetune \
    --esm_model facebook/esm2_t12_35M_UR50D --n_unfrozen_layers 2 \
    --eval holdout --holdout_gene MLH1
```

The broad-panel workflow is unchanged and still available via
`bash run_pipeline.sh` / `.\run_pipeline.ps1` / `python run_pipeline.py`, or
stage by stage as documented in `README.md`.

**GPU note:** ESM-2 extraction is ~20× faster on CUDA and is required in
practice for `--features esm+priors` over large panels. Device selection is
automatic (CUDA → MPS → CPU) and forward passes run under fp16 autocast on GPU.
Keep `--extract_batch_size` small (default 8); raise the training `--batch_size`
freely.

---

## 10. Known limitations and open work

**Scientific limits that no amount of engineering removes**

- No merger can establish clinical truth from heterogeneous public evidence.
  DMS labels remain functional-assay proxies.
- AlphaMissense and the supervised `zs_*` predictors may overlap the training
  data of the clinical benchmark. A temporally external or genuinely held-out
  clinical set is required before any prospective clinical claim.
- The PMS2 pseudogene region stays unresolved until orthogonal confirmation data
  is supplied; the current safe default excludes PMS2 entirely.

**Open engineering work**

1. Run and record **leave-one-protein-out** clinical evaluation — the preferred
   model-selection criterion for deployment to new disease genes.
2. Run the ablation battery and record it: AlphaMissense only; all priors; all
   priors minus each source family; ESM/PLLR only; ESM + priors — with feature
   availability reported alongside performance for the sparse `zs_*` columns.
3. **Multitask / curriculum training**: continuous, within-assay-normalised DMS
   fitness as one task and high-confidence clinical pathogenicity as another;
   fine-tune the clinical head after DMS pretraining rather than binarising DMS
   into the same target. Needs a new head and data interface.
4. Split ClinVar provenance into a high-confidence expert-panel set and a
   lower-confidence down-weighted set rather than one star threshold.
5. Full-model GPU run (`--features esm+priors`, `t33_650M`) with
   `--no_dms_features`; then score all ~966k unlabelled substitutions
   (including the 25k ClinVar VUS) into calibrated risk tiers.
6. Close the two issues left open by the 2026-08-27 review (detail and
   suggested fixes in `docs/CODE_GUIDE.md` §10): prior-imputation medians are
   still computed over the whole stage table rather than the fitting partition
   (a mild transductive leak), and `scaler_mean`/`scaler_scale` are written into
   every transfer checkpoint but never read — wire them into warm-starting or
   drop them, because dead payload in a schema contract invites false
   assumptions.
7. Try the two cheapest untested wins named in `docs/CODE_GUIDE.md` §8:
   gradient-boosted trees on the priors branch (the branch that currently
   produces the project's best number, on ~22 mostly-missing tabular columns —
   the model class that suits that data has not been tried), and LoRA or a
   shallow unfreeze instead of a full 650M-parameter backbone fine-tune against
   ~5k clinical labels.
8. Later plan phases not yet started: **Phase 4** LLM clinical-text branch
   (BioBERT + ClinVar-BERT preprocessing), **Phase 5** fusion of that branch into
   GateWave (the ESM/priors fusion heads already exist and are benchmarked),
   **Phase 6** gene-specific calibration.

**Repository hygiene**

- The recent commit history contains several placeholder commit messages
  (git's own "Please enter the commit message…" template text). Squash or
  amend before this history is shared.
- `PROJECT_PLAN.md` is cited by name across the docs and module docstrings but
  is not in the repository; it should be committed or the references rewritten.
- The `--full_finetune`-by-default switch, the accelerator preflight, and the
  three fixes in §7.2(e) are in the working tree and not yet committed.

---

## 11. Where everything lives

### Documents

| Doc | Contents |
|---|---|
| `README.md` | quickstart, per-stage commands, architecture diagram, repo layout |
| `docs/PROJECT_DOCUMENTATION.md` | **this file** — what/why/improvements, the single entry point |
| `docs/CODE_GUIDE.md` | **how it works** — module mechanics, the merge step by step, and the better alternative for every design decision |
| `docs/DATA_TO_MODEL.md` | the dataset schema and the exact path from a CSV row to a tensor, for all three training paths; what the model never sees, and what is actually verified |
| `docs/DATASETS.md` | exhaustive per-source reference: URLs, versions, licences, schemas, processing rules, caveats (§1–19) |
| `docs/CODE_REVIEW.md` | pre-existing-code review findings + complete changelog |
| `docs/DATA_PIPELINE_HARDENING.md` | ingestion, merge, provenance and conflict-quarantine guarantees |
| `docs/FINE_TUNING_FINDINGS.md` | evidence from saved OOF artefacts, the training-safeguard changes, recommended experiment order |
| `docs/TRAINING_NOTES.md` | data expansion 10→80 genes, the label-inversion bug, the DL stage, results |
| `docs/MMR_TRANSFER_WORKFLOW.md` | the MMR pipeline stage by stage |
| `docs/RUN_REPORT.md` | minute-by-minute build record: commands, checksums, 13 failures and their fixes |
| `docs/RUNLOG.md` | one entry per build/training run, newest first |

### Code

| Path | Role |
|---|---|
| `run_mmr_pipeline.py` (+ `.sh`/`.ps1`) | **recommended entrypoint** — download → clean → train for the MMR project |
| `run_pipeline.py` (+ `.sh`/`.ps1`), `main.py` | original broad-panel workflow and single-/multi-gene training CLI |
| `src/data_loader.py` | ClinVar streaming, HGVS-p parsing, star filtering, multi-gene single-pass |
| `src/external_datasets.py` | resumable checksummed downloader; ProteinGym / AlphaMissense / UniProt parsers; RefSeq→UniProt exact-sequence mapper |
| `src/extended_builder.py` | multi-source merge, label precedence, conflict quarantine, manifest |
| `src/gnomad.py` · `src/structure.py` · `src/interpro.py` · `src/mavedb.py` · `src/cimra.py` | per-source adapters (AF + constraint, AlphaFold pLDDT, InterPro, MaveDB REST, CIMRA OddsPath) |
| `src/mmr_dataset.py` | pinned MMR references, VCEP tiers, PMS2 pseudogene gate, functional-assay attach |
| `src/esm_extractor.py` · `src/mvmamba_features.py` · `src/esm_finetune.py` | frozen embeddings + PLLR · MVmamba WT/VT + masked-marginal scorers · backbone fine-tuning |
| `src/model.py` · `src/fusion.py` · `src/loss.py` | residual MLP head · branch/concat/GateWave fusion · focal & weighted BCE |
| `src/dataset.py` · `src/train.py` · `src/calibration.py` · `src/eval_utils.py` · `src/transfer.py` | group-disjoint splits · CV loop · calibrators & reliability plots · bootstrap CIs & threshold tuning · two-stage transfer |
| `scripts/` | dataset builds, audit, pretraining, transfer, fine-tune benchmarks, backbone comparison, LOPO eval, cluster split, overfitting diagnostic |
| `tests/` | 60 unit tests over parsing, splitting, MMR/gnomAD/MaveDB/CIMRA logic, ESM fine-tuning, and prior-column wiring |
