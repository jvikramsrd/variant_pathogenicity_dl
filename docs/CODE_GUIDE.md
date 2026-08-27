# Code Guide — what each part does, why it works that way, and what could replace it

**Audience:** someone who has to change this code, review it, or defend a number
it produced. `docs/PROJECT_DOCUMENTATION.md` answers *what was built and why it
matters*. This file answers *how it actually works*, walks through the data
merge line by line, and — for every significant design decision — names the
alternative that could be better and the condition under which you should
switch to it.

Verified against the tree on 2026-08-27. Line references are given as
`file.py:line` and will drift; the function names will not.

---

## Table of contents

1. [The one-paragraph mental model](#1-the-one-paragraph-mental-model)
2. [Execution map: what actually runs, in order](#2-execution-map-what-actually-runs-in-order)
3. [The data layer: from six web sources to one row per variant](#3-the-data-layer-from-six-web-sources-to-one-row-per-variant)
4. [The merge, step by step](#4-the-merge-step-by-step)
5. [The feature layer](#5-the-feature-layer)
6. [The model and training layer](#6-the-model-and-training-layer)
7. [The evaluation layer](#7-the-evaluation-layer)
8. [Design decisions and better alternatives](#8-design-decisions-and-better-alternatives)
9. [Invariants you must not break](#9-invariants-you-must-not-break)
10. [Known issues, ranked](#10-known-issues-ranked)

---

## 1. The one-paragraph mental model

Everything in this repository is one of four things: **an adapter** that turns
some public database into a dataframe keyed by
`(uniprot_id, position, wt_aa, mut_aa)`; **a merge** that joins those dataframes
into one master table and decides which source gets to define the training
label; **a featuriser** that turns a row of that table into a numeric vector
(either external prior scores, or ESM-2 hidden states, or both); or **a trainer**
that fits a classifier on those vectors under a split designed so the reported
number is not a lie. The hard part of this project is not the model. It is
keeping the key honest through the merge, and keeping the split honest through
the evaluation.

---

## 2. Execution map: what actually runs, in order

`run_mmr_pipeline.py` is the recommended entrypoint. It is a **process
orchestrator**, not a library: every stage is a subprocess running a script
under `scripts/`, so a stage can be re-run by hand with the exact command the
banner printed, and a crash in stage 5 does not lose stages 1–4.

```
run_mmr_pipeline.py
│
├─ ensure_environment()              run_pipeline.py:61   venv + requirements (stamped)
├─ detect_accelerator(py)            run_mmr_pipeline.py  probes the CHILD interpreter
│                                                         → aborts before stage 2b on CPU
│
├─ stage: unit tests                 tests/*.py
├─ stage: build broad panel          scripts/build_extended_dataset.py
│                                    └─ src/extended_builder.build_extended_dataset()
├─ stage: audit broad panel          scripts/audit_extended_dataset.py   (12 checks)
├─ stage: build MMR panel            scripts/build_mmr_dataset.py
│                                    └─ same builder, 4 genes + MMR-only safeguards
├─ stage: TRAIN 1 — pretrain         scripts/pretrain_esm_80.py
│                                    frozen features → head → checkpoint .pt
├─ stage: TRAIN 2 — transfer         scripts/run_mmr_transfer.py
│                                    warm-start head, leave-one-gene-out + ablations
└─ stage: TRAIN 2b — backbone FT     scripts/finetune_esm_mmr.py
                                     gradients into ESM-2 itself (default; needs GPU)
```

**Why subprocesses rather than function calls.** Stages have incompatible
resource profiles — a build is network- and disk-bound for tens of minutes, a
fine-tune is GPU-bound for hours — and ESM-2 checkpoints hold a lot of GPU
memory that is far easier to reclaim by letting the process exit than by
chasing references. The cost is that data crosses stages as files, not objects,
which is exactly why the schema-drift and cache-staleness checks in §9 exist.

**Stage numbering** is computed from your flags (`StageCounter`), so
`Stage 3/4` always means three of four *real* stages are done. Skipped stages
print an unnumbered `SKIPPED` line.

**The preflight.** Stage 2b is on by default and is the one stage that can turn
an afternoon into a week. `detect_accelerator()` probes the child interpreter —
not this one, because after `ensure_environment()` the stages run in a venv that
may have a different torch build — and the pipeline stops *before* the
downloads and frozen-embedding stages if no CUDA/MPS device is visible.
`--allow_cpu_finetune` overrides; `--no-full_finetune` skips the stage.

> There is a second, older entrypoint: `run_pipeline.py` / `main.py`, the
> broad-panel workflow with K-fold CV and calibration. It is untouched by the
> MMR work and shares the same `src/` primitives.

---

## 3. The data layer: from six web sources to one row per variant

### 3.1 The contract every adapter obeys

| Requirement | Enforced where |
|---|---|
| Emit `(uniprot_id, position, wt_aa, mut_aa)` | each adapter |
| `wt_aa` must equal the canonical UniProt residue at `position` | `extended_builder._wt_ok_mask`, `esm_extractor.validate_and_align` |
| Non-UniProt identifiers resolve by **sequence**, never by string | `external_datasets.map_np_to_panel` |
| Every downloaded byte is checksummed and recorded | `external_datasets.download_file`, `sha256_of` |
| Rows that fail validation are **dropped and counted**, never coerced | `BuildStats` → `manifest.json` |

That last row is the load-bearing one. A dropped row costs you supervision you
can recover by fixing the adapter. A wrongly joined row teaches the model that
a benign variant is pathogenic and is invisible in every downstream metric.

### 3.2 The adapters

| Module | Source | The interesting mechanic |
|---|---|---|
| `data_loader.py` | ClinVar `variant_summary.txt.gz` | Streams a ~4 GB-decompressed TSV. `_stream_genes_variants` serves **all** requested genes in one pass — the single-gene version was O(genes × minutes). `parse_protein_substitution` reads the HGVS-p; `stars_for_review` maps review status → 0–4 stars. |
| `external_datasets.py` | ProteinGym v1.3, AlphaMissense, UniProt | `download_file` stages to `.part` and promotes atomically only after size + checksum validation, so a server ignoring an HTTP Range request cannot corrupt a resumed download. `stream_filter_alphamissense` scans 216M lines once and caches the panel subset. `map_np_to_panel` matches RefSeq `NP_` → UniProt by **exact whole-protein sequence equality**, which simultaneously guarantees identical residue numbering. |
| `gnomad.py` | gnomAD v4 GraphQL | Variant AF → `gnomad_log10_af` plus ACMG `BA1`/`BS1`/`PM2` flags; gene-level constraint (pLI, o/e, z) broadcast per gene. |
| `structure.py` | AlphaFold DB | Per-residue pLDDT + a disorder flag. Structural confidence without a structure pipeline. |
| `interpro.py` | InterPro | Domain/family/superfamily intervals → `in_interpro_domain`. |
| `mavedb.py` | MaveDB REST | MMR functional assays. **Validation only.** |
| `cimra.py` | user-supplied CSV | Calibrated ACMG OddsPath. **Validation only.** (No bulk CIMRA API exists.) |

---

## 4. The merge, step by step

This is the part the rest of the project rests on. It lives in
`src/extended_builder.assemble_master()` (`src/extended_builder.py:509`), and it
runs in a deliberate order. Sequence normalisation and de-duplication happen
first; label resolution happens *last*, after every source has had its say.

### Step 0 — the panel defines the coordinate system

`resolve_panel()` fetches each gene's UniProt accession and canonical sequence.
Every subsequent operation is expressed in *this* numbering. Nothing else is
trusted to agree.

For the MMR build, the four references are additionally **pinned with hard-fail
length checks** (MLH1 P40692/756 aa, MSH2 P43246/934 aa, MSH6 P52701/1360 aa,
PMS2 P54278/862 aa) in `src/mmr_dataset.py`, so a silent upstream reference
update cannot shift numbering underneath a rebuild.

### Step 1 — build the row universe (a union, not a join)

```python
union = pd.concat([
    _base(labeled,   "clinvar"),      # ClinVar P/LP/B/LB
    _base(vus,       "clinvar_vus"),  # ClinVar VUS — scored, never trained on
    _base(dms_panel, "dms"),
    _base(clin_panel,"pg_clinical"),
    _base(am_panel,  "alphamissense"),
], ignore_index=True)
```

**Why a union.** The output must contain every substitution *any* source knows
about — including ~966k unlabelled ones that exist only to be scored. Starting
from one source and left-joining the rest would silently make that source's
coverage the ceiling for the whole project.

### Step 2 — derive identity **before** de-duplicating

```python
master["gene"]   = canonical_gene.fillna(master["gene"])       # panel is authority
master["hgvs_p"] = _render_hgvs_p(wt_aa, position, mut_aa)     # always regenerated
master = master.loc[~(incomplete_key | synonymous)]            # drop non-variants
master = master.drop_duplicates(subset=MASTER_BASE_COLS, keep="first")
```

**Why the order matters.** AlphaMissense rows arrive without a `gene`. Dedupe
first and an AlphaMissense row survives as a *twin* of its ClinVar counterpart —
same variant, two rows, one of them missing metadata — which then breaks the
uniqueness invariant at export. Getting this backwards once left 29,327
duplicate rows (`docs/RUN_REPORT.md` §5).

**Why derived and not merely backfilled.** `gene` and `hgvs_p` sit inside
`MASTER_BASE_COLS`, the de-duplication key — but they *describe* a variant
rather than identify it, and `MASTER_KEY` is what identifies it. Letting each
source keep its own spelling meant a gene alias or a different HGVS rendering
(`p.A10C` vs `p.Ala10Cys`) produced two rows for one variant, which then
collided on `MASTER_KEY` and aborted the build. Deriving both makes
de-duplication on `MASTER_BASE_COLS` equivalent to de-duplication on
`MASTER_KEY` by construction, rather than by every adapter's good behaviour.

### Step 3 — resolve each source's own internal contradictions, separately

`_resolve_binary_evidence()` (`src/extended_builder.py:257`) is applied
**per source, before any cross-source comparison**:

```python
n_labels = labeled.groupby(key)[label_col].nunique()
conflicts = n_labels[n_labels > 1]        # both 0 and 1 for the same variant
# → excluded from that source's supervision entirely, and returned separately
```

Duplicates that *agree* are reduced deterministically by a priority column
(review stars for ClinVar). A key carrying **both** labels is never resolved by
archive order — it is removed from that source's evidence and reported.

**Why not just take the highest-star row.** Archive order is not evidence.
"ClinVar contains both a pathogenic and a benign two-star assertion for this
variant" is a real, informative state, and picking a winner by row order
destroys it while producing a confident-looking label.

### Step 4 — attach every source by left join onto the key

In order: ClinVar labels + stars + review status → ProteinGym clinical →
DMS aggregation → AlphaMissense → zero-shot models → gnomAD AF → gene
constraint → AlphaFold pLDDT → InterPro → UniProt functional sites → UniProt
domains.

Two mechanics are worth noting:

- **DMS is aggregated, not exploded** — median score, median bin, assay count
  and the pipe-joined assay ids per key. Per-assay detail is preserved in a
  separate `dms_scores_long.csv`, so the master stays one row per variant while
  nothing is lost.
- **Every optional source has an explicit `else` branch** that creates its
  columns full of `NaN`. The master's schema therefore does not depend on which
  flags you passed — only the values do. This is what lets a checkpoint's stored
  column order stay meaningful across builds.

The interval-valued sources (UniProt domains, InterPro) are joined by
**position containment**, not equality: distinct positions are joined to
intervals, matches are grouped and pipe-joined into a names string, and presence
becomes a binary flag.

### Step 5 — resolve the label, last

```python
dms_single_ok = (dms_bin_median.notna()) & (n_dms_assays == 1) & dms_bin_median.isin([0, 1])
dms_pathogenic = 1 - master["dms_bin_median"]        # ← the flip

master["label"] = np.where(clinvar_label.notna(), clinvar_label,
                  np.where(clinical_label.notna(), clinical_label,
                  np.where(dms_single_ok, dms_pathogenic, np.nan)))
```

Three things are happening:

1. **Precedence** — ClinVar > ProteinGym-clinical > single-assay DMS. Curated
   clinical assertion beats curated clinical benchmark beats functional proxy.
2. **The flip.** ProteinGym's `DMS_score_bin=1` marks the **top half of assay
   fitness**, i.e. *tolerated*. Mapping it straight to `label=1` inverted ~185k
   labels. Note that cross-validated AUC did **not** move when this was fixed —
   AUC is flip-invariant once the head re-learns the mapping — which is exactly
   why the bug survived ordinary metric-watching. It was caught by two external
   anchors: `P(AlphaMissense ≥ 0.9 | bin=1) = 0.217` vs `0.506` for `bin=0`, and
   mean DMS score −0.8 for pathogenic vs +1.9 for benign on PG-clinical rows.
3. **`n_dms_assays == 1`.** A variant covered by multiple assays gets **no**
   DMS-derived label at all, even when the assays agree. See §8 for why, and
   for the better alternative.

### Step 6 — quarantine cross-source disagreement

```python
cross_source_conflict = clinvar_label.notna() & clinical_label.notna() & (clinvar_label != clinical_label)
master["label_conflict"] = clinvar_conflict | clinical_conflict | cross_source_conflict
master.loc[master["label_conflict"] == 1, "label"] = np.nan
master.loc[master["label_conflict"] == 1, "label_weight"] = 0.0
```

The raw source columns stay in the CSV for investigation; the *resolved target*
is withheld. A conflicted variant is structurally incapable of entering
training.

**Why not majority-vote or trust-the-higher-tier.** Heterogeneous public
evidence cannot establish clinical truth. Making the disagreement explicit and
unusable is honest; picking a winner manufactures confidence that the underlying
data does not support.

### Step 7 — provenance and weights

Every labelled row carries `label_source` (`clinvar` / `pg_clinical` / `dms`)
and `label_weight` (ClinVar by review stars 0.5–1.0; PG-clinical 0.75; DMS 0.20),
which `transfer.stage_sample_weights()` later multiplies by the experiment's
clinical-vs-DMS weighting. A pipe-joined `sources` string and `n_sources` count
record which of the ten source families touched each row.

### Step 8 — refuse to export a table that violates the invariants

`validate_master_for_export()` (`src/extended_builder.py:810`) raises on: null
or duplicate keys, unresolved `gene`/`hgvs_p`, non-binary labels, a conflicted
row that still carries a label, or a label without a recognised source.

**Why a separate function.** It is the last gate before an artefact leaves the
process and becomes something another stage trusts. Inline assertions get
disabled, moved, or quietly deleted; a named export gate does not.

### Step 9 — audit it again, from scratch

`scripts/audit_extended_dataset.py` **re-derives the merge independently** and
runs 12 checks (key uniqueness, wt-residue validation, label-precedence
re-derivation, provenance coherence, cross-source conflicts, feature coverage)
into `audit_report.json`. A builder that validates itself only proves it is
self-consistent. On the 80-gene build it passes 12/12, and the 1,919 variants
covered by both ClinVar and ProteinGym-clinical agree 100%.

### 4.1 What the MMR build adds

`scripts/build_mmr_dataset.py` calls the *same* builder restricted to four
genes, then layers on MMR-specific safeguards:

- **VCEP tiering** (`add_evidence_tiers`) — InSiGHT/ClinGen expert-panel
  assertions get `expert_panel`, `evidence_tier`, `tier_weight`.
- **The PMS2 pseudogene gate, fail-closed** (`apply_pms2_homology_gate`,
  `src/mmr_dataset.py:161`). Short-read calls inside the PMS2CL homology region
  (exons 11–15) are untrustworthy. The build **refuses to run** unless given a
  homology CSV with `orthogonally_confirmed` flags, an explicitly verified
  `--pms2_codon_range`, or `--exclude_pms2`. Unconfirmed labels are withheld,
  never guessed.
- **Orthogonal evidence attached but fenced off** (`attach_mavedb`,
  `attach_cimra`). These columns live in
  `FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS` and are deliberately absent from
  `transfer.TRANSFER_PRIOR_COLS`, so agreement with them remains a real
  check rather than a restatement of a training input.

---

## 5. The feature layer

### 5.1 Frozen ESM-2 embeddings + PLLR (`src/esm_extractor.py`)

For variant `(i, wt → mut)` in sequence `X`:

```
z = [ h_wt ‖ h_mut ‖ (h_mut − h_wt) ‖ |h_mut − h_wt| ‖ PLLR_i ]      →  4d + 1
PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)
```

**How it is computed cheaply.** Both log-probabilities are conditioned on the
*same* masked context, so both come from a **single** forward pass over the
wild-type sequence — the Meier et al. (2021) masked-marginal trick. The mutant
hidden states need their own passes, but only over **unique `(position, mut_aa)`
pairs**, then gathered back to rows. For a gene with a few thousand variants
that is the difference between one pass and thousands.

**Long sequences.** ESM-2's positional capacity is 1,022 residues. Longer chains
use overlapping sliding windows (`_sliding_spans`, 256-residue overlap), hidden
states averaged across covering windows and log-probabilities averaged **in log
space**. MSH6 at 1,360 aa needs this.

**Caching.** `extract_features_cached` writes `{gene}_{model}[+extrasN]_features.npz`
plus a metadata CSV. The cache tag includes the checkpoint name and the
appended-prior count, so adding priors cannot silently reuse an incompatible
cache. A cache hit is now validated by **variant key**, not row count — see §10.

### 5.2 External priors (`src/transfer.prior_matrix`)

~22 numeric columns (`am_pathogenicity`, `in_domain`, gnomAD AF + ACMG flags,
gene constraint, AlphaFold pLDDT, InterPro, functional site, plus every `zs_*`
published-model score), each accompanied by an **`is_missing_*` indicator**.

**Why the indicators are not optional.** Most `zs_*` columns are absent for ~99%
of rows. Median-imputing without a flag makes "this model never scored this
variant" indistinguishable from "this model scored it at exactly the median" —
the model cannot tell absence from a real, confident, mid-range prediction.

**The leakage-safe set is defined once**, in `TRANSFER_PRIOR_COLS`
(`src/transfer.py:58`). DMS-derived columns and the validation-only functional
assay columns are deliberately absent. A regression test
(`tests/test_new_data_sources.py::test_new_prior_columns_are_wired_into_transfer_prior_cols`)
fails if a new source is joined onto the table without being wired through —
that class of bug (gnomAD AF joined but invisible to the model) happened once
and is now closed permanently.

**Imputation constants are persisted across stages.** `prior_impute_values()`
records the fill value per column; stage 1 stores it in the checkpoint and
stage 2 reuses it. Since the imputed constant *is* the column for ~99% of rows,
recomputing medians per stage meant the head was pretrained on the 80-gene
panel's centre and warm-started onto the four MMR genes' centre.

---

## 6. The model and training layer

### 6.1 Heads (`src/model.py`, `src/fusion.py`)

| Head | Shape | Use |
|---|---|---|
| `VariantPathogenicityMLP` | `Linear(d→h)` → N × (`Linear→LayerNorm→GELU→Dropout` + residual) → `Linear(h→1)` | broad-panel pipeline |
| `BranchHead` | single-view MLP | ESM-only and priors-only ablation baselines |
| `ConcatFusionHead` | project each view to a shared dim, concatenate, MLP | plan-default fusion |
| `GateWaveFusionHead` | sigmoid branch gate + softmax per-feature gating + GLU + residual (adapted from MVmamba) | the fusion head to beat |

All four are trained on **identical splits** by `scripts/run_mmr_transfer.py`,
because a fusion architecture that is never compared against its own branches
gets credited for gains that came from one branch alone.

### 6.2 The two-stage transfer (`src/transfer.py`)

**Stage 1 — pretrain** (`scripts/pretrain_esm_80.py`). Fit the head on the broad
panel. `select_stage_rows(stage="pretrain", mode="leave_gene_out")` drops every
MMR gene, so no MMR variant contaminates the representation before transfer.
Split by residue group, `StandardScaler` fitted on train only, early stopping on
validation ROC-AUC with best-weight restoration. The checkpoint stores the model
state, the scaler, **the exact prior-column order**, the ESM block dimension, and
now the imputation constants.

**Stage 2 — fine-tune** (`scripts/run_mmr_transfer.py`). Warm-start from that
checkpoint on **clinical labels only** (`select_stage_rows` filters to
`label_source ∈ {clinvar, pg_clinical}` on MMR genes — DMS-only and functional
assay values never become targets), evaluate leave-one-gene-out with bootstrap
CIs, and run the full ablation battery.

**Schema drift is fatal, by design.** If `esm_dim` or the prior-column order
differs from the checkpoint, stage 2 raises rather than proceeding. Transfer is
invalid when column *order* changes even if dimensions happen to match, and a
silently-invalid transfer run produces plausible numbers.

### 6.3 True backbone fine-tuning (`src/esm_finetune.py`) — stage 2b

This is the only place gradients reach the transformer. Everything above is a
linear probe on frozen features.

```python
h_wt  = backbone(wt_ids).last_hidden_state[batch, wt_pos + 1]    # +1 skips <cls>
h_mut = backbone(mut_ids).last_hidden_state[batch, mut_pos + 1]  # siamese only
feat  = cat([h_wt, h_mut, h_mut - h_wt, |h_mut - h_wt|])         # → MLP → logit
```

- `mode="siamese"` — **ProPath**: separate WT and mutant passes, pooled at the
  mutated residue. Backbone LR 1e-5, batch 8, 10 epochs.
- `mode="wt_site"` — **CSBJ**: a single WT pass, per-residue token classifier.
  Half the compute, no explicit mutant representation.
- `n_unfrozen_layers`: `-1` full backbone, `0` frozen (the ablation floor that
  isolates what backbone gradients actually buy), `N>0` last N layers.
- `--gradient_checkpointing` trades ~30% speed for the memory that usually
  decides whether the 650M checkpoint fits on a consumer GPU.

### 6.4 Losses (`src/loss.py`)

Plain BCE is the default. `WeightedBCELoss` uses a **fixed** class ratio computed
once from the fitting partition — recomputing `pos_weight` per mini-batch made
gradients noisy on small final batches. `FocalLoss` remains available with a
neutral `alpha=0.5`; the inherited 0.25 default was actively wrong here, since
positives are ~43% of all labels and ~52% of clinical labels, so it was
down-weighting the majority-ish class.

**The objective correction that mattered most:** only 5,152 of 190,494 labelled
rows (**2.7%**) carry a clinical label. A loss treating rows equally optimises
assay fitness, not pathogenicity. Hence `--clinical_weight` (default 5.0) and
early stopping monitored on **clinical-only** ROC-AUC whenever both classes are
present in the monitored slice.

---

## 7. The evaluation layer

| Control | Where | What it prevents |
|---|---|---|
| Residue-group disjoint folds | `dataset.make_position_group_folds`, grouped on `"{uniprot_id}:{position}"`, asserted at runtime | different substitutions at one residue landing on both sides |
| Protein / gene holdout | `scripts/eval_leave_one_protein_out.py`; LOGO for MMR | quoting a new-residue number as if it were a new-gene number |
| Cluster-disjoint split | `scripts/build_cluster_split.py` (MMseqs2, 20% id / 20% cov) | paralogues straddling a split |
| Nested calibration | `src/calibration.py` + `src/train.py` | a calibrator fitted on the fold it scores (the cause of the impossible isotonic ECE = 0) |
| Clinical-only slices | reported separately from all-labels | DMS assay performance quoted as clinical performance |
| Target-leakage diagnostic | `scripts/diagnose_overfitting.py` | features that encode the label by construction |
| Bootstrap CIs + tuned thresholds | `src/eval_utils.py`, 10k percentile CIs, MCC-optimal threshold from an inner slice only | point estimates on a few-thousand-row set read as precise |
| Mandatory ablations | `scripts/run_mmr_transfer.py` | fusion credited for a single branch's gains |
| Schema-drift abort | `src/transfer.py` | a transfer run that silently invalidated itself |

**The grouping-key bug worth remembering:** `make_position_group_folds`
originally grouped on the raw `position`. Pooling proteins then collapsed
position 42 of TP53 and position 42 of BRCA1 into one "leakage group" — wrong in
both directions. The key is now `"{uniprot_id}:{position}"`, passed explicitly.

---

## 8. Design decisions and better alternatives

Each row: what the code does now, why, and the specific alternative worth
switching to — with the condition that should trigger the switch.

### 8.1 Data and labels

| Decision | Why it is this way | Better alternative, and when |
|---|---|---|
| **DMS labels only when `n_dms_assays == 1`** | A median across assays measuring different phenotypes is not a coherent target, and disagreement is real signal. | **Use unanimous multi-assay agreement as a label**, weighted by assay count, and quarantine only genuine disagreement. This is strictly more supervision at equal or better confidence, and the current rule discards agreeing evidence for no benefit. **Switch when:** you want more labels — this is the cheapest available win in the data layer. |
| **DMS binarised into the same target as clinical labels** | One head, one loss, one pipeline. | **Multitask or curriculum training**: continuous, within-assay-normalised DMS fitness as a regression task; clinical pathogenicity as a classification head; pretrain on the former, fine-tune the latter. Binarising throws away the magnitude that makes DMS informative. **Switch when:** you have GPU budget for two heads — this is the highest-ceiling change in the repository, and it is already listed as open work. |
| **A single `--min_stars` confidence floor** | One number, easy to defend. | **Split ClinVar provenance into an expert-panel tier and a down-weighted lower-confidence tier** rather than a hard cut. A 1-star assertion is not worthless, it is uncertain, and the weighting machinery to express that already exists. **Switch when:** label scarcity is your binding constraint (it is, on MMR genes). |
| **Cross-source conflicts quarantined entirely** | Public evidence cannot establish clinical truth; a manufactured winner hides the dispute. | Keep it. The alternative — resolve by tier — is defensible only if you *also* report how many labels were decided that way. |
| **Median imputation + missingness flags** | Cheap, stable, and the flag preserves the distinction that matters. | **Learned missingness embeddings**, or gradient-boosted trees that handle NaN natively (see 8.3). For columns missing ~99% of the time, the imputed value carries no information and the flag carries all of it. **Switch when:** you move to a tree model, which makes the whole question disappear. |

### 8.2 Features

| Decision | Why | Better alternative, and when |
|---|---|---|
| **Last-layer hidden states** | The conventional default. | **Mid-layer or layer-averaged representations.** For ESM-2, middle layers are repeatedly reported to carry more structural signal than the last, which is specialised for the MLM objective. **Switch when:** you can afford one sweep — `compare_backbones.py` is the natural place, and it is a cheap experiment with a real chance of a free gain. |
| **`[h_wt ‖ h_mut ‖ Δ ‖ \|Δ\|]` = 4d+1 features** | Gives the head both absolute context and the mutation's direction and magnitude without making it learn subtraction. | **`[Δ ‖ \|Δ\| ‖ PLLR]` alone** (2d+1) if you are overfitting: half the width, and the absolute states are what let the model memorise a residue. **Switch when:** the gap between residue-grouped and gene-held-out performance is large — that gap is the memorisation you would be removing. |
| **PLLR from a single WT forward pass** | Masked-marginal scoring done correctly and cheaply. | Nothing better at this cost. A true masked-marginal *per position* (mask each site explicitly) is marginally more faithful and vastly more expensive. |
| **Sliding windows for >1,022 aa** | Correct, and the log-space averaging is right. | **Mutation-centred asymmetric windows** (the VariPred recipe, already implemented in `mvmamba_features.centered_window_bounds`) for long chains: one window centred on the variant instead of averaging over windows where the variant is peripheral. **Switch when:** working on MSH6 (1,360 aa) or PMS2 — this is already available, just not the default in `esm_extractor`. |

### 8.3 Model and training

| Decision | Why | Better alternative, and when |
|---|---|---|
| **MLP head on frozen embeddings as the default** | Runs anywhere, fast to iterate, and it is one of four published recipes rather than an assumption. | **Gradient-boosted trees (XGBoost/LightGBM) on the priors branch.** For ~22 mostly-missing tabular columns this is the stronger model class almost by default: native NaN handling, no imputation question, no scaling question, better sample efficiency at 5k clinical rows. **Switch when:** you care about the priors-only number — which is currently the *best* number in the project (0.963 clinical AUC). This is the most likely quick win in the repo and it is not yet tried. |
| **Full backbone fine-tune (`n_unfrozen_layers=-1`) as the stage 2b default** | It is the ProPath recipe as published. | **LoRA / adapters, or unfreezing the last 2–4 layers.** With ~5k clinical labels, fine-tuning 650M parameters is enormously over-parameterised; LoRA gets most of the benefit at a fraction of the memory and with far less catastrophic forgetting. `--n_unfrozen_layers 4` is available today and costs nothing to try. **Switch when:** you see a large train/val gap in stage 2b, or you do not have a big GPU. |
| **Per-view `StandardScaler` refitted on the fine-tune train slice** | Standard, and it adapts to the target domain's distribution. | Defensible as-is, but note it means the pretrained first-layer weights meet differently-scaled inputs at warm-start. **Reusing the checkpoint's scaler** makes warm-starting more faithful; refitting adapts better to domain shift. Worth measuring rather than assuming — the checkpoint already carries `scaler_mean`/`scaler_scale` and currently nothing reads them. |
| **BCE + sample weights for class imbalance** | Simple and transparent; the weights carry real evidence-quality meaning. | Fine. Focal loss is available; for a ~43/57 split it is solving a problem you do not have. |
| **Early stopping on clinical-only ROC-AUC** | Optimises the target you actually care about. | **Clinical AUPRC** if the deployment slice is heavily imbalanced — AUC over-reports on rare positives. **Switch when:** you move from balanced clinical slices to prospective VUS scoring, where positives are rare. |

### 8.4 Orchestration

| Decision | Why | Better alternative, and when |
|---|---|---|
| **Subprocess-per-stage orchestration** | Clean resource reclamation, restartable, every command is copy-pasteable from a banner. | **A real workflow engine (Snakemake/Nextflow/Airflow)** once you want content-addressed caching, automatic re-run of only what changed, and parallel fan-out over genes. The current design already re-implements the easy 20% of this (`--skip_build`, feature caches, checksummed manifests). **Switch when:** you find yourself hand-managing which stages need re-running after a source update. |
| **CSV as the inter-stage format** | Universally inspectable; you can `grep` a bug. | **Parquet** — typed, ~5–10× smaller, ~10× faster to load, and it would eliminate the `pd.to_numeric(errors="coerce")` defensive conversions scattered through the code that exist purely because CSV loses dtypes. **Switch when:** load times or the dtype coercions start costing you; keep a CSV export for inspection. |
| **Stage 2b on by default** | It is the actual DL training stage; making it opt-in led to it never being run. | Keep, now that the accelerator preflight stops a CPU-only box before it wastes a day. Consider defaulting `--n_unfrozen_layers` to a small N rather than `-1` so the default is *runnable* as well as *real*. |

---

## 9. Invariants you must not break

1. **The key is `(uniprot_id, position, wt_aa, mut_aa)`** and `wt_aa` must match
   the canonical sequence. Any new source validates against the sequence before
   joining, or it does not join.
2. **A new prior column must be added to `TRANSFER_PRIOR_COLS`** or the model
   never sees it. There is a test that enforces this. Do not delete it.
3. **Validation-only columns must never enter `TRANSFER_PRIOR_COLS`.** MaveDB and
   CIMRA are held out so agreement with them means something.
4. **Splits group on `"{uniprot_id}:{position}"`**, never bare `position`.
5. **Calibrators are fitted outside the fold they score.** An ECE of exactly zero
   means you broke this.
6. **A conflicted row must not carry a label.** `validate_master_for_export`
   enforces it; keep it that way.
7. **Checkpoint schema is a contract** — prior-column order, ESM dimension, and
   imputation constants. Mismatch aborts loudly rather than proceeding.

---

## 10. Known issues, ranked

**Fixed on 2026-08-27** (recorded here because the failure modes are worth
recognising elsewhere in the code):

1. **Feature cache validated by row count only.** `extract_features_cached`
   checked `len(meta) != len(df)` and nothing else, while callers rely on
   positional alignment *and* read prior columns straight off the cached
   metadata. A variant table rebuilt with the same row count in a different
   order silently returned another variant's embeddings. Now validated by
   variant key (`_assert_cache_matches`), with a regression test.
2. **Imputation constants recomputed per stage.** Stage 1 pretrained on the
   80-gene panel's medians; stage 2 warm-started onto the four MMR genes'
   medians. Since `zs_*` columns are ~99% missing, the imputed constant is the
   column for almost every row. Now persisted in the checkpoint and reused, with
   a warning when an older checkpoint has no stored values.
3. **A half-built venv counted as ready.** `ensure_environment` gated on the
   interpreter existing. `python -m venv` takes a second; `pip install` takes
   minutes — so any interrupted install left an environment that every later run
   silently adopted and then failed inside a stage with an ImportError. Now
   gated on a stamp file written after a successful install, which also forces a
   reinstall when switching between the CPU and CUDA requirement sets.

**Also fixed on 2026-08-27, in the merge itself.** Each of these produced
either silently wrong data or an aborted 80-gene build from one bad row:

4. **The zero-shot join key was built without an integer cast.** `master["mutant"]`
   was `wt + str(position) + mut`; the other side of the join uses
   `.astype(int).astype(str)`. One NaN position anywhere in one source frame
   makes the whole column float, rendering `"A10.0C"` — and **all 17 published
   zero-shot model scores join zero rows**, with no error. This was the only
   position-derived string key in the codebase missing the cast. Fixed at the
   root by `_normalise_key_dtypes()`, applied to every source frame before the
   merge, plus a `logger.error` when a non-empty zero-shot panel joins nothing.
5. **`gene` and `hgvs_p` were source-supplied but part of the de-duplication
   key.** Two sources disagreeing on a gene alias or an HGVS rendering produced
   two rows for one variant, which then collided on `MASTER_KEY` and aborted the
   build at export. Both are now *derived*: the panel owns the gene symbol, and
   `_render_hgvs_p()` renders every row canonically. De-duplication on
   `MASTER_BASE_COLS` is now equivalent to de-duplication on `MASTER_KEY` by
   construction.
6. **A non-standard residue aborted the whole build.** `ONE_TO_THREE` covers only
   the 20 standard residues, so a selenocysteine (`U`) — which genuinely occurs
   in UniProt canonical sequences — or an ambiguity code yielded a NaN `hgvs_p`
   and a failed export. `_THREE_LETTER` extends the map (Sec, Pyl, Asx, Glx,
   Xle, Xaa) and falls back to the raw letter, so `hgvs_p` can never be NaN.
7. **Residue case was not normalised.** A source emitting `"a"` split one variant
   into two rows that then collided on the key. Now upper-cased and stripped
   alongside the position coercion.
8. **Synonymous rows (`wt_aa == mut_aa`) were kept.** Not missense variants at
   all, and a single-assay DMS row like this acquired a pathogenicity label.
   Now dropped and counted (`master_dropped_synonymous`), along with rows
   carrying an incomplete key.
9. **Imputation medians spanned the split.** Stage 2 computed them over
   `pool = ft_df + ho_df`, letting the holdout gene influence the constant that
   *is* the feature for ~99%-missing `zs_*` columns. Stage 2 now derives them
   from the fine-tune partition only (or reuses the checkpoint's); stage 1
   re-imputes from its training fold after the split and stores those values.

`assemble_master` now has direct test coverage for all of the above
(`tests/test_merge.py`, 16 tests) — it previously had none.

**Open:**

10. **`scaler_mean`/`scaler_scale` are written into every transfer checkpoint and
    never read.** Either wire them into warm-starting (see §8.3) or drop them;
    dead payload in a schema contract invites false assumptions.
11. **Agreeing multi-assay DMS evidence is still discarded** (§8.1). This is a
    label-policy decision, not a defect — but it is free supervision being
    thrown away, and the cost is measurable on a real build with
    `(n_dms_assays > 1) & (dms_bin_nunique == 1)`.
12. **`PROJECT_PLAN.md` does not exist** in the repository despite being cited by
    name across the docs and module docstrings. Commit it or rewrite the
    references.

## See also

- `docs/PROJECT_DOCUMENTATION.md` — what was built, why, and what it improves on
- `docs/DATASETS.md` — exhaustive per-source reference: URLs, licences, schemas
- `docs/DATA_PIPELINE_HARDENING.md` — ingestion/merge/provenance guarantees
- `docs/CODE_REVIEW.md` — findings against the pre-existing code + changelog
- `docs/FINE_TUNING_FINDINGS.md`, `docs/TRAINING_NOTES.md` — results and evidence
- `docs/RUN_REPORT.md`, `docs/RUNLOG.md` — build record and dated run history
