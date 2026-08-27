# From dataset to tensor — how the data reaches the model

**What this covers:** the exact path a variant takes from a row in
`extended_dataset.csv` to a number in a loss function, for each of the three
training paths in this repository. Every transformation, in order, with the
function that performs it and the shape at each step.

- The *sources* and their processing rules: `docs/DATASETS.md`
- The *merge* that produces the table: `docs/CODE_GUIDE.md` §4
- The *dataset schema and label policy*: `docs/DATASETS.md` §8–9

Verified against the tree on 2026-08-27.

---

## 1. The three paths

There is no single "the model". Three distinct paths consume the dataset, and
they do **not** see the same features:

| Path | Entry point | Sees priors? | Sees ESM? | Gradients into ESM? |
|---|---|---|---|---|
| **A. Broad-panel CV** | `run_pipeline.py` → `src/train.py` | yes | optional | no |
| **B. Two-stage transfer** | `pretrain_esm_80.py` → `run_mmr_transfer.py` | yes | optional (`--features`) | no |
| **C. Backbone fine-tune** | `finetune_esm_mmr.py` | **no** | yes | **yes** |

That third row is the one people get wrong. Path C is **sequence-only** — it
reads `gene, position, wt_aa, mut_aa, label, label_weight` and nothing else. No
AlphaMissense, no gnomAD, no zero-shot scores. It is a different model on
different inputs, not a better version of paths A/B.

---

## 2. Path B, stage by stage (the recommended pipeline)

### Step 1 — read and select rows

```python
master = pd.read_csv("data/processed/extended/extended_dataset_train.csv")
selected = select_stage_rows(master, stage=..., mode=..., holdout_gene=...)
```

`src/transfer.select_stage_rows` is the **anti-circularity gate**, applied
before any fitting:

| Stage | Rows kept |
|---|---|
| `pretrain`, `leave_gene_out` | everything **except** MLH1/MSH2/MSH6/PMS2 |
| `pretrain`, `practical` | everything |
| `finetune` | MMR genes only, **and** `label_source ∈ {clinvar, pg_clinical}` |

The fine-tune filter is why DMS-derived and functional-assay values can never
become targets in stage 2: they are excluded by label *source*, not by column.
Quarantined rows are already excluded upstream — their `label` is `NaN`, so they
are absent from `extended_dataset_train.csv` entirely.

### Step 2 — build the two feature branches

`src/transfer.assemble_features` returns a `FeatureBundle` with two matrices
that stay row-aligned with `meta`:

```
FeatureBundle
├── X_esm    [n, 4d+1]   None in --features priors mode
├── X_prior  [n, 2p]     p values + p is_missing_* flags
├── prior_cols           the exact ordered column names
├── impute_values        fill value per source column
└── meta                 row-aligned dataframe (labels, genes, positions)
```

**The prior branch** (`prior_matrix`):

```
raw      = df[base_cols].apply(pd.to_numeric, errors="coerce")   [n, p]
missing  = raw.isna().astype(float32)                            [n, p]   ← the flags
values   = raw.fillna(fill)                                      [n, p]
X_prior  = concat([values, missing], axis=1)                     [n, 2p]
```

Three things matter here:

1. **Which columns.** Exactly `TRANSFER_PRIOR_COLS` plus every `zs_*` column —
   defined once, in `src/transfer.py:58`. A column present in the CSV but absent
   from that tuple is **invisible to the model**. This is not hypothetical: it
   happened to gnomAD allele frequency, which was joined onto the table and
   silently ignored. A regression test now fails if a new source is added
   without being wired through.
2. **Missingness flags are half the matrix.** Most `zs_*` columns are absent for
   ~99% of rows. Without the flag, "this model never scored this variant" is
   indistinguishable from "this model scored it at exactly the median".
3. **Fill values are pinned across stages.** Stage 1 stores its per-column
   medians in the checkpoint; stage 2 reuses them. Because the imputed constant
   *is* the feature for ~99% of rows, recomputing medians per stage meant the
   pretrained head was warm-started onto a differently-centred input.

**The ESM branch** (`esm_branch_matrix` → `extract_features_cached`), per gene:

```
h_wt, logp = one forward pass over the WT sequence
PLLR_i     = logp[i, mut] - logp[i, wt]                    ← masked-marginal, free
h_mut      = forward pass per UNIQUE (position, mut_aa), gathered back to rows
X_esm      = [h_wt ‖ h_mut ‖ Δ ‖ |Δ| ‖ PLLR]              [n, 4d+1]
```

`d` = 1280 for `esm2_t33_650M`, 480 for `esm2_t12_35M`. Sequences over 1,022
residues use overlapping sliding windows with hidden states averaged across
covering windows and log-probabilities averaged in log space.

Cached to `{gene}_{model}[+extrasN]_features.npz`. A cache hit is validated by
**variant key**, not row count — a table rebuilt with the same row count in a
different order used to return another variant's embeddings silently.

### Step 3 — split, *then* scale

```python
groups     = meta["uniprot_id"] + ":" + meta["position"]     ← the leakage group
tr, va     = make_position_group_folds(positions, y, k_folds=5, groups=groups)
scaler     = StandardScaler().fit(X[tr])                     ← train partition ONLY
X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])
```

The group key is `"{uniprot_id}:{position}"`, never bare `position` — pooling
proteins otherwise collapses position 42 of TP53 and position 42 of BRCA1 into
one "leakage group", which is wrong in both directions.

In `run_mmr_transfer.py` the holdout gene is carved out first, and the scaler is
refitted per view on the fine-tune train slice only.

**Imputation constants are derived from the fitting partition only.**
`assemble_features` has to run before the split (the split needs `meta`, and in
`esm+priors` mode extraction decides which rows survive), so its first pass
necessarily sees validation rows. Both stages correct for this:

- Stage 1 re-imputes from its training fold after the split and stores those
  values in the checkpoint. The prior matrix is cheap to rebuild — no ESM
  re-extraction.
- Stage 2 reuses the checkpoint's values, or, with no checkpoint, derives them
  from `ft_df` alone — never from `pool`, which deliberately contains the
  holdout gene so its features can be built in the same call.

This matters because ~99% of `zs_*` values are missing, so the imputed constant
*is* the feature for almost every row. The `StandardScaler` was never affected —
it is fitted after the split.

### Step 4 — sample weights

```python
w = stage_sample_weights(meta, clinical_weight=5.0, dms_weight=1.0)
#   = (5.0 if label_source in {clinvar, pg_clinical} else 1.0) * label_weight
```

Two multiplied axes: **task priority** (clinical evidence matters 5× more than
assay fitness — only 2.7% of labelled rows are clinical, so an unweighted loss
optimises the wrong target) and **evidence quality** (`label_weight` from
`docs/DATASETS.md` §9). A quarantined row would get 0.0, but cannot appear here
anyway.

### Step 5 — tensors and the training loop

```python
tensors = [torch.as_tensor(x, dtype=torch.float32) for x in Xs_train]
perm    = torch.randperm(n)                       # shuffled each epoch
for idx in batches(perm, batch_size=256):
    logits = model(*(t[idx].to(device) for t in tensors))
    cw     = where(y > 0.5, pos_weight, 1.0)      # fixed ratio from the fit partition
    loss   = (BCEWithLogits(logits, y, reduction="none") * w[idx] * cw).mean()
```

`pos_weight = n_neg / n_pos` is computed **once** from the fitting partition, not
per mini-batch — recomputing it per batch made gradients noisy on small final
batches.

Early stopping monitors validation ROC-AUC, restricted to the **clinical-only**
slice whenever that slice has both classes (`monitor_mask`), with best-weight
restoration. Selecting on all-labels AUC would select for assay-fitness
performance.

### Step 6 — what crosses the stage boundary

Stage 1 writes a checkpoint that is a **schema contract**, not just weights:

| Key | Used by stage 2 for |
|---|---|
| `model_state_dict` | warm-start initialisation |
| `feature_columns` | pinning the exact prior-column **order** |
| `prior_impute_values` | pinning the imputation **constants** |
| `config["esm_dim"]` | asserting the ESM block width matches |
| `scaler_mean`, `scaler_scale` | **nothing — currently written and never read** |

A mismatch on order or `esm_dim` **aborts**. Transfer is invalid when column
order changes even if the dimensions happen to line up, and a silently-invalid
transfer run still produces plausible-looking numbers.

---

## 3. Path C: how the backbone fine-tune is fed

Completely different plumbing — raw sequence text, not a feature matrix.

```
row (gene, position, wt_aa, mut_aa, label, label_weight)
  │  build_examples()          src/esm_finetune.py:163
  │    - look up the canonical sequence for `gene`
  │    - DROP the row if seq[position-1] != wt_aa        ← same contract as extraction
  │    - crop a mutation-CENTRED window of ≤1022 residues
  │    - siamese mode: also build the mutated window
  ▼
FineTuneExample(wt_seq, wt_pos0, mut_seq, mut_pos0, label, weight)
  │  make_collate_fn()  → tokenizer(..., padding=True)
  ▼
batch: wt_ids [B,L] · wt_mask [B,L] · wt_pos [B] · (mut_* …) · labels [B] · weights [B]
  │  forward()
  ▼
h      = backbone(ids, mask).last_hidden_state          [B, L, d]
site   = h[arange(B), pos + 1]                          ← +1 skips the <cls> token
feat   = cat([site_wt, site_mut, Δ, |Δ|])  (siamese)    [B, 4d]
       = site_wt                           (wt_site)    [B, d]
logit  = out(head(feat))                                [B]
```

Two details worth knowing:

- **`pos + 1` is not an off-by-one.** ESM tokenizers prepend `<cls>`, so the
  residue at 0-based `pos` sits at token index `pos + 1`.
- **The window is mutation-centred**, unlike the sliding-window averaging in the
  frozen extractor. For MSH6 (1,360 aa) this keeps the variant near the middle of
  the context instead of at the edge of some windows.

`n_unfrozen_layers` controls what actually receives gradients: `-1` the whole
backbone, `0` none (the ablation floor that isolates what backbone gradients
buy), `N>0` the last N transformer layers.

Path C writes `esm_finetune_results_*.csv` and **nothing in the repository reads
it**. There is no path from a fine-tuned backbone into the prior-fusion heads
that produce the project's best number.

---

## 4. What the model never sees

Deliberate exclusions. Each is a column that exists in the CSV and is kept out
of `TRANSFER_PRIOR_COLS` on purpose:

| Excluded | Why |
|---|---|
| `dms_score_median`, `dms_bin_median`, `n_dms_assays` | On single-assay rows (97.3% of labelled data) `dms_bin_median` equals `1 − label` **by construction**. Including it produced a 0.9987 ROC-AUC that was pure circularity; the honest all-labels number is 0.716. |
| MaveDB scores, CIMRA OddsPath | Held out so that agreement with orthogonal experimental evidence is a real check, not a restatement of a training input (`FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS`). |
| `label_*`, `*_conflict`, `sources`, `stars` | Supervision and provenance metadata, not features. |
| Quarantined rows | Never have a `label`, so they never reach `extended_dataset_train.csv`. |

**How to check what a run actually fed the model** — the honest answer is not
"read the code", it is:

```python
from src.transfer import prior_columns_of
import pandas as pd
df = pd.read_csv("data/processed/extended/extended_dataset_train.csv")
cols = prior_columns_of(df)
print(len(cols), "prior columns")
print(df[cols].notna().mean().sort_values())     # per-column coverage
```

A column with 0.00 coverage is a broken join, not a real absence.

---

## 5. Is the dataset correct? What is actually verified

Honest answer: **the merge logic is verified; a specific build is not, and no
merge can make the underlying evidence true.** Three separate levels:

### Level 1 — the merge logic (verified in code, 2026-08-27)

`tests/test_merge.py` pins the behaviours that fail silently, exercising
`assemble_master` directly with synthetic edge cases (16 tests):

| Test | Pins |
|---|---|
| `test_label_precedence_clinvar_beats_clinical_beats_dms` | precedence order and the full weight ladder |
| `test_dms_bin_is_flipped_into_pathogenicity` | the `1 − bin` flip that AUC cannot detect |
| `test_multi_assay_dms_yields_no_label` | agreeing *and* disagreeing multi-assay rows stay unlabelled |
| `test_contradictory_clinvar_assertions_are_quarantined` | never resolved by star count or row order |
| `test_cross_source_disagreement_is_quarantined` | label, weight and source all cleared |
| `test_quarantined_rows_cannot_reach_the_training_export` | the export gate holds |
| `test_row_universe_is_a_union_not_a_join` | unlabelled rows survive to be scored |
| `test_metadata_is_backfilled_before_deduplication` | no AlphaMissense "twin" rows |
| `test_zeroshot_join_survives_a_float_position_column` | the silent-join regression |
| `test_key_dtype_is_normalised_across_sources` | mixed int/float keys do not split a variant |
| `test_provenance_records_every_contributing_source` | `sources` / `n_sources` coherence |
| `test_gene_alias_from_one_source_does_not_split_a_variant` | the panel owns the gene symbol |
| `test_hgvs_p_is_derived_not_taken_from_the_source` | one canonical rendering per variant |
| `test_non_standard_residues_do_not_abort_the_build` | `U`/`X`/`B` render, never NaN |
| `test_residue_case_is_normalised` | `"a"` and `"A"` are one variant |
| `test_synonymous_rows_are_dropped` | `wt == mut` never becomes a label |

Before this file existed, `assemble_master` — the function that decides every
training label — had **no direct test coverage at all**.

### Level 2 — a specific build (verified by the auditor, per build)

`scripts/audit_extended_dataset.py` **re-derives the merge independently** and
runs 12 checks into `audit_report.json`; exit code 0 iff every hard check
passes. The 80-gene build passed 12/12, and the 1,919 variants covered by both
ClinVar and ProteinGym-clinical agreed 100%.

A builder that validates itself only proves it is self-consistent, which is why
the auditor is a separate implementation. **Run it after every build** — the
pipeline does this automatically as its own stage.

Additional per-build guarantees: every remote artefact is SHA-256 checksummed
into `manifest.json` with URL, version, byte count and date; downloads stage to
`.part` and promote atomically only after size and checksum validation.

### Level 3 — the evidence itself (not verifiable here, ever)

- DMS labels are **functional-assay proxies**, not clinical truth.
- AlphaMissense and the supervised `zs_*` predictors were themselves trained on
  ClinVar, so the clinical evaluation slice may be temporally contaminated.
- The PMS2 exon 11–15 pseudogene region stays unresolved until orthogonal
  confirmation data is supplied; the safe default excludes PMS2 entirely.

No merge discipline fixes any of these. They bound what the numbers can mean.

### What the merge now guarantees

Every one of these was a real defect on 2026-08-27 — four of them aborted an
entire 80-gene build from a single bad row, one silently discarded all 17
zero-shot score columns. All are fixed and asserted in `tests/test_merge.py`:

| Input | Was | Now |
|---|---|---|
| A float `position` anywhere | zero-shot join matched **0 rows**, silently | key dtype normalised across every source frame; a zero-match join logs `ERROR` |
| Two sources, different gene alias | two rows → duplicate key → build aborts | panel owns the symbol; one row |
| Two sources, different HGVS rendering | two rows → duplicate key → build aborts | `hgvs_p` always derived canonically |
| Selenocysteine (`U`), `X`, `B` residue | `hgvs_p` = NaN → build aborts | rendered `Sec` / `Xaa` / `Asx`; never NaN |
| Lowercase or padded residue letter | two rows → duplicate key → build aborts | upper-cased and stripped |
| `wt_aa == mut_aa` (synonymous) | kept, and could acquire a pathogenicity label | dropped and counted |

### Remaining gaps

1. **Agreeing multi-assay DMS evidence is discarded.** A variant covered by two
   assays that *agree* still gets no label. This is a label-policy decision, not
   a defect — but it is free supervision being thrown away. Count what it costs
   on a real build with `(n_dms_assays > 1) & (dms_bin_nunique == 1)`.
2. **The evidence itself is still what it is** (Level 3 above). No merge
   discipline makes a functional assay into a clinical outcome.

---

## 6. Verifying a build yourself

```bash
# 1. Merge invariants (offline, no data needed, ~2 s)
python tests/test_merge.py

# 2. Independent audit of an actual build -> exit 0 iff all 12 checks pass
python scripts/audit_extended_dataset.py
cat data/processed/extended/audit_report.json

# 3. Provenance of every byte that went in
python -c "import json;m=json.load(open('data/processed/extended/manifest.json'));print(json.dumps(m['sources'],indent=2))"

# 4. What the model will actually be fed (coverage per prior column)
python -c "
import pandas as pd; from src.transfer import prior_columns_of
df = pd.read_csv('data/processed/extended/extended_dataset_train.csv')
print(df[prior_columns_of(df)].notna().mean().sort_values().to_string())"

# 5. Label composition
python -c "
import pandas as pd
df = pd.read_csv('data/processed/extended/extended_dataset_train.csv')
print(df['label_source'].value_counts(dropna=False))
print(df.groupby('label_source')['label'].mean())"
```

Step 4 is the one people skip. A prior column at 0.00 coverage means a broken
join, and it will not raise anything.
