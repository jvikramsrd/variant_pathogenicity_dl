# Gene-level holdout and anti-leakage protocol

Static audit only — no code in this document was executed against real data
in this environment. Every claim below is either (a) a direct reading of the
cited file/line, labelled CONFIRMED SAFE or CONFIRMED RISK, or (b) explicitly
marked as a design note rather than a defect.

## What "gene-level holdout" means here

An entire protein — every one of its variants, across every joined source
(ClinVar, ProteinGym, gnomAD, MaveDB, CIMRA) — is absent from training,
preprocessing-fit, and model/hyperparameter selection for the split(s) where
it is held out. See `src/gene_aliases.py`'s module docstring for how this is
distinguished from sample-level, patient-level, batch-level and
random-feature holdout, all of which are different guarantees this codebase
also uses in places (nested inner folds; feature-group ablation) but must
not be confused with gene-level holdout.

## Fit-boundary audit

For each leakage-sensitive fitting step (imputation medians, standardisation
mean/std, feature-column selection), the question is: does the training-time
statistic ever read a row from the gene held out for that split?

| Location | Fits on | Holdout excluded? | Verdict |
|---|---|---|---|
| `scripts/finetune_esm_mmr.py:245-253` (`build_prior_inputs`) | `columns` from `ft_df` (fine-tune partition only); `impute` from `train_rows = ft_df.iloc[list(tr_idx)]` (inner-train slice of the fine-tune partition) | Yes — `ft_df` itself already excludes the holdout gene (`prepare_split`), and `tr_idx` further excludes the inner-val slice | **CONFIRMED SAFE** |
| `scripts/finetune_esm_mmr.py:263-266` (mean/scale) | `x_tr` (same `train_rows`) | Yes, same as above | **CONFIRMED SAFE** |
| `scripts/run_mmr_transfer.py` `run_one_split` (`fixed_impute = prior_impute_values(ft_df)`) | `ft_df`, not `pool` (which contains the holdout gene) | Yes | **CONFIRMED SAFE** |
| `scripts/run_mmr_transfer.py::scale_views` (`StandardScaler().fit(m[tr_idx])`) | `tr_idx = ft_idx_all[inner_tr_local]`, itself a subset of `ft_idx_all = flatnonzero(~is_holdout)` | Yes | **CONFIRMED SAFE** |
| `scripts/eval_leave_one_protein_out.py:155` (`StandardScaler().fit(X[tr_idx])`) | `tr_idx = tr_pool_idx[tr_local]`, and `tr_pool_idx = flatnonzero(~is_held)` | Yes | **CONFIRMED SAFE** |
| `src/train.py:267` (`StandardScaler().fit(features[fit_idx])`) | `fit_idx`, which excludes both the outer `val_idx` fold and two further inner slices (`early_local`, `calibration_local`) reserved for early-stopping/calibration | Yes, relative to the outer validation fold this call's own loop iteration holds out | **CONFIRMED SAFE** (note: this is `make_position_group_folds`-based CV, gene-level only if the caller passes gene-derived `groups` — see Design notes) |
| `scripts/run_mmr_transfer.py::main()` — `add_within_gene_rank_features(master, raw_score_cols)` runs on the full `master` table before any per-gene split | Full table, all genes | No exclusion, but see below | **CONFIRMED SAFE, with reasoning** |

### On the one "runs on the full table before split" case

`add_within_gene_rank_features` is applied to the entire master table before
`run_one_split` is called per holdout gene. This looks, at first glance, like
exactly the anti-pattern items 1-6 above avoid. It is not a leak, because the
transform is **within-gene**: a variant's rank is computed only against
other variants *of the same gene* (labelled and VUS alike), never against
another gene's variants, and it does not use the label. Concretely:

* It contains no cross-gene statistic (no global mean/std/median), so
  running it once up front and running it separately per split (recomputing
  only on non-holdout genes each time) produce byte-identical output for
  every gene's rows — the up-front computation is an efficiency choice, not
  a shortcut that changes results.
* It does not read `label`, so it cannot encode which partition a row will
  later fall into.

If a future change to `add_within_gene_rank_features` introduces any
cross-gene statistic (e.g. a global rank normalisation instead of per-gene),
this reasoning stops applying and the call would need to move inside
`run_one_split`, fit on `ft_df` only, exactly like every other case above.

## No confirmed leakage bugs found

This audit found zero confirmed leakage defects across the four call chains
inspected (`scripts/finetune_esm_mmr.py`, `scripts/run_mmr_transfer.py`,
`scripts/train_extended.py` → `src/train.py`, `scripts/eval_leave_one_protein_out.py`).
This is not a claim that the pipeline is leakage-free in general — only that
every *imputation/scaling/column-selection* fit boundary checked resolves
correctly to the training partition. It does not cover: `src/calibration.py`'s
temperature/isotonic fitting (out of scope for this pass — audit separately
before trusting calibrated probabilities across a holdout boundary),
cluster-split construction in `scripts/build_cluster_split.py` (not read for
this pass), or any leakage route through the LLM/fusion branches described in
`PROJECT_PLAN.md` Phases 4-5, which do not yet have training code to audit.

## New protections added (this change)

* `src/gene_aliases.py` — canonical UniProt-accession resolution
  (`resolve_gene_id`, `UnknownGeneError`) and `assert_disjoint_gene_sets`,
  which catches a gene appearing in more than one partition even when the
  partitions spell it differently (`"MLH1"` vs `"P40692"`).
* `src/split_manifest.py` — `SplitManifest` / `build_split_manifest`, which
  bundles canonical gene sets, a seed, and a dataset SHA-256
  (`src.provenance.sha256_file`) into one versioned, on-disk, re-verifiable
  record. `verify_against_dataset` proves a dataset file is byte-identical
  to the one a manifest's genes were assigned against before anything is
  trained from it.
* Neither module edits or duplicates `src.mmr_dataset.write_leave_one_gene_out_manifest`;
  they are an additive, stricter layer callers can adopt incrementally.

## Design notes (not defects)

* `src/train.py`'s CV harness (`make_position_group_folds`) is
  group-disjoint by construction, but "group" is whatever the caller passes
  — it is gene-level only when `groups` is derived from gene identity.
  Callers using it for a non-gene grouping (e.g. protein-position clusters)
  are doing sample-level or cluster-level holdout, not gene-level; this is
  correct usage, not a bug, but worth naming explicitly per the user's
  requirement to distinguish holdout types.
* No test in this codebase currently asserts, at the `SplitManifest` layer,
  that a *specific* production script's train/val/test/holdout assignment is
  disjoint end-to-end — the fit-boundary audit above is per-callsite static
  reading, not an automated invariant. Wiring `assert_disjoint_gene_sets`
  into each script's split-construction code (not done in this change — no
  existing file was edited) is the natural next step.

## Remaining risks / follow-ups

* `src/calibration.py` fit boundaries not audited in this pass.
* `scripts/build_cluster_split.py` not audited in this pass (different
  leakage surface — sequence-similarity clustering, not gene-symbol
  aliasing).
* This document and the two new modules are **not executed in this
  environment**; `tests/test_split_manifest.py` covers `src/split_manifest.py`
  and `src/gene_aliases.py` with synthetic in-memory data only and has been
  syntax-checked (`python3 -m py_compile`) but not run (no working Python
  environment with pytest/pandas available here).
