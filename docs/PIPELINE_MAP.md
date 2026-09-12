# Pipeline Map

Read-only inspection artifact. Every claim below cites `file:line` in this
checkout as of this writing; nothing here was executed to produce it — it is
a static trace of the code, not a measured behavior. Where a claim depends on
data that isn't present in this checkout (all of `data/raw/`, `data/*/processed/**`
are gitignored — see `.gitignore` and the "Retention" section below), that is
stated explicitly rather than assumed.

## 1. Entry points

| Script | Produces | Notes |
|---|---|---|
| `scripts/make_expanded_panel.py` | `data/raw/uniprot/expanded_panel.json` | Gene panel (all human ProteinGym DMS proteins by default). Input to `build_extended_dataset.py --panel_file`. |
| `scripts/build_extended_dataset.py` | `data/processed/extended/extended_dataset.csv` + `manifest.json` | Calls `src.extended_builder.build_extended_dataset`; the broad multi-gene table (README: 1,152,863 rows × 68 cols at last audit). |
| `scripts/audit_extended_dataset.py` | `data/processed/extended/audit_report.json` + a `*_train.csv` emission | 12-point audit + train-CSV emission (`scripts/audit_extended_dataset.py:21` lists checks incl. `C11 provenance coherence`). |
| `scripts/build_mmr_dataset.py` | `data/mmr/processed/extended/extended_dataset.csv` + `manifest.json` | MMR-specific (MLH1/MSH2/MSH6/PMS2) two-phase build: calls `build_extended_dataset` then joins gnomAD (stages 3-4), CIMRA (stage 7), MaveDB (stage 8), applies the PMS2 homology gate, and calls `refresh_manifest` once at the end (`scripts/build_mmr_dataset.py:325`). |
| `run_pipeline.py` / `run_pipeline.sh` / `run_pipeline.ps1` | Orchestrates the original single-/multi-gene workflow end to end | Reads `FEATURES`, `K_FOLDS`, `ESM_MODEL` env vars (`run_pipeline.py:127,134,139`). |
| `run_mmr_pipeline.py` / `.sh` / `.ps1` | Orchestrates the Lynch-syndrome MMR workflow (download → clean/process → train, stops before calibration/LLM/fusion by design per `README.md`) | Same `FEATURES`/`ESM_MODEL` env-var pattern (`run_mmr_pipeline.py:94,99`). |
| `scripts/repair_manifest.py` | Re-stamps an existing `manifest.json` in place | Wraps `src.extended_builder.refresh_manifest`; does not rebuild or modify the dataset (per its own `--dry-run` support). |
| `main.py` | (not traced in this pass) | Out of scope for this map; not touched by this session's changes. |

No `os.environ`/`os.getenv` usage exists anywhere in `src/` or `scripts/`
except `run_pipeline.py` (lines 25, 51, 108, 127, 129, 134, 139, 166) and
`run_mmr_pipeline.py` (lines 94, 99, 500) — confirmed by grep across
`src/*.py`, `scripts/*.py`, `main.py`, `prime_gnomad.py`. Both only read
`FEATURES`, `ESM_MODEL`, `K_FOLDS`, `VIRTUAL_ENV`, `PYTHON` to set argparse
defaults or locate the interpreter; nothing else in the pipeline is
env-configured.

## 2. Full data flow (broad-panel build)

```
raw source ingestion (ClinVar, ProteinGym DMS+clinical+scores, AlphaMissense,
UniProt sequences/domains, gnomAD, MaveDB, CIMRA, InterPro, AlphaFold)
        |
        v
per-source schema normalization (src/external_datasets.py, src/gnomad.py,
src/mavedb.py, src/interpro.py, src/structure.py — each returns a frame keyed
on MASTER_KEY = ["uniprot_id", "position", "wt_aa", "mut_aa"]
(src/extended_builder.py:141), or convertible to it)
        |
        v
gene/identifier mapping (UniProt accession <-> gene symbol via the panel
dict passed into assemble_master; ClinVar/ProteinGym rows mapped to
uniprot_id via `np_accession` -> uniprot_id mapping, src/extended_builder.py:496-497)
        |
        v
merge/join into one master table (src.extended_builder.assemble_master,
src/extended_builder.py:574 — 11 `.merge()` calls, see section 3)
        |
        v
dedup + conflict resolution (`_resolve_binary_evidence`,
src/extended_builder.py:260 — contradictory same-key binary labels are
EXCLUDED from supervision, not resolved by row order; counted into
stats.clinvar_conflicting_keys / stats.clinical_conflicting_keys /
stats.dms_conflicting_keys)
        |
        v
missing-value handling (label columns: left as NaN, never imputed here —
imputation is deferred to the model-input stage, see below; prior-score
columns: no fill applied in extended_builder.py itself)
        |
        v
export: master.to_csv(master_path) (src/extended_builder.py:1158) + manifest.json
        |
        v
[MMR path only] scripts/build_mmr_dataset.py: gnomAD join (stages 3-4),
CIMRA join (stage 7, src/cimra.py via attach_cimra), MaveDB join (stage 8,
src/mavedb.py via attach_mavedb), PMS2 homology gate
(src/mmr_dataset.py:263-275, nulls labels for unconfirmed-homology rows),
re-export master.to_csv (scripts/build_mmr_dataset.py:319), refresh_manifest
(scripts/build_mmr_dataset.py:325)
        |
        v
prior-feature scaling/imputation, FIT ONLY ON THE TRAIN/FINE-TUNE PARTITION
(src/transfer.py:prior_impute_values line 303, prior_matrix line 319 —
called with fixed_impute derived from `ft_df` only, never from the pool
that includes the holdout gene; see scripts/run_mmr_transfer.py:245-251
and scripts/finetune_esm_mmr.py's build_prior_inputs)
        |
        v
train/eval split (leave-one-gene-out or single-holdout; see section 4)
        |
        v
model input (src/transfer.py FeatureBundle: X_esm optional, X_prior always
present, both row-aligned to `meta`)
```

## 3. Merge inventory

None of the 11 `.merge()` calls in `src/extended_builder.py` pass
`validate=`; two joins elsewhere in the codebase already do
(`src/mmr_dataset.py:342` and `src/gnomad.py:436`, both `validate="m:1"`).
This asymmetry was flagged in this session's earlier audit and is the
motivating gap for `src/data_validation.py`'s
`validate_merge_cardinality` (new in this change — see below), not yet
retrofitted onto the merges themselves (a change to production merge
semantics was deliberately not made without a working test environment to
verify against; see this session's prior "Deliberately not fixed" note).

| File:line | `on=` | `how=` | Cardinality (declared) | Duplicate/conflict handling |
|---|---|---|---|---|
| `src/extended_builder.py:275` | `key` (MASTER_KEY) | left | undeclared | pre-filtered by `_resolve_binary_evidence` upstream |
| `src/extended_builder.py:496` | `np_accession` | inner | undeclared | drops non-mapped accessions by construction (inner join) |
| `src/extended_builder.py:497` | `np_accession` | (same call chain) | undeclared | — |
| `src/extended_builder.py:699` | `key` | left | undeclared | `cv_meta` pre-deduped via `drop_duplicates(subset=key, keep="first")` (line 696-697) sorted by star rating |
| `src/extended_builder.py:702` | `key` | left | undeclared | source (`cv`) already deduped above |
| `src/extended_builder.py:705` | `key` | left | undeclared | joins a conflict-flag frame; result filled 0 if absent (line 706) |
| `src/extended_builder.py:713` | `key` | left | undeclared | `cp` deduped inside `_resolve_binary_evidence` |
| `src/extended_builder.py:715` | `key` | left | undeclared | same conflict-flag pattern as 705 |
| `src/extended_builder.py:734` | `key` | left | undeclared | `agg` is a `groupby(key).agg(...)` — one row per key by construction |
| `src/extended_builder.py:738` | `key` | left | undeclared | `am_cols` pre-deduped via `drop_duplicates(subset=key)` (line 737) |
| `src/extended_builder.py:752` | `["uniprot_id","mutant"]` | left | undeclared | `zs_small` pre-deduped via `drop_duplicates(subset=["uniprot_id","mutant"])` (line 745); has an explicit zero-match guard (`logger.error`, lines 756-762) — the only merge in this file with any runtime cardinality check today |
| `src/extended_builder.py:823` | `uniprot_id` | inner | intentionally 1:many (a domain spans many positions) | fan-out is the intended semantics, not a bug |
| `src/extended_builder.py:832` | `["uniprot_id","position"]` | left | undeclared | joins a `groupby(...).agg(...)` result — one row per key |
| `src/mmr_dataset.py:342` | `key` | left | `validate="m:1"` (declared) | — |
| `src/gnomad.py:328` | `gene` | (join into `target`) | undeclared | — |
| `src/gnomad.py:436` | `key` | left | `validate="m:1"` (declared) | — |
| `src/mavedb.py` / `src/interpro.py:144,151` | `uniprot_id` / `["uniprot_id","position"]` | inner / left | undeclared | InterPro position-membership join has the same 1:many-by-domain shape as `extended_builder.py:823` |

**Known row-count effects already documented in this repo** (not re-derived
here, cited from prior findings in this session and `MISSING_EVIDENCE.md`):
the 21-row `stats.master_label_counts` (17,124) vs. `label_source` tally
(17,103) discrepancy in the MMR manifest was root-caused this session to the
PMS2 homology gate (`src/mmr_dataset.py:263-275`) mutating labels *after*
`build_extended_dataset` had already stamped the manifest — now fixed by
making `refresh_manifest` recompute `stats.master_label_counts` from the
on-disk CSV (`src/extended_builder.py::refresh_manifest`, this session's
earlier change). The zero-shot join's zero-match guard
(`extended_builder.py:756-762`) is the one place in this file that already
fails loudly on a broken key rather than silently returning an
all-missing column.

## 4. Splits (as they exist today)

- **Leave-one-MMR-gene-out (LOPO)**: `scripts/finetune_esm_mmr.py`'s
  `run_one_split`/`prepare_split`, `scripts/run_mmr_transfer.py`'s
  `run_one_split`/`prepare_split` (line 156) — holds out one of
  `MMR_GENES = ("MLH1","MSH2","MSH6","PMS2")` entirely; the fine-tune/inner-val
  split within the remaining genes uses
  `src.dataset.make_position_group_folds` with `groups = uniprot_id:position`
  (gene-and-position-scoped, not variant-level) — this is a **gene-level**
  holdout for the outer split and a **position-group** holdout for the inner
  one, not a random row split.
- **Single holdout**: `--eval holdout --holdout_gene <GENE>` in the same
  scripts — one designated gene held out, the rest used for
  fine-tune/inner-val.
- **Leave-one-protein-out (broad panel)**: `scripts/eval_leave_one_protein_out.py`
  — same shape as LOPO but over the full pretraining panel rather than the
  4 MMR genes.
- Feature-imputation constants (`prior_impute_values`) are derived from the
  fine-tune partition only, explicitly excluding the holdout gene
  (`scripts/run_mmr_transfer.py:245-251`'s comment states this rationale
  directly) — i.e. preprocessing-fit-before-split leakage was already
  designed against in this exact spot.

## 5. Config, dependencies, tests, outputs

- **Config**: argparse defaults per script (no central config file/YAML
  exists in this repo; confirmed via `find . -iname '*.yml' -o -iname '*.yaml'`
  in this session's earlier audit — no matches outside `.venv`/`.git`).
- **Dependencies**: `requirements.txt` / `requirements-cuda.txt` — all
  floating `>=` bounds, no exact pins (flagged in this session's earlier
  audit as a reproducibility gap, not fixed there; still open).
- **Tests**: `tests/test_datasets.py`, `test_esm_finetune.py`,
  `test_finetune_grid.py`, `test_merge.py`, `test_metrics.py`,
  `test_mmr_modules.py`, `test_new_data_sources.py`, `test_provenance.py`,
  `test_recalibration.py`, and this session's new `test_transfer_scripts.py`
  / `test_provenance.py` additions. No CI runs them (no `.github/` or other
  CI config exists in this repo, confirmed earlier this session).
- **Outputs / retention**: `.gitignore` excludes `data/raw/`,
  `data/mmr/raw/`, and `data/processed/**` / `data/mmr/processed/**`
  wholesale, then re-includes specific small artifacts:
  `manifest.json`, `*_summary*.json`, `*comparison*`, `audit_report.json`,
  `leave_one_gene_out_splits.json` — the large CSVs and per-variant
  prediction dumps are never committed (confirmed in this session's earlier
  audit, including the one exception where several
  `*_predictions_*.csv` files were force-added and remain tracked despite
  matching the ignore pattern — see git history, not re-derived here).
  **This checkout has no files under `data/raw/` or `data/*/processed/**`
  present locally** — every claim above about row counts, manifest content,
  etc. that isn't cited from a previously-read manifest excerpt is a
  structural/code claim, not a claim about this checkout's actual data.
