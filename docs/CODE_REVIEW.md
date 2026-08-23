# Code Review & Changelog

Review performed on the pre-existing codebase before extending the data
layer, plus a complete changelog of everything added afterwards.

---

## Part 1 — Issues found in the original code

### Bugs fixed

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| B1 | `src/esm_extractor.py :: validate_and_align` | `np.char.str_len(np.asarray(df["mut_aa"], dtype="U1")) == 1` truncates every value to ≤1 char, so the mut-aa check was vacuous | explicit per-row check `len(m)==1 and m in VALID_AA` |
| B2 | `src/esm_extractor.py` | missing import of `VALID_AA` after fix B1 | imported from `data_loader` (single source of truth) |
| B3 | `main.py` stage-2 cache flow | features cached under `{gene}_{model}_features.npz` regardless of appended columns; adding priors would silently reuse an incompatible cache | cache tag now includes `+extras{n}` when prior features are attached |

### Design weaknesses addressed

| # | Location | Weakness | Improvement |
|---|----------|----------|-------------|
| D1 | `data_loader._stream_gene_variants` | one full ~4 GB-decompressed pass over ClinVar **per gene**; O(genes × minutes) | new `_stream_genes_variants` serves all requested genes in ONE pass (`build_multi_gene_dataset`); old function kept as a wrapper |
| D2 | `dataset.make_position_group_folds` | grouping key was raw `position`; pooling proteins would collapse equal positions of different proteins into one leakage group and distort folds | optional explicit `groups` parameter; `train.run_pipeline` now passes `"{uniprot}:{position}"` keys automatically when a `uniprot_id` column exists |
| D3 | `loss.WeightedBCELoss` | `pos_weight` recomputed per batch → noisy gradients on small final batches | documented; focal loss remains default (unchanged semantics for reproducibility) |
| D4 | `esm_extractor._embed_spans` | dead if/else branch accumulating window pieces identically in both arms | collapsed to unconditional append |
| D5 | `esm_extractor.extract_features_cached` | GPU memory freed only after npz compression | extractor deleted before compression |

### Verified-correct behaviours (no change needed)

* Temperature scaling fitted on each fold's validation split — intentional
  for this benchmark; caveat already documented in the docstring.
* PLLR trick reading both pseudo-log-likelihoods from one masked-context
  forward pass matches Meier et al. 2021.
* StratifiedGroupKFold partition completeness + leakage assertion.
* Logging hierarchy: root at WARNING while project loggers honour `--debug`
  (records propagate to root handlers irrespective of ancestor levels).

---

## Part 2 — New additions (this change)

### New modules

| File | Purpose |
|------|---------|
| `src/external_datasets.py` | resumable downloader with SHA-256 verification; ProteinGym DMS / clinical / zero-shot parsers; AlphaMissense streaming filter; RefSeq→UniProt exact-sequence mapper; UniProt domain fetcher |
| `src/extended_builder.py` | gene-panel resolution, multi-source assembly into a master table, label precedence rules, provenance manifest |
| `scripts/build_extended_dataset.py` | CLI for the above (panel selection, per-source toggles, cache overwrite) |
| `tests/test_datasets.py` | 10 unit tests: token parsing, significance classification, fold disjointness incl. multi-protein groups, alignment validation, column-name normalisation |
| `docs/DATASETS.md` | exhaustive per-dataset documentation (URLs, versions, licences, schemas, processing rules, caveats) |
| `README.md` | project overview, quickstart, architecture |

### Modified files

| File | Change |
|------|--------|
| `src/data_loader.py` | added `_stream_genes_variants`, `_dedupe_and_split`, `build_multi_gene_dataset`; single-gene API preserved |
| `src/dataset.py` | `make_position_group_folds(..., groups=None)` for protein-aware leakage groups |
| `src/train.py` | `cross_validate(..., groups=None)`; automatic `uniprot:position` group keys in `run_pipeline`; `uniprot_id` included in OOF metadata columns |
| `src/esm_extractor.py` | fixes B1/B2/D4/D5; `extract_features_cached(..., extra_features=...)` appends external priors and versioned cache tag |
| `main.py` | comma-separated `--gene` pooling across proteins (per-protein ESM extraction then vertical concat); `--extra_features {none,zeroshot,alphamissense,dms,structure,all}` joining extended-dataset priors onto variants with median imputation |

### Data artefacts added under `data/raw/`

| Path | Source | Bytes |
|------|--------|-------|
| `proteingym/DMS_ProteinGym_substitutions.zip` | ProteinGym v1.3 | 43,021,128 |
| `proteingym/DMS_substitutions_reference.csv` | ProteinGym v1.3 | 208,734 |
| `proteingym/clinical_ProteinGym_substitutions.zip` | ProteinGym v1.3 | 3,356,697 |
| `proteingym/zero_shot_clinical_substitutions_scores.zip` | ProteinGym v1.3 | 13,241,411 |
| `alphamissense/AlphaMissense_aa_substitutions.tsv.gz` | Google DeepMind | 1,207,278,510 |
| `variant_summary.txt.gz` | NCBI ClinVar (pre-existing) | 442,283,371 |
| `uniprot_domains/{ACC}.json` | UniProt REST (per panel gene) | ~50 KB total |

(SHA-256 for every artefact is recorded in
`data/processed/extended/manifest.json` at build time.)
