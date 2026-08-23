# Run Report — extended dataset build & integration

Minute-by-minute record of everything executed on 2026-08-23, including the
exact commands, artefact checksums, intermediate failures and their fixes.

---

## 1. Environment

| Item | Value |
|------|-------|
| Machine | Linux, 12 CPU cores, 14 GB RAM, 424 GB free (no GPU) |
| Python | 3.14.7 (project `.venv`) |
| Key packages | torch 2.13.0+cpu · transformers 5.15.1 · pandas 3.0.5 · numpy 2.5.2 · scikit-learn 1.9.0 · scipy 1.18.1 |
| Network | NCBI FTP, Zenodo / Harvard mirror, GCS, UniProt REST all reachable |

## 2. Datasets evaluated for viability

| Source | Verdict | Reason |
|--------|---------|--------|
| ClinVar `variant_summary` | **adopted** (already present; upgraded to multi-gene single-pass) | primary supervision |
| ProteinGym v1.3 DMS substitutions | **adopted** | 217 assays / 2.5M mutants, direct UniProt keys |
| ProteinGym v1.3 clinical benchmark | **adopted** | independent curated labels, 63K variants |
| ProteinGym v1.3 zero-shot clinical scores | **adopted** | EVE/ESM1b/GEMME/… priors for free |
| AlphaMissense aa-substitutions | **adopted** | full-proteome prior keyed by UniProt accession |
| UniProt domain annotations | **adopted** | cheap REST call, structural flag |
| dbNSFP 4.x | rejected | ~30 GB unpacked; overlaps zs_* columns |
| gnomAD v4 | rejected | Hail toolchain too heavy here |
| EVE GitHub repo | rejected as source | ships MSAs only; EVE scores arrive via ProteinGym bundle instead |
| MaveDB | rejected | per-experiment registration scraping |

## 3. Downloads executed

| File | URL | Bytes | SHA-256 (first 16) |
|------|-----|-------|--------------------|
| `variant_summary.txt.gz` | ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/ | 442,283,371 | `d2a8c9c2f038c703` |
| `DMS_ProteinGym_substitutions.zip` | marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ | 43,021,128 | `3a83766254ac9ac9` |
| `DMS_substitutions.csv` (reference) | same base | 208,734 | `a8f498011532a74a` |
| `clinical_ProteinGym_substitutions.zip` | same base | 3,356,697 | `afe711af49365bc1` |
| `zero_shot_clinical_substitutions_scores.zip` | same base | 13,241,411 | `6ae0dd2c61ea3adc` |
| `AlphaMissense_aa_substitutions.tsv.gz` | storage.googleapis.com/dm_alphamissense/ | 1,207,278,510 | see manifest.json |

Total new download volume: **≈ 1.27 GB**. Zip integrity verified with
`zipfile.testzip()` (218 + 2,525 members OK). AlphaMissense size matches the
GCS `Content-Length` exactly.

## 4. Build pipeline execution

Command: `python scripts/build_extended_dataset.py`

Stages and observed behaviour:

1. Panel resolution via UniProt REST — 10/10 genes → accessions
   (`panel_sequences.json` cached).
2. ClinVar single pass over 9 genes (TP53 served from existing cache) —
   ~90 s; per-gene labelled/VUS CSVs rewritten.
3. ProteinGym DMS parse — 96 human assays → 329,664 substitutions;
   entry-name→accession translation (`BRCA1_HUMAN`→P38398 …);
   panel filter + wt validation → **45,528 rows / 11 assays**
   (57 mutants dropped: isoform numbering mismatch).
4. Clinical benchmark + zero-shot scores — 62,727 clinical variants parsed;
   string labels `Pathogenic/Benign` mapped to 1/0;
   RefSeq NP_→panel mapping by exact sequence equality → **7 proteins /
   1,308 label rows** joined to 17 model-score columns.
5. AlphaMissense streaming filter — scanned **216,175,352 lines**, kept
   110,124 rows for the panel (~7 min one-off; cached thereafter).
6. UniProt domains — 10 REST JSONs cached.
7. Assembly → `extended_dataset.csv`: **110,124 unique substitutions,
   zero duplicate keys, zero null gene/hgvs_p.**

## 5. Failures hit during development & their fixes

All of these were caught by running the build end-to-end and fixed before the
final green run:

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `KeyError: '_stars'` in assembly | cached per-gene CSVs persist only `FINAL_COLUMNS`; stars dropped by `_dedupe_and_split` | re-derive stars from `review_status` via `stars_for_review` |
| `IndexError: arrays used as indices must be integer` | float positions from pandas-3 CSV reads (empty VUS frames) | defensive `pd.to_numeric` coercion in `validate_and_align` |
| `KeyError 'uniprot_id'` on zs join | zero-shot frame merged with `np_accession` only | include `uniprot_id` in the mapping merge |
| `KeyError 'wt_aa'` in master | `MASTER_KEY` omitted `wt_aa`, so base union never carried it | key now `[uniprot_id, position, wt_aa, mut_aa]` + explicit base-column list |
| `KeyError 'gene'` on DMS part | DMS table has no gene column | attach gene from panel map |
| DMS panel = 0 rows | reference `UniProt_ID` holds *entry names* (`BRCA1_HUMAN`), not accessions | added `resolve_uniprot_entry_names` (batched REST, cached); note TP53's name is `P53_HUMAN` |
| clinical labels all NaN | `DMS_bin_score` contains strings (`Pathogenic`/`Benign`), not numbers | string→binary vocabulary mapping + drop-and-count unmapped |
| hgvs_p all NaN in new parsers | mapped one-letter codes through `THREE_TO_ONE` instead of `ONE_TO_THREE` | corrected direction in 3 parsers + builder backfill; purged stale AM cache |
| 29,327 exact-duplicate master rows | union de-duplication ran **before** NaN-gene backfill, so AM twins survived | reorder: backfill → dedupe (verified 110,124 = unique keys) |
| `transformers` EsmConfig import failure | torchvision ABI break (`operator torchvision::nms does not exist`) vs torch 2.13+cpu | `pip install --force-reinstall --no-deps torchvision==0.28.0+cpu` from the PyTorch CPU index |
| `UnboundLocalError: c` in `load_extra_features` | comprehension typo | renamed loop variable |

## 6. Verification runs

| Check | Command | Result |
|-------|---------|--------|
| Unit tests (10) | `python tests/test_datasets.py` | all pass |
| Static lint | `python -m pyflakes src/*.py main.py scripts/*.py tests/*.py` | clean |
| Smoke train w/ priors, pooled genes | `python main.py --gene NUDT15,CALM1 --debug --extra_features all --esm_model facebook/esm2_t12_35M_UR50D` | completed 31 s: 22 prior cols appended (d=1943), 5-fold grouped CV, calibration tables, VUS predictions written |
| Backward-compat single-gene run | `python main.py --gene TP53 --debug` (cached t12 features) | completed 3 s, original artifacts regenerated |
| Master-table integrity | duplicate-key / null-meta scan on `extended_dataset.csv` | 110,124 rows = 110,124 unique keys; no nulls |

Note on the smoke metrics: NUDT15+CALM1 have 17 labelled ClinVar variants,
all pathogenic — folds are single-class, so ROC-AUC is undefined there. This
validates plumbing, not biology; use larger genes or the pooled panel for real
evaluation.
