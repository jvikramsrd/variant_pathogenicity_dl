# DL data pipeline audit — local code audit, 2026-09-24

Scope: `vpdl/dl/{canonical,hgvs,splits,leakage,feature_store,functional,genomic,homology,structure,context}.py`,
`vpdl/{assemble,features,splits,experiment}.py`, `vpdl/sources/**`. Existing pipeline only;
nothing downloaded. The research splits were **not** changed.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| B-1 | MEDIUM (data correctness; touches existing DGX results) | `vpdl/assemble.py:176-180` (`groupby(key)[other].first()`), `vpdl/sources/gnomad.py:296-325`, `vpdl/dl/canonical.py:481-503` | When several nucleotide changes give one protein change, gnomAD emits one row per allele and assembly keeps the **first** allele's `feature_gnomad_log10_af` and ACMG flags (confirmed with a synthetic two-allele example: −6.0 and −2.0 → kept −6.0). `canonical.py` recomputes the display column `gnomad_af` from all records and counts disagreement (`gnomad_first_allele_differs: 655` of 74,328 rows), but the model-facing feature is not corrected. **On `canonical_full.csv` — byte-identical (sha256 `6cea58db…`) to the table behind `runs/dl/main` — 7 of the 449 ClinVar-labelled rows disagree (MSH2 5, MSH6 1, MLH1 1).** The disagreement goes both ways (median log10 gap −5: some rows where the recomputed AF is ~0 while the feature records an observed allele), so multi-allele collapsing is not the only cause; the recomputation's filtering should be checked too. On `canonical.csv` (with DMS labels) 174 labelled rows disagree. | **OPEN — decision needed.** A fix changes feature values behind results already produced on the DGX, so it needs a table rebuild and re-run, not a silent edit. At 1.6% of rows in one of six features the effect on the headline is probably small, but that is for the re-run to show. |
| B-2 | LOW | `vpdl/dl/canonical.py:394` | Without `clinvar_records`, the canonical table falls back to the order-dependent assembled label. Documented; the canonical build always passes the records. | OPEN — suggest a loud warning if omitted |
| O-4 | MEDIUM | `vpdl/dl/cli.py` `cmd_embed`, `vpdl/dl/feature_store.py` | The embedding `--dtype` was not part of the feature-store identity: a float16 entry was "reused" for a float32 request, and a float32 entry could then never be written. | **FIXED** — dtype joins the identity only when not float16, so every existing entry keeps its key |
| B-3 | INFO | `vpdl/experiment.py:296` | Gene-constant columns are dropped using the full table (including the held-out gene). By design (identical schema across arms); only affects which columns exist, never fitted values. | Documented |

## Verified correct

- Standardisation and imputation are fitted on the inner-training fold only
  (`build_feature_matrix`, `Augmenter`); validation and test use `transform`.
- The decision threshold comes from inner-validation scores only.
- Leakage groups are protein-qualified (`uniprot_id:position`, landmine L3);
  `assert_no_group_straddle` runs every fold; `tests/dl/test_splits_leakage.py` passes.
- Fold seeds are keyed on the gene name via SHA-256 (`derive_seed`), not loop position or
  Python's salted `hash()` (landmine L7).
- Feature-store keys include model, sequence, dataset versions and the full identity; a
  lookup for a missing variant raises instead of zero-filling (L5). Entries are written
  atomically.
- `pg_dms.to_label` maps `DMS_score_bin == 1` to benign (L1); ClinVar VUS/conflicting map to
  NaN, never a class; orientation is checked after merging against an independent anchor.
- ClinVar multi-record protein changes are resolved order-independently and discordant
  ones are quarantined; ClinVar-vs-DMS contradictions are quarantined, not resolved.
- The PMS2 homology gate is fail-closed (L9).
- The manifest is written after the final write of every file (L10).

## Deferred to the data phase

- The real prevalence and effect of B-1 on the headline numbers (re-run after the fix).
- The leakage reports for `logo_purged`, `family` and `random_debug` exist in
  `runs/dl/leakage/`; `docs/dl/DATA_LEAKAGE_REPORT.md` part B still reads "pending".
- Strict-mode PLM pretraining corpora (`vpdl/dl/pretrain/corpus.py`) have not been
  exercised against a real UniProt corpus.
