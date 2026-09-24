# DL ↔ SLM interface audit — local code audit, 2026-09-24

Scope: `vpdl/dl/interface.py` (DL contract), `vpdl/dl/cli.py` `cmd_export`,
`vpdl/dl/runner.py` (representation export), `vpdl/slm/interface.py` (SLM contract and
`DLInput`), `vpdl/slm/modeling/finetune.py` (the only consumer),
`docs/slm/GENOMIC_SLM_DL_INTERFACE.md`. **Final fusion is not implemented and was not
started.** No DL export exists yet (`runs/dl/` has no `export/`), so everything below ran
on synthetic records written by the real writers.

## Field mapping

| Expected | DL (`DLRepresentation` / `DL_OUTPUT_SCHEMA`) | SLM (`SLMRepresentation` / `SLM_OUTPUT_SCHEMA`) |
|---|---|---|
| variant_id | `variant_id` (`GENE:pos:W>M`, protein level) | `variant_id` (`clinvar:<id>`) + `protein_variant_id` for the join |
| gene | `gene` | `gene` |
| embedding | `embedding` (float32, inline or `.npy` row) | `slm_embedding` |
| evidence embedding | — | `evidence_embedding` |
| pathogenicity / class probabilities | `pathogenicity_score` | `class_probabilities`, `pathogenic_probability`, `benign_probability`, `vus_probability` |
| uncertainty | `uncertainty` + `uncertainty_method` | same |
| evidence / explanation | — | `evidence`, `acmg`, `explanation` (not filled by `vpdl-slm embed`; flagged) |
| model_version / feature_version | required | required |
| quality_flags | **not in the DL contract** — the free `metadata` object can carry it | `quality_flags` |

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| I-1 | HIGH | `vpdl/slm/interface.py` | Class probabilities were checked only for summing to ~1, so 1.6 and −0.6 passed and `pathogenic_probability` read 1.0. | **FIXED** — each value finite and in [0, 1], in the dataclass and the validator |
| I-2 | HIGH | `vpdl/dl/interface.py`, `vpdl/slm/interface.py` | Embeddings stored as `.npy` references (the default) were never checked: a NaN row passed validation and would reach the SLM's LayerNorm. | **FIXED** — both readers check width and finiteness of every referenced row |
| I-3 | HIGH | `vpdl/slm/interface.py` `fold_check` | The out-of-gene check compared the DL record's fold to the gene by string equality — true only for `logo`/`logo_purged`. Every variant from a `family` fold (`family-MutL`) was reported as possibly in-fold. | **FIXED** — family folds resolve to their genes from the DL branch's own tables; `random_debug` folds are still (correctly) not out-of-gene |
| I-4 | HIGH | `vpdl/slm/modeling/finetune.py` | `fold_check` was never called in training; the interface doc promised a fold report for every joined variant. | **FIXED** — computed, logged as a warning when any joined record is not out-of-gene, and stored in the registry record and `model.pt`. It does not stop training (a decision for the data phase). |
| I-5 | MEDIUM | both readers | Embeddings stayed memory-mapped views; on Windows the `.npy` could not be rewritten or deleted while any representation lived (reproduced: `OSError`, `WinError 32`). | **FIXED** — rows are copied |
| MASTER-2 | LOW | `DLInput.from_outputs` | A variant appearing twice in a DL export silently kept the last record (possible with `--score-rows all`). | **FIXED** — refused with the ids |
| — | LOW | `vpdl/slm/interface.py` | A NaN uncertainty passed validation and was written as a bare `NaN` token (not JSON). `vpdl-slm embed` passed NaN when predictions lacked entropy. | **FIXED** — validator requires finite; embed writes null plus flag `uncertainty_missing` |
| I-6 | MEDIUM | `vpdl/dl/interface.py` | No `quality_flags` on the DL side. | OPEN — changing the frozen DL contract is a decision; use `metadata.quality_flags` meanwhile |
| I-7 | LOW | `finetune.py` | DL versions were computed and discarded. | **FIXED** — recorded with the fold report |
| M-7 | MEDIUM | `vpdl/slm/build/features.py` | The structured feature `dl_score` is declared (default on) but never filled, and has no fold-aware resolver or leakage check. | OPEN — wire it only with a fold check |

## Verified correct

`SLMRepresentation` is the DL branch's `ReasoningRepresentation`; dimensions come from the
records; a variant with no DL record gets zeros and mask 0 and the model ignores the masked
vector; float32 everywhere; the join key (`protein_variant_id`) is read from the DL
canonical table, so spelling drift between branches cannot happen; gene symbols with
lowercase or hyphens do not crash the writer. The cross-branch test (DL writer → SLM reader)
exists and passes.

## Deferred to the DGX

The first real `vpdl-dl export` and `vpdl-slm finetune --dl-outputs` (EXP-018): read the fold
report before trusting any DL-augmented number.
