# Regression audit — local code audit, 2026-09-24

Baseline commit: `faf8de2` (branch `v2/rebuild`, clean tree). Machine: Windows 11 PC, CPU only.
Command (from the repository root, `PYTHONPATH=.`, in a throw-away `uv` environment):

```bash
uv run --no-project --python 3.14 --with pytest --with numpy --with pandas --with scikit-learn --with scipy --with torch --with transformers --with tokenizers --with pyarrow --with matplotlib --with tabulate --with xgboost --with lightgbm --with requests --with seaborn --with biopython --with tqdm python -m pytest tests -q -p no:cacheprovider
```

Environment B is the same command with `"pandas<3"` and `"scikit-learn<1.9"` (see
DEPENDENCY_AUDIT.md for why: Windows Application Control started blocking the pandas 3.0.6
and scikit-learn 1.9.1 DLLs mid-session).

## Results

| Run | Code | Env | Passed | Skipped | Failed | Errors |
|---|---|---|---|---|---|---|
| 0 | baseline `faf8de2` | A — first try, without v1 requirements | — | — | — | 9 collection errors (`requests`, `seaborn` missing: environment, not code) |
| 1 | baseline `faf8de2` | A | 686 | 1 | 1 | 9 |
| 2 | baseline `faf8de2` (extracted with `git archive`) | B | 686 | 1 | 1 | 9 |
| 3 | after fixes | B | **767** | 1 | **0** | **0** |

Runs 1 and 2 are identical, so the baseline is reproducible and the change of environment
does not move any result. Run 3 against run 2, test by test:

- every test that passed at baseline passes after (0 regressions);
- the 10 baseline non-passes now pass (see below);
- 71 new tests, all passing;
- the one skip is the CUDA-only landmine test (`test_landmines.py:1221`, no GPU here).

Per suite (run 2 → run 3): DL 108 + 10 failing → 132; KB 47 → 47; regression 66 (+1 skip) →
66 (+1 skip); SLM 192 → 249; v1 archive 273 → 273.

**No regression detected in the tests executed.**

## The 10 pre-existing failures

All ten were one test-infrastructure bug, not DL code: `tests/dl/conftest.py` and
`tests/slm/conftest.py` both import as the module `conftest` (neither directory is a package),
so when the whole suite ran, `from conftest import PANEL` in the DL tests got the SLM file.
`tests/dl` alone passed 118/118. Fixed by moving the DL builders to
`tests/dl/dl_helpers.py` (a unique name) and importing from there; no assertion changed.

## Do the new tests detect the bugs?

The new test files were run against the unfixed baseline code: **42 of the 71 fail there**
(every test aimed at a fix). The 29 that pass on both are guards (evidence sentences that
must stay unmasked, every shipped config loads).

## Changes, one line each

| File | Change | Reason | Risk | Test | Result |
|---|---|---|---|---|---|
| `vpdl/slm/text/conclusion.py` | verb-less / abbreviated verdict patterns; broader tripwire | CRITICAL label leakage | over-masking evidence | 17 masker cases + existing text tests | pass |
| `vpdl/slm/build/leakage.py` | `input_gate`; missing clusters critical under text/laboratory | gate gaps | a run without `dedup` now fails the audit under those schemes (intended) | `test_training_refuses…`, `test_an_isolating_split…` | pass |
| `vpdl/slm/modeling/finetune.py` | threshold on calibrated scores; real split to calibrator guard; calibrated MC draws; input gate; DL fold report; backbone/DL provenance | metric correctness, leakage, reproducibility | none on results (nothing trained) | `test_the_threshold…`, `test_finetune_itself…` | pass |
| `vpdl/slm/modeling/predict.py` | ACMG gene mask at inference; calibrator transform for MC draws | C-2, H-2 | — | `test_acmg_codes…` | pass |
| `vpdl/slm/modeling/continued.py` | per-step RNG seeding; `base_model_prefix` | exact resume; non-BERT encoders | dropout draws differ from an old unseeded run (none exist) | `test_a_resumed_pretraining_run…` | pass |
| `vpdl/slm/evaluation/calibration.py` | refuse fitting on zero labelled rows | silent T = 20 | — | full suite | pass |
| `vpdl/slm/interface.py` | per-class bounds, finite uncertainty, finite referenced rows, copy rows, duplicate refusal, family folds | contract | stricter validation can refuse records that passed before (they were invalid) | 5 interface tests | pass |
| `vpdl/slm/cli.py` | embed quality flags / NaN; evaluate VUS join; `--lr 0`; pretrain-corpus requires exclusions | C-1, O-2, O-14, M-5 | runbook already passes `--exclusions` | `test_vus_outcomes…`, `test_a_pretraining_corpus…` | pass |
| `vpdl/slm/teacher.py` | unit validation; missing confidence rejected; temperature/seed refused | G-3/G-4/G-5 | — | 3 teacher tests | pass |
| `vpdl/slm/experiments.py` | EXP-017 note: not runnable yet | honesty | — | matrix test | pass |
| `vpdl/slm/smoke.py` | registry restore in `finally` | robustness | — | smoke run | ok |
| `vpdl/slm/build/{records,inventory}.py`, `vpdl/provenance.py`, `vpdl/dl/tracking.py` | paths stored with `/`; `command` in run records | portability, reproducibility | none on Linux | `test_manifest_paths…`, `test_a_run_record…` | pass |
| `vpdl/dl/leakage.py` | label-proxy check in the per-cell gate | M-1 | could stop a cell with a ≥ 0.99 proxy (intended); real DGX table passes | `test_the_cell_gate…`, real-table dry run | pass |
| `vpdl/dl/trainer.py` | `TrainConfig` validation | N-1 | refuses configs that used to be clamped | 5 parametrised tests | pass |
| `vpdl/dl/plm/finetune.py` | `checkpoint_trainable_only=True` | A-1 | old PLM checkpoints would not resume (none exist) | `test_plm_finetune_checkpoints…` | pass |
| `vpdl/dl/interface.py` | copy + check referenced rows | I-2, I-5 | — | 2 tests | pass |
| `vpdl/dl/failure.py` | match on `fold` when present | H-5 | logo output unchanged | `test_overfitting…` | pass |
| `vpdl/dl/cli.py` | `train --dry-run`; dtype in embed identity when not float16 | audit brief; O-4 | float16 keys unchanged | 2 dry-run tests + real-table dry run | pass |
| `tests/dl/{conftest.py,dl_helpers.py,test_*.py}` | helpers moved to a unique module | K-1 | test-only | full suite | pass |

## Accidental-change check on the DL branch

- The real-table dry run (`data/built/canonical_full.csv`, sha256 identical to the DGX runs'
  dataset) produces the same four folds (449 test rows) and the same cell names as
  `runs/dl/main`, and the stricter gate passes with 0 critical findings.
- The DGX's own full leakage report had already run the label-proxy check ("no feature
  agrees with the label at ≥ 0.95"), so M-1's fix does not bear on existing results.
- Tracked result files (`runs/`, `results/`, `data/`) are unchanged (`git status`).
- The DL model code paths (`run_cell`, models, features, splits) were not edited.
