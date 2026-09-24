# Calibration audit — local code audit, 2026-09-24

Scope: `vpdl/dl/calibration.py` (binary), `vpdl/slm/evaluation/calibration.py` (five-class),
`vpdl/slm/evaluation/{metrics,uncertainty}.py`, every call site that fits a calibrator
(`vpdl/dl/cli.py` `cmd_calibrate`, `vpdl/slm/modeling/finetune.py`).

**No calibrator is fitted on test data in either branch.** The DL branch fits on each
fold's inner-validation predictions (`valpreds_*.csv`) and applies to that fold's held-out
predictions; the SLM fits on the `val` / `mmr_val` rows.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| MASTER-1 / O-1 | HIGH | `vpdl/slm/modeling/finetune.py:287-312` | The binary decision threshold was chosen on **raw** validation probabilities, then applied to **calibrated** scores on every split (and saved next to the calibrator in `model.pt`). Temperature scaling changes (P+LP)/(P+LP+LB+B), so MCC, sensitivity, specificity, precision, F1 and balanced accuracy were computed at a mismatched threshold in every config (all use `calibration = "temperature"`). Found independently by the master auditor and subagent O. | **FIXED** — threshold chosen on calibrated validation scores; `model.pt` records `threshold_basis`; test `test_the_threshold_is_chosen_on_the_scores_it_is_applied_to` |
| H-1 | MEDIUM | `finetune.py:284-286` | `fit_calibrator`'s refusal of test rows was fed `np.full(n, "val")`, so it checked the caller's intention, not the data. | **FIXED** — passes the rows' own split labels |
| — | LOW | `vpdl/slm/evaluation/calibration.py` | Fitting on zero labelled rows returned the search bound (T = 20) as if it were a result. | **FIXED** — raises; finetune skips calibration when validation has no labelled row |
| H-2 | MEDIUM | `predict.py`, `finetune.py` | Mutual information on uncalibrated MC draws beside calibrated entropy. | **FIXED** |
| H-3 | LOW | both modules | Binary Brier (one term) and multi-class Brier (sum over classes) differ in scale by design; do not compare the numbers across branches. | Documented |
| H-4 | LOW | `vpdl/dl/failure.py` | Per-gene calibration in the DL failure report has no small-n pooling (the SLM pools groups under 30). | OPEN |
| H-5 | LOW | `vpdl/dl/failure.py` `overfitting` | Under the `family` split the inner-validation rows were matched by gene and none matched, so the gap was silently NaN. | **FIXED** — matches on `fold` when both files carry it (as `cmd_calibrate` already did) |
| H-6 | LOW | `finetune.py` | Early stopping scores on `val + mmr_val`; calibration fits on `val` only when both exist. | Documented |

## Verified correct (synthetic checks)

Binary temperature: T = 0.988 on calibrated data, finite on separable data. Multi-class
temperature: softmax(z/T) on logits, T bounded [0.05, 20] by golden-section search in log
space. Platt: finite under complete separation (ridge + logit clipping). Isotonic: monotone,
clips out-of-range inputs. ECE = 0 on a perfectly calibrated example. Vector-scaling gradient
matches the analytic softmax cross-entropy gradient.
