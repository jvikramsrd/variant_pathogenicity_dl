# SLM VUS / uncertainty audit — local code audit, 2026-09-24

Scope: `vpdl/slm/{labels,schema,interface}.py`, `vpdl/slm/evaluation/{vus,uncertainty,metrics}.py`,
`vpdl/slm/modeling/predict.py`, label mapping in `clinvar_text.py` / `build/records.py` / `build/examples.py`.

## VUS is a class, not "the model does not know"

- `vus_probability` is a real softmax output (`class_probabilities["vus"]`).
- `abstained` is independent: set from `max_prob`, mutual information and evidence counts,
  never from the predicted class or `p(VUS)`.
- VUS rows are kept in training (one-hot at the VUS index) and in five-class evaluation
  (macro-F1, balanced accuracy, per-class, confusion, MCC). The binary panel excludes them
  on purpose, as documented.

## Label mapping — run on every distinct `ClinicalSignificance` in the local ClinVar release

9,048,962 rows, 108 distinct strings: five_class 88.2%, missing 5.6%, conflicting 3.7%,
pair (e.g. "Pathogenic/Likely pathogenic" → 0.5/0.5) 2.4%, out_of_scope 0.10%.
VUS-high / -mid / -low (standalone and "Uncertain significance/VUS-high") map to VUS with a
tier modifier. "Likely pathogenic, low penetrance" keeps the class with a modifier.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| O-2 | HIGH | `vpdl/slm/cli.py` `cmd_evaluate` | The VUS-reclassification evaluation (EXP-015's scoring) always joined **zero** rows: the examples were cut to columns without `variant_id`, so documents were joined against variant ids, and `ranking_metrics` quietly returned `n = 0`. | **FIXED** — keeps `variant_id`, averages priority per variant, reports `variants_joined`, warns on an empty join; test `test_vus_outcomes_are_joined_per_variant` |
| H-2 | MEDIUM | `vpdl/slm/modeling/finetune.py`, `predict.py` | MC-dropout mutual information was computed on uncalibrated draws while entropy and margin beside it used calibrated probabilities. | **FIXED** — each draw goes through the fitted calibrator |
| E-1 | LOW | `vpdl/slm/labels.py` | NaN / `pd.NA` map to `out_of_scope`, not `missing`. Unreachable today. | OPEN |
| E-2 | LOW | `vpdl/slm/labels.py` | "Uncertain significance/Uncertain risk allele" (284 rows) is out_of_scope as a whole — the documented rule, but it slightly undercounts VUS. | Documented |
| E-3 | INFO | `docs/slm/GENOMIC_SLM_VUS.md:53` | Sub-tier counts are stale: local release has VUS-high 140, VUS-mid 102, VUS-low 46. | Documented |
| E-4 | LOW | `vpdl/slm/evaluation/explain.py:99` | An abstaining explanation reports `classification: "abstain"` and drops the argmax class from that field (probabilities are still emitted). | OPEN |
| — | INFO | `vpdl/slm/interface.py` | `score` / `pathogenic_probability` = P + LP (not renormalised against VUS) while evaluation uses (P+LP)/(P+LP+LB+B). Deliberate and documented in the interface docstring; a fusion layer must use the right one. | Documented |

## Verified correct (numerically)

Entropy (nats): one-hot → 0, uniform-5 → ln 5. Margin = top1 − top2. Mutual information
`H(mean) − mean(H)` clamped ≥ 0: one draw → 0, identical draws → 0, two opposite one-hot
draws → ln 2. Gorodkin's multi-class MCC. Abstention reasons are attributed correctly.

## Deferred

VUS reclassification scoring needs an archived ClinVar release (not downloaded).
Abstention thresholds are not yet chosen on validation data — do that before reporting any
abstention number.
