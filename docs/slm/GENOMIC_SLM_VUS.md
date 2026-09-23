# VUS and uncertainty — a first-class task, not "low confidence"

Code: `vpdl/slm/evaluation/vus.py`, `vpdl/slm/evaluation/uncertainty.py`.
Status: **implemented and unit-tested; no VUS result exists.**

## 1. Why VUS is its own class

2,372,536 of the 4,558,681 variants in the local ClinVar release are of
uncertain significance — 52%. Treating them as "cases the model was unsure
about" throws away the majority of the data and the actual clinical problem.
So VUS is one of the five classes the model predicts, and the model is asked
three separate questions:

1. **What is it?** — five calibrated probabilities.
2. **How sure is it, and why?** — entropy, margin, disagreement between passes,
   and how much informative evidence the input actually contained.
3. **Which VUS deserve attention first?** — a ranking, scored against what
   later happened to those variants.

## 2. Scores

```
priority         p(P) + p(LP)     how pathogenic-leaning a VUS is
benign_priority  p(LB) + p(B)
resolution       max of the two   how far from "uncertain" the model thinks it is
```

## 3. How a VUS ranking can honestly be scored

A VUS has no label, so it cannot be scored against one. It can be scored
against **what happened next**: take an older ClinVar release, keep the
variants that were VUS then, and look them up in a newer one. Those now
Pathogenic/Likely pathogenic are positives, those now Likely benign/Benign are
negatives, and those still VUS are not scored at all
(`reclassification_outcomes`).

Metrics: ROC-AUC and average precision of `priority`; precision and enrichment
over the base rate in the top-k (k = 10, 50, 100, 5%, 10%); calibration of
`priority` on the reclassified set.

**Needs an archived ClinVar release** — ClinVar publishes monthly archives, so
this is a download, not a new data source. Not downloaded here:
**TBD — RUN ON DGX SPARK**.

Two cautions that belong with any number this produces: variants that get
reclassified are not a random sample of VUS (they are the ones someone
studied), and a model trained on later ClinVar data would see the answer — so
this evaluation is run against a model trained on the older release only
(the temporal split exists for exactly this).

## 4. ClinVar's own VUS sub-tiers

The local release already carries `VUS-high` (48 variants) and `VUS-mid` (42) —
a sub-ordering by the submitters themselves, newly present in ClinVar and
previously unused in this project. They are parsed as VUS with the tier kept as
a modifier, and `subtier_agreement` reports the Spearman correlation between
the model's `priority` and that tier. Small today; a free, independent check as
it grows.

## 5. Uncertainty that is measured

Signals: predictive entropy, top probability, margin, mutual information across
MC-dropout passes or ensemble members, spread of the pathogenic score, and the
count of informative evidence units in the input.

An uncertainty is only useful if it predicts error, so `evaluate_uncertainty`
reports, for each signal, the ROC-AUC of that signal for identifying wrong
predictions, plus the risk–coverage curve and its area (AURC): the error rate
among the most confident fraction, at every coverage level.

## 6. Abstention

The model may decline. `abstain()` returns a mask **and the reasons**:

* `low_confidence` — top probability below a validation-chosen threshold;
* `high_disagreement` — mutual information above its threshold;
* `no_informative_evidence` — the input carried no evidence unit with a
  direction. A confident prediction from a narrative containing no evidence is
  a prediction about a template, and this is the case that catches it.

Thresholds are chosen on validation, like every other threshold here, and
abstention is reported as coverage against accuracy — never as a way to make a
headline number look better.
