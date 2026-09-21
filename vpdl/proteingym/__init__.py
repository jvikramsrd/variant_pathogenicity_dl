"""ProteinGym per-assay benchmark: reproduce published results, then combine.

Step 1 of the combined-vs-individual study over ProteinGym's 217 substitution
assays. Before any claim about combining datasets, each dataset's own model has
to reproduce the number ProteinGym publishes for it — the same principle as the
v1 oracle check, applied to a published benchmark.

Protocol, taken from ProteinGym's code rather than assumed (checked 2026-09-21):

* Folds: ProteinGym's own assignments (``fold_random_5`` / ``fold_modulo_5`` /
  ``fold_contiguous_5``) from ``cv_folds_singles_substitutions.zip``.
* Metric: ONE Spearman per assay, over all out-of-fold predictions pooled
  (``proteingym/merge_supervised.py``, line 112) — not a mean of per-fold values.
* One-hot baseline: ProteinNPT's ``OHE_not_augmented`` config — a linear model on
  the one-hot-encoded mutated sequence.
"""
