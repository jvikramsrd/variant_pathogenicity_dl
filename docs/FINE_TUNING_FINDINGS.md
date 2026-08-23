# Fine-tuning findings and implementation notes

*Prepared 2026-08-23. No training, feature extraction, or tests were run as
part of these changes.*

## Evidence from the current dataset and saved OOF artefacts

- `extended_dataset_train.csv` has 190,494 labelled substitutions.  Only
  5,152 (2.7%) have a ClinVar or ProteinGym-clinical label; approximately
  188k have a DMS aggregate.  A loss treating every row equally therefore
  optimizes assay fitness much more strongly than clinical pathogenicity.
- The leakage-clean `ext80` priors run reports clinical-only ROC-AUC 0.961
  (temperature) and 0.963 (isotonic), compared with 0.945 for AlphaMissense.
  This is promising but is a *new-residue-in-known-protein* estimate because
  the current outer folds group by `uniprot_id:position`, not protein.
- Most `zs_*` columns are absent for roughly 99% of rows.  Median imputation
  without an indicator previously made "not scored" indistinguishable from a
  real median prediction.
- The prior calibration implementation fitted temperature and isotonic models
  on each outer validation fold and then scored that same fold.  Calibration,
  Brier, ECE, and threshold metrics were therefore optimistic.  Isotonic ECE
  of exactly zero is a symptom of this in-sample fitting.
- The prior default focal alpha was 0.25 even though positives are 43% of all
  labels and about 52% of clinical labels.  This down-weighted the positive
  class and was not a sensible default for the intended clinical target.

## Changes made

1. **Nested calibration and model selection.** Each outer validation fold is
   now untouched.  Its outer-training portion is group-split into model-fit,
   early-stopping, and calibration partitions.  Both calibrators are fit only
   on the dedicated calibration partition and then applied to the outer fold.
2. **Clinical-aware optimization.** `train_extended.py` now gives clinical
   labels a configurable loss weight (`--clinical_weight`, default 5.0) and
   monitors clinical-only ROC-AUC for early stopping whenever both classes are
   present. DMS-only rows remain useful, but cannot numerically dominate the
   objective as completely.
3. **Stable losses.** Added ordinary BCE as the default.  Weighted BCE now
   uses a fixed class ratio from the fitting partition rather than a different
   ratio in every mini-batch.  Focal loss remains available for comparison,
   with neutral default alpha 0.5.
4. **Missingness features.** Every numerical prior is now accompanied by an
   `is_missing_*` indicator before median imputation.
5. **Fixed ESM-prior assembly.** The `esm+priors` extraction pool previously
   retained only metadata columns. As a result, its prior matrix had zero
   columns and the advertised external priors were not appended to ESM
   embeddings. The pool now carries the raw prior columns for both labelled
   substitutions and VUS rows; the changed cache width intentionally creates
   a new feature-cache key.
6. **Tunable capacity.** Exposed `--n_blocks`; revised the extended-trainer
   starting point to 256 hidden units, 0.15 dropout, 60 maximum epochs,
   patience 10, and learning rate 3e-4. These are starting values, not claimed
   improvements until evaluated.

## Recommended experiment order

1. Compare the old equivalent configuration to the new defaults under the
   same 5-fold residue-grouped split.  Treat this only as a regression check.
2. Run **leave-one-protein-out** clinical evaluation. This is the preferred
   model-selection criterion for deployment to new disease genes.
3. Run ablations: AlphaMissense only; all priors; all priors without each
   source family; ESM/PLLR only; ESM plus priors. Record availability alongside
   performance for sparse `zs_*` features.
4. Train a multitask model: continuous, within-assay-normalized DMS fitness as
   one task and high-confidence clinical pathogenicity as another. Fine-tune
   the clinical head after DMS pretraining rather than treating DMS bins as the
   same target.
5. Hold out a temporal or genuinely external clinical set before making a
   headline claim. In particular, check overlap between evaluation labels and
   the training data of AlphaMissense and supervised `zs_*` predictors.

## Important limitations still not solved in this change

- The dataset needs a protein-level split implementation for an honest unseen-
  gene estimate; residue-level grouping remains appropriate only for local
  generalization within proteins.
- DMS binarization remains a proxy target. The recommended multitask/
  continuous-fitness design needs a new model head and data interface.
- Source provenance should be used to create a high-confidence clinical set
  (for example, expert-panel/practice-guideline ClinVar assertions) and a
  lower-confidence, down-weighted training set.
