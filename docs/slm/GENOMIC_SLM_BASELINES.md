# Baselines — the numbers every claim has to clear

Code: `vpdl/slm/baselines.py`. Command: `vpdl-slm baselines --examples … --kinds …`.
Status: **implemented and tested on synthetic data; not run on real data.**

## 1. The list

| # | Baseline | What it tests | How it is run |
|---|---|---|---|
| 1 | Majority | the floor: predicts the training class distribution | `--kinds majority` |
| 2 | Per-gene label rate | the gene shortcut — reported by the leakage audit itself | automatic (`gene_shortcut`) |
| 3 | TF-IDF + logistic regression | how much of the task is surface wording | `--kinds tfidf_lr` |
| 4 | TF-IDF + linear SVM | the same, different loss | `--kinds tfidf_svm` |
| 5 | Structured features only, no text | the other half of the clinical-text ablation | `--kinds structured_lr --records …` |
| 6 | BioBERT | biomedical encoder, no genomic pretraining | EXP-004 |
| 7 | PubMedBERT (BiomedBERT) | idem | EXP-005 |
| 8 | BioClinicalBERT | clinical-note encoder | EXP-006 |
| 9 | ClinVar-BERT-style | BioBERT + this project's conclusion/template pipeline | EXP-007 |
| 10 | The existing project LLM (`medgemma:27b` in the knowledge base) | context, not a classification arm (below) | `vpdl kb-eval` |
| 11 | Custom SLM, and each ablation on top of it | EXP-009 … EXP-020 | see the experiment plan |

## 2. Fairness rules

* Same examples, same split, same calibration protocol, same metrics — the
  baselines go through the same `five_class_metrics` / `binary_metrics` /
  `calibration_summary` code as the neural arms, and their thresholds come
  from validation too.
* Same tuning budget. A baseline is not left at a default while a neural arm
  is searched: the regularisation grid for the linear models and the learning
  rate / epochs for the neural arms are declared in their configs and reported.
* Everything is recorded per run: dataset, split hash, parameters, trainable
  parameters, tokenizer, hyper-parameters, seed, hardware, command, metrics,
  checkpoint (`runs/slm_genomic/registry.jsonl`).

## 3. Why the linear models are fitted here rather than by scikit-learn

The TF-IDF vectoriser is scikit-learn's. The classifiers are fitted with SciPy
L-BFGS on the sparse matrix, because on the Windows machine this repository is
developed on an Application Control policy blocks scikit-learn's compiled
`_loss` module, so `sklearn.linear_model` cannot be imported at all (the DL
audit records the same four test failures). The objectives are the standard
ones — L2 multinomial logistic regression and one-vs-rest L2 squared-hinge —
and `--backend sklearn` runs scikit-learn's own estimators where they import,
so the choice can be checked rather than trusted.

## 4. The knowledge base is not a classification baseline

`vpdl kb-ask` retrieves passages, answers with citations, and refuses when the
sources do not answer. Its rule 4 is that it **never classifies a variant**
(docs/kb/DESIGN.md). It appears in the matrix as EXP-008 for context — what the
current system does on the same questions — and its numbers are not comparable
with a classifier's. Two further reasons to keep them apart: its evaluation set
became a development set after the prompt was changed (docs/RUNLOG.md,
2026-09-22), and `medgemma:27b` reads retrieved passages rather than a single
submitter's narrative.

## 5. Preserved, not replaced

Nothing in the v1 or v2 codebase was deleted for this branch. The from-scratch
SLM pipeline, the knowledge base, the DL branch and the v1 archive all still
run as they did; the new commands live under `vpdl-slm`.
