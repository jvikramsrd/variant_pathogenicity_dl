# DL experiment plan

Status: **design only — no DL result exists yet.** Every table here is a shape
to fill from `vpdl paired` / cell summaries on the DGX, never typed by hand.
Commands: [DGX_SPARK_RUNBOOK.md](DGX_SPARK_RUNBOOK.md).

## Goal

Stronger generalisation under a leakage-safe protocol — not a higher number
than a paper. Published results are context: MVmamba reports AUC 0.901 / MCC
0.656 on 18,731 genome-wide variants; that is a different task (random splits,
many genes) and is not a bar this four-gene leave-one-gene-out protocol is
expected to clear. Claims are made only against baselines run here, on the
same held-out variants.

## Fixed protocol (every cell)

* Table: `data/built/canonical_full.csv` — one table, one hash, one test set.
* Train on ClinVar (≥ 2★, QC-passed) unless the arm says otherwise; **score on
  the same held-out ClinVar variants** (`eval_source = clinvar`).
* Primary split `logo` (MLH1+MSH2+MSH6→PMS2, …, MSH2+MSH6+PMS2→MLH1); secondary
  `family` and `logo_purged`; `random_debug` only for debugging.
* Threshold from inner validation (training genes only). Three seeds (42/43/44)
  minimum. 10,000 bootstrap resamples.
* Metrics: **MCC primary**; ROC-AUC (the fair cross-arm comparison — no
  threshold), PR-AUC, F1, sensitivity, specificity, Brier, ECE. PMS2 (n ≈ 21,
  4 benign) is reported but excluded from headline means (`n < 50`).
* Differences: paired ΔAUC with bootstrap CI on identical variants
  (`vpdl paired`), stratified mean over informative genes. A difference smaller
  than the seed spread (MCC SD up to 0.056 here) is not a finding.

## The matrix

| block | arms | reference |
|---|---|---|
| **R — existing models** | gbm, mlp, bilstm (+ gbm early-stopped) on the six published features | RUNLOG 2026-09-21 numbers |
| **S — sequence baselines** | aa_mlp, cnn, bilstm_attn, transformer (± tabular); sequence-only | existing mlp, bilstm |
| **Z — zero-shot** | masked vs wild-type marginal × ESM-1b / ESM-2 650M | AlphaMissense feature baseline |
| **B — backbones** | existing (= ESM-2 650M) vs ESM-1b; optional ESM-2 150M | ESM-2 650M probe |
| **W — WT/VT representations** | site/local/global × wt, vt, vt−wt, abs(vt−wt), wt+vt, concat4 | site.concat4 |
| **L — MSH6 context** | full, centered, asymmetric, hierarchical, sliding (MSH6 fold only differs) | full |
| **A/B — population** | fusion without vs with population features | Model A |
| **M — modality ablation** | sequence (PLM) only, structure only, genomic only, population only; PLM + structure / population / genomic; PLM + structure + population; all DL modalities | all DL modalities |
| **F — fusion form** | concat vs gated | concat |
| **T — fine-tuning** | frozen, LoRA, adapters, last-N (full only with evidence) | frozen probe |
| **P — pretraining** | P0, P1, P2, P3, P4 (strict per fold; transductive P1 as a cost comparison) | P0 |
| **X — splits** | logo vs family vs logo_purged for the best tabular and best PLM arm | logo |

External priors (AlphaMissense) are not a DL modality: they appear only in
block R (continuity with the published arms) and as a zero-training baseline.
"All DL modalities" means sequence/PLM + structure + genomic + population (+
UniProt annotation where present).

## Regression path (existing vs new)

Block R re-runs the existing models on the canonical table, so every new arm
has a paired comparison against them on identical variants. If a new model is
worse, the table says so and the old model stays the reference; nothing is
deleted. Differences between block R and the RUNLOG numbers are attributable to
canonical QC (`label_changes_vs_assembled`, discordant ClinVar records now
withheld) and are reported with those counts.

## Independent validation

* **Functional** — held-out-gene scores (`--score-rows functional`) against the
  continuous MSH2 DMS assay, and CIMRA OddsPath when the CSV is supplied:
  Spearman with CI, and class AUC where the assay defines classes. Only under
  gene-disjoint splits (the gate refuses otherwise). Note that the MSH2 assay is
  also a label source in pooled arms; functional validation is reported for
  ClinVar-trained arms only.
* **Calibration** — per gene, fitted on inner validation (temperature / Platt /
  isotonic): ECE, Brier, slope, intercept, reliability diagrams.
* **Uncertainty** — MC-dropout SD per variant (fusion) and seed-ensemble SD; the
  check is that higher uncertainty goes with lower accuracy (risk–coverage).

## Failure analysis (run on the best arm of each block)

| failure | how it is checked |
|---|---|
| gene shortcut | nearest-centroid gene accuracy from features; per-gene base-rate drift |
| population shortcut | AUC among gnomAD-absent vs observed variants; AF-alone AUC |
| sequence leakage | homologous twins per fold; logo vs logo_purged vs family |
| duplicate variants | leakage report (critical) |
| overfitting | inner-validation vs held-out AUC gap |
| MSH6 truncation | MSH6 AUC within vs beyond residue 1,022; context-policy block L |
| class imbalance | per-gene prevalence (= random PR-AUC) beside every PR-AUC |
| poor calibration | per-gene ECE and slope |
| pretraining regression | P1–P4 paired vs P0 with CI excluding zero (flagged) |

## Order and stopping rules

1. R and Z first — they bound everything else.
2. W and B on frozen embeddings (cheap), then M and A/B.
3. T only if frozen probes beat the tabular baseline or come close; start with
   LoRA; full fine-tuning never without T's evidence.
4. P only after T (pretraining changes the backbone every later block uses);
   transductive P1 first; strict P1 for all folds; P2–P4 only if P1 is not a
   regression.
5. X for the final best arms.

A negative result is a result: if PLM representations do not beat six tabular
features under leave-one-gene-out, that is the finding, stated with its CI.
