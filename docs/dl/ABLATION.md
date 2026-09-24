# DL ablation study — each data source, and all combined, against every model

One script, two axes, 3 seeds, the same evaluation throughout: leave-one-gene-out over
MLH1 / MSH2 / MSH6 / PMS2, scored on the held-out gene's **ClinVar** labels. Arms differ
only in what they train on (landmine L17), so every arm is comparable with every other.

```bash
bash scripts/dl_ablation.sh dry     # seconds: every arm's table, folds, leakage gate and models
bash scripts/dl_ablation.sh run     # the grid, then the paper tables (results/dl)
```

`run` is resumable: finished cells are skipped (`--skip-existing`), so after an interruption
run the same command again. The log goes to `runs/dl/ablation_<time>.log`.

## Models (7)

Feature models: `gbm`, `mlp`. Sequence-window models (use the protein window plus whatever
features the arm allows): `aa_mlp`, `cnn`, `bilstm`, `bilstm_attn`, `transformer`.

## Axis A — feature sources (training labels: ClinVar) → `runs/dl/ablation_features`

| Arm | Features | Arm name in the tables |
|---|---|---|
| A0 | none (sequence models only) | `train-clinvar__drop-domains-genomic-gnomad-prior_scores-structure` |
| A1 | gnomAD population + ACMG frequency flags | `train-clinvar__drop-domains-genomic-prior_scores-structure` |
| A2 | AlphaMissense | `train-clinvar__drop-domains-genomic-gnomad-structure` |
| A3 | AlphaFold structure | `train-clinvar__drop-domains-genomic-gnomad-prior_scores` |
| A4 | genomic / exon position | `train-clinvar__drop-domains-gnomad-prior_scores-structure` |
| A5 | all sources combined | `train-clinvar` |
| (main) | gnomAD + AlphaMissense (already run) | `train-clinvar__drop-domains-genomic-structure` in `runs/dl/main` |

## Axis B — training-label sources (all features) → `runs/dl/ablation_labels`

| Arm | Trained on | Rows with a training label |
|---|---|---|
| A5 | ClinVar | 449 (all four genes) |
| B1 | ProteinGym DMS only | 16,748 — **MSH2 only** |
| B2 | ClinVar + DMS combined | 449 + 16,748 |
| B3 | ClinVar expert-panel only | 252 |

## What the numbers do and do not license — read before writing

- **Three seeds are the minimum.** A difference smaller than the seed spread is not a result
  (a single-seed 0.007–0.009 AUROC "win" vanished at three seeds on 2026-09-06).
- **B1 is confounded with gene identity.** Every DMS label is MSH2, so the MSH2 fold has
  nothing to train on (the cell reports it as untrainable, by design) and the other folds
  learn from one gene's assay. B1 measures "transfer from one MSH2 assay", not "DMS vs ClinVar".
- **B2 is dominated by DMS rows** (16,748 vs 449). The earlier MMR pooling study found
  pooling DMS with ClinVar hurt held-out-gene prediction for GBM, MLP and BiLSTM; B2 is the
  replication in this pipeline. Read it with the paired table, per gene.
- **A2 is a joint arm.** AlphaMissense carries allele-frequency signal, so "AlphaMissense
  without gnomAD" is not a clean removal; the run records `allow_proxy_leak`.
- **PMS2 is not scoreable** (21 ClinVar variants, 4 benign): its AUROC is reported but kept
  out of the headline mean.
- **Open data issue B-1** (docs/audit/DL_DATA_PIPELINE_AUDIT.md): the gnomAD frequency
  feature disagrees with the recomputed value on 7 of the 449 labelled rows; it touches A1,
  A5, B2, B3 and main.

## Save everything for the paper

The script's last step writes `results/dl` (CSV, Markdown and LaTeX tables, paired
comparisons, `checks.json`, `MANIFEST.json`). Then, as after every phase:

```bash
git add runs/dl results/dl docs/dl docs/RUNLOG.md
```

```bash
git status
```

Check: **no `deleted:` lines.**

```bash
git commit -m "dl results: ablation grid" && git push origin v2/rebuild
```

Add one `docs/RUNLOG.md` entry: the command, the outcome, where the files are, and the
caveats above.
