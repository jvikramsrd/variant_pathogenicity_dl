# DL branch architecture

Status (2026-09-22): **built and unit-tested; nothing trained.** Every number
this branch will produce comes from the DGX Spark (see
[DGX_SPARK_RUNBOOK.md](DGX_SPARK_RUNBOOK.md)).

## The one design rule

Every DL model is **a model in the existing protocol**, not a new pipeline.
It is registered in `vpdl.models.MODELS` and run by `vpdl.experiment.run_cell`,
so it is scored on the **same held-out ClinVar variants**, with the same
inner-validation threshold, the same bootstrap metrics and the same provenance
block as the existing `mlp` / `bilstm` / `gbm` arms. That is what makes
"new DL model vs existing DL model" a paired comparison (`vpdl paired`) rather
than two numbers from two protocols.

`run_cell`'s original leave-one-gene-out behaviour is unchanged: predictions and
inner-validation predictions for `mlp` and `bilstm` are byte-identical to a
snapshot taken before the change (the results CSVs only gained `f1`, `brier`,
`ece` columns).

## Layers

```
sources/ (ClinVar, gnomAD, AlphaMissense, ProteinGym, UniProt)      existing
   │  iter_variant_summary / iter_gnomad_records: every record, not first-wins
   ▼
assemble.py  ── one row per substitution, per-source labels ──────── existing
   ▼
dl/canonical.py  identity, HGVS, genomic coords, QC, PMS2 rule,       new
                 functional values (validation only), clusters, splits
   ▼
dl/structure.py, dl/genomic.py  → feature_struct_*, feature_genomic_*  new
dl/plm/embed.py, dl/plm/zeroshot.py → dl/feature_store.py              new
   ▼
experiment.run_cell  (split scheme, embedding blocks, leakage gate)    extended
   │  models: gbm mlp bilstm (existing) | aa_mlp cnn bilstm_attn
   │          transformer (dl) | fusion (dl) | plm_finetune (dl)
   ▼
predictions / valpreds / scores / representations / summary+provenance
   ▼
vpdl paired · dl/calibration · dl/functional · dl/failure · dl/interface
```

## Modules

| module | role |
|---|---|
| `dl/hgvs.py` | variant identity (`uniprot:pos:wt>mut`, = `vpdl.splits.variant_keys`), HGVS p./c. normalisation, MANE transcripts, c./p. codon consistency |
| `dl/canonical.py` | canonical table + QC report + flagged records; see [DATASET.md](DATASET.md) |
| `dl/homology.py` | vectorised Gotoh aligner (BLOSUM62), paralog families, homology clusters, homologous-twin counts |
| `dl/functional.py` | CIMRA / MaveDB / continuous DMS as a validation table; orientation check; functional validation |
| `dl/splits.py` | `logo`, `logo_purged`, `family`, `random_debug` folds |
| `dl/leakage.py` | leakage checks, report, and the gate `run_cell` calls |
| `dl/context.py` | long-chain policies: `full`, `sliding`, `centered`, `asymmetric`, `hierarchical` |
| `dl/plm/backbones.py` | ESM-1b, ESM-2 (8M–3B), `existing` (= ESM-2 650M, v1's backbone); alphabet checked against the checkpoint |
| `dl/plm/embed.py` | WT/VT site, local-window and global vectors; representations derived at load |
| `dl/plm/zeroshot.py` | true masked marginal + wild-type marginal |
| `dl/plm/peft.py` | frozen / LoRA / adapters / last-N / partial / full (guarded) |
| `dl/plm/finetune.py` | siamese PLM fine-tuning (frame model) |
| `models/seqwin.py` | sequence baselines: AA-embedding MLP, CNN, BiLSTM+attention, Transformer |
| `dl/fusion.py` | modality projection → concat (or gate) → MLP; masks, modality dropout, MC-dropout |
| `dl/structure.py` | WT features from AlphaFold; VT via a precomputed-structure provider |
| `dl/genomic.py` | exon-structure features; re-derives the PMS2 homology codon range |
| `dl/pretrain/` | MMR corpus, label-free objectives, P0–P4 ([PRETRAINING_DESIGN.md](PRETRAINING_DESIGN.md)) |
| `dl/trainer.py` | one torch loop: bf16, accumulation, fused AdamW, compile, early stop, exact `--resume` |
| `dl/feature_store.py` | versioned, memory-mapped feature cache |
| `dl/calibration.py` | ECE, Brier, slope/intercept, reliability, Platt/temperature/isotonic |
| `dl/failure.py` | shortcut / overfitting / MSH6 / imbalance / calibration / pretraining-regression checks |
| `dl/tracking.py` | run registry (`runs/dl/registry.jsonl`) |
| `dl/report.py` | every run -> paper tables (CSV / Markdown / LaTeX), comparability checks, checksummed manifest |
| `dl/interface.py` | the frozen output contract |
| `dl/cli.py` | `vpdl-dl` (separate from `vpdl/cli.py`, which hosts the LLM commands) |

## Models

| name | inputs | what it tests |
|---|---|---|
| `gbm`, `mlp`, `bilstm` | unchanged | the existing-model baseline |
| `aa_mlp`, `cnn`, `bilstm_attn`, `transformer` | WT/VT ±7 windows + tabular | does another sequence inductive bias find signal the BiLSTM did not? |
| `mlp` / `gbm` + `--embedding-blocks` | tabular + cached PLM vectors | frozen-PLM probe (VariPred recipe) |
| `fusion` | named modalities → projection → concat/gate → MLP | multi-modal DL fusion |
| `plm_finetune` | variant rows + sequences (+ tabular) | end-to-end PLM with PEFT |
| zero-shot | no training (`vpdl paired --feature-baseline`) | masked-marginal floor |

## WT / VT representations

For every variant six raw vectors are cached: `{site,local,global}_{wt,vt}`.
Representations are derived at read time as `<level>.<pairing>` with pairing in
`wt | vt | vt_minus_wt | abs_diff | wt_plus_vt | concat4`, so comparing them is
reading one cache five ways, never re-extraction. The variant window is always
cut from the mutated chain (landmine L12, re-tested for every policy).

## MSH6

MSH6 is the only chain over 1,022 residues. Nothing is truncated by default:
`full` (ESM-2 only; extrapolation beyond the training crop, measured not
assumed), `sliding` (exact: only windows containing the mutation are
recomputed), `centered`, `asymmetric` (VariPred) and `hierarchical` (centred
local + sliding global) are all first-class and recorded in the feature
store's metadata. Because the choice affects one gene only, it is a
gene-identity confound under leave-one-gene-out and is reported as an axis.

## Integration boundary (future reasoning branch)

Not implemented; only the contract exists (`vpdl/dl/interface.py`):

```
DL branch ──► DLRepresentation          (variant_id, embedding, score, uncertainty, metadata)
future    ──► ReasoningRepresentation   (the same shape, produced by nothing here)
future fusion layer: one more modality in build_fusion_net — no DL model changes
```

The export (`vpdl-dl export`) writes JSONL records validated against
`DL_OUTPUT_SCHEMA` (`schema_version = dl-output/1.0`):

```json
{"schema_version": "dl-output/1.0", "variant_id": "P43246:10:A>V", "gene": "MSH2",
 "embedding": {"file": "dl_outputs_<cell>.embeddings.npy", "row": 0}, "embedding_dim": 128,
 "pathogenicity_score": 0.87, "uncertainty": 0.08, "uncertainty_method": "mc_dropout",
 "model_version": "<cell slug>@<git commit>", "feature_version": "<feature-store entries>",
 "dataset_version": "<canonical sha256>", "fold": "MSH2", "metadata": {}}
```

`pathogenicity_score` is always an **out-of-gene** prediction (from the fold that
held that gene out) and is a research output, not a clinical classification.
Clinical text is not used anywhere in the DL branch.
