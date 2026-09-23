# Architecture of the genomic SLM

Code: `vpdl/slm/modeling/`. Status: **implemented and unit-tested on tiny
random-weight models; no real backbone has been loaded or trained here.**

## 1. Shape

```
  narrative text (conclusions masked) ─► backbone ─► pooled text vector ┐
  structured genomic fields ─► embeddings + scaled numbers + masks ─────┼─► fuse ─► slm_embedding
  DLRepresentation (optional, with a presence bit) ─────────────────────┘            │
                                                                                     ├─► class head        P / LP / VUS / LB / B
                                                                                     └─► ACMG head         28 codes, multi-label,
                                                                                                           masked per gene specification
  one evidence sentence ─► backbone ─► unit vector ─► evidence-type head (multi-label)
                                                   └─ polarity head (pathogenic / benign / neutral / mixed)
```

Heads are switched on per experiment, so "does this head help?" is an ablation
(EXP-012, EXP-013, EXP-014) rather than an assumption.

## 2. Backbones

One spec string chooses the model: `hf:<id or dir>` (any Hugging Face encoder),
`slm:<dir>` (a model saved by `vpdl slm-train` or `vpdl-slm pretrain`),
`tiny-bert` / `tiny-llama` (random weights, for tests and dry runs).
Named baselines, with licence tags checked on Hugging Face 2026-09-22:

| Name | Id | Licence tag | Lineage |
|---|---|---|---|
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | none on the card — confirm before release | BERT → PubMed/PMC |
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | MIT | from scratch on PubMed + PMC |
| `bioclinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | MIT | BioBERT → MIMIC-III notes |

The intended path is the plan's: **general → biomedical → broad genomic
continued pretraining → custom genomic SLM → task training.** A model trained
from scratch is not the default, because this project's own from-scratch plan
(docs/slm/PLAN.md) already states the honest expectation at DGX scale. The
from-scratch decoder remains a first-class arm (`slm:runs/slm/small/final`), so
"was starting from biomedical weights worth it?" stays an experiment.

## 3. Structured genomic input

Structured fields enter as their own modality — embeddings for categories,
standardised numbers with a **missing mask** — not pasted into the text, so an
ablation can switch them off cleanly and the leakage audit can see exactly
which ones a run used.

On by default: consequence, variant type, origin, chromosome class, whether a
protein change exists, log10 allele frequency (masked when absent — which is
most of ClinVar today), DL score.
Off by default: gene identity (a shortcut), collection method (a provenance
shortcut).
Forbidden, and refused by the encoder: review status, stars, number of
submitters, submitter, and anything in `NEVER_INPUT`.

## 4. The DL representation as one more modality

`dl_dim` comes from the DL export itself; a variant without a DL record gets a
zero vector **and a mask bit of 0**, never an imputed embedding — the same rule
the DL branch's own fusion uses for missing modalities. `dl_dropout` randomly
drops the modality during training so the text path cannot come to depend on
it. This is EXP-018, an ablation of the SLM — **not** the project's final
multimodal fusion, which is explicitly out of scope for this phase.

## 5. Heads and losses

* **Class head** — five logits. The target is a probability vector: one-hot for
  a five-class label, 0.5/0.5 for ClinVar's "Pathogenic/Likely pathogenic" and
  "Benign/Likely benign" (never forced onto one side). Optional class weights
  and label smoothing; an optional ordinal term (squared earth-mover's distance
  on the ordered scale) so a far mistake costs more than a neighbouring one.
* **ACMG head** — 28 multi-label logits, supervised only on documents that
  state codes, and masked to the codes the gene's specification allows
  (`GENE_SPECS`, both recorded as `verified=False` until checked against the
  ClinGen CSpec registry).
* **Evidence heads** — per sentence: multi-label type, single-label polarity.
  Their targets are rule-derived weak labels; the loss weights are low by
  default and their value is an ablation.

Every term sees only the rows that carry its supervision, so adding a head
never silently trains on zeros.

## 6. Parameter-efficient tuning

`frozen | lora | adapters | last_n | partial | full`, reusing the DL branch's
implementation for encoders (same strategies, same refusal to fully fine-tune a
large backbone on few labels) and the same strategies for decoders. Every run
records trainable and total parameters and the trainable fraction.

## 7. Training

Supervised training runs on `vpdl.dl.trainer.Trainer`: bf16 where supported,
gradient accumulation with a correct short last window, warm-up + cosine, early
stopping on a validation score, and checkpoints holding model, optimiser,
scheduler, scaler, epoch, step, best weights and every RNG. The dataset
version, split hash and head configuration are part of the checkpoint's
identity, so resuming against different data is refused rather than silently
mixing two experiments.

Validation score: pathogenic-vs-benign ROC-AUC when the validation split has
both classes, else macro F1, else negative Brier — chosen once, recorded in the
config.

## 8. Size, decided by measurement

`vpdl-slm sizing --corpus-tokens … --examples … --memory-gib …` lists
candidates with estimated memory and hours **for this corpus and this machine**
(`vpdl/slm/modeling/sizing.py`): a biomedical encoder base (~110M), a large one
(~335M), and the from-scratch decoders (~110M / ~340M). It prints its
assumptions (6·N·D + attention FLOPs, ~16 bytes per trainable parameter,
measured TFLOPS) and labels every figure an estimate; the runbook's
`pretrain --benchmark` replaces them with measurements. No size is hard-coded
anywhere, and the decision waits for EXP-009/010.

## 9. Output

`SLMRepresentation` (see GENOMIC_SLM_DL_INTERFACE.md): the fused embedding, an
evidence embedding, five calibrated probabilities, pathogenic/benign/VUS
aggregates, uncertainty and its method, abstention, the evidence used, the
explanation, ACMG context, and model/feature/dataset versions.
