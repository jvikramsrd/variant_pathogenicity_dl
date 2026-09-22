# Pretraining design (DL branch)

Status: **built and unit-tested on a tiny random model; not run.** No protein
foundation model is trained from scratch here. Every arm starts from a
published ESM checkpoint.

```
pretrained ESM (P0) ──► continued MLM on MMR proteins (P1)
                    ──► + variant-aware objectives (P2, P3, P4)
                    ──► the SAME downstream protocol for every arm
```

## Corpus (`vpdl-dl corpus`)

* **Content:** the four panel proteins plus mismatch-repair homologs across
  species — UniProt reviewed entries in Pfam MutS (PF00488) and MutL (PF01119),
  and entries with keyword KW-0234 (DNA mismatch repair). Queries are pinned in
  `vpdl/dl/pretrain/corpus.py::CORPUS_QUERIES` and recorded in the manifest.
* **Filters:** exact-duplicate removal; > 5 % non-standard residues dropped;
  < 50 residues dropped.
* **Validation split:** 5 % by sequence hash, for early stopping on MLM loss
  (label-free). Near-duplicates may straddle it; that only affects the stopping
  estimate, never the downstream evaluation.

### The leakage rule

Pretraining uses no labels, but under leave-one-gene-out it can still adapt the
model to the held-out protein's family.

| mode | corpus | models | use |
|---|---|---|---|
| `strict` (default) | per fold: remove the held-out gene and every corpus sequence ≥ 50 % identical to it (its orthologs; 3-mer prefilter, then alignment) | one per arm **per fold** | reported P1–P4 results |
| `transductive` | everything, including all four panel proteins | one per arm | a cheaper comparison, labelled as such — no more exposure than ESM's own UniRef pretraining |

Paralogs (≈ 26 % identity) stay in a strict corpus, consistent with
leave-one-gene-out keeping them in supervised training. A per-fold backbone is
used through a `perfold=<mapping.json>:<representation>` embedding block; each
fold's entry records its corpus hold-out, and `run_cell` refuses a fold whose
backbone did not hold out that fold's gene (`LeakageError`, tested).

## Objectives (`vpdl/dl/pretrain/objectives.py`)

Each has its own weight; every weight defaults to **zero**, MLM included, so an
arm is exactly what it names.

| objective | definition | passes / sample |
|---|---|---|
| `mlm` | 15 % of residues; 80 % `<mask>`, 10 % random, 10 % kept (ESM/BERT) | 1 |
| `mutation_position` | one synthetic substitution per crop; per-token logit, softmax over residues, CE to the substituted position | 1 (shared VT pass) |
| `substitution` | at the substituted position, predict the ORIGINAL residue from the variant context | shares the VT pass |
| `contrastive` | WT/VT triplet on site embeddings: conservative substitution (BLOSUM62 ≥ 1) closer to WT than radical (≤ −2) by margin 0.2 (cosine distance) | 3 (WT + 2 VT) |
| `paired` | WT/VT paired regression of BLOSUM62/4 from `[h_wt, h_vt, h_vt − h_wt]` | 1 WT + shared VT |

Label-free by construction: substitutions are synthetic and positions uniform —
never ClinVar positions, which would leak where pathogenic variants cluster.
Severity comes from BLOSUM62, a generic matrix with no knowledge of these genes.
The contrastive and paired objectives inject BLOSUM's view of severity; whether
that helps, hurts, or re-learns what ESM already knows is what the ablation
measures.

## Arms

| arm | objectives |
|---|---|
| P0 | none — the original checkpoint |
| P1 | mlm 1.0 |
| P2 | mlm 1.0 + mutation_position 0.5 + substitution 0.5 |
| P3 | mlm 1.0 + contrastive 0.5 |
| P4 | P2 ∪ P3 |

`paired` is available (`--objectives '{"paired": 0.25}'`) but in no preset arm:
objectives are not all combined by default.

## Training

LoRA (rank 16, α 32) on query/key/value/dense of every layer by default — the
frozen base cannot drift far from P0, and checkpoints hold only trainable
weights. `last_n` / `partial` are the alternatives; adapters are refused for
pretraining because their deltas cannot be merged. 512-residue random crops,
bf16, gradient checkpointing, warmup + cosine, early stopping on held-out MLM
loss (fixed masks, so epochs are comparable). Resumable (`--resume`), with
batch contents keyed on the batch's own indices, so a resumed run replays the
same crops, substitutions and masks.

Output per arm/fold: `backbone_delta.pt` (format `vpdl-dl-backbone-delta-v1`)
+ `pretrain_summary.json` + a registry record. `load_adapted_backbone` applies
the delta, merges LoRA into the base weights and unwraps it, so downstream
fine-tuning applies its own strategy on a plain backbone.

## Downstream protocol (identical for P0–P4)

1. `vpdl-dl embed --arm Pk --fold G --adapted-from runs/dl/pretrain/Pk-G --blocks-file runs/dl/pretrain/Pk_blocks.json`
2. `vpdl-dl zeroshot` with the same backbone (masked marginal)
3. probe / fusion cells with `--embedding-blocks "perfold=runs/dl/pretrain/Pk_blocks.json:site.concat4"`
4. `vpdl paired --reference <P0 arm>` — ΔAUC on identical variants, 3 seeds
5. `vpdl-dl failure --paired <table>` lists any arm significantly worse than P0

A pretraining arm that loses to P0 is a documented regression and stays in the
results; nothing is deleted.

## What would make a positive result believable

* The gain appears under `strict` corpora, not only `transductive`.
* It survives three seeds and the paired CI excludes zero.
* It appears for the probe AND the zero-shot score (a representation change,
  not a head artefact).
* It is not confined to one gene (in particular not only MSH6, whose context
  policy is itself a confound).
