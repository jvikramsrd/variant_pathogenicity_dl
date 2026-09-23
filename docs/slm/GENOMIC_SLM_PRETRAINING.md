# Broad genomic continued pretraining

Code: `vpdl/slm/modeling/continued.py`, `vpdl/slm/build/pretrain_corpus.py`.
Commands: `vpdl-slm pretrain-corpus`, `pretrain-pack`, `pretrain`
(`--dry-run`, `--benchmark N`). Status: **implemented; run only on synthetic
data with a tiny random model. Real pretraining is NOT RUN — REQUIRES DGX SPARK.**

## 1. Where this sits

```
general pretrained model → biomedical pretrained model → BROAD GENOMIC
CONTINUED PRETRAINING (here) → custom genomic SLM → task training
```

The objective follows the backbone: masked language modelling for an encoder,
causal for a decoder. Both are supported; `auto` picks by backbone kind.

## 2. The corpus

Built from what the project already has readers for, reusing
`vpdl.slm.corpus.iter_documents` unchanged:

* **PubMed abstracts** — retractions, retraction notices and expressions of
  concern dropped (`vpdl.slm.pubmed`), non-English dropped, every decision
  counted;
* **knowledge-base passages** — GeneReviews and whatever else `vpdl kb-build`
  indexed; passages marked `private` (owned textbooks) never enter weights;
* **ClinVar narratives** — only with `--include-narratives`, only for
  TRAINING-role variants, and only conclusion-masked.

Planned additions with public-domain or CC terms (docs/kb/SOURCES.md):
MedlinePlus Genetics, ClinGen curations, Orphanet, then PMC `oa_comm`.

### What is deliberately missing from it

A pretraining corpus is built once and reused by every experiment, so anything
an evaluation will later be scored on must not be inside it:

* narratives of **reserved variants** — the MMR test set, the temporal test
  set, the functional holdout, the unseen-gene panel;
* in strict mode, **PubMed abstracts whose PMID ClinVar cites for those
  variants** — the paper that describes the test variant;
* the independent functional datasets' own publications, always.

Both counts land in `stats.json` (`pubmed_excluded_cited_by_evaluation_variants`,
`narratives_excluded_reserved`), so the cost of the exclusion is visible.

This is variant-level, not topic-level: Lynch syndrome literature stays in the
corpus. Removing it would make the MMR transfer experiment measure a corpus
gap rather than transfer. The residual risk — a paper describing a test variant
that ClinVar does not cite for it — is stated, not solved.

## 3. Packing

`pretrain-pack --backbone <spec>` tokenises the corpus once with **that
backbone's** tokenizer into flat uint16 token files, and records the
tokenizer's fingerprint. `pretrain` refuses token files whose fingerprint does
not match the model it is about to train — a wrong pairing fails on the first
command instead of producing a model trained on another vocabulary's ids.

(Windows scale note: a vocabulary of 65,536 or more does not fit uint16 and is
refused outright, with the message saying so.)

## 4. The loop

Step-based, because a real run lasts days:

* checkpoints every `checkpoint_every` steps **and** at least every
  `checkpoint_minutes`; the last `keep_last` kept, written atomically;
* Ctrl-C / SIGTERM finish the current step, save, and exit;
* a resume continues from (seed, step), so the batches are exactly those an
  uninterrupted run would have seen; a resume with different settings is
  refused, and a resume onto a finished run says so and changes nothing;
* a non-finite loss stops the run with the last checkpoint intact;
* `run.json` records config, objective, device, GPU, precision, PEFT summary,
  data meta, torch version, git state and parameter count; `log.jsonl` records
  loss, learning rate, gradient norm, tokens/s and validation loss.

Those properties — and `TokenWindows`, the schedule, the checkpoint helpers and
the Python-headers check — are imported from `vpdl/slm/train.py`, the
from-scratch loop this project already runs on the DGX.

## 5. Masking

BERT-style: 15% of non-special positions, of which 80% become `[MASK]`, 10% a
random token, 10% unchanged; labels are `-100` elsewhere. The generator is
seeded from (seed, step, micro-batch), so masking is part of the deterministic
resume. Windows are cut from the packed stream, so a window can begin
mid-document — standard for domain-adaptive pretraining, and noted here rather
than left implicit.

## 6. What to watch, and what would say it worked

* validation loss (masked-LM or causal) falling steadily;
* gradient norm small and stable;
* throughput near the machine's benchmark (measured for the from-scratch small
  model: 34.7k tokens/s, 30.7 TFLOPS with `torch.compile`; the encoder's rate
  is **TBD — measure with `pretrain --benchmark 20`**).

Pretraining is not the result. The result is whether the continued-pretrained
backbone beats the same backbone without it on held-out genes (EXP-009 vs
EXP-010, hypothesis H1), on three seeds, with the leakage audit clean.

## 7. Size and budget

`vpdl-slm sizing` gives candidates and estimates for the measured corpus; the
decision is taken after EXP-009/010, not before. The from-scratch arm's
budget is already documented in docs/slm/PLAN.md (2.5B tokens ≈ 20 h measured
for the small model). For continued pretraining the Chinchilla ratio does not
apply — the backbone has already seen far more — so the budget is set by the
corpus (one to a few passes) and checked against validation loss.
