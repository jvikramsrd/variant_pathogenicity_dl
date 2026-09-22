# A small language model trained from scratch — plan

Status: **steps 1–4 built, not yet run on the DGX** (2026-09-22). Code:
`vpdl/slm/`; how to run: [RUNBOOK.md](RUNBOOK.md).

Measured on the DGX (2026-09-22): peak **90.1 TFLOPS** BF16 (matrix multiply);
training the small model reached **30.7 TFLOPS with `torch.compile`** (20.4
without) — **~20 hours** for 2.5B tokens. Medium (~340M, 7B tokens): ~7 days
by the same arithmetic, to be benchmarked. Details: docs/RUNLOG.md.

Decided while building: pretraining uses a **2,048-token** context (4,096 would
add ~70% compute for the small model through attention alone); the context is
extended to 4,096 during task tuning, where the reader needs six passages.

## What "from scratch" means here

Random weights → pretrain on biomedical text → teach the one task we need
(answer only from the given passages, cite every sentence, say `NOT_FOUND`)
→ run it through `vpdl kb-eval`: same questions, same citation check, same
refusal test as medgemma:27b.

## What to expect — honestly

- **At the size the DGX can train (about 100–350M parameters) the model will
  write fluent biomedical text but reason far worse than medgemma:27b**, which
  was built with orders of magnitude more compute. Reference points:
  - BioGPT: 347M parameters, trained from scratch on 15M PubMed abstracts,
    10 days on 8 V100 GPUs.
  - BioMedLM: 2.7B parameters, 300B tokens, 128 A100 GPUs for ~6.25 days
    (~$38,000 at the authors' placeholder price).
  - The GeneTuring benchmark found both performed poorly on genomics questions.
- **So the small model does not replace the safety design.** It becomes the
  reader inside the same retrieve → cite → refuse system, and `kb-eval`
  decides. medgemma:27b stays the default until the small model matches it on
  *fresh* questions, with every unanswerable question refused.
- What it is good for: a research contribution (a genetics model trained on
  one desktop machine, measured head to head against a 27B model), full
  control of data and weights, and a model small enough to be very fast.

## Compute budget

Training cost per token ≈ 6 × parameters, plus attention over the context
(12 × layers × width × context). Measured on the DGX: 90.1 TFLOPS peak; 30.7
TFLOPS sustained while training the small model with `torch.compile`.

| Size | Tokens | Time on the DGX |
|---|---|---|
| small (~110M) | 2.5B (20 tokens per parameter, the Chinchilla rule) | **20 h (measured projection)** |
| small | 10B | ~3.3 days |
| medium (~340M) | 7B | ~7 days (estimate; benchmark first) |
| 1B | 20B | ~6–7 weeks — not practical here |

## Data

Pretraining corpus. The trained model is **research-only and not
distributed** until the licences have been reviewed for releasing weights.

| Source | Size | Terms |
|---|---|---|
| PubMed abstracts | Baseline: 1,334 files, 51.8 GB compressed (checked 2026-09-22). Roughly 5–7B tokens of abstract text (estimate; The Pile's 2020 copy held 15.5M abstracts in 19.3 GiB) | NLM does not own abstract copyright; publishers may. Standard research practice (BioGPT, BioMedLM) |
| GeneReviews | ~44,000 passages, already built | Non-commercial research |
| MedlinePlus Genetics, ClinGen, Orphanet | Small, on-target | Public domain / CC0 / CC BY |
| PMC Open Access (`oa_comm`) | Large full text | CC0 / BY / BY-SA / BY-ND — later |

1% of the text is held out to measure perplexity.

## Model

- **Tokenizer:** our own byte-level byte-pair vocabulary (32k) trained on the
  corpus (BioGPT also trained its own: 42,384 tokens). Any text round-trips
  exactly; identifiers split at letter/digit/punctuation boundaries
  ("MLH1" → "MLH" + "1"; "c.199G>A" → "c" "." "199" "G" ">" "A") rather than at
  arbitrary points, and `tokenizer_meta.json` records how probe terms split.
- **Architecture:** Llama-style decoder (RoPE, RMSNorm, SwiGLU).
  small (~110M with the 32k vocabulary) = 12 layers × 768 wide;
  medium (~340M) = 24 layers × 1024 wide.
- **Context:** 2,048 tokens in pretraining; extended to 4,096 in task tuning,
  where the reader must see 6 passages (~3–4k tokens).
- Built as a Hugging Face `LlamaForCausalLM` with **random** weights, so it
  exports to GGUF → Ollama → `kb-eval` unchanged, GPU check included.

## Teaching the task

A pretrained model only continues text. It has to learn to answer only from
the passages, cite `[S#]` on every sentence, and say `NOT_FOUND`.

- Training examples are generated locally from our passages by **qwen3:32b**.
  Its weights are Apache-2.0 and its outputs may be used to train another
  model. Not medgemma: whether its Health AI Developer Foundations terms
  allow training other models on its outputs has not been checked.
- Every generated example must pass our citation check. About a third are
  unanswerable (passages that do not contain the answer → `NOT_FOUND`).
- Cost: qwen3 took 35 s per answer with its "thinking" on; with it off, perhaps
  ~5 s → 20,000 examples in about 1–2 days.

## Steps

0. **Measure the DGX's training speed** (5 minutes). This fixes the size.
1. **Corpus:** download PubMed abstracts; add GeneReviews, MedlinePlus, ClinGen
   text; remove duplicates; hold out 1% (1–2 days, mostly downloading).
2. **Tokenizer** (hours).
3. **Pretrain small (~110M)** as the pipeline check (~17 hours). Pass: held-out perplexity
   falls steadily; samples read as biomedical text.
4. **Pretrain medium (~340M)** if the small run is sane (~6 days).
5. **Generate and filter** task examples (1–2 days); fine-tune (hours).
6. **Export and evaluate** with `kb-eval` on fresh questions against
   medgemma:27b. The bar: `refused_correctly` = 1.0 first, then correct facts.

## Risks

- It may not be good enough to answer. That result is still worth reporting,
  with the evaluation behind it.
- Contamination: the fresh evaluation questions must come from text not used
  to generate the training examples.
- Licensing if the weights are ever shared.
