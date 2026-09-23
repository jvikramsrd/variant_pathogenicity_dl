# Edge deployment — desirable, and secondary to being right

**Nothing in this document has been measured.** Every figure below is
`TBD — RUN ON DGX SPARK` or later; no quantised model, ONNX graph or latency
number exists in this repository. It is a plan, written so the design does not
foreclose it, not a claim.

## 1. Why it is secondary

A small model that runs anywhere but is miscalibrated on unseen genes is worse
than useless in a clinical setting — it is confidently wrong at scale. So the
order is: leakage-safe data → measured accuracy and calibration → grounded
explanations → **then** size, latency and packaging. Deployment work that
would compromise any earlier item is not done.

## 2. What already points that way

* The classification head sits on a **base-sized encoder** (~110M) by default;
  the whole point of the size study is that the smallest model the data
  supports wins.
* **LoRA / adapters** mean a specialisation (for example the MMR adapter) is
  megabytes, not gigabytes — one base model plus small deltas.
* The **deterministic explainer** needs no generation at inference: the
  explanation is assembled from evidence units and the model's own
  probabilities, so the expensive part is one forward pass.
* Retrieval is **BM25**, which needs no embedding model at query time.
* The from-scratch arm already exports to Hugging Face format and, per
  docs/slm/PLAN.md, on to GGUF for Ollama.

## 3. What would be measured, in order

| Question | How | Status |
|---|---|---|
| How large is the model on disk, and in memory at batch 1? | `vpdl-slm sizing`, then the real checkpoint | TBD |
| Latency per variant, CPU and GPU, batch 1 and 32 | time one forward pass over the fine-tuned model | TBD |
| Does dynamic int8 quantisation change any metric? | quantise, re-run `vpdl-slm evaluate` on the same predictions protocol | TBD |
| Does ONNX / TensorRT export change any metric? | export, re-score, compare exactly | TBD |
| Does a smaller backbone lose accuracy or only speed? | the size study's smaller candidates | TBD |

The rule for all of them: **a deployment format is only acceptable if the
metrics are recomputed after conversion and match.** A quantised model is a
different model; it gets its own row in the registry, not a footnote.

## 4. Constraints that do not change at the edge

* Research use only; not a medical device; the disclaimer travels with every
  output.
* Variant facts are looked up, never generated — the knowledge base's rule, and
  the reason `vpdl kb-ask` and this classifier stay separate systems.
* Nothing leaves the machine: the teacher client refuses any host but loopback,
  and the same must hold for any deployed service.
* Weights are **not distributed** until the licences of everything they were
  trained on have been reviewed (docs/slm/PLAN.md says the same for the
  from-scratch model; GeneReviews is non-commercial research, PubMed abstracts
  carry publisher rights, OMIM-sourced narratives are excluded by flag).

## 5. What would have to be true before any clinical pilot

Not a deployment checklist — a scientific one, and none of it is met yet:
calibrated on the gene and disease it is used for; evaluated on variants no
part of the pipeline saw; explanations reviewed by a clinical geneticist
against their cited sources; abstention behaviour measured; and a named person
responsible for the decision the output informs.
