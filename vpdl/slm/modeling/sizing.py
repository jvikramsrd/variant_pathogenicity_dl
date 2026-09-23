"""How big should the model be? Computed from the data and the machine, not chosen in advance.

Inputs are measured, not assumed: corpus tokens (``vpdl-slm stats`` /
``pretrain-pack``), supervised examples (``vpdl-slm examples``), the device's
memory (``vpdl-slm hardware``) and a measured throughput. Outputs are
candidate configurations with the numbers behind them, clearly labelled
estimates. Nothing here decides; the experiment plan decides after EXP-009/010
are run.

Rules used, each with its source:

* **tokens per parameter** — Hoffmann et al. 2022 ("Chinchilla") train-optimal
  ratio ~20:1 for pretraining FROM SCRATCH. Continued pretraining starts from a
  model that already saw far more, so this bounds the FROM-SCRATCH arm only.
* **training FLOPs** ~ 6 x parameters x tokens, plus attention
  12 x layers x width x context (PaLM appendix) — the same arithmetic
  ``vpdl.slm.model.flops_per_token`` uses for the from-scratch runs.
* **memory** ~ 16 bytes per trainable parameter (fp32 master weight + gradient +
  two Adam moments) + activations ~ 2 x layers x context x width x micro-batch
  bytes in bf16, halved with gradient checkpointing. A rough envelope, not a
  measurement: the runbook's benchmark step measures the real number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Candidate", "CANDIDATES", "estimate_memory_gib", "estimate_hours", "size_report"]


@dataclass(frozen=True)
class Candidate:
    name: str
    parameters: float          # total, in millions
    layers: int
    width: int
    kind: str                  # encoder | decoder
    source: str                # where the model comes from
    note: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("biomedical-encoder-base", 110, 12, 768, "encoder",
              "pretrained (BiomedBERT / BioBERT / Bio_ClinicalBERT) + continued pretraining",
              "the default: biomedical language already learnt; 512-token inputs fit a narrative"),
    Candidate("biomedical-encoder-large", 335, 24, 1024, "encoder",
              "pretrained large + continued pretraining",
              "3x the compute per step; only if base shows the data can use it"),
    Candidate("from-scratch-small", 110, 12, 768, "decoder", "vpdl slm-train --size small",
              "already built and benchmarked on the DGX (30.7 TFLOPS with compile)"),
    Candidate("from-scratch-medium", 340, 24, 1024, "decoder", "vpdl slm-train --size medium",
              "~7 days for 7B tokens by the same arithmetic (estimate)"),
)


def estimate_memory_gib(parameters_m: float, layers: int, width: int, context: int,
                        micro_batch: int, trainable_fraction: float = 1.0,
                        gradient_checkpointing: bool = False, bytes_per_state: int = 16) -> dict[str, float]:
    parameters = parameters_m * 1e6
    weights = parameters * 2 / 2 ** 30                              # bf16 copy held for the forward pass
    states = parameters * trainable_fraction * bytes_per_state / 2 ** 30
    activations = 2 * layers * context * width * micro_batch * 2 / 2 ** 30
    if gradient_checkpointing:
        activations /= max(1.0, layers ** 0.5)
    return {"weights_gib": round(weights, 2), "optimizer_states_gib": round(states, 2),
            "activations_gib": round(activations, 2),
            "total_gib": round(weights + states + activations, 2)}


def estimate_hours(parameters_m: float, layers: int, width: int, context: int, tokens: float,
                   tflops: float) -> float:
    per_token = 6 * parameters_m * 1e6 + 12 * layers * width * context
    return round(per_token * tokens / (tflops * 1e12) / 3600, 1)


def size_report(corpus_tokens: float | None, supervised_examples: int | None,
                memory_gib: float | None, tflops: float = 30.7, context: int = 512,
                micro_batch: int = 16, candidates: Sequence[Candidate] = CANDIDATES,
                pretraining_epochs: float = 1.0) -> dict[str, Any]:
    """Candidates with estimated memory and hours for THIS corpus and machine."""
    rows = []
    for candidate in candidates:
        tokens = (corpus_tokens or 0) * pretraining_epochs
        memory = estimate_memory_gib(candidate.parameters, candidate.layers, candidate.width,
                                     context, micro_batch)
        lora = estimate_memory_gib(candidate.parameters, candidate.layers, candidate.width, context,
                                   micro_batch, trainable_fraction=0.01)
        rows.append({
            "name": candidate.name, "parameters_m": candidate.parameters, "kind": candidate.kind,
            "source": candidate.source, "note": candidate.note,
            "continued_pretraining_hours_estimate": estimate_hours(
                candidate.parameters, candidate.layers, candidate.width, context, tokens, tflops)
            if corpus_tokens else None,
            "chinchilla_tokens_if_from_scratch_b": round(candidate.parameters * 20 / 1000, 2),
            "memory_full_finetune_gib": memory["total_gib"],
            "memory_lora_gib": lora["total_gib"],
            "fits_in_memory": None if memory_gib is None else memory["total_gib"] < memory_gib,
            "fits_with_lora": None if memory_gib is None else lora["total_gib"] < memory_gib,
        })
    supervision_note = None
    if supervised_examples is not None:
        supervision_note = (
            f"{supervised_examples:,} supervised examples. Full fine-tuning of a 110M-parameter "
            "backbone on fewer than ~10^5 examples overfits silently (the DL branch refuses full "
            "fine-tuning above 150M parameters for the same reason): LoRA or last-n first, full "
            "only with a validation result that justifies it.")
    return {"assumptions": {"tflops": tflops, "context": context, "micro_batch": micro_batch,
                            "bytes_per_parameter_state": 16,
                            "note": "ESTIMATES — measure with `vpdl-slm pretrain --benchmark`"},
            "corpus_tokens": corpus_tokens, "supervised_examples": supervised_examples,
            "memory_gib": memory_gib, "supervision_note": supervision_note, "candidates": rows}
