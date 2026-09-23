"""Batched, memory-lean forward passes through an ESM backbone.

Two v1 lessons are built in: only the hidden states actually used are kept
(``output_hidden_states=True`` retains all 34 layers — a ~34x peak-memory spike
for one layer's worth of output), and the LM head is applied only at the
positions whose logits are read, not across every token.

Sized for the DGX Spark rather than for a small card:

* ``batch_size="auto"`` sizes each batch from the GPU memory actually free
  right now (the pool is shared with the CPU and with other jobs), and
* a batch that still runs out of memory is split in half and retried, so an
  optimistic size costs a retry, never a crashed run. Results do not depend on
  the batch size: every sequence is encoded independently.
* ``to_numpy=False`` keeps outputs on the GPU so callers reduce them there
  (a mean over residues) instead of copying every per-residue vector to the CPU.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["inference_autocast", "resolve_batch_size", "hidden_states", "site_log_probs",
           "AUTO_BATCH_CAP"]

# Upper bound for an automatic batch. Past this the GB10 is compute-bound and a
# bigger batch only raises the cost of an out-of-memory retry.
AUTO_BATCH_CAP = 256
_CPU_BATCH = 8


def inference_autocast(device):
    import torch
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def resolve_batch_size(requested, backbone, max_length: int, budget: float = 0.4) -> int:
    """An integer batch size; ``"auto"`` sizes it from free GPU memory.

    Per sequence: the widest activation of one layer (the 4x FFN, bf16) plus a
    float32 copy of the output, and — in case attention falls back to the
    unfused kernel — the full ``T x T`` score matrix per head in float32. A
    deliberate over-estimate; the out-of-memory retry covers the rest.
    """
    if requested not in (None, "auto"):
        return max(1, int(requested))
    if backbone.device.type != "cuda":
        return _CPU_BATCH
    import torch

    free, _ = torch.cuda.mem_get_info(backbone.device)
    config = backbone.model.config
    tokens = max_length + 2
    per_sequence = (tokens * config.hidden_size * 96
                    + config.num_attention_heads * tokens * tokens * 4)
    size = int(budget * free // max(1, per_sequence))
    return max(1, min(AUTO_BATCH_CAP, size))


def _is_oom(error: BaseException) -> bool:
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(error, oom)) or "out of memory" in str(error).lower()


def _run_batched(order: np.ndarray, batch_size: int, run: Callable[[np.ndarray], None],
                 device) -> None:
    """Apply `run` to consecutive chunks of `order`, halving a chunk on OOM."""
    import torch

    start, size, warned = 0, max(1, batch_size), False
    while start < len(order):
        chunk = order[start:start + size]
        try:
            run(chunk)
        except (RuntimeError, MemoryError) as error:
            if not _is_oom(error) or size == 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            size = max(1, size // 2)
            if not warned:
                logger.warning("out of GPU memory at batch %d; continuing at %d",
                               len(chunk), size)
                warned = True
            continue
        start += len(chunk)


def _encoder_output(backbone, ids, attention, layer: int):
    """Hidden states ``[B, T, d]`` from the final layer or a hooked earlier one."""
    encoder = backbone.encoder()
    if layer == -1:
        return encoder(input_ids=ids, attention_mask=attention).last_hidden_state
    captured = {}
    block = encoder.encoder.layer[layer]
    handle = block.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    try:
        encoder(input_ids=ids, attention_mask=attention)
    finally:
        handle.remove()
    return captured["h"]


def hidden_states(backbone, sequences: Sequence[str], batch_size="auto",
                  layer: int = -1, to_numpy: bool = True) -> list:
    """Per-residue hidden states ``[len(seq), d]`` (float32) for each sequence.

    Sequences are length-sorted into batches so padding stays small, and the
    results are returned in the input order. With ``to_numpy=False`` each
    result is a float32 tensor left on the backbone's device.
    """
    import torch

    if not sequences:
        return []
    order = np.argsort([len(s) for s in sequences], kind="mergesort")
    out: list = [None] * len(sequences)
    device = backbone.device
    size = resolve_batch_size(batch_size, backbone, max(len(s) for s in sequences))
    backbone.model.eval()

    def run(chunk: np.ndarray) -> None:
        ids, attention = backbone.alphabet.encode([sequences[i] for i in chunk])
        hidden = _encoder_output(backbone, ids.to(device, non_blocking=True),
                                 attention.to(device, non_blocking=True), layer).float()
        if to_numpy:
            hidden = hidden.cpu().numpy()
        for row, index in enumerate(chunk):
            item = hidden[row, 1:1 + len(sequences[index])]
            out[index] = item if to_numpy else item.clone()

    with torch.inference_mode(), inference_autocast(device):
        _run_batched(order, size, run, device)
    return out


def site_log_probs(backbone, sequences: Sequence[str], sites: Sequence[int],
                   masked: bool, batch_size="auto") -> np.ndarray:
    """``[n, 20]`` log-probabilities of the 20 residues at one site per sequence.

    ``masked=True`` replaces the site with ``<mask>`` first — the masked
    marginal. ``masked=False`` reads the unmasked pass — the wild-type
    marginal (what v1 computed and mislabelled). The softmax runs over the
    full vocabulary and the 20 standard residues are then selected, matching
    how ESM scores variants.
    """
    import torch

    out = np.zeros((len(sequences), 20), dtype=np.float32)
    if not sequences:
        return out
    order = np.argsort([len(s) for s in sequences], kind="mergesort")
    device = backbone.device
    aa_ids = torch.tensor(backbone.alphabet.aa_ids, device=device)
    size = resolve_batch_size(batch_size, backbone, max(len(s) for s in sequences))
    backbone.model.eval()

    def run(chunk: np.ndarray) -> None:
        chunk_sites = [int(sites[i]) for i in chunk]
        ids, attention = backbone.alphabet.encode(
            [sequences[i] for i in chunk], mask_positions=chunk_sites if masked else None)
        hidden = _encoder_output(backbone, ids.to(device, non_blocking=True),
                                 attention.to(device, non_blocking=True), -1)
        rows = torch.arange(len(chunk), device=device)
        at_site = hidden[rows, torch.tensor(chunk_sites, device=device) + 1]
        logits = backbone.model.lm_head(at_site).float()
        out[chunk] = torch.log_softmax(logits, dim=-1)[:, aa_ids].cpu().numpy()

    with torch.inference_mode(), inference_autocast(device):
        _run_batched(order, size, run, device)
    return out
