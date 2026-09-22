"""Batched, memory-lean forward passes through an ESM backbone.

Two v1 lessons are built in: only the hidden states actually used are kept
(``output_hidden_states=True`` retains all 34 layers — a ~34x peak-memory spike
for one layer's worth of output), and the LM head is applied only at the
positions whose logits are read, not across every token.
"""

from __future__ import annotations

import contextlib
from typing import Sequence

import numpy as np

__all__ = ["inference_autocast", "hidden_states", "site_log_probs"]


def inference_autocast(device):
    import torch
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


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


def hidden_states(backbone, sequences: Sequence[str], batch_size: int = 8,
                  layer: int = -1) -> list[np.ndarray]:
    """Per-residue hidden states ``[len(seq), d]`` (float32) for each sequence.

    Sequences are length-sorted into batches so padding stays small, and the
    results are returned in the input order.
    """
    import torch

    order = np.argsort([len(s) for s in sequences], kind="mergesort")
    out: list[np.ndarray | None] = [None] * len(sequences)
    device = backbone.device
    backbone.model.eval()
    with torch.inference_mode(), inference_autocast(device):
        for start in range(0, len(order), max(1, batch_size)):
            chunk = order[start:start + batch_size]
            ids, attention = backbone.alphabet.encode([sequences[i] for i in chunk])
            hidden = _encoder_output(backbone, ids.to(device), attention.to(device), layer)
            hidden = hidden.float().cpu().numpy()
            for row, index in enumerate(chunk):
                out[index] = hidden[row, 1:1 + len(sequences[index])]
    return out  # type: ignore[return-value]


def site_log_probs(backbone, sequences: Sequence[str], sites: Sequence[int],
                   masked: bool, batch_size: int = 16) -> np.ndarray:
    """``[n, 20]`` log-probabilities of the 20 residues at one site per sequence.

    ``masked=True`` replaces the site with ``<mask>`` first — the masked
    marginal. ``masked=False`` reads the unmasked pass — the wild-type
    marginal (what v1 computed and mislabelled). The softmax runs over the
    full vocabulary and the 20 standard residues are then selected, matching
    how ESM scores variants.
    """
    import torch

    order = np.argsort([len(s) for s in sequences], kind="mergesort")
    out = np.zeros((len(sequences), 20), dtype=np.float32)
    device = backbone.device
    aa_ids = torch.tensor(backbone.alphabet.aa_ids, device=device)
    backbone.model.eval()
    with torch.inference_mode(), inference_autocast(device):
        for start in range(0, len(order), max(1, batch_size)):
            chunk = order[start:start + batch_size]
            chunk_sites = [int(sites[i]) for i in chunk]
            ids, attention = backbone.alphabet.encode(
                [sequences[i] for i in chunk], mask_positions=chunk_sites if masked else None)
            hidden = _encoder_output(backbone, ids.to(device), attention.to(device), -1)
            rows = torch.arange(len(chunk), device=device)
            at_site = hidden[rows, torch.tensor(chunk_sites, device=device) + 1]
            logits = backbone.model.lm_head(at_site).float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, aa_ids]
            out[chunk] = log_probs.cpu().numpy()
    return out
