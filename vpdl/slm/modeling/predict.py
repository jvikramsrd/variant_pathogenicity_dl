"""Predictions, stochastic (MC-dropout) samples and embeddings, in fixed-size batches."""

from __future__ import annotations

import contextlib

import numpy as np

from vpdl.slm.modeling.data import TaskTensors, UnitTensors, to_batch, unit_batch

__all__ = ["predict_documents", "predict_units", "mc_dropout_samples"]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict_documents(model, tensors: TaskTensors, indices: np.ndarray, device, batch_size: int = 64,
                      autocast=None) -> dict[str, np.ndarray]:
    import torch
    model.eval()
    logits, embeddings, text, acmg = [], [], [], []
    with torch.inference_mode(), (autocast() if autocast else contextlib.nullcontext()):
        for start in range(0, len(indices), batch_size):
            batch = to_batch(tensors, indices[start:start + batch_size], device)
            out = model(**batch)
            logits.append(out["class_logits"].float().cpu().numpy())
            embeddings.append(out["embedding"].float().cpu().numpy())
            text.append(out["text_embedding"].float().cpu().numpy())
            if "acmg_logits" in out:
                acmg.append(torch.sigmoid(out["acmg_logits"].float()).cpu().numpy())
    empty = np.zeros((0, 0), dtype=np.float32)
    result = {"logits": np.concatenate(logits) if logits else empty,
              "embedding": np.concatenate(embeddings) if embeddings else empty,
              "text_embedding": np.concatenate(text) if text else empty}
    result["probs"] = _softmax(result["logits"]) if len(result["logits"]) else empty
    if acmg:
        result["acmg_probs"] = np.concatenate(acmg)
    return result


def predict_units(model, units: UnitTensors, indices: np.ndarray, device, batch_size: int = 128,
                  autocast=None) -> dict[str, np.ndarray]:
    import torch
    model.eval()
    types, polarity, embeddings = [], [], []
    with torch.inference_mode(), (autocast() if autocast else contextlib.nullcontext()):
        for start in range(0, len(indices), batch_size):
            batch = unit_batch(units, indices[start:start + batch_size], device)
            out = model.forward_units(batch["unit_input_ids"], batch["unit_attention_mask"])
            types.append(torch.sigmoid(out["type_logits"].float()).cpu().numpy())
            polarity.append(torch.softmax(out["polarity_logits"].float(), -1).cpu().numpy())
            embeddings.append(out["unit_embedding"].float().cpu().numpy())
    return {"type_probs": np.concatenate(types), "polarity_probs": np.concatenate(polarity),
            "unit_embedding": np.concatenate(embeddings)}


def mc_dropout_samples(model, tensors: TaskTensors, indices: np.ndarray, device, samples: int = 20,
                       batch_size: int = 64, seed: int = 0) -> np.ndarray:
    """[S, N, 5] probabilities with dropout active (weights unchanged, RNG restored after)."""
    import torch
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    model.train()
    draws = []
    try:
        with torch.no_grad():
            for _ in range(samples):
                logits = []
                for start in range(0, len(indices), batch_size):
                    out = model(**to_batch(tensors, indices[start:start + batch_size], device))
                    logits.append(out["class_logits"].float().cpu().numpy())
                draws.append(_softmax(np.concatenate(logits)))
    finally:
        model.eval()
        torch.set_rng_state(state)
    return np.stack(draws)
