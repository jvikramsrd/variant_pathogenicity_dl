"""WT / VT embeddings per variant, at three levels, generated once and cached.

For each missense variant six raw vectors are extracted, each ``d`` wide:

    site_wt,   site_vt     hidden state at the mutated residue
    local_wt,  local_vt    mean over the mutation +/- radius window (MVmamba's
                           optimum radius 3; configurable)
    global_wt, global_vt   mean over the whole context the policy provides

and stored raw. Every representation the experiment compares is DERIVED from
those at load time (:func:`derive`), never re-extracted:

    wt | vt | vt_minus_wt | abs_diff | wt_plus_vt | concat4 ([wt, vt, diff, |diff|])

so evaluating "VT - WT vs |VT - WT| vs WT + VT" is five cheap reads of one cache.

Efficiency: one wild-type pass (per window) per protein; one variant-type pass
per unique ``(position, mut)``; under ``sliding``, only the windows containing
the mutation are recomputed (a transformer's other windows cannot see it), so
the variant's chain mean is exact without re-running the whole chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from vpdl.dl.context import CONTEXT_POLICIES, ContextSummary, site_span, sliding_spans
from vpdl.dl.plm.forward import hidden_states

logger = logging.getLogger(__name__)

__all__ = ["RAW_BLOCKS", "LEVELS", "PAIRINGS", "derive", "EmbeddingExtractor"]

LEVELS = ("site", "local", "global")
RAW_BLOCKS = tuple(f"{level}_{side}" for level in LEVELS for side in ("wt", "vt"))
PAIRINGS = ("wt", "vt", "vt_minus_wt", "abs_diff", "wt_plus_vt", "concat4")


def derive(blocks: dict[str, np.ndarray], representation: str) -> np.ndarray:
    """``"<level>.<pairing>"`` (e.g. ``site.vt_minus_wt``) from the raw blocks."""
    level, _, pairing = representation.partition(".")
    if level not in LEVELS or pairing not in PAIRINGS:
        raise ValueError(f"representation must be <level>.<pairing> with level in {LEVELS} "
                         f"and pairing in {PAIRINGS}; got {representation!r}")
    wt = blocks[f"{level}_wt"].astype(np.float32)
    vt = blocks[f"{level}_vt"].astype(np.float32)
    diff = vt - wt
    return {"wt": wt, "vt": vt, "vt_minus_wt": diff, "abs_diff": np.abs(diff),
            "wt_plus_vt": wt + vt,
            "concat4": np.concatenate([wt, vt, diff, np.abs(diff)], axis=1)}[pairing]


def _mutate(sequence: str, key: tuple[int, str]) -> str:
    """The variant chain — always cut from the MUTATED sequence (landmine L12)."""
    position, mut = key
    return sequence[:position - 1] + mut + sequence[position:]


@dataclass
class _Chain:
    """Wild-type per-window outputs for one chain, reused by every variant."""

    sequence: str
    spans: list[tuple[int, int]]
    outputs: dict[tuple[int, int], np.ndarray]
    summed: np.ndarray | None = None
    counts: np.ndarray | None = None


class EmbeddingExtractor:
    def __init__(self, backbone, policy: str = "full", local_radius: int = 3,
                 batch_size: int = 8, layer: int = -1, chunk: int = 64) -> None:
        if policy not in CONTEXT_POLICIES:
            raise ValueError(f"unknown policy {policy!r}; known {CONTEXT_POLICIES}")
        self.backbone = backbone
        self.policy = policy
        self.radius = int(local_radius)
        self.batch_size = int(batch_size)
        self.layer = int(layer)
        self.chunk = max(1, int(chunk))
        self.max_residues = backbone.spec.max_residues
        # Refuse up front rather than on the first long chain.
        if policy == "full" and not backbone.spec.supports_full_length:
            logger.info("%s has a positional limit: 'full' applies only to chains "
                        "<= %d residues and raises on longer ones.",
                        backbone.spec.name, self.max_residues)

    def describe(self, length: int) -> dict:
        return ContextSummary.of(self.policy, length, self.max_residues).__dict__ | {
            "local_radius": self.radius, "layer": self.layer}

    # -- helpers -------------------------------------------------------------
    def _run(self, sequences: Sequence[str]) -> list[np.ndarray]:
        return hidden_states(self.backbone, sequences, self.batch_size, self.layer)

    def _local(self, hidden: np.ndarray, pos0_in_window: int) -> np.ndarray:
        lo = max(0, pos0_in_window - self.radius)
        hi = min(len(hidden), pos0_in_window + self.radius + 1)
        return hidden[lo:hi].mean(axis=0)

    def _sliding_chain(self, sequence: str) -> _Chain:
        spans = sliding_spans(len(sequence), self.max_residues)
        outputs = dict(zip(spans, self._run([sequence[s:e] for s, e in spans])))
        d = next(iter(outputs.values())).shape[1]
        summed = np.zeros((len(sequence), d), dtype=np.float64)
        counts = np.zeros(len(sequence), dtype=np.float64)
        for (s, e), h in outputs.items():
            summed[s:e] += h
            counts[s:e] += 1
        return _Chain(sequence, spans, outputs, summed, counts)

    # -- extraction ----------------------------------------------------------
    def extract(self, sequence: str, variants: Sequence[tuple[int, str, str]]
                ) -> dict[str, np.ndarray]:
        """Raw blocks ``{name: [n_variants, d]}`` for ``(position, wt, mut)`` rows."""
        for position, wt, _ in variants:
            if not 1 <= int(position) <= len(sequence) or sequence[int(position) - 1] != wt:
                raise ValueError(f"wild-type mismatch at {position}{wt}: refusing to embed")
        unique = sorted({(int(p), m) for p, _, m in variants})
        by_key: dict[tuple[int, str], dict[str, np.ndarray]] = {}

        needs_sliding = self.policy in ("sliding", "hierarchical")
        chain = self._sliding_chain(sequence) if needs_sliding else None
        if chain is not None:
            wt_full = (chain.summed / chain.counts[:, None]).astype(np.float32)
            wt_global = wt_full.mean(axis=0)

        if self.policy != "sliding":
            # One window per variant: full, centred or asymmetric.
            local_policy = "centered" if self.policy == "hierarchical" else self.policy
            spans = {key: site_span(local_policy, len(sequence), key[0] - 1, self.max_residues,
                                    self.backbone.spec.supports_full_length)
                     for key in unique}
            wt_spans = sorted(set(spans.values()))
            wt_out = dict(zip(wt_spans, self._run([sequence[s:e] for s, e in wt_spans])))
            # Chunked: per-residue outputs for 17k assay variants at once would
            # be ~80 GB; each chunk is reduced to six vectors as it arrives.
            for start in range(0, len(unique), self.chunk):
                keys = unique[start:start + self.chunk]
                outs = self._run([_mutate(sequence, key)[spans[key][0]:spans[key][1]]
                                  for key in keys])
                for key, h_vt in zip(keys, outs):
                    (s, _), p0 = spans[key], key[0] - 1
                    h_wt = wt_out[spans[key]]
                    by_key[key] = {"site_wt": h_wt[p0 - s], "site_vt": h_vt[p0 - s],
                                   "local_wt": self._local(h_wt, p0 - s),
                                   "local_vt": self._local(h_vt, p0 - s),
                                   "global_wt": h_wt.mean(axis=0),
                                   "global_vt": h_vt.mean(axis=0)}

        if chain is not None:
            # Only windows containing the mutation change; recompute those.
            for start in range(0, len(unique), self.chunk):
                keys = unique[start:start + self.chunk]
                jobs, owners = [], []
                for key in keys:
                    p0, mutated = key[0] - 1, _mutate(sequence, key)
                    for span in chain.spans:
                        if span[0] <= p0 < span[1]:
                            jobs.append(mutated[span[0]:span[1]])
                            owners.append((key, span))
                deltas: dict[tuple[int, str], list] = {}
                for (key, span), h in zip(owners, self._run(jobs)):
                    deltas.setdefault(key, []).append((span, h))
                for key in keys:
                    p0 = key[0] - 1
                    summed = chain.summed.copy()
                    for (s, e), h in deltas[key]:
                        summed[s:e] += h - chain.outputs[(s, e)]
                    vt_full = (summed / chain.counts[:, None]).astype(np.float32)
                    if self.policy == "sliding":
                        by_key[key] = {"site_wt": wt_full[p0], "site_vt": vt_full[p0],
                                       "local_wt": self._local(wt_full, p0),
                                       "local_vt": self._local(vt_full, p0)}
                    by_key[key]["global_wt"] = wt_global
                    by_key[key]["global_vt"] = vt_full.mean(axis=0)

        blocks = {name: np.stack([by_key[(int(p), m)][name] for p, _, m in variants])
                  .astype(np.float32) for name in RAW_BLOCKS}
        logger.info("embedded %d variants (%d unique) on a %d-residue chain, policy=%s",
                    len(variants), len(unique), len(sequence), self.policy)
        return blocks
