"""Zero-shot variant scores from a protein language model — no training at all.

Two methods, named for what they compute (Meier et al. 2021 define both):

``masked_marginal``   mask the site, read log P(mut) - log P(wt) at it. One
                      forward pass per residue position with variants, so the
                      model never sees the wild-type residue it is scoring.
``wt_marginal``       one unmasked pass over the wild-type chain, read the same
                      difference. Cheaper (one pass per window), and what v1
                      called "masked marginal" (CODEBASE_AUDIT finding 1).

Scores are stored raw (log-ratio: NEGATIVE for damaging) and oriented only at
the edge, with :func:`vpdl.features.pllr_to_pathogenicity` — the one function
landmine L6 pins. The 20 log-probabilities at each scored position are cached,
so every substitution at a position is scored from one pass.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.context import site_span, sliding_spans
from vpdl.dl.plm.backbones import AA20
from vpdl.dl.plm.forward import site_log_probs

logger = logging.getLogger(__name__)

__all__ = ["ZEROSHOT_METHODS", "position_log_probs", "score_variants", "zeroshot_frame"]

ZEROSHOT_METHODS = ("masked_marginal", "wt_marginal")
_AA_INDEX = {aa: i for i, aa in enumerate(AA20)}


def position_log_probs(backbone, sequence: str, positions: Iterable[int],
                       method: str = "masked_marginal", policy: str = "full",
                       batch_size="auto") -> dict[int, np.ndarray]:
    """``{position (1-based): log-probs [20]}`` for each requested position.

    For chains longer than the window the context follows `policy`
    (:mod:`vpdl.dl.context`); ``sliding`` averages the site's log-probs over
    every window containing it, as v1 did for the wild-type marginal.
    """
    if method not in ZEROSHOT_METHODS:
        raise ValueError(f"unknown zero-shot method {method!r}; known {ZEROSHOT_METHODS}")
    spec = backbone.spec
    positions = sorted({int(p) for p in positions})
    windows: list[str] = []
    sites: list[int] = []
    owners: list[int] = []
    for position in positions:
        pos0 = position - 1
        if policy in ("sliding", "hierarchical") and len(sequence) > spec.max_residues:
            spans = [s for s in sliding_spans(len(sequence), spec.max_residues)
                     if s[0] <= pos0 < s[1]]
        else:
            spans = [site_span("centered" if policy == "hierarchical" else policy,
                               len(sequence), pos0, spec.max_residues,
                               spec.supports_full_length)]
        for start, end in spans:
            windows.append(sequence[start:end])
            sites.append(pos0 - start)
            owners.append(position)
    log_probs = site_log_probs(backbone, windows, sites, masked=method == "masked_marginal",
                               batch_size=batch_size)
    totals: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for owner, row in zip(owners, log_probs):
        totals[owner] = totals.get(owner, 0) + row
        counts[owner] = counts.get(owner, 0) + 1
    return {p: totals[p] / counts[p] for p in positions}


def score_variants(log_probs: dict[int, np.ndarray],
                   variants: Sequence[tuple[int, str, str]]) -> np.ndarray:
    """Raw log-ratio ``log P(mut) - log P(wt)`` per ``(position, wt, mut)``."""
    return np.array([log_probs[int(p)][_AA_INDEX[m]] - log_probs[int(p)][_AA_INDEX[w]]
                     for p, w, m in variants], dtype=np.float64)


def zeroshot_frame(backbone, table: pd.DataFrame, sequences: dict[str, str],
                   method: str = "masked_marginal", policy: str = "full",
                   batch_size="auto") -> pd.DataFrame:
    """Score every row of `table` (``uniprot_id, position, wt_aa, mut_aa``).

    Returns ``variant_key, raw_llr, pathogenicity`` where ``pathogenicity`` is
    oriented higher = more damaging, ready for ``vpdl paired --feature-baseline``.
    """
    from vpdl.features import pllr_to_pathogenicity
    from vpdl.splits import variant_keys

    parts = []
    for accession, rows in table.groupby("uniprot_id"):
        sequence = sequences[accession]
        mismatched = [p for p, w in zip(rows["position"], rows["wt_aa"])
                      if sequence[int(p) - 1] != w]
        if mismatched:
            raise ValueError(f"{accession}: wild-type mismatch at {mismatched[:5]} — "
                             "refusing to score against the wrong residue")
        log_probs = position_log_probs(backbone, sequence, rows["position"], method, policy,
                                       batch_size)
        raw = score_variants(log_probs, list(zip(rows["position"], rows["wt_aa"],
                                                 rows["mut_aa"])))
        parts.append(pd.DataFrame({"variant_key": variant_keys(rows), "raw_llr": raw,
                                   "pathogenicity": pllr_to_pathogenicity(raw)}))
        logger.info("%s: %s scored %d variants at %d positions (%s context)", accession,
                    method, len(rows), rows["position"].nunique(), policy)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["variant_key", "raw_llr", "pathogenicity"])
