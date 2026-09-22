"""Sequence-context policies — how a protein longer than a model's window is seen.

Only MSH6 (1,360 residues) exceeds the 1,022-residue window ESM models were
trained on, so on this panel every policy gives identical input for MLH1, MSH2
and PMS2 and differs only for MSH6. That makes the choice a potential
gene-identity confound under leave-one-gene-out (v1 shipped exactly that bug,
landmine L12), so the policy is recorded in every feature-store entry and is a
reported axis of the experiment rather than a hidden constant.

    full          the whole chain in one pass. Needs rotary positions (ESM-2);
                  ESM-1b's learned absolute positions stop at 1,022 and it is
                  refused. Beyond 1,022 this is extrapolation for ESM-2 too —
                  measured, not assumed, on the DGX.
    sliding       overlapping 1,022-residue windows, per-residue outputs
                  averaged (v1's aggregation). Exact for a variant: only windows
                  containing the mutation are recomputed.
    centered      one window centred on the mutation (VariPred 510/511 split,
                  clamped at the termini).
    asymmetric    VariPred's recipe: the first 1,022 residues if the mutation
                  lies in them, else the last 1,022 if it lies there, else
                  centred.
    hierarchical  local (site, window) from the centred pass + global (chain
                  mean) from the sliding pass: in-distribution local context
                  and a whole-chain summary.

Positions here are 0-based residue indices; spans are half-open ``[start, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CONTEXT_POLICIES", "MAX_RESIDUES", "WINDOW_OVERLAP", "sliding_spans",
           "centered_span", "asymmetric_span", "site_span", "ContextError"]

CONTEXT_POLICIES = ("full", "sliding", "centered", "asymmetric", "hierarchical")
MAX_RESIDUES = 1022
WINDOW_OVERLAP = 256


class ContextError(ValueError):
    """A policy cannot be applied to this backbone/sequence combination."""


def sliding_spans(length: int, max_residues: int = MAX_RESIDUES,
                  overlap: int = WINDOW_OVERLAP) -> list[tuple[int, int]]:
    """Half-open windows covering ``[0, length)`` (ported from v1 esm_extractor)."""
    if length <= max_residues:
        return [(0, length)]
    step = max(1, max_residues - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_residues, length)
        spans.append((start, end))
        if end == length:
            break
        start += step
    return spans


def centered_span(length: int, pos0: int, max_residues: int = MAX_RESIDUES) -> tuple[int, int]:
    """Mutation-centred window, clamped to the chain (v1 centered_window_bounds)."""
    if not 0 <= pos0 < length:
        raise ValueError(f"position {pos0} outside chain of length {length}")
    span = min(max_residues, length)
    start = min(max(pos0 - span // 2, 0), max(0, length - span))
    return start, start + span


def asymmetric_span(length: int, pos0: int, max_residues: int = MAX_RESIDUES) -> tuple[int, int]:
    """VariPred: nearest-terminus window when it contains the mutation, else centred."""
    if not 0 <= pos0 < length:
        raise ValueError(f"position {pos0} outside chain of length {length}")
    if length <= max_residues:
        return 0, length
    if pos0 < max_residues:
        return 0, max_residues
    if pos0 >= length - max_residues:
        return length - max_residues, length
    return centered_span(length, pos0, max_residues)


def site_span(policy: str, length: int, pos0: int, max_residues: int = MAX_RESIDUES,
              supports_full_length: bool = True) -> tuple[int, int]:
    """The single window a site-level representation is read from.

    For ``sliding`` there is no single window; callers aggregate over
    :func:`sliding_spans`. ``hierarchical`` reads its local part from the
    centred window.
    """
    if policy not in CONTEXT_POLICIES:
        raise ContextError(f"unknown context policy {policy!r}; known {CONTEXT_POLICIES}")
    if policy == "full":
        if length > max_residues and not supports_full_length:
            raise ContextError(
                f"policy 'full' on a {length}-residue chain needs a backbone without a "
                f"{max_residues}-residue positional limit (ESM-1b has one). Use "
                "'centered', 'asymmetric', 'sliding' or 'hierarchical'.")
        return 0, length
    if policy == "asymmetric":
        return asymmetric_span(length, pos0, max_residues)
    if policy == "sliding":
        raise ContextError("'sliding' has no single site window; aggregate over sliding_spans")
    return centered_span(length, pos0, max_residues)


@dataclass(frozen=True)
class ContextSummary:
    """What a policy does to one chain — recorded in feature-store metadata."""

    policy: str
    length: int
    max_residues: int
    exceeds_window: bool
    n_sliding_windows: int

    @classmethod
    def of(cls, policy: str, length: int, max_residues: int = MAX_RESIDUES) -> "ContextSummary":
        return cls(policy, length, max_residues, length > max_residues,
                   len(sliding_spans(length, max_residues)))
