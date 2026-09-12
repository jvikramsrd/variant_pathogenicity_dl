"""Canonical gene-ID resolution for cross-partition leakage checks.

This project's gene-level holdout (``docs/HOLDOUT_PROTOCOL.md``) means an
entire protein -- every one of its variants, across every source table
(ClinVar, ProteinGym, gnomAD, MaveDB, CIMRA) -- is withheld from training,
preprocessing-fit, and hyperparameter/model selection, not just scored
separately at the end. That guarantee only holds if "the same gene" is
recognised under every spelling it appears as. Four distinct notions of
"holdout" exist in ML generally, and this module protects exactly one of
them:

* **Gene-level** (what this module enforces) -- an entire protein's data is
  absent from every partition but one. The unit of disjointness is the
  canonical UniProt accession, not a row, not a patient, not a batch.
* **Sample-level** -- a random subset of one gene's variants held out while
  other variants of the *same* gene remain in training. This project does
  this too (inner train/val folds within the fine-tune partition, see
  ``src/train.py``'s nested ``make_position_group_folds`` calls), but it is
  a different, weaker guarantee: it says nothing about a whole protein's
  identity ever leaking, only about which specific rows of an already-seen
  protein were used for early stopping vs. gradient updates.
* **Patient-level** -- not applicable to this codebase: every table here is
  keyed by (gene, position, wt_aa, mut_aa), not by an individual/sample
  donor, so there is no patient axis to leak across.
* **Batch-level** -- a held-out sequencing batch/cohort/lab. Not what any
  split in this codebase does; noted here only so a reader doesn't conflate
  it with gene-level holdout.
* **Random-feature holdout** (feature ablation) -- withholding a *feature
  column*, not a gene, e.g. ``--drop_prior_groups`` in
  ``scripts/finetune_esm_mmr.py``. Orthogonal to this module: a feature can
  be dropped from every gene at once, or a gene can be dropped with every
  feature intact; the two are independent axes.

Canonical IDs are sourced from :data:`src.mmr_dataset.MMR_UNIPROT`, the only
place in this codebase that pairs an HGNC symbol with a verified UniProt
accession (each entry is validated against the live UniProt sequence length
at data-build time -- see ``resolve_mmr_panel`` in that module). This module
does not invent, guess, or fetch any additional gene/accession pairs: an
unresolvable name raises :class:`UnknownGeneError` rather than being
silently passed through or fuzzy-matched. Fuzzy/partial matching is
deliberately absent -- two distinct genes with visually similar aliases
silently resolving to one canonical ID would be a leakage bug indistinguishable
from a correct disjoint split until someone checked by hand.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

from .mmr_dataset import MMR_UNIPROT

#: HGNC symbol -> canonical UniProt accession, taken verbatim from
#: src.mmr_dataset.MMR_UNIPROT (which pairs each symbol with an
#: expected-sequence-length-verified accession). Not independently
#: sourced: this module must never disagree with the dataset-build code
#: about what a gene's canonical ID is.
CANONICAL_GENE_IDS: Dict[str, str] = {
    symbol: accession for symbol, (accession, _expected_len) in MMR_UNIPROT.items()
}

#: Reverse lookup: canonical accession -> HGNC symbol. Public so callers that
#: only have a resolved accession (e.g. after :func:`resolve_gene_id`) can
#: still report the human-readable symbol in an error message.
ACCESSION_TO_SYMBOL: Dict[str, str] = {acc: sym for sym, acc in CANONICAL_GENE_IDS.items()}


class UnknownGeneError(ValueError):
    """Raised when a gene name/accession cannot be resolved to a canonical ID.

    Deliberately not a silent fallback: an unresolvable name is far more
    likely to be a typo or an alias this module has not been told about than
    a gene that is genuinely absent, and treating the two the same way would
    let a misspelled holdout gene silently fail to be held out.
    """


def resolve_gene_id(name: str) -> str:
    """Normalize a gene symbol or UniProt accession to its canonical accession.

    Accepts an HGNC symbol (case-insensitive, e.g. ``"mlh1"``) or the
    canonical UniProt accession itself (e.g. ``"P40692"``) -- the latter so a
    caller who already has an accession does not need a branch. Anything else
    raises :class:`UnknownGeneError`; there is no fuzzy matching (see module
    docstring for why).
    """
    if not isinstance(name, str) or not name.strip():
        raise UnknownGeneError(f"Gene identifier must be a non-empty string, got {name!r}.")
    key = name.strip().upper()
    if key in CANONICAL_GENE_IDS:
        return CANONICAL_GENE_IDS[key]
    if key in ACCESSION_TO_SYMBOL:
        return key
    raise UnknownGeneError(
        f"{name!r} is not a known gene symbol or UniProt accession. "
        f"Known symbols: {sorted(CANONICAL_GENE_IDS)}. "
        "This module does not guess or fuzzy-match aliases -- add the "
        "verified (symbol, accession) pair to src.mmr_dataset.MMR_UNIPROT "
        "first if this is a genuinely new gene.")


def resolve_gene_ids(names: Iterable[str]) -> tuple:
    """Resolve every name in *names*, preserving order, via :func:`resolve_gene_id`."""
    return tuple(resolve_gene_id(n) for n in names)


def assert_disjoint_gene_sets(train_genes: Sequence[str] = (),
                              val_genes: Sequence[str] = (),
                              test_genes: Sequence[str] = (),
                              holdout_genes: Sequence[str] = ()) -> None:
    """Raise if any gene (by canonical accession) appears in more than one partition.

    Every input is resolved through :func:`resolve_gene_id` first, so
    ``["MLH1"]`` in one partition and ``["P40692"]`` in another are correctly
    recognised as the same gene and flagged -- the exact alias-collision
    protection a raw string-equality check would miss.

    Partitions may be empty (e.g. no held-out set for a non-LOPO run); empty
    partitions trivially cannot conflict with anything.
    """
    named_partitions = (
        ("train", train_genes), ("val", val_genes),
        ("test", test_genes), ("holdout", holdout_genes),
    )
    resolved = {name: set(resolve_gene_ids(genes)) for name, genes in named_partitions}

    conflicts = []
    names = [n for n, _ in named_partitions]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = resolved[a] & resolved[b]
            if overlap:
                genes_readable = sorted(
                    ACCESSION_TO_SYMBOL.get(acc, acc) for acc in overlap)
                conflicts.append(f"{a} & {b} share {genes_readable}")
    if conflicts:
        raise ValueError(
            "Gene-level holdout violated -- the same gene appears in more "
            "than one partition:\n  " + "\n  ".join(conflicts))


__all__ = [
    "CANONICAL_GENE_IDS", "ACCESSION_TO_SYMBOL", "UnknownGeneError",
    "resolve_gene_id", "resolve_gene_ids", "assert_disjoint_gene_sets",
]
