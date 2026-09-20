"""Table to matrices: feature families, ablation resolution, leak guards.

Satisfies regression landmines L2 (no feature is the label in disguise),
L5 (caches key on their schema), L6 (PLLR orientation), L11 (ablations refuse
to leave a proxy behind) and L12 (the WT/VT window contrast is real).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.evaluate import symmetric_agreement
from vpdl.provenance import schema_hash

logger = logging.getLogger(__name__)

__all__ = [
    "PRIOR_GROUPS",
    "PROXY_FOR",
    "group_of",
    "drop_gene_constant",
    "assert_no_label_proxy",
    "cache_key",
    "pllr_to_pathogenicity",
    "resolve_ablation",
    "local_windows",
    "FeatureMatrix",
    "build_feature_matrix",
]

# Feature families, matched by keyword rather than exact name so a renamed
# column does not silently fall out of its own ablation group.
PRIOR_GROUPS: dict[str, tuple[str, ...]] = {
    "gnomad": ("gnomad", "acmg_ba1", "acmg_bs1", "acmg_pm2", "allele_freq"),
    "prior_scores": ("alphamissense", "revel", "eve", "esm1b", "gemme",
                     "tranception", "primateai", "_score"),
    "structure": ("plddt", "disorder", "sasa", "secondary_structure"),
    "domains": ("interpro", "domain", "functional_site", "uniprot_site"),
}

# Which family acts as a stand-in for which other. AlphaMissense was trained on
# population and clinical data, so it carries allele-frequency signal: a
# "without gnomAD" ablation that leaves it in removes the legible copy of the
# signal and nothing else, and is uninterpretable.
PROXY_FOR: dict[str, frozenset[str]] = {
    "prior_scores": frozenset({"gnomad"}),
}


def group_of(column: str) -> str | None:
    """Family a feature column belongs to, or None when it belongs to none."""
    lowered = column.lower()
    for group, keywords in PRIOR_GROUPS.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return None


def drop_gene_constant(
    df: pd.DataFrame,
    columns: Sequence[str],
    gene_column: str = "gene",
    tolerance: float = 1e-12,
) -> list[str]:
    """Remove features that are constant within every gene.

    Under leave-one-gene-out such a column carries **gene identity and nothing
    else**: it takes one value across the whole training set for each gene, and
    a single unseen value on the held-out fold. The model can only use it to
    ask "which gene is this?", which is the one question the protocol exists to
    prevent it from answering.

    gnomAD's gene-level constraint columns (pLI, o/e missense, missense Z) are
    exactly this case, which is why the gnomAD source emits them and this
    function removes them again for LOPO. v1 did the same via
    ``prior_columns_of(df, drop_gene_constant=(eval == "lopo"))``; the guard is
    restated here because the trap is invisible — such a feature looks
    informative in any within-gene split and only misleads across genes.
    """
    if gene_column not in df.columns:
        return list(columns)

    kept: list[str] = []
    dropped: list[str] = []

    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        spread = values.groupby(df[gene_column]).transform(
            lambda series: series.max() - series.min()
        )
        varies = bool((spread.fillna(0.0) > tolerance).any())
        (kept if varies else dropped).append(column)

    if dropped:
        logger.info(
            "Dropped %d gene-constant feature(s) for leave-one-gene-out: %s. "
            "Within a gene these take a single value, so they encode gene "
            "identity rather than variant biology.",
            len(dropped), dropped,
        )
    return kept


def assert_no_label_proxy(
    df: pd.DataFrame,
    labels: Sequence[int],
    max_agreement: float = 0.95,
) -> None:
    """Raise if any feature column reproduces the label, in either orientation.

    `dms_bin_median` was the flipped label on 97.3% of rows in v1 and was fed to
    the model as a feature, producing ROC-AUC 0.9987 — a number good enough that
    it was nearly written up before anyone asked why. Orientation-symmetric on
    purpose: a perfectly *inverted* column is just as much a leak, and a
    one-sided check reads it as uninformative.
    """
    offenders: list[tuple[str, float]] = []
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        agreement = symmetric_agreement(labels, values)
        if np.isfinite(agreement) and agreement >= max_agreement:
            offenders.append((column, agreement))

    if offenders:
        detail = ", ".join(f"{name} ({value:.4f})" for name, value in offenders)
        raise ValueError(
            f"Label proxy detected — these columns reproduce the label and would "
            f"leak: {detail}. Agreement is measured symmetrically, so a flipped "
            "column counts. Drop them or justify each one explicitly."
        )


def cache_key(gene: str, model: str, columns: Iterable[str]) -> str:
    """Cache identity that includes the feature schema.

    v1 cached as ``{gene}_{model}_features.npz`` regardless of which prior
    columns had been appended, so a run *with* priors would silently load a
    cache built *without* them (CODE_REVIEW B3). Column order is excluded: it is
    not a modelling decision.
    """
    return f"{gene}__{model}__{schema_hash(columns)}"


def pllr_to_pathogenicity(raw_pllr: float | np.ndarray) -> float | np.ndarray:
    """Negate raw PLLR so that higher means more pathogenic.

    Raw pseudo-log-likelihood-ratio is NEGATIVE for damaging variants. Comparing
    it directly against a pathogenic=1 label reports 1 − AUC; v1 published
    ROC-AUC 0.03–0.18 for two strong 650M backbones this way.
    """
    return -np.asarray(raw_pllr) if isinstance(raw_pllr, np.ndarray) else -raw_pllr


def resolve_ablation(
    columns: Sequence[str],
    drop_groups: Sequence[str],
    allow_proxy_leak: bool = False,
) -> list[str]:
    """Drop feature families, refusing when a proxy for one remains.

    `allow_proxy_leak=True` is permitted but must be recorded in the run summary
    so the resulting arm is read as a joint bound rather than a clean removal.
    """
    to_drop = set(drop_groups)
    unknown = to_drop - set(PRIOR_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {sorted(unknown)}. "
            f"Known: {sorted(PRIOR_GROUPS)}."
        )

    kept = [column for column in columns if group_of(column) not in to_drop]

    if not allow_proxy_leak:
        remaining = {group_of(column) for column in kept} - {None}
        for dropped in sorted(to_drop):
            leaking = sorted(
                group for group in remaining
                if dropped in PROXY_FOR.get(group, frozenset())
            )
            if leaking:
                raise ValueError(
                    f"Refusing ablation: dropping '{dropped}' while {leaking} "
                    "remains leaves a proxy for the family being removed, so the "
                    "arm cannot be read as a clean removal. Drop both, or pass "
                    "allow_proxy_leak=True to report it as a joint bound."
                )

    return kept


def local_windows(
    sequence: str,
    position: int,
    wt_aa: str,
    mut_aa: str,
    radius: int = 3,
    pad: str = "-",
) -> tuple[str, str]:
    """Return ``(wild_type_window, variant_window)`` centred on the mutation.

    The variant window is cut from the MUTATED sequence. v1 sliced it from the
    wild-type sequence whenever the chain exceeded the model's positional
    capacity, so the two windows were identical, the (vt − wt) and |vt − wt|
    feature blocks were identically zero, and exactly one gene (MSH6, 1360 aa)
    had a different feature space from the other three — inside a
    leave-one-gene-out design.

    Window width is ``2 * radius + 1`` and the mutated residue is always at the
    centre index, padded at chain ends so that stays true.
    """
    if not 1 <= position <= len(sequence):
        raise ValueError(
            f"position {position} outside sequence of length {len(sequence)}"
        )
    if sequence[position - 1] != wt_aa:
        raise ValueError(
            f"wild-type mismatch at {position}: sequence has "
            f"{sequence[position - 1]!r}, variant claims {wt_aa!r}"
        )

    variant_sequence = sequence[: position - 1] + mut_aa + sequence[position:]

    def window(source: str) -> str:
        start = position - 1 - radius
        end = position + radius
        left_pad = pad * max(0, -start)
        right_pad = pad * max(0, end - len(source))
        return left_pad + source[max(0, start): min(len(source), end)] + right_pad

    return window(sequence), window(variant_sequence)


@dataclass
class FeatureMatrix:
    X: np.ndarray
    columns: list[str]
    means: np.ndarray
    scales: np.ndarray

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the TRAINING statistics to new rows. Never refits."""
        raw = df.reindex(columns=self.columns)
        values = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.means)
        return (values - self.means) / self.scales

    @property
    def schema(self) -> str:
        return schema_hash(self.columns)


def build_feature_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[int] | None = None,
    check_leaks: bool = True,
    max_agreement: float = 0.95,
) -> FeatureMatrix:
    """Standardise selected columns, imputing with the training median.

    Standardisation statistics are computed here and carried on the returned
    object so held-out folds and single-variant inference reuse them rather than
    refitting — refitting on the holdout is a leak that is easy to write and
    hard to see.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns absent from table: {missing}")

    frame = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")

    if check_leaks and labels is not None:
        assert_no_label_proxy(frame, labels, max_agreement=max_agreement)

    values = frame.to_numpy(dtype=float)
    means = np.nanmedian(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    values = np.where(np.isfinite(values), values, means)

    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)

    return FeatureMatrix(
        X=(values - means) / scales,
        columns=list(columns),
        means=means,
        scales=scales,
    )
