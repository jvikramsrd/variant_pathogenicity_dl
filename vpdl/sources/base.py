"""The contract every data source implements, and the validation it must pass.

Sources know nothing about each other. That independence is what makes the
central experiment — one source alone versus all of them pooled — expressible
at all, so it is enforced here rather than left as a convention.

Satisfies regression landmines L4 (amino-acid validation is not vacuous) and
the coordinate-safety half of L9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "VALID_AA",
    "RECORD_COLUMNS",
    "SourceCapabilities",
    "Source",
    "validate_substitution",
    "validate_against_sequence",
    "validate_frame",
]

# The 20 standard proteinogenic residues. Ambiguity codes (B, Z, X, J) are NOT
# members: a variant we cannot resolve to a specific residue is not a variant we
# can score.
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Every source emits exactly these columns and nothing else. Extra information
# belongs in prefixed `feature_*` columns, so that assembling N sources is a
# concatenation rather than a negotiation.
RECORD_COLUMNS: tuple[str, ...] = (
    "uniprot_id",
    "position",
    "wt_aa",
    "mut_aa",
    "gene",
    "label",
    "label_source",
    "evidence_tier",
)


@dataclass(frozen=True)
class SourceCapabilities:
    """What a source can contribute, declared rather than inferred."""

    name: str
    supplies_labels: bool
    feature_columns: tuple[str, ...] = ()
    label_precedence: int | None = None
    licence: str = "unknown"
    notes: str = ""


@runtime_checkable
class Source(Protocol):
    """Structural contract. A module satisfying this is a source; no base class."""

    def fetch(self, cache_dir: Path) -> Path: ...

    def load(self, path: Path) -> pd.DataFrame: ...

    def provides(self) -> SourceCapabilities: ...


def _residue_problem(value: object, field: str) -> str | None:
    if not isinstance(value, str):
        return f"{field}={value!r} is not a string"
    if len(value) != 1:
        return f"{field}={value!r} is not a single residue (length {len(value)})"
    if value not in VALID_AA:
        return f"{field}={value!r} is not one of the 20 standard residues"
    return None


def validate_substitution(wt_aa: object, mut_aa: object, position: object) -> None:
    """Raise unless this is a well-formed single-residue substitution.

    Deliberately written per-value in plain Python. v1 vectorised this as
    ``np.char.str_len(np.asarray(df["mut_aa"], dtype="U1")) == 1`` — the cast to
    ``U1`` truncates every value to one character *before* the length check, so
    the check was always true and validated nothing (CODE_REVIEW B1). Any
    vectorised rewrite must keep :func:`validate_frame`'s tests passing.
    """
    problems = [
        problem for problem in (
            _residue_problem(wt_aa, "wt_aa"),
            _residue_problem(mut_aa, "mut_aa"),
        ) if problem
    ]

    if isinstance(position, bool) or not isinstance(position, (int,)):
        problems.append(f"position={position!r} is not an integer")
    elif position < 1:
        problems.append(f"position={position!r} is not 1-indexed and positive")

    if problems:
        raise ValueError("Invalid substitution: " + "; ".join(problems))


def validate_against_sequence(sequence: str, position: int, wt_aa: str) -> bool:
    """True when `wt_aa` is what the canonical sequence holds at `position`.

    UniProt is the coordinate authority for the whole project. Rows failing this
    are dropped and counted, never silently repositioned — roughly 50,000
    isoform-mismatched rows entered v1's table before this check existed.
    """
    if not isinstance(position, int) or position < 1 or position > len(sequence):
        return False
    return bool(sequence[position - 1] == wt_aa)


def validate_frame(
    df: pd.DataFrame,
    sequences: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Vectorised validation that keeps L4's semantics. Returns (kept, dropped).

    Correct-by-construction against B1: residue checks compare against
    :data:`VALID_AA` membership on the original object, with no dtype coercion
    anywhere in the path.
    """
    dropped: dict[str, int] = {}

    def _valid_residues(column: pd.Series) -> pd.Series:
        return column.map(lambda v: isinstance(v, str) and v in VALID_AA)

    ok = _valid_residues(df["wt_aa"]) & _valid_residues(df["mut_aa"])
    dropped["invalid_residue"] = int((~ok).sum())

    before = int(ok.sum())
    positions = pd.to_numeric(df["position"], errors="coerce")
    ok &= positions.notna() & (positions >= 1)
    dropped["invalid_position"] = before - int(ok.sum())

    before = int(ok.sum())
    ok &= df["wt_aa"] != df["mut_aa"]
    dropped["synonymous"] = before - int(ok.sum())

    if sequences:
        def _matches(row: pd.Series) -> bool:
            sequence = sequences.get(row["uniprot_id"])
            if sequence is None:
                return False
            position = pd.to_numeric(row["position"], errors="coerce")
            # Guarded because apply() visits rows that already failed the
            # position check above; int(nan) would take the whole build down.
            if pd.isna(position):
                return False
            return validate_against_sequence(sequence, int(position), row["wt_aa"])

        before = int(ok.sum())
        ok &= df.apply(_matches, axis=1)
        dropped["sequence_mismatch"] = before - int(ok.sum())
    else:
        # The coordinate-authority check is what stops isoform drift from
        # silently repositioning variants. Skipping it quietly is not an option.
        logger.warning(
            "No canonical sequences supplied — wild-type residue validation is "
            "SKIPPED for %d rows. Positions are being trusted as given, which "
            "is how ~50,000 isoform-mismatched rows entered the v1 table.",
            len(df),
        )
        dropped["sequence_mismatch"] = 0

    return df.loc[ok].reset_index(drop=True), dropped
