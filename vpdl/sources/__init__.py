"""Data sources, one module each, mutually ignorant by design.

A source implements :class:`vpdl.sources.base.Source`: ``fetch``, ``load``,
``provides``. It emits :data:`vpdl.sources.base.RECORD_COLUMNS` and nothing
else, so assembling any subset is concatenation rather than special-casing.
"""

from vpdl.sources.base import (
    RECORD_COLUMNS,
    VALID_AA,
    Source,
    SourceCapabilities,
    validate_against_sequence,
    validate_frame,
    validate_substitution,
)

__all__ = [
    "RECORD_COLUMNS",
    "VALID_AA",
    "Source",
    "SourceCapabilities",
    "validate_against_sequence",
    "validate_frame",
    "validate_substitution",
]
