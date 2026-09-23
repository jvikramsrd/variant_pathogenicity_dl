"""ClinVar classification words -> the five classes, without guessing.

ClinVar writes one classification per submission ("Likely pathogenic") and one
aggregate per variant, which may combine terms ("Pathogenic/Likely pathogenic",
"Pathogenic; risk factor", "Conflicting classifications of pathogenicity").

Rules, each one a decision a reader can check:

* one five-class term -> that class (``five_class``);
* "Pathogenic/Likely pathogenic" or "Benign/Likely benign" -> no hard class,
  a soft target of 0.5/0.5 over the two (``pair``). Picking either would
  invent a distinction the submitters did not make;
* "conflicting" -> ``conflicting``. Kept as rows — these are exactly the
  variants whose evidence disagrees — but never a five-class target;
* ", low penetrance" keeps the class and adds the modifier ``low_penetrance``;
* risk alleles, drug response, association, protective, "other" ->
  ``out_of_scope`` (not the ACMG/AMP Mendelian scale);
* terms after ";" are secondary (e.g. "; risk factor") and become modifiers;
* anything unrecognised -> ``out_of_scope`` with the raw text kept, counted by
  the inventory rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vpdl.slm.schema import CLASSES

__all__ = ["LabelInfo", "normalize_classification", "binary_label", "soft_target",
           "CATEGORIES"]

CATEGORIES = ("five_class", "pair", "conflicting", "out_of_scope", "missing")

_TERM = {
    "pathogenic": "pathogenic",
    "likely pathogenic": "likely_pathogenic",
    "uncertain significance": "vus",
    "variant of uncertain significance": "vus",
    "likely benign": "likely_benign",
    "benign": "benign",
}
# ClinVar's VUS sub-tiers (seen in the local 2026 release: "VUS-high" 48 and
# "VUS-mid" 42 variant-level records). They stay VUS; the tier is kept as a
# modifier because it is exactly the ordering VUS prioritisation is about.
_VUS_TIERS = {"vus-high": "vus_high", "vus-mid": "vus_mid", "vus-low": "vus_low"}
_MISSING = {"", "-", "not provided", "na", "none",
            "no classification for the single variant",
            "no classifications from unflagged records",
            "no classification provided"}


@dataclass(frozen=True)
class LabelInfo:
    raw: str
    category: str
    label5: str | None = None
    soft: tuple[float, ...] | None = None
    modifiers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def binary(self) -> float | None:
        return binary_label(self)


def soft_target(classes: set[str]) -> tuple[float, ...]:
    share = 1.0 / len(classes)
    return tuple(share if name in classes else 0.0 for name in CLASSES)


def normalize_classification(raw: object) -> LabelInfo:
    text = "" if raw is None else str(raw).strip()
    lowered = text.lower()
    if lowered in _MISSING:
        return LabelInfo(text, "missing")
    primary, *secondary = [part.strip() for part in lowered.split(";")]
    modifiers = [s for s in secondary if s]
    if "conflicting" in primary:
        return LabelInfo(text, "conflicting", modifiers=tuple(modifiers))
    if primary in _MISSING:
        return LabelInfo(text, "missing", modifiers=tuple(modifiers))

    classes: set[str] = set()
    unknown = []
    for term in primary.split("/"):
        term = term.strip()
        if term.endswith(", low penetrance"):
            term = term[: -len(", low penetrance")].strip()
            if "low_penetrance" not in modifiers:
                modifiers.append("low_penetrance")
        mapped = _TERM.get(term)
        if mapped is None and term in _VUS_TIERS:
            mapped = "vus"
            if _VUS_TIERS[term] not in modifiers:
                modifiers.append(_VUS_TIERS[term])
        if mapped is None:
            unknown.append(term)
        else:
            classes.add(mapped)
    if unknown or not classes:
        return LabelInfo(text, "out_of_scope", modifiers=tuple(modifiers + unknown))
    if len(classes) == 1:
        (only,) = classes
        return LabelInfo(text, "five_class", only, soft_target(classes), tuple(modifiers))
    if classes in ({"pathogenic", "likely_pathogenic"}, {"benign", "likely_benign"}):
        return LabelInfo(text, "pair", None, soft_target(classes), tuple(modifiers))
    # e.g. "Likely pathogenic/Uncertain significance": not a ClinVar aggregate term
    # today; if one appears, it disagrees with itself — treat as conflicting.
    return LabelInfo(text, "conflicting", modifiers=tuple(modifiers))


def binary_label(info: LabelInfo) -> float | None:
    """1 = pathogenic side (P, LP, P/LP), 0 = benign side, None otherwise (VUS included)."""
    if info.category == "five_class":
        return {"pathogenic": 1.0, "likely_pathogenic": 1.0,
                "benign": 0.0, "likely_benign": 0.0}.get(info.label5)
    if info.category == "pair" and info.soft is not None:
        return 1.0 if info.soft[0] > 0 else 0.0
    return None
