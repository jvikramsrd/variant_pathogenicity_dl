"""ClinGen Evidence Repository (expert-panel curations) -> documents with met / not-met codes.

The ERepo is the best-supervised text this project can get: each record is an
expert panel's (VCEP) interpretation of one variant, with the ACMG codes the
panel applied, the ones it considered and did not apply, and a summary.

**The column names below are NOT verified.** ClinGen's download documentation
could not be reached on 2026-09-22. :data:`DEFAULT_COLUMNS` is a best-effort
map that must be checked against the first real download on the DGX; the
reader refuses to run if any mapped column is missing and prints the columns
the file actually has, so a wrong guess fails loudly on the first run instead
of producing an empty table. Pass ``columns=`` to override.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

from vpdl.slm.clinvar_text import parse_date
from vpdl.slm.text.acmg import normalize_code

__all__ = ["DEFAULT_COLUMNS", "iter_erepo", "parse_code_list", "COLUMNS_VERIFIED"]

COLUMNS_VERIFIED = False

DEFAULT_COLUMNS: dict[str, str] = {
    "variation_id": "ClinVar Variation Id",
    "gene": "HGNC Gene Symbol",
    "disease": "Disease",
    "disease_id": "Mondo Id",
    "classification": "Assertion",
    "codes_met": "Applied Evidence Codes (Met)",
    "codes_not_met": "Applied Evidence Codes (Not Met)",
    "summary": "Summary of interpretation",
    "pmids": "PubMed Articles",
    "panel": "Expert Panel",
    "approved": "Approval Date",
    "record_id": "Uuid",
}

_CODE_ITEM = re.compile(r"(PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])(?:[_ -]?([A-Za-z-]+))?")


def parse_code_list(value: str | None) -> list[str]:
    """'PM2_Supporting, PP3, PS3_Moderate' -> normalised codes, order kept."""
    out = []
    for code, suffix in _CODE_ITEM.findall(value or ""):
        normalized, _ = normalize_code(code, suffix or None)
        if normalized not in out:
            out.append(normalized)
    return out


def iter_erepo(path: Path | str, columns: Mapping[str, str] | None = None,
               delimiter: str = "\t", stats: Counter | None = None) -> Iterator[dict]:
    columns = dict(DEFAULT_COLUMNS | dict(columns or {}))
    stats = stats if stats is not None else Counter()
    with Path(path).open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = reader.fieldnames or []
        missing = [f"{key} -> {name!r}" for key, name in columns.items() if name not in header]
        if missing:
            raise ValueError(
                f"{path}: ERepo columns not found: {missing}. The file has: {header}. "
                "These names were never verified (vpdl/slm/erepo.py); pass columns={...} "
                "with the real names and record them in docs/slm/GENOMIC_SLM_KNOWLEDGE_BASE.md.")
        for row in reader:
            try:
                variation_id = int(float(row[columns["variation_id"]]))
            except (TypeError, ValueError):
                stats["erepo_without_clinvar_id"] += 1
                continue
            stats["erepo_records"] += 1
            yield {
                "variation_id": variation_id,
                "record_id": row.get(columns["record_id"], ""),
                "gene": row.get(columns["gene"], ""),
                "disease": row.get(columns["disease"], ""),
                "disease_id": row.get(columns["disease_id"], ""),
                "classification_raw": row.get(columns["classification"], ""),
                "codes_met": parse_code_list(row.get(columns["codes_met"])),
                "codes_not_met": parse_code_list(row.get(columns["codes_not_met"])),
                "text": row.get(columns["summary"], "") or "",
                "pmids": re.findall(r"\d{5,9}", row.get(columns["pmids"], "") or ""),
                "submitter": row.get(columns["panel"], "") or "ClinGen expert panel",
                "date": parse_date(row.get(columns["approved"])),
            }
