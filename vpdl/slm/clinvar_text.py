"""ClinVar's tab-delimited releases -> variant, document and citation records.

Three files from ``ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/`` (column
names checked against NCBI's README on 2026-09-22):

    variant_summary.txt.gz     one row per variant per assembly: identity, location,
                               aggregate classification, conditions. Already on this
                               PC (data/raw/, 442 MB).
    submission_summary.txt.gz  one row per submission (SCV). ``Description`` is
                               "an optional free text description of the basis of the
                               interpretation" — the lab narratives. NOT on this PC.
    var_citations.txt          variant -> citation (PubMed, PMC, Bookshelf ...).
                               NOT on this PC.

Readers stream (the files are GBs uncompressed) and never guess: a column the
release lacks is an error naming the columns it has, and every skipped row is
counted. ClinVar's own words are kept beside every normalised value.

Unlike ``vpdl.sources.clinvar.iter_variant_summary`` (missense in named genes
only, for the DL branch), these readers keep every gene and every variant
type — the SLM is a broad model.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

from vpdl.slm.labels import normalize_classification
from vpdl.slm.schema import clinvar_variant_id, genomic_key
from vpdl.slm.variants import consequence, parse_name
from vpdl.sources.clinvar import _review_stars

logger = logging.getLogger(__name__)

__all__ = ["parse_date", "parse_phenotype_ids", "parse_reported_phenotypes", "CATCH_ALL_CONDITIONS",
           "UMBRELLA_CONDITIONS", "parse_conditions", "specific_condition",
           "iter_variant_summary_all", "iter_submissions", "iter_citations",
           "VARIANT_SUMMARY_COLUMNS", "SUBMISSION_COLUMNS", "CITATION_COLUMNS"]

VARIANT_SUMMARY_COLUMNS = ("VariationID", "Type", "Name", "GeneSymbol", "ClinicalSignificance",
                           "ReviewStatus", "Assembly")
SUBMISSION_COLUMNS = ("VariationID", "ClinicalSignificance", "DateLastEvaluated", "Description",
                      "ReviewStatus", "Submitter", "SCV")
CITATION_COLUMNS = ("VariationID", "citation_source", "citation_id")

# Condition names that say "no specific disease". A disease-holdout split can
# neither hold these out nor treat two variants sharing them as related.
CATCH_ALL_CONDITIONS = frozenset({
    "not provided", "not specified", "see cases", "none provided", "not applicable",
    "variant of unknown significance", "all conditions", "unspecified", "",
})
# Umbrella terms some laboratories submit instead of a specific disease. In the
# local 2026-09 release they are among the five most frequent condition ids
# (MedGen C0950123 "Inborn genetic diseases" 362,633 variants; C0027672
# "Hereditary cancer-predisposing syndrome" 202,567; CN230736 "Cardiovascular
# phenotype" 93,976 — measured by `vpdl-slm inventory`). Grouping by them would
# make one "disease" of most of a laboratory's submissions.
UMBRELLA_CONDITIONS = frozenset({
    "inborn genetic diseases", "hereditary cancer-predisposing syndrome",
    "cardiovascular phenotype", "hereditary disease", "genetic disease",
})

_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%Y%m%d", "%d %b %Y", "%Y/%m/%d", "%b %Y", "%Y")


def parse_date(value: object) -> str | None:
    """ClinVar dates ("Dec 17, 2024", "2024-12-17", "-") -> ISO "2024-12-17" or None."""
    text = str(value or "").strip()
    if not text or text in ("-", "na", "NA", "None"):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _open_text(path: Path | str):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace",
                                newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def _header(handle, first_column: str) -> tuple[list[str], str | None]:
    """Skip ``##`` comment lines; return the header (``#`` stripped) and the first data line."""
    for line in handle:
        if line.startswith("##"):
            continue
        header = line.lstrip("#").rstrip("\r\n").split("\t")
        if header and header[0].strip() == first_column.lstrip("#"):
            return [h.strip() for h in header], None
        if line.startswith("#"):
            continue
        # No header line matched: the release changed; fail loudly below.
        return [h.strip() for h in header], line
    return [], None


def _require(header: list[str], required, path) -> dict[str, int]:
    index = {name: i for i, name in enumerate(header)}
    missing = [c for c in required if c not in index]
    if missing:
        raise ValueError(f"{path}: expected columns {missing} not in header {header[:20]} — "
                         "ClinVar changed the file layout; update vpdl/slm/clinvar_text.py.")
    return index


def parse_phenotype_ids(value: str | None) -> list[str]:
    """variant_summary ``PhenotypeIDS``: '|' between conditions, ',' between ids of one.

    One id per condition: MedGen when present (ClinVar's own concept), else MONDO,
    else the first id given.
    """
    out = []
    for condition in str(value or "").split("|"):
        ids = [part.strip() for part in condition.split(",") if part.strip() and part.strip() != "-"]
        if not ids:
            continue
        medgen = next((i for i in ids if i.startswith("MedGen:")), None)
        mondo = next((i for i in ids if i.startswith("MONDO:")), None)
        chosen = medgen or (mondo.replace("MONDO:MONDO:", "MONDO:") if mondo else ids[0])
        if chosen not in out:
            out.append(chosen)
    return out


def parse_conditions(ids_field: str | None, names_field: str | None) -> list[tuple[str | None, str]]:
    """variant_summary conditions as aligned ``(id, name)`` pairs ('|' separates both)."""
    names = [n.strip() for n in str(names_field or "").split("|")]
    id_groups = str(ids_field or "").split("|")
    pairs = []
    for position, name in enumerate(names):
        ids = parse_phenotype_ids(id_groups[position]) if position < len(id_groups) else []
        if not name or name == "-":
            continue
        pairs.append((ids[0] if ids else None, name))
    return pairs


def specific_condition(name: str | None) -> bool:
    """False for catch-all and umbrella condition names."""
    key = (name or "").strip().lower()
    return key not in CATCH_ALL_CONDITIONS and key not in UMBRELLA_CONDITIONS


_REPORTED = re.compile(r"^(?P<id>(?:C|CN)\d+)\s*:\s*(?P<name>.*)$")


def parse_reported_phenotypes(value: str | None) -> tuple[list[str], list[str]]:
    """submission_summary ``ReportedPhenotypeInfo``: 'C0009405:Lynch syndrome;...'."""
    ids, names = [], []
    for part in re.split(r"[;|]", str(value or "")):
        part = part.strip()
        if not part or part == "-":
            continue
        match = _REPORTED.match(part)
        if match:
            identifier, name = f"MedGen:{match.group('id')}", match.group("name").strip()
        else:
            identifier, name = None, part
        if identifier and identifier not in ids:
            ids.append(identifier)
        if name and name not in names:
            names.append(name)
    return ids, names


def iter_variant_summary_all(path: Path | str, genes: set[str] | None = None,
                             stats: Counter | None = None,
                             prefer_assembly: str = "GRCh38") -> Iterator[dict]:
    """One record per VariationID (GRCh38 row preferred), every gene and type.

    Rows arrive once per assembly; the file is grouped by variant, so the two
    rows of one variant are adjacent in practice — but that is not relied on:
    records are buffered by VariationID and flushed when a new id appears, with
    a final check that no id is emitted twice.
    """
    stats = stats if stats is not None else Counter()
    emitted: set[int] = set()
    with _open_text(path) as handle:
        header, pending = _header(handle, "#AlleleID")
        index = _require(header, VARIANT_SUMMARY_COLUMNS, path)

        def get(fields, column):
            position = index.get(column)
            return fields[position] if position is not None and position < len(fields) else None

        buffer: dict[int, dict] = {}
        lines = ([pending] if pending else [])
        for line in _chain(lines, handle):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                stats["variant_summary_malformed_rows"] += 1
                continue
            try:
                variation_id = int(get(fields, "VariationID"))
            except (TypeError, ValueError):
                stats["variant_summary_bad_variation_id"] += 1
                continue
            symbols = [s for s in re.split(r"[;|]", get(fields, "GeneSymbol") or "") if s and s != "-"]
            if genes is not None and not (set(symbols) & genes):
                continue
            assembly = get(fields, "Assembly") or ""
            if variation_id in buffer:
                if assembly == prefer_assembly:
                    buffer[variation_id] = _variant_record(fields, get, symbols, variation_id, path)
                continue
            for old_id in list(buffer):
                if old_id != variation_id:
                    yield from _flush(buffer.pop(old_id), emitted, stats)
            buffer[variation_id] = _variant_record(fields, get, symbols, variation_id, path)
        for record in buffer.values():
            yield from _flush(record, emitted, stats)


def _chain(first, rest):
    yield from first
    yield from rest


def _flush(record: dict, emitted: set[int], stats: Counter) -> Iterator[dict]:
    if record["variation_id"] in emitted:
        stats["variant_summary_non_adjacent_duplicate"] += 1
        return
    emitted.add(record["variation_id"])
    stats["variants"] += 1
    yield record


def _variant_record(fields, get, symbols, variation_id, path) -> dict:
    name = get(fields, "Name") or ""
    parsed = parse_name(name)
    classification = get(fields, "ClinicalSignificance") or ""
    label = normalize_classification(classification)
    review_status = get(fields, "ReviewStatus") or ""
    assembly = get(fields, "Assembly") or ""
    chromosome = get(fields, "Chromosome")
    start, stop = _int(get(fields, "Start")), _int(get(fields, "Stop"))
    length = (stop - start + 1) if start is not None and stop is not None else None
    variant_type = get(fields, "Type") or ""
    flags = []
    if assembly != "GRCh38":
        flags.append(f"assembly_{assembly or 'unknown'}")
    if len(symbols) > 1:
        flags.append("multiple_genes")
    if label.category == "out_of_scope":
        flags.append("classification_out_of_scope")
    conditions = parse_conditions(get(fields, "PhenotypeIDS") or get(fields, "PhenotypeIDs"),
                                  get(fields, "PhenotypeList"))
    return {
        "variant_id": clinvar_variant_id(variation_id),
        "variation_id": variation_id,
        "allele_id": _int(get(fields, "AlleleID") or get(fields, "#AlleleID")),
        "gene": parsed.gene if parsed.gene in symbols else (symbols[0] if symbols else ""),
        "genes": symbols,
        "name": name,
        "transcript": parsed.transcript,
        "hgvs_c": parsed.change,
        "hgvs_p": parsed.protein,
        "variant_type": variant_type,
        "consequence": consequence(variant_type, name, length),
        "chromosome": chromosome if assembly == "GRCh38" else None,
        "start": start if assembly == "GRCh38" else None,
        "stop": stop if assembly == "GRCh38" else None,
        "genomic_key": genomic_key(chromosome, get(fields, "PositionVCF"),
                                   get(fields, "ReferenceAlleleVCF"),
                                   get(fields, "AlternateAlleleVCF"))
        if assembly == "GRCh38" and _int(get(fields, "PositionVCF")) not in (None, -1) else None,
        "rsid": _rsid(get(fields, "RS# (dbSNP)")),
        "clinvar_classification": classification,
        "label5": label.label5,
        "label_category": label.category,
        "review_status": review_status,
        "stars": _review_stars(review_status),
        "last_evaluated": parse_date(get(fields, "LastEvaluated")),
        "number_submitters": _int(get(fields, "NumberSubmitters")),
        "origin": get(fields, "OriginSimple"),
        # Aligned lists: disease_ids[i] names disease_names[i] ("" when no id was given).
        "disease_ids": [identifier or "" for identifier, _ in conditions],
        "disease_names": [name for _, name in conditions],
        "protein_variant_id": None,
        "quality_flags": flags,
        "source_id": "clinvar_variant_summary",
        "source_version": "",
    }


def _int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _rsid(value) -> str | None:
    text = str(value or "").strip()
    return f"rs{text}" if text.isdigit() and text != "-1" else None


def iter_submissions(path: Path | str, stats: Counter | None = None) -> Iterator[dict]:
    """One dict per SCV, the narrative (``Description``) exactly as published."""
    stats = stats if stats is not None else Counter()
    with _open_text(path) as handle:
        header, pending = _header(handle, "#VariationID")
        index = _require(header, SUBMISSION_COLUMNS, path)

        def get(fields, column):
            position = index.get(column)
            return fields[position] if position is not None and position < len(fields) else None

        for line in _chain([pending] if pending else [], handle):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                stats["submission_malformed_rows"] += 1
                continue
            variation_id = _int(get(fields, "VariationID"))
            if variation_id is None:
                stats["submission_bad_variation_id"] += 1
                continue
            description = get(fields, "Description") or ""
            if description.strip() == "-":
                description = ""
            reported_ids, reported_names = parse_reported_phenotypes(get(fields, "ReportedPhenotypeInfo"))
            stats["submissions"] += 1
            yield {
                "variation_id": variation_id,
                "classification_raw": get(fields, "ClinicalSignificance") or "",
                "date": parse_date(get(fields, "DateLastEvaluated")),
                "text": description,
                "submitted_phenotype": get(fields, "SubmittedPhenotypeInfo") or "",
                "disease_ids": reported_ids,
                "disease_names": reported_names,
                "review_status": get(fields, "ReviewStatus") or "",
                "collection_method": get(fields, "CollectionMethod") or "",
                "origin_counts": get(fields, "OriginCounts") or "",
                "submitter": get(fields, "Submitter") or "",
                "scv": get(fields, "SCV") or "",
                "submitted_gene": get(fields, "SubmittedGeneSymbol") or "",
                "explanation_of_interpretation": get(fields, "ExplanationOfInterpretation") or "",
            }


def iter_citations(path: Path | str, stats: Counter | None = None) -> Iterator[dict]:
    stats = stats if stats is not None else Counter()
    with _open_text(path) as handle:
        header, pending = _header(handle, "#AlleleID")
        index = _require(header, CITATION_COLUMNS, path)
        for line in _chain([pending] if pending else [], handle):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                stats["citation_malformed_rows"] += 1
                continue
            variation_id = _int(fields[index["VariationID"]])
            if variation_id is None:
                continue
            stats["citations"] += 1
            yield {"variation_id": variation_id,
                   "allele_id": _int(fields[index["AlleleID"]]) if "AlleleID" in index else None,
                   "citation_source": fields[index["citation_source"]].strip(),
                   "citation_id": fields[index["citation_id"]].strip()}
