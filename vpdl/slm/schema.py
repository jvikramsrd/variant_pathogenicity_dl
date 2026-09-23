"""Tables of the genomic SLM, and the checks every build runs on them.

Four tables, each one row per unit, joined by ids (docs/slm/GENOMIC_SLM_DATASET.md):

    variants        one ClinVar variation (VariationID) — identity, location,
                    aggregate classification, diseases
    documents       one text with provenance — a ClinVar submission (SCV)
                    narrative, a ClinGen expert-panel summary, a passage
    evidence_units  one sentence of a document, with its role (description /
                    evidence / conclusion ...), evidence types, polarity and
                    ACMG codes, and the character span it came from
    citations       variant -> publication links (ClinVar ``var_citations``)

Nothing collapses evidence into one irreversible label: every document keeps
its own submitter, date and classification, and the variant's aggregate
classification is a separate column.

``classification_context`` on an evidence unit is the label of the document
the sentence came from. It exists for audits (e.g. how often a polarity cue
agrees with the document's label) and is NEVER an input: the example builder
refuses any feature named in :data:`NEVER_INPUT`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

__all__ = ["SLM_SCHEMA_VERSION", "CLASSES", "CLASS_INDEX", "TABLES", "NEVER_INPUT",
           "DATA_ROLES", "Column", "validate_table", "empty_table", "genomic_key",
           "clinvar_variant_id", "list_column", "records_to_frame", "arrow_schema", "FLOAT_LISTS",
           "as_list"]

SLM_SCHEMA_VERSION = "slm-genomic/1.0"

# Ordered pathogenic -> benign, so the class index is also an ordinal scale.
CLASSES = ("pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign")
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}

DATA_ROLES = ("KNOWLEDGE_BASE", "PRETRAINING", "TRAINING", "VALIDATION", "TEST",
              "INDEPENDENT_VALIDATION", "REFERENCE_ONLY", "EXCLUDED")

# Columns that describe the answer, not the evidence. The example builder
# refuses to put any of these into a model input (docs/slm/GENOMIC_SLM_DATA_LEAKAGE_REPORT.md).
NEVER_INPUT = frozenset({
    "clinvar_classification", "label5", "label_category", "label_soft",
    "classification_raw", "classification_context", "explanation_of_interpretation",
})


@dataclass(frozen=True)
class Column:
    name: str
    kind: str          # str | int | float | bool | list | date
    required: bool = True
    meaning: str = ""


def _columns(*specs: tuple) -> dict[str, Column]:
    return {spec[0]: Column(*spec) for spec in specs}


TABLES: dict[str, dict[str, Column]] = {
    "variants": _columns(
        ("variant_id", "str", True, "clinvar:<VariationID>"),
        ("variation_id", "int", True, "ClinVar VariationID"),
        ("allele_id", "int", False, "ClinVar AlleleID"),
        ("gene", "str", True, "primary gene symbol ('' when ClinVar gives none)"),
        ("genes", "list", True, "every gene symbol ClinVar lists"),
        ("name", "str", True, "ClinVar Name, e.g. NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)"),
        ("transcript", "str", False, "RefSeq transcript parsed from Name"),
        ("hgvs_c", "str", False, "coding / non-coding c. or n. change"),
        ("hgvs_p", "str", False, "protein change as written by ClinVar"),
        ("variant_type", "str", True, "ClinVar Type column"),
        ("consequence", "str", True, "coarse consequence derived from HGVS (vpdl.slm.variants)"),
        ("chromosome", "str", False, "GRCh38 chromosome"),
        ("start", "int", False, "GRCh38 start"),
        ("stop", "int", False, "GRCh38 stop"),
        ("genomic_key", "str", False, "GRCh38:chrom:pos:ref:alt (VCF) — the identity across transcripts"),
        ("rsid", "str", False, "dbSNP rs id"),
        ("clinvar_classification", "str", True, "aggregate classification, ClinVar's words"),
        ("label5", "str", False, "aggregate classification mapped to CLASSES (None if not one)"),
        ("label_category", "str", True, "five_class | pair | conflicting | out_of_scope | missing"),
        ("review_status", "str", True, "aggregate review status"),
        ("stars", "int", True, "0-4 review stars"),
        ("last_evaluated", "date", False, "aggregate last-evaluated date (ISO)"),
        ("number_submitters", "int", False, "submitters behind the aggregate"),
        ("origin", "str", False, "germline / somatic / ..."),
        ("disease_ids", "list", True, "MedGen/MONDO/OMIM ids from PhenotypeIDS"),
        ("disease_names", "list", True, "PhenotypeList"),
        ("protein_variant_id", "str", False, "DL-branch join key UniProt:pos:wt>mut, when mapped"),
        ("quality_flags", "list", True, "reasons to treat the row with care"),
        ("source_id", "str", True, "catalog source id"),
        ("source_version", "str", True, "sha256 of the file read"),
    ),
    "documents": _columns(
        ("document_id", "str", True, "SCV accession.version, or source-specific id"),
        ("variant_id", "str", True, "clinvar:<VariationID>"),
        ("variation_id", "int", True, "ClinVar VariationID"),
        ("gene", "str", True, "submitted gene symbol, else the variant's"),
        ("source_id", "str", True, "catalog source id"),
        ("submitter", "str", True, "laboratory / expert panel / curator"),
        ("collection_method", "str", False, "clinical testing / literature only / research ..."),
        ("review_status", "str", True, "this submission's review status"),
        ("classification_raw", "str", True, "the submitter's own words"),
        ("label5", "str", False, "classification_raw mapped to CLASSES"),
        ("label_category", "str", True, "five_class | pair | conflicting | out_of_scope | missing"),
        ("label_soft", "list", False, "5 probabilities for pair terms (P/LP, B/LB)"),
        ("date", "date", False, "date last evaluated (ISO)"),
        ("text", "str", True, "the narrative exactly as published ('' when none)"),
        ("disease_ids", "list", True, "MedGen ids from ReportedPhenotypeInfo"),
        ("disease_names", "list", True, "condition names"),
        ("tier", "int", True, "1-5 context tier (vpdl.slm.text.quality) — not a truth ranking"),
        ("restricted", "bool", True, "text under third-party terms (e.g. OMIM) — kept out of weights"),
        ("restricted_reason", "str", False, "why"),
        ("text_pmids", "list", True, "PMIDs written inside the text"),
    ),
    "evidence_units": _columns(
        ("evidence_id", "str", True, "<document_id>#<sentence index>"),
        ("document_id", "str", True, ""),
        ("variant_id", "str", True, ""),
        ("gene", "str", True, ""),
        ("source_id", "str", True, ""),
        ("char_start", "int", True, "offset into documents.text"),
        ("char_end", "int", True, "exclusive"),
        ("text_span", "str", True, "documents.text[char_start:char_end]"),
        ("sentence_role", "str", True, "description | evidence | conclusion | external_classification | boilerplate | other"),
        ("evidence_types", "list", True, "vpdl.slm.text.evidence.EVIDENCE_TYPES"),
        ("evidence_polarity", "str", True, "pathogenic | benign | neutral | mixed"),
        ("evidence_strength", "str", False, "from an ACMG strength suffix, else None"),
        ("acmg_codes", "list", True, "codes stated as applied, normalised (PM2_Supporting)"),
        ("acmg_codes_not_met", "list", True, "codes stated as not met"),
        ("pmids", "list", True, "PMIDs in the sentence"),
        ("date", "date", False, "the document's date"),
        ("provenance", "str", True, "how the unit was labelled, e.g. rule:evidence-lexicon/v1"),
        ("confidence", "float", True, "rule cue strength in [0, 1] — not a probability"),
        ("classification_context", "str", False, "the document's label — AUDIT ONLY, never input"),
    ),
    "citations": _columns(
        ("variation_id", "int", True, ""),
        ("allele_id", "int", False, ""),
        ("citation_source", "str", True, "PubMed | PubMedCentral | NCBIBookShelf | ..."),
        ("citation_id", "str", True, "the id in that source"),
    ),
    # Structured ACMG supervision: codes an expert panel recorded as met / not met
    # (ClinGen ERepo). Narrative-parsed codes live on evidence_units instead.
    "acmg_labels": _columns(
        ("document_id", "str", True, ""),
        ("variant_id", "str", True, ""),
        ("gene", "str", True, ""),
        ("codes_met", "list", True, "normalised codes"),
        ("codes_not_met", "list", True, "normalised codes"),
        ("source_id", "str", True, ""),
        ("provenance", "str", True, "expert_panel_structured"),
    ),
}

# Parquet types. Lists are lists of strings except where noted.
FLOAT_LISTS = frozenset({"label_soft"})


def arrow_schema(name: str):
    """The pyarrow schema of table `name` — explicit, so every chunk writes the same types."""
    import pyarrow as pa
    kinds = {"str": pa.string(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_(),
             "date": pa.string()}
    fields = []
    for column, spec in TABLES[name].items():
        if spec.kind == "list":
            kind = pa.list_(pa.float64() if column in FLOAT_LISTS else pa.string())
        else:
            kind = kinds[spec.kind]
        fields.append(pa.field(column, kind))
    return pa.schema(fields)


def empty_table(name: str) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in TABLES[name]})


def _is_listlike(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or (hasattr(value, "tolist")
                                                and not isinstance(value, str))


def validate_table(frame: pd.DataFrame, name: str, sample: int = 1000) -> list[str]:
    """Problems with `frame` as table `name`. Empty list = valid.

    Checks required columns, unknown columns, id uniqueness and — on a sample
    of rows — value kinds. Kinds are checked on a sample because the tables
    reach tens of millions of rows; the id checks run on every row.
    """
    schema = TABLES[name]
    problems = []
    missing = [c for c, spec in schema.items() if spec.required and c not in frame.columns]
    if missing:
        problems.append(f"{name}: missing required columns {missing}")
    extra = sorted(set(frame.columns) - set(schema))
    if extra:
        problems.append(f"{name}: undeclared columns {extra}")
    key = {"variants": "variant_id", "documents": "document_id",
           "evidence_units": "evidence_id"}.get(name)
    if key and key in frame.columns and frame[key].duplicated().any():
        problems.append(f"{name}: duplicate {key} "
                        f"(e.g. {frame.loc[frame[key].duplicated(), key].iloc[0]!r})")
    rows = frame.head(sample)
    for column, spec in schema.items():
        if column not in rows.columns or rows.empty:
            continue
        values = rows[column]
        if spec.kind == "list":
            bad = values.map(lambda v: not _is_listlike(v))
        elif spec.kind == "bool":
            bad = values.map(lambda v: not isinstance(v, (bool,)) and str(v) not in ("True", "False"))
        elif spec.kind == "int":
            bad = values.map(lambda v: v is not None and not pd.isna(v)
                             and not float(v).is_integer())
        elif spec.kind == "float":
            bad = values.map(lambda v: v is not None and not isinstance(v, (int, float)))
        else:
            bad = pd.Series(False, index=values.index)
        if spec.required and spec.kind in ("str",):
            bad = bad | values.isna()
        if bool(bad.any()):
            problems.append(f"{name}.{column}: {int(bad.sum())} value(s) of the wrong kind "
                            f"(expected {spec.kind}) in the first {len(rows)} rows")
    if name == "evidence_units" and {"char_start", "char_end"} <= set(rows.columns):
        backwards = rows["char_end"] < rows["char_start"]
        if bool(backwards.any()):
            problems.append("evidence_units: char_end < char_start")
    return problems


def genomic_key(chromosome: Any, position: Any, ref: Any, alt: Any,
                assembly: str = "GRCh38") -> str | None:
    """``GRCh38:17:43045712:G:A`` — one change however many transcripts name it."""
    values = [chromosome, position, ref, alt]
    if any(v is None or (isinstance(v, float) and pd.isna(v)) or str(v) in ("", "na", "-")
           for v in values):
        return None
    return f"{assembly}:{chromosome}:{int(position)}:{ref}:{alt}"


def clinvar_variant_id(variation_id: Any) -> str:
    return f"clinvar:{int(variation_id)}"


def as_list(value: Any) -> list:
    """A list, whatever Parquet handed back (numpy array, list, None, NaN).

    ``value or []`` is not safe here: an empty numpy array raises on truth-testing
    and a non-empty one is ambiguous.
    """
    if value is None:
        return []
    if isinstance(value, float) and value != value:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def list_column(values: Iterable[Any]) -> list:
    return [list(v) if _is_listlike(v) else ([] if v is None else [v]) for v in values]


def records_to_frame(records: Iterable[Mapping[str, Any]], name: str) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return empty_table(name)
    for column in TABLES[name]:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(TABLES[name])]
