"""ClinVar (+ ClinGen ERepo) files -> the SLM's tables, streamed, with a manifest.

    vpdl-slm build-records --variant-summary data/raw/variant_summary.txt.gz \
        --submission-summary data/raw/clinvar/submission_summary.txt.gz \
        --citations data/raw/clinvar/var_citations.txt --out data/slm_genomic

Writes ``variants / documents / evidence_units / citations / acmg_labels``
as Parquet (explicit Arrow schemas, written in row groups so memory stays
flat), plus ``manifest.json``: the sha256 and size of every input, row counts,
every reader's skip counts, the schema and preprocessing versions and the git
state. Two builds are comparable only when their manifests' input hashes and
versions agree.

Evidence extraction is the slow part (regex over millions of narratives) and
runs in ``workers`` processes; output order is the input order either way.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from vpdl.slm.clinvar_text import iter_citations, iter_submissions, iter_variant_summary_all
from vpdl.slm.labels import normalize_classification
from vpdl.slm.schema import (SLM_SCHEMA_VERSION, TABLES, arrow_schema, as_list,
                             clinvar_variant_id)
from vpdl.slm.text.evidence import PMID, RULE_PROVENANCE, extract_units
from vpdl.slm.text.quality import assign_tier, restriction
from vpdl.slm.variants import DLJoin

logger = logging.getLogger(__name__)

__all__ = ["TABLE_FILES", "build_records", "load_tables", "units_for_document", "variant_view",
           "build_edges", "PREPROCESSING_VERSION"]

PREPROCESSING_VERSION = "slm-genomic-prep/1"
TABLE_FILES = {name: f"{name}.parquet" for name in TABLES}


class _ChunkWriter:
    def __init__(self, path: Path, table: str, chunk_size: int):
        import pyarrow.parquet as pq
        self.path, self.table, self.chunk_size = path, table, chunk_size
        self.schema = arrow_schema(table)
        self.writer = pq.ParquetWriter(str(path), self.schema)
        self.buffer: list[dict] = []
        self.rows = 0

    def add(self, record: Mapping[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa
        if not self.buffer:
            return
        columns = {name: [row.get(name) for row in self.buffer] for name in TABLES[self.table]}
        self.writer.write_table(pa.Table.from_pydict(columns, schema=self.schema))
        self.rows += len(self.buffer)
        self.buffer = []

    def close(self) -> int:
        self.flush()
        self.writer.close()
        return self.rows


def units_for_document(document: Mapping[str, Any]) -> list[dict]:
    """Evidence-unit rows for one document (top-level: runs in worker processes)."""
    rows = []
    for unit in extract_units(document.get("text") or ""):
        rows.append({
            "evidence_id": f"{document['document_id']}#{unit.index}",
            "document_id": document["document_id"],
            "variant_id": document["variant_id"],
            "gene": document.get("gene") or "",
            "source_id": document["source_id"],
            "char_start": unit.start, "char_end": unit.end, "text_span": unit.text,
            "sentence_role": unit.sentence_role,
            "evidence_types": list(unit.evidence_types),
            "evidence_polarity": unit.evidence_polarity,
            "evidence_strength": unit.evidence_strength,
            "acmg_codes": list(unit.acmg_codes),
            "acmg_codes_not_met": list(unit.acmg_codes_not_met),
            "pmids": list(unit.pmids),
            "date": document.get("date"),
            "provenance": RULE_PROVENANCE,
            "confidence": float(unit.confidence),
            "classification_context": document.get("label5") or document.get("label_category"),
        })
    return rows


def _units_batch(documents: list[dict]) -> list[list[dict]]:
    return [units_for_document(d) for d in documents]


def _file_identity(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    from vpdl.slm.build.inventory import sha256_file
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _document_record(submission: Mapping[str, Any], variant_gene: str | None,
                     row_number: int) -> dict:
    label = normalize_classification(submission["classification_raw"])
    submitted = (submission.get("submitted_gene") or "").strip()
    reason = restriction(submission.get("submitter"))
    text = submission.get("text") or ""
    return {
        "document_id": submission.get("scv") or f"clinvar-row-{row_number}",
        "variant_id": clinvar_variant_id(submission["variation_id"]),
        "variation_id": submission["variation_id"],
        "gene": submitted if submitted and submitted != "-" else (variant_gene or ""),
        "source_id": "clinvar_submission_summary",
        "submitter": submission.get("submitter") or "",
        "collection_method": submission.get("collection_method") or "",
        "review_status": submission.get("review_status") or "",
        "classification_raw": submission["classification_raw"],
        "label5": label.label5,
        "label_category": label.category,
        "label_soft": list(label.soft) if label.soft else None,
        "date": submission.get("date"),
        "text": text,
        "disease_ids": list(submission.get("disease_ids") or []),
        "disease_names": list(submission.get("disease_names") or []),
        "tier": assign_tier("clinvar_submission_summary", submission.get("review_status"),
                            submission.get("collection_method"), text),
        "restricted": reason is not None,
        "restricted_reason": reason,
        "text_pmids": list(dict.fromkeys(PMID.findall(text))),
    }


def _erepo_document(record: Mapping[str, Any]) -> dict:
    label = normalize_classification(record["classification_raw"])
    text = record.get("text") or ""
    return {
        "document_id": f"erepo:{record.get('record_id') or record['variation_id']}",
        "variant_id": clinvar_variant_id(record["variation_id"]),
        "variation_id": record["variation_id"],
        "gene": record.get("gene") or "",
        "source_id": "clingen_erepo",
        "submitter": record.get("submitter") or "ClinGen expert panel",
        "collection_method": "curation",
        "review_status": "reviewed by expert panel",
        "classification_raw": record["classification_raw"],
        "label5": label.label5, "label_category": label.category,
        "label_soft": list(label.soft) if label.soft else None,
        "date": record.get("date"), "text": text,
        "disease_ids": [record["disease_id"]] if record.get("disease_id") else [],
        "disease_names": [record["disease"]] if record.get("disease") else [],
        "tier": 1, "restricted": False, "restricted_reason": None,
        "text_pmids": list(record.get("pmids") or []),
    }


def build_records(out_dir: Path | str, variant_summary: Path | str,
                  submission_summary: Path | str | None = None,
                  var_citations: Path | str | None = None,
                  erepo: Path | str | None = None, erepo_columns: Mapping[str, str] | None = None,
                  dl_canonical: Path | str | None = None, genes: Iterable[str] | None = None,
                  limit_documents: int | None = None, workers: int = 1,
                  chunk_size: int = 100_000) -> dict[str, Any]:
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    wanted = set(genes) if genes else None
    join = DLJoin.from_canonical(dl_canonical) if dl_canonical else DLJoin.empty()
    inputs = {"variant_summary": _file_identity(variant_summary),
              "submission_summary": _file_identity(submission_summary),
              "var_citations": _file_identity(var_citations),
              "erepo": _file_identity(erepo), "dl_canonical": _file_identity(dl_canonical)}

    # 1. variants ----------------------------------------------------------------
    variant_gene: dict[int, str] = {}
    writer = _ChunkWriter(out / TABLE_FILES["variants"], "variants", chunk_size)
    version = inputs["variant_summary"]["sha256"]
    for record in iter_variant_summary_all(variant_summary, genes=wanted, stats=stats):
        record["source_version"] = version
        record["protein_variant_id"] = join.get(record["variation_id"])
        variant_gene[record["variation_id"]] = record["gene"]
        writer.add(record)
    counts = {"variants": writer.close()}
    logger.info("variants: %d", counts["variants"])

    # 2. documents + evidence units -------------------------------------------------
    doc_writer = _ChunkWriter(out / TABLE_FILES["documents"], "documents", chunk_size)
    unit_writer = _ChunkWriter(out / TABLE_FILES["evidence_units"], "evidence_units", chunk_size)
    acmg_writer = _ChunkWriter(out / TABLE_FILES["acmg_labels"], "acmg_labels", chunk_size)
    seen_documents: set[str] = set()

    def documents() -> Iterator[dict]:
        if submission_summary is not None:
            for row, submission in enumerate(iter_submissions(submission_summary, stats)):
                if submission["variation_id"] not in variant_gene:
                    stats["documents_for_variants_not_kept"] += 1
                    continue
                yield _document_record(submission, variant_gene.get(submission["variation_id"]), row)
        if erepo is not None:
            from vpdl.slm.erepo import iter_erepo
            for record in iter_erepo(erepo, erepo_columns, stats=stats):
                if record["variation_id"] not in variant_gene:
                    stats["erepo_for_variants_not_kept"] += 1
                    continue
                document = _erepo_document(record)
                acmg_writer.add({"document_id": document["document_id"],
                                 "variant_id": document["variant_id"], "gene": document["gene"],
                                 "codes_met": record["codes_met"],
                                 "codes_not_met": record["codes_not_met"],
                                 "source_id": "clingen_erepo",
                                 "provenance": "expert_panel_structured"})
                yield document

    def batches(size: int = 2000) -> Iterator[list[dict]]:
        batch = []
        for document in documents():
            if document["document_id"] in seen_documents:
                stats["duplicate_document_id"] += 1
                continue
            seen_documents.add(document["document_id"])
            doc_writer.add(document)
            batch.append(document)
            if limit_documents and len(seen_documents) >= limit_documents:
                break
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    pool = multiprocessing.Pool(workers) if workers > 1 else None
    try:
        results = (pool.imap(_units_batch, batches()) if pool else map(_units_batch, batches()))
        for unit_lists in results:
            for units in unit_lists:
                for unit in units:
                    unit_writer.add(unit)
    finally:
        if pool is not None:
            pool.terminate()
    counts["documents"] = doc_writer.close()
    counts["evidence_units"] = unit_writer.close()
    counts["acmg_labels"] = acmg_writer.close()

    # 3. citations -------------------------------------------------------------------
    cite_writer = _ChunkWriter(out / TABLE_FILES["citations"], "citations", chunk_size)
    if var_citations is not None:
        for citation in iter_citations(var_citations, stats):
            if citation["variation_id"] in variant_gene:
                cite_writer.add(citation)
    counts["citations"] = cite_writer.close()

    from vpdl.provenance import _git_state
    manifest = {
        "schema_version": SLM_SCHEMA_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "evidence_provenance": RULE_PROVENANCE, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 1), "inputs": inputs,
        "genes_filter": sorted(wanted) if wanted else None, "limit_documents": limit_documents,
        "dl_join": {"source": join.source, "variation_ids_mapped": len(join)},
        "counts": counts, "reader_stats": dict(stats), "git": _git_state(),
        "files": {name: TABLE_FILES[name] for name in TABLES},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def load_tables(directory: Path | str, names: Iterable[str] = tuple(TABLES),
                columns: Mapping[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    tables = {}
    for name in names:
        path = directory / TABLE_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `vpdl-slm build-records` first.")
        tables[name] = pd.read_parquet(path, columns=(columns or {}).get(name))
        for column, spec in TABLES[name].items():
            if spec.kind == "list" and column in tables[name].columns:
                tables[name][column] = tables[name][column].map(
                    lambda v: list(v) if v is not None and not isinstance(v, float) else [])
    return tables


def variant_view(tables: Mapping[str, pd.DataFrame], variant_id: str) -> dict[str, Any]:
    """The variant-centric record (GENOMIC_SLM_KNOWLEDGE_BASE.md): everything known, with provenance.

    Nothing is collapsed: each document keeps its own classification, date and
    submitter; evidence stays per sentence. Sources that are not joined yet
    (population frequency beyond the DL panel, functional data) are present as
    empty containers with a quality flag, never filled with defaults.
    """
    variants = tables["variants"]
    row = variants.loc[variants["variant_id"] == variant_id]
    if row.empty:
        raise KeyError(variant_id)
    v = row.iloc[0].to_dict()
    documents = tables.get("documents", pd.DataFrame())
    docs = documents.loc[documents["variant_id"] == variant_id] if not documents.empty else documents
    units = tables.get("evidence_units", pd.DataFrame())
    unit_rows = units.loc[units["variant_id"] == variant_id] if not units.empty else units
    citations = tables.get("citations", pd.DataFrame())
    cites = citations.loc[citations["variation_id"] == v["variation_id"]] if not citations.empty else citations
    flags = as_list(v.get("quality_flags"))
    flags += ["population_frequency_not_joined", "functional_evidence_not_joined"]
    return {
        "variant_id": v["variant_id"], "chromosome": v.get("chromosome"), "position": v.get("start"),
        "genomic_key": v.get("genomic_key"), "gene": v.get("gene"), "transcript": v.get("transcript"),
        "hgvs_c": v.get("hgvs_c"), "hgvs_p": v.get("hgvs_p"), "variant_type": v.get("variant_type"),
        "molecular_consequence": v.get("consequence"),
        "protein_variant_id": v.get("protein_variant_id"),
        "disease": list(zip(as_list(v.get("disease_ids")), as_list(v.get("disease_names")))),
        "population_frequency": {},
        "clinical_interpretations": [
            {"document_id": d["document_id"], "submitter": d["submitter"],
             "classification": d["classification_raw"], "date": d["date"],
             "review_status": d["review_status"], "tier": int(d["tier"]),
             "restricted": bool(d["restricted"])} for _, d in docs.iterrows()],
        "aggregate": {"classification": v.get("clinvar_classification"),
                      "review_status": v.get("review_status"), "stars": v.get("stars"),
                      "last_evaluated": v.get("last_evaluated")},
        "evidence": [
            {"evidence_id": u["evidence_id"], "types": list(u["evidence_types"]),
             "polarity": u["evidence_polarity"], "acmg_codes": list(u["acmg_codes"]),
             "text": u["text_span"], "role": u["sentence_role"], "provenance": u["provenance"]}
            for _, u in unit_rows.iterrows() if u["sentence_role"] in ("evidence", "code_list")],
        "literature_evidence": [f"{c['citation_source']}:{c['citation_id']}" for _, c in cites.iterrows()],
        "functional_evidence": [], "computational_evidence": [],
        "source_provenance": [{"source_id": v.get("source_id"), "version": v.get("source_version")}],
        "quality_flags": flags,
    }


def _present(value: Any) -> bool:
    """True for a real value; False for None, NaN (Parquet's missing string) and ''."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    return bool(value)


def build_edges(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Entity relationships (variant-gene, variant-disease, document-variant, ...) with provenance."""
    edges: list[dict] = []
    for _, v in tables["variants"].iterrows():
        for gene in as_list(v["genes"]) or ([v["gene"]] if v["gene"] else []):
            edges.append({"subject": v["variant_id"], "predicate": "in_gene", "object": f"gene:{gene}",
                          "source_id": v["source_id"], "document_id": None})
        for identifier, name in zip(as_list(v["disease_ids"]), as_list(v["disease_names"])):
            edges.append({"subject": v["variant_id"], "predicate": "asserted_for",
                          "object": identifier or f"condition:{name}", "source_id": v["source_id"],
                          "document_id": None})
        if _present(v.get("protein_variant_id")):
            edges.append({"subject": v["variant_id"], "predicate": "encodes_protein_change",
                          "object": f"protein:{v['protein_variant_id']}", "source_id": "dl_canonical",
                          "document_id": None})
    for _, d in tables.get("documents", pd.DataFrame()).iterrows():
        edges.append({"subject": d["document_id"], "predicate": "interprets", "object": d["variant_id"],
                      "source_id": d["source_id"], "document_id": d["document_id"]})
        for pmid in as_list(d["text_pmids"]):
            edges.append({"subject": d["document_id"], "predicate": "cites", "object": f"PMID:{pmid}",
                          "source_id": d["source_id"], "document_id": d["document_id"]})
    for _, c in tables.get("citations", pd.DataFrame()).iterrows():
        edges.append({"subject": clinvar_variant_id(c["variation_id"]), "predicate": "cited_by",
                      "object": f"{c['citation_source']}:{c['citation_id']}",
                      "source_id": "clinvar_var_citations", "document_id": None})
    for _, u in tables.get("evidence_units", pd.DataFrame()).iterrows():
        for code in as_list(u["acmg_codes"]):
            edges.append({"subject": u["evidence_id"], "predicate": "states_acmg_code",
                          "object": f"acmg:{code}", "source_id": u["source_id"],
                          "document_id": u["document_id"]})
    return pd.DataFrame(edges, columns=["subject", "predicate", "object", "source_id", "document_id"])
