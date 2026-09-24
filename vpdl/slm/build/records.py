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

The build is sharded: the main process only decompresses and cuts each file
into chunks of raw lines; every worker process parses its chunk, extracts the
evidence units and writes its own part file. ``workers`` defaults to every
core (20 on the DGX Spark). Each table is a directory of
``part-NNNNN.parquet`` files named by chunk index, so output order is input
order and the tables are identical for any number of workers.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.slm.clinvar_text import (SUBMISSION_COLUMNS, VARIANT_SUMMARY_COLUMNS, iter_citations,
                                   iter_line_chunks, parse_submission_lines, parse_variant_lines)
from vpdl.slm.parallel import default_workers, run_windowed
from vpdl.slm.labels import normalize_classification
from vpdl.slm.schema import (SLM_SCHEMA_VERSION, TABLES, arrow_schema, as_list,
                             clinvar_variant_id)
from vpdl.slm.text.evidence import PMID, RULE_PROVENANCE, extract_units
from vpdl.slm.text.quality import assign_tier, restriction
from vpdl.slm.variants import DLJoin

logger = logging.getLogger(__name__)

__all__ = ["TABLE_FILES", "build_records", "load_tables", "units_for_document", "variant_view",
           "build_edges", "PREPROCESSING_VERSION", "VariantIndex"]

PREPROCESSING_VERSION = "slm-genomic-prep/1"
TABLE_FILES = {name: f"{name}.parquet" for name in TABLES}


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


def _file_identity(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    from vpdl.slm.build.inventory import sha256_file
    path = Path(path)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


PART = "part-{:05d}.parquet"
VARIANT_CHUNK_LINES = 100_000
SUBMISSION_CHUNK_LINES = 10_000


def _reset_table(out: Path, name: str) -> Path:
    """The table's directory, emptied — a rebuild never mixes with an older one."""
    path = out / TABLE_FILES[name]
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()                       # the single-file layout of earlier builds
    path.mkdir(parents=True)
    return path


def _write_part(directory: Path, table: str, index: int, records: list[Mapping[str, Any]]) -> int:
    """One part file, written with the table's explicit schema (an empty part is valid)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = arrow_schema(table)
    columns = {name: [row.get(name) for row in records] for name in TABLES[table]}
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), str(directory / PART.format(index)))
    return len(records)


class VariantIndex:
    """VariationID -> gene, as sorted numpy arrays.

    Handed to every worker. Numpy arrays, not a 4.5-million-entry dict: forked
    workers share the pages instead of copying them as Python touches reference
    counts, so twenty workers cost one copy of the index, not twenty.
    """

    def __init__(self, ids: np.ndarray, genes: Sequence[str]):
        order = np.argsort(ids, kind="stable")
        self.ids = np.asarray(ids, dtype=np.int64)[order]
        names, codes = np.unique(np.asarray(genes, dtype=object).astype(str), return_inverse=True)
        self.codes = codes.astype(np.int32)[order]
        self.names = names.tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def gene(self, variation_id: int) -> str | None:
        """The variant's gene ('' when ClinVar gives none), or None if the variant was not kept."""
        position = int(np.searchsorted(self.ids, variation_id))
        if position < len(self.ids) and self.ids[position] == variation_id:
            return self.names[self.codes[position]]
        return None

    def contains(self, variation_id: int) -> bool:
        return self.gene(variation_id) is not None


_STATE: dict[str, Any] = {}


def _init_worker(state: Mapping[str, Any]) -> None:
    _STATE.clear()
    _STATE.update(state)


def _variant_chunk(payload: tuple) -> dict[str, Any]:
    index, header, lines = payload
    records, stats = parse_variant_lines(header, lines, _STATE.get("genes"),
                                         path=_STATE.get("variant_path", ""))
    join = _STATE.get("join") or {}
    for record in records:
        record["source_version"] = _STATE["version"]
        record["protein_variant_id"] = join.get(record["variation_id"])
    _write_part(_STATE["out"] / TABLE_FILES["variants"], "variants", index, records)
    return {"rows": len(records), "stats": stats,
            "ids": np.array([r["variation_id"] for r in records], dtype=np.int64),
            "genes": [r["gene"] for r in records]}


def _submission_chunk(payload: tuple) -> dict[str, Any]:
    index, header, lines, row_offset = payload
    variants: VariantIndex = _STATE["variant_index"]
    submissions, stats = parse_submission_lines(header, lines)
    documents = []
    for row, submission in enumerate(submissions):
        gene = variants.gene(submission["variation_id"])
        if gene is None:
            stats["documents_for_variants_not_kept"] += 1
            continue
        documents.append(_document_record(submission, gene, row_offset + row))
    units = [unit for document in documents for unit in units_for_document(document)]
    out = _STATE["out"]
    _write_part(out / TABLE_FILES["documents"], "documents", index, documents)
    _write_part(out / TABLE_FILES["evidence_units"], "evidence_units", index, units)
    return {"documents": len(documents), "units": len(units), "stats": stats}


def _deduplicate(directory: Path, table: str, key: str) -> int:
    """Drop repeated keys (first kept) across parts. Returns how many were dropped.

    Only reads the key column unless there is something to drop — the common
    case costs one column scan.
    """
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = pd.concat([pq.read_table(p, columns=[key]).to_pandas()[key] for p in parts],
                     ignore_index=True)
    duplicates = int(keys.duplicated().sum())
    if not duplicates:
        return 0
    seen: set = set()
    for part in parts:
        # Filtered in Arrow, not via pandas: a round trip turns nulls into NaN
        # and the table's explicit schema then refuses them.
        data = pq.read_table(part, schema=arrow_schema(table))
        keep = []
        for value in data.column(key).to_pylist():
            keep.append(value not in seen)
            seen.add(value)
        if not all(keep):
            pq.write_table(data.filter(pa.array(keep, type=pa.bool_())), str(part))
    return duplicates


def build_records(out_dir: Path | str, variant_summary: Path | str,
                  submission_summary: Path | str | None = None,
                  var_citations: Path | str | None = None,
                  erepo: Path | str | None = None, erepo_columns: Mapping[str, str] | None = None,
                  dl_canonical: Path | str | None = None, genes: Iterable[str] | None = None,
                  limit_documents: int | None = None, workers: int | None = None,
                  variant_chunk_lines: int = VARIANT_CHUNK_LINES,
                  submission_chunk_lines: int = SUBMISSION_CHUNK_LINES) -> dict[str, Any]:
    """Build every table, sharded across `workers` processes (default: every core).

    The main process only decompresses and cuts the input into chunks of raw
    lines; each worker parses its chunk, extracts evidence units, and writes
    its own part file (``<table>.parquet/part-NNNNN.parquet``, named by chunk
    index, so output order is input order). Nothing funnels through one
    process, which is what kept an earlier version of this function on one
    core of twenty. The result does not depend on the number of workers.
    """
    started = time.time()
    workers = default_workers() if workers is None else max(1, int(workers))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    wanted = set(genes) if genes else None
    join = DLJoin.from_canonical(dl_canonical) if dl_canonical else DLJoin.empty()
    inputs = {"variant_summary": _file_identity(variant_summary),
              "submission_summary": _file_identity(submission_summary),
              "var_citations": _file_identity(var_citations),
              "erepo": _file_identity(erepo), "dl_canonical": _file_identity(dl_canonical)}
    for name in TABLES:
        _reset_table(out, name)
    timings: dict[str, float] = {}
    logger.info("building records with %d worker processes", workers)

    # 1. variants: parsed and written in parallel ---------------------------------------
    phase = time.time()
    ids: list[np.ndarray] = []
    genes_of: list[str] = []
    counts: Counter = Counter()

    def collect_variants(result: Mapping[str, Any]) -> None:
        counts["variants"] += result["rows"]
        stats.update(result["stats"])
        ids.append(result["ids"])
        genes_of.extend(result["genes"])

    state = {"out": out, "genes": wanted, "join": dict(join.as_dict()), "version":
             inputs["variant_summary"]["sha256"], "variant_path": str(variant_summary)}
    chunks = iter_line_chunks(variant_summary, "#AlleleID", VARIANT_SUMMARY_COLUMNS,
                              variant_chunk_lines, key_column="VariationID")
    run_windowed(_variant_chunk, ((i, header, lines) for i, (header, lines) in enumerate(chunks)),
                 workers, initializer=_init_worker, initargs=(state,),
                 label="variant_summary chunks", on_result=collect_variants)
    all_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    duplicates = _deduplicate(out / TABLE_FILES["variants"], "variants", "variation_id")
    if duplicates:
        stats["variant_summary_non_adjacent_duplicate"] += duplicates
        counts["variants"] -= duplicates
        first = ~pd.Series(all_ids).duplicated().to_numpy()
        all_ids, genes_of = all_ids[first], [g for g, keep in zip(genes_of, first) if keep]
    variant_index = VariantIndex(all_ids, genes_of)
    timings["variants_s"] = round(time.time() - phase, 1)
    logger.info("variants: %d in %.0f s", counts["variants"], timings["variants_s"])

    # 2. documents + evidence units: parsed, extracted and written in parallel ----------
    phase = time.time()
    n_chunks = 0
    if submission_summary is not None:
        def payloads() -> Iterator[tuple]:
            nonlocal n_chunks
            offset = 0
            for index, (header, lines) in enumerate(
                    iter_line_chunks(submission_summary, "#VariationID", SUBMISSION_COLUMNS,
                                     submission_chunk_lines)):
                if limit_documents is not None:
                    remaining = limit_documents - offset
                    if remaining <= 0:
                        break
                    lines = lines[:remaining]
                n_chunks = index + 1
                yield index, header, lines, offset
                offset += len(lines)

        def collect_documents(result: Mapping[str, Any]) -> None:
            counts["documents"] += result["documents"]
            counts["evidence_units"] += result["units"]
            stats.update(result["stats"])

        run_windowed(_submission_chunk, payloads(), workers, initializer=_init_worker,
                     initargs=({"out": out, "variant_index": variant_index},),
                     label="submission_summary chunks", on_result=collect_documents)

    acmg_rows: list[dict] = []
    if erepo is not None:
        from vpdl.slm.erepo import iter_erepo
        documents = []
        for record in iter_erepo(erepo, erepo_columns, stats=stats):
            if not variant_index.contains(record["variation_id"]):
                stats["erepo_for_variants_not_kept"] += 1
                continue
            document = _erepo_document(record)
            documents.append(document)
            acmg_rows.append({"document_id": document["document_id"],
                              "variant_id": document["variant_id"], "gene": document["gene"],
                              "codes_met": record["codes_met"], "codes_not_met": record["codes_not_met"],
                              "source_id": "clingen_erepo", "provenance": "expert_panel_structured"})
        units = [unit for document in documents for unit in units_for_document(document)]
        counts["documents"] += _write_part(out / TABLE_FILES["documents"], "documents", n_chunks, documents)
        counts["evidence_units"] += _write_part(out / TABLE_FILES["evidence_units"], "evidence_units",
                                                n_chunks, units)
    counts["acmg_labels"] = _write_part(out / TABLE_FILES["acmg_labels"], "acmg_labels", 0, acmg_rows)
    duplicate_documents = _deduplicate(out / TABLE_FILES["documents"], "documents", "document_id")
    if duplicate_documents:
        stats["duplicate_document_id"] += duplicate_documents
        counts["documents"] -= duplicate_documents
        counts["evidence_units"] -= _deduplicate(out / TABLE_FILES["evidence_units"], "evidence_units",
                                                 "evidence_id")
    timings["documents_s"] = round(time.time() - phase, 1)
    logger.info("documents: %d, evidence units: %d in %.0f s", counts["documents"],
                counts["evidence_units"], timings["documents_s"])

    # 3. citations (small: one pass in this process) ------------------------------------
    citations = []
    if var_citations is not None:
        citations = [c for c in iter_citations(var_citations, stats)
                     if variant_index.contains(c["variation_id"])]
    counts["citations"] = _write_part(out / TABLE_FILES["citations"], "citations", 0, citations)
    for name in TABLES:                     # every table readable, even when empty
        directory = out / TABLE_FILES[name]
        if not any(directory.glob("part-*.parquet")):
            _write_part(directory, name, 0, [])

    from vpdl.provenance import _git_state
    manifest = {
        "schema_version": SLM_SCHEMA_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "evidence_provenance": RULE_PROVENANCE, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 1), "timings": timings, "workers": workers,
        "inputs": inputs, "genes_filter": sorted(wanted) if wanted else None,
        "limit_documents": limit_documents,
        "dl_join": {"source": join.source, "variation_ids_mapped": len(join)},
        "counts": dict(counts), "reader_stats": dict(stats), "git": _git_state(),
        "files": {name: TABLE_FILES[name] for name in TABLES},
        "layout": "each table is a directory of part-NNNNN.parquet files in input order",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _read_table(path: Path, columns: list[str] | None) -> pd.DataFrame:
    if path.is_dir():
        import pyarrow as pa
        import pyarrow.parquet as pq
        parts = sorted(path.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"{path} has no part files; rebuild with `vpdl-slm build-records`.")
        return pa.concat_tables([pq.read_table(p, columns=columns) for p in parts]).to_pandas()
    return pd.read_parquet(path, columns=columns)


def load_tables(directory: Path | str, names: Iterable[str] = tuple(TABLES),
                columns: Mapping[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    tables = {}
    for name in names:
        path = directory / TABLE_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `vpdl-slm build-records` first.")
        tables[name] = _read_table(path, (columns or {}).get(name))
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
