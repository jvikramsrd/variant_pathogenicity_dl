"""The broad genomic continued-pretraining corpus — with the evaluation held out of it.

Built from the sources the project already has readers for: PubMed abstracts
(``vpdl.slm.pubmed``, retractions dropped) and the knowledge-base passages
(GeneReviews and whatever else ``vpdl kb-build`` indexed), reusing
``vpdl.slm.corpus.iter_documents`` unchanged; the wider free text of
``vpdl.slm.textsources`` (PMC full text, MedlinePlus Genetics, Orphanet, MONDO,
UniProt) when named; plus — only if asked — ClinVar narratives of TRAINING
variants with their conclusions masked.

What makes this corpus different from the one ``vpdl slm-corpus`` builds is
what it leaves out. A pretraining corpus is built once and reused by every
experiment, so anything an evaluation will later be scored on must not be in
it (``vpdl.slm.build.roles.pretraining_exclusions``):

    * narratives of reserved variants (MMR test, temporal test, functional
      holdout, the unseen-gene panel) — never;
    * in strict mode, PubMed abstracts whose PMID ClinVar cites for one of
      those variants — the paper that describes the test variant.

Both are counted in ``stats.json``: how many documents were dropped and why.
The output format is the one ``vpdl.slm.corpus`` writes, so the tokenizer,
packer and training loop read it unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from vpdl.slm.corpus import in_validation, iter_documents
from vpdl.slm.text.conclusion import mask_conclusions

logger = logging.getLogger(__name__)

__all__ = ["build_pretrain_corpus"]


def _narratives(records_dir: Path | str, training_documents: Iterable[str] | None,
                stats: Counter) -> Iterable[dict]:
    from vpdl.slm.build.records import load_tables
    tables = load_tables(records_dir, ["documents"])
    documents = tables["documents"]
    allowed = set(training_documents) if training_documents is not None else None
    for row in documents.itertuples(index=False):
        if not (row.text or "").strip():
            continue
        if row.restricted:
            stats["narratives_restricted"] += 1
            continue
        if allowed is not None and row.document_id not in allowed:
            stats["narratives_not_training_role"] += 1
            continue
        masked = mask_conclusions(row.text)
        if not masked.text.strip():
            stats["narratives_empty_after_masking"] += 1
            continue
        stats["narratives_kept"] += 1
        stats["conclusion_sentences_masked"] += masked.n_masked
        yield {"id": f"scv:{row.document_id}", "source": "clinvar_narrative", "text": masked.text}


def build_pretrain_corpus(out_dir: Path | str, kb_dir: Path | str | None = None,
                          pubmed_dir: Path | str | None = None,
                          records_dir: Path | str | None = None,
                          exclusions: Mapping[str, Iterable[str]] | None = None,
                          include_narratives: bool = False,
                          training_documents: Iterable[str] | None = None,
                          limit_files: int | None = None, workers: int | None = None,
                          shard_documents: int = 500_000,
                          text_sources: Mapping[str, Path | str | None] | None = None) -> dict[str, Any]:
    """`text_sources` adds the readers of vpdl.slm.textsources — ``{"medlineplus": file,
    "orphanet": file, "mondo": file, "uniprot_text": file, "pmc": directory}`` — under the
    same exclusions: a PMC article whose PMID is cited for an evaluation variant is dropped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()
    started = time.time()
    stats: Counter = Counter()
    excluded_pmids = {str(p) for p in (exclusions or {}).get("pmids", ())}
    excluded_documents = {str(d) for d in (exclusions or {}).get("documents", ())}
    per_source: dict[str, Counter] = {}
    unreadable: list[str] = []

    shard, in_shard = 0, 0
    train = (out / f"train-{shard:05d}.jsonl").open("w", encoding="utf-8")
    val = (out / "val.jsonl").open("w", encoding="utf-8")

    def emit(document: dict) -> None:
        nonlocal shard, in_shard, train
        split = "val" if in_validation(document["id"]) else "train"
        counts = per_source.setdefault(document["source"], Counter())
        counts[f"{split}_documents"] += 1
        counts[f"{split}_characters"] += len(document["text"])
        line = json.dumps(document, ensure_ascii=False) + "\n"
        if split == "val":
            val.write(line)
            return
        if in_shard >= shard_documents:
            train.close()
            shard, in_shard = shard + 1, 0
            train = (out / f"train-{shard:05d}.jsonl").open("w", encoding="utf-8")
        train.write(line)
        in_shard += 1

    try:
        if pubmed_dir is not None or kb_dir is not None:
            for document in iter_documents(pubmed_dir, kb_dir, stats, limit_files, unreadable, workers):
                identifier = str(document["id"])
                if identifier.startswith("pmid:") and identifier.split(":", 1)[1] in excluded_pmids:
                    stats["pubmed_excluded_cited_by_evaluation_variants"] += 1
                    continue
                if identifier in excluded_documents:
                    stats["documents_excluded_by_id"] += 1
                    continue
                emit(document)
        if text_sources:
            from vpdl.slm.textsources import iter_text_sources
            for document in iter_text_sources(text_sources, stats):
                if document.get("pmid") and str(document["pmid"]) in excluded_pmids:
                    stats[f"{document['source']}_excluded_cited_by_evaluation_variants"] += 1
                    continue
                if str(document["id"]) in excluded_documents:
                    stats["documents_excluded_by_id"] += 1
                    continue
                emit(document)
        if include_narratives:
            if records_dir is None:
                raise ValueError("include_narratives needs records_dir")
            for document in _narratives(records_dir, training_documents, stats):
                if document["id"].removeprefix("scv:") in excluded_documents:
                    stats["narratives_excluded_reserved"] += 1
                    continue
                emit(document)
    finally:
        train.close()
        val.close()

    summary = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - started, 1),
        "pubmed_dir": str(pubmed_dir) if pubmed_dir else None,
        "kb_dir": str(kb_dir) if kb_dir else None,
        "records_dir": str(records_dir) if records_dir else None,
        "text_sources": {k: str(v) for k, v in (text_sources or {}).items() if v},
        "include_narratives": include_narratives,
        "exclusions": {"pmids": len(excluded_pmids), "documents": len(excluded_documents),
                       "strict_literature": bool(excluded_pmids)},
        "decisions": dict(stats), "unreadable_files": unreadable,
        "sources": {name: dict(counts) for name, counts in per_source.items()},
        "train_shards": shard + 1,
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    return summary
