"""Assemble the pretraining text: PubMed abstracts + GeneReviews passages.

Output (``data/slm/corpus/``): ``train-00000.jsonl`` ... shards, ``val.jsonl``,
and ``stats.json`` recording every count — documents and characters per
source, and every reason a PubMed record was dropped.

The validation split is chosen by a hash of each document's id, not at
random: the same document lands in the same split on every run and every
machine, so a validation loss measured today is comparable with one measured
after the corpus is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterator

from vpdl.slm.pubmed import iter_abstracts

logger = logging.getLogger(__name__)

__all__ = ["in_validation", "iter_documents", "build_corpus", "iter_texts"]

VALIDATION_PER_MILLE = 10          # 1%


def in_validation(doc_id: str, per_mille: int = VALIDATION_PER_MILLE) -> bool:
    digest = hashlib.sha1(doc_id.encode()).hexdigest()
    return int(digest[:8], 16) % 1000 < per_mille


def _read_file(path: str) -> tuple[str, list[tuple[str, str]], Counter, str | None]:
    """One PubMed file -> its kept abstracts. Runs in a worker process."""
    stats: Counter = Counter()
    try:
        return path, [(a.pmid, a.text) for a in iter_abstracts(path, stats)], stats, None
    except (EOFError, OSError, ET.ParseError) as error:
        return path, [], stats, f"{type(error).__name__}: {error}"


def iter_documents(pubmed_dir: Path | str | None, kb_dir: Path | str | None,
                   stats: Counter, limit_files: int | None = None,
                   unreadable: list[str] | None = None,
                   workers: int | None = None) -> Iterator[dict]:
    """PubMed files are parsed in parallel (``workers`` processes, default all
    cores) but consumed in file order, so the corpus is identical to a
    single-process build — same documents, same order, same duplicates dropped."""
    unreadable = unreadable if unreadable is not None else []
    if pubmed_dir is not None:
        files = sorted(Path(pubmed_dir).glob("pubmed*.xml*"))
        if limit_files:
            files = files[:limit_files]
        if not files:
            raise FileNotFoundError(f"No pubmed*.xml.gz files in {pubmed_dir}.")
        workers = max(1, min(workers or os.cpu_count() or 1, len(files)))
        seen: set[str] = set()
        pool = multiprocessing.Pool(workers) if workers > 1 else None
        try:
            results = (pool.imap(_read_file, map(str, files)) if pool
                       else map(_read_file, map(str, files)))
            for index, (path, abstracts, file_stats, error) in enumerate(results, start=1):
                stats.update(file_stats)
                if error:
                    # A truncated download must not cost an hour-long build: the
                    # whole file is skipped and named in stats.json; re-download it
                    # (wget -c) and rebuild.
                    stats["unreadable_files"] += 1
                    unreadable.append(Path(path).name)
                    logger.warning("%s could not be read (%s); skipped.", Path(path).name, error)
                for pmid, text in abstracts:
                    if pmid in seen:
                        stats["duplicate_pmid"] += 1
                        continue
                    seen.add(pmid)
                    yield {"id": f"pmid:{pmid}", "source": "pubmed", "text": text}
                if index % 50 == 0 or index == len(files):
                    logger.info("PubMed: %d/%d files, %d abstracts kept (%d processes)",
                                index, len(files), stats["kept"], workers)
        finally:
            if pool is not None:
                pool.terminate()

    if kb_dir is not None:
        path = Path(kb_dir) / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `vpdl kb-build` first.")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                chunk = json.loads(line)
                if chunk.get("private"):
                    continue                 # owned textbooks never go into weights
                yield {"id": chunk["chunk_id"], "source": chunk["source"].lower(),
                       "text": f"{chunk['doc_title']} > {chunk['section']}\n{chunk['text']}"}


def build_corpus(out_dir: Path | str, pubmed_dir: Path | str | None = None,
                 kb_dir: Path | str | None = None, shard_docs: int = 500_000,
                 limit_files: int | None = None, workers: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()                         # a rebuild never mixes with an old corpus
    started = time.time()
    stats: Counter = Counter()
    unreadable: list[str] = []
    per_source: dict[str, Counter] = {}
    shard, in_shard = 0, 0
    train = (out / f"train-{shard:05d}.jsonl").open("w", encoding="utf-8")
    val = (out / "val.jsonl").open("w", encoding="utf-8")
    try:
        for doc in iter_documents(pubmed_dir, kb_dir, stats, limit_files, unreadable, workers):
            split = "val" if in_validation(doc["id"]) else "train"
            counts = per_source.setdefault(doc["source"], Counter())
            counts[f"{split}_documents"] += 1
            counts[f"{split}_characters"] += len(doc["text"])
            line = json.dumps(doc, ensure_ascii=False) + "\n"
            if split == "val":
                val.write(line)
                continue
            if in_shard >= shard_docs:
                train.close()
                shard, in_shard = shard + 1, 0
                train = (out / f"train-{shard:05d}.jsonl").open("w", encoding="utf-8")
            train.write(line)
            in_shard += 1
    finally:
        train.close()
        val.close()

    summary = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pubmed_dir": str(pubmed_dir) if pubmed_dir else None,
        "pubmed_files_limit": limit_files,
        "kb_dir": str(kb_dir) if kb_dir else None,
        "validation_per_mille": VALIDATION_PER_MILLE,
        "pubmed_decisions": dict(stats),
        "unreadable_files": unreadable,
        "sources": {name: dict(counts) for name, counts in per_source.items()},
        "train_shards": shard + 1,
        "seconds": round(time.time() - started, 1),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    return summary


def iter_texts(corpus_dir: Path | str, split: str = "train",
               every: int = 1) -> Iterator[str]:
    """Texts of one split; ``every=k`` yields every k-th document (an even sample)."""
    corpus = Path(corpus_dir)
    files = sorted(corpus.glob("train-*.jsonl")) if split == "train" else [corpus / "val.jsonl"]
    index = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if index % every == 0:
                    yield json.loads(line)["text"]
                index += 1
