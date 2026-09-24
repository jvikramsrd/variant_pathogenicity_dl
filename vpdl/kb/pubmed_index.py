"""PubMed abstracts in the knowledge base: an on-disk exact-word index, merged at question time.

The GeneReviews index (``vpdl.kb.search.KnowledgeIndex``) holds ~44,000
passages in memory with an embedding each. PubMed's baseline holds ~36
million abstracts: in memory that would need hundreds of GB, and embeddings
for all of them ~150 GB of vectors and days of embedding. So PubMed gets its
own index on disk — SQLite full-text search (FTS5, BM25-ranked) — and a
question searches both, merging the two lists by reciprocal-rank fusion, as the
GeneReviews index already merges its exact-word and meaning searches.

Words are indexed with the knowledge base's own tokenizer (``vpdl.kb.search.tokenize``),
so "c.199G>A", "MLH1" and "rs63750217" are matched exactly as they are in
GeneReviews. Abstracts are read with ``vpdl.slm.pubmed`` — English only, and
retractions, retraction notices and expressions of concern are dropped.

What a reader sees is the abstract word for word, with its PMID, a link to
PubMed and a credit line: the copyright of an abstract stays with its
publisher, and NLM's terms ask that the source be acknowledged. The index is
separate from ``chunks.jsonl``, so the SLM pretraining corpus (which reads
``chunks.jsonl`` and PubMed on its own) does not see PubMed twice.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from vpdl.kb.chunks import Chunk
from vpdl.kb.search import CANDIDATES, reciprocal_rank_fusion, tokenize

logger = logging.getLogger(__name__)

__all__ = ["build_pubmed_index", "PubMedIndex", "CombinedIndex", "GENETICS_FILTER"]

# Optional narrowing for a smaller index: titles/abstracts that talk about genetics at all.
GENETICS_FILTER = re.compile(
    r"\b(?:gene|genes|genetic|genetics|genomic|genome|germline|variant|variants|mutation|mutations|"
    r"allele|alleles|hereditary|inherited|exome|sequencing|carrier|syndrome|pathogenic|"
    r"polymorphism|heterozygous|homozygous|mismatch repair|lynch)\b", re.I)
_TOKENCHARS = "._:>+-*"            # the characters vpdl.kb.search.tokenize keeps inside a token


def _parse_file(job: tuple[str, str | None]) -> tuple[str, list[tuple], Counter, str | None]:
    """One baseline file -> rows (pmid, year, title, abstract, terms). Runs in a worker."""
    from vpdl.slm.pubmed import iter_abstracts
    path, pattern = job
    matcher = re.compile(pattern, re.I) if pattern else None
    stats: Counter = Counter()
    rows = []
    try:
        for record in iter_abstracts(path, stats):
            if not record.pmid.isdigit():
                stats["kb_bad_pmid"] += 1
                continue
            if matcher is not None and not matcher.search(record.text):
                stats["kb_filtered_out"] += 1
                continue
            title, _, abstract = record.text.partition("\n")
            if not abstract:
                title, abstract = "", title
            rows.append((int(record.pmid), record.year, title, abstract,
                         " ".join(tokenize(record.text))))
        return path, rows, stats, None
    except Exception as error:                          # noqa: BLE001 — named, counted, skipped
        return path, [], stats, f"{type(error).__name__}: {error}"


def build_pubmed_index(pubmed_dir: Path | str, out: Path | str, workers: int | None = None,
                       limit_files: int | None = None, genetics_only: bool = False) -> dict:
    """Parse every ``pubmed*.xml.gz`` in parallel and write ``out`` (SQLite, built beside and
    renamed into place, so a half-built index never replaces a working one)."""
    files = sorted(Path(pubmed_dir).glob("pubmed*.xml*"))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"no pubmed*.xml.gz files in {pubmed_dir}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_suffix(".building")
    staging.unlink(missing_ok=True)
    started = time.time()
    db = sqlite3.connect(staging)
    try:
        db.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        db.execute("DROP TABLE probe")
    except sqlite3.OperationalError as error:
        db.close()
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"this Python's SQLite has no FTS5 ({error}); the PubMed index needs it")
    db.executescript(f"""
        PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;
        CREATE TABLE abstracts (pmid INTEGER PRIMARY KEY, year TEXT, title TEXT, abstract TEXT);
        CREATE VIRTUAL TABLE terms USING fts5(t, content='',
            tokenize="unicode61 remove_diacritics 0 tokenchars '{_TOKENCHARS}'");
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    stats: Counter = Counter()
    unreadable: list[str] = []
    seen: set[int] = set()
    workers = max(1, min(workers or os.cpu_count() or 1, len(files)))
    pattern = GENETICS_FILTER.pattern if genetics_only else None
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    try:
        jobs = [(str(f), pattern) for f in files]
        results = pool.imap(_parse_file, jobs) if pool else map(_parse_file, jobs)
        for index, (path, rows, file_stats, error) in enumerate(results, start=1):
            stats.update(file_stats)
            if error:
                stats["unreadable_files"] += 1
                unreadable.append(Path(path).name)
                logger.warning("%s could not be read (%s); skipped.", Path(path).name, error)
            fresh = [r for r in rows if r[0] not in seen]
            stats["duplicate_pmid"] += len(rows) - len(fresh)
            seen.update(r[0] for r in fresh)
            with db:
                db.executemany("INSERT INTO abstracts VALUES (?, ?, ?, ?)", [r[:4] for r in fresh])
                db.executemany("INSERT INTO terms(rowid, t) VALUES (?, ?)", [(r[0], r[4]) for r in fresh])
            stats["indexed"] += len(fresh)
            if index % 25 == 0 or index == len(files):
                logger.info("PubMed index: %d/%d files, %d abstracts (%.0f min)", index, len(files),
                            stats["indexed"], (time.time() - started) / 60)
    finally:
        if pool is not None:
            pool.terminate()
    summary = {"files": len(files), "abstracts": stats["indexed"], "genetics_only": genetics_only,
               "decisions": dict(stats), "unreadable_files": unreadable,
               "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "minutes": round((time.time() - started) / 60, 1)}
    with db:
        db.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in summary.items()])
    db.execute("INSERT INTO terms(terms) VALUES ('optimize')")
    db.commit()
    db.close()
    os.replace(staging, out)
    return summary


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


class PubMedIndex:
    """Read side: BM25 search over the terms table, abstracts returned as knowledge-base passages."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path}: no PubMed index; build it with `vpdl kb-build --pubmed`")
        self.db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True,
                                  check_same_thread=False)
        self.meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def search_ids(self, query: str, k: int = CANDIDATES) -> list[int]:
        """PMIDs, best first. All query words first (precise, fast); if that finds fewer than
        `k`, any of them (broader) fills the rest."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        found: list[int] = []
        for joiner in (" AND ", " OR "):
            if len(found) >= k or (joiner == " OR " and len(terms) == 1):
                break
            rows = self.db.execute("SELECT rowid FROM terms WHERE terms MATCH ? ORDER BY rank LIMIT ?",
                                   (joiner.join(_quote(t) for t in terms), k)).fetchall()
            found += [r[0] for r in rows if r[0] not in found]
        return found[:k]

    def passages(self, pmids: Sequence[int]) -> list[Chunk]:
        if not pmids:
            return []
        marks = ",".join("?" * len(pmids))
        rows = {r[0]: r for r in self.db.execute(
            f"SELECT pmid, year, title, abstract FROM abstracts WHERE pmid IN ({marks})", list(pmids))}
        return [self._chunk(*rows[p]) for p in pmids if p in rows]

    @staticmethod
    def _chunk(pmid: int, year: str, title: str, abstract: str) -> Chunk:
        return Chunk(chunk_id=f"pubmed:{pmid}", source="PubMed", doc_id=str(pmid),
                     doc_title=title or f"PMID {pmid}", section=f"Abstract ({year})" if year else "Abstract",
                     section_id=f"pubmed:{pmid}", kind="text", text=abstract,
                     url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                     attribution=f"PubMed abstract, PMID {pmid}. Courtesy of the U.S. National Library "
                                 "of Medicine; copyright of the abstract remains with its publisher.",
                     licence="NLM PubMed terms; publisher copyright; research use")

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        return self.passages(self.search_ids(query, k))


class CombinedIndex:
    """GeneReviews (exact-word + meaning) and PubMed (exact-word), merged by reciprocal-rank fusion.

    Looks like a ``KnowledgeIndex`` to ``vpdl.kb.answer.ask``: same ``search``
    signature, and the embedding attributes of the GeneReviews index it wraps.
    """

    def __init__(self, knowledge, pubmed: PubMedIndex):
        self.knowledge = knowledge
        self.pubmed = pubmed
        self.embeddings = knowledge.embeddings
        self.embed_model = knowledge.embed_model
        self.chunks = knowledge.chunks

    def search(self, query: str, k: int = 6, embed=None) -> list[Chunk]:
        curated = self.knowledge.search(query, k=CANDIDATES, embed=embed)
        literature = self.pubmed.search(query, k=CANDIDATES)
        by_id = {c.chunk_id: c for c in [*curated, *literature]}
        order = reciprocal_rank_fusion([[c.chunk_id for c in curated], [c.chunk_id for c in literature]])
        return [by_id[i] for i in order[:k]]
