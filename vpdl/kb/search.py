"""Two searches, merged: exact words (BM25) and meaning (embeddings).

Genetics text is full of identifiers — MLH1 vs MSH2, c.199G>A, rs63750217 — that
meaning-based search blurs together; exact-word search keeps them apart. Plain
questions ("how often should colonoscopy be done?") are the reverse. Each finds
what the other misses, and reciprocal-rank fusion merges the two lists without
having to put their scores on one scale.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from vpdl.kb.chunks import Chunk, chunks_digest, load_chunks, save_chunks

logger = logging.getLogger(__name__)

__all__ = ["tokenize", "BM25", "reciprocal_rank_fusion", "KnowledgeIndex"]

# Keeps identifiers whole: "c.199G>A", "p.Gly67Arg", "MSH6", "rs63750217", "1-2".
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._:>+\-*][A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in is it its of on or "
    "should that the their this to was what when which who why will with".split())

# GeneReviews tables abbreviate systematically ("every 1-2 yrs", "assoc w/");
# questions do not. Both sides are searched in the long form. Search only: the
# excerpt a reader sees is never altered.
_EXPANSIONS = {
    "yrs": "years", "yr": "year", "mos": "months", "wks": "weeks", "assoc": "associated",
    "mgmt": "management", "incl": "including", "esp": "especially", "dx": "diagnosis",
    "tx": "treatment", "hx": "history", "pv": "pathogenic variant", "pvs": "pathogenic variants",
    "often": "frequency",
}
# Chapter-specific abbreviations stay out: "LS" is Lynch syndrome in one
# chapter and Leigh syndrome in another.

RRF_K = 60          # the constant from Cormack et al. 2009; not tuned
CANDIDATES = 40     # from each search, before fusion

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def tokenize(text: str) -> list[str]:
    """Whole identifiers plus their parts, lower-cased.

    "c.199G>A" yields "c.199g>a" (exact match) and "199g" (partial match), so a
    query written as "199G>A" still finds it.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        word = match.group(0).lower()
        if word in _STOPWORDS:
            continue
        if word in _EXPANSIONS:
            tokens.extend(_EXPANSIONS[word].split())
            continue
        tokens.append(word)
        parts = re.split(r"[._:>+\-*]", word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) >= 2 and p not in _STOPWORDS)
    return tokens


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(len(documents))
        for index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.lengths[index] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.n = len(documents)
        self.mean_length = float(self.lengths.mean()) if self.n else 0.0

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (self.n - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.mean_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores


def _top(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    if k == 0:
        return []
    candidates = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(candidates, key=lambda i: (-scores[i], i))
    return [int(i) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] += 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda item: (-fused[item], item))


class KnowledgeIndex:
    """Passages + their embeddings, saved together and checked against each other."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None,
                 embed_model: str | None = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.bm25 = BM25([c.search_text() for c in chunks])

    # -- building ---------------------------------------------------------
    @staticmethod
    def embed_chunks(chunks: list[Chunk], embed: EmbedFn, batch: int = 64) -> np.ndarray:
        vectors = []
        started = time.time()
        for start in range(0, len(chunks), batch):
            texts = [c.search_text() for c in chunks[start:start + batch]]
            vectors.append(np.asarray(embed(texts), dtype=np.float32))
            done = start + len(texts)
            if done % (batch * 20) == 0 or done == len(chunks):
                logger.info("embedded %d/%d passages (%.0fs)", done, len(chunks),
                            time.time() - started)
        matrix = np.vstack(vectors)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, directory / "chunks.jsonl")
        meta = {"passages": len(self.chunks), "chunks_sha256": chunks_digest(self.chunks),
                "embed_model": self.embed_model,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)
            meta["dimensions"] = int(self.embeddings.shape[1])
        (directory / "index.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path | str) -> "KnowledgeIndex":
        directory = Path(directory)
        if not (directory / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No knowledge base at {directory}; run `vpdl kb-build`.")
        chunks = load_chunks(directory / "chunks.jsonl")
        meta = json.loads((directory / "index.json").read_text())
        embeddings = None
        if (directory / "embeddings.npy").exists():
            # Embeddings belong to one exact set of passages. Re-chunking without
            # re-embedding would silently pair vectors with the wrong text.
            if meta.get("chunks_sha256") != chunks_digest(chunks):
                raise ValueError(f"{directory}: embeddings do not match the passages; "
                                 "re-run `vpdl kb-build`.")
            embeddings = np.load(directory / "embeddings.npy")
        return cls(chunks, embeddings, meta.get("embed_model"))

    # -- searching --------------------------------------------------------
    def search(self, query: str, k: int = 6, embed: EmbedFn | None = None) -> list[Chunk]:
        rankings = [_top(self.bm25.scores(query), CANDIDATES)]
        if self.embeddings is not None and embed is not None:
            vector = np.asarray(embed([query]), dtype=np.float32)[0]
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            similarity = self.embeddings @ vector
            rankings.append(_top(similarity - similarity.min() + 1e-9, CANDIDATES))
        elif self.embeddings is not None:
            logger.warning("No embedding function given; exact-word search only.")
        return [self.chunks[i] for i in reciprocal_rank_fusion(rankings)[:k]]
