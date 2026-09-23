"""Duplicates and laboratory templates: exact hashes, MinHash, Jaccard, clusters.

Clinical labs write from templates. Two narratives from one lab about two
different variants can be 90% identical words; if one is in training and the
other in test, a model can score well by recognising the template's ending.
Three levels are computed, each a column the split builder and the leakage
audit use:

    exact_group      sha256 of the whitespace/case-normalised text
    near_dup_cluster documents whose word-shingle Jaccard >= `near_threshold`
                     (default 0.8) — the same narrative with small edits
    template_cluster documents whose TEMPLATE Jaccard >= `template_threshold`
                     (default 0.6), where the template is the text with the
                     variant-specific parts replaced by placeholders:
                     HGVS -> <HGVS>, numbers -> <NUM>, amino acids -> <AA>,
                     gene symbols -> <GENE>, PMIDs/rsIDs -> <ID>

Clustering is MinHash + banded LSH to find candidates, then the exact Jaccard
of the shingle sets decides; candidates are linked by union-find. Large
buckets (thousands of copies of one template) are linked through their first
member instead of all pairs, so the cost stays linear in the bucket.

MinHash uses the universal family h(x) = (a*x + b) mod (2^31 - 1) on 32-bit
shingle hashes, with a, b < 2^31 so every product fits in uint64 without
overflow — the values are exact, not wrapped.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["normalize_text", "template_text", "shingles", "jaccard", "MinHasher",
           "UnionFind", "cluster_documents", "ClusterResult", "exact_group"]

_MERSENNE = np.uint64((1 << 31) - 1)
_HGVS = re.compile(r"\b(?:[cgnmrp]\.(?:\(?[A-Za-z*]{0,3}[-*]?\d+[^\s,;)]*\)?)|[NX][MRCGP]_\d+(?:\.\d+)?)")
_ID = re.compile(r"\b(?:rs\d+|PMID:?\s*\d+|\d{6,9})\b", re.I)
_AMINO = re.compile(r"\b(?:alanine|arginine|asparagine|aspartic\s+acid|aspartate|cysteine|glutamine|"
                    r"glutamic\s+acid|glutamate|glycine|histidine|isoleucine|leucine|lysine|methionine|"
                    r"phenylalanine|proline|serine|threonine|tryptophan|tyrosine|valine|"
                    r"Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_GENE_LIKE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?\b")
_WORD = re.compile(r"[a-z0-9<>_]+")


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def exact_group(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def template_text(text: str, genes: Iterable[str] = ()) -> str:
    """The narrative with its variant-specific parts replaced by placeholders.

    Placeholders are lower-case (``_hgvs_``, ``_gene_``...) so later
    substitutions cannot match inside them.
    """
    out = _HGVS.sub(" _hgvs_ ", text or "")
    out = _ID.sub(" _id_ ", out)
    for gene in sorted({g for g in genes if g}, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(gene)}\b", " _gene_ ", out)
    # Remaining all-caps tokens are mostly gene symbols of other genes named in passing.
    out = _GENE_LIKE.sub(" _gene_ ", out)
    out = _AMINO.sub(" _aa_ ", out)
    out = _NUMBER.sub(" _num_ ", out)
    return normalize_text(out)


def shingles(text: str, k: int = 5) -> np.ndarray:
    """32-bit hashes of word k-grams (a short text contributes its whole word list)."""
    words = _WORD.findall(normalize_text(text))
    if not words:
        return np.zeros(0, dtype=np.uint64)
    grams = [" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))]
    values = {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little")
              for g in grams}
    return np.fromiter(values, dtype=np.uint64, count=len(values))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    union = len(np.union1d(a, b))
    return len(np.intersect1d(a, b, assume_unique=True)) / union if union else 0.0


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)

    def signature(self, shingle_hashes: np.ndarray) -> np.ndarray:
        if len(shingle_hashes) == 0:
            return np.full(self.num_perm, int(_MERSENNE), dtype=np.uint64)
        x = (shingle_hashes % _MERSENNE)[:, None]              # < 2^31
        values = (x * self.a[None, :] + self.b[None, :]) % _MERSENNE   # < 2^62 + 2^31: no overflow
        return values.min(axis=0)

    @staticmethod
    def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:                           # path compression
            self.parent[i], i = root, self.parent[i]
        return int(root)

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


@dataclass
class ClusterResult:
    labels: np.ndarray               # cluster id per document (smallest member index)
    n_clusters: int
    largest: int
    pairs_checked: int
    pairs_linked: int

    def sizes(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))


def _shingle_and_sign(payload: tuple) -> tuple[list[np.ndarray], np.ndarray]:
    """Shingles and MinHash signatures for one chunk (runs in a worker process).

    Each worker builds its own :class:`MinHasher` from the same seed, so the
    permutations — and therefore the signatures — are identical however many
    processes are used.
    """
    chunk, k, num_perm, seed = payload
    hasher = MinHasher(num_perm, seed)
    sets = [shingles(text, k) for text in chunk]
    block = (np.stack([hasher.signature(s) for s in sets]) if sets
             else np.zeros((0, num_perm), dtype=np.uint64))
    return sets, block


def _signatures(texts: Sequence[str], k: int, num_perm: int, seed: int,
                workers: int | None) -> tuple[list[np.ndarray], np.ndarray]:
    """Shingle and sign every document, across `workers` processes (default: all cores)."""
    import multiprocessing
    import os

    workers = workers if workers is not None else (os.cpu_count() or 1)
    chunk_size = max(256, math.ceil(len(texts) / max(1, workers * 4)))
    chunks = [(list(texts[i:i + chunk_size]), k, num_perm, seed)
              for i in range(0, len(texts), chunk_size)]
    if workers <= 1 or len(chunks) <= 1:
        results = [_shingle_and_sign(chunk) for chunk in chunks]
    else:
        with multiprocessing.Pool(min(workers, len(chunks))) as pool:
            results = pool.map(_shingle_and_sign, chunks)
    sets: list[np.ndarray] = []
    blocks = []
    for chunk_sets, block in results:
        sets.extend(chunk_sets)
        blocks.append(block)
    return sets, (np.vstack(blocks) if blocks else np.zeros((0, num_perm), dtype=np.uint64))


def cluster_documents(texts: Sequence[str], threshold: float, num_perm: int = 128,
                      bands: int | None = None, k: int = 5, seed: int = 1,
                      max_bucket_pairs: int = 50, workers: int | None = 1) -> ClusterResult:
    """Union-find clusters of documents whose shingle Jaccard >= `threshold`.

    `bands` defaults to the banding whose LSH threshold (1/b)^(1/r) sits just
    below `threshold`, so true pairs are rarely missed; every candidate is then
    checked exactly. In a bucket larger than `max_bucket_pairs`, each member is
    checked against the bucket's first member only (linear, not quadratic).

    ``workers`` shingles and signs across processes — the part that dominates on
    millions of narratives. ``None`` uses every core; the result is identical
    whatever the number (each worker seeds its own permutations the same way),
    which ``tests/slm`` checks.
    """
    n = len(texts)
    finder = UnionFind(n)
    if n < 2:
        return ClusterResult(np.arange(n), n, 1 if n else 0, 0, 0)
    sets, signatures = _signatures(texts, k, num_perm, seed, workers)
    if bands is None:
        bands = _choose_bands(num_perm, threshold)
    rows = num_perm // bands
    checked = linked = 0
    seen: set[tuple[int, int]] = set()
    for band in range(bands):
        block = signatures[:, band * rows:(band + 1) * rows]
        keys = _band_keys(block)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        for bucket in np.split(order, boundaries):
            if len(bucket) < 2:
                continue
            members = [int(i) for i in bucket if len(sets[int(i)])]
            if len(members) < 2:
                continue
            if len(members) * (len(members) - 1) // 2 > max_bucket_pairs:
                pairs = [(members[0], m) for m in members[1:]]
            else:
                pairs = [(a, b) for x, a in enumerate(members) for b in members[x + 1:]]
            for a, b in pairs:
                key = (a, b) if a < b else (b, a)
                if key in seen or finder.find(a) == finder.find(b):
                    continue
                seen.add(key)
                checked += 1
                if jaccard(sets[a], sets[b]) >= threshold:
                    finder.union(a, b)
                    linked += 1
    labels = finder.labels()
    _, counts = np.unique(labels, return_counts=True)
    return ClusterResult(labels, len(counts), int(counts.max()), checked, linked)


def _band_keys(block: np.ndarray) -> np.ndarray:
    """One 64-bit key per row of a band — deterministic across processes.

    (Python's ``hash`` of bytes is salted per process; a polynomial hash with
    uint64 wrap-around is not. Unequal rows that collide only add candidates,
    which the exact Jaccard check then rejects.)
    """
    keys = np.zeros(len(block), dtype=np.uint64)
    multiplier = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        for column in range(block.shape[1]):
            keys = keys * multiplier + block[:, column] + np.uint64(column + 1)
    return keys


def _choose_bands(num_perm: int, threshold: float) -> int:
    best = 1
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        if (1.0 / bands) ** (1.0 / rows) <= threshold - 0.1:
            best = bands
            break
        best = bands
    return best
