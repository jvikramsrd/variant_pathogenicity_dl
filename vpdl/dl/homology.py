"""Sequence homology for leakage control: alignment, families, homologous twins.

Why this exists. Leave-one-gene-out on this panel is *not* a sequence-cluster
split: MLH1 and PMS2 are MutL homologs, MSH2 and MSH6 are MutS homologs.
Holding out PMS2 while training on MLH1 leaves aligned, homologous residues —
sometimes carrying the very same substitution — on the training side. That is
the "sequence similarity" leak the leakage report measures and the family split
removes.

Nothing here shells out. A vectorised Gotoh (affine-gap) global aligner is
enough for a four-protein panel and for screening a pretraining corpus; MMseqs2
remains an option on the DGX for very large corpora, but is not required.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "BLOSUM62",
    "PROTEIN_FAMILIES",
    "family_of",
    "Alignment",
    "align",
    "kmer_containment",
    "pairwise_identity",
    "cluster_by_identity",
    "homology_groups",
    "homologous_twins",
    "sequence_sha",
]

_AA = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_MATRIX = np.array([[int(v) for v in row.split()]
                    for row in _BLOSUM62_ROWS.strip().splitlines()], dtype=np.int32)
# Index 20 = any non-standard residue (X, B, Z, U, O): scored -1 against all.
_SCORES = np.full((21, 21), -1, dtype=np.int32)
_SCORES[:20, :20] = _MATRIX
_INDEX = {aa: i for i, aa in enumerate(_AA)}

BLOSUM62: dict[tuple[str, str], int] = {
    (a, b): int(_MATRIX[i, j]) for i, a in enumerate(_AA) for j, b in enumerate(_AA)
}

# Established MMR biology, not a clustering output: MutL homologs dimerise
# (MLH1-PMS2, MutLalpha), MutS homologs dimerise (MSH2-MSH6, MutSalpha). The
# leakage report checks this against computed identities rather than trusting
# it; `cluster_by_identity` is the data-driven alternative.
PROTEIN_FAMILIES: dict[str, str] = {
    "P40692": "MutL",   # MLH1
    "P54278": "MutL",   # PMS2
    "P43246": "MutS",   # MSH2
    "P52701": "MutS",   # MSH6
}

_NEG = np.int32(-(10 ** 8))


def family_of(uniprot_id: str, families: Mapping[str, str] = PROTEIN_FAMILIES) -> str:
    """Family label; an unknown protein is its own family (never merged by guess)."""
    return families.get(uniprot_id, uniprot_id)


def sequence_sha(sequence: str) -> str:
    """Short, stable identity for a sequence version (feature-store key)."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:12]


def _encode(sequence: str) -> np.ndarray:
    return np.array([_INDEX.get(c, 20) for c in sequence.upper()], dtype=np.int64)


@dataclass
class Alignment:
    """Global alignment of `a` (rows) against `b` (columns)."""

    score: int
    a_to_b: np.ndarray          # len(a); aligned b index (0-based) or -1 for a gap
    matches: int
    aligned_pairs: int
    length_a: int
    length_b: int
    extra: dict = field(default_factory=dict)

    @property
    def identity_aligned(self) -> float:
        """Identical / aligned (non-gap) columns. Inflated for short overlaps."""
        return self.matches / self.aligned_pairs if self.aligned_pairs else 0.0

    @property
    def identity_shorter(self) -> float:
        """Identical / length of the shorter sequence — the conservative one."""
        shorter = min(self.length_a, self.length_b)
        return self.matches / shorter if shorter else 0.0

    @property
    def coverage_shorter(self) -> float:
        shorter = min(self.length_a, self.length_b)
        return self.aligned_pairs / shorter if shorter else 0.0

    def map_position(self, position_a: int) -> int | None:
        """1-based position in `a` -> 1-based aligned position in `b`, or None."""
        if not 1 <= position_a <= self.length_a:
            return None
        target = int(self.a_to_b[position_a - 1])
        return None if target < 0 else target + 1


def align(a: str, b: str, gap_open: int = 11, gap_extend: int = 1,
          free_end_gaps: bool = True) -> Alignment:
    """Gotoh global alignment with BLOSUM62, vectorised along each row.

    A gap of length k costs ``gap_open + (k - 1) * gap_extend`` (BLAST's
    defaults, 11/1). ``free_end_gaps`` does not penalise terminal overhangs, so
    a domain-length difference between paralogs does not dominate the score.

    The horizontal-gap recurrence is a max-plus scan: with open >= extend, the
    best horizontal gap ending at j starts from a non-gap cell k, so
    ``E[j] = max_{k<j}(G[k] + ext*k) - open - ext*(j-1)`` — one exclusive
    running maximum per row instead of an inner loop.
    """
    if gap_open < gap_extend:
        raise ValueError("gap_open must be >= gap_extend for the row scan to be exact")
    sa, sb = _encode(a), _encode(b)
    n, m = len(sa), len(sb)
    if n == 0 or m == 0:
        return Alignment(0, np.full(n, -1), 0, 0, n, m)

    cols = np.arange(m + 1, dtype=np.int64)
    # Traceback state: which move produced H, whether G was diag or vertical,
    # where a horizontal gap started, and whether a vertical gap opened here.
    hsrc = np.zeros((n + 1, m + 1), dtype=np.int8)       # 0 diag, 1 vert, 2 horiz
    gsrc = np.zeros((n + 1, m + 1), dtype=np.int8)       # 0 diag, 1 vert
    estart = np.zeros((n + 1, m + 1), dtype=np.int32)
    fopen = np.zeros((n + 1, m + 1), dtype=bool)

    if free_end_gaps:
        h_prev = np.zeros(m + 1, dtype=np.int64)
    else:
        h_prev = np.where(cols == 0, 0, -(gap_open + (cols - 1) * gap_extend))
        hsrc[0, 1:] = 2
        estart[0, 1:] = 0
    f_prev = np.full(m + 1, _NEG, dtype=np.int64)
    last_row = h_prev.copy()
    last_col = np.zeros(n + 1, dtype=np.int64)
    last_col[0] = h_prev[m]

    for i in range(1, n + 1):
        f_open = h_prev - gap_open
        f_ext = f_prev - gap_extend
        f_cur = np.maximum(f_open, f_ext)
        fopen[i] = f_open >= f_ext

        m_cur = np.full(m + 1, _NEG, dtype=np.int64)
        m_cur[1:] = h_prev[:-1] + _SCORES[sa[i - 1], sb]

        g = np.maximum(m_cur, f_cur)
        gsrc[i] = (f_cur > m_cur).astype(np.int8)
        g[0] = 0 if free_end_gaps else -(gap_open + (i - 1) * gap_extend)

        v = g + gap_extend * cols
        running = np.maximum.accumulate(v)
        arg = np.maximum.accumulate(np.where(v >= running, cols, 0))
        exclusive = np.empty_like(running)
        exclusive[0], exclusive[1:] = _NEG, running[:-1]
        e_start = np.zeros_like(arg)
        e_start[1:] = arg[:-1]
        e_cur = exclusive - gap_open - gap_extend * (cols - 1)
        e_cur[0] = _NEG

        h_cur = np.maximum(g, e_cur)
        src = np.where(e_cur > g, 2, gsrc[i]).astype(np.int8)
        src[0] = 1 if not free_end_gaps else 0
        hsrc[i] = src
        estart[i] = e_start
        h_prev, f_prev = h_cur, f_cur
        last_col[i] = h_cur[m]
    last_row = h_prev

    if free_end_gaps:
        best_row, best_col = int(np.argmax(last_row)), int(np.argmax(last_col))
        if last_row[best_row] >= last_col[best_col]:
            i, j, score = n, best_row, int(last_row[best_row])
        else:
            i, j, score = best_col, m, int(last_col[best_col])
    else:
        i, j, score = n, m, int(last_row[m])

    a_to_b = np.full(n, -1, dtype=np.int64)
    matches = pairs = 0
    state = int(hsrc[i, j])
    while i > 0 and j > 0:
        if state == 0:
            a_to_b[i - 1] = j - 1
            pairs += 1
            matches += int(sa[i - 1] == sb[j - 1] and sa[i - 1] != 20)
            i, j = i - 1, j - 1
            state = int(hsrc[i, j])
        elif state == 1:
            opened = bool(fopen[i, j])
            i -= 1
            state = int(hsrc[i, j]) if opened else 1
        else:
            j = int(estart[i, j])
            state = int(gsrc[i, j])
    return Alignment(score=score, a_to_b=a_to_b, matches=matches,
                     aligned_pairs=pairs, length_a=n, length_b=m,
                     extra={"gap_open": gap_open, "gap_extend": gap_extend,
                            "free_end_gaps": free_end_gaps})


def kmer_containment(query: str, target: str, k: int = 3) -> float:
    """Fraction of `query`'s distinct k-mers present in `target` (prefilter)."""
    if len(query) < k:
        return 0.0
    q = {query[i:i + k] for i in range(len(query) - k + 1)}
    t = {target[i:i + k] for i in range(len(target) - k + 1)}
    return len(q & t) / len(q)


def pairwise_identity(sequences: Mapping[str, str]) -> dict[tuple[str, str], Alignment]:
    """Align every unordered pair once. Keys are sorted ``(id_a, id_b)``."""
    ids = sorted(sequences)
    return {(x, y): align(sequences[x], sequences[y])
            for index, x in enumerate(ids) for y in ids[index + 1:]}


def cluster_by_identity(alignments: Mapping[tuple[str, str], Alignment],
                        ids: Iterable[str], threshold: float = 0.25,
                        metric: str = "identity_shorter") -> dict[str, str]:
    """Single-linkage clusters at `threshold`; returns ``{id: representative}``."""
    parent = {i: i for i in ids}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (x, y), alignment in alignments.items():
        if getattr(alignment, metric) >= threshold:
            rx, ry = root(x), root(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    return {i: root(i) for i in parent}


def homology_groups(sequences: Mapping[str, str],
                    families: Mapping[str, str] = PROTEIN_FAMILIES,
                    alignments: Mapping[tuple[str, str], Alignment] | None = None,
                    ) -> dict[tuple[str, int], str]:
    """``{(uniprot_id, position): cluster_id}`` — aligned residues share an id.

    Within a family, residues aligned to each other get one id
    (``MutL:P40692:123``, anchored on the lexicographically first member);
    unaligned residues and proteins outside any family keep their own
    ``uniprot:position``. Splitting on these ids keeps homologous positions on
    one side of every partition.
    """
    groups: dict[tuple[str, int], str] = {
        (acc, pos): f"{acc}:{pos}"
        for acc, seq in sequences.items() for pos in range(1, len(seq) + 1)
    }
    by_family: dict[str, list[str]] = {}
    for acc in sequences:
        by_family.setdefault(family_of(acc, families), []).append(acc)

    for family, members in by_family.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            key = (anchor, other)
            alignment = (alignments or {}).get(key) or align(sequences[anchor],
                                                             sequences[other])
            for pos in range(1, len(sequences[anchor]) + 1):
                mapped = alignment.map_position(pos)
                label = f"{family}:{anchor}:{pos}"
                groups[(anchor, pos)] = label
                if mapped is not None:
                    groups[(other, mapped)] = label
    return groups


def homologous_twins(
    test_keys: Sequence[tuple[str, int, str, str]],
    train_keys: Sequence[tuple[str, int, str, str]],
    alignments: Mapping[tuple[str, str], Alignment],
) -> dict[str, int]:
    """Count test variants with a homologous counterpart in training.

    ``position`` twins: a training variant sits at the aligned residue of a
    paralog. ``exact`` twins: same aligned residue, same wild-type and mutant
    residue — the strongest form of cross-gene leakage this panel allows.
    """
    by_position: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for acc, pos, wt, mut in train_keys:
        by_position.setdefault((acc, int(pos)), set()).add((wt, mut))

    # Directed 0-based maps source -> target, built once per alignment.
    maps: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (x, y), alignment in alignments.items():
        forward = alignment.a_to_b
        backward = np.full(alignment.length_b, -1, dtype=np.int64)
        aligned = forward >= 0
        backward[forward[aligned]] = np.flatnonzero(aligned)
        maps.setdefault(x, []).append((y, forward))
        maps.setdefault(y, []).append((x, backward))

    position_twins = exact_twins = 0
    for acc, pos, wt, mut in test_keys:
        hit_pos = hit_exact = False
        for other, mapping in maps.get(acc, ()):
            index = int(pos) - 1
            if not 0 <= index < len(mapping) or mapping[index] < 0:
                continue
            substitutions = by_position.get((other, int(mapping[index]) + 1))
            if substitutions:
                hit_pos = True
                hit_exact = hit_exact or (wt, mut) in substitutions
        position_twins += int(hit_pos)
        exact_twins += int(hit_exact)
    return {"test_variants": len(test_keys), "position_twins": position_twins,
            "exact_twins": exact_twins}
