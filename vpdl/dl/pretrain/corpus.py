"""The unlabelled MMR protein corpus for continued pretraining.

Content: the four panel proteins and mismatch-repair homologs across species —
UniProt entries in the MutS (Pfam PF00488) and MutL (PF01119) families and
those annotated with the DNA-mismatch-repair keyword (KW-0234). Justification:
continued masked-residue pretraining helps when the corpus is close to the
target distribution; MMR homologs are the closest there is, and a few thousand
sequences is the scale at which LoRA-based domain adaptation is cheap.

The leakage rule is the design decision that matters. Pretraining uses no
labels, but under leave-one-gene-out it still adapts the model to the held-out
protein's sequence family. Two modes, recorded in the corpus manifest:

    transductive   all panel proteins and all homologs. Label-free, and no more
                   exposure than ESM's own UniRef pretraining already gave.
    strict         per held-out gene: remove the gene itself and every corpus
                   sequence >= `max_identity` identical to it (its orthologs).
                   One corpus — and one pretrained model — per fold.

`strict` is the default for reported P1-P4 results; `transductive` is kept
for the cost-bounded comparison and must be labelled as such.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from vpdl.dl.homology import align, kmer_containment

logger = logging.getLogger(__name__)

__all__ = ["UNIPROT_STREAM", "CORPUS_QUERIES", "read_fasta", "write_fasta",
           "fetch_uniprot_fasta", "CorpusReport", "build_corpus"]

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
CORPUS_QUERIES = {
    "mutS_family": "(xref:pfam-PF00488) AND (reviewed:true)",
    "mutL_family": "(xref:pfam-PF01119) AND (reviewed:true)",
    "mismatch_repair_keyword": "(keyword:KW-0234) AND (reviewed:true)",
}


def read_fasta(path: Path | str) -> dict[str, str]:
    records: dict[str, str] = {}
    name, parts = None, []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.upper())
    if name is not None:
        records[name] = "".join(parts)
    return records


def write_fasta(records: Mapping[str, str], path: Path | str, width: int = 60) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for name, sequence in records.items():
            handle.write(f">{name}\n")
            for i in range(0, len(sequence), width):
                handle.write(sequence[i:i + width] + "\n")


def fetch_uniprot_fasta(query: str, out_path: Path | str, timeout: int = 300) -> Path:
    """Stream one UniProt query as FASTA (a few MB for these families)."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    url = f"{UNIPROT_STREAM}?{urllib.parse.urlencode({'query': query, 'format': 'fasta'})}"
    logger.info("fetching %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.read())
    return out_path


@dataclass
class CorpusReport:
    mode: str
    holdout: str | None
    n_input: int
    n_duplicates: int
    n_nonstandard: int
    n_length: int
    n_excluded_homologs: int
    n_train: int
    n_val: int
    excluded: list[dict]
    sha256: dict[str, str]


def _nonstandard_fraction(sequence: str) -> float:
    return sum(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sequence) / max(1, len(sequence))


def build_corpus(fasta_paths: Sequence[Path | str], panel: Mapping[str, str],
                 out_dir: Path | str, mode: str = "strict", holdout: str | None = None,
                 max_identity: float = 0.5, prefilter: float = 0.05, min_length: int = 50,
                 max_nonstandard: float = 0.05, val_fraction: float = 0.05,
                 seed: int = 0) -> CorpusReport:
    """Deduplicate, filter, apply the leakage rule, split train/val, write.

    `panel` maps accession -> sequence for the four panel proteins; `holdout`
    is the held-out accession in ``strict`` mode.
    """
    if mode not in ("strict", "transductive"):
        raise ValueError("mode must be strict or transductive")
    if mode == "strict" and holdout not in panel:
        raise ValueError("strict mode needs the held-out panel accession")
    records: dict[str, str] = {}
    for path in fasta_paths:
        records.update(read_fasta(path))
    n_input = len(records)
    for accession, sequence in panel.items():
        records.setdefault(f"panel|{accession}", sequence)

    seen: dict[str, str] = {}
    duplicates = nonstandard = short = 0
    for name, sequence in records.items():
        if sequence in seen:
            duplicates += 1
            continue
        if _nonstandard_fraction(sequence) > max_nonstandard:
            nonstandard += 1
            continue
        if len(sequence) < min_length:
            short += 1
            continue
        seen[sequence] = name

    excluded: list[dict] = []
    kept: dict[str, str] = {}
    target = panel.get(holdout) if holdout else None
    for sequence, name in seen.items():
        if target is not None:
            if sequence == target:
                excluded.append({"name": name, "identity": 1.0})
                continue
            if kmer_containment(target, sequence, k=3) >= prefilter:
                identity = align(target, sequence).identity_shorter
                if identity >= max_identity:
                    excluded.append({"name": name, "identity": round(identity, 3)})
                    continue
        kept[name] = sequence

    train, val = {}, {}
    for name, sequence in kept.items():
        digest = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()
        (val if int.from_bytes(digest[:4], "big") / 2 ** 32 < val_fraction else train)[name] = \
            sequence
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(train, out_dir / "train.fasta")
    write_fasta(val, out_dir / "val.fasta")
    sha = {split: hashlib.sha256((out_dir / f"{split}.fasta").read_bytes()).hexdigest()
           for split in ("train", "val")}
    report = CorpusReport(mode, holdout, n_input, duplicates, nonstandard, short,
                          len(excluded), len(train), len(val), excluded, sha)
    (out_dir / "corpus_manifest.json").write_text(json.dumps(
        report.__dict__ | {"max_identity": max_identity, "prefilter": prefilter,
                           "queries": CORPUS_QUERIES, "sources": [str(p) for p in fasta_paths],
                           "seed": seed}, indent=2))
    logger.info("corpus %s (holdout=%s): %d train / %d val, %d homologs excluded",
                mode, holdout, len(train), len(val), len(excluded))
    return report
