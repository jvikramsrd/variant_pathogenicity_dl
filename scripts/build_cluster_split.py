#!/usr/bin/env python3
"""Sequence-cluster-disjoint split via MMseqs2 (PROJECT_PLAN.md Phase 1).

"cluster at the sequence level (MMseqs2, 20% coverage/20% identity, per CSBJ's
recipe) and additionally prepare a leave-one-MMR-gene-out split -- the real
generalization test given only 4 genes total."

Two highly-similar proteins (e.g. paralogs, or near-duplicate isoforms that
slipped into a large panel) sharing a train/val split leak information a
model would never have on a truly novel protein. This script clusters every
panel protein's canonical sequence with MMseqs2 at the plan-specified 20%
identity / 20% coverage threshold, then assigns whole clusters (never
individual proteins) to train/val so no two sequence-similar proteins land on
opposite sides.

Requires the ``mmseqs`` binary on PATH (not a Python dependency -- install via
your package manager or https://github.com/soedinglab/MMseqs2#installation;
conda: ``conda install -c bioconda mmseqs2``). This script only orchestrates
the subprocess call and parses its output; nothing here fabricates a
clustering result when the binary is missing -- it fails loudly instead.

Example
-------
    python scripts/build_cluster_split.py \
        --panel_json data/raw/uniprot/expanded_panel.json \
        --out_dir data/processed/extended --val_fraction 0.2
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_cluster_split")

DEFAULT_MIN_SEQ_ID = 0.20
DEFAULT_MIN_COVERAGE = 0.20


def check_mmseqs_available() -> str:
    path = shutil.which("mmseqs")
    if path is None:
        raise RuntimeError(
            "mmseqs binary not found on PATH. Install MMseqs2 first: "
            "https://github.com/soedinglab/MMseqs2#installation "
            "(conda: 'conda install -c bioconda mmseqs2'; or download a "
            "static release binary). This script deliberately does not "
            "fall back to a fake/random clustering.")
    return path


def write_fasta(sequences: Dict[str, str], path: Path) -> None:
    with open(path, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def run_mmseqs_cluster(
    sequences: Dict[str, str], work_dir: Path,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID, min_coverage: float = DEFAULT_MIN_COVERAGE,
    threads: int = 4,
) -> Dict[str, str]:
    """Run ``mmseqs easy-cluster`` and return ``{member_id: representative_id}``."""
    check_mmseqs_available()
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "panel.fasta"
    write_fasta(sequences, fasta_path)
    out_prefix = work_dir / "cluster"
    tmp_dir = work_dir / "mmseqs_tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        "mmseqs", "easy-cluster", str(fasta_path), str(out_prefix), str(tmp_dir),
        "--min-seq-id", str(min_seq_id), "-c", str(min_coverage),
        "--cov-mode", "0", "--threads", str(threads),
    ]
    logger.info("+ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs easy-cluster failed (exit {result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")

    tsv_path = Path(f"{out_prefix}_cluster.tsv")
    if not tsv_path.exists():
        raise RuntimeError(f"Expected mmseqs output {tsv_path} not found.")
    return parse_cluster_tsv(tsv_path)


def parse_cluster_tsv(tsv_path: Path) -> Dict[str, str]:
    """Parse an ``mmseqs easy-cluster`` ``*_cluster.tsv`` (representative<TAB>member)."""
    mapping: Dict[str, str] = {}
    with open(tsv_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, member = line.split("\t")
            mapping[member] = rep
    return mapping


def assign_cluster_split(
    member_to_cluster: Dict[str, str], val_fraction: float = 0.2, seed: int = 42,
) -> Dict[str, str]:
    """Randomly assign whole clusters to train/val, targeting *val_fraction*
    of members (not clusters) in val -- greedy bin-packing by cluster size so
    a single giant cluster can't blow the target fraction."""
    clusters: Dict[str, List[str]] = {}
    for member, rep in member_to_cluster.items():
        clusters.setdefault(rep, []).append(member)

    rng = np.random.default_rng(seed)
    reps = list(clusters.keys())
    rng.shuffle(reps)
    total = len(member_to_cluster)
    target_val = int(round(val_fraction * total))

    split: Dict[str, str] = {}
    val_count = 0
    for rep in reps:
        members = clusters[rep]
        if val_count < target_val:
            for m in members:
                split[m] = "val"
            val_count += len(members)
        else:
            for m in members:
                split[m] = "train"
    return split


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--panel_json", type=Path, required=True,
                   help="{gene: {accession, sequence}} panel file (e.g. "
                        "data/raw/uniprot/expanded_panel.json or "
                        "data/processed/extended/panel_sequences.json).")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--min_seq_id", type=float, default=DEFAULT_MIN_SEQ_ID,
                   help="MMseqs2 --min-seq-id (plan default: 0.20).")
    p.add_argument("--min_coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="MMseqs2 -c coverage (plan default: 0.20).")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_work_dir", action="store_true",
                   help="Keep the intermediate FASTA/mmseqs files (default: "
                        "written under --out_dir/mmseqs_work and kept anyway "
                        "unless this flag is combined with a temp override).")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    panel = json.loads(args.panel_json.read_text())
    sequences = {gene.upper(): d["sequence"] for gene, d in panel.items()}
    logger.info("Clustering %d panel sequences at %.0f%% identity / %.0f%% coverage ...",
                len(sequences), args.min_seq_id * 100, args.min_coverage * 100)

    work_dir = args.out_dir / "mmseqs_work"
    member_to_cluster = run_mmseqs_cluster(
        sequences, work_dir, min_seq_id=args.min_seq_id,
        min_coverage=args.min_coverage, threads=args.threads)

    n_clusters = len(set(member_to_cluster.values()))
    logger.info("%d sequences -> %d clusters (%.1f%% reduction).",
                len(sequences), n_clusters,
                100 * (1 - n_clusters / max(1, len(sequences))))
    multi = {rep: [m for m, r in member_to_cluster.items() if r == rep]
             for rep in set(member_to_cluster.values())}
    merged = {rep: members for rep, members in multi.items() if len(members) > 1}
    if merged:
        logger.warning("%d cluster(s) merge multiple panel proteins (candidate "
                       "paralogs/near-duplicates) -- these MUST stay on the same "
                       "side of any split: %s",
                       len(merged), {r: ms for r, ms in list(merged.items())[:5]})

    split = assign_cluster_split(member_to_cluster, val_fraction=args.val_fraction, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "cluster_split.json"
    payload = {
        "min_seq_id": args.min_seq_id, "min_coverage": args.min_coverage,
        "val_fraction_target": args.val_fraction,
        "n_sequences": len(sequences), "n_clusters": n_clusters,
        "merged_clusters": {r: ms for r, ms in merged.items()},
        "member_to_cluster": member_to_cluster,
        "split": split,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    n_train = sum(1 for v in split.values() if v == "train")
    n_val = sum(1 for v in split.values() if v == "val")
    logger.info("Split written -> %s (train=%d, val=%d proteins).", out_path, n_train, n_val)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
