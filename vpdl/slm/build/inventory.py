"""What data already exists, measured — before anything new is downloaded.

``inventory()`` looks in the places this repository writes to, measures what
it finds (rows, variants, genes, diseases, labels, types, dates, hashes) and
says plainly what it did not find. Nothing is downloaded; nothing is assumed.
The result is the evidence behind docs/slm/LLM_CODEBASE_AUDIT.md section 2.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from vpdl.slm.clinvar_text import iter_variant_summary_all, specific_condition
from vpdl.slm.variants import MMR_GENES

logger = logging.getLogger(__name__)

__all__ = ["measure_variant_summary", "inventory", "sha256_file"]


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _top(counter: Counter, n: int = 25) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(n)}


def measure_variant_summary(path: Path | str, limit: int | None = None) -> dict[str, Any]:
    """Counts over every variant in a ClinVar ``variant_summary`` (one pass)."""
    stats: Counter = Counter()
    by = {name: Counter() for name in ("label_category", "label5", "consequence", "variant_type",
                                       "review_status", "stars", "origin", "year", "classification_raw")}
    genes: Counter = Counter()
    diseases: Counter = Counter()
    mmr = {name: Counter() for name in ("label5", "consequence", "label_category")}
    mmr_variants = 0
    n = 0
    for record in iter_variant_summary_all(path, stats=stats):
        n += 1
        by["label_category"][record["label_category"]] += 1
        by["label5"][record["label5"] or "-"] += 1
        by["consequence"][record["consequence"]] += 1
        by["variant_type"][record["variant_type"]] += 1
        by["review_status"][record["review_status"]] += 1
        by["stars"][record["stars"]] += 1
        by["origin"][record["origin"] or "-"] += 1
        by["year"][(record["last_evaluated"] or "none")[:4]] += 1
        by["classification_raw"][record["clinvar_classification"]] += 1
        for gene in record["genes"]:
            genes[gene] += 1
        for disease, name in zip(record["disease_ids"], record["disease_names"]):
            diseases[disease or f"name:{name}"] += 1
            if not specific_condition(name):
                stats["condition_mentions_catch_all_or_umbrella"] += 1
        if record["disease_names"] and not any(specific_condition(n) for n in record["disease_names"]):
            stats["variants_without_a_specific_condition"] += 1
        if record["gene"] in MMR_GENES:
            mmr_variants += 1
            mmr["label5"][record["label5"] or "-"] += 1
            mmr["consequence"][record["consequence"]] += 1
            mmr["label_category"][record["label_category"]] += 1
        if limit and n >= limit:
            break
    return {
        "path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
        "variants": n, "reader_counts": dict(stats),
        "genes": len(genes), "diseases_distinct_ids": len(diseases),
        "top_genes": _top(genes), "top_diseases": _top(diseases),
        "genes_with_at_least": {str(k): sum(1 for v in genes.values() if v >= k)
                                for k in (1, 10, 100, 1000)},
        **{f"by_{name}": dict(sorted(counter.items(), key=lambda kv: -kv[1]))
           if name != "classification_raw" else _top(counter, 40) for name, counter in by.items()},
        "mmr": {"genes": list(MMR_GENES), "variants": mmr_variants,
                **{f"by_{k}": dict(v) for k, v in mmr.items()}},
        "limit": limit,
    }


def _jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def inventory(root: Path | str = ".", variant_summary_limit: int | None = None,
              measure_clinvar: bool = True) -> dict[str, Any]:
    root = Path(root)
    found: dict[str, Any] = {"root": str(root.resolve())}

    raw = root / "data" / "raw"
    clinvar = {}
    for name in ("variant_summary.txt.gz", "submission_summary.txt.gz", "var_citations.txt",
                 "var_citations.txt.gz"):
        for folder in (raw, raw / "clinvar"):
            path = folder / name
            if path.exists():
                clinvar[name] = {"path": str(path), "bytes": path.stat().st_size}
    found["clinvar_files"] = clinvar
    missing = [n for n in ("submission_summary.txt.gz", "var_citations.txt") if not any(
        k.startswith(n.split(".")[0]) for k in clinvar)]
    found["clinvar_missing"] = missing
    if measure_clinvar and "variant_summary.txt.gz" in clinvar:
        found["variant_summary"] = measure_variant_summary(
            clinvar["variant_summary.txt.gz"]["path"], variant_summary_limit)

    kb = root / "data" / "kb"
    found["kb"] = ({"chunks": _jsonl_count(kb / "chunks.jsonl"),
                    "index": json.loads((kb / "index.json").read_text())
                    if (kb / "index.json").exists() else None}
                   if (kb / "chunks.jsonl").exists() else "absent on this machine (built on the DGX)")
    slm = root / "data" / "slm"
    found["slm_corpus"] = (json.loads((slm / "corpus" / "stats.json").read_text())
                           if (slm / "corpus" / "stats.json").exists()
                           else "absent on this machine (built on the DGX)")
    found["slm_tokens"] = (json.loads((slm / "data_meta.json").read_text()).get("splits")
                           if (slm / "data_meta.json").exists() else "absent on this machine")

    mavedb = {}
    for path in sorted((root / "data").rglob("urn_mavedb_*_metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        scores = path.with_name(path.name.replace("_metadata.json", "_scores.csv"))
        publications = [
            {"db": p.get("dbName"), "id": p.get("identifier"), "doi": p.get("doi"),
             "year": p.get("publicationYear")}
            for p in meta.get("primaryPublicationIdentifiers") or []]
        mavedb[path.name.replace("_metadata.json", "")] = {
            "title": meta.get("title"),
            "target": ((meta.get("targetGenes") or [{}])[0] or {}).get("name"),
            "rows": sum(1 for _ in scores.open(encoding="utf-8")) - 1 if scores.exists() else None,
            "publications": publications, "path": str(path.parent)}
    found["mavedb"] = mavedb

    mmr = root / "data" / "mmr" / "processed"
    found["mmr_v1_tables"] = {p.name: sum(1 for _ in p.open(encoding="utf-8")) - 1
                              for p in sorted(mmr.glob("*_variants.csv"))} if mmr.exists() else {}
    built = root / "data" / "built"
    found["dl_canonical"] = sorted(str(p) for p in built.glob("canonical*.csv")) if built.exists() else []
    found["dl_outputs"] = sorted(str(p) for p in (root / "runs" / "dl").rglob("dl_outputs_*.jsonl")) \
        if (root / "runs" / "dl").exists() else []
    found["cimra"] = sorted(str(p) for p in (root / "data").rglob("*cimra*.csv"))
    found["pubmed_files"] = len(list((raw / "pubmed").glob("pubmed*.xml.gz"))) if (raw / "pubmed").exists() else 0
    found["checkpoints"] = {
        "slm_runs": sorted(str(p) for p in (root / "runs" / "slm").glob("*/run.json"))
        if (root / "runs" / "slm").exists() else [],
        "pt_files_in_repo": sorted(str(p) for p in root.rglob("*.pt")
                                   if ".venv" not in p.parts and "node_modules" not in p.parts)[:50],
    }
    return found
