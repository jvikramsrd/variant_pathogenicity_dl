#!/usr/bin/env python3
"""Build the expanded gene panel covering every human ProteinGym DMS protein.

Maps every human ``UniProt_ID`` entry name in the ProteinGym DMS reference to
its reviewed primary accession (cached resolver), then batch-fetches the HGNC
primary gene symbol and canonical sequence from UniProt.  The result is
written to ``data/raw/uniprot/expanded_panel.json`` as
``{GENE: {"accession": ..., "sequence": ...}}`` for use with
``scripts/build_extended_dataset.py --panel_file``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import make_session
from src.external_datasets import (
    UNIPROT_SEARCH_URL,
    load_dms_assays,
    resolve_uniprot_entry_names,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("expanded_panel")


def main() -> int:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    out_path = raw_dir / "uniprot" / "expanded_panel.json"

    dms = load_dms_assays(raw_dir, taxon="human")
    entry_names = sorted(dms["uniprot_id"].unique())
    log.info("%d human DMS entry names", len(entry_names))

    name_map = resolve_uniprot_entry_names(entry_names, raw_dir)
    accs = {name_map[n] for n in entry_names if n in name_map}
    log.info("%d unique primary accessions", len(accs))

    sess = make_session()
    records = {}
    acc_list = sorted(accs)
    CHUNK = 25
    for i in range(0, len(acc_list), CHUNK):
        chunk = acc_list[i:i + CHUNK]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        params = {
            "query": f"({query}) AND (organism_id:9606) AND (reviewed:true)",
            "format": "json",
            "fields": "accession,id,gene_primary,sequence",
            "size": str(CHUNK),
        }
        resp = sess.get(UNIPROT_SEARCH_URL, params=params, timeout=90)
        if resp.status_code == 429:
            time.sleep(10)
            resp = sess.get(UNIPROT_SEARCH_URL, params=params, timeout=90)
        resp.raise_for_status()
        for res in resp.json().get("results", []):
            genes = res.get("genes") or []
            symbol = None
            for g in genes:
                gp = (g.get("geneName") or {}).get("value")
                if gp:
                    symbol = gp
                    break
            seq = (res.get("sequence") or {}).get("value")
            if symbol and seq:
                records[symbol.upper()] = {
                    "accession": res["primaryAccession"],
                    "sequence": seq,
                }
        log.info("resolved %d/%d accessions", len(records) // 1 or i + len(chunk),
                 len(acc_list))
        time.sleep(0.5)

    # Collisions: two entry names may share a gene symbol -> keep first.
    dupes = len(records)
    panel = {g: v for g, v in sorted(records.items())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(panel, indent=2))
    log.info("Wrote %d genes -> %s", dupes, out_path)

    # Coverage report against the DMS assays.
    mapped_accs = {v["accession"] for v in panel.values()}
    covered = dms[dms["uniprot_id"].map(name_map).isin(mapped_accs)]
    log.info("Panel covers %d/%d human DMS assays (%d mutants)",
             covered["dms_id"].nunique(), dms["dms_id"].nunique(), len(covered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
