"""InterPro domain/family/superfamily annotations (complementary to UniProt).

UniProt's own ``Domain``/``Region`` feature annotations
(:func:`src.external_datasets.fetch_uniprot_domains`) are curated but sparse
for some proteins. InterPro integrates Pfam, PROSITE, SMART, CATH-Gene3D,
SUPERFAMILY and PRINTS into one consensus call per residue range, giving
broader structural/functional coverage for the same accession at essentially
no extra acquisition cost (one REST call per protein, same caching pattern as
every other source here).

API: https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{accession}/
(no auth, no rate-limit tier for this volume).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import requests

from .data_loader import make_session

logger = logging.getLogger(__name__)

INTERPRO_API_BASE = "https://www.ebi.ac.uk/interpro/api"

#: InterPro entry types worth keeping as structural/functional priors.
#: ("unintegrated" member-database-only signatures are noisier; excluded.)
KEPT_ENTRY_TYPES = ("domain", "family", "homologous_superfamily",
                    "conserved_site", "repeat", "active_site", "binding_site")


def fetch_interpro_entries(uniprot_id: str, raw_dir: Path,
                           session: Optional[requests.Session] = None,
                           overwrite: bool = False, timeout: int = 60,
                           page_size: int = 100) -> List[dict]:
    """Cached raw InterPro entry-match list for one UniProt accession."""
    cache_dir = raw_dir / "interpro"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{uniprot_id}.json"
    if cache.exists() and not overwrite:
        return json.loads(cache.read_text())

    sess = session or make_session()
    url = (f"{INTERPRO_API_BASE}/entry/all/protein/uniprot/{uniprot_id}/"
          f"?page_size={page_size}")
    results: List[dict] = []
    while url:
        resp = sess.get(url, timeout=timeout)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        payload = resp.json()
        results.extend(payload.get("results", []))
        url = payload.get("next")
    cache.write_text(json.dumps(results))
    return results


def parse_interpro_intervals(entries: List[dict], uniprot_id: str) -> pd.DataFrame:
    """Flatten raw entry-match records into ``[uniprot_id, accession, name,
    entry_type, member_databases, start, end]`` (1-based inclusive)."""
    rows = []
    for entry in entries:
        meta = entry.get("metadata", {})
        entry_type = meta.get("type")
        if entry_type not in KEPT_ENTRY_TYPES:
            continue
        member_dbs = "|".join(sorted((meta.get("member_databases") or {}).keys()))
        for prot in entry.get("proteins", []):
            for loc in prot.get("entry_protein_locations") or []:
                for frag in loc.get("fragments") or []:
                    start, end = frag.get("start"), frag.get("end")
                    if start is None or end is None:
                        continue
                    rows.append({
                        "uniprot_id": uniprot_id,
                        "accession": meta.get("accession"),
                        "name": meta.get("name"),
                        "entry_type": entry_type,
                        "member_databases": member_dbs,
                        "start": int(start), "end": int(end),
                    })
    return pd.DataFrame(rows, columns=["uniprot_id", "accession", "name",
                                       "entry_type", "member_databases",
                                       "start", "end"])


def load_panel_interpro_intervals(
    panel: pd.DataFrame, raw_dir: Path,
    session: Optional[requests.Session] = None, overwrite: bool = False,
) -> pd.DataFrame:
    """InterPro interval table for every accession in *panel* (``[uniprot_id]``)."""
    sess = session or make_session()
    frames = []
    for uniprot_id in panel["uniprot_id"].unique():
        entries = fetch_interpro_entries(uniprot_id, raw_dir, session=sess, overwrite=overwrite)
        df = parse_interpro_intervals(entries, uniprot_id)
        if len(df):
            frames.append(df)
        logger.info("InterPro %s: %d entry intervals kept.", uniprot_id, len(df))
    if not frames:
        return pd.DataFrame(columns=["uniprot_id", "accession", "name",
                                     "entry_type", "member_databases", "start", "end"])
    return pd.concat(frames, ignore_index=True)


def attach_interpro_features(master: pd.DataFrame, interpro_intervals: pd.DataFrame) -> pd.DataFrame:
    """Add ``in_interpro_domain`` (0/1) and ``interpro_names`` per variant position.

    Same interval-join pattern as ``in_domain``/``domain_names`` in
    :mod:`src.extended_builder` -- kept as a separate, independently-sourced
    column rather than merged into ``in_domain`` so a discrepancy between the
    two curations is visible rather than silently averaged away.
    """
    if not len(interpro_intervals) or not len(master):
        out = master.copy()
        out["in_interpro_domain"] = 0
        out["interpro_names"] = ""
        return out
    positions = master[["uniprot_id", "position"]].drop_duplicates()
    pairs = positions.merge(interpro_intervals, on="uniprot_id", how="inner")
    inside = (pairs["start"] <= pairs["position"]) & (pairs["position"] <= pairs["end"])
    names_by_pos = (
        pairs.loc[inside]
        .groupby(["uniprot_id", "position"])["name"]
        .agg(lambda s: "|".join(sorted({str(n) for n in s if pd.notna(n) and str(n)})))
    )
    joined = master[["uniprot_id", "position"]].merge(
        names_by_pos.rename("interpro_names").reset_index(),
        on=["uniprot_id", "position"], how="left")
    out = master.copy()
    out["in_interpro_domain"] = joined["interpro_names"].notna().astype(int).to_numpy()
    out["interpro_names"] = joined["interpro_names"].fillna("").to_numpy()
    return out


INTERPRO_FEATURE_COLUMNS = ["in_interpro_domain"]

__all__ = [
    "INTERPRO_API_BASE", "INTERPRO_FEATURE_COLUMNS", "KEPT_ENTRY_TYPES",
    "fetch_interpro_entries", "parse_interpro_intervals",
    "load_panel_interpro_intervals", "attach_interpro_features",
]
