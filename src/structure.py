"""AlphaFold DB structural-confidence features (PROJECT_PLAN.md Phase 3 step 1).

The plan's primary recipe calls for "generate wild-type (WT) ... structures
(AlphaFold for WT ...)" as part of the structure-informed feature pipeline.
Running a full structural pipeline (FoldX mutant modelling, DSSP solvent
accessibility) is out of scope for a dependency-light repo, but the AlphaFold
DB REST API (https://alphafold.ebi.ac.uk/api) freely serves a precomputed
model + per-residue confidence (pLDDT) for every reviewed human UniProt
accession with no auth and no huge bulk download -- exactly the kind of
"real, live, verifiable" source this project's other integrations use.

Per-residue pLDDT is a genuinely informative, cheap-to-extract structural
proxy: low-confidence (pLDDT < 50) residues are typically intrinsically
disordered regions, which tolerate substitutions very differently than a
well-packed structured core (pLDDT >= 90) -- exactly the kind of signal
AlphaMissense and MVmamba's own structural branch exploit implicitly.

Only ``biopython`` (already a project dependency) is needed to parse the PDB
file; no DSSP binary, no FoldX, no GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .data_loader import make_session

logger = logging.getLogger(__name__)

ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"

#: AlphaFold's own official pLDDT confidence bins
#: (https://alphafold.ebi.ac.uk/faq -- "How confident should I be in a prediction?").
PLDDT_BINS = (
    (0.0, 50.0, "very_low"),
    (50.0, 70.0, "low"),
    (70.0, 90.0, "confident"),
    (90.0, 100.01, "very_high"),
)


def plddt_bin(value: float) -> str:
    for lo, hi, name in PLDDT_BINS:
        if lo <= value < hi:
            return name
    return "unknown"


def fetch_alphafold_metadata(uniprot_id: str, raw_dir: Path,
                             session: Optional[requests.Session] = None,
                             overwrite: bool = False, timeout: int = 60) -> Optional[dict]:
    """Cached AlphaFold DB prediction-metadata entry for one UniProt accession.

    Returns ``None`` (rather than raising) when AlphaFold DB has no model for
    this accession -- some proteins are outside AlphaFold's coverage; callers
    should treat that as "no structural feature available", not an error.
    """
    cache_dir = raw_dir / "alphafold"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{uniprot_id}_metadata.json"
    if cache.exists() and not overwrite:
        import json
        payload = json.loads(cache.read_text())
        return payload if payload else None

    sess = session or make_session()
    resp = sess.get(f"{ALPHAFOLD_API_BASE}/{uniprot_id}", timeout=timeout)
    if resp.status_code in (400, 404):
        # 404 = no model for this (well-formed) accession; 400 = the API's
        # response to a malformed/unrecognised id. Either way there is no
        # structural feature to extract -- treat both as "unavailable", not
        # a hard failure that should abort a whole panel build.
        cache.write_text("null")
        logger.info("AlphaFold DB: no model for %s (HTTP %d).", uniprot_id, resp.status_code)
        return None
    resp.raise_for_status()
    payload = resp.json()
    entry = payload[0] if isinstance(payload, list) and payload else None
    import json
    cache.write_text(json.dumps(entry) if entry else "null")
    return entry


def fetch_alphafold_pdb(uniprot_id: str, pdb_url: str, raw_dir: Path,
                        session: Optional[requests.Session] = None,
                        overwrite: bool = False, timeout: int = 120) -> Path:
    """Download (cached) the AlphaFold PDB model file for one accession."""
    cache_dir = raw_dir / "alphafold"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{uniprot_id}.pdb"
    if dest.exists() and not overwrite:
        return dest
    sess = session or make_session()
    resp = sess.get(pdb_url, timeout=timeout)
    resp.raise_for_status()
    dest.write_text(resp.text)
    return dest


def parse_plddt_per_residue(pdb_path: Path) -> Dict[int, float]:
    """1-based residue position -> pLDDT (AlphaFold stores it in the B-factor column)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af_model", str(pdb_path))
    plddt: Dict[int, float] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                pos = int(residue.id[1])
                plddt[pos] = float(residue["CA"].get_bfactor())
        break  # AlphaFold single-model PDB; only need the first MODEL record.
    return plddt


def load_alphafold_plddt(uniprot_id: str, raw_dir: Path,
                         session: Optional[requests.Session] = None,
                         overwrite: bool = False) -> Dict[int, float]:
    """Cached per-residue pLDDT dict for one accession (``{}`` if unavailable)."""
    meta = fetch_alphafold_metadata(uniprot_id, raw_dir, session=session, overwrite=overwrite)
    if meta is None or not meta.get("pdbUrl"):
        return {}
    pdb_path = fetch_alphafold_pdb(uniprot_id, meta["pdbUrl"], raw_dir,
                                   session=session, overwrite=overwrite)
    try:
        return parse_plddt_per_residue(pdb_path)
    except Exception as exc:  # noqa: BLE001 - a malformed/partial PDB must not abort the panel
        logger.warning("AlphaFold: failed to parse pLDDT for %s (%s).", uniprot_id, exc)
        return {}


def load_panel_alphafold_features(
    panel: pd.DataFrame, raw_dir: Path,
    session: Optional[requests.Session] = None, overwrite: bool = False,
) -> pd.DataFrame:
    """Per-residue AlphaFold pLDDT for every accession in *panel* (``[gene, uniprot_id, sequence]``).

    Returns a tidy long frame ``[uniprot_id, position, af_plddt, af_plddt_bin,
    af_disordered]`` ready to left-join onto a variant table by
    ``(uniprot_id, position)``.
    """
    sess = session or make_session()
    rows = []
    for uniprot_id in panel["uniprot_id"].unique():
        plddt = load_alphafold_plddt(uniprot_id, raw_dir, session=sess, overwrite=overwrite)
        for pos, val in plddt.items():
            rows.append({"uniprot_id": uniprot_id, "position": pos, "af_plddt": val})
        if plddt:
            logger.info("AlphaFold: %s -> %d residues with pLDDT.", uniprot_id, len(plddt))
    if not rows:
        return pd.DataFrame(columns=["uniprot_id", "position", "af_plddt",
                                     "af_plddt_bin", "af_disordered"])
    out = pd.DataFrame(rows)
    out["af_plddt_bin"] = out["af_plddt"].apply(plddt_bin)
    out["af_disordered"] = (out["af_plddt"] < 50.0).astype(int)
    return out


def attach_alphafold_features(master: pd.DataFrame, af_panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-residue AlphaFold features onto a variant table."""
    key = ["uniprot_id", "position"]
    sub = af_panel[key + ["af_plddt", "af_plddt_bin", "af_disordered"]].drop_duplicates(subset=key)
    return master.merge(sub, on=key, how="left")


ALPHAFOLD_FEATURE_COLUMNS = ["af_plddt", "af_disordered"]

__all__ = [
    "ALPHAFOLD_API_BASE", "ALPHAFOLD_FEATURE_COLUMNS", "PLDDT_BINS",
    "plddt_bin", "fetch_alphafold_metadata", "fetch_alphafold_pdb",
    "parse_plddt_per_residue", "load_alphafold_plddt",
    "load_panel_alphafold_features", "attach_alphafold_features",
]
