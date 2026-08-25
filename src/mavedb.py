"""MaveDB deep-mutational-scan acquisition (PROJECT_PLAN.md Phase 2).

PROJECT_PLAN.md Phase 2 calls for the MSH2 DMS dataset (Jia et al. 2021) as a
**first-class input** for MSH2 specifically ("per D6, it outperforms every
computational predictor for MSH2 classification"), and asks to check
MaveDB/Atlas of Variant Effects for anything newer before assuming no
equivalent data exists for MLH1/MSH6/PMS2.

Verified against the live MaveDB REST API (https://api.mavedb.org/api/v1):

* **MSH2** — ``urn:mavedb:00000050-a-1``, Jia et al. 2021 (PMID 33357406),
  "Massively parallel functional testing of MSH2 missense variants conferring
  Lynch syndrome risk" — 17,746 variants, loss-of-function score in HAP1 cells
  (positive = LOF = pathogenic-consistent). This is the flagship, most-complete
  dataset and matches the plan's expectation.
* **MLH1** — ``urn:mavedb:00001218-a-1`` (2025), "Cellular abundance of MLH1"
  via DHFR-PCA in yeast — 5,056 variants. This is a *stability/abundance*
  assay, not a repair-function assay, so it is tagged with a different
  ``mave_assay_type`` and should not be treated as equivalent evidence to the
  MSH2 LOF screen without that caveat.
* **MSH6, PMS2** — no MaveDB score set exists as of this writing (searched via
  the MaveDB text-search endpoint). :func:`search_mavedb_for_gene` re-runs
  that search programmatically so a newer submission is picked up automatically
  rather than silently assumed absent.

Per Phase 2's non-circularity requirement, this data is deliberately **not**
merged into :mod:`src.transfer`'s ``TRANSFER_PRIOR_COLS`` (ESM branch
pretraining/fine-tuning features) — treat it as a held-out validation axis
only, exactly like the existing ProteinGym DMS exclusion.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

from .data_loader import THREE_TO_ONE, VALID_AA, make_session

logger = logging.getLogger(__name__)

MAVEDB_API_BASE = "https://api.mavedb.org/api/v1"

#: Verified score-set URNs for the four MMR genes (see module docstring for
#: provenance). Empty list = confirmed absent at last check, not "unchecked".
KNOWN_MMR_SCORESETS: Dict[str, List[str]] = {
    "MLH1": ["urn:mavedb:00001218-a-1"],
    "MSH2": ["urn:mavedb:00000050-a-1"],
    "MSH6": [],
    "PMS2": [],
}

#: ``p.Met1Ala`` style (MaveDB's hgvs_pro column; no enclosing parentheses).
#: The ``(?![a-z])`` guard rejects Ter/fs/del/ins/dup extension tokens whose
#: third group would otherwise false-match a 3-letter run (e.g. "Ter").
_HGVS_PRO_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?![a-zA-Z])$")


def parse_hgvs_pro(hgvs_pro: str) -> Optional[tuple[str, int, str]]:
    """Parse a MaveDB ``hgvs_pro`` token into one-letter ``(wt, pos, mut)``.

    Returns ``None`` for synonymous (``p.Met1=``), stop-gain/frameshift
    (``p.Met1Ter`` / ``...fs``), multi-variant, or missing (``NA``) tokens.
    """
    s = str(hgvs_pro or "").strip()
    m = _HGVS_PRO_RE.match(s)
    if not m:
        return None
    wt = THREE_TO_ONE.get(m.group(1), "")
    mut = THREE_TO_ONE.get(m.group(3), "")
    if not wt or not mut or wt == mut or wt not in VALID_AA or mut not in VALID_AA:
        return None
    return wt, int(m.group(2)), mut


def search_mavedb_for_gene(gene: str, session: Optional[requests.Session] = None,
                           timeout: int = 30) -> List[dict]:
    """Live text-search of MaveDB score sets mentioning *gene*.

    Use this before trusting :data:`KNOWN_MMR_SCORESETS` as exhaustive — the
    plan explicitly asks to re-check for newer submissions rather than assume
    absence. Returns the raw ``scoreSets`` result list (urn, title, numVariants,
    targetGenes, ...); callers should still eyeball the title/target before
    treating a hit as the right assay (MaveDB's text search matches on free
    text, e.g. "PMS2" also matches MLH1 score sets whose description mentions
    a PMS2 interaction partner).
    """
    sess = session or make_session()
    resp = sess.post(f"{MAVEDB_API_BASE}/score-sets/search",
                     json={"text": gene.upper()}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("scoreSets", []) if isinstance(payload, dict) else list(payload)


def _extract_uniprot_accession(metadata: dict) -> Optional[str]:
    """Pull the UniProt accession cross-reference out of score-set metadata."""
    for target in metadata.get("targetGenes", []):
        for ext in target.get("externalIdentifiers", []):
            ident = ext.get("identifier", {})
            if ident.get("dbName") == "UniProt":
                return ident.get("identifier")
    return None


def fetch_scoreset_metadata(urn: str, raw_dir: Path,
                            session: Optional[requests.Session] = None,
                            overwrite: bool = False, timeout: int = 60) -> dict:
    """Cached JSON metadata for one MaveDB score set."""
    cache_dir = raw_dir / "mavedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = urn.replace(":", "_")
    cache = cache_dir / f"{safe_name}_metadata.json"
    if cache.exists() and not overwrite:
        return json.loads(cache.read_text())
    sess = session or make_session()
    resp = sess.get(f"{MAVEDB_API_BASE}/score-sets/{urn}", timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    cache.write_text(json.dumps(payload, indent=2))
    return payload


def fetch_scoreset_scores(urn: str, raw_dir: Path,
                          session: Optional[requests.Session] = None,
                          overwrite: bool = False, timeout: int = 120) -> pd.DataFrame:
    """Cached raw per-variant scores CSV for one MaveDB score set.

    Columns are whatever MaveDB exports (``accession, hgvs_nt, hgvs_splice,
    hgvs_pro, score, ...``); use :func:`load_mavedb_scoreset` for the tidy,
    UniProt-addressable frame.
    """
    cache_dir = raw_dir / "mavedb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = urn.replace(":", "_")
    cache = cache_dir / f"{safe_name}_scores.csv"
    if cache.exists() and not overwrite:
        return pd.read_csv(cache)
    sess = session or make_session()
    resp = sess.get(f"{MAVEDB_API_BASE}/score-sets/{urn}/scores", timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(cache, index=False)
    return df


def load_mavedb_scoreset(
    urn: str,
    gene: str,
    raw_dir: Path,
    session: Optional[requests.Session] = None,
    overwrite: bool = False,
    assay_type: str = "lof_repair",
) -> pd.DataFrame:
    """Tidy, UniProt-addressable frame for one MaveDB score set.

    Returns columns ``[gene, uniprot_id, position, wt_aa, mut_aa, hgvs_p,
    mave_score, mave_urn, mave_assay_type]``. ``mave_score`` is the raw
    assay-scale score (sign/units differ per assay; see the score set's
    ``methodText`` in the metadata cache before comparing across assays).
    Rows with unparsable / synonymous / stop-gain HGVS tokens are dropped.
    """
    meta = fetch_scoreset_metadata(urn, raw_dir, session=session, overwrite=overwrite)
    uniprot_id = _extract_uniprot_accession(meta)
    if uniprot_id is None:
        logger.warning("%s: no UniProt cross-reference in MaveDB metadata; "
                       "rows will carry uniprot_id=None.", urn)

    raw = fetch_scoreset_scores(urn, raw_dir, session=session, overwrite=overwrite)
    if "hgvs_pro" not in raw.columns or "score" not in raw.columns:
        raise RuntimeError(
            f"{urn}: expected 'hgvs_pro' and 'score' columns, got {list(raw.columns)}")

    parsed = raw["hgvs_pro"].map(parse_hgvs_pro)
    ok = parsed.notna() & raw["score"].notna()
    n_dropped = int((~ok).sum())
    if n_dropped:
        logger.info("%s: dropped %d/%d rows (synonymous/stop/NA hgvs_pro or missing score)",
                    urn, n_dropped, len(raw))
    sub = raw.loc[ok].copy()
    wpm = pd.DataFrame(parsed.loc[ok].tolist(), columns=["wt_aa", "position", "mut_aa"])
    out = pd.concat([sub.reset_index(drop=True), wpm], axis=1)
    out["gene"] = gene.upper()
    out["uniprot_id"] = uniprot_id
    out["hgvs_p"] = ("p." + out["wt_aa"].map(lambda a: _one_to_three(a))
                     + out["position"].astype(int).astype(str)
                     + out["mut_aa"].map(lambda a: _one_to_three(a)))
    out["mave_score"] = pd.to_numeric(out["score"], errors="coerce")
    out["mave_urn"] = urn
    out["mave_assay_type"] = assay_type
    key = ["gene", "position", "wt_aa", "mut_aa"]
    out = out.drop_duplicates(subset=key, keep="first")
    result = out[key + ["uniprot_id", "hgvs_p", "mave_score", "mave_urn",
                        "mave_assay_type"]].reset_index(drop=True)
    logger.info("MaveDB %s (%s): %d usable variants (%d dropped)",
                urn, gene.upper(), len(result), n_dropped)
    return result


def _one_to_three(aa: str) -> str:
    from .data_loader import ONE_TO_THREE
    return ONE_TO_THREE[aa]


def add_rank_normalized_score(df: pd.DataFrame, score_col: str = "mave_score",
                              out_col: str = "mave_score_pct") -> pd.DataFrame:
    """Append a within-scoreset percentile-rank column (0-1, higher = larger raw score).

    Raw MaveDB scores are assay-specific in scale and sign; rank-normalizing
    within each ``mave_urn`` makes cross-assay comparison (e.g. MLH1 abundance
    vs MSH2 LOF) meaningful for a diagnostic plot even though the two are not
    validated as equivalent evidence.
    """
    out = df.copy()
    out[out_col] = out.groupby("mave_urn")[score_col].rank(pct=True)
    return out


def load_mmr_mavedb_features(
    raw_dir: Path,
    genes: Sequence[str] = ("MLH1", "MSH2", "MSH6", "PMS2"),
    session: Optional[requests.Session] = None,
    overwrite: bool = False,
    check_for_new: bool = False,
) -> pd.DataFrame:
    """Load every known MaveDB score set for the requested MMR genes.

    Parameters
    ----------
    check_for_new:
        When ``True``, re-run :func:`search_mavedb_for_gene` for genes with no
        entry in :data:`KNOWN_MMR_SCORESETS` and log (but do not silently
        auto-ingest) any hits found, so a human can vet and add the URN.
    """
    sess = session or make_session()
    frames: List[pd.DataFrame] = []
    for gene in genes:
        gene_up = gene.upper()
        urns = KNOWN_MMR_SCORESETS.get(gene_up, [])
        if not urns:
            if check_for_new:
                hits = search_mavedb_for_gene(gene_up, session=sess)
                if hits:
                    logger.warning(
                        "MaveDB search found %d possible score set(s) for %s not "
                        "in KNOWN_MMR_SCORESETS: %s -- review and add manually, "
                        "titles are free-text and may not be a real match.",
                        len(hits), gene_up,
                        [(h.get("urn"), h.get("title")) for h in hits[:5]])
            logger.info("MaveDB: no known score set for %s.", gene_up)
            continue
        for urn in urns:
            assay_type = "abundance" if gene_up == "MLH1" else "lof_repair"
            frames.append(load_mavedb_scoreset(
                urn, gene_up, raw_dir, session=sess, overwrite=overwrite,
                assay_type=assay_type))
    if not frames:
        return pd.DataFrame(columns=["gene", "position", "wt_aa", "mut_aa",
                                     "uniprot_id", "hgvs_p", "mave_score",
                                     "mave_urn", "mave_assay_type"])
    out = pd.concat(frames, ignore_index=True)
    out = add_rank_normalized_score(out)
    return out


__all__ = [
    "MAVEDB_API_BASE", "KNOWN_MMR_SCORESETS",
    "parse_hgvs_pro", "search_mavedb_for_gene",
    "fetch_scoreset_metadata", "fetch_scoreset_scores",
    "load_mavedb_scoreset", "add_rank_normalized_score",
    "load_mmr_mavedb_features",
]
