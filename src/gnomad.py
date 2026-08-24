"""gnomAD v4 allele-frequency adapter for the MMR pathogenicity project.

Why this module exists (PROJECT_PLAN.md, Phase 1 + Phase 3)
----------------------------------------------------------
* **Phase 1** needs gnomAD v4 allele frequencies for the four Lynch-syndrome
  MMR genes (MLH1/MSH2/MSH6/PMS2) to apply the ACMG frequency criteria
  BA1 / BS1 / PM2 as *filtering evidence*.
* **Phase 3** (MVmamba recipe) additionally uses gnomAD allele frequency as an
  *explicit input feature*: MVmamba's own ablation showed that adding it on top
  of a strong structure+sequence model improves every metric
  (AUC 0.895 -> 0.901).

How data is fetched
-------------------
Downloading whole-chromosome sites VCFs just to cover four genes would pull in
tens of gigabytes.  Instead we query the official gnomAD GraphQL API
(https://gnomad.broadinstitute.org/api) once per gene for ``gnomad_r4``
(v4) variants and keep only missense substitutions with a protein-level HGVS
on any transcript.  Every row is then validated against the canonical UniProt
sequence (wt residue must match at the parsed position), which automatically
removes transcript/isoform numbering mismatches — the same numbering-safety
contract used by every other source in this repository.

ACMG frequency flags (Richards et al. 2015; thresholds configurable)
--------------------------------------------------------------------
* ``acmg_ba1``  — AF > 0.05 in the combined population (stand-alone benign).
* ``acmg_bs1``  — AF greater than expected for an autosomal-dominant
                  late-onset? no: for Lynch syndrome (AD, early onset) we use
                  a conservative default of AF > 1e-3.
* ``acmg_pm2``  — absent from (or extremely rare in) control populations:
                  joint AF < ``pm2_threshold`` (default 1e-5).

These are exported as integer flag columns next to the raw frequencies so
downstream code can both filter on them (BA1) and feed them as features.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from .data_loader import ONE_TO_THREE, THREE_TO_ONE, VALID_AA, make_session

logger = logging.getLogger(__name__)

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET = "gnomad_r4"          # v4.1 joint release
GNOMAD_REFERENCE_GENOME = "GRCh38"

#: ACMG frequency-threshold defaults (see module docstring).
DEFAULT_BA1_AF = 0.05
DEFAULT_BS1_AF = 1e-3                 # AD disease, early onset -> conservative
DEFAULT_PM2_AF = 1e-5

_GENE_VARIANTS_QUERY = """
query($gene: String!, $dataset: DatasetId!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    symbol
    chrom
    variants(dataset: $dataset) {
      variant_id
      consequence
      hgvsp
      transcript_id
      genome { af ac an }
      exome  { af ac an }
    }
  }
}
"""

#: ``p.Arg273His`` / ``p.(Arg273His)`` -> wt3, pos, mut3.  Rejects fs/ext/*.
_HGVSP_RE = re.compile(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")


def parse_hgvsp(hgvsp: str) -> Optional[tuple[str, int, str]]:
    """Parse a protein-level HGVS string into ``(wt, pos, mut)`` one-letter."""
    m = _HGVSP_RE.match(str(hgvsp or "").strip())
    if not m:
        return None
    wt = THREE_TO_ONE.get(m.group(1), "")
    mut = THREE_TO_ONE.get(m.group(3), "")
    if not wt or not mut or wt == mut or wt not in VALID_AA or mut not in VALID_AA:
        return None
    return wt, int(m.group(2)), mut


def _gql(session: requests.Session, query: str, variables: Dict[str, str],
         max_retries: int = 5, timeout: int = 120) -> dict:
    """POST one GraphQL request with exponential-backoff retries."""
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.post(
                GNOMAD_API_URL,
                json={"query": query, "variables": variables},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - transport errors are retried
            last = exc
            wait = 5.0 * (attempt + 1)
            logger.warning("gnomAD API attempt %d failed (%s); retry in %.0fs",
                           attempt + 1, exc, wait)
            time.sleep(wait)
            continue
        if payload.get("errors"):
            raise RuntimeError(f"gnomAD GraphQL error(s): {payload['errors']}")
        return payload["data"]
    raise RuntimeError(f"gnomAD API unreachable after {max_retries} attempts: {last}")


def fetch_gene_gnomad_variants(gene: str,
                               session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Pull every gnomAD v4 missense substitution reported for *gene*.

    Returns a tidy frame keyed by protein substitution:

        [uniprot-independent] gene, position, wt_aa, mut_aa, hgvs_p,
        variant_id, consequence, transcript_id,
        gnomad_af_genome, gnomad_ac_genome, gnomad_an_genome,
        gnomad_af_exome,  gnomad_ac_exome,  gnomad_an_exome,
        gnomad_af_joint,  gnomad_ac_joint,  gnomad_an_joint
    """
    sess = session or make_session()
    data = _gql(sess, _GENE_VARIANTS_QUERY, {"gene": gene.upper(), "dataset": GNOMAD_DATASET})
    gene_payload = data.get("gene")
    if gene_payload is None:
        raise RuntimeError(f"gnomAD returned no gene object for '{gene}'.")
    rows: List[dict] = []
    for v in gene_payload.get("variants") or []:
        parsed = parse_hgvsp(v.get("hgvsp"))
        if parsed is None:
            continue
        wt, pos, mut = parsed
        genome = v.get("genome") or {}
        exome = v.get("exome") or {}
        af_g = genome.get("af")
        af_e = exome.get("af")
        # Joint frequency approximates the max coverage union; gnomAD exposes
        # exome/genome separately here, so combine by AC/AN summation when both
        # exist (standard browser behaviour for the joint display).
        if af_g is not None and af_e is not None:
            ac_j = float(genome.get("ac", 0)) + float(exome.get("ac", 0))
            an_j = float(genome.get("an", 0)) + float(exome.get("an", 0))
            af_j = ac_j / an_j if an_j else np.nan
        else:
            src = genome if af_g is not None else exome
            other = exome if af_g is not None else genome
            ac_j = float(src.get("ac", 0)) + float(other.get("ac") or 0)
            an_j = float(src.get("an", 0)) + float(other.get("an") or 0)
            af_j = ac_j / an_j if an_j else np.nan
        rows.append({
            "gene": gene.upper(),
            "position": pos,
            "wt_aa": wt,
            "mut_aa": mut,
            "hgvs_p": f"p.{ONE_TO_THREE[wt]}{pos}{ONE_TO_THREE[mut]}",
            "variant_id": v.get("variant_id"),
            "consequence": v.get("consequence"),
            "transcript_id": v.get("transcript_id"),
            "gnomad_af_genome": af_g,
            "gnomad_ac_genome": genome.get("ac"),
            "gnomad_an_genome": genome.get("an"),
            "gnomad_af_exome": af_e,
            "gnomad_ac_exome": exome.get("ac"),
            "gnomad_an_exome": exome.get("an"),
            "gnomad_af_joint": af_j,
            "gnomad_ac_joint": ac_j,
            "gnomad_an_joint": an_j,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("gnomAD: no usable missense substitutions for %s.", gene)
        return df
    # Multiple gnomAD variants can collapse onto one protein substitution
    # (alternate codon changes); keep the highest-AF observation per protein
    # substitution — the conservative choice for benign-ness evidence.
    df = (df.sort_values("gnomad_af_joint", ascending=False, na_position="last")
          .drop_duplicates(subset=["position", "wt_aa", "mut_aa"], keep="first")
          .sort_values(["position", "mut_aa"])
          .reset_index(drop=True))
    logger.info("gnomAD %s: %d unique missense protein substitutions.",
                gene.upper(), len(df))
    return df


def validate_against_sequence(df: pd.DataFrame, sequence: str) -> pd.DataFrame:
    """Drop rows whose wt residue disagrees with the canonical UniProt sequence."""
    seq_arr = np.array(list(sequence), dtype="U1")
    pos = pd.to_numeric(df["position"], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(pos) & (pos >= 1) & (pos <= len(sequence))
    idx = np.flatnonzero(ok)
    wt_ok = seq_array_check(seq_arr, pos[idx].astype("int64"), df["wt_aa"].to_numpy()[idx])
    ok[idx] &= wt_ok
    dropped = int((~ok).sum())
    if dropped:
        logger.warning("gnomAD: dropped %d rows failing canonical-sequence "
                       "validation (isoform numbering).", dropped)
    return df.loc[ok].reset_index(drop=True)


def seq_array_check(seq_arr: np.ndarray, positions_int: np.ndarray,
                    wt_aas: np.ndarray) -> np.ndarray:
    positions_int = np.asarray(positions_int, dtype=np.int64)
    chars = seq_arr[np.clip(positions_int - 1, 0, len(seq_arr) - 1)]
    return chars == np.asarray(wt_aas, dtype="U1")


def add_frequency_flags(df: pd.DataFrame,
                        ba1_af: float = DEFAULT_BA1_AF,
                        bs1_af: float = DEFAULT_BS1_AF,
                        pm2_af: float = DEFAULT_PM2_AF) -> pd.DataFrame:
    """Append ACMG BA1 / BS1 / PM2 integer flag columns from joint AF.

    Missing frequencies are treated as *absent from gnomAD* for PM2 (the
    variant was simply never observed in these cohorts) but produce flag=0 for
    BA1/BS1 (no evidence of a high frequency).
    """
    out = df.copy()
    if "gnomad_af_joint" not in out.columns:
        out["gnomad_af_joint"] = np.nan
    af = pd.to_numeric(out["gnomad_af_joint"], errors="coerce")
    out["acmg_ba1"] = (af > ba1_af).fillna(False).astype(int)
    out["acmg_bs1"] = (af > bs1_af).fillna(False).astype(int)
    # PM2: absent (NaN) or below threshold.
    out["acmg_pm2"] = ((af.isna()) | (af < pm2_af)).astype(int)
    # Log-scaled explicit input features (MVmamba Phase-3 recipe): log10(AF)
    # with absent variants mapped below the smallest observed value so the
    # ordering "absent < rare < common" is preserved numerically.
    floor = -9.0
    log_af = np.log10(af.where(af > 0)).fillna(floor)
    out["gnomad_log10_af"] = log_af.clip(lower=floor).astype(float)
    out.attrs["freq_thresholds"] = {"ba1": ba1_af, "bs1": bs1_af, "pm2": pm2_af}
    return out


GNOMAD_FEATURE_COLUMNS = [
    "gnomad_af_joint", "gnomad_af_genome", "gnomad_af_exome",
    "gnomad_ac_joint", "gnomad_an_joint",
    "gnomad_log10_af", "acmg_ba1", "acmg_bs1", "acmg_pm2",
]


def load_or_fetch_gene(gene: str, raw_dir: Path,
                       sequence: Optional[str] = None,
                       session: Optional[requests.Session] = None,
                       overwrite: bool = False,
                       ba1_af: float = DEFAULT_BA1_AF,
                       bs1_af: float = DEFAULT_BS1_AF,
                       pm2_af: float = DEFAULT_PM2_AF) -> pd.DataFrame:
    """Cached per-gene gnomAD table with flags applied (data/raw/gnomad/)."""
    cache_dir = Path(raw_dir) / "gnomad"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{gene.upper()}_gnomad_v4.csv"
    if path.exists() and not overwrite:
        logger.info("Using cached gnomAD table: %s", path)
        df = pd.read_csv(path)
    else:
        df = fetch_gene_gnomad_variants(gene, session=session)
        if sequence is not None and len(df):
            df = validate_against_sequence(df, sequence)
        df.to_csv(path, index=False)
        logger.info("Cached gnomAD table -> %s", path)
    return add_frequency_flags(df, ba1_af=ba1_af, bs1_af=bs1_af, pm2_af=pm2_af)


def join_gnomad_features(target: pd.DataFrame, gnomad_df: pd.DataFrame,
                         feature_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Left-join gnomAD frequency features/flags onto a variant table.

    The join key is ``(gene, position, wt_aa, mut_aa)``, identical to every
    other source merge in this project.  Rows without gnomAD observations keep
    NaN frequencies (PM2-consistent) and flag=0.
    """
    cols = list(feature_columns or GNOMAD_FEATURE_COLUMNS)
    key = ["gene", "position", "wt_aa", "mut_aa"]
    source_cols = [c for c in key + cols if c in gnomad_df.columns]
    missing = set(cols) - set(source_cols)
    if missing:
        raise ValueError(f"gnomAD frame lacks requested feature columns: {sorted(missing)}")
    sub = gnomad_df[source_cols].drop_duplicates(subset=key)
    merged = target.merge(sub, on=key, how="left", validate="m:1")
    for c in cols:
        if c.endswith(("ba1", "bs1")):
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)
        elif c == "acmg_pm2":
            # Absent from the join => absent from gnomAD => PM2 evidence applies.
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(1).astype(int)
    return merged


def fetch_mmr_genes(raw_dir: Path, sequences: Dict[str, str],
                    genes: Sequence[str] = ("MLH1", "MSH2", "MSH6", "PMS2"),
                    overwrite: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch/cache gnomAD tables for all four MMR genes."""
    session = make_session()
    out: Dict[str, pd.DataFrame] = {}
    for gene in genes:
        seq = sequences.get(gene)
        out[gene.upper()] = load_or_fetch_gene(gene, raw_dir, sequence=seq,
                                               session=session, overwrite=overwrite)
    return out


__all__ = [
    "GNOMAD_API_URL", "GNOMAD_DATASET",
    "DEFAULT_BA1_AF", "DEFAULT_BS1_AF", "DEFAULT_PM2_AF",
    "parse_hgvsp", "fetch_gene_gnomad_variants", "validate_against_sequence",
    "add_frequency_flags", "load_or_fetch_gene", "join_gnomad_features",
    "fetch_mmr_genes", "GNOMAD_FEATURE_COLUMNS",
]
