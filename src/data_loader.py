"""Data acquisition and preprocessing for gene-specific ClinVar variant datasets.

Responsibilities
----------------
1. Resolve a human gene symbol (e.g. TP53, KCNQ1, BRCA1, PTEN) to its canonical
   UniProt accession and download the canonical wild-type amino-acid sequence.
2. Download (and cache) the NCBI ClinVar ``variant_summary.txt.gz`` table,
   stream it, and filter it down to high-confidence missense substitutions of
   the requested gene.
3. Produce two tidy dataframes:
     * labelled set  -> Pathogenic / Likely-Pathogenic = 1 vs Benign / Likely-Benign = 0
     * held-out VUS set (label = NaN) reserved for prospective inference.

Only single-residue substitutions (``p.XaaPosYaa`` or ``p.XPosY``) are kept.
Labelled variants require a review status of at least one star; conflicting
interpretations are always excluded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import requests
import urllib3
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
UNIPROT_SEARCH_URL: str = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL: str = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
CLINVAR_SUMMARY_URL: str = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
ORGANISM_TAXID_HUMAN: str = "9606"

THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
ONE_TO_THREE: dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}
VALID_AA: frozenset[str] = frozenset(THREE_TO_ONE.values())

#: ClinVar review-status string -> star rating.
#: https://www.ncbi.nlm.nih.gov/clinvar/docs/review/ (star-based system).
REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 0,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no assertion for the individual variant": 0,
}

PATHOGENIC_TERMS: frozenset[str] = frozenset({"pathogenic", "likely pathogenic"})
BENIGN_TERMS: frozenset[str] = frozenset({"benign", "likely benign"})
VUS_TERMS: frozenset[str] = frozenset({"uncertain significance"})

#: ``p.R273H`` style (ClinVar ProteinChange column format).
_ONE_LETTER_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
#: ``p.(Arg273His)`` embedded inside the full variant Name field.
#: ``(?![a-z])`` rejects frameshift/extension forms such as ``p.Arg273HisfsTer12``.
_THREE_LETTER_RE = re.compile(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?![a-z])")

FINAL_COLUMNS = ["gene", "position", "wt_aa", "mut_aa", "hgvs_p", "label", "review_status"]

# --------------------------------------------------------------------------- #
# HTTP session helpers (requests + urllib3 retries)
# --------------------------------------------------------------------------- #


def make_session(max_retries: int = 5, backoff_factor: float = 1.5) -> requests.Session:
    """Build a :class:`requests.Session` with exponential-backoff retries."""
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": "variant-pathogenicity-dl/1.0"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Surface urllib3 warnings (e.g. TLS) at WARNING level instead of crashing.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


# --------------------------------------------------------------------------- #
# UniProt access
# --------------------------------------------------------------------------- #


def fetch_uniprot_accession(gene: str, session: Optional[requests.Session] = None) -> str:
    """Resolve a HGNC gene symbol to its reviewed human UniProt accession.

    Uses the UniProt REST search API restricted to SwissProt (reviewed) human
    entries. Raises :class:`ValueError` if no unique match can be determined;
    callers can then fall back to an explicit accession via ``--uniprot``.
    """
    sess = session or make_session()
    params = {
        "query": (
            f"(gene_exact:{gene}) AND (organism_id:{ORGANISM_TAXID_HUMAN}) "
            "AND (reviewed:true)"
        ),
        "format": "json",
        "fields": "accession,id,protein_name,length",
        "size": "5",
    }
    logger.info("Resolving gene symbol %s via UniProt search ...", gene)
    resp = sess.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(
            f"No reviewed human UniProt entry found for gene '{gene}'. "
            "Pass an explicit accession with --uniprot (e.g. --uniprot P04637)."
        )
    entry = results[0]
    accession = entry["primaryAccession"]
    logger.info("Resolved %s -> %s (%s)", gene.upper(), accession, entry.get("id"))
    return accession


def fetch_uniprot_sequence(accession: str, session: Optional[requests.Session] = None) -> str:
    """Download the canonical FASTA sequence for a UniProt accession."""
    sess = session or make_session()
    url = UNIPROT_FASTA_URL.format(accession=accession)
    logger.info("Fetching canonical sequence from %s", url)
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">")).upper()
    invalid = set(seq) - VALID_AA - {"X"}
    if invalid:
        raise ValueError(
            f"Sequence for {accession} contains non-standard residues: {sorted(invalid)}"
        )
    logger.info("Canonical sequence length: %d aa", len(seq))
    return seq


# --------------------------------------------------------------------------- #
# ClinVar variant_summary download (cached)
# --------------------------------------------------------------------------- #


def download_clinvar_summary(raw_dir: Path, overwrite: bool = False) -> Path:
    """Download the ClinVar tab-delimited summary table into *raw_dir*.

    The archive is ~40 MB compressed and is only re-downloaded when missing or
    when *overwrite* is requested.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "variant_summary.txt.gz"
    if dest.exists() and not overwrite:
        logger.info("Using cached ClinVar summary: %s", dest)
        return dest

    logger.info("Downloading ClinVar variant summary (%s) ...", CLINVAR_SUMMARY_URL)
    sess = make_session()
    with sess.get(CLINVAR_SUMMARY_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="ClinVar")
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                progress.update(len(chunk))
        progress.close()
    tmp.rename(dest)
    logger.info("Saved ClinVar summary to %s", dest)
    return dest


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def stars_for_review(review_status: str) -> int:
    """Map a ClinVar ReviewStatus string to its star rating."""
    return REVIEW_STATUS_STARS.get(str(review_status).strip().lower(), 0)


def parse_protein_substitution(protein_change: str, name: str) -> Optional[Tuple[str, int, str]]:
    """Extract ``(wt_aa, position, mut_aa)`` from ClinVar fields.

    Tries the one-letter ``ProteinChange`` column first (e.g. ``R273H``), then
    falls back to the three-letter HGVS-p expression embedded in ``Name``
    (e.g. ``TP53 c.818G>A (p.Arg273His)``).

    Returns ``None`` for anything that is not a clean single-substitution.
    """
    pc = str(protein_change or "").strip()
    match = _ONE_LETTER_RE.match(pc)
    if match:
        wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
        if wt in VALID_AA and mut in VALID_AA and wt != mut:
            return wt, pos, mut

    nm = str(name or "")
    match = _THREE_LETTER_RE.search(nm)
    if match:
        wt3, pos, mut3 = match.group(1), int(match.group(2)), match.group(3)
        wt, mut = THREE_TO_ONE.get(wt3, ""), THREE_TO_ONE.get(mut3, "")
        if wt and mut and wt != mut:
            return wt, pos, mut
    return None


def classify_clinical_significance(
    clinical_significance: str, review_stars: int, min_stars: int = 1
) -> Optional[Tuple[str, float]]:
    """Classify a ClinVar ClinicalSignificance string.

    Returns
    -------
    tuple(kind, label) where kind is one of ``{"labelled", "vus"}`` and label
    is 0.0 / 1.0 (or NaN for VUS); or ``None`` when the row must be discarded
    (conflicting interpretations, multi-category assertions, or review status
    below *min_stars*).
    """
    cats = {
        part.strip().lower()
        for group in str(clinical_significance).split("|")
        for part in group.split("/")
        if part.strip()
    }
    if not cats or any("conflicting" in c for c in cats):
        return None  # conflicting interpretations are always excluded
    if cats <= PATHOGENIC_TERMS:
        return ("labelled", 1.0) if review_stars >= min_stars else None
    if cats <= BENIGN_TERMS:
        return ("labelled", 0.0) if review_stars >= min_stars else None
    if cats == VUS_TERMS:
        return ("vus", float("nan"))  # VUS is kept regardless of star count
    return None  # mixed categories (risk factor, drug response, ...) -> discard


def _stream_gene_variants(summary_path: Path, gene: str, min_stars: int) -> pd.DataFrame:
    """Backward-compatible single-gene wrapper around :func:`_stream_genes_variants`."""
    frames = _stream_genes_variants(summary_path, [gene], min_stars=min_stars)
    if not frames:
        raise RuntimeError(
            f"No usable ClinVar variants found for gene '{gene}'. Check spelling "
            "of the gene symbol."
        )
    return frames[gene.upper()]


def _stream_genes_variants(
    summary_path: Path, genes: Sequence[str], min_stars: int
) -> dict[str, pd.DataFrame]:
    """Stream the gzipped ClinVar table once and keep rows for *all* genes.

    A single decompression pass serves every requested gene, which matters:
    ``variant_summary.txt.gz`` is ~440 MB compressed / ~4 GB uncompressed, so
    per-gene passes would multiply an already slow step.

    Returns
    -------
    Mapping from upper-case gene symbol to a raw record frame with columns
    ``[gene, position, wt_aa, mut_aa, hgvs_p, label, review_status, _kind,
    _stars]``.  Genes with zero usable variants are simply absent from the
    result (callers decide whether that is fatal).
    """
    wanted_genes = {g.casefold(): g.upper() for g in genes}
    wanted_cols = {"#AlleleID", "GeneSymbol", "Name", "ProteinChange",
                   "ClinicalSignificance", "ReviewStatus", "VariationType"}
    records: dict[str, list[dict]] = {up: [] for up in wanted_genes.values()}
    reader = pd.read_csv(
        summary_path, sep="\t", compression="gzip", dtype=str,
        chunksize=200_000, low_memory=False,
    )
    for chunk_idx, chunk in enumerate(reader):
        available = [c for c in wanted_cols if c in chunk.columns]
        sub = chunk[available]
        sub = sub[
            sub["GeneSymbol"].str.casefold().isin(wanted_genes.keys())
        ]
        if sub.empty:
            logger.debug("chunk %d: no rows for any requested gene", chunk_idx)
            continue
        for row in sub.itertuples(index=False):
            rowd = row._asdict()
            gene_up = wanted_genes[str(rowd["GeneSymbol"]).casefold()]
            parsed = parse_protein_substitution(rowd.get("ProteinChange"), rowd.get("Name"))
            if parsed is None:
                continue
            wt_aa, position, mut_aa = parsed
            stars = stars_for_review(rowd.get("ReviewStatus", ""))
            verdict = classify_clinical_significance(
                rowd.get("ClinicalSignificance", ""), stars, min_stars=min_stars
            )
            if verdict is None:
                continue
            kind, label = verdict
            records[gene_up].append({
                "gene": gene_up,
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "hgvs_p": f"p.{ONE_TO_THREE[wt_aa]}{position}{ONE_TO_THREE[mut_aa]}",
                "label": label,
                "review_status": str(rowd.get("ReviewStatus", "")),
                "_kind": kind,
                "_stars": stars,
            })
        logger.info(
            "chunk %d: cumulative qualifying variants per gene: %s",
            chunk_idx, {g: len(rs) for g, rs in records.items()},
        )
    return {
        gene: pd.DataFrame.from_records(recs)
        for gene, recs in records.items() if recs
    }


def build_gene_dataset(
    gene: str,
    raw_dir: Path,
    processed_dir: Path,
    min_stars: int = 1,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build (and cache) the labelled and VUS dataframes for *gene*.

    Returns
    -------
    labeled_df : DataFrame with columns
        ``[gene, position, wt_aa, mut_aa, hgvs_p, label, review_status]``
    vus_df : same schema, ``label`` set to NaN.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = processed_dir / f"{gene}_labeled_variants.csv"
    vus_path = processed_dir / f"{gene}_vus_variants.csv"
    if labeled_path.exists() and vus_path.exists() and not overwrite:
        logger.info("Loading cached ClinVar datasets for %s", gene)
        labeled = pd.read_csv(labeled_path)
        vus = pd.read_csv(vus_path)
        return labeled[FINAL_COLUMNS], vus[FINAL_COLUMNS]

    summary_path = download_clinvar_summary(raw_dir, overwrite=False)
    frames = _stream_genes_variants(summary_path, [gene], min_stars=min_stars)
    if not frames:
        raise RuntimeError(
            f"No usable ClinVar variants found for gene '{gene}'. Check spelling "
            "of the gene symbol."
        )
    frame = frames[gene.upper()]

    labeled, vus = _dedupe_and_split(frame)

    counts = labeled["label"].value_counts().to_dict()
    logger.info(
        "%s: %d labelled variants (pathogenic=%d, benign=%d), %d VUS held out",
        gene.upper(), len(labeled), counts.get(1, 0), counts.get(0, 0), len(vus),
    )

    labeled.to_csv(labeled_path, index=False)
    vus.to_csv(vus_path, index=False)
    logger.info("Cached labelled set -> %s | VUS set -> %s", labeled_path, vus_path)
    return labeled, vus


def _dedupe_and_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """De-duplicate on hgvs_p (highest-reviewed assertion wins) and split."""
    frame = (
        frame.sort_values("_stars", ascending=False)
        .drop_duplicates(subset=["hgvs_p"], keep="first")
        .reset_index(drop=True)
    )
    labeled = frame[frame["_kind"] == "labelled"].copy()
    labeled["label"] = labeled["label"].astype(int)
    vus = frame[frame["_kind"] == "vus"].copy()
    vus["label"] = pd.NA
    return labeled[FINAL_COLUMNS].reset_index(drop=True), \
        vus[FINAL_COLUMNS].reset_index(drop=True)


def build_multi_gene_dataset(
    genes: Sequence[str],
    raw_dir: Path,
    processed_dir: Path,
    min_stars: int = 1,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build combined labelled + VUS frames for several genes in ONE pass.

    Per-gene CSV caches are still written (same naming as
    :func:`build_gene_dataset`), and any gene whose cache already exists is
    served from disk; the decompression pass runs only for the missing ones.
    Genes with no usable ClinVar entries are logged and skipped.

    Returns
    -------
    (labeled_all, vus_all): concatenated frames over all requested genes with
    the standard ``FINAL_COLUMNS`` schema.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    missing: list[str] = []
    for gene in genes:
        gene_up = gene.upper()
        labeled_path = processed_dir / f"{gene_up}_labeled_variants.csv"
        vus_path = processed_dir / f"{gene_up}_vus_variants.csv"
        if labeled_path.exists() and vus_path.exists() and not overwrite:
            labeled = pd.read_csv(labeled_path)[FINAL_COLUMNS]
            vus = pd.read_csv(vus_path)[FINAL_COLUMNS]
            results[gene_up] = (labeled, vus)
        else:
            missing.append(gene_up)

    if missing:
        logger.info("Single-pass ClinVar extraction for: %s", ", ".join(missing))
        summary_path = download_clinvar_summary(raw_dir, overwrite=False)
        frames = _stream_genes_variants(summary_path, missing, min_stars=min_stars)
        for gene_up in missing:
            if gene_up not in frames or frames[gene_up].empty:
                logger.warning("No usable ClinVar variants for '%s'; skipped.", gene_up)
                continue
            labeled, vus = _dedupe_and_split(frames[gene_up])
            counts = labeled["label"].value_counts().to_dict()
            logger.info(
                "%s: %d labelled variants (pathogenic=%d, benign=%d), %d VUS held out",
                gene_up, len(labeled), counts.get(1, 0), counts.get(0, 0), len(vus),
            )
            labeled.to_csv(processed_dir / f"{gene_up}_labeled_variants.csv", index=False)
            vus.to_csv(processed_dir / f"{gene_up}_vus_variants.csv", index=False)
            results[gene_up] = (labeled, vus)

    if not results:
        raise RuntimeError(
            f"None of the requested genes {list(genes)} yielded usable ClinVar "
            "variants. Check gene symbols."
        )
    labeled_all = pd.concat([lv[0] for lv in results.values()], ignore_index=True)
    vus_all = pd.concat([lv[1] for lv in results.values()], ignore_index=True)
    order = [g.upper() for g in genes if g.upper() in results]
    labeled_all["gene"] = pd.Categorical(labeled_all["gene"], categories=order, ordered=True)
    labeled_all = (
        labeled_all.sort_values(["gene", "position"]).reset_index(drop=True)
    )
    labeled_all["gene"] = labeled_all["gene"].astype(str)
    vus_all = vus_all.set_index("gene").loc[order].reset_index() if len(vus_all) else vus_all
    n_pos = int((labeled_all["label"] == 1).sum())
    n_neg = int((labeled_all["label"] == 0).sum())
    logger.info("Combined: %d genes | %d labelled (%d P/LP, %d B/LB) | %d VUS",
                len(order), len(labeled_all), n_pos, n_neg, len(vus_all))
    return labeled_all, vus_all
