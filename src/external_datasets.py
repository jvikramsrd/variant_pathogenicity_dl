"""External dataset acquisition, parsing and caching.

This module turns four independently published resources into tidy,
UniProt-addressable pandas tables so that they can be merged into a single
extended variant-pathogenicity dataset (see :mod:`src.extended_builder`).

Sources handled here
--------------------
1. ProteinGym v1.3 DMS substitutions
   217 deep-mutational-scan assays / ~2.5M mutants with continuous fitness
   scores (``DMS_score``) and an assay-specific binary label
   (``DMS_score_bin``).  Every assay is keyed by a UniProt accession in the
   reference file, which makes protein-level joins trivial.
2. ProteinGym v1.3 clinical benchmark
   ~63K clinically annotated missense substitutions spread over 2.5K human
   proteins, distributed as one CSV per **RefSeq NP_ protein accession**.
   Labels live in ``DMS_bin_score``.
3. ProteinGym v1.3 zero-shot clinical model scores
   One CSV per RefSeq NP_ accession containing the scores of ~30 published
   predictors (EVE, GEMME, ESM1b, TranceptEVE_L, PoET, REVEL, CADD, SIFT4G,
   PolyPhen2-HVAR, MetaRNN, VEST4, ...) for exactly the clinical-benchmark
   variants.  These are joined onto our ClinVar rows as *prior features*.
   Because the files are keyed by RefSeq accessions we translate NP_ -> panel
   proteins via exact whole-protein sequence matching against the canonical
   UniProt sequences of the gene panel (see :func:`map_np_to_panel`), which
   also guarantees identical residue numbering before any positional join.
4. AlphaMissense aa-substitution table (2023-09-18 release)
   All ~71M single-residue substitutions of the human proteome scored by
   AlphaMissense, keyed directly by UniProt accession.  The full table is
   1.2 GB compressed; :func:`stream_filter_alphamissense` streams it once and
   writes only rows belonging to the requested accessions to a small cache.

Additionally:

5. UniProt domain annotations (REST) -- per-accession domain/region intervals
   used to attach an ``in_domain`` boolean feature downstream.
6. A generic, resumable downloader (:func:`download_file`) with SHA-256
   verification support that every source uses.

Every public function returns/accepts plain paths under ``data/raw/<source>/``
and never mutates global state.  Nothing in this module needs torch or
transformers, so it can be used standalone from the CLI in
``scripts/build_extended_dataset.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from .data_loader import ONE_TO_THREE, VALID_AA, make_session

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source registry: URLs, versions, licences (mirrored into the manifest)
# --------------------------------------------------------------------------- #

PROTEINGYM_VERSION = "v1.3"
PROTEINGYM_BASE_HARVARD = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
PROTEINGYM_ZENODO_RECORD = "https://zenodo.org/records/15293562"

ALPHAMISSENSE_AA_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)
ALPHAMISSENSE_VERSION = "1.0 (2023-09-18 release)"
UNIPROT_DOMAINS_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

#: Model score columns kept from the ProteinGym zero-shot clinical bundle.
#: The list intentionally mixes orthogonal families:
#:   * generative PLMs on MSAs / retrievers: EVE, GEMME, ESM1b, TranceptEVE_L,
#:     PoET, TranceptEVE (kept for completeness)
#:   * classical ensemble/pathogenicity tools: REVEL, CADD, MetaRNN, VEST4,
#:     VARITY(R), M_CAP? (not present) -> BayesDel(addAF), MPC
#:   * simple alignment heuristics: SIFT4G, PolyPhen2(HVAR), FATHMM, Provean
ZERO_SHOT_MODEL_COLUMNS: Tuple[str, ...] = (
    "EVE", "TranceptEVE_L", "GEMME", "ESM1b", "PoET",
    "REVEL", "CADD", "MetaRNN", "VEST4",
    "BayesDel (addAF)", "MPC", "SIFT4G", "PolyPhen2 (HVAR)",
    "Provean", "FATHMM", "DEOGEN2", "MutationAssessor",
)

_MUTANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# --------------------------------------------------------------------------- #
# Generic download helper
# --------------------------------------------------------------------------- #
def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    session: Optional[requests.Session] = None,
    overwrite: bool = False,
    expected_sha256: Optional[str] = None,
    min_bytes: int = 1024,
    timeout: int = 180,
    max_attempts: int = 5,
) -> Path:
    """Download *url* into *dest* with resume + retry + optional checksum.

    Downloads are written to ``.part`` and atomically promoted only after
    validation. An existing *dest* is trusted when it is at least *min_bytes*
    bytes and
    passes *expected_sha256* (if provided); otherwise the transfer resumes via
    HTTP Range requests when the server supports them.

    *max_attempts* bounds the resume loop. Five is fine for a small artefact,
    but a multi-hundred-megabyte file on a link that drops every few megabytes
    needs one attempt per drop -- each attempt continues from the ``.part``
    rather than restarting, so a larger budget costs nothing when the link is
    healthy.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        if expected_sha256 is None or sha256_of(dest) == expected_sha256:
            logger.info("Using cached %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
            return dest
        logger.warning("%s failed checksum; re-downloading.", dest)
        dest.unlink()

    sess = session or make_session()
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers: Dict[str, str] = {}
    mode = "wb"
    if tmp.exists() and tmp.stat().st_size > 0:
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            with sess.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                if resp.status_code == 416:  # range not satisfiable -> restart clean
                    tmp.unlink(missing_ok=True)
                    headers.pop("Range", None)
                    mode = "wb"
                    continue
                resp.raise_for_status()
                # Some mirrors ignore Range and return 200 with the complete
                # resource. Appending that response corrupts the cache, so
                # restart from zero rather than trusting the server.
                if mode == "ab" and resp.status_code != 206:
                    logger.warning("%s ignored Range request; restarting download.", dest.name)
                    tmp.unlink(missing_ok=True)
                    headers.pop("Range", None)
                    mode = "wb"
                    continue
                total_hdr = resp.headers.get("Content-Length")
                start_off = tmp.stat().st_size if mode == "ab" else 0
                total = int(total_hdr) + start_off if total_hdr else None
                with open(tmp, mode) as fh:
                    done = start_off
                    t_last = 0.0
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if total and now - t_last > 10:
                            t_last = now
                            pct = 100.0 * done / total
                            logger.info("  %s: %.1f/%.1f MB (%.1f%%)",
                                        dest.name, done / 1e6, total / 1e6, pct)
            if not tmp.exists() or tmp.stat().st_size < min_bytes:
                raise IOError(f"Downloaded file is unexpectedly small: {tmp}")
            if expected_sha256 is not None:
                got = sha256_of(tmp)
                if got != expected_sha256:
                    raise IOError(f"SHA256 mismatch for {dest.name}: {got}")
            # os.replace is atomic on the same filesystem and never leaves a
            # partially written destination visible to parsers.
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last_err = exc
            wait = min(3.0 * (attempt + 1), 30.0)
            logger.warning("Download attempt %d failed (%s); retrying in %.0fs",
                           attempt + 1, exc, wait)
            time.sleep(wait)
            # After a failure the .part may be longer than what the server
            # still has; fall back to a fresh connection with updated Range.
            if tmp.exists():
                headers["Range"] = f"bytes={tmp.stat().st_size}-"
                mode = "ab"
    raise RuntimeError(f"Could not download {url}: {last_err}")


def validate_zip_file(path: Path) -> None:
    """Fail closed when a cached archive is unreadable or has corrupt members."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise IOError(f"corrupt member {bad_member!r}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise IOError(f"Invalid ZIP archive {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# 1+2+3. ProteinGym v1.3 bundles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProteinGymFiles:
    dms_zip: Path
    dms_reference: Path
    clinical_zip: Path
    zeroshot_zip: Path


def proteingym_dir(raw_dir: Path) -> Path:
    return raw_dir / "proteingym"


def download_proteingym(raw_dir: Path, overwrite: bool = False) -> ProteinGymFiles:
    """Fetch all four ProteinGym v1.3 artefacts needed downstream."""
    out_dir = proteingym_dir(raw_dir)
    spec = {
        "dms_zip": ("DMS_ProteinGym_substitutions.zip",),
        "dms_reference": ("DMS_substitutions.csv",),
        "clinical_zip": ("clinical_ProteinGym_substitutions.zip",),
        "zeroshot_zip": ("zero_shot_clinical_substitutions_scores.zip",),
    }
    paths: Dict[str, Path] = {}
    sess = make_session()
    for key, (fname,) in spec.items():
        target = out_dir / fname
        if overwrite and target.exists():
            target.unlink()
        url = f"{PROTEINGYM_BASE_HARVARD}/{fname}"
        paths[key] = download_file(url, target, session=sess)
        if target.suffix == ".zip":
            validate_zip_file(paths[key])
    return ProteinGymFiles(**paths)


def parse_mutant_token(mutant: str) -> Optional[Tuple[str, int, str]]:
    """Parse ``A119D`` style tokens -> ``(wt, pos, mut)`` or ``None``."""
    m = _MUTANT_RE.match(str(mutant).strip())
    if not m:
        return None
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    if wt not in VALID_AA or mut not in VALID_AA or wt == mut:
        return None
    return wt, pos, mut


def load_dms_assays(raw_dir: Path, taxon: str = "human") -> pd.DataFrame:
    """Tidy long-format table of every ProteinGym substitution mutant.

    Returns columns::

        uniprot_id, dms_id, taxon, position, wt_aa, mut_aa, mutant,
        hgvs_p, dms_score, dms_bin, first_author, year, selection_type

    ``taxon`` filters the reference table (case-insensitive substring match;
    pass ``None`` to keep all organisms).
    """
    pgf = download_proteingym(raw_dir)
    ref = pd.read_csv(pgf.dms_reference, dtype=str)
    if taxon is not None:
        keep = ref["taxon"].str.contains(taxon, case=False, na=False)
        ref = ref[keep]
        logger.info("ProteinGym reference: %d %s assays of %d total",
                    len(ref), taxon, len(pd.read_csv(pgf.dms_reference, usecols=[0])))

    frames: List[pd.DataFrame] = []
    with zipfile_ctx(pgf.dms_zip) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".csv")}
        wanted = dict(zip(ref["DMS_filename"], ref.to_dict("records")))
        for fname, meta_row in wanted.items():
            member = members.get(fname)
            if member is None:
                logger.warning("Reference lists %s but zip lacks it; skipped.", fname)
                continue
            df = pd.read_csv(zf.open(member))
            required = {"mutant", "DMS_score", "DMS_score_bin"}
            missing_cols = required - set(df.columns)
            if df.empty or missing_cols:
                logger.warning("Skipping DMS file %s: missing %s", fname,
                               sorted(missing_cols))
                continue
            parsed = df["mutant"].map(parse_mutant_token)
            ok = parsed.notna()
            n_bad = int((~ok).sum())
            if n_bad:
                logger.debug("%s: dropped %d unparsable mutants", fname, n_bad)
            sub = df.loc[ok].copy()
            wt_pos_mut = pd.DataFrame(parsed.loc[ok].tolist(),
                                      columns=["wt_aa", "position", "mut_aa"])
            sub = pd.concat([sub.reset_index(drop=True), wt_pos_mut], axis=1)
            sub["uniprot_id"] = meta_row["UniProt_ID"]
            sub["dms_id"] = meta_row["DMS_id"]
            sub["taxon"] = meta_row["taxon"]
            sub["first_author"] = meta_row.get("first_author")
            sub["year"] = meta_row.get("year")
            sub["selection_type"] = meta_row.get("selection_type")
            frames.append(sub[[
                "uniprot_id", "dms_id", "taxon", "wt_aa", "position", "mut_aa",
                "mutant", "DMS_score", "DMS_score_bin",
                "first_author", "year", "selection_type",
            ]])
    if not frames:
        raise RuntimeError("No valid ProteinGym DMS assay files were parsed.")
    out = pd.concat(frames, ignore_index=True)
    out["hgvs_p"] = (
        "p."
        + out["wt_aa"].map(ONE_TO_THREE)
        + out["position"].astype(int).astype(str)
        + out["mut_aa"].map(ONE_TO_THREE)
    )
    out = out.rename(columns={"DMS_score": "dms_score", "DMS_score_bin": "dms_bin"})
    out["dms_score"] = pd.to_numeric(out["dms_score"], errors="coerce")
    out["dms_bin"] = pd.to_numeric(out["dms_bin"], errors="coerce").astype("Int64")
    missing_score = out["dms_score"].isna()
    if missing_score.any():
        logger.warning("Dropping %d DMS rows without a numeric assay score.",
                       int(missing_score.sum()))
        out = out.loc[~missing_score].copy()
    invalid_bin = out["dms_bin"].notna() & ~out["dms_bin"].isin([0, 1])
    if invalid_bin.any():
        logger.warning("Dropping %d DMS rows with non-binary score bins.",
                       int(invalid_bin.sum()))
        out = out.loc[~invalid_bin].copy()
    # An assay should contribute at most one measurement per substitution.
    # Exact duplicates are harmless; discordant duplicates are not resolved by
    # row order and are removed from the training source.
    assay_key = ["dms_id", "wt_aa", "position", "mut_aa"]
    conflicting = (out.groupby(assay_key, dropna=False)["dms_bin"].nunique(dropna=True) > 1)
    if conflicting.any():
        bad = conflicting[conflicting].reset_index()[assay_key]
        out = out.merge(bad.assign(_drop=True), on=assay_key, how="left")
        logger.warning("Dropping %d DMS rows with within-assay label conflicts.",
                       int(out["_drop"].notna().sum()))
        out = out[out["_drop"].isna()].drop(columns="_drop")
    out = out.drop_duplicates(subset=assay_key, keep="first").reset_index(drop=True)
    logger.info("ProteinGym DMS table: %d substitutions across %d assays",
                len(out), out["dms_id"].nunique())
    return out


def load_clinical_benchmark(raw_dir: Path) -> pd.DataFrame:
    """Load every clinical-benchmark CSV -> one long dataframe.

    Output columns::

        np_accession, position, wt_aa, mut_aa, mutant, clinical_label

    ``np_accession`` is the RefSeq protein accession (e.g. ``NP_000007.1``);
    mapping to UniProt happens separately in :func:`map_np_to_panel`.
    """
    pgf = download_proteingym(raw_dir)
    frames: List[pd.DataFrame] = []
    with zipfile_ctx(pgf.clinical_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            df = pd.read_csv(zf.open(name))
            required = {"mutant", "protein_sequence", "DMS_bin_score"}
            missing_cols = required - set(df.columns)
            if df.empty or missing_cols:
                logger.warning("Skipping clinical benchmark %s: missing %s", name,
                               sorted(missing_cols))
                continue
            acc = Path(name).stem
            parsed = df["mutant"].map(parse_mutant_token)
            ok = parsed.notna()
            sub = df.loc[ok].copy()
            wpm = pd.DataFrame(parsed.loc[ok].tolist(),
                               columns=["wt_aa", "position", "mut_aa"])
            sub = pd.concat([sub.reset_index(drop=True), wpm], axis=1)
            sub["np_accession"] = acc
            sub["protein_sequence_hash"] = df["protein_sequence"].iloc[0]
            frames.append(sub[["np_accession", "wt_aa", "position", "mut_aa",
                               "mutant", "DMS_bin_score", "protein_sequence_hash"]])
    if not frames:
        raise RuntimeError("No valid ProteinGym clinical benchmark files were parsed.")
    out = pd.concat(frames, ignore_index=True)
    out["hgvs_p"] = (
        "p." + out["wt_aa"].map(ONE_TO_THREE)
        + out["position"].astype(int).astype(str)
        + out["mut_aa"].map(ONE_TO_THREE)
    )
    # DMS_bin_score holds string class labels ("Pathogenic"/"Benign").
    _label_map = {"pathogenic": 1, "likely pathogenic": 1,
                  "benign": 0, "likely benign": 0}
    raw_labels = (out["DMS_bin_score"].astype(str).str.strip().str.lower()
                  .map(_label_map))
    n_unmapped = int(raw_labels.isna().sum())
    if n_unmapped:
        logger.warning("Clinical benchmark: %d rows with unmapped label "
                       "strings; dropped.", n_unmapped)
    out["clinical_label"] = raw_labels.astype("Int64")
    # Keep one representative sequence per accession for the mapping step,
    # then drop the bulky column before returning.
    seq_map = out.groupby("np_accession")["protein_sequence_hash"].first()
    out = out.drop(columns=["DMS_bin_score", "protein_sequence_hash"])
    out = out.dropna(subset=["clinical_label"]).reset_index(drop=True)
    key = ["np_accession", "wt_aa", "position", "mut_aa"]
    conflict = out.groupby(key)["clinical_label"].nunique() > 1
    if conflict.any():
        bad = conflict[conflict].reset_index()[key]
        out = out.merge(bad.assign(_drop=True), on=key, how="left")
        logger.warning("Dropping %d conflicting clinical-benchmark rows.",
                       int(out["_drop"].notna().sum()))
        out = out[out["_drop"].isna()].drop(columns="_drop")
    out = out.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    out.attrs["np_sequences"] = seq_map.to_dict()
    logger.info("ProteinGym clinical benchmark: %d variants across %d proteins",
                len(out), out["np_accession"].nunique())
    return out


def load_zero_shot_scores(raw_dir: Path) -> pd.DataFrame:
    """Long table of published-model scores for the clinical benchmark.

    Columns: ``np_accession, wt_aa, position, mut_aa, mutant, <model...>``
    where ``<model>`` spans :data:`ZERO_SHOT_MODEL_COLUMNS` (only those
    actually present are kept).
    """
    pgf = download_proteingym(raw_dir)
    frames: List[pd.DataFrame] = []
    with zipfile_ctx(pgf.zeroshot_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            df = pd.read_csv(zf.open(name))
            if df.empty or "mutant" not in df.columns:
                logger.warning("Skipping zero-shot file %s: missing mutant column.", name)
                continue
            cols_present = [c for c in ZERO_SHOT_MODEL_COLUMNS if c in df.columns]
            missing = set(ZERO_SHOT_MODEL_COLUMNS) - set(cols_present)
            if missing:
                logger.debug("%s missing model columns: %s", name, sorted(missing))
            parsed = df["mutant"].map(parse_mutant_token)
            ok = parsed.notna()
            sub = df.loc[ok].copy()
            wpm = pd.DataFrame(parsed.loc[ok].tolist(),
                               columns=["wt_aa", "position", "mut_aa"])
            sub = pd.concat([sub.reset_index(drop=True), wpm], axis=1)
            sub["np_accession"] = Path(name).stem
            frames.append(sub.rename(columns={
                c: model_col_name(c) for c in cols_present}))
    if not frames:
        raise RuntimeError("No valid ProteinGym zero-shot score files were parsed.")
    out = pd.concat(frames, ignore_index=True)
    score_cols = [c for c in out.columns if c.startswith("zs_")]
    out[score_cols] = out[score_cols].apply(pd.to_numeric, errors="coerce")
    # Duplicate score rows should agree. Keep the median only after rejecting
    # per-model disagreement that would otherwise depend on archive ordering.
    key = ["np_accession", "mutant"]
    if out.duplicated(subset=key).any():
        out = out.groupby(key, as_index=False)[score_cols].median()
    logger.info("Zero-shot clinical scores: %d rows, %d models",
                len(out), len(out.columns) - 2 - 3)
    return out


def model_col_name(model: str) -> str:
    """Normalise a raw column header into a snake_case ``zs_*`` feature name."""
    key = (model.strip()
           .replace(" (HDIV)", "_hdiv").replace(" (HVAR)", "_hvar")
           .replace(" (addAF)", "_addaf").replace(" (noAF)", "_noaf")
           .replace("-", "_").replace(".", "").replace(" ", "_"))
    return "zs_" + "".join(ch for ch in key.lower() if ch.isalnum() or ch == "_")


# --------------------------------------------------------------------------- #
# RefSeq NP_ -> panel mapping by exact sequence identity
# --------------------------------------------------------------------------- #
def map_np_to_panel(
    np_sequences: Dict[str, str],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Map RefSeq NP accessions onto panel proteins by exact sequence match.

    Parameters
    ----------
    np_sequences:
        ``{np_accession: protein_sequence}``.
    panel:
        Table with unique rows ``uniprot_id -> sequence`` (one per panel
        protein; canonical sequences).

    Returns
    -------
    DataFrame ``[np_accession, uniprot_id, gene]`` restricted to *exact*
    whole-protein matches.  Exactness matters: residue numbering of the
    clinical benchmark must equal canonical UniProt numbering for positional
    joins to be valid, so near-matches are deliberately discarded.
    """
    seq_lookup = {seq: (acc, gene) for acc, gene, seq in
                  panel[["uniprot_id", "gene", "sequence"]].itertuples(index=False)}
    rows = []
    for np_acc, seq in np_sequences.items():
        hit = seq_lookup.get(seq)
        if hit is not None:
            rows.append({"np_accession": np_acc, "uniprot_id": hit[0],
                         "gene": hit[1]})
    mapped = pd.DataFrame(rows)
    logger.info("RefSeq->panel mapping: %d/%d NP accessions matched exactly.",
                len(mapped), len(np_sequences))
    return mapped


# --------------------------------------------------------------------------- #
# UniProt entry-name -> accession resolution
# --------------------------------------------------------------------------- #
def resolve_uniprot_entry_names(
    entry_names: Sequence[str],
    raw_dir: Path,
    session: Optional[requests.Session] = None,
) -> Dict[str, str]:
    """Map UniProt entry names (``BRCA1_HUMAN``) to primary accessions.

    The ProteinGym reference file's ``UniProt_ID`` column contains entry names
    rather than accessions, so a translation step is needed before filtering
    against our accession-keyed panel. Results are batched (OR queries of 40)
    and cached in ``raw/uniprot/entry_name_map.json``.
    """
    out_dir = raw_dir / "uniprot"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "entry_name_map.json"
    mapping: Dict[str, str] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted({n for n in entry_names if n and n not in mapping})
    if not todo:
        return mapping

    sess = session or make_session()
    CHUNK = 40
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        query = " OR ".join(f"id:{n}" for n in chunk)
        params = {
            "query": f"({query}) AND (organism_id:9606) AND (reviewed:true)",
            "format": "json",
            "fields": "accession,id",
            "size": str(CHUNK),
        }
        resp = sess.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        for res in resp.json().get("results", []):
            name = res.get("uniProtkbId")
            if name:
                mapping[name] = res["primaryAccession"]
    missing = [n for n in todo if n not in mapping]
    if missing:
        logger.warning("No reviewed human entry found for %d UniProt names, "
                       "e.g. %s", len(missing), missing[:5])
    cache.write_text(json.dumps(mapping, indent=2))
    logger.info("Resolved %d/%d UniProt entry names to accessions.",
                len(mapping), len(set(entry_names)))
    return mapping


# --------------------------------------------------------------------------- #
# 5. AlphaMissense streaming filter
# --------------------------------------------------------------------------- #
def download_alphamissense(raw_dir: Path, overwrite: bool = False) -> Path:
    """Download the full aa-substitution table (~1.2 GB, cached)."""
    out_dir = raw_dir / "alphamissense"
    dest = out_dir / "AlphaMissense_aa_substitutions.tsv.gz"
    if overwrite and dest.exists():
        dest.unlink()
    # ~1.2 GB; same reasoning as the ClinVar table -- budget one resume
    # attempt per expected connection drop rather than the default five.
    return download_file(ALPHAMISSENSE_AA_URL, dest, min_bytes=100_000_000,
                         max_attempts=120)


def stream_filter_alphamissense(gz_path: Path, out_path: Path,
                                accessions: Iterable[str]) -> pd.DataFrame:
    """Stream-gzip scan keeping rows whose UniProt id is in *accessions*.

    The AlphaMissense table has no header row inside the archive; the first
    lines are licence comments starting with ``#``.  Data lines are
    tab-separated: ``uniprot_id<TAB>variant<TAB>score<TAB>class``.

    The filtered output is cached as TSV with header
    ``uniprot_id,mutant,am_pathogenicity,am_class``.
    """
    wanted = set(accessions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        df = pd.read_csv(out_path)
        if wanted <= set(df["uniprot_id"].unique()):
            logger.info("Using cached AlphaMissense subset %s", out_path)
            return df
        logger.info("Cached AlphaMissense subset misses some accessions; rescanning.")

    n_seen = n_kept = 0
    records: List[List] = []
    with gzip.open(gz_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            n_seen += 1
            if parts[0] in wanted:
                token = parse_mutant_token(parts[1])
                if token is None:
                    continue
                wt, pos, mut = token
                try:
                    score = float(parts[2])
                except ValueError:
                    continue
                records.append([parts[0], parts[1], wt, pos, mut, score, parts[3]])
                n_kept += 1
    df = pd.DataFrame(records, columns=[
        "uniprot_id", "mutant", "wt_aa", "position", "mut_aa",
        "am_pathogenicity", "am_class"])
    df["hgvs_p"] = (
        "p." + df["wt_aa"].map(ONE_TO_THREE)
        + df["position"].astype(int).astype(str)
        + df["mut_aa"].map(ONE_TO_THREE)
    )
    df.to_csv(out_path, index=False)
    logger.info("AlphaMissense scan: saw %d lines, kept %d for %d accessions -> %s",
                n_seen, n_kept, len(wanted), out_path)
    return df


# --------------------------------------------------------------------------- #
# 5. UniProt domain annotations
# --------------------------------------------------------------------------- #
def fetch_uniprot_domains(accessions: Sequence[str], raw_dir: Path,
                          session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Domain/region intervals per accession (cached JSON per accession).

    Returns tidy frame: ``[uniprot_id, feature_type, description, start, end]``
    with 1-based inclusive coordinates.
    """
    out_dir = raw_dir / "uniprot_domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = session or make_session()
    rows = []
    for acc in accessions:
        cache = out_dir / f"{acc}.json"
        if not cache.exists():
            resp = sess.get(UNIPROT_DOMAINS_URL.format(accession=acc), timeout=60)
            resp.raise_for_status()
            cache.write_text(resp.text)
        payload = json.loads(cache.read_text())
        feats = payload.get("features", [])
        kept = 0
        for feat in feats:
            if feat.get("type") in {"Domain", "Region", "Family", "Coiled coil",
                                    "Compositional bias"}:
                loc = feat.get("location", {})
                start = loc.get("start", {}).get("value")
                end = loc.get("end", {}).get("value")
                if start is None or end is None:
                    continue
                desc = feat.get("description", "")
                rows.append({"uniprot_id": acc, "feature_type": feat["type"],
                             "description": desc, "start": int(start), "end": int(end)})
                kept += 1
        logger.debug("%s: %d interval features kept", acc, kept)
    return pd.DataFrame(rows, columns=["uniprot_id", "feature_type",
                                       "description", "start", "end"])


# --------------------------------------------------------------------------- #
# 5b. UniProt point features (active/binding sites, PTMs, disulfides)
# --------------------------------------------------------------------------- #
#: Point (single-residue) feature types worth a "functional site" flag --
#: distinct from the interval Domain/Region features already captured by
#: fetch_uniprot_domains. A substitution AT one of these positions is
#: mechanistically much more likely to be damaging regardless of domain
#: membership (e.g. a catalytic residue, a phosphosite, a disulfide-forming
#: cysteine) than the coarser domain/region annotation alone would suggest.
POINT_FEATURE_TYPES = (
    "Active site", "Binding site", "Site", "Metal binding",
    "Modified residue", "Disulfide bond", "Cross-link",
)


def fetch_uniprot_point_features(accessions: Sequence[str], raw_dir: Path,
                                 session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Single-residue functional-site annotations per accession.

    Reuses the same per-accession UniProt JSON cache as
    :func:`fetch_uniprot_domains` (``data/raw/uniprot_domains/{ACC}.json``) --
    both feature families come from one REST call, so this never re-fetches
    if the domain cache already exists.

    Returns tidy frame ``[uniprot_id, position, feature_type, description]``
    (one row per point feature; ``Disulfide bond`` -- a two-residue feature --
    contributes a row at both endpoints).
    """
    out_dir = raw_dir / "uniprot_domains"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = session or make_session()
    rows = []
    for acc in accessions:
        cache = out_dir / f"{acc}.json"
        if not cache.exists():
            resp = sess.get(UNIPROT_DOMAINS_URL.format(accession=acc), timeout=60)
            resp.raise_for_status()
            cache.write_text(resp.text)
        payload = json.loads(cache.read_text())
        for feat in payload.get("features", []):
            ftype = feat.get("type")
            if ftype not in POINT_FEATURE_TYPES:
                continue
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start is None:
                continue
            desc = feat.get("description", "")
            positions = {int(start)} if end is None else {int(start), int(end)}
            for pos in positions:
                rows.append({"uniprot_id": acc, "position": pos,
                            "feature_type": ftype, "description": desc})
    return pd.DataFrame(rows, columns=["uniprot_id", "position", "feature_type", "description"])


def attach_functional_site_features(master: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_functional_site`` (0/1) and ``functional_site_types`` per variant."""
    if not len(sites) or not len(master):
        out = master.copy()
        out["is_functional_site"] = 0
        out["functional_site_types"] = ""
        return out
    types_by_pos = (
        sites.groupby(["uniprot_id", "position"])["feature_type"]
        .agg(lambda s: "|".join(sorted(set(s))))
    )
    joined = master[["uniprot_id", "position"]].merge(
        types_by_pos.rename("functional_site_types").reset_index(),
        on=["uniprot_id", "position"], how="left")
    out = master.copy()
    out["is_functional_site"] = joined["functional_site_types"].notna().astype(int).to_numpy()
    out["functional_site_types"] = joined["functional_site_types"].fillna("").to_numpy()
    return out


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
class zipfile_ctx:
    """Thin wrapper so callers don't need to import zipfile themselves."""

    def __init__(self, path: Path) -> None:
        import zipfile
        self._zf = zipfile.ZipFile(path)

    def __enter__(self):
        return self._zf

    def __exit__(self, *exc) -> None:
        self._zf.close()


__all__ = [
    "download_file", "sha256_of",
    "ProteinGymFiles", "download_proteingym",
    "load_dms_assays", "load_clinical_benchmark", "load_zero_shot_scores",
    "parse_mutant_token",
    "map_np_to_panel",
    "download_alphamissense", "stream_filter_alphamissense",
    "fetch_uniprot_domains",
    "POINT_FEATURE_TYPES", "fetch_uniprot_point_features",
    "attach_functional_site_features",
    "ZERO_SHOT_MODEL_COLUMNS",
]
