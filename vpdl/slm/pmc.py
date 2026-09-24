"""PubMed Central full text for pretraining: search, pick a licence-clear version, fetch — resumably.

Since August 2026 PMC's article datasets exist only as a public S3 bucket
(``pmc-oa-opendata``): one prefix per article version holding ``.json``
metadata (licence, retraction flag, PMID) and the JATS ``.xml``. There are no
bulk tarballs any more (https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/, read
2026-09-24). So:

1. **Search** — NCBI ESearch on ``db=pmc`` with a topic query and a licence
   filter. ESearch returns at most 9,999 ids per query, so the PMC release-date
   range is split in halves until every piece fits.
2. **Choose a version** — list the article's versions, read each version's
   metadata, keep one whose licence is CC0 / CC BY / CC BY-SA and which is not
   retracted, preferring the published version over an author manuscript.
3. **Fetch** — the XML, gzipped, beside its metadata:
   ``<out>/<last two digits>/PMC<n>.<v>.json`` and ``.xml.gz``.

Re-running skips what is already on disk, so an interrupted download resumes.
Every decision is counted in ``<out>/manifest.json``. No e-mail address or
other personal detail is sent; set ``NCBI_API_KEY`` for NCBI's higher rate.

NLM's terms: acknowledge NLM as the source (the manifest records the citation
text), do not use the PMC wordmark, redistribute only licensed data.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from vpdl.slm.textsources import ALLOWED_PMC_LICENCES, normalise_licence

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TOPIC", "LICENCE_FILTER", "search_ids", "choose_version", "fetch_article",
           "download", "NotFound", "CITATION"]

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
MAX_IDS = 9999
CITATION = ("NIH NLM NCBI PubMed Central (PMC) Article Datasets - Full-Text Biomedical and Life "
            "Sciences Journal Articles on AWS, accessed on {date} from "
            "https://registry.opendata.aws/ncbi-pmc.")

# Medical genetics, broadly: germline variants, their interpretation, testing and the
# conditions they cause. 421,690 CC0/CC BY/CC BY-SA open-access articles on 2026-09-24.
DEFAULT_TOPIC = ('("germline"[Abstract] OR "pathogenic variant"[Abstract] OR "pathogenic variants"'
                 '[Abstract] OR "likely pathogenic"[Abstract] OR "variant of uncertain significance"'
                 '[Abstract] OR "variants of uncertain significance"[Abstract] OR "missense variant"'
                 '[Abstract] OR "hereditary"[Abstract] OR "genetic testing"[Abstract] OR '
                 '"Lynch syndrome"[Abstract] OR "mismatch repair"[Abstract] OR "ACMG"[Abstract] OR '
                 '"exome sequencing"[Abstract] OR "genetic variant"[Abstract] OR '
                 '"rare disease"[Abstract])')
LICENCE_FILTER = ('(cc0_license[Filter] OR cc_by_license[Filter] OR cc_by-sa_license[Filter]) '
                  'AND open_access[Filter]')


class NotFound(Exception):
    pass


class Http:
    """GET with retries and back-off; NCBI calls are paced to NCBI's published rate."""

    def __init__(self, api_key: str | None = None, timeout: float = 60.0, retries: int = 5):
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        self._last_ncbi = 0.0
        self._interval = 0.11 if self.api_key else 0.35

    def __call__(self, url: str) -> bytes:
        if url.startswith(ESEARCH):
            with self._lock:
                wait = self._interval - (time.monotonic() - self._last_ncbi)
                if wait > 0:
                    time.sleep(wait)
                self._last_ncbi = time.monotonic()
            if self.api_key:
                url += "&api_key=" + urllib.parse.quote(self.api_key)
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "vpdl-slm (research)"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code in (403, 404):
                    raise NotFound(url) from error
                if error.code not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
            time.sleep(min(60.0, 2.0 ** attempt))
        raise RuntimeError(f"unreachable: {url}")


def _esearch(http: Callable[[str], bytes], term: str, retmax: int) -> dict[str, Any]:
    url = (f"{ESEARCH}?db=pmc&format=json&tool=vpdl-slm&retmax={retmax}&term="
           + urllib.parse.quote(term))
    return json.loads(http(url))["esearchresult"]


def _dates(start: dt.date, end: dt.date) -> str:
    return f"{start:%Y/%m/%d}:{end:%Y/%m/%d}[pmcrdat]"      # the form PMC's README documents


def search_ids(query: str, http: Callable[[str], bytes], start: dt.date = dt.date(1990, 1, 1),
               end: dt.date | None = None) -> tuple[list[str], dict[str, int]]:
    """Every PMCID (as ``PMC<n>``) matching `query`, splitting the release-date range as needed."""
    end = end or dt.date.today()
    total = int(_esearch(http, query, 0)["count"])
    ids: list[str] = []
    pending = [(start, end)]
    requests = 1
    while pending:
        low, high = pending.pop()
        result = _esearch(http, f"({query}) AND {_dates(low, high)}", MAX_IDS)
        requests += 1
        count = int(result["count"])
        if count > MAX_IDS:
            if low == high:
                raise RuntimeError(f"{count} articles released on {low}; narrow the query")
            middle = low + (high - low) // 2
            pending += [(middle + dt.timedelta(days=1), high), (low, middle)]
            continue
        ids.extend(result["idlist"])
    unique = sorted({f"PMC{i}" for i in ids}, key=lambda s: int(s[3:]))
    return unique, {"query_total": total, "found_by_date": len(unique), "esearch_requests": requests}


def _versions(pmcid: str, http: Callable[[str], bytes]) -> list[str]:
    listing = http(f"{S3}/?list-type=2&delimiter=/&prefix={pmcid}.").decode("utf-8", "replace")
    return sorted(set(re.findall(r"<Prefix>(" + re.escape(pmcid) + r"\.\d+)/</Prefix>", listing)),
                  key=lambda v: int(v.rsplit(".", 1)[1]))


def choose_version(metas: Iterable[dict[str, Any]],
                   allowed: Iterable[str] = ALLOWED_PMC_LICENCES) -> tuple[dict[str, Any] | None, str]:
    """(metadata of the version to use, reason). Published before manuscript, newest first."""
    allowed = set(allowed)
    metas = list(metas)
    usable = [m for m in metas
              if normalise_licence(m.get("license_code")) in allowed
              and str(m.get("is_retracted", "")).lower() not in ("yes", "true", "1")]
    if not usable:
        if any(str(m.get("is_retracted", "")).lower() in ("yes", "true", "1") for m in metas):
            return None, "retracted"
        codes = sorted({normalise_licence(m.get("license_code")) or "none" for m in metas})
        return None, "licence:" + "+".join(codes)
    usable.sort(key=lambda m: (str(m.get("is_manuscript", "")).lower() in ("yes", "true", "1"),
                               -int(m.get("version") or 0)))
    return usable[0], "ok"


def fetch_article(pmcid: str, out_dir: Path, http: Callable[[str], bytes],
                  allowed: Iterable[str] = ALLOWED_PMC_LICENCES) -> tuple[str, str | None]:
    """(status, licence). Status: ok | exists | no_version | retracted | licence:<codes>."""
    shard = out_dir / pmcid[-2:]
    if any(shard.glob(f"{pmcid}.*.xml.gz")):
        return "exists", None
    versions = _versions(pmcid, http)
    if not versions:
        return "no_version", None
    metas = []
    for version in versions:
        try:
            meta = json.loads(http(f"{S3}/metadata/{version}.json"))
        except NotFound:
            continue
        meta["_version"] = version
        metas.append(meta)
    chosen, reason = choose_version(metas, allowed)
    if chosen is None:
        return reason, None
    version = chosen["_version"]
    xml = http(f"{S3}/{version}/{version}.xml")
    shard.mkdir(parents=True, exist_ok=True)
    temporary = shard / f".{version}.xml.gz.tmp"
    temporary.write_bytes(gzip.compress(xml))
    (shard / f"{version}.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    os.replace(temporary, shard / f"{version}.xml.gz")        # the XML last: its presence = done
    return "ok", normalise_licence(chosen.get("license_code"))


def download(out: Path | str, query: str | None = None, workers: int = 16, limit: int | None = None,
             http: Callable[[str], bytes] | None = None, refresh_ids: bool = False) -> dict[str, Any]:
    """Search once (ids cached in ``ids.txt``), then fetch every article not yet on disk."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    http = http or Http()
    query = query or f"{DEFAULT_TOPIC} AND {LICENCE_FILTER}"
    ids_path = out / "ids.txt"
    search: dict[str, Any] = {}
    if ids_path.exists() and not refresh_ids:
        ids = ids_path.read_text().split()
        search = {"reused_ids_file": str(ids_path)}
    else:
        ids, search = search_ids(query, http)
        ids_path.write_text("\n".join(ids) + "\n")
    if limit:
        ids = ids[:limit]
    started = time.time()
    counts: Counter = Counter()
    licences: Counter = Counter()
    lock = threading.Lock()
    last = [started]

    def one(pmcid: str) -> None:
        try:
            status, licence = fetch_article(pmcid, out, http)
        except Exception as error:                       # noqa: BLE001 — counted, logged, retried next run
            status, licence = f"error:{type(error).__name__}", None
            logger.warning("%s: %s", pmcid, error)
        with lock:
            counts[status if not status.startswith("licence:") else "licence_excluded"] += 1
            if licence:
                licences[licence] += 1
            if time.time() - last[0] > 30:
                last[0] = time.time()
                done = sum(counts.values())
                logger.info("PMC: %d/%d articles (%s)", done, len(ids), dict(counts))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, ids))
    manifest = {"query": query, "search": search, "articles_requested": len(ids),
                "status": dict(counts), "licences_fetched": dict(licences),
                "allowed_licences": sorted(ALLOWED_PMC_LICENCES),
                "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.time() - started, 1),
                "citation": CITATION.format(date=dt.date.today().isoformat())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
