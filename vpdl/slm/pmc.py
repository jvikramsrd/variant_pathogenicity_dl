"""PubMed Central full text for pretraining: search, pick a licence-clear version, fetch — resumably.

Since August 2026 PMC's article datasets exist only as a public S3 bucket
(``pmc-oa-opendata``): one prefix per article version holding ``.json``
metadata (licence, retraction flag, PMID) and the JATS ``.xml``. There are no
bulk tarballs any more (https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/, read
2026-09-24). So:

1. **Search** — NCBI ESearch on ``db=pmc`` with a topic query and a licence
   filter. ESearch returns at most 9,999 ids per query, so the PMC release-date
   range is split in halves until every piece fits.
2. **Choose a version** — the bucket's daily inventory (read once) says which
   versions each article has; read each version's metadata, keep one whose
   licence is CC0 / CC BY / CC BY-SA and which is not retracted, preferring the
   published version over an author manuscript.

Speed comes from latency, not bandwidth: every worker thread keeps its HTTPS
connection open, and the inventory replaces one listing request per article,
so an article costs two small requests on a warm connection.
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
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from vpdl.slm.textsources import ALLOWED_PMC_LICENCES, normalise_licence

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TOPIC", "LICENCE_FILTER", "search_ids", "choose_version", "fetch_article",
           "download", "inventory_versions", "Http", "NotFound", "CITATION"]

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
    """GET with kept-alive connections, retries and back-off; NCBI calls paced to NCBI's rate.

    Each worker thread keeps one HTTPS connection per host open between requests.
    A fresh connection per request (what ``urllib.request`` does) costs a TCP and
    TLS handshake — several round trips to us-east-1 — before every small file,
    and that latency, not bandwidth, limited the download to ~5 articles/s.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 60.0, retries: int = 5):
        import ssl
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        self._last_ncbi = 0.0
        self._interval = 0.11 if self.api_key else 0.35
        self._local = threading.local()
        self._tls = ssl.create_default_context()

    def _connection(self, scheme: str, host: str):
        import http.client
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = self._local.pool = {}
        connection = pool.get((scheme, host))
        if connection is None:
            connection = (http.client.HTTPSConnection(host, timeout=self.timeout, context=self._tls)
                          if scheme == "https" else http.client.HTTPConnection(host, timeout=self.timeout))
            pool[(scheme, host)] = connection
        return connection

    def _drop(self, scheme: str, host: str) -> None:
        connection = getattr(self._local, "pool", {}).pop((scheme, host), None)
        if connection is not None:
            connection.close()

    def __call__(self, url: str) -> bytes:
        import http.client
        if url.startswith(ESEARCH):
            with self._lock:
                wait = self._interval - (time.monotonic() - self._last_ncbi)
                if wait > 0:
                    time.sleep(wait)
                self._last_ncbi = time.monotonic()
            if self.api_key:
                url += "&api_key=" + urllib.parse.quote(self.api_key)
        parts = urllib.parse.urlsplit(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        for attempt in range(self.retries):
            if attempt:
                time.sleep(min(60.0, 2.0 ** attempt))
            connection = self._connection(parts.scheme, parts.netloc)
            try:
                connection.request("GET", target, headers={"User-Agent": "vpdl-slm (research)"})
                response = connection.getresponse()
                data = response.read()
            except (http.client.HTTPException, OSError):
                # A kept-alive connection the server has closed fails here; reconnect and retry.
                self._drop(parts.scheme, parts.netloc)
                if attempt == self.retries - 1:
                    raise
                continue
            if response.will_close:
                self._drop(parts.scheme, parts.netloc)
            if response.status == 200:
                return data
            if response.status in (403, 404):
                raise NotFound(url)
            if response.status not in (429, 500, 502, 503, 504) or attempt == self.retries - 1:
                raise RuntimeError(f"HTTP {response.status} for {url}")
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


INVENTORY = "inventory-reports/pmc-oa-opendata/metadata/"


def inventory_versions(http: Callable[[str], bytes], wanted: Iterable[str],
                       workers: int = 8) -> tuple[dict[str, list[str]], str]:
    """``{pmcid: [version prefixes]}`` for the wanted articles, from the bucket's daily inventory.

    One pass over ~230 MB of CSV instead of one listing request per article; an
    article absent from the inventory has no version in the bucket. Returns the
    inventory date used.
    """
    import csv
    import io
    wanted = set(wanted)
    listing = http(f"{S3}/?list-type=2&delimiter=/&prefix={INVENTORY}").decode("utf-8", "replace")
    dates = sorted(re.findall(re.escape(INVENTORY) + r"(\d{4}-\d\d-\d\dT[\d-]+Z)/", listing),
                   reverse=True)
    manifest = None
    for date in dates[:3]:                      # the newest can still be being written
        try:
            manifest = json.loads(http(f"{S3}/{INVENTORY}{date}/manifest.json"))
            break
        except NotFound:
            continue
    if manifest is None:
        raise NotFound(f"{S3}/{INVENTORY}: no readable inventory manifest")

    def read(key: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        text = io.TextIOWrapper(io.BytesIO(gzip.decompress(http(f"{S3}/{key}"))), encoding="utf-8")
        for row in csv.reader(text):
            if len(row) < 2 or not row[1].startswith("metadata/"):
                continue
            version = row[1][len("metadata/"):].removesuffix(".json")
            pmcid = version.split(".", 1)[0]
            if pmcid in wanted:
                found.setdefault(pmcid, []).append(version)
        return found

    versions: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(read, [f["key"] for f in manifest["files"]]):
            for pmcid, found in part.items():
                versions.setdefault(pmcid, []).extend(found)
    for pmcid in versions:
        versions[pmcid] = sorted(set(versions[pmcid]), key=lambda v: int(v.rsplit(".", 1)[1]))
    return versions, dates[0] if dates else "unknown"


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
                  allowed: Iterable[str] = ALLOWED_PMC_LICENCES,
                  versions: list[str] | None = None) -> tuple[str, str | None]:
    """(status, licence). Status: ok | exists | no_version | retracted | licence:<codes>.

    `versions` from the inventory saves a listing request; None lists the bucket."""
    shard = out_dir / pmcid[-2:]
    if any(shard.glob(f"{pmcid}.*.xml.gz")):
        return "exists", None
    if versions is None:
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


def download(out: Path | str, query: str | None = None, workers: int = 64, limit: int | None = None,
             http: Callable[[str], bytes] | None = None, refresh_ids: bool = False,
             use_inventory: bool = True) -> dict[str, Any]:
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
    versions: dict[str, list[str]] | None = None
    if use_inventory:
        cache = out / "versions.json"
        if cache.exists() and not refresh_ids:
            versions = json.loads(cache.read_text())
            search["inventory"] = "reused versions.json"
        else:
            try:
                versions, date = inventory_versions(http, ids)
                cache.write_text(json.dumps(versions))
                search["inventory"] = date
                logger.info("inventory %s: %d of %d articles have a version in the bucket",
                            date, len(versions), len(ids))
            except Exception as error:               # noqa: BLE001 - fall back to listing each article
                logger.warning("inventory unavailable (%s); listing each article instead", error)
                search["inventory"] = f"unavailable: {error}"
    started = time.time()
    counts: Counter = Counter()
    licences: Counter = Counter()
    lock = threading.Lock()
    last = [started]

    def one(pmcid: str) -> None:
        try:
            status, licence = fetch_article(
                pmcid, out, http, versions=None if versions is None else versions.get(pmcid, []))
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
