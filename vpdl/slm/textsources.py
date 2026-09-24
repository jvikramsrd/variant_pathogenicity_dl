"""Readers for the free, licence-clear text sources that widen the pretraining corpus.

Every reader yields corpus documents in the format ``vpdl.slm.corpus`` writes
— ``{"id", "source", "text"}`` plus provenance fields (``licence``, ``pmid``
where one exists) — and counts every decision in a ``Counter``, so a build can
say what it kept and why it dropped the rest.

    source                file                                   licence
    medlineplus_genetics  ghr-summaries.xml                      public domain
    orphanet              en_product1.xml (Orphadata)            CC BY 4.0 (stated in the file)
    mondo                 mondo.obo                              CC BY 4.0
    uniprot_text          UniProt REST TSV (function, disease)   CC BY 4.0
    pmc                   PMC article datasets on AWS            per article; CC0 / CC BY / CC BY-SA kept

Formats were checked against the providers on 2026-09-24 (docs/slm/TEXT_SOURCES.md).
None of these sources carries variant-level classifications, with one exception:
PMC articles can describe individual variants and their interpretation, so PMC
documents keep their PMID and the pretraining corpus drops those cited for
evaluation variants (``vpdl.slm.build.roles.pretraining_exclusions``).
"""

from __future__ import annotations

import csv
import gzip
import html
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

__all__ = ["TEXT_SOURCES", "iter_medlineplus", "iter_orphanet", "iter_mondo", "iter_uniprot_text",
           "iter_pmc", "parse_jats", "normalise_licence", "ALLOWED_PMC_LICENCES",
           "iter_text_sources"]

# Licences whose articles may train a model that is later shared: no NC (non-commercial)
# and no ND (no derivatives). Author manuscripts ("TDM") are text-mining only: excluded.
ALLOWED_PMC_LICENCES = frozenset({"cc0", "ccby", "ccbysa"})

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_EVIDENCE = re.compile(r"\s*\{ECO:[^}]*\}")          # UniProt evidence tags


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _clean(text: str | None) -> str:
    """Unescape entities, drop markup, collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(_TAG.sub(" ", html.unescape(text)))
    return _SPACE.sub(" ", text).strip()


def _open(path: Path | str):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


# -- MedlinePlus Genetics ------------------------------------------------------------------------

def iter_medlineplus(path: Path | str, stats: Counter) -> Iterator[dict[str, Any]]:
    """``ghr-summaries.xml``: one document per condition, gene, chromosome or mtDNA summary."""
    with _open(path) as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, element in context:
            kind = _local(element.tag)
            if not kind.endswith("-summary") or kind == "summaries":
                continue
            name = _clean(element.findtext("{*}name"))
            symbol = _clean(element.findtext("{*}gene-symbol"))
            sections = []
            for text in element.iterfind("{*}text-list/{*}text"):
                role = _clean(text.findtext("{*}text-role"))
                node = text.find("{*}html")
                body = _clean("".join(node.itertext())) if node is not None else ""
                if body:
                    sections.append(f"{role.replace('-', ' ').capitalize()}: {body}" if role else body)
            key = element.get("id") or symbol or name
            element.clear()
            if not sections:
                stats["medlineplus_no_text"] += 1
                continue
            heading = f"{symbol} ({name})" if symbol and name else (symbol or name)
            stats["medlineplus_kept"] += 1
            yield {"id": f"medlineplus:{kind.removesuffix('-summary')}:{key}",
                   "source": "medlineplus_genetics", "licence": "public domain",
                   "text": "\n".join([heading, *sections])}


# -- Orphanet ------------------------------------------------------------------------------------

def iter_orphanet(path: Path | str, stats: Counter) -> Iterator[dict[str, Any]]:
    """Orphadata ``en_product1.xml``: disorder name + its text sections (the definition)."""
    licence = None
    with _open(path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            tag = _local(element.tag)
            if tag == "ShortIdentifier" and licence is None:
                licence = (element.text or "").strip()
                if licence and licence != "CC-BY-4.0":
                    raise ValueError(f"{path}: licence {licence!r}, expected CC-BY-4.0 — check the "
                                     "terms before using this release")
            if tag != "Disorder":
                continue
            code = (element.findtext("{*}OrphaCode") or "").strip()
            name = _clean(element.findtext("{*}Name"))
            sections = []
            for section in element.iterfind(".//{*}TextSection"):
                if (section.get("lang") or "en") != "en":
                    continue
                label = _clean(section.findtext("{*}TextSectionType/{*}Name"))
                body = _clean(section.findtext("{*}Contents"))
                if body:
                    sections.append(f"{label}: {body}" if label else body)
            element.clear()
            if not code or not sections:
                stats["orphanet_no_text"] += 1
                continue
            stats["orphanet_kept"] += 1
            yield {"id": f"orpha:{code}", "source": "orphanet", "licence": licence or "CC-BY-4.0",
                   "text": "\n".join([name, *sections])}


# -- MONDO ---------------------------------------------------------------------------------------

_OBO_DEF = re.compile(r'^def:\s*"((?:[^"\\]|\\.)*)"')
_OBO_SYNONYM = re.compile(r'^synonym:\s*"((?:[^"\\]|\\.)*)"\s+EXACT')


def iter_mondo(path: Path | str, stats: Counter) -> Iterator[dict[str, Any]]:
    """``mondo.obo``: name, exact synonyms and textual definition of each live term."""
    term: dict[str, Any] | None = None

    def finish(current):
        if current is None or not current.get("id", "").startswith("MONDO:"):
            return None
        if current.get("obsolete"):
            stats["mondo_obsolete"] += 1
            return None
        if not current.get("definition"):
            stats["mondo_no_definition"] += 1
            return None
        stats["mondo_kept"] += 1
        synonyms = f" Also known as: {'; '.join(current['synonyms'])}." if current["synonyms"] else ""
        return {"id": current["id"].lower(), "source": "mondo", "licence": "CC BY 4.0",
                "text": f"{current.get('name', '')}.{synonyms}\n{current['definition']}"}

    with _open(path) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("["):
                document = finish(term)
                if document:
                    yield document
                term = {"synonyms": []} if line == "[Term]" else None
                continue
            if term is None or ":" not in line:
                continue
            if line.startswith("id: "):
                term["id"] = line[4:].strip()
            elif line.startswith("name: "):
                term["name"] = line[6:].strip()
            elif line.startswith("is_obsolete: true"):
                term["obsolete"] = True
            elif match := _OBO_DEF.match(line):
                term["definition"] = match.group(1).replace('\\"', '"')
            elif match := _OBO_SYNONYM.match(line):
                term["synonyms"].append(match.group(1).replace('\\"', '"'))
        document = finish(term)
        if document:
            yield document


# -- UniProt function / disease text -------------------------------------------------------------

def iter_uniprot_text(path: Path | str, stats: Counter) -> Iterator[dict[str, Any]]:
    """UniProt REST TSV with columns Entry, Gene Names (primary), Protein names,
    Function [CC], Involvement in disease. Evidence tags ``{ECO:...}`` are removed."""
    with _open(path) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
        needed = {"Entry", "Function [CC]"}
        if not needed <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: columns {reader.fieldnames}; expected at least {sorted(needed)} "
                             "(download with fields=accession,gene_primary,protein_name,cc_function,"
                             "cc_disease&format=tsv)")
        for row in reader:
            parts = [_EVIDENCE.sub("", row.get(column) or "").strip()
                     for column in ("Function [CC]", "Involvement in disease")]
            parts = [p for p in parts if p]
            if not parts:
                stats["uniprot_no_text"] += 1
                continue
            heading = " ".join(x for x in (row.get("Gene Names (primary)"), f"({row.get('Protein names')})"
                                           if row.get("Protein names") else "") if x)
            stats["uniprot_kept"] += 1
            yield {"id": f"uniprot:{row['Entry']}", "source": "uniprot_text", "licence": "CC BY 4.0",
                   "text": "\n".join([heading, *parts]) if heading else "\n".join(parts)}


# -- PubMed Central (JATS) -----------------------------------------------------------------------

def normalise_licence(code: str | None) -> str:
    """'CC BY', 'cc-by', 'CC BY 4.0', 'by' -> 'ccby'; 'CC0', 'CC0 1.0' -> 'cc0'; 'TDM' -> 'tdm'."""
    text = re.sub(r"[^a-z0-9]", "", (code or "").lower())
    if text.startswith("cc0") or text in ("0", "zero", "cczero", "publicdomain", "pd"):
        return "cc0"
    text = re.sub(r"\d+$", "", text)                 # version: ccby40 -> ccby
    return "cc" + text if text.startswith("by") else text


# Front-matter and floating material that is not running text.
_JATS_SKIP = frozenset({"ref-list", "table-wrap", "fig", "fig-group", "table", "disp-formula",
                        "inline-formula", "supplementary-material", "graphic", "media",
                        "xref", "ack", "fn-group", "app-group", "glossary", "notes", "tex-math",
                        "math", "alternatives", "object-id", "label"})


def _jats_text(element: ET.Element) -> str:
    """Running text of one element, without citations, floats, formulae or labels."""
    parts: list[str] = [element.text or ""]

    def walk(node: ET.Element) -> None:
        for child in node:
            if _local(child.tag) not in _JATS_SKIP:
                parts.append(child.text or "")
                walk(child)
            parts.append(child.tail or "")

    walk(element)
    return _SPACE.sub(" ", "".join(parts)).strip()


_JATS_CONTAINERS = frozenset({"sec", "boxed-text", "disp-quote", "list", "list-item", "def-list",
                              "def-item", "def", "statement"})


def _walk_body(node: ET.Element, out: list[str]) -> None:
    for child in node:
        name = _local(child.tag)
        if name in _JATS_SKIP:
            continue
        if name == "title":
            text = _jats_text(child)
            if text:
                out.append(f"\n{text}")
        elif name == "p":
            text = _jats_text(child)
            if text:
                out.append(text)
        elif name in _JATS_CONTAINERS:
            _walk_body(child, out)


def parse_jats(data: bytes) -> dict[str, Any]:
    """Title, abstract, section-structured body and article type of one JATS article."""
    root = ET.fromstring(data)
    article = root if _local(root.tag) == "article" else root.find(".//{*}article")
    if article is None:
        raise ET.ParseError("no <article> element")
    meta = article.find("{*}front/{*}article-meta")
    title_node = meta.find("{*}title-group/{*}article-title") if meta is not None else None
    abstracts = meta.findall("{*}abstract") if meta is not None else []
    body_parts: list[str] = []
    body = article.find("{*}body")
    if body is not None:
        _walk_body(body, body_parts)
    return {"title": _jats_text(title_node) if title_node is not None else "",
            "abstract": " ".join(_jats_text(a) for a in abstracts if not a.get("abstract-type")),
            "body": "\n".join(body_parts).strip(),
            "article_type": article.get("article-type", "")}


def iter_pmc(pmc_dir: Path | str, stats: Counter,
             licences: Iterable[str] = ALLOWED_PMC_LICENCES,
             include_abstract_when_in_pubmed: bool = False) -> Iterator[dict[str, Any]]:
    """Articles written by ``vpdl-slm pmc-download``: ``<shard>/PMCnnn.v.json`` + ``.xml.gz``.

    The metadata's licence and retraction flag are checked again here (the
    download already filtered), so a directory filled by other means is held to
    the same rule. An article with a PMID drops its abstract by default: the
    PubMed reader already carries that abstract, and one text twice is noise.
    """
    allowed = set(licences)
    for meta_path in sorted(Path(pmc_dir).glob("*/PMC*.json")):
        xml_path = meta_path.with_suffix(".xml.gz")
        if not xml_path.exists():
            stats["pmc_missing_xml"] += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        licence = normalise_licence(meta.get("license_code"))
        if licence not in allowed:
            stats[f"pmc_licence_excluded_{licence or 'none'}"] += 1
            continue
        if str(meta.get("is_retracted", "")).lower() in ("yes", "true", "1"):
            stats["pmc_retracted"] += 1
            continue
        try:
            parsed = parse_jats(gzip.decompress(xml_path.read_bytes()))
        except (ET.ParseError, OSError, EOFError) as error:
            stats["pmc_unparseable"] += 1
            stats[f"pmc_unparseable_{type(error).__name__}"] += 1
            continue
        pmid = str(meta.get("pmid") or "").strip() or None
        parts = [parsed["title"]]
        if parsed["abstract"] and (include_abstract_when_in_pubmed or not pmid):
            parts.append(parsed["abstract"])
        if parsed["body"]:
            parts.append(parsed["body"])
        text = "\n".join(p for p in parts if p).strip()
        if len(text) < 200:
            stats["pmc_too_short"] += 1
            continue
        stats["pmc_kept"] += 1
        stats["pmc_characters"] += len(text)
        yield {"id": f"pmc:{meta.get('pmcid') or meta_path.stem.split('.')[0]}", "source": "pmc",
               "licence": licence, "pmid": pmid, "article_type": parsed["article_type"],
               "text": text}


# -- dispatch ------------------------------------------------------------------------------------

TEXT_SOURCES: dict[str, tuple[Callable[..., Iterator[dict[str, Any]]], str]] = {
    "medlineplus": (iter_medlineplus, "https://medlineplus.gov/download/ghr-summaries.xml"),
    "orphanet": (iter_orphanet, "https://www.orphadata.com/data/xml/en_product1.xml"),
    "mondo": (iter_mondo, "http://purl.obolibrary.org/obo/mondo.obo"),
    "uniprot_text": (iter_uniprot_text, "https://rest.uniprot.org/uniprotkb/stream (TSV)"),
    "pmc": (iter_pmc, "s3://pmc-oa-opendata via `vpdl-slm pmc-download`"),
}


def iter_text_sources(paths: Mapping[str, Path | str | None], stats: Counter) -> Iterator[dict[str, Any]]:
    """Documents from every source named in `paths` (``{name: file or directory}``), in a fixed order."""
    unknown = set(paths) - set(TEXT_SOURCES)
    if unknown:
        raise ValueError(f"unknown text sources {sorted(unknown)}; known {sorted(TEXT_SOURCES)}")
    for name in TEXT_SOURCES:
        path = paths.get(name)
        if not path:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"{name}: {path} does not exist")
        reader, _ = TEXT_SOURCES[name]
        yield from reader(path, stats)
