"""GeneReviews chapters -> passages, following the chapters' own section structure.

Input is NCBI Bookshelf's open-access archive of the book (``gene_NBK1116.tar.gz``),
which ships every chapter as BITS XML — so passages follow real section
boundaries ("Management > Surveillance") instead of being cut from PDF text at
arbitrary lengths. Structure was read from the Lynch syndrome chapter
(``hnpcc.nxml``) on 2026-09-21.

Terms (from each chapter's own ``<permissions>`` block): non-commercial research
only; every copy credits https://www.genereviews.org and the University of
Washington copyright; a link to the original accompanies it; **no
modifications**. So passages keep the chapter's words as written. What changes
is layout only: tables become one line per row ("Header: value; ..."), lists
become "- item" lines, and superscripts are written as ``^x``.
"""

from __future__ import annotations

import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from vpdl.kb.chunks import Chunk

logger = logging.getLogger(__name__)

__all__ = ["read_chapter_ids", "iter_nxml", "parse_chapter", "load_genereviews"]

MAX_WORDS = 350

# Sections that are bibliography or editorial history, not clinical content.
# Skipping "References" also keeps 100+ citation strings from swamping search.
SKIP_SECTIONS = frozenset({
    "references", "literature cited", "chapter notes", "author notes",
    "author history", "revision history", "acknowledgments", "acknowledgements",
})

_BLOCK_TAGS = frozenset({"p", "list", "table-wrap", "sec", "ref-list", "title",
                         "fig", "boxed-text", "disp-quote"})
_SEPARATED = frozenset({"p", "list", "list-item", "break"})


def read_chapter_ids(path: Path | str) -> dict[str, tuple[str, str]]:
    """``{shortname: (title, NBK id)}`` from NCBI's ``GRtitle_shortname_NBKid.txt``.

    The chapter XML does not carry its own NBK id, and the id is what the
    required link to the original is built from.
    """
    mapping: dict[str, tuple[str, str]] = {}
    with Path(path).open("rb") as handle:
        for raw in handle:
            # NCBI's file mixes encodings: UTF-8 throughout except a few titles
            # in Latin-1 ("Cant\xfa syndrome", 2026-09-21). Decode line by line.
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode("cp1252", errors="replace")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[2].startswith("NBK"):
                mapping[fields[0]] = (fields[1], fields[2])
    return mapping


def iter_nxml(source: Path | str) -> Iterator[tuple[str, bytes]]:
    """Yield ``(filename stem, xml bytes)`` from the archive or an unpacked folder.

    The archive is read in place — nothing is extracted to disk.
    """
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.nxml")):
            yield path.stem, path.read_bytes()
        return
    with tarfile.open(source, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".nxml"):
                handle = archive.extractfile(member)
                if handle is not None:
                    yield Path(member.name).stem, handle.read()


def _clean(text: str) -> str:
    return " ".join(text.split())


def _inline(element: ET.Element) -> str:
    """Text of an element, leaving out nested blocks (they are rendered on their own)."""
    parts = [element.text or ""]
    for child in element:
        if child.tag == "break":
            parts.append(" ")
        elif child.tag == "sup":
            parts.append("^" + _clean(_inline(child)))
        elif child.tag not in _BLOCK_TAGS:
            parts.append(_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _flat(element: ET.Element) -> str:
    """All text of an element, with block boundaries kept as spaces (table cells)."""
    parts = [element.text or ""]
    for child in element:
        inner = _flat(child)
        if child.tag == "sup":
            inner = "^" + _clean(inner)
        parts.append(f" {inner} " if child.tag in _SEPARATED else inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _list_lines(list_element: ET.Element, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for item in list_element.findall("list-item"):
        first = True
        for child in item:
            if child.tag == "p":
                text = _clean(_inline(child))
                if text:
                    lines.append("  " * depth + ("- " if first else "  ") + text)
                    first = False
                for nested in child.findall("list"):
                    lines.extend(_list_lines(nested, depth + 1))
            elif child.tag == "list":
                lines.extend(_list_lines(child, depth + 1))
        if first and _clean(item.text or ""):
            lines.append("  " * depth + "- " + _clean(item.text or ""))
    return lines


def _grid(rows: list[ET.Element], repeat_colspan: bool) -> tuple[list[list[str]], list[dict]]:
    """Lay table rows out on a grid, honouring rowspan/colspan.

    A value spanning several rows is repeated into each, so every rendered row
    stands on its own ("Colorectal cancer" belongs to each of its rows). A value
    spanning several body columns is written once, and ``spans`` records how
    many columns it covers; a header spanning several columns labels each.
    """
    grid: list[list[str]] = []
    spans: list[dict[int, int]] = []
    carried: dict[int, tuple[str, int, int]] = {}
    for tr in rows:
        row: dict[int, str] = {}
        row_spans: dict[int, int] = {}
        for column, (value, left, span) in list(carried.items()):
            row[column] = value
            if span > 1:
                row_spans[column] = span
            if left <= 1:
                del carried[column]
            else:
                carried[column] = (value, left - 1, span)
        column = 0
        for cell in (c for c in tr if c.tag in ("th", "td")):
            while column in row:
                column += 1
            text = _clean(_flat(cell))
            colspan, rowspan = _span(cell.get("colspan")), _span(cell.get("rowspan"))
            for offset in range(colspan):
                value = text if (offset == 0 or repeat_colspan) else ""
                row[column + offset] = value
                if rowspan > 1:
                    carried[column + offset] = (value, rowspan - 1,
                                                colspan if offset == 0 else 1)
            if colspan > 1:
                row_spans[column] = colspan
            column += colspan
        grid.append([row.get(c, "") for c in range(max(row) + 1)] if row else [])
        spans.append(row_spans)
    return grid, spans


def _span(value: str | None) -> int:
    """A rowspan/colspan attribute as a count; anything unreadable counts as 1."""
    match = re.match(r"\s*(\d+)", value or "")
    return max(1, min(int(match.group(1)), 100)) if match else 1


def _span_header(labels: list[str]) -> str:
    """Headers of the columns one cell covers: "Frequency / TST1, TST2"."""
    labels = list(dict.fromkeys(label for label in labels if label))
    if len(labels) <= 1:
        return labels[0] if labels else ""
    parts = [label.split(" / ") for label in labels]
    common = []
    for level in zip(*parts):
        if len(set(level)) != 1:
            break
        common.append(level[0])
    rest = ", ".join(" / ".join(p[len(common):]) for p in parts)
    return f"{' / '.join(common)} / {rest}" if common else rest


def _table(table_wrap: ET.Element) -> tuple[str, list[str], list[str]]:
    """``(title, row lines, footnote lines)`` for one ``<table-wrap>``."""
    label = table_wrap.find("label")
    caption = " ".join(_inline(p) for p in table_wrap.findall("caption/p"))
    if not caption and table_wrap.find("caption/title") is not None:
        caption = _inline(table_wrap.find("caption/title"))
    title = _clean(f"{_inline(label) if label is not None else ''} {caption}")

    table = table_wrap.find(".//table")
    rows_out: list[str] = []
    if table is not None:
        head_rows = table.findall("thead/tr")
        body_rows = table.findall("tbody/tr") + table.findall("tr")
        header_grid, _ = _grid(head_rows, repeat_colspan=True)
        width = max((len(r) for r in header_grid), default=0)
        headers = []
        for column in range(width):
            labels: list[str] = []
            for row in header_grid:
                if column < len(row) and row[column] and row[column] not in labels:
                    labels.append(row[column])
            headers.append(" / ".join(labels))
        body_grid, body_spans = _grid(body_rows, repeat_colspan=False)
        for row, row_spans in zip(body_grid, body_spans):
            cells = []
            for column, value in enumerate(row):
                if not value:
                    continue
                covered = range(column, column + row_spans.get(column, 1))
                header = _span_header([headers[c] for c in covered if c < len(headers)])
                cells.append(f"{header}: {value}" if header else value)
            if cells:
                rows_out.append("; ".join(cells))

    footnotes = [_clean(_flat(fn)) for fn in table_wrap.findall("table-wrap-foot//fn")]
    footnotes += [_clean(_inline(p)) for p in table_wrap.findall("table-wrap-foot/p")]
    return title, rows_out, [f for f in footnotes if f]


def _blocks(element: ET.Element) -> Iterator[tuple[str, object]]:
    """A section's own content, in order: ``("text", str)`` or ``("table", element)``."""
    for child in element:
        tag = child.tag
        if tag in ("title", "sec", "ref-list", "label"):
            continue
        if tag == "p":
            text = _clean(_inline(child))
            if text:
                yield "text", text
            for nested in child:
                if nested.tag == "list":
                    yield "text", "\n".join(_list_lines(nested))
                elif nested.tag == "table-wrap":
                    yield "table", nested
        elif tag == "list":
            lines = _list_lines(child)
            if lines:
                yield "text", "\n".join(lines)
        elif tag == "table-wrap":
            yield "table", child
        elif len(child):
            yield from _blocks(child)            # boxed-text, fig captions, ...


def _section_title(section: ET.Element) -> str:
    title = section.find("title")
    return _clean(_inline(title)) if title is not None else ""


def parse_chapter(
    xml: bytes | str,
    chapter_ids: dict[str, tuple[str, str]],
    max_words: int = MAX_WORDS,
) -> list[Chunk]:
    """One chapter -> passages. Chapters absent from ``chapter_ids`` yield none."""
    root = ET.fromstring(xml)
    shortname = root.get("id", "")
    if shortname not in chapter_ids:
        return []
    title, nbk = chapter_ids[shortname]
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk}/"

    copyright_text = _clean(" ".join(root.find(".//copyright-statement").itertext())) \
        if root.find(".//copyright-statement") is not None else ""
    years = re.search(r"(\d{4})\s*-\s*(\d{4})", copyright_text)
    span = f"{years.group(1)}-{years.group(2)} " if years else ""
    attribution = (f"GeneReviews®, https://www.genereviews.org. "
                   f"© {span}University of Washington, Seattle. Original: {url}")
    licence_element = root.find(".//license")
    licence_url = next((v for k, v in (licence_element.attrib.items()
                                       if licence_element is not None else [])
                        if k.endswith("href")), "https://www.ncbi.nlm.nih.gov/books/NBK138602/")
    licence = f"Non-commercial research use only; excerpts unmodified; terms: {licence_url}"

    chunks: list[Chunk] = []

    def emit(section_path: list[str], section_id: str, kind: str, text: str) -> None:
        chunks.append(Chunk(
            chunk_id=f"{shortname}:{section_id}:{sum(c.section_id == section_id for c in chunks)}",
            source="GeneReviews", doc_id=shortname, doc_title=title,
            section=" > ".join(section_path) or "Summary", section_id=section_id,
            kind=kind, text=text, url=url, attribution=attribution, licence=licence,
        ))

    def walk(section: ET.Element, path: list[str]) -> None:
        heading = _section_title(section) or ("Summary" if section.tag == "abstract" else "")
        if heading.lower() in SKIP_SECTIONS:
            return
        here = path + [heading] if heading else path
        section_id = section.get("id") or re.sub(
            r"[^A-Za-z0-9]+", "_", f"{shortname}.{'_'.join(here) or 'body'}")

        pending: list[str] = []
        count = 0

        def flush() -> None:
            nonlocal pending, count
            if pending:
                emit(here, section_id, "text", "\n".join(pending))
            pending, count = [], 0

        for kind, block in _blocks(section):
            if kind == "text":
                words = len(block.split())
                if pending and count + words > max_words:
                    flush()
                pending.append(block)
                count += words
                continue
            flush()
            table_title, rows, notes = _table(block)
            group: list[str] = []
            for row in rows + ([f"Note: {n}" for n in notes]):
                if group and len(" ".join(group).split()) + len(row.split()) > max_words:
                    emit(here, section_id, "table", "\n".join([table_title, *group]))
                    group = ["(continued)"]
                group.append(row)
            if group:
                emit(here, section_id, "table", "\n".join([table_title, *group]))
        flush()

        for subsection in section.findall("sec"):
            walk(subsection, here)

    meta = root.find(".//book-part-meta")
    if meta is not None and meta.find("abstract") is not None:
        walk(meta.find("abstract"), [])
    body = root.find(".//book-part/body")
    if body is None:
        body = root.find(".//body")
    if body is not None:
        for section in body.findall("sec"):
            walk(section, [])
    return chunks


def load_genereviews(archive: Path | str, chapter_ids_path: Path | str) -> list[Chunk]:
    """Every current GeneReviews chapter in the archive, as passages."""
    chapter_ids = read_chapter_ids(chapter_ids_path)
    chunks: list[Chunk] = []
    chapters = skipped = failed = 0
    for stem, xml in iter_nxml(archive):
        try:
            parsed = parse_chapter(xml, chapter_ids)
        except (ET.ParseError, ValueError, TypeError, AttributeError) as error:
            # One malformed chapter must not cost a 900-chapter build; it is
            # counted and named so it can be looked at.
            failed += 1
            logger.warning("%s: could not be read (%s: %s); skipped.",
                           stem, type(error).__name__, error)
            continue
        if not parsed:
            skipped += 1
            continue
        chapters += 1
        chunks.extend(parsed)
        if chapters % 100 == 0:
            logger.info("GeneReviews: %d chapters, %d passages", chapters, len(chunks))
    logger.info("GeneReviews: %d chapters -> %d passages; %d files not in the chapter "
                "list (retired or non-chapter pages), %d unparseable",
                chapters, len(chunks), skipped, failed)
    if chapters == 0:
        raise ValueError(f"No GeneReviews chapters found in {archive}.")
    return chunks
