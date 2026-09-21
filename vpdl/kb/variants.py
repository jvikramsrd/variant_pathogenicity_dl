"""Variant facts, looked up — never generated.

Rule 4 of docs/kb/DESIGN.md: the model never classifies a variant. When a
question names one, its ClinVar record is read from a local table and printed
as recorded (classification, review status, date, submitters, link), beside —
never through — the model's answer.

The table is built from ClinVar's ``variant_summary.txt.gz`` and keeps every
record for the chosen genes: uncertain and conflicting ones included, with
ClinVar's own wording. (vpdl's training table cannot serve here: it keeps only
2-star records and reduces them to 0/1 labels.)
"""

from __future__ import annotations

import gzip
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from vpdl.sources.clinvar import _THREE_TO_ONE, _review_stars

logger = logging.getLogger(__name__)

__all__ = ["LYNCH_GENES", "build_variant_db", "VariantDB", "VariantRecord",
           "VariantReport", "EvidenceTable", "parse_mentions"]

LYNCH_GENES = ("MLH1", "MSH2", "MSH6", "PMS2", "EPCAM")

_NAME = re.compile(r"^(?P<tx>[A-Z]{2}_\d+(?:\.\d+)?)\((?P<gene>[^)]+)\):"
                   r"(?P<cdna>c\.[^\s(]+)(?:\s+\((?P<protein>p\.[^)]+)\))?")
_PROTEIN_3 = re.compile(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)\)?", re.I)
_PROTEIN_1 = re.compile(r"(?<![A-Za-z0-9])(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})"
                        r"([ACDEFGHIKLMNPQRSTVWY*])(?![A-Za-z0-9])")
_CDNA = re.compile(r"(?<![A-Za-z0-9])c\.[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?"
                   r"(?:[ACGT]>[ACGT]|delins[ACGT]+|del[ACGT]*|dup[ACGT]*|ins[ACGT]+)",
                   re.I)
_RSID = re.compile(r"(?<![A-Za-z0-9])rs\d{3,}(?![0-9])", re.I)
_WORD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9-]{1,14}(?![A-Za-z0-9])")

_SCHEMA = """
CREATE TABLE variants (
    variation_id INTEGER PRIMARY KEY,
    gene TEXT, name TEXT, cdna TEXT, protein TEXT, protein_1 TEXT, rsid TEXT,
    classification TEXT, review_status TEXT, stars INTEGER,
    last_evaluated TEXT, submitters INTEGER, conditions TEXT, type TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _one_letter(protein: str | None) -> str | None:
    """``p.Gly67Arg`` -> ``G67R``; ``p.Gly67Ter`` -> ``G67*``; others -> None."""
    if not protein:
        return None
    match = re.fullmatch(r"p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)\)?", protein)
    if not match:
        return None
    wt, position, mut = match.groups()
    wt_1 = _THREE_TO_ONE.get(wt)
    mut_1 = "*" if mut == "Ter" else ("=" if mut == "=" else _THREE_TO_ONE.get(mut))
    return f"{wt_1}{position}{mut_1}" if wt_1 and mut_1 else None


def _normalise_cdna(text: str) -> str:
    """``c.199g>a`` -> ``c.199G>A``; ``c.1234DEL`` -> ``c.1234del``."""
    body = text[2:].lower()
    return "c." + re.sub(r"[acgt]", lambda m: m.group(0).upper(), body)


def build_variant_db(
    variant_summary: Path | str,
    out: Path | str,
    genes: Sequence[str] | None = LYNCH_GENES,
) -> int:
    """Stream ClinVar's summary file once into a lookup table. ``genes=None``: all."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    wanted = set(genes) if genes else None

    connection = sqlite3.connect(temporary)
    connection.executescript(_SCHEMA)
    rows = 0
    opener = gzip.open if str(variant_summary).endswith(".gz") else open
    with opener(variant_summary, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().lstrip("#").rstrip("\n").split("\t")
        column = {name: i for i, name in enumerate(header)}
        significance = next((c for c in ("ClinicalSignificance", "GermlineClassification")
                             if c in column), None)
        if significance is None or "VariationID" not in column:
            raise ValueError(f"{variant_summary}: unrecognised ClinVar header {header[:12]}")
        batch = []
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            name = fields[column["Name"]]
            parsed = _NAME.match(name)
            gene = parsed.group("gene") if parsed else fields[column["GeneSymbol"]]
            if wanted is not None and gene not in wanted:
                continue
            protein = parsed.group("protein") if parsed else None
            rsid = fields[column["RS# (dbSNP)"]] if "RS# (dbSNP)" in column else "-1"
            batch.append((
                int(fields[column["VariationID"]]), gene, name,
                _normalise_cdna(parsed.group("cdna")) if parsed else None,
                protein, _one_letter(protein),
                f"rs{rsid}" if rsid not in ("-1", "", "na") else None,
                fields[column[significance]],
                fields[column["ReviewStatus"]],
                _review_stars(fields[column["ReviewStatus"]]),
                fields[column["LastEvaluated"]] if "LastEvaluated" in column else "",
                int(fields[column["NumberSubmitters"]] or 0)
                if "NumberSubmitters" in column else None,
                fields[column["PhenotypeList"]] if "PhenotypeList" in column else "",
                fields[column["Type"]] if "Type" in column else "",
            ))
            if len(batch) >= 50_000:
                rows += _insert(connection, batch)
                batch = []
        rows += _insert(connection, batch)

    connection.executescript("""
        CREATE INDEX by_cdna ON variants (gene, cdna);
        CREATE INDEX by_protein ON variants (gene, protein);
        CREATE INDEX by_protein_1 ON variants (gene, protein_1);
        CREATE INDEX by_rsid ON variants (rsid);""")
    source = Path(variant_summary)
    connection.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("source", source.name),
        ("source_modified", time.strftime("%Y-%m-%d", time.localtime(source.stat().st_mtime))),
        ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("genes", ",".join(sorted(wanted)) if wanted else "all"),
        ("records", str(rows)),
    ])
    connection.commit()
    connection.close()
    temporary.replace(out)
    logger.info("ClinVar lookup table: %d records -> %s", rows, out)
    return rows


def _insert(connection: sqlite3.Connection, batch: list[tuple]) -> int:
    # Each variant appears once per genome assembly with the same record;
    # the primary key keeps the first.
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO variants VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    return connection.total_changes - before


@dataclass(frozen=True)
class VariantRecord:
    variation_id: int
    gene: str
    name: str
    classification: str
    review_status: str
    stars: int
    last_evaluated: str
    submitters: int | None
    conditions: str
    protein_1: str | None

    @property
    def url(self) -> str:
        return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{self.variation_id}/"


@dataclass
class VariantReport:
    mentions: list[str]
    records: list[VariantRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    evidence: dict[int, list[str]] = field(default_factory=dict)   # variation_id -> lines


def parse_mentions(question: str, known_genes: Iterable[str]) -> dict[str, list[str]]:
    """Genes, cDNA changes, protein changes and rsIDs named in a question."""
    known = set(known_genes)
    genes = []
    for match in _WORD.finditer(question):
        word = match.group(0)
        # Exact case always; lower case only for symbols containing a digit
        # ("mlh1"). Otherwise, with all genes loaded, "met" would become MET.
        candidate = word if word in known else (
            word.upper() if any(ch.isdigit() for ch in word) and word.upper() in known
            else None)
        if candidate and candidate not in genes:
            genes.append(candidate)

    proteins: list[str] = []
    for match in _PROTEIN_3.finditer(question):
        wt, position, mut = (match.group(1).capitalize(), match.group(2),
                             match.group(3) if match.group(3) == "=" else match.group(3).capitalize())
        one = _one_letter(f"p.{wt}{position}{mut}")
        if one and one not in proteins:
            proteins.append(one)
    if genes:
        for match in _PROTEIN_1.finditer(question):
            one = "".join(match.groups())
            if one not in proteins:
                proteins.append(one)

    return {
        "genes": genes,
        "cdna": list(dict.fromkeys(_normalise_cdna(m.group(0)) for m in _CDNA.finditer(question))),
        "protein": proteins,
        "rsid": list(dict.fromkeys(m.group(0).lower() for m in _RSID.finditer(question))),
    }


class VariantDB:
    def __init__(self, path: Path | str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No ClinVar lookup table at {path}; run `vpdl kb-build`.")
        self.connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.meta = dict(self.connection.execute("SELECT key, value FROM meta"))
        self.genes = {g for (g,) in self.connection.execute("SELECT DISTINCT gene FROM variants")}

    def _records(self, sql: str, params: tuple) -> list[VariantRecord]:
        query = ("SELECT variation_id, gene, name, classification, review_status, stars, "
                 "last_evaluated, submitters, conditions, protein_1 FROM variants WHERE "
                 + sql + " ORDER BY stars DESC, variation_id")
        return [VariantRecord(*row) for row in self.connection.execute(query, params)]

    def lookup(self, question: str) -> VariantReport | None:
        """None when the question names no variant."""
        found = parse_mentions(question, self.genes)
        if not (found["cdna"] or found["protein"] or found["rsid"]):
            return None
        genes = found["genes"]
        report = VariantReport(
            mentions=[*found["cdna"], *found["protein"], *found["rsid"]],
            source=(f"ClinVar {self.meta.get('source', '')} (downloaded "
                    f"{self.meta.get('source_modified', '?')}; genes: {self.meta.get('genes')})"))

        def add(records: list[VariantRecord], label: str) -> None:
            new = [r for r in records if r not in report.records]
            if new:
                report.records.extend(new)
            elif not records:
                report.notes.append(
                    f"No ClinVar record for {label} in this table. Absence from ClinVar "
                    "says nothing about whether a variant is harmful.")

        for rsid in found["rsid"]:
            add(self._records("rsid = ?", (rsid,)), rsid)
        for change in found["cdna"] + found["protein"]:
            if not genes:
                report.notes.append(f"{change}: name the gene (e.g. 'MLH1 {change}'); the "
                                    "same change exists in many genes.")
                continue
            for gene in genes:
                column = "cdna" if change.startswith("c.") else "protein_1"
                add(self._records(f"gene = ? AND {column} = ?", (gene, change)),
                    f"{gene} {change}")
        return report


class EvidenceTable:
    """AlphaMissense and gnomAD values for missense variants, from vpdl's built table.

    Reported as the sources' numbers with their own caveats — a computational
    score is not a classification.
    """

    COLUMNS = ["gene", "position", "wt_aa", "mut_aa", "feature_alphamissense_score",
               "feature_gnomad_log10_af", "feature_gnomad_observed"]

    def __init__(self, path: Path | str):
        import pandas as pd
        header = pd.read_csv(path, nrows=0).columns
        frame = pd.read_csv(path, usecols=[c for c in self.COLUMNS if c in header],
                            low_memory=False)
        self.rows = {
            (r.gene, f"{r.wt_aa}{int(r.position)}{r.mut_aa}"): r
            for r in frame.itertuples(index=False)
        }

    def lines(self, record: VariantRecord) -> list[str]:
        import math
        row = self.rows.get((record.gene, record.protein_1 or ""))
        if row is None:
            return []
        out = []
        score = getattr(row, "feature_alphamissense_score", float("nan"))
        if isinstance(score, float) and not math.isnan(score):
            out.append(f"AlphaMissense score {score:.3f} (computational prediction, "
                       "CC BY 4.0; not a classification)")
        observed = getattr(row, "feature_gnomad_observed", float("nan"))
        log_af = getattr(row, "feature_gnomad_log10_af", float("nan"))
        if observed == 1.0 and isinstance(log_af, float) and not math.isnan(log_af):
            out.append(f"gnomAD allele frequency {10 ** log_af:.2e}")
        elif observed == 0.0:
            out.append("gnomAD: not observed")
        return out
