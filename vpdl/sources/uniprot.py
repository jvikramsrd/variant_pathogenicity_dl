"""UniProt — the coordinate authority for the entire project.

Not a source in the labelling sense: it supplies the canonical sequence every
other source's positions are validated against, plus domain and functional-site
features. Nothing enters the assembled table without agreeing with the sequence
here, which is what stops isoform drift from silently repositioning variants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from vpdl.sources.base import SourceCapabilities

logger = logging.getLogger(__name__)

__all__ = ["UNIPROT_REST", "MMR_ACCESSIONS", "provides", "fetch", "load_sequences",
           "domain_features"]

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/{accession}.json"

# Pinned for the Lynch panel. Lengths are recorded so a silently different
# isoform is caught at load rather than after training.
MMR_ACCESSIONS: dict[str, tuple[str, int]] = {
    "MLH1": ("P40692", 756),
    "MSH2": ("P43246", 934),
    "MSH6": ("P52701", 1360),
    "PMS2": ("P54278", 862),
}


def provides() -> SourceCapabilities:
    return SourceCapabilities(
        name="uniprot",
        supplies_labels=False,
        feature_columns=("feature_in_domain", "feature_is_functional_site"),
        licence="CC BY 4.0",
        notes="Coordinate authority; also supplies domain and site features.",
    )


def fetch(accession: str, cache_dir: Path) -> Path:
    """Download one accession's JSON record, cached by accession."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{accession}.json"
    if target.exists():
        return target

    import urllib.request
    url = UNIPROT_REST.format(accession=accession)
    logger.info("Fetching %s", url)
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def load_sequences(
    cache_dir: Path,
    accessions: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, str]:
    """Return ``{accession: sequence}``, asserting the expected length.

    A length mismatch means UniProt has published a different canonical isoform
    than the one this panel was pinned to. That must stop the build: every
    position in every other source is interpreted against this sequence.
    """
    accessions = accessions or MMR_ACCESSIONS
    sequences: dict[str, str] = {}

    for gene, (accession, expected_length) in accessions.items():
        record = json.loads(fetch(accession, cache_dir).read_text())
        sequence = record["sequence"]["value"]
        if len(sequence) != expected_length:
            raise ValueError(
                f"{gene} ({accession}): UniProt returned a {len(sequence)}-residue "
                f"sequence, pinned length is {expected_length}. The canonical "
                "isoform has changed; every downstream coordinate is affected."
            )
        sequences[accession] = sequence

    return sequences


def domain_features(
    cache_dir: Path,
    accessions: Mapping[str, tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Per-residue domain and functional-site flags for the panel."""
    accessions = accessions or MMR_ACCESSIONS
    rows: list[dict] = []

    for gene, (accession, _) in accessions.items():
        record = json.loads(fetch(accession, cache_dir).read_text())
        domains: list[tuple[int, int]] = []
        sites: set[int] = set()

        for feature in record.get("features", []):
            kind = feature.get("type", "")
            location = feature.get("location", {})
            start = location.get("start", {}).get("value")
            end = location.get("end", {}).get("value")
            if start is None or end is None:
                continue
            if kind in {"Domain", "Region", "Repeat"}:
                domains.append((int(start), int(end)))
            elif kind in {"Active site", "Binding site", "Site", "Metal binding"}:
                sites.update(range(int(start), int(end) + 1))

        length = len(record["sequence"]["value"])
        for position in range(1, length + 1):
            rows.append({
                "uniprot_id": accession,
                "gene": gene,
                "position": position,
                "feature_in_domain": int(
                    any(start <= position <= end for start, end in domains)
                ),
                "feature_is_functional_site": int(position in sites),
            })

    return pd.DataFrame(rows)
