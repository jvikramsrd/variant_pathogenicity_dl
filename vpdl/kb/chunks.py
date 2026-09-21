"""A passage: the unit that is searched, cited and shown."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

__all__ = ["Chunk", "save_chunks", "load_chunks", "chunks_digest"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # stable: "<doc_id>:<section_id>:<n>"
    source: str          # "GeneReviews"
    doc_id: str          # "hnpcc"
    doc_title: str       # "Lynch Syndrome"
    section: str         # "Management > Surveillance"
    section_id: str      # "hnpcc.Surveillance"
    kind: str            # "text" | "table"
    text: str            # exactly what is shown to a reader as the excerpt
    url: str             # link to the original, required by GeneReviews' terms
    attribution: str     # credit line shown with every excerpt
    licence: str
    private: bool = False  # used to answer, never displayed (owned textbooks)

    @property
    def heading(self) -> str:
        return f"{self.source} - {self.doc_title} - {self.section}"

    def search_text(self) -> str:
        """What the search indexes: the excerpt plus where it sits.

        "Surveillance" rarely appears inside the surveillance table itself; the
        section path carries it.
        """
        return f"{self.doc_title} > {self.section}\n{self.text}"


def save_chunks(chunks: Iterable[Chunk], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_chunks(path: Path | str) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def chunks_digest(chunks: Iterable[Chunk]) -> str:
    """Fingerprint of the passages, so stored embeddings can prove they match."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(b"\0")
        digest.update(chunk.search_text().encode())
        digest.update(b"\0")
    return digest.hexdigest()
