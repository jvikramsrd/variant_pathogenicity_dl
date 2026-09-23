"""Sentences with their character offsets — the unit evidence is cut into.

Offsets matter: every evidence unit points back into the document text it came
from (``char_start``/``char_end``), so an extracted span, a masked conclusion
or a cited explanation can always be checked against the original words.

Variant notation is full of full stops that do not end sentences
("c.199G>A", "p.Gly67Arg", "NM_000249.4", "0.01%", "et al. 2020"), so a split
needs a full stop, whitespace, AND a sentence-like start after it (capital,
digit or bracket), and never after a known abbreviation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Sentence", "split_sentences"]

_ABBREVIATIONS = frozenset({
    "al", "e.g", "i.e", "eg", "ie", "vs", "fig", "figs", "approx", "no", "nos", "ca", "cf",
    "dr", "mr", "ms", "mrs", "st", "spp", "ref", "refs", "resp", "suppl", "vol", "p", "c",
    "n", "g", "m", "r", "chr", "ex", "ivs", "etc", "jr", "sr", "inc", "ltd",
    "co", "dept", "univ", "u.s",
})
_BOUNDARY = re.compile(r"[.!?][\"')\]]*(?=\s)")
_START = re.compile(r"\s+([\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str


def _previous_word(text: str, index: int) -> str:
    """The word ending at `index` (the boundary character), lower-cased, dots kept inside."""
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "._"):
        start -= 1
    return text[start:index].lower().strip(".")


def split_sentences(text: str) -> list[Sentence]:
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\n]+", text):
        line_start = line_match.start()
        line = line_match.group(0)
        begin = 0
        for boundary in _BOUNDARY.finditer(line):
            stop = boundary.end()
            follow = _START.match(line, stop)
            if not follow:
                continue
            if line[boundary.start()] == "." and _previous_word(line, boundary.start()) in _ABBREVIATIONS:
                continue
            spans.append((line_start + begin, line_start + stop))
            begin = follow.start(1)
        spans.append((line_start + begin, line_start + len(line)))

    sentences = []
    for start, end in spans:
        # trim whitespace but keep offsets exact
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            sentences.append(Sentence(start, end, text[start:end]))
    return sentences
