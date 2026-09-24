"""PubMed abstracts in the knowledge base: the on-disk index and its merge with GeneReviews."""

from __future__ import annotations

import gzip

import pytest

from vpdl.kb.chunks import Chunk
from vpdl.kb.pubmed_index import CombinedIndex, PubMedIndex, build_pubmed_index
from vpdl.kb.search import KnowledgeIndex


def _article(pmid, title, abstract, types=("Journal Article",), year="2021"):
    kinds = "".join(f"<PublicationType>{t}</PublicationType>" for t in types)
    return (f'<PubmedArticle><MedlineCitation><PMID Version="1">{pmid}</PMID><Article>'
            f'<Journal><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>'
            f'<ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText>{abstract}</AbstractText>'
            f'</Abstract><Language>eng</Language><PublicationTypeList>{kinds}</PublicationTypeList>'
            f'</Article></MedlineCitation></PubmedArticle>')


def _pubmed(folder, name, articles):
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / name, "wt", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?><PubmedArticleSet>')
        handle.write("".join(articles))
        handle.write("</PubmedArticleSet>")


@pytest.fixture
def pubmed_dir(tmp_path):
    folder = tmp_path / "pubmed"
    _pubmed(folder, "pubmed26n0001.xml.gz", [
        _article(101, "MLH1 c.199G&gt;A in Lynch syndrome",
                 "The germline MLH1 variant c.199G&gt;A segregated with colorectal cancer."),
        _article(102, "MSH2 deletions", "Large MSH2 deletions cause Lynch syndrome in families."),
        _article(103, "Retracted", "MLH1 c.199G&gt;A fabricated.",
                 types=("Journal Article", "Retracted Publication")),
    ])
    _pubmed(folder, "pubmed26n0002.xml.gz", [
        _article(104, "Hip fracture surgery", "Outcomes of hip fracture surgery in older adults."),
        _article(101, "duplicate", "The same PMID again in a later file."),
    ])
    return folder


def test_the_index_holds_kept_abstracts_once_and_is_built_atomically(tmp_path, pubmed_dir):
    out = tmp_path / "kb" / "pubmed.sqlite"
    summary = build_pubmed_index(pubmed_dir, out, workers=1)
    assert summary["abstracts"] == 3
    assert summary["decisions"]["retracted_or_concern"] == 1
    assert summary["decisions"]["duplicate_pmid"] == 1
    assert out.exists() and not out.with_suffix(".building").exists()
    assert PubMedIndex(out).meta["abstracts"] == 3


def test_identifiers_are_matched_exactly_and_passages_carry_their_source(tmp_path, pubmed_dir):
    out = tmp_path / "pubmed.sqlite"
    build_pubmed_index(pubmed_dir, out, workers=1)
    index = PubMedIndex(out)
    hits = index.search("What is known about MLH1 c.199G>A?", k=3)
    assert hits and hits[0].chunk_id == "pubmed:101"
    assert "103" not in {h.doc_id for h in hits}                      # retracted: never indexed
    first = hits[0]
    assert first.source == "PubMed" and first.url == "https://pubmed.ncbi.nlm.nih.gov/101/"
    assert "copyright" in first.attribution and first.section == "Abstract (2021)"
    assert first.text.startswith("The germline MLH1 variant c.199G>A")
    assert index.search("MSH2 hip", k=5)                              # nothing has both words: OR fills
    assert index.search("", k=5) == []


def test_the_genetics_filter_leaves_unrelated_abstracts_out(tmp_path, pubmed_dir):
    summary = build_pubmed_index(pubmed_dir, tmp_path / "g.sqlite", workers=1, genetics_only=True)
    # hip-fracture abstract + the duplicate record (no genetics words either)
    assert summary["abstracts"] == 2 and summary["decisions"]["kb_filtered_out"] == 2


def _genereviews_chunk():
    return Chunk(chunk_id="hnpcc:Surveillance:0", source="GeneReviews", doc_id="hnpcc",
                 doc_title="Lynch Syndrome", section="Management > Surveillance",
                 section_id="hnpcc.Surveillance", kind="text",
                 text="Colonoscopy every one to two years is recommended for MLH1 heterozygotes.",
                 url="https://www.ncbi.nlm.nih.gov/books/NBK1211/", attribution="GeneReviews",
                 licence="GeneReviews terms")


def test_questions_search_genereviews_and_pubmed_together(tmp_path, pubmed_dir):
    build_pubmed_index(pubmed_dir, tmp_path / "pubmed.sqlite", workers=1)
    combined = CombinedIndex(KnowledgeIndex([_genereviews_chunk()]),
                             PubMedIndex(tmp_path / "pubmed.sqlite"))
    sources = {c.source for c in combined.search("MLH1 colonoscopy c.199G>A", k=4)}
    assert sources == {"GeneReviews", "PubMed"}
    assert combined.embeddings is None and combined.embed_model is None


def test_the_cli_builds_the_index_and_ask_can_leave_it_out(tmp_path, pubmed_dir):
    import argparse

    from vpdl.cli import _open_kb, main
    kb = tmp_path / "kb"
    KnowledgeIndex([_genereviews_chunk()]).save(kb)
    assert main(["kb-build", "--kb", str(kb), "--pubmed", str(pubmed_dir),
                 "--pubmed-workers", "1"]) == 0
    with_pubmed, _, _ = _open_kb(argparse.Namespace(kb=str(kb), no_pubmed=False))
    without, _, _ = _open_kb(argparse.Namespace(kb=str(kb), no_pubmed=True))
    assert isinstance(with_pubmed, CombinedIndex) and isinstance(without, KnowledgeIndex)
