"""Knowledge base: the rules of docs/kb/DESIGN.md, as tests.

Each test pins one way a retrieve-and-cite system goes quietly wrong: a table
read with its headers misaligned, an identifier lost by search, an uncited
sentence slipping through, a variant classification passing through the
model, a request leaving the machine, a prompt silently truncated.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import urllib.error

import numpy as np
import pytest

CHAPTER = """<?xml version="1.0" encoding="UTF-8"?>
<book-part-wrapper content-type="chapter" id="testsyn" dtd-version="2.0"
    xmlns:xlink="http://www.w3.org/1999/xlink">
  <book-meta>
    <permissions>
      <copyright-statement>Copyright © 1993-2026, University of Washington, Seattle.</copyright-statement>
      <license license-type="open-access" xlink:href="https://www.ncbi.nlm.nih.gov/books/NBK138602/">
        <license-p>terms</license-p></license>
    </permissions>
    <abstract><p>Book-level blurb that is not part of any chapter.</p></abstract>
  </book-meta>
  <book-part book-part-type="chapter">
    <book-part-meta>
      <title-group><title>Test Syndrome</title></title-group>
      <abstract id="testsyn.Summary"><title>Summary</title>
        <p>Test syndrome is caused by <italic>TST1</italic> variants.</p></abstract>
    </book-part-meta>
    <body>
      <sec id="testsyn.Management"><title>Management</title>
        <sec id="testsyn.Surveillance"><title>Surveillance</title>
          <p>Screening with colonoscopy is recommended for all carriers.</p>
          <list list-type="bullet"><list-item><p>Colonoscopy</p>
            <list list-type="bullet"><list-item><p>every 1-2 yrs</p></list-item></list>
          </list-item></list>
          <table-wrap id="testsyn.T1"><label>Table 1. </label>
            <caption><p>Recommended Surveillance</p></caption>
            <table><thead>
              <tr><th rowspan="2">Concern</th><th colspan="2">Frequency</th></tr>
              <tr><th>TST1</th><th>TST2</th></tr>
            </thead><tbody>
              <tr><td rowspan="2">Colon</td><td>Every 1-2 yrs</td><td>Every 3 yrs</td></tr>
              <tr><td colspan="2">Annual review</td></tr>
            </tbody></table>
            <table-wrap-foot><fn><p>1. Intervals are individualized.</p></fn></table-wrap-foot>
          </table-wrap>
        </sec>
      </sec>
      <sec id="testsyn.References"><title>References</title>
        <p>Smith et al 2020, a citation that must not be indexed.</p></sec>
    </body>
  </book-part>
</book-part-wrapper>""".encode("utf-8")

IDS = {"testsyn": ("Test Syndrome", "NBK9999")}


def _chunks():
    from vpdl.kb.genereviews import parse_chapter
    return parse_chapter(CHAPTER, IDS)


# -- reading GeneReviews ----------------------------------------------------------

def test_sections_keep_their_path_and_bibliography_is_skipped():
    chunks = _chunks()
    sections = {c.section for c in chunks}
    assert "Management > Surveillance" in sections
    assert "Summary" in sections
    assert not any("Smith et al" in c.text for c in chunks), "references were indexed"
    assert not any("Book-level blurb" in c.text for c in chunks), \
        "the book's own abstract was taken for the chapter's"


def test_every_passage_carries_the_attribution_and_link_the_licence_requires():
    for chunk in _chunks():
        assert "https://www.genereviews.org" in chunk.attribution
        assert "© 1993-2026 University of Washington" in chunk.attribution
        assert chunk.url == "https://www.ncbi.nlm.nih.gov/books/NBK9999/"
        assert "NBK138602" in chunk.licence


def test_nested_lists_keep_their_structure():
    text = next(c.text for c in _chunks() if c.kind == "text" and "Colonoscopy" in c.text)
    assert "- Colonoscopy\n  - every 1-2 yrs" in text


def test_table_rows_carry_their_headers_through_rowspan_and_colspan():
    """A misaligned header puts a surveillance interval under the wrong gene."""
    table = next(c.text for c in _chunks() if c.kind == "table")
    lines = table.splitlines()
    assert lines[0] == "Table 1. Recommended Surveillance"
    assert ("Concern: Colon; Frequency / TST1: Every 1-2 yrs; "
            "Frequency / TST2: Every 3 yrs") in lines
    # The row below still belongs to "Colon" (rowspan), and "Annual review"
    # covers BOTH genes (colspan) — not just the first column.
    assert "Concern: Colon; Frequency / TST1, TST2: Annual review" in lines
    assert "Note: 1. Intervals are individualized." in lines


def test_ncbis_mixed_encoding_chapter_list_is_read(tmp_path):
    """The real file is UTF-8 except two Latin-1 titles; it crashed the first DGX build."""
    from vpdl.kb.genereviews import read_chapter_ids

    path = tmp_path / "ids.txt"
    path.write_bytes(
        "#GR_shortname\tGR_Title\tNBK_id\tPMID\n".encode()
        + "cantu\tCantú syndrome\tNBK246980\t25275207\n".encode("latin-1")
        + "hnpcc\tLynch Syndrome\tNBK1211\t20301390\n".encode("utf-8"))
    ids = read_chapter_ids(path)
    assert ids["cantu"] == ("Cantú syndrome", "NBK246980")
    assert ids["hnpcc"] == ("Lynch Syndrome", "NBK1211")


def test_chapters_missing_from_ncbis_id_list_are_skipped():
    from vpdl.kb.genereviews import parse_chapter
    assert parse_chapter(CHAPTER, {"other": ("Other", "NBK1")}) == []


def test_archive_is_read_in_place(tmp_path):
    from vpdl.kb.genereviews import load_genereviews

    archive = tmp_path / "gene.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("gene_NBK1116/testsyn.nxml")
        info.size = len(CHAPTER)
        tar.addfile(info, io.BytesIO(CHAPTER))
    ids = tmp_path / "ids.txt"
    ids.write_text("#GR_shortname\tGR_Title\tNBK_id\tPMID\ntestsyn\tTest Syndrome\tNBK9999\t1\n")
    chunks = load_genereviews(archive, ids)
    assert chunks and all(c.doc_id == "testsyn" for c in chunks)
    assert not (tmp_path / "gene_NBK1116").exists(), "archive was extracted to disk"


# -- search ---------------------------------------------------------------------

def test_exact_identifiers_survive_tokenising():
    from vpdl.kb.search import tokenize
    tokens = tokenize("MLH1 c.199G>A (p.Gly67Arg)")
    assert {"mlh1", "c.199g>a", "p.gly67arg", "199g"} <= set(tokens)


def test_keyword_search_tells_similar_genes_apart():
    from vpdl.kb.search import BM25
    documents = ["MLH1 c.199G>A is a missense variant in exon 2.",
                 "MSH2 c.1906G>C is an Ashkenazi Jewish founder variant.",
                 "Colonoscopy is advised every one to two years."]
    bm25 = BM25(documents)
    assert int(np.argmax(bm25.scores("199G>A"))) == 0
    assert int(np.argmax(bm25.scores("MSH2 founder"))) == 1


def test_reciprocal_rank_fusion_rewards_agreement():
    from vpdl.kb.search import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 1]]) == [1, 3, 2]


def test_meaning_search_adds_what_exact_words_miss():
    from vpdl.kb.chunks import Chunk
    from vpdl.kb.search import KnowledgeIndex

    def chunk(i, text):
        return Chunk(f"d:s:{i}", "T", "d", "Doc", "S", "s", "text", text, "u", "a", "l")

    chunks = [chunk(0, "Colonoscopy is advised every one to two years."),
              chunk(1, "Aspirin reduces colorectal cancer risk.")]
    # Toy embedding: "bowel screening" means colonoscopy.
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = KnowledgeIndex(chunks, embeddings, "toy")
    embed = lambda texts: np.array([[1.0, 0.0]])   # noqa: E731
    found = index.search("bowel screening interval", k=1, embed=embed)
    assert found[0].chunk_id == "d:s:0"


def test_embeddings_that_no_longer_match_the_passages_are_refused(tmp_path):
    from vpdl.kb.search import KnowledgeIndex

    chunks = _chunks()
    KnowledgeIndex(chunks, np.ones((len(chunks), 3), dtype=np.float32), "toy").save(tmp_path)
    assert len(KnowledgeIndex.load(tmp_path).chunks) == len(chunks)

    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["text"] = "edited after embedding"
    lines[0] = json.dumps(record)
    (tmp_path / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        KnowledgeIndex.load(tmp_path)


# -- citations -------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "Colonoscopy is advised every 1-2 years [S1]. It begins at age 20-25 years [S1][S2].",
    "Colonoscopy is advised every 1-2 years. [S1] It begins at age 20-25 years. [S2]",
    "Cancers (e.g. colon and endometrium) are common in carriers [S1, S2].",
    "**Surveillance:**\n- Colonoscopy every 1-2 years for carriers [S1]",
])
def test_cited_answers_pass(answer):
    from vpdl.kb.answer import check_citations
    assert check_citations(answer, n_passages=2).ok


def test_one_uncited_sentence_withholds_the_whole_answer():
    from vpdl.kb.answer import check_citations
    result = check_citations("Colonoscopy is advised every 1-2 years [S1]. "
                             "Aspirin prevents every cancer in all carriers.", 2)
    assert not result.ok and result.reason == "uncited_sentence"


def test_citing_a_passage_that_was_not_retrieved_withholds_the_answer():
    from vpdl.kb.answer import check_citations
    result = check_citations("Colonoscopy is advised every 1-2 years [S7].", 6)
    assert not result.ok and result.reason == "invalid_citation"


def test_not_found_is_an_answer_not_a_failure():
    from vpdl.kb.answer import check_citations
    assert check_citations("NOT_FOUND", 6).reason == "not_found"


# -- the model server --------------------------------------------------------------

def test_the_client_refuses_any_machine_but_this_one():
    from vpdl.kb.ollama import LocalOllama
    for url in ("http://10.0.0.5:11434", "https://api.example.com", "http://0.0.0.0:11434"):
        with pytest.raises(ValueError, match="only talks to this machine"):
            LocalOllama(url)
    assert LocalOllama("http://localhost:11434").url == "http://localhost:11434"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_ollama(monkeypatch, size_vram_share=1.0, calls=None):
    """A stand-in Ollama: /api/chat, /api/embed, and /api/ps reporting where the
    model sits (share of its bytes in GPU memory)."""
    import vpdl.kb.ollama as ollama

    sent = {}

    def fake_urlopen(request, timeout):
        path = request.full_url.rsplit(":11434", 1)[-1]
        if calls is not None:
            calls.append(path)
        if path == "/api/ps":
            size = 5_000_000_000
            return _Response(json.dumps({"models": [
                {"name": "m:latest", "model": "m:latest", "size": size,
                 "size_vram": int(size * size_vram_share)}]}).encode())
        sent.update(json.loads(request.data))
        if path == "/api/embed":
            return _Response(json.dumps({"embeddings": [[0.1, 0.2]]}).encode())
        return _Response(json.dumps(
            {"message": {"content": "<think>hidden</think>Answer [S1]."}}).encode())

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    return ollama, sent


def test_chat_sets_a_context_window_big_enough_for_the_passages(monkeypatch):
    """Ollama's default window truncates from the start: the rules go first."""
    ollama, sent = _fake_ollama(monkeypatch)
    reply = ollama.LocalOllama().chat([{"role": "user", "content": "q"}], "m")
    assert reply == "Answer [S1]."
    assert sent["options"]["num_ctx"] >= 8192
    assert sent["options"]["temperature"] == 0
    assert sent["stream"] is False


def test_a_model_that_ollama_put_on_the_cpu_stops_the_run(monkeypatch):
    """Ollama falls back to the CPU without an error; answers just arrive 10-50x
    slower. The client asks where the model loaded instead of assuming."""
    ollama, _ = _fake_ollama(monkeypatch, size_vram_share=0.0)
    with pytest.raises(RuntimeError, match="on the CPU"):
        ollama.LocalOllama().chat([{"role": "user", "content": "q"}], "m")
    with pytest.raises(RuntimeError, match="on the CPU"):
        ollama.LocalOllama().embed(["x"], "m")


def test_a_partly_offloaded_model_also_stops_the_run(monkeypatch):
    ollama, _ = _fake_ollama(monkeypatch, size_vram_share=0.8)
    with pytest.raises(RuntimeError, match="20% on the CPU"):
        ollama.LocalOllama().chat([{"role": "user", "content": "q"}], "m")


def test_allow_cpu_runs_anyway_but_records_where(monkeypatch):
    ollama, _ = _fake_ollama(monkeypatch, size_vram_share=0.0)
    client = ollama.LocalOllama(require_gpu=False)
    assert client.chat([{"role": "user", "content": "q"}], "m") == "Answer [S1]."
    assert client.gpu_share["m"] == 0.0


def test_the_gpu_is_checked_once_per_model_not_per_question(monkeypatch):
    calls = []
    ollama, _ = _fake_ollama(monkeypatch, calls=calls)
    client = ollama.LocalOllama()
    for _ in range(3):
        client.chat([{"role": "user", "content": "q"}], "m")
    assert calls.count("/api/ps") == 1
    assert client.gpu_share == {"m": 1.0}


def test_untagged_model_names_match_ollamas_latest_tag(monkeypatch):
    ollama, _ = _fake_ollama(monkeypatch)
    assert ollama.LocalOllama().placement("m") == 1.0          # listed as "m:latest"
    assert ollama.LocalOllama().placement("other") is None


def test_a_stopped_server_gives_an_instruction_not_a_traceback(monkeypatch):
    import vpdl.kb.ollama as ollama

    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="not reachable"):
        ollama.LocalOllama().embed(["x"], "bge-m3")


# -- variant lookup --------------------------------------------------------------

HEADER = ["#AlleleID", "Type", "Name", "GeneID", "GeneSymbol", "ClinicalSignificance",
          "LastEvaluated", "RS# (dbSNP)", "PhenotypeList", "Assembly", "ReviewStatus",
          "NumberSubmitters", "VariationID"]


def _variant_db(tmp_path):
    from vpdl.kb.variants import VariantDB, build_variant_db

    rows = [
        # Same variant on two assemblies: one record.
        ["1", "single nucleotide variant", "NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)", "4292",
         "MLH1", "Pathogenic", "Jun 19, 2019", "63750217", "Lynch syndrome 2", "GRCh37",
         "reviewed by expert panel", "12", "17093"],
        ["1", "single nucleotide variant", "NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)", "4292",
         "MLH1", "Pathogenic", "Jun 19, 2019", "63750217", "Lynch syndrome 2", "GRCh38",
         "reviewed by expert panel", "12", "17093"],
        ["2", "single nucleotide variant", "NM_000251.3(MSH2):c.1906G>C (p.Ala636Pro)", "4436",
         "MSH2", "Pathogenic", "Sep 5, 2013", "63750875", "Lynch syndrome 1", "GRCh38",
         "reviewed by expert panel", "20", "1753"],
        ["3", "single nucleotide variant", "NM_000249.4(MLH1):c.100A>G (p.Thr34Ala)", "4292",
         "MLH1", "Uncertain significance", "-", "-1", "not provided", "GRCh38",
         "criteria provided, single submitter", "1", "99001"],
        ["4", "single nucleotide variant", "NM_007294.4(BRCA1):c.68_69del (p.Glu23fs)", "672",
         "BRCA1", "Pathogenic", "-", "-1", "Breast cancer", "GRCh38",
         "reviewed by expert panel", "40", "17662"],
    ]
    source = tmp_path / "variant_summary.txt.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    count = build_variant_db(source, tmp_path / "clinvar.sqlite")
    assert count == 3, "assembly duplicates must collapse; other genes must be dropped"
    return VariantDB(tmp_path / "clinvar.sqlite")


@pytest.mark.parametrize("question, expected", [
    ("What does ClinVar say about MLH1 c.199G>A?", "c.199G>A"),
    ("mlh1 c.199g>a", "c.199G>A"),
    ("Is MSH2 p.Ala636Pro pathogenic?", "p.Ala636Pro"),
    ("msh2 A636P classification", "p.Ala636Pro"),
    ("rs63750217", "c.199G>A"),
])
def test_variants_are_found_however_they_are_written(tmp_path, question, expected):
    report = _variant_db(tmp_path).lookup(question)
    assert report is not None and any(expected in r.name for r in report.records)


def test_uncertain_classifications_are_kept_in_clinvars_own_words(tmp_path):
    report = _variant_db(tmp_path).lookup("MLH1 c.100A>G")
    assert report.records[0].classification == "Uncertain significance"


def test_a_change_without_a_gene_is_not_guessed(tmp_path):
    report = _variant_db(tmp_path).lookup("What is c.199G>A?")
    assert report.records == [] and "name the gene" in report.notes[0]


def test_absence_from_clinvar_is_not_reported_as_benign(tmp_path):
    report = _variant_db(tmp_path).lookup("MLH1 c.99999G>A")
    assert report.records == []
    assert "says nothing about whether a variant is harmful" in report.notes[0]


def test_questions_without_a_variant_skip_the_lookup(tmp_path):
    assert _variant_db(tmp_path).lookup("How is Lynch syndrome inherited?") is None


# -- the whole path ------------------------------------------------------------------

class FakeModel:
    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def chat(self, messages, model):
        self.messages = messages
        return self.reply(messages) if callable(self.reply) else self.reply

    def embed(self, texts, model):
        return np.ones((len(texts), 3), dtype=np.float32)


def _index():
    from vpdl.kb.search import KnowledgeIndex
    return KnowledgeIndex(_chunks())


def test_a_cited_answer_is_shown_with_verbatim_attributed_sources():
    from vpdl.kb.answer import DISCLAIMER, ask, render
    answer = ask("How often is colonoscopy recommended?", _index(),
                 FakeModel("Colonoscopy is recommended every 1-2 yrs [S1]."), "m")
    text = render(answer)
    assert answer.answered and text.startswith(DISCLAIMER)
    cited = answer.passages[answer.check.cited[0] - 1]
    assert cited.text.splitlines()[0] in text, "excerpt must be shown word for word"
    assert "https://www.genereviews.org" in text


def test_variant_facts_are_shown_but_never_given_to_the_model(tmp_path):
    from vpdl.kb.answer import ask, render
    model = FakeModel("NOT_FOUND")
    answer = ask("Is MLH1 c.199G>A pathogenic, and how often is colonoscopy recommended?",
                 _index(), model, "m", variant_db=_variant_db(tmp_path))
    assert model.messages is not None, "the model was never asked; the test proves nothing"
    prompt = json.dumps(model.messages)
    assert "expert panel" not in prompt and "clinvar/variation" not in prompt
    text = render(answer)
    assert "ClinVar classification: Pathogenic" in text
    assert "https://www.ncbi.nlm.nih.gov/clinvar/variation/17093/" in text


def test_an_answer_failing_the_check_is_withheld_not_shown():
    from vpdl.kb.answer import ask, render
    reply = "Colonoscopy is recommended every 1-2 yrs [S1]. Also take vitamin D daily for life."
    text = render(ask("How often is colonoscopy recommended?", _index(),
                      FakeModel(reply), "m"))
    assert "ANSWER WITHHELD" in text
    assert "vitamin D" not in text, "the withheld claim leaked through the explanation"


def test_private_sources_answer_but_are_never_displayed():
    import dataclasses

    from vpdl.kb.answer import ask, render
    from vpdl.kb.search import KnowledgeIndex
    private = [dataclasses.replace(c, private=True) for c in _chunks()]
    answer = ask("How often is colonoscopy recommended?", KnowledgeIndex(private),
                 FakeModel("Colonoscopy is recommended every 1-2 yrs [S1]."), "m")
    text = render(answer)
    assert "private source: excerpt not displayed" in text
    assert answer.passages[0].text.splitlines()[0] not in text


# -- evaluation ---------------------------------------------------------------------

def test_evaluation_scores_answers_refusals_and_lookups(tmp_path):
    from vpdl.kb.evaluate import evaluate

    def reply(messages):
        question = messages[-1]["content"].rsplit("Question:", 1)[1]
        return ("Colonoscopy is recommended every 1-2 yrs [S1]."
                if "colonoscopy" in question.lower() else "NOT_FOUND")

    questions = [
        {"id": "a", "type": "answerable", "question": "How often is colonoscopy recommended?",
         "expected_sections": ["testsyn.Surveillance"], "answer_keywords": [["1-2 yrs"]]},
        {"id": "r", "type": "refusal", "question": "What is the capital of Australia?"},
        {"id": "v", "type": "variant", "question": "MLH1 c.199G>A",
         "expected_name_contains": "c.199G>A", "expect_found": True},
    ]
    [summary] = evaluate(questions, _index(), FakeModel(reply), ["m"], tmp_path,
                         variant_db=_variant_db(tmp_path))
    assert summary["retrieval_hit"] == 1.0
    assert summary["correct_facts"] == 1.0
    assert summary["refused_correctly"] == 1.0
    assert summary["variant_lookup_correct"] == "1/1"
    written = (tmp_path / "answers_m.jsonl").read_text(encoding="utf-8")
    assert "Screening with colonoscopy is recommended" not in written, \
        "evaluation output must reference passages by id, not republish them"


def test_the_shipped_question_set_is_well_formed():
    from pathlib import Path

    from vpdl.kb.evaluate import load_questions
    questions = load_questions(Path(__file__).parents[2] / "docs/kb/eval_questions.jsonl")
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    kinds = {q["type"] for q in questions}
    assert kinds == {"answerable", "refusal", "variant"}
    for question in questions:
        if question["type"] == "answerable":
            assert question["expected_sections"] and question["answer_keywords"]
