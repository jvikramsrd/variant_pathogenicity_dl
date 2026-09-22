"""Small language model pipeline: each test pins a way a multi-day run goes wrong.

Retracted papers entering the training text; a validation split that shifts
between rebuilds; a vocabulary that cannot round-trip "c.199G>A"; token files
that disagree with their tokenizer; a resumed run that silently differs from
an uninterrupted one; a diverged run that keeps going for days.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("tokenizers")


def _article(pmid, abstract=True, language="eng", types=("Journal Article",),
             retraction_link=False, year="2020"):
    comments = ('<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn">'
                '<PMID>999</PMID></CommentsCorrections></CommentsCorrectionsList>'
                if retraction_link else "")
    body = ('<Abstract><AbstractText Label="BACKGROUND">Lynch syndrome is common.</AbstractText>'
            '<AbstractText Label="RESULTS">We found <i>MLH1</i> c.199G&gt;A.</AbstractText>'
            '</Abstract>') if abstract else ""
    kinds = "".join(f"<PublicationType>{t}</PublicationType>" for t in types)
    return (f'<PubmedArticle><MedlineCitation><PMID Version="1">{pmid}</PMID>{comments}<Article>'
            f'<Journal><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>'
            f'<ArticleTitle>Study {pmid} of MLH1.</ArticleTitle>{body}'
            f'<Language>{language}</Language>'
            f'<PublicationTypeList>{kinds}</PublicationTypeList>'
            f'</Article></MedlineCitation></PubmedArticle>')


def _pubmed_file(path, articles):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?><PubmedArticleSet>')
        handle.write("".join(articles))
        handle.write("</PubmedArticleSet>")


# -- PubMed -----------------------------------------------------------------------

def test_retracted_and_unusable_records_never_reach_the_training_text(tmp_path):
    from vpdl.slm.pubmed import iter_abstracts

    path = tmp_path / "pubmed26n0001.xml.gz"
    _pubmed_file(path, [
        _article(1),
        _article(2, abstract=False),
        _article(3, language="fre"),
        _article(4, types=("Journal Article", "Retracted Publication")),
        _article(5, retraction_link=True),
        _article(6, types=("Expression of Concern",)),
    ])
    stats = Counter()
    kept = list(iter_abstracts(path, stats))
    assert [a.pmid for a in kept] == ["1"]
    assert stats == Counter(kept=1, no_abstract=1, not_english=1, retracted_or_concern=3)


def test_structured_abstract_labels_and_inline_markup_are_kept_as_text(tmp_path):
    from vpdl.slm.pubmed import iter_abstracts

    path = tmp_path / "pubmed26n0001.xml.gz"
    _pubmed_file(path, [_article(1)])
    [abstract] = iter_abstracts(path)
    assert abstract.text.splitlines() == [
        "Study 1 of MLH1.", "BACKGROUND: Lynch syndrome is common.",
        "RESULTS: We found MLH1 c.199G>A."]
    assert abstract.year == "2020"


# -- corpus -----------------------------------------------------------------------

def test_validation_split_is_fixed_by_document_id():
    from vpdl.slm.corpus import in_validation

    ids = [f"pmid:{i}" for i in range(20_000)]
    first = [in_validation(i) for i in ids]
    assert first == [in_validation(i) for i in ids], "the split changed between calls"
    assert 0.007 < sum(first) / len(ids) < 0.013          # ~1%


def _kb(tmp_path, private=False):
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    records = [{"chunk_id": "hnpcc:s:0", "source": "GeneReviews", "doc_title": "Lynch Syndrome",
                "section": "Surveillance", "text": "Colonoscopy every 1-2 yrs.", "private": False},
               {"chunk_id": "book:s:0", "source": "Textbook", "doc_title": "Owned book",
                "section": "Ch 1", "text": "Copyrighted text.", "private": True}]
    (kb / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return kb


def test_corpus_counts_every_decision_and_keeps_private_books_out(tmp_path):
    from vpdl.slm.corpus import build_corpus

    pubmed = tmp_path / "pubmed"
    pubmed.mkdir()
    _pubmed_file(pubmed / "pubmed26n0001.xml.gz", [_article(1), _article(4, types=("Retracted Publication",))])
    _pubmed_file(pubmed / "pubmed26n0002.xml.gz", [_article(1), _article(7)])   # PMID 1 again

    summary = build_corpus(tmp_path / "corpus", pubmed_dir=pubmed, kb_dir=_kb(tmp_path))
    assert summary["pubmed_decisions"]["duplicate_pmid"] == 1
    assert summary["pubmed_decisions"]["retracted_or_concern"] == 1
    documents = [json.loads(line) for path in sorted((tmp_path / "corpus").glob("*.jsonl"))
                 for line in path.read_text(encoding="utf-8").splitlines()]
    ids = {d["id"] for d in documents}
    assert ids == {"pmid:1", "pmid:7", "hnpcc:s:0"}, "private or duplicate text got in"
    assert json.loads((tmp_path / "corpus" / "stats.json").read_text())["train_shards"] == 1


# -- tokenizer and token files ----------------------------------------------------------

def _corpus_with_text(tmp_path, documents=400):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rng = np.random.default_rng(0)
    words = ["Lynch", "syndrome", "MLH1", "MSH2", "colonoscopy", "variant", "c.199G>A",
             "p.Gly67Arg", "risk", "every", "1-2", "years", "pathogenic", "gene", "≥2.5%", "Cantú"]
    lines, characters = [], 0
    for index in range(documents):
        text = " ".join(rng.choice(words, 30))
        characters += len(text)
        lines.append(json.dumps({"id": f"d{index}", "source": "test", "text": text},
                                ensure_ascii=False))
    (corpus / "train-00000.jsonl").write_text("\n".join(lines[:-40]) + "\n", encoding="utf-8")
    (corpus / "val.jsonl").write_text("\n".join(lines[-40:]) + "\n", encoding="utf-8")
    (corpus / "stats.json").write_text(json.dumps(
        {"sources": {"test": {"train_characters": characters}}}))
    return corpus


def test_the_vocabulary_round_trips_clinical_text_exactly(tmp_path):
    from vpdl.slm.tokenizer import SPECIAL_TOKENS, load_tokenizer, train_tokenizer

    corpus = _corpus_with_text(tmp_path)
    meta = train_tokenizer(corpus, tmp_path / "tok" / "tokenizer.json", vocab_size=400)
    tokenizer = load_tokenizer(tmp_path / "tok" / "tokenizer.json")
    for text in ["MLH1 c.199G>A (p.Gly67Arg)", "risk ≥2.5% in Cantú syndrome", "a\ttab and  spaces"]:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert list(meta["special_tokens"].values()) == list(range(len(SPECIAL_TOKENS)))
    assert "MLH1 c.199G>A (p.Gly67Arg)" in meta["probes"]


def _packed(tmp_path):
    from vpdl.slm.pack import pack
    from vpdl.slm.tokenizer import train_tokenizer

    corpus = _corpus_with_text(tmp_path)
    train_tokenizer(corpus, tmp_path / "tok" / "tokenizer.json", vocab_size=400)
    meta = pack(corpus, tmp_path / "tok" / "tokenizer.json", tmp_path / "data")
    return corpus, meta


def test_token_files_hold_every_document_plus_one_end_marker_each(tmp_path):
    from vpdl.slm.corpus import iter_texts
    from vpdl.slm.tokenizer import EOS, load_tokenizer

    corpus, meta = _packed(tmp_path)
    tokenizer = load_tokenizer(tmp_path / "data" / "tokenizer.json")   # copied beside the data
    expected = sum(len(tokenizer.encode(t).ids) + 1 for t in iter_texts(corpus, "train"))
    tokens = np.fromfile(tmp_path / "data" / "train.bin", dtype=np.uint16)
    assert len(tokens) == expected == meta["splits"]["train"]["tokens"]
    assert (tokens == tokenizer.token_to_id(EOS)).sum() == meta["splits"]["train"]["documents"]
    assert tokens.max() < tokenizer.get_vocab_size()


# -- model and training ------------------------------------------------------------------

def _torch():
    """Model and training tests need torch + transformers; the text pipeline does not."""
    pytest.importorskip("transformers")
    return pytest.importorskip("torch")


def test_model_sizes_are_what_the_plan_says():
    _torch()
    from vpdl.slm.model import count_parameters

    assert 100e6 < count_parameters("small") < 120e6
    assert 320e6 < count_parameters("medium") < 360e6


def test_learning_rate_warms_up_then_decays_to_its_floor():
    pytest.importorskip("torch")
    from vpdl.slm.train import TrainConfig, learning_rate

    config = TrainConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1, micro_batch=1,
                         context=8, tokens_per_step=8, total_tokens=8 * 100)
    assert learning_rate(0, config) == pytest.approx(1e-4)
    assert learning_rate(9, config) == pytest.approx(1e-3)
    assert learning_rate(99, config) == pytest.approx(1e-4, rel=0.01)
    assert learning_rate(50, config) < learning_rate(20, config)


def _tiny_data(tmp_path):
    """A repeating token pattern: learnable in a few dozen steps."""
    from vpdl.slm.tokenizer import train_tokenizer

    data = tmp_path / "data"
    data.mkdir()
    train_tokenizer(_corpus_with_text(tmp_path), data / "tokenizer.json", vocab_size=400)
    pattern = np.tile(np.arange(10, 60, dtype=np.uint16), 400)
    pattern.tofile(data / "train.bin")
    pattern[:2000].tofile(data / "val.bin")
    return data


def _tiny_config(**changes):
    from vpdl.slm.train import TrainConfig
    settings = dict(size="tiny", context=32, micro_batch=4, tokens_per_step=256,
                    total_tokens=256 * 40, lr=3e-3, warmup_steps=5, eval_every=20,
                    eval_batches=2, checkpoint_every=10, log_every=5)
    settings.update(changes)
    return TrainConfig(**settings)


def test_training_from_random_weights_learns_and_exports(tmp_path):
    torch = _torch()
    from transformers import AutoModelForCausalLM

    from vpdl.slm.train import train

    data = _tiny_data(tmp_path)
    result = train(data, tmp_path / "run", _tiny_config(), device="cpu")
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < log[0]["loss"] - 1.0, "loss did not fall"
    assert "val_loss" in log[-1]
    model = AutoModelForCausalLM.from_pretrained(result["final"])
    out = model.generate(torch.tensor([[10, 11, 12]]), max_new_tokens=3, do_sample=False)
    assert out.shape == (1, 6)


def test_a_resumed_run_ends_with_exactly_the_weights_of_an_uninterrupted_one(tmp_path, monkeypatch):
    torch = _torch()
    import vpdl.slm.train as training

    data = _tiny_data(tmp_path)
    config = _tiny_config(total_tokens=256 * 20)
    training.train(data, tmp_path / "straight", config, device="cpu")

    real = training.learning_rate

    def crash_at_step_10(step, cfg):
        if step == 10:
            raise KeyboardInterrupt("power cut")
        return real(step, cfg)

    monkeypatch.setattr(training, "learning_rate", crash_at_step_10)
    with pytest.raises(KeyboardInterrupt):
        training.train(data, tmp_path / "resumed", config, device="cpu")
    monkeypatch.setattr(training, "learning_rate", real)
    training.train(data, tmp_path / "resumed", config, device="cpu")   # picks up at step 10

    first = torch.load(tmp_path / "straight" / "checkpoint.pt", weights_only=True)["model"]
    second = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=True)["model"]
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_resuming_with_different_settings_is_refused(tmp_path):
    torch = _torch()
    from vpdl.slm.train import train

    data = _tiny_data(tmp_path)
    train(data, tmp_path / "run", _tiny_config(total_tokens=256 * 10), device="cpu")
    with pytest.raises(ValueError, match="different configuration"):
        train(data, tmp_path / "run", _tiny_config(total_tokens=256 * 10, lr=1e-3), device="cpu")


def test_a_diverged_run_stops_instead_of_training_on(tmp_path, monkeypatch):
    torch = _torch()
    import vpdl.slm.train as training

    data = _tiny_data(tmp_path)
    monkeypatch.setattr(training, "_loss", lambda model, x, y: torch.tensor(float("nan"),
                                                                            requires_grad=True))
    with pytest.raises(RuntimeError, match="Loss became nan"):
        training.train(data, tmp_path / "run", _tiny_config(), device="cpu")


def test_missing_python_headers_become_one_instruction(monkeypatch, tmp_path):
    """First DGX run died inside Triton for want of Python.h."""
    _torch()
    import sysconfig

    import vpdl.slm.train as training

    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"include": str(tmp_path)})
    message = training.missing_python_headers()
    assert message and "sudo apt install python" in message and "-dev" in message
    (tmp_path / "Python.h").write_text("")
    assert training.missing_python_headers() is None


def test_benchmark_reports_speed_and_saves_nothing(tmp_path):
    torch = _torch()
    from vpdl.slm.train import train

    data = _tiny_data(tmp_path)
    result = train(data, tmp_path / "bench", _tiny_config(), device="cpu", benchmark_steps=4)
    assert result["tokens_per_s"] > 0 and result["projected_hours"] >= 0
    assert not (tmp_path / "bench" / "checkpoint.pt").exists()
