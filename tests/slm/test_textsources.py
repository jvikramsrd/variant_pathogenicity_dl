"""Wider pretraining text: MedlinePlus, Orphanet, MONDO, UniProt, PubMed Central.

Fixtures copy the structure of the real files as read from the providers on
2026-09-24 (docs/slm/TEXT_SOURCES.md). The PMC downloader runs against a fake
HTTP layer — no network.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
from collections import Counter

import pytest

from vpdl.slm import pmc
from vpdl.slm.textsources import (iter_medlineplus, iter_mondo, iter_orphanet, iter_pmc,
                                  iter_text_sources, iter_uniprot_text, normalise_licence,
                                  parse_jats)

MEDLINEPLUS = """<?xml version="1.0" encoding="utf-8"?>
<summaries xmlns="https://medlineplus.gov/download/ghr-summaries-20250602.xsd" xmlns:html="http://www.w3.org/1999/xhtml">
<health-condition-summary id="20395"><name>Lynch syndrome</name>
<ghr-page>https://medlineplus.gov/genetics/condition/lynch-syndrome</ghr-page>
<text-list><text><text-role>description</text-role><html><html:p>Lynch syndrome, often called hereditary nonpolyposis colorectal cancer (HNPCC), is an inherited disorder.</html:p><html:p>People with Lynch syndrome may have colon polyps.</html:p></html></text></text-list>
<related-gene-list><related-gene><gene-symbol>MLH1</gene-symbol></related-gene></related-gene-list>
</health-condition-summary>
<gene-summary id="21777"><gene-symbol>MLH1</gene-symbol><name>mutL homolog 1</name>
<text-list><text><text-role>function</text-role><html><html:p>The MLH1 gene provides instructions for making a protein that plays an essential role in repairing DNA.</html:p></html></text></text-list>
</gene-summary>
<gene-summary id="1"><gene-symbol>EMPTY</gene-symbol><name>no text</name></gene-summary>
</summaries>"""

ORPHANET = """<?xml version="1.0" encoding="UTF-8"?>
<JDBOR date="2026-06-23 07:53:50" copyright="Orphanet (c) 2026">
  <Availability><Licence><FullName lang="en">Creative Commons Attribution 4.0 International</FullName>
    <ShortIdentifier>CC-BY-4.0</ShortIdentifier></Licence></Availability>
  <DisorderList count="2">
    <Disorder id="17601"><OrphaCode>166024</OrphaCode>
      <Name lang="en">Multiple epiphyseal dysplasia-macrocephaly-facial dysmorphism syndrome</Name>
      <SynonymList count="1"><Synonym lang="en">Multiple epiphyseal dysplasia, Al-Gazali type</Synonym></SynonymList>
      <SummaryInformationList count="1"><SummaryInformation id="63385" lang="en"><TextSectionList count="1">
        <TextSection id="83697" lang="en"><TextSectionType id="16907"><Name lang="en">Definition</Name></TextSectionType>
          <Contents>A rare primary bone dysplasia with &lt;i&gt;genu valgum&lt;/i&gt; and macrocephaly.</Contents>
        </TextSection></TextSectionList></SummaryInformation></SummaryInformationList>
    </Disorder>
    <Disorder id="17603"><OrphaCode>166032</OrphaCode><Name lang="en">No text here</Name>
      <SummaryInformationList count="0"></SummaryInformationList></Disorder>
  </DisorderList>
</JDBOR>"""

MONDO = """format-version: 1.2
ontology: mondo

[Term]
id: MONDO:0005835
name: Lynch syndrome
def: "An autosomal dominant \\"hereditary\\" cancer syndrome caused by germline mismatch repair defects." [PMID:123]
synonym: "HNPCC" EXACT []
synonym: "colon cancer thing" RELATED []

[Term]
id: MONDO:0000001
name: disease
is_obsolete: true
def: "obsolete term" []

[Term]
id: MONDO:0000002
name: no definition here

[Typedef]
id: part_of
name: part of
"""

UNIPROT = ("Entry\tGene Names (primary)\tProtein names\tFunction [CC]\tInvolvement in disease\n"
           "P40692\tMLH1\tDNA mismatch repair protein Mlh1\tFUNCTION: Heterodimerizes with PMS2 to "
           "form MutL alpha. {ECO:0000269|PubMed:16873062, ECO:0000269|PubMed:18206974}.\t"
           "DISEASE: Lynch syndrome 2 (LYNCH2) [MIM:609310]: A form of Lynch syndrome. "
           "{ECO:0000269|PubMed:1}.\n"
           "Q00000\tNONE\tUncharacterised\t\t\n")

JATS = """<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.4//EN" "JATS-archivearticle1-4.dtd">
<article xml:lang="en" article-type="case-report"><front><article-meta>
<title-group><article-title>A germline <italic>MLH1</italic> variant in a family</article-title></title-group>
<abstract><p>We describe a family with Lynch syndrome.</p></abstract>
<abstract abstract-type="graphical"><p>graphical abstract</p></abstract>
</article-meta></front>
<body>
<sec><title>Introduction</title><p>Mismatch repair deficiency causes cancer <xref ref-type="bibr" rid="B1">[1]</xref>.</p>
<fig id="F1"><label>Figure 1</label><caption><p>Pedigree caption that is not running text.</p></caption></fig>
<sec><title>Case</title><p>The proband carried the variant and had colorectal cancer at 41.</p>
<table-wrap><table><tr><td>table cell text</td></tr></table></table-wrap></sec></sec>
</body>
<back><ref-list><ref><mixed-citation>Reference text must not appear.</mixed-citation></ref></ref-list></back>
</article>"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_medlineplus_summaries_become_one_document_each(tmp_path):
    stats = Counter()
    docs = list(iter_medlineplus(_write(tmp_path, "ghr.xml", MEDLINEPLUS), stats))
    assert [d["id"] for d in docs] == ["medlineplus:health-condition:20395", "medlineplus:gene:21777"]
    assert docs[0]["text"].startswith("Lynch syndrome\nDescription: Lynch syndrome, often called")
    assert "colon polyps" in docs[0]["text"] and "<" not in docs[0]["text"]
    assert docs[1]["text"].startswith("MLH1 (mutL homolog 1)\nFunction:")
    assert stats == Counter({"medlineplus_kept": 2, "medlineplus_no_text": 1})


def test_orphanet_definitions_keep_the_licence_and_drop_markup(tmp_path):
    stats = Counter()
    docs = list(iter_orphanet(_write(tmp_path, "en_product1.xml", ORPHANET), stats))
    assert len(docs) == 1 and docs[0]["id"] == "orpha:166024" and docs[0]["licence"] == "CC-BY-4.0"
    assert "Definition: A rare primary bone dysplasia with genu valgum" in docs[0]["text"]
    assert stats["orphanet_no_text"] == 1


def test_orphanet_refuses_a_release_under_another_licence(tmp_path):
    path = _write(tmp_path, "en_product1.xml", ORPHANET.replace("CC-BY-4.0", "CC-BY-NC-4.0"))
    with pytest.raises(ValueError, match="licence"):
        list(iter_orphanet(path, Counter()))


def test_mondo_keeps_live_defined_terms_with_exact_synonyms(tmp_path):
    stats = Counter()
    docs = list(iter_mondo(_write(tmp_path, "mondo.obo", MONDO), stats))
    assert [d["id"] for d in docs] == ["mondo:0005835"]
    assert docs[0]["text"] == ('Lynch syndrome. Also known as: HNPCC.\nAn autosomal dominant "hereditary" '
                               "cancer syndrome caused by germline mismatch repair defects.")
    assert stats["mondo_obsolete"] == 1 and stats["mondo_no_definition"] == 1


def test_uniprot_text_drops_evidence_tags_and_empty_entries(tmp_path):
    stats = Counter()
    docs = list(iter_uniprot_text(_write(tmp_path, "uniprot.tsv", UNIPROT), stats))
    assert len(docs) == 1 and docs[0]["id"] == "uniprot:P40692"
    assert "ECO:" not in docs[0]["text"] and "DISEASE: Lynch syndrome 2" in docs[0]["text"]
    assert docs[0]["text"].startswith("MLH1 (DNA mismatch repair protein Mlh1)")
    assert stats["uniprot_no_text"] == 1


def test_uniprot_text_names_the_columns_it_needs(tmp_path):
    with pytest.raises(ValueError, match="Function"):
        list(iter_uniprot_text(_write(tmp_path, "bad.tsv", "Entry\tSequence\nP1\tMKV\n"), Counter()))


def test_jats_keeps_running_text_and_drops_citations_floats_and_references():
    parsed = parse_jats(JATS.encode())
    assert parsed["title"] == "A germline MLH1 variant in a family"
    assert parsed["abstract"] == "We describe a family with Lynch syndrome."
    assert parsed["article_type"] == "case-report"
    body = parsed["body"]
    assert "Introduction" in body and "Case" in body and "colorectal cancer at 41" in body
    assert "causes cancer ." in body                                   # "[1]" citation removed
    for absent in ("Pedigree caption", "table cell", "Reference text", "graphical abstract"):
        assert absent not in body


@pytest.mark.parametrize("code, expected", [
    ("CC BY", "ccby"), ("cc-by", "ccby"), ("CC BY 4.0", "ccby"), ("by", "ccby"), ("CC0", "cc0"),
    ("CC0 1.0", "cc0"), ("CC BY-SA", "ccbysa"), ("CC BY-NC", "ccbync"), ("CC BY-NC-ND", "ccbyncnd"),
    ("CC BY-ND", "ccbynd"), ("TDM", "tdm"), (None, "")])
def test_licence_codes_are_normalised(code, expected):
    assert normalise_licence(code) == expected


def _pmc_article(root, pmcid, version=1, licence="CC BY", pmid=111, retracted=False, xml=JATS):
    shard = root / pmcid[-2:]
    shard.mkdir(parents=True, exist_ok=True)
    name = f"{pmcid}.{version}"
    (shard / f"{name}.json").write_text(json.dumps({
        "pmcid": pmcid, "version": version, "pmid": pmid, "license_code": licence,
        "is_retracted": retracted, "is_manuscript": False}), encoding="utf-8")
    (shard / f"{name}.xml.gz").write_bytes(gzip.compress(xml.encode()))


def test_pmc_articles_are_filtered_by_licence_and_retraction(tmp_path):
    long_body = JATS.replace("had colorectal cancer at 41.", "had colorectal cancer at 41. " * 20)
    _pmc_article(tmp_path, "PMC100", xml=long_body)
    _pmc_article(tmp_path, "PMC101", licence="CC BY-NC", xml=long_body)
    _pmc_article(tmp_path, "PMC102", retracted=True, xml=long_body)
    _pmc_article(tmp_path, "PMC103", pmid=None, xml=long_body)
    stats = Counter()
    docs = list(iter_pmc(tmp_path, stats))
    assert [d["id"] for d in docs] == ["pmc:PMC100", "pmc:PMC103"]
    assert docs[0]["pmid"] == "111" and docs[0]["licence"] == "ccby"
    assert "We describe a family" not in docs[0]["text"]        # PubMed already has this abstract
    assert "We describe a family" in docs[1]["text"]            # no PMID: keep it
    assert stats["pmc_licence_excluded_ccbync"] == 1 and stats["pmc_retracted"] == 1


def test_the_published_version_is_chosen_over_a_manuscript_and_nc_is_refused():
    metas = [{"version": 1, "license_code": "CC BY", "is_manuscript": True, "is_retracted": False},
             {"version": 2, "license_code": "CC BY", "is_manuscript": False, "is_retracted": False}]
    assert pmc.choose_version(metas)[0]["version"] == 2
    assert pmc.choose_version([{"version": 1, "license_code": "CC BY-NC"}]) == (None, "licence:ccbync")
    assert pmc.choose_version([{"version": 1, "license_code": "CC BY", "is_retracted": True}]) == \
        (None, "retracted")


class FakeHttp:
    """The S3 bucket and ESearch, in memory: 3 ids released per day from 2024-01-01 to 2024-01-10."""

    def __init__(self):
        self.calls = Counter()
        self.days = {dt.date(2024, 1, 1) + dt.timedelta(days=i): [str(1000 + 3 * i + k) for k in range(3)]
                     for i in range(10)}

    def __call__(self, url: str) -> bytes:
        if url.startswith(pmc.ESEARCH):
            self.calls["esearch"] += 1
            from urllib.parse import parse_qs, urlparse
            term = parse_qs(urlparse(url).query)["term"][0]
            match = re.search(r"(\d{4}/\d\d/\d\d):(\d{4}/\d\d/\d\d)\[pmcrdat\]", term)
            if match:
                low, high = (dt.datetime.strptime(x, "%Y/%m/%d").date() for x in match.groups())
                ids = [i for day, day_ids in self.days.items() if low <= day <= high for i in day_ids]
            else:
                ids = [i for day_ids in self.days.values() for i in day_ids]
            retmax = int(parse_qs(urlparse(url).query)["retmax"][0])
            return json.dumps({"esearchresult": {"count": str(len(ids)),
                                                 "idlist": ids[:retmax]}}).encode()
        if "list-type=2" in url and "inventory-reports" in url:
            self.calls["inventory"] += 1
            return (b"<ListBucketResult><CommonPrefixes><Prefix>inventory-reports/pmc-oa-opendata/"
                    b"metadata/2026-09-23T01-00Z/</Prefix></CommonPrefixes></ListBucketResult>")
        if url.endswith("manifest.json"):
            return json.dumps({"files": [{"key": "inventory-reports/pmc-oa-opendata/metadata/data/a.csv.gz"}]}).encode()
        if url.endswith("a.csv.gz"):
            # Every article has version 1 except PMC1029 (absent from the bucket); PMC1000 also has .2.
            rows = [f'"pmc-oa-opendata","metadata/PMC{n}.1.json","2026-09-01","x"'
                    for day_ids in self.days.values() for n in day_ids if n != "1029"]
            rows.append('"pmc-oa-opendata","metadata/PMC1000.2.json","2026-09-01","x"')
            return gzip.compress("\n".join(rows).encode())
        if "list-type=2" in url:
            self.calls["list"] += 1
            prefix = re.search(r"prefix=(PMC\d+)\.", url).group(1)
            return (f"<ListBucketResult><CommonPrefixes><Prefix>{prefix}.1/</Prefix></CommonPrefixes>"
                    f"</ListBucketResult>").encode()
        if "/metadata/" in url:
            self.calls["metadata"] += 1
            version = url.rsplit("/", 1)[1].removesuffix(".json")
            number = int(version[3:].split(".")[0])
            return json.dumps({"pmcid": version.split(".")[0], "version": 1, "pmid": number,
                               "license_code": "CC BY-NC" if number == 1001 else "CC BY",
                               "is_retracted": False, "is_manuscript": False}).encode()
        if url.endswith(".xml"):
            self.calls["xml"] += 1
            return JATS.encode()
        raise pmc.NotFound(url)


def test_search_splits_the_date_range_under_the_esearch_cap(monkeypatch):
    monkeypatch.setattr(pmc, "MAX_IDS", 7)
    http = FakeHttp()
    ids, info = pmc.search_ids("topic", http, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 10))
    assert len(ids) == 30 == info["query_total"] == info["found_by_date"]
    assert ids[0] == "PMC1000" and http.calls["esearch"] > 2


def test_download_fetches_licence_clear_articles_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(pmc, "MAX_IDS", 7)
    http = FakeHttp()
    first = pmc.download(tmp_path, "topic", workers=4, limit=6, http=http, use_inventory=False)
    assert first["status"] == {"ok": 5, "licence_excluded": 1}
    assert len(list(tmp_path.glob("*/PMC*.xml.gz"))) == 5
    assert "NLM" in first["citation"] or "PubMed Central" in first["citation"]
    xml_calls = http.calls["xml"]
    again = pmc.download(tmp_path, "topic", workers=4, limit=6, http=http, use_inventory=False)
    assert again["status"]["exists"] == 5 and http.calls["xml"] == xml_calls   # nothing re-fetched
    assert (tmp_path / "ids.txt").read_text().split()[:2] == ["PMC1000", "PMC1001"]


def test_the_pretraining_corpus_takes_the_new_sources_and_drops_evaluation_papers(tmp_path):
    from vpdl.slm.build.pretrain_corpus import build_pretrain_corpus
    long_body = JATS.replace("had colorectal cancer at 41.", "had colorectal cancer at 41. " * 20)
    _pmc_article(tmp_path / "pmc", "PMC100", pmid=111, xml=long_body)
    _pmc_article(tmp_path / "pmc", "PMC200", pmid=222, xml=long_body)
    sources = {"medlineplus": _write(tmp_path, "ghr.xml", MEDLINEPLUS),
               "orphanet": _write(tmp_path, "orpha.xml", ORPHANET),
               "mondo": _write(tmp_path, "mondo.obo", MONDO),
               "uniprot_text": _write(tmp_path, "uniprot.tsv", UNIPROT), "pmc": tmp_path / "pmc"}
    summary = build_pretrain_corpus(tmp_path / "corpus", exclusions={"pmids": ["222"], "documents": []},
                                    text_sources=sources)
    written = [json.loads(line)["id"] for path in (tmp_path / "corpus").glob("*.jsonl")
               for line in path.read_text(encoding="utf-8").splitlines()]
    assert "pmc:PMC100" in written and "pmc:PMC200" not in written
    assert {"medlineplus_genetics", "orphanet", "mondo", "uniprot_text", "pmc"} <= set(summary["sources"])
    assert summary["decisions"]["pmc_excluded_cited_by_evaluation_variants"] == 1


def test_an_unknown_text_source_is_an_error():
    with pytest.raises(ValueError, match="unknown text sources"):
        list(iter_text_sources({"omim": "x"}, Counter()))


def test_the_cli_reports_each_source_before_a_build(tmp_path, capsys):
    from vpdl.slm.cli import main
    code = main(["text-sources", "--medlineplus", str(_write(tmp_path, "ghr.xml", MEDLINEPLUS)),
                 "--mondo", str(_write(tmp_path, "mondo.obo", MONDO))])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["medlineplus"]["documents_read"] == 2
    assert report["mondo"]["decisions"]["mondo_kept"] == 1


def test_the_inventory_replaces_per_article_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(pmc, "MAX_IDS", 7)
    http = FakeHttp()
    manifest = pmc.download(tmp_path, "topic", workers=4, http=http)
    assert http.calls["list"] == 0 and http.calls["inventory"] == 1
    assert manifest["search"]["inventory"] == "2026-09-23T01-00Z"
    assert manifest["status"]["no_version"] == 1                        # PMC1029: not in the bucket
    assert manifest["status"]["ok"] == 28 and manifest["status"]["licence_excluded"] == 1
    versions = json.loads((tmp_path / "versions.json").read_text())
    assert versions["PMC1000"] == ["PMC1000.1", "PMC1000.2"]
    calls = dict(http.calls)
    pmc.download(tmp_path, "topic", workers=4, http=http)                # resume: cached, nothing new
    assert http.calls["inventory"] == calls["inventory"] and http.calls["xml"] == calls["xml"]


def test_the_http_client_reuses_one_connection_per_thread():
    import http.server
    import threading

    connections = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            connections.append(self.client_address)
            super().setup()

        def do_GET(self):
            body = b"missing" if self.path.startswith("/missing") else f"ok {self.path}".encode()
            self.send_response(404 if self.path.startswith("/missing") else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = pmc.Http(retries=2)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert [client(f"{base}/a{i}") for i in range(5)] == [f"ok /a{i}".encode() for i in range(5)]
        with pytest.raises(pmc.NotFound):
            client(f"{base}/missing")
        assert len(connections) == 1                                      # one TCP connection, reused
    finally:
        server.shutdown()
