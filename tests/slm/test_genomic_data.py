"""Data layer: labels, variant identity, readers, tables, splits, roles, examples, leakage.

The fixture is a small synthetic ClinVar release in ClinVar's own file layouts
(``vpdl.slm.build.synthetic``) — invented variants, invented narratives.
"""

from __future__ import annotations

import functools
import gzip

import pandas as pd
import pytest

from vpdl.slm.build.clusters import document_clusters
from vpdl.slm.build.examples import build_examples, holdout_patterns, model_input
from vpdl.slm.build.features import FEATURES, StructuredEncoder, feature_frame
from vpdl.slm.build.leakage import AuditInputs, LeakageError, run_audit
from vpdl.slm.build.records import build_edges, build_records, load_tables, variant_view
from vpdl.slm.build.roles import assign_roles, pretraining_exclusions, reserved_variants
from vpdl.slm.build.splits import SCHEMES, SplitConfig, SplitError, make_split, split_frame, variant_groups
from vpdl.slm.build.stats import corpus_stats
from vpdl.slm.build.synthetic import HOLDOUT_PMID, write_synthetic_clinvar
from vpdl.slm.catalog import HOLDOUT_PUBLICATIONS, SOURCES, validate_catalog
from vpdl.slm.clinvar_text import parse_conditions, parse_date, specific_condition
from vpdl.slm.labels import normalize_classification
from vpdl.slm.parallel import pmap, run_windowed
from vpdl.slm.schema import TABLES, validate_table
from vpdl.slm.text.acmg import task_policy
from vpdl.slm.variants import DLJoin, consequence, parse_name


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("genomic")
    paths = write_synthetic_clinvar(root / "clinvar", variants_per_gene=8, seed=11)
    manifest = build_records(root / "records", paths["variant_summary"], paths["submission_summary"],
                             paths["var_citations"], workers=1)
    tables = load_tables(root / "records")
    return {"root": root, "paths": paths, "manifest": manifest, "tables": tables}


# -- labels ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,category,label", [
    ("Pathogenic", "five_class", "pathogenic"),
    ("Likely benign", "five_class", "likely_benign"),
    ("Uncertain significance", "five_class", "vus"),
    ("Pathogenic/Likely pathogenic", "pair", None),
    ("Benign/Likely benign", "pair", None),
    ("Conflicting classifications of pathogenicity", "conflicting", None),
    ("drug response", "out_of_scope", None),
    ("not provided", "missing", None),
    ("-", "missing", None),
])
def test_clinvar_words_map_to_the_five_classes(raw, category, label):
    info = normalize_classification(raw)
    assert (info.category, info.label5) == (category, label)


def test_a_paired_term_gets_a_soft_target_rather_than_a_coin_flip():
    info = normalize_classification("Pathogenic/Likely pathogenic")
    assert info.soft[:2] == (0.5, 0.5) and sum(info.soft) == 1.0
    assert info.binary == 1.0


def test_low_penetrance_and_vus_subtiers_keep_the_class_and_record_the_modifier():
    assert normalize_classification("Pathogenic, low penetrance").label5 == "pathogenic"
    assert "low_penetrance" in normalize_classification("Pathogenic, low penetrance").modifiers
    vus_high = normalize_classification("VUS-high")
    assert vus_high.label5 == "vus" and "vus_high" in vus_high.modifiers


def test_secondary_terms_become_modifiers_not_a_different_class():
    info = normalize_classification("Pathogenic; risk factor")
    assert info.label5 == "pathogenic" and "risk factor" in info.modifiers


# -- variant identity and consequence ----------------------------------------------------

@pytest.mark.parametrize("name,variant_type,expected", [
    ("NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)", "single nucleotide variant", "missense"),
    ("NM_000249.4(MLH1):c.1731G>A (p.Ser577=)", "single nucleotide variant", "synonymous"),
    ("NM_000251.3(MSH2):c.2131C>T (p.Arg711Ter)", "single nucleotide variant", "nonsense"),
    ("NM_000251.3(MSH2):c.1077-2A>G", "single nucleotide variant", "splice_site"),
    ("NM_000251.3(MSH2):c.1077+5G>A", "single nucleotide variant", "splice_region"),
    ("NM_000251.3(MSH2):c.1077+60G>A", "single nucleotide variant", "intronic"),
    ("NM_000249.4(MLH1):c.-27C>A", "single nucleotide variant", "utr5_or_upstream"),
    ("NM_000249.4(MLH1):c.*30G>T", "single nucleotide variant", "utr3_or_downstream"),
    ("NM_000179.3(MSH6):c.3261dup (p.Phe1088LeufsTer5)", "Duplication", "frameshift"),
    ("NM_000179.3(MSH6):c.3260_3262del (p.Phe1088del)", "Deletion", "inframe_indel"),
    ("NM_000249.4(MLH1):c.1A>G (p.Met1?)", "single nucleotide variant", "start_loss"),
    ("GRCh38/hg38 3p22.2(chr3:36993548-37050846)x1", "copy number loss", "cnv"),
])
def test_consequence_classes_cover_more_than_missense(name, variant_type, expected):
    assert consequence(variant_type, name) == expected


def test_a_clinvar_name_is_parsed_into_its_parts():
    parsed = parse_name("NM_000249.4(MLH1):c.199G>A (p.Gly67Arg)")
    assert (parsed.transcript, parsed.gene, parsed.change, parsed.protein) == \
        ("NM_000249.4", "MLH1", "c.199G>A", "p.Gly67Arg")


def test_dates_and_conditions_are_read_or_left_empty():
    assert parse_date("Dec 17, 2024") == "2024-12-17"
    assert parse_date("-") is None and parse_date("") is None
    pairs = parse_conditions("MedGen:C1333991,OMIM:609310|MedGen:C3661900", "Lynch syndrome 2|not provided")
    assert pairs == [("MedGen:C1333991", "Lynch syndrome 2"), ("MedGen:C3661900", "not provided")]
    assert specific_condition("Lynch syndrome 2") and not specific_condition("not provided")
    assert not specific_condition("Hereditary cancer-predisposing syndrome")


def test_the_dl_join_comes_from_the_dl_branchs_own_table(tmp_path):
    table = tmp_path / "canonical.csv"
    pd.DataFrame({"variant_id": ["P40692:67:G>R"], "clinvar_variation_ids": ["12345;67890"]}).to_csv(
        table, index=False)
    join = DLJoin.from_canonical(table)
    assert join.get(12345) == join.get(67890) == "P40692:67:G>R"
    assert join.get(1) is None and len(DLJoin.empty()) == 0


# -- records ------------------------------------------------------------------------------

def test_every_table_matches_its_schema(built):
    for name, frame in built["tables"].items():
        assert validate_table(frame, name) == [], name
    assert set(built["tables"]) == set(TABLES)


def test_the_manifest_records_inputs_hashes_counts_and_versions(built):
    manifest = built["manifest"]
    assert manifest["inputs"]["variant_summary"]["sha256"]
    assert manifest["counts"]["variants"] > 0 and manifest["counts"]["documents"] > 0
    assert manifest["schema_version"] and manifest["preprocessing_version"]
    assert manifest["evidence_provenance"].startswith("rule:")


def test_narratives_keep_their_own_submitter_date_and_classification(built):
    documents = built["tables"]["documents"]
    many = documents.groupby("variant_id")["document_id"].count().max()
    assert many > 1                              # several submissions per variant exist
    assert documents["submitter"].nunique() > 1
    assert documents["label5"].notna().any()


def test_omim_narratives_are_flagged_restricted(built):
    documents = built["tables"]["documents"]
    omim = documents.loc[documents["submitter"] == "OMIM"]
    assert len(omim) and omim["restricted"].all()
    assert not documents.loc[documents["submitter"] != "OMIM", "restricted"].any()


def test_the_variant_view_keeps_evidence_separate_from_the_aggregate(built):
    tables = built["tables"]
    view = variant_view(tables, tables["variants"].variant_id.iloc[0])
    assert view["aggregate"]["classification"]
    assert isinstance(view["clinical_interpretations"], list)
    assert "population_frequency_not_joined" in view["quality_flags"]


def test_relationships_are_derivable_as_edges(built):
    edges = build_edges(built["tables"])
    assert set(edges["predicate"]) >= {"in_gene", "asserted_for", "interprets", "cited_by"}


def test_statistics_measure_the_corpus(built):
    stats = corpus_stats(built["tables"])
    assert stats["documents"] == len(built["tables"]["documents"])
    assert stats["unique_genes"] >= 10
    assert stats["characters"]["median"] > 0
    assert stats["documents_with_direct_conclusion"] > 0
    assert 0 <= stats["mmr_share"]["documents"] <= 1
    assert set(stats["breakdowns"]) >= {"gene", "disease", "variant_type", "laboratory", "year"}


# -- every core, same answer ------------------------------------------------------------------

def _assert_same_tables(left, right):
    assert set(left) == set(right)
    for name in left:
        a, b = left[name].reset_index(drop=True), right[name].reset_index(drop=True)
        assert list(a.columns) == list(b.columns) and a.shape == b.shape, name
        for column in a.columns:
            assert a[column].astype(str).tolist() == b[column].astype(str).tolist(), (name, column)


def test_a_sharded_build_on_many_processes_writes_the_single_process_tables(built, tmp_path):
    paths = built["paths"]
    manifest = build_records(tmp_path / "records", paths["variant_summary"],
                             paths["submission_summary"], paths["var_citations"], workers=2,
                             variant_chunk_lines=17, submission_chunk_lines=13)
    assert manifest["workers"] == 2
    assert len(list((tmp_path / "records" / "documents.parquet").glob("part-*.parquet"))) > 1
    assert manifest["counts"] == built["manifest"]["counts"]
    _assert_same_tables(built["tables"], load_tables(tmp_path / "records"))


def test_rows_of_one_variant_far_apart_in_the_file_still_give_one_record(built, tmp_path):
    with gzip.open(built["paths"]["variant_summary"], "rt", encoding="utf-8") as handle:
        header, *rows = handle.read().splitlines()
    assembly = header.split("\t").index("Assembly")
    first = [r for r in rows if r.split("\t")[assembly] == "GRCh37"]
    shuffled = tmp_path / "variant_summary.txt.gz"
    with gzip.open(shuffled, "wt", encoding="utf-8") as handle:     # every GRCh37 row first
        handle.write("\n".join([header, *first, *[r for r in rows if r not in first]]) + "\n")
    manifest = build_records(tmp_path / "records", shuffled, workers=1, variant_chunk_lines=5)
    variants = load_tables(tmp_path / "records", ["variants"])["variants"]
    assert variants["variation_id"].is_unique
    assert set(variants["variation_id"]) == set(built["tables"]["variants"]["variation_id"])
    assert manifest["reader_stats"]["variant_summary_non_adjacent_duplicate"] > 0


def test_the_parallel_helpers_keep_input_order():
    items = list(range(300))
    assert pmap(str, items, workers=2, minimum=1) == [str(i) for i in items]
    seen: list = []
    assert run_windowed(str, iter(items), workers=2, window=3, on_result=seen.append) == len(items)
    assert seen == [str(i) for i in items]
    single: list = []
    run_windowed(str, ["only"], workers=4, on_result=single.append)      # one chunk: no pool
    assert single == ["only"]


def test_masking_examples_and_the_leakage_scan_match_on_many_processes(built, frame, monkeypatch):
    import vpdl.slm.build.clusters as clusters_module
    import vpdl.slm.build.examples as examples_module
    import vpdl.slm.build.leakage as leakage_module
    split = make_split(frame, SplitConfig("variant"))
    documents = built["tables"]["documents"]
    serial_clusters = document_clusters(documents, workers=1)
    serial = build_examples(built["tables"], split, "classify", HOLDOUT_PUBLICATIONS, workers=1)
    serial.loc[0, "input_text"] += " Therefore, this variant is classified as pathogenic."
    eager = functools.partial(pmap, minimum=1)          # force the pool even on tiny data
    for module in (clusters_module, examples_module, leakage_module):
        monkeypatch.setattr(module, "pmap", eager)
    pd.testing.assert_frame_equal(serial_clusters, document_clusters(documents, workers=2))
    parallel = build_examples(built["tables"], split, "classify", HOLDOUT_PUBLICATIONS, workers=2)
    parallel.loc[0, "input_text"] += " Therefore, this variant is classified as pathogenic."
    pd.testing.assert_frame_equal(serial, parallel)
    reports = [run_audit(AuditInputs("variant", serial, built["tables"]["variants"], documents,
                                     built["tables"]["citations"], workers=w)) for w in (1, 2)]
    assert repr(reports[0].findings) == repr(reports[1].findings)
    assert any(f.check == "conclusion_leakage" for f in reports[1].critical)


# -- splits -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frame(built):
    clusters = document_clusters(built["tables"]["documents"])
    return split_frame(built["tables"], clusters)


def test_every_scheme_runs_and_assigns_every_document(frame):
    for scheme in SCHEMES:
        kwargs = {}
        if scheme == "temporal":
            kwargs = {"cutoff": "2023-01-01"}
        if scheme == "functional":
            kwargs = {"functional_variants": ("P43246:1:A>C",)}
        split = make_split(frame, SplitConfig(scheme, **kwargs))
        assert len(split) == len(frame)
        assert set(split["split"]) <= {"train", "val", "test", "excluded", "broad_test",
                                       "mmr_train", "mmr_val"}


def test_a_variant_never_straddles_a_variant_isolated_split(frame):
    split = make_split(frame, SplitConfig("variant"))
    for _, rows in split.groupby("variant_group"):
        sides = set(rows["split"]) - {"excluded"}
        assert len(sides) <= 1


def test_variant_groups_join_two_spellings_of_one_change():
    frame = pd.DataFrame({
        "document_id": ["d1", "d2", "d3"], "variant_id": ["clinvar:1", "clinvar:2", "clinvar:3"],
        "gene": ["MLH1", "MLH1", "MSH2"], "hgvs_p": ["p.Gly67Arg", "p.Gly67Arg", "p.Ser1Leu"],
        "genomic_key": ["GRCh38:3:1:G>A", None, "GRCh38:2:9:C>T"],
        "protein_variant_id": [None, None, None]})
    groups = variant_groups(frame)
    assert groups[0] == groups[1] != groups[2]


def test_the_gene_holdout_keeps_the_reserved_genes_out_of_the_test_set(frame):
    split = make_split(frame, SplitConfig("gene", test_fraction=0.4))
    merged = split.merge(frame[["document_id", "gene"]], on="document_id")
    held_out = set(merged.loc[merged["split"] == "test", "gene"])
    assert held_out and not held_out & {"MLH1", "MSH2", "MSH6", "PMS2"}


def test_the_mmr_split_separates_broad_training_from_mmr_evaluation(frame):
    split = make_split(frame, SplitConfig("mmr"))
    merged = split.merge(frame[["document_id", "gene"]], on="document_id")
    mmr = merged["gene"].isin(["MLH1", "MSH2", "MSH6", "PMS2"])
    assert not merged.loc[~mmr, "split"].isin(["test", "mmr_train", "mmr_val"]).any()
    assert set(merged.loc[mmr, "split"]) <= {"test", "mmr_train", "mmr_val", "excluded"}


def test_the_temporal_split_never_trains_on_the_future(frame):
    split = make_split(frame, SplitConfig("temporal", cutoff="2023-01-01"))
    merged = split.merge(frame[["document_id", "date"]], on="document_id")
    train = merged.loc[merged["split"] == "train"]
    assert (train["date"] < "2023-01-01").all()
    assert merged.loc[merged["split"] == "test", "date"].ge("2023-01-01").all()


def test_undated_documents_are_excluded_from_a_temporal_split():
    frame = pd.DataFrame({"document_id": ["a", "b"], "variant_id": ["clinvar:1", "clinvar:2"],
                          "gene": ["G1", "G2"], "hgvs_p": [None, None], "genomic_key": [None, None],
                          "protein_variant_id": [None, None], "date": [None, "2024-01-01"],
                          "restricted": [False, False], "submitter": ["lab", "lab"],
                          "disease_names": [[], []]})
    split = make_split(frame, SplitConfig("temporal", cutoff="2023-01-01"))
    assert split.loc[split["document_id"] == "a", "split"].iloc[0] == "excluded"


def test_a_split_that_cannot_isolate_anything_fails_loudly():
    identical = "the same narrative repeated word for word in every record"
    frame = pd.DataFrame({"document_id": [f"d{i}" for i in range(6)],
                          "variant_id": [f"clinvar:{i}" for i in range(6)],
                          "gene": ["G"] * 6, "hgvs_p": [None] * 6, "genomic_key": [None] * 6,
                          "protein_variant_id": [None] * 6, "restricted": [False] * 6,
                          "submitter": ["lab"] * 6, "disease_names": [[]] * 6,
                          "near_dup_cluster": ["one"] * 6, "template_cluster": ["t"] * 6,
                          "text": [identical] * 6})
    with pytest.raises(SplitError, match="largest"):
        make_split(frame, SplitConfig("text"))


def test_the_laboratory_holdout_removes_a_test_variants_training_lab_documents(frame):
    split = make_split(frame, SplitConfig("laboratory", test_fraction=0.4, val_fraction=0.2))
    merged = split.merge(frame[["document_id", "submitter"]], on="document_id")
    for _, rows in merged.groupby("variant_group"):
        assert len(set(rows["split"]) - {"excluded"}) <= 1


# -- roles ---------------------------------------------------------------------------------

def test_roles_follow_the_split_and_independent_data_overrides_it(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    roles = assign_roles(split)
    assert set(roles) <= {"TRAINING", "VALIDATION", "TEST", "EXCLUDED"}
    first = split["variant_id"].iloc[0]
    assert assign_roles(split, [first]).iloc[0] == "INDEPENDENT_VALIDATION"


def test_pretraining_exclusions_list_reserved_narratives_and_their_publications(built, frame):
    split = make_split(frame, SplitConfig("mmr"))
    reserved = reserved_variants({"mmr": split})
    exclusions = pretraining_exclusions(built["tables"], reserved, HOLDOUT_PUBLICATIONS)
    assert exclusions["documents"] and exclusions["reserved_variants"] == len(reserved)
    assert HOLDOUT_PMID in exclusions["pmids"]


# -- examples and their leakage policy ----------------------------------------------------

def test_classification_inputs_have_no_verdict_and_no_codes(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify", HOLDOUT_PUBLICATIONS)
    assert len(examples)
    assert examples["conclusions_removed"].sum() > 0
    for text in examples["input_text"]:
        assert "classified as" not in text.lower()
        assert "PM2" not in text and "BA1" not in text
    assert set(examples["target_label"].dropna()) <= {"pathogenic", "likely_pathogenic", "vus",
                                                      "likely_benign", "benign"}


def test_restricted_documents_never_become_examples(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    restricted = set(built["tables"]["documents"].loc[
        built["tables"]["documents"]["restricted"], "document_id"])
    assert not set(examples["document_id"]) & restricted


def test_the_codes_task_keeps_the_codes_as_a_target_not_an_input(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "acmg_codes")
    assert len(examples) and examples["target_codes"].map(len).sum() > 0
    assert all("PM2" not in text for text in examples["input_text"])


def test_interpreting_stated_codes_is_a_separate_task_that_may_see_them(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify_from_codes")
    assert len(examples) and any("PM2" in text or "BS3" in text for text in examples["input_text"])
    assert task_policy("classify_from_codes").notes


def test_a_holdout_publication_is_removed_from_the_input():
    policy = task_policy("classify")
    text = ("The variant is absent from gnomAD. A multiplexed assay reported loss of function "
            f"(PMID: {HOLDOUT_PMID}). Therefore it is pathogenic.")
    masked, stats = model_input(text, policy, holdout_patterns(HOLDOUT_PUBLICATIONS))
    assert HOLDOUT_PMID not in masked and stats.holdout_removed == 1
    assert "absent from gnomAD" in masked


def test_forbidden_features_cannot_be_asked_for(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    with pytest.raises(ValueError, match="leakage policy"):
        build_examples(built["tables"], split, "classify", features=["review_status"])


# -- structured features ---------------------------------------------------------------------

def test_the_encoder_is_fitted_on_training_rows_only(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    raw = feature_frame(examples, built["tables"]["variants"], built["tables"]["documents"])
    encoder = StructuredEncoder().fit(raw, examples["split"])
    assert encoder.fitted_on == "train"
    encoded = encoder.transform(raw)
    assert encoded["categorical"].shape[0] == len(examples)
    assert encoded["numeric_mask"].shape == encoded["numeric"].shape
    unseen = raw.assign(consequence="a_consequence_never_seen")
    assert (encoder.transform(unseen)["categorical"][:, 0] == 0).all()     # <unk>, not a crash


def test_label_describing_features_are_refused_by_the_encoder():
    with pytest.raises(ValueError, match="forbidden"):
        StructuredEncoder(("review_status",))
    assert not FEATURES["gene"].default          # gene identity is an ablation, not a default


# -- leakage audit --------------------------------------------------------------------------

def _audit(built, split, examples, **kwargs):
    clusters = document_clusters(built["tables"]["documents"])
    return run_audit(AuditInputs(
        scheme=kwargs.pop("scheme", "variant"),
        examples=examples.merge(clusters, on="document_id", how="left"),
        variants=built["tables"]["variants"], documents=built["tables"]["documents"],
        citations=built["tables"]["citations"], holdout_publications=HOLDOUT_PUBLICATIONS, **kwargs))


def test_a_clean_variant_split_has_no_critical_finding(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify", HOLDOUT_PUBLICATIONS)
    report = _audit(built, split, examples, features=("consequence",))
    assert report.critical == []
    assert {f.check for f in report.findings} >= {"conclusion_leakage", "variant_duplicates",
                                                  "literature_leakage", "gene_shortcut"}


def test_a_verdict_left_in_an_input_is_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    examples.loc[0, "input_text"] += " Therefore, this variant is classified as pathogenic."
    report = _audit(built, split, examples)
    assert any(f.check == "conclusion_leakage" and f.severity == "critical" for f in report.findings)
    with pytest.raises(LeakageError):
        report.assert_clean()


def test_a_label_describing_feature_is_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    report = _audit(built, split, examples, features=("review_status",))
    assert any(f.check == "label_leakage" and f.severity == "critical" for f in report.findings)


def test_a_variant_on_both_sides_is_critical_except_under_the_random_scheme(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    counts = examples.loc[examples["split"] == "train", "variant_id"].value_counts()
    train_variant = counts[counts >= 2].index[0]          # a variant with several submissions
    leaked = examples.copy()
    rows = leaked.index[leaked["variant_id"] == train_variant]
    leaked.loc[rows[0], "split"] = "test"                 # one of them moves to the test side
    strict = _audit(built, split, leaked, scheme="variant")
    assert any(f.check == "variant_duplicates" and f.severity == "critical" for f in strict.findings)
    lenient = _audit(built, split, leaked, scheme="random")
    assert all(f.severity != "critical" for f in lenient.findings if f.check == "variant_duplicates")


def test_functional_holdout_leakage_is_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")     # no holdout filtering
    report = _audit(built, split, examples)
    finding = next(f for f in report.findings if f.check == "functional_leakage")
    assert finding.severity == "critical" and "holdout functional publication" in finding.message


def test_teacher_outputs_for_evaluation_rows_are_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    evaluated = examples.loc[examples["split"] == "test", "example_id"].iloc[:1]
    report = _audit(built, split, examples, teacher=pd.DataFrame({"example_id": evaluated}))
    assert any(f.check == "teacher_contamination" and f.severity == "critical" for f in report.findings)


def test_evaluation_narratives_in_the_pretraining_corpus_are_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    evaluated = examples.loc[examples["split"] == "test", "document_id"].tolist()[:2]
    report = _audit(built, split, examples,
                    pretraining={"kind": "corpus", "documents": evaluated, "pmids": []})
    assert any(f.check == "benchmark_contamination" and f.severity == "critical"
               for f in report.findings)


def test_an_exclusion_list_built_for_another_split_is_caught(built, frame):
    """The list says what pretraining must NOT hold; it has to cover THIS split."""
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    exclusions = pretraining_exclusions(built["tables"], reserved_variants({"variant": split}),
                                        HOLDOUT_PUBLICATIONS)
    covering = _audit(built, split, examples, pretraining=exclusions)
    assert all(f.severity != "critical" for f in covering.findings
               if f.check == "benchmark_contamination")
    other_split = make_split(frame, SplitConfig("mmr"))
    other = pretraining_exclusions(built["tables"], reserved_variants({"mmr": other_split}),
                                   HOLDOUT_PUBLICATIONS)
    report = _audit(built, split, examples, pretraining=other)
    finding = next(f for f in report.findings if f.check == "benchmark_contamination")
    assert finding.severity == "critical" and "different split" in finding.message


def test_a_retrieval_index_holding_evaluation_documents_is_critical(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    evaluated = examples.loc[examples["split"] == "test", "document_id"].tolist()[:1]
    report = _audit(built, split, examples, retrieval_documents=evaluated)
    assert any(f.check == "retrieval_contamination" and f.severity == "critical"
               for f in report.findings)


def test_the_gene_shortcut_is_measured_and_reported(built, frame):
    split = make_split(frame, SplitConfig("variant"))
    examples = build_examples(built["tables"], split, "classify")
    report = _audit(built, split, examples)
    finding = next(f for f in report.findings if f.check == "gene_shortcut")
    assert "auc" in finding.details


# -- the catalogue ------------------------------------------------------------------------

def test_the_source_catalogue_is_consistent():
    assert validate_catalog() == []
    assert {s.source_id for s in SOURCES} >= {"clinvar_variant_summary", "clinvar_submission_summary",
                                              "clingen_erepo", "gnomad", "pubmed", "genereviews"}
    for source in SOURCES:
        assert source.license, source.source_id


def test_independent_validation_sources_carry_the_publications_to_keep_out():
    assert f"PMID:{HOLDOUT_PMID}" in HOLDOUT_PUBLICATIONS
