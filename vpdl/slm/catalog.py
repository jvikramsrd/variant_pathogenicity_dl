"""The knowledge-source catalogue: every source, its role, terms and leakage risk.

Knowledge base is not training data. A source can inform retrieval or
explanations (KNOWLEDGE_BASE), shape the language model (PRETRAINING), supply
supervised labels (TRAINING / VALIDATION / TEST), stay untouched until the end
(INDEPENDENT_VALIDATION) or only be consulted by people and code
(REFERENCE_ONLY) — and each of those has different leakage consequences.

Facts here were checked, not remembered: licences against the providers'
terms pages (docs/kb/SOURCES.md, 2026-09-21), Hugging Face licence tags and
ClinVar column names on 2026-09-22, local files by sha256 (``vpdl-slm catalog
--hash``). ``status`` says honestly where each source stands:

    local          on this machine (hash computed on request)
    on_dgx         built or downloaded on the DGX, not on this PC
    not_downloaded reader exists or is planned; nothing downloaded yet
    not_incorporated  deliberately not used (terms, access, or no data)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from vpdl.slm.schema import DATA_ROLES

__all__ = ["Source", "SOURCES", "source", "validate_catalog", "catalog_markdown",
           "catalog_json", "STATUSES", "LEAKAGE_RISKS", "HOLDOUT_PUBLICATIONS"]

STATUSES = ("local", "on_dgx", "not_downloaded", "not_incorporated", "generated")
LEAKAGE_RISKS = ("none", "low", "medium", "high", "critical")
PREPROCESSING_VERSION = "slm-genomic-prep/1"


@dataclass(frozen=True)
class Source:
    source_id: str
    source_name: str
    source_type: str                    # clinical | population | functional | literature |
                                        # annotation | phenotype | standard | model_output
    intended_roles: tuple[str, ...]
    license: str
    access_method: str
    status: str
    biological_scope: str
    gene_scope: str = "all"
    disease_scope: str = "all"
    variant_scope: str = "all"
    evidence_types: tuple[str, ...] = ()
    label_types: tuple[str, ...] = ()
    provenance: str = ""
    leakage_risk: str = "low"
    leakage_notes: str = ""
    quality_level: str = ""
    update_frequency: str = ""
    version: str | None = None
    release_date: str | None = None
    access_date: str | None = None
    local_paths: tuple[str, ...] = ()
    source_hash: str | None = None
    preprocessing_version: str = PREPROCESSING_VERSION
    publications: tuple[str, ...] = ()   # identifiers whose text must not reach training
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


S = Source
SOURCES: tuple[Source, ...] = (
    # -- clinical -------------------------------------------------------------
    S("clinvar_variant_summary", "ClinVar variant_summary", "clinical",
      ("TRAINING", "VALIDATION", "TEST", "KNOWLEDGE_BASE"), "public domain (NCBI)",
      "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz", "local",
      "aggregate germline classification per variant, conditions, location",
      evidence_types=("clinical interpretation",), label_types=("five_class", "aggregate", "conflicting"),
      provenance="NCBI ClinVar monthly release; one row per variant per assembly",
      leakage_risk="high",
      leakage_notes="The aggregate classification IS a label: never an input. Same variant under "
                    "several VariationIDs/transcripts is tied by genomic_key.",
      quality_level="review stars 0-4", update_frequency="monthly",
      access_date="2026-08-28 (file mtime)", local_paths=("data/raw/variant_summary.txt.gz",),
      source_hash="7e5f0c798858831cf15051f257c80fa5c90611ba982e1ebf89c1ff7787d5d101",
      notes="Measured 2026-09-22: 4,558,681 variants, 35,013 gene symbols. The v1 MMR manifest "
            "recorded a different sha256 (186ad283...) — an earlier release."),
    S("clinvar_submission_summary", "ClinVar submission_summary (SCV narratives)", "clinical",
      ("TRAINING", "VALIDATION", "TEST"), "public domain (NCBI); OMIM-submitted text carries "
      "OMIM's terms (flagged restricted)",
      "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz",
      "not_downloaded", "one row per submission: classification, date, submitter, free-text basis",
      evidence_types=("lab narrative", "expert-panel summary", "literature-only summary"),
      label_types=("five_class per submission",),
      provenance="Description = 'an optional free text description of the basis of the "
                 "interpretation' (NCBI README, checked 2026-09-22)",
      leakage_risk="critical",
      leakage_notes="Narratives state their own verdict (conclusion masking), repeat lab templates "
                    "(near-duplicate clusters), and cite the same papers as other variants "
                    "(literature leakage).",
      quality_level="tiers 1-5 (vpdl.slm.text.quality)", update_frequency="monthly"),
    S("clinvar_var_citations", "ClinVar var_citations", "literature",
      ("REFERENCE_ONLY",), "public domain (NCBI)",
      "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/var_citations.txt",
      "not_downloaded", "variant -> PubMed / PMC / Bookshelf citation",
      provenance="columns AlleleID, VariationID, rs, nsv, citation_source, citation_id (README)",
      leakage_risk="medium",
      leakage_notes="Used to BLOCK leakage: publications cited by evaluation variants are removed "
                    "from pretraining text in strict mode.", update_frequency="monthly"),
    S("clingen_erepo", "ClinGen Evidence Repository (VCEP curations)", "clinical",
      ("TRAINING", "TEST", "KNOWLEDGE_BASE"), "CC0 (ClinGen terms of use; docs/kb/SOURCES.md)",
      "https://erepo.clinicalgenome.org/evrepo/ (tab-delimited download)", "not_downloaded",
      "expert-panel variant interpretations with met / not-met ACMG codes and a summary",
      evidence_types=("expert curation", "ACMG codes"),
      label_types=("five_class", "acmg_met", "acmg_not_met"), leakage_risk="high",
      leakage_notes="Same variants as ClinVar expert-panel SCVs: joined by ClinVar VariationID "
                    "and split together. Codes are targets for the acmg task, masked elsewhere.",
      quality_level="tier 1", notes="Column names NOT verified (documentation unreachable "
                                    "2026-09-22); vpdl.slm.erepo reads by a column map that "
                                    "must be confirmed against the first download."),
    S("clingen_cspec", "ClinGen Criteria Specification Registry", "standard",
      ("REFERENCE_ONLY",), "ClinGen terms (check)", "https://cspec.genome.network",
      "not_downloaded", "gene/disease-specific ACMG criteria specifications",
      leakage_risk="none",
      notes="vpdl.slm.text.acmg.GENE_SPECS holds two specifications, both verified=False."),
    S("insight_lovd", "InSiGHT LOVD (MMR variants)", "clinical", ("REFERENCE_ONLY",),
      "terms not confirmed", "https://www.insight-database.org", "not_incorporated",
      "MMR variant database", gene_scope="MLH1, MSH2, MSH6, PMS2",
      notes="InSiGHT expert-panel classifications already reach ClinVar as expert-panel SCVs."),
    # -- population -----------------------------------------------------------
    S("gnomad", "gnomAD v4", "population", ("TRAINING", "KNOWLEDGE_BASE"), "CC0",
      "GraphQL API (vpdl.sources.gnomad) / sites VCFs", "local",
      "allele frequencies", gene_scope="per-gene files for the DL panel and ProteinGym genes only",
      evidence_types=("population",), leakage_risk="low",
      leakage_notes="Frequency is evidence (PM2/BS1/BA1), not a label. PMS2CL region unreliable.",
      local_paths=("data/raw/gnomad/", "data/mmr/raw/gnomad/"),
      notes="DATA GAP: no genome-wide gnomAD on this PC; the SLM's broad variants have no "
            "frequency until sites VCFs are joined on the DGX."),
    # -- functional (independent by default) ------------------------------------
    S("mavedb_msh2_jia2021", "MaveDB urn:mavedb:00000050-a-1 — MSH2 LOF scores (HAP1)",
      "functional", ("INDEPENDENT_VALIDATION",), "CC0 (MaveDB record)", "MaveDB API", "local",
      "MSH2 missense loss-of-function scores", gene_scope="MSH2", variant_scope="missense",
      evidence_types=("functional",), label_types=("continuous",), leakage_risk="critical",
      leakage_notes="Held out. ClinVar narratives cite this study; in strict functional-holdout "
                    "mode sentences citing it are removed from training text.",
      local_paths=("data/raw/mavedb/urn_mavedb_00000050-a-1_scores.csv",),
      publications=("PMID:33357406", "DOI:10.1016/j.ajhg.2020.12.003",
                    "DOI:10.1101/2020.06.03.133017"),
      notes="17,746 score rows (local file)."),
    S("mavedb_mlh1_abundance", "MaveDB urn:mavedb:00001218-a-1 — cellular abundance of MLH1",
      "functional", ("INDEPENDENT_VALIDATION",), "CC0 (MaveDB record)", "MaveDB API", "local",
      "MLH1 variant abundance", gene_scope="MLH1", variant_scope="missense",
      evidence_types=("functional",), label_types=("continuous",), leakage_risk="critical",
      local_paths=("data/raw/mavedb/urn_mavedb_00001218-a-1_scores.csv",),
      publications=("DOI:10.1101/2024.07.28.605491",), notes="5,056 score rows (local file)."),
    S("proteingym_dms", "ProteinGym v1.3 DMS substitutions", "functional",
      ("INDEPENDENT_VALIDATION",), "MIT (as recorded in vpdl/sources/proteingym_dms.py)",
      "zip download", "local",
      "217 DMS assays", evidence_types=("functional",), label_types=("continuous", "binary"),
      leakage_risk="high", local_paths=("data/raw/proteingym/DMS_ProteinGym_substitutions.zip",),
      leakage_notes="Contains the MSH2 Jia 2021 assay again: the MaveDB holdout applies to it too."),
    S("cimra", "CIMRA (MMR in-vitro activity, OddsPath)", "functional",
      ("INDEPENDENT_VALIDATION",), "user-supplied; terms per publication", "CSV (vpdl.dl.functional)",
      "not_downloaded", "MMR functional classes", gene_scope="MLH1, MSH2, MSH6, PMS2",
      leakage_risk="critical", notes="No CIMRA file on this PC. DATA GAP."),
    S("atlas_of_variant_effects", "Atlas of Variant Effects", "functional",
      ("INDEPENDENT_VALIDATION",), "per dataset", "https://www.varianteffect.org", "not_incorporated",
      "MAVE datasets", notes="No MMR dataset bundled (docs/dl/DATASET.md)."),
    # -- literature -------------------------------------------------------------
    S("pubmed", "PubMed baseline abstracts", "literature", ("PRETRAINING",),
      "NLM does not own abstract copyright; research use as in BioGPT/BioMedLM (docs/slm/PLAN.md)",
      "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ (1,334 files)", "on_dgx",
      "biomedical abstracts", evidence_types=("literature",), leakage_risk="medium",
      leakage_notes="Abstracts describing evaluation variants are removed in strict mode "
                    "(publications cited by reserved variants).",
      update_frequency="annual baseline + daily updates",
      notes="Download started 2026-09-22 on the DGX; retractions and expressions of concern dropped."),
    S("pmc_oa_comm", "PMC article datasets (CC0 / CC BY / CC BY-SA)", "literature", ("PRETRAINING",),
      "per article; kept: CC0, CC BY, CC BY-SA (no NC, no ND, no author-manuscript TDM); "
      "acknowledge NLM as the source",
      "s3://pmc-oa-opendata via `vpdl-slm pmc-download` (FTP bulk files withdrawn Aug 2026)",
      "not_downloaded", "full-text articles, medical-genetics topic query", leakage_risk="medium",
      leakage_notes="Articles can describe evaluation variants: each keeps its PMID and the "
                    "pretraining corpus drops those cited for evaluation variants (strict mode).",
      notes="421,690 articles matched the default topic + licence query on 2026-09-24 "
            "(vpdl.slm.pmc.DEFAULT_TOPIC)."),
    S("genereviews", "GeneReviews", "literature", ("PRETRAINING", "KNOWLEDGE_BASE"),
      "non-commercial research only; credit + link; no modifications when displayed",
      "NCBI Bookshelf BITS XML archive (vpdl.kb.genereviews)", "on_dgx",
      "expert-written chapters on inherited conditions", leakage_risk="low",
      leakage_notes="Chapters name some variants (founder alleles); counted, not removed.",
      notes="44,476 passages from 892 chapters (docs/RUNLOG.md 2026-09-22)."),
    S("medlineplus_genetics", "MedlinePlus Genetics", "literature", ("PRETRAINING", "KNOWLEDGE_BASE"),
      "public domain", "https://medlineplus.gov/download/ghr-summaries.xml", "not_downloaded",
      "plain-language gene and condition summaries", notes="reader: vpdl.slm.textsources.iter_medlineplus"),
    S("uniprot_text", "UniProtKB function and disease comments (human, reviewed)", "annotation",
      ("PRETRAINING", "KNOWLEDGE_BASE"), "CC BY 4.0",
      "https://rest.uniprot.org/uniprotkb/stream (TSV: accession, gene_primary, protein_name, "
      "cc_function, cc_disease)", "not_downloaded", "protein function and disease involvement text",
      leakage_notes="Gene/protein-level text; variant features (ft_variant) are not downloaded.",
      notes="reader: vpdl.slm.textsources.iter_uniprot_text"),
    # -- annotation / phenotype ------------------------------------------------
    S("uniprot", "UniProtKB", "annotation", ("KNOWLEDGE_BASE", "REFERENCE_ONLY"), "CC BY 4.0",
      "REST API (vpdl.sources.uniprot)", "local", "protein sequence, domains, function",
      local_paths=("data/raw/uniprot/", "data/raw/uniprot_domains/")),
    S("refseq_mane", "RefSeq / MANE Select", "annotation", ("REFERENCE_ONLY",),
      "public domain (NCBI)", "NCBI FTP", "not_downloaded", "transcripts",
      notes="The DL branch pins MANE transcripts for the four MMR genes only."),
    S("ensembl_vep", "Ensembl VEP consequences", "annotation", ("REFERENCE_ONLY",),
      "Apache-2.0 (VEP)", "local install on the DGX", "not_downloaded",
      "transcript-aware consequence", notes="PROPOSED to replace the notation-level consequence."),
    S("alphamissense", "AlphaMissense", "annotation", ("TRAINING",),
      "CC BY 4.0 (relicensed 2024-03-13; file header still says CC BY-NC-SA 4.0 — "
      "vpdl/sources/alphamissense.py)", "zip",
      "local", "missense pathogenicity predictions", variant_scope="missense",
      evidence_types=("computational",), leakage_risk="medium",
      leakage_notes="Thresholds were calibrated on ClinVar: circular when evaluated on ClinVar.",
      local_paths=("data/raw/alphamissense/AlphaMissense_aa_substitutions.tsv.gz",)),
    S("mondo", "Mondo disease ontology", "phenotype", ("PRETRAINING", "REFERENCE_ONLY"), "CC BY 4.0",
      "http://purl.obolibrary.org/obo/mondo.obo", "not_downloaded",
      "disease identifiers, hierarchy and textual definitions (reader: iter_mondo)",
      notes="Would let disease holdout group sub-types under one parent. Until then MedGen ids "
            "are grouped as given."),
    S("hpo", "Human Phenotype Ontology", "phenotype", ("REFERENCE_ONLY",),
      "free, must be cited, content not altered (confirm before redistribution)", "OBO download",
      "not_downloaded", "phenotype terms"),
    S("orphanet", "Orphanet / Orphadata", "phenotype", ("PRETRAINING", "KNOWLEDGE_BASE"), "CC BY 4.0",
      "https://www.orphadata.com/data/xml/en_product1.xml", "not_downloaded",
      "rare-disease definitions (11,645 disorders, release 2026-06-23; reader: iter_orphanet)"),
    S("gencc", "GenCC gene-disease validity", "clinical", ("KNOWLEDGE_BASE",), "CC0",
      "TSV download", "not_downloaded", "gene-disease validity"),
    # -- standards ------------------------------------------------------------
    S("acmg_amp_2015", "ACMG/AMP 2015 guideline (Richards et al.)", "standard", ("REFERENCE_ONLY",),
      "journal copyright — not redistributed", "publication", "not_incorporated",
      "variant classification framework",
      notes="Encoded as the criteria table in vpdl.slm.text.acmg (codes, direction, default "
            "strength, paraphrased summaries); the guideline text itself is not used."),
    S("omim", "OMIM", "clinical", ("REFERENCE_ONLY",),
      "no incorporation into software without a licence", "—", "not_incorporated",
      "allelic variant descriptions",
      notes="OMIM text also arrives inside ClinVar as OMIM 'literature only' submissions; those "
            "documents are flagged restricted and kept out of weights."),
    # -- project-internal ---------------------------------------------------------
    S("kb_eval_questions", "Knowledge-base evaluation questions", "clinical", ("TEST",), "ours",
      "docs/kb/eval_questions.jsonl", "local", "37 KB questions", leakage_risk="critical",
      leakage_notes="Never indexed, never trained on (docs/kb/SOURCES.md section 4)."),
    S("mmr_v1_tables", "v1 MMR labelled / VUS tables", "clinical", ("REFERENCE_ONLY",),
      "derived from ClinVar", "data/mmr/processed/", "local", "v1 archive tables",
      gene_scope="MLH1, MSH2, MSH6, PMS2",
      notes="Preserved as the v1 baseline's inputs; derived from an earlier ClinVar release."),
    S("dl_branch_outputs", "DL branch DLRepresentation export", "model_output", ("TRAINING",),
      "ours", "runs/dl/**/dl_outputs_*.jsonl (vpdl-dl export)", "not_downloaded",
      "out-of-gene DL embeddings and scores", gene_scope="DL panel genes", variant_scope="missense",
      leakage_risk="high",
      leakage_notes="Each score is out-of-fold for its own gene only; the DL model saw other genes' "
                    "labels, so SLM test variants must be checked against the DL training set."),
    S("teacher_outputs", "Local teacher model outputs", "model_output", ("TRAINING",),
      "qwen3:32b Apache-2.0; medgemma terms on training-from-outputs NOT checked",
      "vpdl.slm.teacher via local Ollama", "not_downloaded", "noisy evidence labels / explanations",
      leakage_risk="high", leakage_notes="Never generated for VALIDATION/TEST/INDEPENDENT rows."),
)


def source(source_id: str) -> Source:
    for item in SOURCES:
        if item.source_id == source_id:
            return item
    raise KeyError(f"no source {source_id!r} in the catalogue")


HOLDOUT_PUBLICATIONS: tuple[str, ...] = tuple(
    identifier for item in SOURCES if "INDEPENDENT_VALIDATION" in item.intended_roles
    for identifier in item.publications)


def validate_catalog(sources: Iterable[Source] = SOURCES) -> list[str]:
    problems = []
    seen = set()
    for item in sources:
        if item.source_id in seen:
            problems.append(f"duplicate source_id {item.source_id}")
        seen.add(item.source_id)
        bad_roles = [r for r in item.intended_roles if r not in DATA_ROLES]
        if bad_roles:
            problems.append(f"{item.source_id}: unknown roles {bad_roles}")
        if item.status not in STATUSES:
            problems.append(f"{item.source_id}: unknown status {item.status!r}")
        if item.leakage_risk not in LEAKAGE_RISKS:
            problems.append(f"{item.source_id}: unknown leakage_risk {item.leakage_risk!r}")
        if not item.license:
            problems.append(f"{item.source_id}: licence/terms must be recorded")
        if "INDEPENDENT_VALIDATION" in item.intended_roles and {"TRAINING", "PRETRAINING"} & set(
                item.intended_roles):
            problems.append(f"{item.source_id}: independent validation cannot also train")
    return problems


def with_hashes(root: Path | str = ".") -> list[dict[str, Any]]:
    """Catalogue rows with sha256 of every local file (recomputed, not trusted)."""
    from vpdl.slm.build.inventory import sha256_file
    rows = []
    for item in SOURCES:
        row = item.as_dict()
        hashes = {}
        for relative in item.local_paths:
            path = Path(root) / relative
            if path.is_file():
                hashes[relative] = sha256_file(path)
        row["computed_hashes"] = hashes
        rows.append(row)
    return rows


def catalog_json(root: Path | str | None = None) -> str:
    rows = with_hashes(root) if root is not None else [s.as_dict() for s in SOURCES]
    return json.dumps(rows, indent=2)


def catalog_markdown(sources: Iterable[Source] = SOURCES) -> str:
    lines = ["| source_id | type | roles | status | licence / terms | leakage risk |",
             "|---|---|---|---|---|---|"]
    for item in sources:
        lines.append(f"| `{item.source_id}` | {item.source_type} | {', '.join(item.intended_roles)} "
                     f"| {item.status} | {item.license} | {item.leakage_risk} |")
    return "\n".join(lines)
