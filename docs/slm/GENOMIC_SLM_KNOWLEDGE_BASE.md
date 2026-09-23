# The genomic knowledge system: sources, roles and the variant record

The machine-readable catalogue is `vpdl/slm/catalog.py`
(`vpdl-slm catalog --markdown`, `--json`, `--hash`). This file explains it.

## 1. A knowledge base is not a training set

Every source is given a **role**, and the role decides what may happen to it:

| Role | Meaning | Consequence |
|---|---|---|
| `KNOWLEDGE_BASE` | consulted at answer time | may be retrieved and cited; never a label |
| `PRETRAINING` | shapes the language model | text only; evaluation material removed first |
| `TRAINING` / `VALIDATION` / `TEST` | supervised learning | split-controlled, leakage-audited |
| `INDEPENDENT_VALIDATION` | touched once, at the end | never trains, never tunes, never retrieved during training |
| `REFERENCE_ONLY` | for people and code | e.g. ACMG criteria definitions, ontologies |
| `EXCLUDED` | kept out | third-party terms, or contaminated |

A source may inform without supervising. gnomAD frequency is evidence for
PM2/BS1/BA1 — it is a feature, not a label. ClinVar's aggregate classification
is a label — so it can never be an input (`vpdl.slm.schema.NEVER_INPUT`).

## 2. The catalogue

`vpdl-slm catalog --markdown` prints the current table; every row records
source type, roles, licence/terms, access method, status, scope, evidence and
label types, provenance, leakage risk and (for local files) a hash. Status is
one of `local`, `on_dgx`, `not_downloaded`, `not_incorporated`, `generated` —
so the document never implies that something has been obtained when it has not.

Highlights, with the terms as checked (2026-09-21 against each provider's own
page, recorded in `docs/kb/SOURCES.md`; Hugging Face licence tags 2026-09-22;
ClinVar column names against NCBI's README 2026-09-22):

**Clinical.** ClinVar `variant_summary` (public domain, local, measured) ·
ClinVar `submission_summary` — the narratives, **not downloaded** · ClinVar
`var_citations` — variant→publication, **not downloaded** · ClinGen Evidence
Repository (CC0, **not downloaded**; its column names are unverified and its
reader fails loudly rather than guessing) · InSiGHT LOVD — terms not confirmed,
**not incorporated** (its expert-panel calls reach us through ClinVar anyway).

**Population.** gnomAD v4 (CC0) — present only for panel genes; genome-wide is
a DATA GAP.

**Functional — independent by default.** MaveDB MSH2 LOF (CC0, PMID 33357406)
and MLH1 abundance (CC0, DOI 10.1101/2024.07.28.605491), ProteinGym v1.3 (MIT),
CIMRA (**no file here**). These never train. Their **publications** are
registered so that narratives citing them can be removed from training text
(`HOLDOUT_PUBLICATIONS`).

**Literature.** PubMed baseline (downloading on the DGX; retractions and
expressions of concern dropped by `vpdl.slm.pubmed`) · PMC OA `oa_comm` (later)
· GeneReviews (non-commercial research; no modification when displayed) ·
MedlinePlus Genetics (public domain, next).

**Annotation / phenotype.** UniProt (CC BY 4.0, local) · RefSeq/MANE ·
Ensembl VEP (proposed, would replace the notation-level consequence) ·
AlphaMissense (CC BY 4.0 since 2024-03-13; **calibrated on ClinVar**, so
circular when evaluating on ClinVar) · Mondo, HPO, Orphanet, GenCC.

**Standards.** ACMG/AMP 2015 is encoded as a criteria *table* (code, direction,
default strength, a paraphrase) in `vpdl/slm/text/acmg.py`; the guideline text
itself is copyrighted and is not reproduced.

**Excluded.** OMIM — its terms forbid incorporation into software without a
licence. OMIM text also arrives *inside* ClinVar as OMIM "literature only"
submissions; those documents are flagged `restricted` and kept out of weights
and examples by default.

## 3. The variant-centric record

`vpdl.slm.build.records.variant_view(tables, variant_id)` assembles everything
known about one variant without collapsing it:

```
variant_id, genomic_key (GRCh38:chr:pos:ref:alt), gene, transcript, hgvs_c,
hgvs_p, variant_type, molecular_consequence, protein_variant_id (DL join),
disease [(id, name), ...], population_frequency {}, 
clinical_interpretations [ per submission: submitter, classification, date,
                           review status, tier, restricted ],
aggregate { classification, review_status, stars, last_evaluated },
evidence [ per sentence: types, polarity, ACMG codes, text, role, provenance ],
literature_evidence [PubMed:…], functional_evidence [], computational_evidence [],
source_provenance [{source_id, version}], quality_flags [...]
```

Two rules are visible in that shape. **Nothing is imputed**: a source that has
not been joined leaves an empty container and a quality flag
(`population_frequency_not_joined`), never a default value. **Nothing is
collapsed**: each submission keeps its own classification and date, so
disagreement stays visible — which is the point for VUS and for conflicting
evidence.

## 4. Entities and relationships

`build_edges(tables)` emits a provenance-carrying edge table over the entity
graph the prompt describes:

```
variant  -in_gene->            gene            variant -asserted_for->  disease
variant  -encodes_protein_change-> protein     variant -cited_by->      publication
document -interprets->         variant         document -cites->        publication
evidence -states_acmg_code->   ACMG criterion
```

Every edge carries `source_id` and, where it came from a document, its
`document_id`. This is a derived table, not a graph database: it exists so
that "where did this come from" is one query, and so retrieval and explanation
can cite an actual record.

## 5. Provenance and versions

`build-records` writes `manifest.json`: the sha256, size and path of every
input file, per-table row counts, every reader's skip counts (malformed rows,
unreadable files, documents whose variant was filtered out), the schema and
preprocessing versions, the evidence-rule version, and the git commit. Two
builds are comparable only when those agree — the same rule the rest of this
project uses before pooling any two runs (`vpdl.provenance.assert_comparable`).
