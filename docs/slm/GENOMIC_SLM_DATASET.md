# The dataset: tables, corpus measurements, roles, splits and data gaps

Code: `vpdl/slm/build/`. Commands: `vpdl-slm build-records | stats | dedup |
splits | roles | examples`. Schema: `vpdl/slm/schema.py`.

## 1. Four tables and one identity

| Table | One row per | Key columns |
|---|---|---|
| `variants` | ClinVar VariationID | `variant_id` (`clinvar:<id>`), `genomic_key` (`GRCh38:chr:pos:ref:alt`), gene(s), transcript, `hgvs_c`, `hgvs_p`, `variant_type`, `consequence`, aggregate classification + review stars + date, conditions, `protein_variant_id` (the DL branch's key), `quality_flags`, source version |
| `documents` | one text with provenance (a ClinVar SCV narrative, a ClinGen expert-panel summary) | `document_id`, submitter, collection method, review status, its own classification (+ `label5`, soft target), date, text, conditions, `tier`, `restricted` |
| `evidence_units` | one sentence of a document | `evidence_id`, character span, `sentence_role`, `evidence_types`, `evidence_polarity`, `acmg_codes` (met / not met), PMIDs, `provenance`, `confidence` |
| `citations` | variant → publication | `variation_id`, `citation_source`, `citation_id` |
| `acmg_labels` | expert-panel codes (ERepo) | `codes_met`, `codes_not_met`, `provenance` |

`variant_id` is ClinVar's VariationID because the SLM reads text about any
variant type. `protein_variant_id` (`UniProt:pos:wt>mut`) is filled from the DL
branch's own canonical table when a row is one of its protein substitutions, so
the two branches join on an identity the DL branch has already validated
against UniProt.

## 2. What the corpus contains — measure before claiming

`vpdl-slm stats --records data/slm_genomic --markdown docs/slm/corpus_stats.md`
produces every count the plan asks for: documents, documents with text, unique
variants / genes / diseases / submitters, expert-panel and VCEP records,
records with a direct conclusion, with citations, with each evidence type, with
ACMG codes, per-class counts, character and word length (median, p25, p75,
max), token counts for a chosen tokenizer, the MMR share, and breakdowns by
gene, disease, variant type, source, laboratory, year, review status,
classification and tier.

**Status: TBD — RUN ON DGX SPARK.** The narratives are not on this machine
(LLM_CODEBASE_AUDIT.md finding 1), so the clinical corpus has not been
measured and is **not** called sufficient. What *is* measured is the variant
layer of the same release (audit §2.3): 4,558,681 variants, 35,013 genes,
22,111 conditions, 55% missense, 22,390 expert-panel variants, MMR 0.70% — and
55% of variants with no specific condition, which is why the disease-holdout
split can only be built on the other 45%.

The upper bound on text is ClinVar's submission count; how much of it is free
text, and how long, is exactly what `stats` will answer first.

## 3. Quality tiers (context, not truth)

1 expert-curated (ClinGen VCEP, practice guideline) · 2 clinical laboratory
with criteria and a narrative · 3 literature-derived / research submissions ·
4 literature and reviews (PubMed, GeneReviews, MedlinePlus) · 5 weakly
structured (no criteria, no or very short text).

Tiers are reported alongside results so a number can be read per tier; they are
never used to overrule a source. OMIM-submitted documents are additionally
flagged `restricted` (terms, not quality) and excluded from weights and
examples.

## 4. Evidence units, and how honest they are

Each sentence becomes a unit with a role (`description`, `evidence`,
`conclusion`, `external_classification`, `code_list`, `boilerplate`, `other`),
its evidence types, a polarity, and any ACMG codes it states. These labels come
from keyword rules (`provenance = rule:evidence-lexicon/v1`).

**Their precision is unknown.** No clinician-annotated sample exists in this
repository, so evidence-extraction quality cannot be reported yet — it is a
DATA GAP, and the evidence heads are an ablation (EXP-012), never an assumed
gain. A gold set of a few hundred sentences, annotated by a clinical geneticist,
is the smallest thing that would turn these into measurable tasks.

## 5. Data roles and reserved evaluation

`vpdl-slm roles --splits …` assigns roles per document and computes what
pretraining must not see:

* documents of **reserved variants** — every variant that is validation, test
  or independent in the MMR, temporal, functional or gene-holdout split — are
  excluded from the pretraining corpus;
* in strict mode, PubMed abstracts whose PMID ClinVar cites **for those
  variants** are excluded too (§19 of the plan: the paper that describes the
  test variant);
* the two independent functional datasets' own publications are always
  excluded.

Both counts appear in `pretrain_corpus/stats.json`, so the cost of the
exclusion is visible rather than assumed.

## 6. Splits

Nine schemes (`vpdl/slm/build/splits.py`), each with a stated guarantee; all
but `random` keep a **variant group** — VariationID ∪ genomic change ∪ gene +
protein change — wholly on one side:

`random` (the optimistic baseline) · `variant` · `text` (variant groups joined
through near-duplicate clusters) · `laboratory` (submitters held out) · `gene`
(MMR reserved out of the gene test set) · `disease` (catch-all and umbrella
conditions never form a group) · `temporal` (`novel` or `reinterpretation`;
undated documents excluded) · `functional` (variants with independent assay
data are test only) · `mmr` (non-MMR train/val/`broad_test`; MMR split into
`mmr_train`/`mmr_val` for an adapter and `test`).

A split that cannot isolate anything fails loudly: if the largest group holds
more than `max_group_share` of the documents, `make_split` raises rather than
producing a split that is one group.

Every split writes a summary with per-split document and variant counts, label
distribution, gene counts and a `split_sha256`.

## 7. Examples, per task, under that task's policy

`vpdl-slm examples --task classify|classify_from_codes|acmg_codes|evidence_type|evidence_polarity`
builds model inputs by applying the task's leakage policy
(`vpdl.slm.text.acmg.TASK_POLICIES`) and records how much it removed per
example: conclusions, other laboratories' verdicts, code-only sentences, masked
codes, sentences citing a holdout publication.

## 8. Data gaps, plainly

| Gap | Effect | What would close it |
|---|---|---|
| ClinVar narratives not downloaded | no clinical text corpus yet | `submission_summary.txt.gz` on the DGX (runbook step 2) |
| `var_citations` not downloaded | literature-leakage control runs on in-text PMIDs only | `var_citations.txt` |
| ClinGen ERepo not downloaded, columns unverified | no gold ACMG codes, no tier-1 evaluation set | one download + confirming the column names |
| Genome-wide gnomAD absent | `log10_af` NaN outside panel genes | gnomAD sites VCFs, or a per-variant join |
| No CIMRA file | one independent functional set missing | the CSV the DL branch's loader expects |
| No annotated evidence sample | evidence extraction unmeasurable | ~300 sentences annotated by a clinical geneticist |
| No archived older ClinVar release | VUS reclassification cannot be scored | one monthly archive file (ClinVar keeps them) |
| Transcript-aware consequence | notation-level classes only | Ensembl VEP on the DGX |
