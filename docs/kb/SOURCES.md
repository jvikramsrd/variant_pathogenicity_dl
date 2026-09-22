# Knowledge sources for the clinical genetics knowledge base

Licences and access terms were checked on 2026-09-21 against each provider's own
terms page (linked). Terms change: re-check before publishing or sharing
anything built from them.

## How sources are used — three roles, never mixed

1. **Read and cite.** Text the local model summarises, shown to the reader word
   for word with credit (GeneReviews today).
2. **Look up.** Facts printed exactly as the database records them, never passed
   through the model (ClinVar today).
3. **Test.** Questions with known answers, used to measure the system. Never
   indexed, or the test would contain its own answers.

The small model is **not trained on these facts**. A model trained on
textbooks learns to sound like them and misremembers details with confidence
(DESIGN.md, rule 1). If we fine-tune later, it is to teach behaviour — cite
every sentence, say "not found", never classify a variant — using examples we
write, not the sources' content.

## 1. Read and cite

| Source | What it adds | Licence / terms | Format | Status |
|---|---|---|---|---|
| [GeneReviews](https://www.ncbi.nlm.nih.gov/books/NBK1116/) | ~890 expert-written chapters on inherited conditions: diagnosis, management, surveillance, counselling | Non-commercial research only; credit + link with every excerpt; **no modifications** | BITS XML archive | **in use** |
| [MedlinePlus Genetics](https://medlineplus.gov/about/using/usingcontent/) (NLM) | Plain-language summaries of genetic conditions, genes and chromosomes; the "Help Me Understand Genetics" handbook | **Public domain**; credit "Courtesy of MedlinePlus from the National Library of Medicine" | One XML file: `medlineplus.gov/download/ghr-summaries.xml` | next |
| [ClinGen](https://clinicalgenome.org/docs/terms-of-use/) curations | Gene–disease validity, dosage sensitivity, clinical actionability reports | **CC0**; attribution and access date requested | CSV/TSV downloads, APIs | next |
| [Orphanet / Orphadata](https://www.orphadata.com/faq/) | Rare-disease definitions, associated genes, epidemiology, natural history | **CC BY 4.0** (commercial use allowed with credit) | XML/JSON, released July and December | next |
| [CPIC](https://cpicpgx.org/) guidelines | Gene–drug prescribing guidelines (pharmacogenomics) | Public domain (CC0) | Web, API | next |
| [ClinPGx](https://blog.clinpgx.org/changes-to-pharmgkb-data-licensing/) (formerly PharmGKB) | Curated variant–drug clinical annotations | **CC BY-SA 4.0** — share-alike: derived data must carry the same licence | TSV, no account needed | later |
| [PMC Open Access subset](https://pmc.ncbi.nlm.nih.gov/tools/ftp/) | Full-text research papers, filtered to genetics | Per article. `oa_comm` = CC0/BY/BY-SA/BY-ND; `oa_noncomm` = NC licences; `oa_other` = unclear — skip | Bulk XML | later (large; papers disagree, so needs date and evidence handling) |

## 2. Look up (facts shown as recorded)

| Source | What | Licence / terms | Status |
|---|---|---|---|
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | Variant classifications with review status, conditions, submitters | Public domain (NCBI) | **in use** |
| [gnomAD](https://gnomad.broadinstitute.org/policies) | Population allele frequencies | **CC0**; attribution requested | in use (Lynch genes, via `vpdl build`) |
| [AlphaMissense](https://github.com/google-deepmind/alphamissense) | Missense pathogenicity *predictions* (a score, not a classification) | CC BY 4.0 (relicensed 2024-03-13; see `vpdl/sources/alphamissense.py`) | in use (Lynch genes) |
| [ClinGen expert-panel variant curations](https://clinicalgenome.org/docs/terms-of-use/) | Expert-panel variant interpretations with the ACMG criteria they met | **CC0** | next |
| [GenCC](https://thegencc.org/terms) | Gene–disease validity, harmonised across curation groups | **CC0**; download excludes OMIM's rows | next |
| [CIViC](https://civicdb.org/) | Clinical evidence for cancer (somatic) variants | **CC0** | if cancer questions are in scope |
| [Mondo](https://mondo.monarchinitiative.org/pages/download/) | Disease names and synonyms, to match a question to the right condition | **CC BY 4.0** | next |
| [Human Phenotype Ontology](https://hpo.jax.org/) | Phenotype terms and disease–phenotype annotations | Free; must be cited; **content must not be altered** (terms on hpo.jax.org — confirm before redistributing) | next |

## 3. Restricted — do not put in the shared index

| Source | Why not | What to do instead |
|---|---|---|
| [OMIM](https://www.omim.org/help/agreement) | Research use only; no copying for redistribution; incorporating it into software needs a Johns Hopkins licence | Link out to OMIM entries; don't index its text |
| [COSMIC](https://www.cosmickb.org/licensing/) | Free for academic non-profit use with registration, but no redistribution | Use CIViC (CC0) for cancer variants |
| [OncoKB](https://faq.oncokb.org/licensing) | Free on the website for academic research; API needs registration; **clinical use needs a paid licence** | Not usable here: this system supports clinical decisions |
| Owned textbooks | Copyrighted | Phase 2 only, flagged private (used to answer, never displayed) |
| InSiGHT LOVD (MMR variants) | Terms not confirmed | Its expert-panel classifications already reach us through ClinVar |
| NCCN guidelines | Not openly licensed | Cite by name when a passage refers to them |
| Patient records | Personal health data | Phase 3 only: ethics approval, de-identification, separate store (DESIGN.md) |

## 4. Test (never indexed)

| Set | What | Licence | Use |
|---|---|---|---|
| `docs/kb/eval_questions.jsonl` | 37 questions: 24 answerable, 8 unanswerable, 5 variant lookups | ours | **in use** |
| [GeneTuring](https://pmc.ncbi.nlm.nih.gov/articles/PMC10054955/) | 16 genomics tasks, 1,600 questions (gene names, locations, functions, sequences) | check before use | Measures the base model's genomic knowledge; the paper found even the best LLMs failed some tasks outright |
| [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) | 193,155 AIIMS / NEET-PG exam questions | Apache-2.0 | General medicine; its genetics questions can screen candidate base models |

## Base models (all run locally through Ollama)

| Model | Terms | Note |
|---|---|---|
| llama3.1:8b, qwen3:32b | Their own open-weight licences | Current candidates; `kb-eval` decides |
| [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma) 4B / 27B | Health AI Developer Foundations terms (open weights; research and commercial use under those terms) | Google states it is not clinical-grade out of the box. Worth adding as a candidate |

## Recommended order

1. Finish Phase 1 on the DGX (GeneReviews + ClinVar) and read the evaluation.
2. Add **MedlinePlus Genetics, ClinGen, Orphanet, GenCC**: all public domain or
   CC, and they cover what GeneReviews does not (genes without a chapter, rare
   diseases, how strong a gene–disease link is). One reader each, same passage
   format, same citation rules.
3. Add ClinGen expert-panel variant curations to the lookup, beside ClinVar.
4. **Before indexing each new source, write its evaluation questions**, both
   answerable and unanswerable, so it is measured, not assumed.
5. PMC Open Access last: largest, and papers contradict each other, so answers
   will need publication dates and evidence levels shown.
