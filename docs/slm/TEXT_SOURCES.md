# Wider pretraining text — what, where from, and how to get it on the DGX

Everything here is free to download and licensed for this use; each source has a reader in
`vpdl/slm/textsources.py` (formats checked against the providers on 2026-09-24) and goes
into the continued-pretraining corpus through `vpdl-slm pretrain-corpus`, under the same
evaluation exclusions as PubMed.

| Source | What | Licence | Size |
|---|---|---|---|
| PubMed abstracts | the biomedical literature | NLM terms | already on the DGX |
| GeneReviews | expert chapters per condition | non-commercial research | already on the DGX (`data/kb`) |
| ClinVar narratives | lab interpretations, conclusions masked, training variants only | public domain | `submission_summary.txt.gz` |
| **PubMed Central full text** | medical-genetics articles, full body | per article: CC0, CC BY, CC BY-SA only | 421,690 articles (2026-09-24) |
| **MedlinePlus Genetics** | gene and condition summaries | public domain | one XML file |
| **Orphanet** | 11,645 rare-disease definitions | CC BY 4.0 (checked in the file) | one XML file |
| **MONDO** | disease names, synonyms, definitions | CC BY 4.0 | one OBO file |
| **UniProt** | human protein function and disease text | CC BY 4.0 | one TSV |

Not used: OMIM (licence forbids use in software), NC / ND / author-manuscript PMC articles,
patient records (ethics approval first), MIMIC and other credentialed datasets, HPO (terms
not confirmed), GenCC (structured, not prose).

## 1. Download the small sources (minutes)

```bash
mkdir -p ~/variant_pathogenicity_dl/data/raw/text && cd ~/variant_pathogenicity_dl/data/raw/text
```
```bash
wget -c https://medlineplus.gov/download/ghr-summaries.xml
```
```bash
wget -c https://www.orphadata.com/data/xml/en_product1.xml
```
```bash
wget -c http://purl.obolibrary.org/obo/mondo.obo
```
```bash
curl -L -o uniprot_human_text.tsv.gz "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,gene_primary,protein_name,cc_function,cc_disease&query=%28reviewed%3Atrue%29%20AND%20%28organism_id%3A9606%29"
```

## 2. PubMed Central (hours; resumable)

PMC's bulk tarballs were withdrawn in August 2026; articles now come one by one from the
public S3 bucket `pmc-oa-opendata`. `pmc-download` searches PMC (medical-genetics topic,
CC0/CC BY/CC BY-SA, open access), picks each article's published, non-retracted version
and saves its metadata and gzipped XML under `data/raw/pmc/`.

```bash
cd ~/variant_pathogenicity_dl
```
```bash
vpdl-slm pmc-download --count-only
```
Check: about 420,000 articles.

```bash
vpdl-slm pmc-download --limit 200
```
A trial: 200 articles in a minute or two. `data/raw/pmc/manifest.json` shows the licences
fetched and anything excluded.

```bash
vpdl-slm pmc-download
```
The full run: roughly 1.3 million small requests, a few hours at 16 threads; about 5–10 GB
on disk (an estimate). If it stops, run the same command again — finished articles are
skipped. An NCBI API key (`export NCBI_API_KEY=...`) speeds up the search step.

## 3. Check what each source yields before building

```bash
vpdl-slm text-sources --medlineplus data/raw/text/ghr-summaries.xml --orphanet data/raw/text/en_product1.xml --mondo data/raw/text/mondo.obo --uniprot-text data/raw/text/uniprot_human_text.tsv.gz --pmc data/raw/pmc --limit 2000
```
Each source reports documents read, characters, every drop reason, and two examples.

## 4. Build the corpus with everything

After `vpdl-slm roles` has written the exclusions (GENOMIC_SLM_DGX_RUNBOOK.md phase 5):

```bash
vpdl-slm pretrain-corpus --kb data/kb --pubmed data/raw/pubmed --records data/slm_genomic --exclusions data/slm_genomic/pretrain_exclusions.json --medlineplus data/raw/text/ghr-summaries.xml --orphanet data/raw/text/en_product1.xml --mondo data/raw/text/mondo.obo --uniprot-text data/raw/text/uniprot_human_text.tsv.gz --pmc data/raw/pmc --out data/slm_genomic/pretrain_corpus
```
`stats.json` counts documents and characters per source and every exclusion — including
PMC articles dropped because ClinVar cites them for an evaluation variant.

## Leakage notes

PMC articles can describe individual variants and how they were classified. Each PMC
document keeps its PMID, and the corpus drops those cited for evaluation variants when the
exclusions are strict (the `roles` default). A paper that describes a test variant without
ClinVar citing it can still get in; that limit is the same as for PubMed and is recorded in
GENOMIC_SLM_DATA_LEAKAGE_REPORT.md. The other four sources are gene- and disease-level
text with no variant classifications.

Cite NLM as the source of PMC text (`manifest.json` carries the citation line).
