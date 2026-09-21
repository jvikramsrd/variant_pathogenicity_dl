# Knowledge base — running it on the DGX

Phase 1 of [DESIGN.md](DESIGN.md): GeneReviews + a ClinVar lookup, answered by a
local model that must cite every sentence. Everything runs on the DGX.

## 1. Install the model server (once)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
```bash
curl http://127.0.0.1:11434/api/version
```
The second command should print a version number.

Ollama listens on `127.0.0.1` only, which is what we want. **Do not set
`OLLAMA_HOST=0.0.0.0`** — that would open the model to the whole network. The
`vpdl` client refuses to talk to anything but this machine anyway.

## 2. Download the models (once)

```bash
ollama pull bge-m3
```
```bash
ollama pull llama3.1:8b
```
```bash
ollama pull qwen3:32b
```
`bge-m3` (~1.2 GB) turns text into vectors for meaning search. The other two
(~5 GB and ~20 GB) are the answering candidates; the evaluation decides which
one is used. More candidates can be added the same way.

## 3. Download the sources (once; refresh monthly)

```bash
wget -c -P data/raw https://ftp.ncbi.nlm.nih.gov/pub/litarch/ca/84/gene_NBK1116.tar.gz
```
```bash
wget -P data/raw https://ftp.ncbi.nlm.nih.gov/pub/GeneReviews/GRtitle_shortname_NBKid.txt
```
The first is every GeneReviews chapter as structured XML (~640 MB, updated
2026-09-20). The second maps each chapter to its NCBI page, which every
excerpt must link to. ClinVar's `variant_summary.txt.gz` is already in
`data/raw` from the main pipeline.

## 4. Build

```bash
vpdl kb-build --genereviews data/raw/gene_NBK1116.tar.gz --clinvar data/raw/variant_summary.txt.gz
```
Reads the archive in place, splits ~900 chapters into passages along their
own sections, embeds every passage with `bge-m3`, and builds the ClinVar
lookup table for the Lynch genes (`--genes all` for every gene). Output goes to
`data/kb/` (not committed: GeneReviews text is not ours to republish).

## 5. Ask

```bash
vpdl kb-ask "How often should people with an MLH1 pathogenic variant have a colonoscopy?"
```
```bash
vpdl kb-ask "What does ClinVar say about MLH1 c.199G>A?"
```
Every answer shows: the research-use label; any variant's ClinVar record as
published; the model's answer with a citation after each sentence; and the
cited passages word for word, with GeneReviews' credit line and link. An answer
that fails the citation check is withheld, and "not found in the sources" is
a normal outcome.

## 6. Evaluate — this decides which model is trusted

```bash
vpdl kb-eval --models llama3.1:8b qwen3:32b
```
Runs the 32 fixed questions in [eval_questions.jsonl](eval_questions.jsonl)
through each model (plus 5 variant-lookup checks) and prints, per model:

| field | meaning | want |
|---|---|---|
| `retrieval_hit` | right section among the 6 passages found | high |
| `answered` | answerable questions answered with valid citations | high |
| `correct_facts` | answers containing the key facts (e.g. "20-25", "38%") | high |
| `refused_correctly` | unanswerable questions answered "not found" | **1.0** |
| `withheld_for_citations` | answers blocked by the citation check | low |
| `variant_lookup_correct` | lookups that found (or correctly did not find) the record | 5/5 |

A model that answers everything scores well on the first three and fails the
fourth. `refused_correctly` below 1.0 disqualifies a model for clinical use,
whatever else it scores.

Each model's answers are written to `runs/kb/answers_<model>.jsonl` (passage ids,
not passage text). Reading them is the faithfulness check — whether each
sentence really says what its cited passage says — and it needs a person.

## Honest limits of the evaluation

- All 24 answerable questions come from one chapter (Lynch syndrome). Searching
  ~900 chapters is harder than searching one; the DGX run is the real test.
- The search was changed once after seeing these questions: GeneReviews'
  table abbreviations ("yrs", "assoc") are expanded, which fixed the one miss
  (exact-word search alone: 23/24 -> 24/24 on the single chapter). New
  questions written after that change are needed before claiming retrieval
  quality in anything published.
- Keyword matching checks that the key fact appears, not that the answer is
  right in every word.
