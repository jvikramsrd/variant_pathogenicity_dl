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
ollama pull medgemma:27b
```
```bash
ollama pull qwen3:32b
```
`bge-m3` (~1.2 GB) turns text into vectors for meaning search. `medgemma:27b`
(17 GB, Google's medical model) is the default answering model and `qwen3:32b`
(~20 GB) the second candidate — chosen by `kb-eval` on 2026-09-22 (RUNLOG).
`llama3.1:8b` was tested and disqualified: it gave a cited but clinically
misleading answer.

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

### 4b. Add PubMed abstracts (optional; large)

The PubMed baseline already downloaded for the language model (`data/raw/pubmed/`,
~1,334 files, ~52 GB compressed) can be searched too. It goes into its own on-disk
index, `data/kb/pubmed.sqlite`: exact-word search (SQLite FTS5, BM25), with no
embeddings — for ~36 million abstracts those would take days to compute and ~150 GB to
store. At question time GeneReviews and PubMed are searched separately and merged by
rank, so the passages given to the model come from both.

Check the disk first. The full index is large (my estimate: 60–100 GB; not measured):

```bash
df -h ~/variant_pathogenicity_dl/data && du -sh ~/variant_pathogenicity_dl/data/raw/pubmed
```

Trial on 5 files (minutes):
```bash
vpdl kb-build --pubmed data/raw/pubmed --pubmed-limit-files 5
```
Then the full build (hours; uses every core; the old index is replaced only when the
new one is complete):
```bash
vpdl kb-build --pubmed data/raw/pubmed
```
If the disk is short, index only abstracts that mention genes, variants or inheritance
(`--pubmed-genetics-only`); it is several times smaller.

Retractions, retraction notices and expressions of concern are left out, as in the
language-model corpus. Each PubMed passage is shown word for word with its PMID, a link
and a credit line (the abstract's copyright stays with its publisher).

**Measure before trusting it.** PubMed is primary literature: single studies,
sometimes contradicting each other or the guidelines, where GeneReviews is curated.
Run the evaluation both ways and compare:
```bash
vpdl kb-eval --no-pubmed --out runs/kb/no_pubmed
```
```bash
vpdl kb-eval --out runs/kb/with_pubmed
```
The 37 questions are a development set (see below), so a difference is a signal, not a
result. `vpdl kb-ask --no-pubmed ...` answers from GeneReviews alone.

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
vpdl kb-eval --models medgemma:27b qwen3:32b
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

## Is it on the GPU?

Ollama decides by itself whether a model runs on the GPU, and when it cannot use
the GPU it quietly runs on the CPU: answers still come, 10-50x slower. So every
`kb-*` command checks. After each model's first use it asks Ollama where the
model loaded, logs `bge-m3: 100% on GPU`, and **stops** if any part is on the
CPU. `kb-eval` also records it per model as `gpu_share` (1.0 = all on GPU).

To look yourself while something is running:
```bash
ollama ps
```
The `PROCESSOR` column should read `100% GPU`. Anything with `CPU` in it means
the GPU is not being used (fully or partly).

If a command stops with "running ...% on the CPU":

1. Check the driver sees the GPU (it should list the GB10):
```bash
nvidia-smi
```
2. Restart Ollama so it looks for the GPU again, then re-run the command:
```bash
sudo systemctl restart ollama
```
3. Still on the CPU? See what Ollama found when it started, and paste this:
```bash
journalctl -u ollama --no-pager | grep -iE "gpu|cuda|inference compute" | tail -20
```

`--allow-cpu` runs anyway. Use it only to test, never for an evaluation you
will report: the timing would be wrong, and it hides the problem. If `ollama ps`
says `100% GPU` but `vpdl` says otherwise, paste both outputs.

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
- After the first DGX run (2026-09-22) the prompt and the citation reader were
  changed in response to its answers (docs/RUNLOG.md). The 37 questions are now
  a development set; accuracy must be measured on new questions.
- A cited, verbatim answer can still be wrong for the question — llama3.1:8b did
  exactly that on the first run. Reading the answers is part of every
  evaluation, not optional.
