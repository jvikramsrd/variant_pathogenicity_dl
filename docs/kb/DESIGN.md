# Clinical genetics knowledge base — design

Status: **Phase 1 built, not yet run on the DGX** (2026-09-21). Code: `vpdl/kb/`; tests: `tests/kb/`; how to run: [RUNBOOK.md](RUNBOOK.md).

A local question-answering system over clinical genetics sources, for three uses:
learning and research questions, variant lookups, and supporting (never making)
clinical decisions.

---

## The rules that everything else follows from

Accuracy on sensitive clinical questions is the requirement. These rules are what make
that achievable; none of them is optional.

1. **Knowledge lives in documents, not in model weights.** No fine-tuning a model to
   "learn" the books. A model trained on textbooks learns to sound like them and
   misremembers details with full confidence — the failure this system cannot afford.
   The model's only job is to read retrieved passages and summarise them.
2. **Every claim is cited, or it is not said.** Answers cite the passages they came
   from. An answer with no valid citation is withheld, not shown.
3. **"Not found in the sources" is a correct answer.** When retrieval does not cover the
   question, the system says so instead of guessing.
4. **The model never classifies a variant.** Variant questions are answered by looking
   the variant up in the assembled ClinVar table (`data/built/mmr.csv`) and reporting
   what the sources record — ClinVar classification with its review status, AlphaMissense,
   gnomAD frequency. The model reports evidence; it never makes the call.
5. **Everything runs on the DGX.** Models, search and data all stay on the machine.
   Nothing is sent to an external service.
6. **Research use only — decision *support* for qualified experts.** Every answer
   carries that label. It is not a medical device and must not be used as one.

---

## How a question flows

```
question
   |
   +-- mentions a variant? --> look it up in the ClinVar table --> cited facts
   |                                                               |
   +-- search the documents:                                       |
         exact-word search (identifiers like MLH1, c.199G>A)       |
       + meaning search (paraphrased questions)                    |
       -> merge and rank -> top passages ------------------------> |
                                                                   v
                                 local model, told to answer ONLY from these
                                                                   |
                                 citation check: every claim cited? every
                                 citation real? if not -> "not found"
                                                                   |
                                                                   v
                                 answer + verbatim source excerpts with attribution
```

**Why two kinds of search.** Genetics text is full of exact identifiers — gene symbols,
HGVS variant names, rsIDs. Meaning-based search alone blurs them together (MLH1 and
MSH2 look alike to it); exact-word search catches them. Using both, merged, gets both
kinds of question right.

**Why excerpts are shown word for word.** GeneReviews forbids modification (below), and
a clinician needs to see what the source actually says, not a paraphrase of it. The
model's summary is clearly separated from the quoted source text.

---

## Sources and their terms — verified from the artefacts, not from memory

| Source | What | Terms | Phase |
|---|---|---|---|
| **GeneReviews** | ~900 expert-written clinical genetics chapters, incl. Lynch syndrome (NBK1211, 41 sections) | Non-commercial research only. Each copy/excerpt must credit genereviews.org and "© 1993-2026 University of Washington", with a link. **No modifications.** Excerpts in lab reports and clinic notes explicitly permitted. | 1 |
| **Our ClinVar table** | 74,328 MMR variants with ClinVar labels, AlphaMissense, gnomAD | Built from public sources; see `vpdl` source licences | 1 |
| Open-access papers (PubMed Central OA) | Research literature | Per-article licence (CC BY, CC BY-NC, ...) — recorded per document | 1 (later) |
| **Your textbooks** | Owned PDFs | Copyrighted — private internal use only, never shown outside | 2 |
| **Patient records** | Clinical data | Personal health information — see Phase 3 | 3 |

GeneReviews terms were read from the `<permissions>` block of `hnpcc.nxml` in the
NCBI Bookshelf open-access archive on 2026-09-21. The archive
(`ftp.ncbi.nlm.nih.gov/pub/litarch/ca/84/gene_NBK1116.tar.gz`, ~608 MB, updated
2026-09-20) ships each chapter as structured XML, so chunks can follow real section
boundaries instead of being cut from PDF text.

Every document is stored with its source, licence, link, and a **private** flag.
Private material is used to answer but never displayed or exported verbatim.

---

## Phases

**Phase 1 — public knowledge + variant lookup.** GeneReviews and the ClinVar table.
Local model, citations, refusal, verbatim excerpts with attribution. Plus an evaluation
set (below). Buildable now.

**Phase 2 — your textbooks.** Same system, documents flagged private.

**Phase 3 — patient records. Not built until all of these exist:**
- ethics committee (IEC/IRB) approval for this use;
- de-identification of the records *before* they reach this machine's index;
- a separate store, with access control and a log of every query that touched it;
- a decision on where outputs that mention patient data may go.

Patient records never share an index with public knowledge. This is the legal and
ethical baseline for health data (India's DPDP Act 2023 and ICMR research guidelines
apply here), not extra caution.

---

## Models — local, via Ollama

Ollama runs a model on the DGX and exposes the same API shape as OpenAI's, so the code
works unchanged if we later move to vLLM or NVIDIA NIM. Confirmed available in
Ollama's library on 2026-09-21:

- **Answering:** `llama3.1`, `qwen3`, `gpt-oss`, and `medgemma` (Google's medical-tuned
  model). The DGX's memory fits even the large variants.
- **Search embeddings:** `nomic-embed-text`, `bge-m3`.

**Which model is chosen by measurement, not reputation:** each candidate is scored on
the evaluation set, and the one that answers most accurately with valid citations wins.

---

## Evaluation — how we know it is accurate

A set of questions with known answers and known source passages, written before the
system is tuned:

- **Retrieval:** does the right passage appear in the top results?
- **Faithfulness:** is every sentence in the answer supported by its cited passage?
- **Citation validity:** does every citation point at a passage that was actually retrieved?
- **Refusal:** for questions the sources do not cover, does it say so instead of answering?
- **Variant facts:** do reported ClinVar classifications match the table exactly?

The refusal questions matter as much as the answerable ones. A system that always
answers will score well on answerable questions and be dangerous on the rest.

---

## What it will not do

- Classify a variant itself.
- Answer from the model's own memory when the sources are silent.
- Send any data off the DGX.
- Touch patient records before Phase 3's conditions are met.
- Show copyrighted textbook text outside the machine.
