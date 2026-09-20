# How these datasets have been used, and what is left unasked

Compiled 2026-09-20 from literature search. **Every citation below was returned
by a search and its URL recorded; none is reconstructed from memory.** Entries
marked ⚠ still need the primary PDF read before they go in a manuscript —
a search snippet is not a read paper.

The purpose of this document is narrow: establish what is already known about
*which data source contributes what*, so the new paper's claim can be stated
against it rather than beside it.

---

## 1. The datasets, and their established role

### ClinVar
The default supervision source for clinical variant effect prediction, and the
default circularity hazard: most published predictors were trained on it, so
evaluating on it rewards agreement with the training distribution rather than
with biology. ESM1b was benchmarked on ~150,000 ClinVar/HGMD missense variants
([Brandes et al., *Nature Genetics* 2023](https://www.nature.com/articles/s41588-023-01465-0)).

**Recurring design pattern worth copying:** work in this area selects the few
proteins with the most ClinVar labels for *testing* and trains on the rest —
e.g. TP53, GCK, CBS, HMBS, BRCA1 as the held-out set, with remaining clusters
supplying training variants ([Sequence UNET, PMC10169183](https://pmc.ncbi.nlm.nih.gov/articles/PMC10169183/)).
Our leave-one-gene-out protocol is the four-gene analogue.

### ProteinGym
The standard benchmark suite: 43 DMS assays with open licences, reduced to 24
assays and 115,093 missense variants after removing sequence-similarity overlap
with training data ([ProteinGym, NeurIPS 2023 D&B](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf);
[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full)).
Note the overlap-removal step — it is the same leakage problem our cluster-split
addresses, solved at benchmark-construction time.

Used overwhelmingly as an **evaluation** set. Its use as *training* supervision
is newer and is where the interesting results are (next entry).

### DMS as training signal, not just benchmark
The most directly relevant methodological precedent: fine-tuning protein
language models on DMS assays with a Normalised Log-odds Ratio head improves
performance on *both* independent DMS and clinical (ClinVar) benchmarks
([Fine-tuning PLMs with DMS, arXiv 2405.06729](https://arxiv.org/pdf/2405.06729);
[ICLR 2024 MLGenX workshop version](https://openreview.net/pdf?id=d7AdTpYuPC)).
Reported Pearson correlation of 0.78 between clinical-VEP gains and human DMS
gains — i.e. the two supervision sources are related but far from identical,
which is precisely the gap a combined-vs-individual comparison measures. ⚠

Cross-protein transfer learning trained on DMS from only five proteins reaches
state-of-the-art clinical variant interpretation for unseen proteins
([Jagota et al., PubMed 37550700](https://pubmed.ncbi.nlm.nih.gov/37550700/);
[bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.15.516532v1.full)). ⚠
Directly relevant to our leave-one-gene-out framing: it is evidence that
cross-protein generalisation from few proteins is achievable, which our
four-gene panel needs to be true.

### AlphaMissense
Supplies a pathogenicity prior. **Licence: CC BY 4.0 — attribution only**,
verified at [google-deepmind/alphamissense](https://github.com/google-deepmind/alphamissense).
The v1 manuscript stated CC BY-NC-SA 4.0 in two places; that was wrong and has
been corrected. Because it was trained on population and clinical data it
carries allele-frequency signal, which is why `vpdl.features.PROXY_FOR` refuses
a gnomAD ablation that leaves it in place.

### gnomAD
Allele frequency and gene-level constraint, used both as a hard filter
(BA1/BS1/PM2) and as an explicit input feature. The distinction matters: a
frequency threshold and a frequency feature are different modelling decisions
and the literature does not always separate them.

### MaveDB
The community database for multiplexed assays of variant effect: over seven
million variant effects, 2,752 datasets, covering 713 human genes of which 487
have a known disease relationship
([MaveDB, PMC6827219](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6827219/)).

**Current state of the art for clinical use** is quantitative calibration to
ACMG/AMP evidence strengths rather than raw scores. MaveMD (2025) is the
clinical interface: 476,076 measurements from 82 MAVE datasets over 39
disease-associated genes, integrating ClinVar/ClinGen and exporting ACMG/AMP
evidence, reported to classify 75% of ClinVar VUS in those genes
([MaveMD, medRxiv 2025.11.15.25336228](https://www.medrxiv.org/content/10.1101/2025.11.15.25336228.full.pdf);
[PubMed 41332838](https://pubmed.ncbi.nlm.nih.gov/41332838/)). ⚠
Related tooling: [ClinMAVE, *NAR* 2026](https://academic.oup.com/nar/article/54/D1/D1355/8322704)
and [acmgscaler, bioRxiv 2025.05.16.654507](https://www.biorxiv.org/content/10.1101/2025.05.16.654507.full.pdf)
for standardised gene-level score calibration. ⚠

**Implication for us:** MaveDB is being consumed as *calibrated clinical
evidence*, not as raw training labels. That supports keeping it validation-only
(our standing decision) and makes "MaveDB as training supervision" an arm that
needs justifying, not a default.

---

## 2. The closest prior work — read this before claiming novelty

**AnnotateMissense** ([arXiv 2605.24520](https://arxiv.org/pdf/2605.24520)) ⚠ —
a genome-wide annotation and benchmarking framework that ran
**circularity-controlled ablation experiments** to measure the contribution of
individual feature categories, explicitly controlling for predictors trained on
overlapping clinical databases. Its reported findings:

- Removing prior pathogenicity predictors **and** population-frequency scores
  substantially reduced XGBoost performance.
- Removing the broader set of clinically-trained tools produced a *nearly
  identical* reduction — so the marginal contribution of tools with known
  ClinVar overlap, over and above other annotations, was limited.
- **Removing AlphaMissense and ESM-derived scores alone produced a negligible
  reduction** — protein-language-model features did not independently explain
  performance beyond the broader annotation set.

That third finding independently corroborates v1's own result, which nobody
believed at the time: a curated-feature head with no ESM input beat every cell
in our ESM grid. Two independent observations of the same effect is worth
considerably more than one, and it should be cited rather than re-discovered.

**It also constrains our novelty claim.** AnnotateMissense already did
feature-category ablation with gradient boosting. If our paper is described as
"which features matter", it is a replication on four genes.

---

## 3. What is actually unasked, and is therefore ours

The distinction the paper must draw, in one sentence:

> Prior work ablates **feature families inside one pooled model**. We compare
> **supervision sources as separate training regimes** — a model trained on
> ClinVar alone, versus on DMS alone, versus on both — and ask whether pooling
> helps, hurts, or merely averages.

These are different questions. Removing AlphaMissense's *score column* from a
ClinVar-trained model asks what that feature adds. Training one model on
ClinVar labels and another on DMS labels asks whether the two supervision
signals agree about what pathogenicity *is*. The arXiv 2405.06729 result — 0.78
correlation between clinical and DMS gains, high but well short of 1.0 — implies
the answer is "related but not interchangeable", and nobody appears to have
measured the pooling effect directly per source.

Secondary contributions that follow for free from the design:

1. **Per-source leave-one-gene-out**, so the comparison is on unseen-gene
   transfer rather than in-distribution fit.
2. **A recurrent baseline reported honestly** rather than omitted — the
   literature asserts transformers beat LSTMs here; on a four-gene panel we can
   show the number (see `vpdl/models/bilstm.py` for why it is framed as a
   baseline).
3. **A negative-result-safe protocol.** Three seeds minimum, provenance-gated
   pooling, orientation anchors. If pooling turns out not to help, that is a
   reportable result rather than a failed experiment.

---

## 4. Verification still owed

Before any of this reaches a manuscript:

- [ ] Read the four ⚠ papers in full; snippets are not evidence.
- [ ] Confirm AnnotateMissense's exact ablation protocol — whether its
      "sources" are features or supervision sets decides how close it really is.
- [ ] Check whether MaveMD/ClinMAVE cover any of MLH1/MSH2/MSH6/PMS2; if they
      do, their calibrated thresholds are a baseline we must clear, not ignore.
- [ ] Resolve every citation to authors/title/year/DOI. v1's `references.bib`
      carried entries with none of these, and no citation should be guessed.
