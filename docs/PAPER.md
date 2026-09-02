# Circularity by Default: A Leakage-Audited Multi-Source Benchmark for Missense Variant Pathogenicity Prediction, with a Lynch-Syndrome Instantiation

**Authors:** Jyothi Vikrama Simha Reddy Dirisinapu¹, [Co-author 2]¹, [Co-author 3]¹; supervised by [Supervisor], [Designation]¹

¹ Department of Information Technology, Chaitanya Bharathi Institute of Technology, Hyderabad, India

**Corresponding author:** [email]

**Running title:** Leakage-audited missense variant benchmarking

**Manuscript status:** working draft generated from the project content pack (`docs/PAPER_DRAFT.md`) and run log (`docs/RUNLOG.md`). Every quantitative claim is traced to a recorded run; slots in `[brackets]` and the items in §7.2 are not yet complete and must be filled or removed before submission. See the checklist in §12.

---

## Abstract

Computational prediction of missense variant pathogenicity increasingly relies on datasets assembled from heterogeneous public sources — clinical archives, deep mutational scanning (DMS) assays, published predictor scores, and proteome-wide priors. We show that such assembly is a systematic source of circular evaluation, and that the resulting inflation is largely invisible to ordinary metric monitoring.

We construct a unified table of 1,156,625 missense substitutions across 80 human proteins, integrating ClinVar, ProteinGym v1.3 (DMS assays, a clinical benchmark, and 17 published zero-shot scores), AlphaMissense, UniProt, gnomAD v4, AlphaFold DB, and InterPro under a single validated key `(uniprot_id, position, wt_aa, mut_aa)`, with SHA-256 provenance for every artefact and an independent 12-check integrity audit (12/12 pass). Wild-type residues are validated against canonical UniProt sequences before joining; 50,304 isoform-mismatched DMS rows and 152 AlphaMissense rows are discarded rather than merged.

Instrumenting this pipeline exposes five failure modes. (i) A DMS label-orientation inversion mislabelled approximately 185,000 variants while leaving cross-validated AUC *unchanged*. (ii) A DMS-derived feature equalled the inverted label by construction on 97.3% of labelled rows, producing a spurious ROC-AUC of 0.9987; the leakage-clean value is 0.716. (iii) Calibrators fitted on the folds they scored yielded an impossible isotonic expected calibration error of exactly zero. (iv) With only 2.7% of labelled rows clinically annotated, an unweighted objective optimises assay fitness rather than pathogenicity. (v) A family of silent implementation faults — a protein-language-model fine-tuner structurally unable to compute the one zero-shot score that already works, a feature cache validated by row count alone, a leave-one-gene-out split in which one gene silently carried a different feature space — none of which produced a bad metric.

After correction, a fused linear head over published priors attains ROC-AUC 0.963 and Matthews correlation coefficient (MCC) 0.78 on the clinical slice under residue-group-disjoint cross-validation, versus 0.945 and 0.75 for AlphaMissense alone; this is a new-residue-in-a-known-protein estimate, not an unseen-gene one. On a Lynch-syndrome instantiation (MLH1, MSH2, MSH6, PMS2) with a fail-closed PMS2 pseudogene gate and functional-assay evidence withheld from training by construction, leave-one-gene-out ESM-2 backbone fine-tuning reaches a mean ROC-AUC of 0.880 over the three genes with adequate sample size — currently *below* a frozen-feature baseline, a negative result we report rather than suppress. We release the pipeline, the audit, and the complete run log including every build failure and its root cause.

**Keywords:** missense variant interpretation; variants of uncertain significance; data leakage; circularity; protein language models; ClinVar; ProteinGym; AlphaMissense; Lynch syndrome; benchmark integrity

---

## 1. Introduction

### 1.1 The clinical problem

A missense variant substitutes one amino acid for another in a protein sequence. Clinical genetics must classify each such variant as pathogenic or benign under the ACMG/AMP guidelines (Richards et al. 2015). Most observed missense variants receive neither call: they are **Variants of Uncertain Significance (VUS)**, and the VUS backlog is the single largest bottleneck in returning actionable results to patients. ClinVar holds millions of submissions, but only a few thousand missense assertions per gene panel reach the two-review-star confidence threshold that supports supervised learning; this project's 80-gene build yields 5,152 clinically labelled rows against roughly 25,000 ClinVar VUS [recount and pin the ClinVar snapshot date before submission; see §3.2]. For a hereditary cancer syndrome the stakes are concrete: a VUS in *MLH1* leaves a family without a basis for colonoscopic surveillance or risk-reducing surgery.

### 1.2 Why the problem is computationally hard

Three properties of the data act as hard design constraints.

| Property | Consequence |
|---|---|
| Clinical labels are scarce and unevenly distributed | Supervision must be pooled across proteins and supplemented; single-gene deep models have too few labels |
| The plentiful labels are proxies | DMS measures *assay fitness*, not clinical consequence; published predictor scores were themselves trained on ClinVar |
| Leakage is the default outcome, not the exception | Same-residue variants across folds, paralogues straddling a split, assay-derived features that encode the label, calibrators fitted on the fold they score |

### 1.3 What the field currently does, and where it breaks

Protein language models (ESM-1b/ESM-2; Rives et al. 2021; Lin et al. 2023) produce strong zero-shot variant scores from pseudo-log-likelihood ratios (Meier et al. 2021). Supervised predictors such as AlphaMissense (Cheng et al. 2023) and generative models of evolutionary data such as EVE (Frazer et al. 2021) push accuracy further. ProteinGym (Notin et al. 2023) standardised large-scale benchmarking. But the standard recipe — pool ClinVar with DMS and published scores, cross-validate, report AUC — reintroduces exactly the circularity that Grimm et al. (2015) identified a decade ago, in a new form: **type-1 circularity**, where variants from the same protein or residue appear on both sides of a split, and **type-2 circularity**, where the label and a feature share a source. The metric looks fine; almost nobody instruments for the failure.

### 1.4 Our position

A variant-effect pipeline should be judged on the defensibility of its evidence as much as on its AUC. We therefore treat leakage control as a first-class subsystem rather than a preprocessing footnote, and we report the corrections that changed — or conspicuously failed to change — our own numbers as primary results.

### 1.5 Contributions

1. A **leakage-audited multi-source variant table** (1,156,625 substitutions, 80 proteins) with a single validated key, sequence-level wild-type validation, SHA-256 provenance, explicit label precedence, cross-source conflict quarantine, and a 12-check audit implemented *independently of the builder*.
2. **Five reproducible circularity and silent-failure findings**, each paired with the diagnostic that caught it — including one (label inversion) that left AUC unchanged and one (target leakage) that cost 0.28 AUC.
3. An **evaluation protocol** combining residue-group-disjoint, protein-disjoint, and MMseqs2 cluster-disjoint splits with nested calibration, clinical-only slices, 10,000-iteration bootstrap confidence intervals, and mandatory branch/fusion ablations on identical splits.
4. A **Lynch-syndrome instantiation** with pinned canonical references, a fail-closed PMS2 pseudogene gate, InSiGHT/ClinGen expert-panel tiering, and MaveDB/CIMRA functional evidence held out for validation only.
5. **Open, single-command reproducibility** across Linux, macOS, and Windows, with a closed-form GPU-memory preflight, a pytest suite, and a run log that records every build failure and its root cause.

---

## 2. Related work

**Zero-shot protein language models.** ESM-1b/ESM-2 score a variant by its pseudo-log-likelihood ratio, `PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)` (Meier et al. 2021). No labels are required, so there is no ClinVar circularity — but also no clinical calibration. EVE (Frazer et al. 2021) and GEMME (Laine et al. 2019) use MSA-based generative models instead. *Our difference:* we ingest all 17 ProteinGym zero-shot scores as features **and** retain them as baselines on identical splits, and we do not assume ESM-2 outperforms ESM-1b — it is tested with true masked-marginal scoring.

**Supervised proteome-wide predictors.** AlphaMissense (Cheng et al. 2023) is the current general-purpose prior. *Our difference:* it is used as both a feature and the baseline to beat, and — critically — we state explicitly that because AlphaMissense was trained on ClinVar, our clinical-slice comparison is plausibly temporally contaminated in AlphaMissense's favour. Most papers that use AlphaMissense as a baseline do not say this.

**Fine-tuning recipes for PLM-based variant effect prediction.** Four distinct recipes recur in the literature and are usually adopted rather than compared: a frozen linear probe (VariPred†), a Siamese wild-type/variant backbone fine-tune (ProPath†), a per-residue token classifier (CSBJ†), and frozen global-plus-local pooled wild-type/variant features with gated fusion (MVmamba†). *Our difference:* all four are implemented in one codebase and benchmarked head-to-head on identical mismatch-repair splits, so the recipe choice becomes an experimental result rather than an inherited assumption.

**Benchmarks and circularity.** Grimm et al. (2015) established the two circularity types. ProteinGym (Notin et al. 2023) standardised DMS-based benchmarking. *Our difference:* we instrument for circularity inside the pipeline — a target-leakage diagnostic, an independent audit, a nested calibration protocol — and we report the resulting metric collapse (0.9987 → 0.716) as a finding rather than quietly fixing it.

**Clinical variant curation for Lynch syndrome.** InSiGHT/ClinGen Variant Curation Expert Panels (Thompson et al. 2014 and successors) provide expert-panel mismatch-repair classifications. Functional assays are calibrated to ACMG evidence strengths via OddsPath (Brnich et al. 2020; Tavtigian et al. 2018). *Our difference:* expert-panel tier is an explicit training weight; MaveDB (Esposito et al. 2019) and CIMRA† evidence is withheld from training by construction so that agreement with it is a genuine external check; and PMS2 exon 11–15 pseudogene-region variants are gated fail-closed.

A structured comparison against these resources is given in Table 1.

**Table 1 — Positioning against prior methods and resources.**

| Method / resource | Supervision | Label sources | Split discipline reported | Circularity instrumentation | Calibration | Functional-assay use |
|---|---|---|---|---|---|---|
| ESM-1v / ESM-2 PLLR (Meier 2021; Lin 2023) | none (zero-shot) | — | n/a | n/a | none | — |
| EVE (Frazer 2021) | unsupervised (MSA) | — | n/a | n/a | GMM-based | — |
| GEMME (Laine 2019) | unsupervised (MSA) | — | n/a | n/a | none | — |
| AlphaMissense (Cheng 2023) | weakly supervised | population + ClinVar-adjacent | proteome-level | none reported | none | — |
| ProteinGym v1.3 (Notin 2023) | benchmark suite | DMS + clinical | assay-level | partial (DMS/clinical separated) | none | DMS is the target |
| VariPred† / ProPath† / CSBJ† / MVmamba† | supervised probe or fine-tune | ClinVar-family | residue-level (typical) | none reported | none | — |
| **This work** | supervised, provenance-weighted | ClinVar (≥2★) > PG-clinical > DMS, conflicts quarantined | residue-disjoint + protein-disjoint + MMseqs2 cluster-disjoint | 12-check independent audit + target-leakage diagnostic | nested temperature + isotonic | withheld from training; validation only |

† Reference details to be verified against the primary source before submission (§11).

---

## 3. Methods

### 3.1 Problem formulation

Given a protein sequence `X = x₁ … x_L` and a substitution `(i, wt → mut)` with `x_i = wt`, predict `P(pathogenic)`. The clinical target is the ACMG/AMP pathogenic-or-likely-pathogenic (P/LP) versus benign-or-likely-benign (B/LB) dichotomy. DMS assay fitness is treated as a **proxy** target and is deliberately kept as a separate evaluation slice throughout.

### 3.2 Data sources and integration

The table integrates ClinVar (clinical assertions), ProteinGym v1.3 (DMS assays, a clinical benchmark, and 17 published zero-shot predictor scores), AlphaMissense (a proteome-wide supervised prior), UniProt (canonical sequences and functional-site annotation), gnomAD v4 (population allele frequencies and constraint), AlphaFold DB (per-residue pLDDT and disorder), InterPro (domain membership), and MaveDB and CIMRA (functional assays, validation only).

Four integration rules carry the methodological substance.

1. **One key.** Every source is normalised to `(uniprot_id, position, wt_aa, mut_aa)`.
2. **Sequence-level validation before joining.** Rows whose wild-type residue disagrees with the canonical UniProt sequence are **dropped, not merged** — 50,304 DMS rows and 152 AlphaMissense rows on the 80-gene build. RefSeq `NP_` accessions are matched by exact whole-protein sequence equality, never by identifier string. A silent isoform or numbering mismatch is the most common way a multi-source variant table becomes quietly wrong: a dropped row is recoverable, a wrongly joined row is invisible.
3. **Label precedence and weighting.** ClinVar (≥2 review stars by default) takes precedence over ProteinGym-clinical, which takes precedence over a single-assay DMS bin. Every label carries a `label_source` tag and a `label_weight` (ClinVar weighted by review stars, ProteinGym-clinical 0.75, DMS 0.20).
4. **Conflict quarantine.** Cross-source disagreement leaves the `label` field blank and sets `label_conflict = 1`; the raw evidence is retained but the row cannot enter training. Heterogeneous public evidence cannot establish clinical truth; making disagreement explicit and unusable is more honest than picking a winner.

Every remote artefact is recorded in a SHA-256 provenance manifest with URL, version, byte count, and retrieval date. Downloads use atomic `.part` staging so that a server ignoring HTTP Range headers cannot corrupt a resumed transfer, and ZIP archives are integrity-checked before parsing. Two sources were rejected and the rejection recorded: dbNSFP (~30 GB and redundant with the ingested `zs_*` columns) and the EVE repository (ships multiple sequence alignments only). Recording a rejection is what stops the same evaluation being redone.

**Table 2 — Dataset composition.**

| | 10-gene build | 80-gene build |
|---|---|---|
| Proteins | 10 | 80 (every human protein with a ProteinGym v1.3 DMS assay) |
| Master table rows | 110,124 | **1,156,625** unique substitutions |
| Labelled rows | 14,131 | **190,494** (82,149 P/LP; 108,345 B/LB) |
| Clinical-slice rows | — | 5,152 (**2.7%** of labelled) |
| AlphaMissense coverage | 10 proteomes | 80 proteomes (~1.16 M scored) |
| Rows dropped by wild-type validation | — | 50,304 DMS + 152 AlphaMissense |
| ClinVar ∩ ProteinGym-clinical | — | 1,919 overlapping variants, **100%** label agreement |
| Independent audit | — | **12 / 12 checks pass** |

*Caveat to resolve before submission:* the master-table ClinVar-labelled count (~5,466) and the training-set clinical-slice count (5,152) are measured on different tables — the 1,156,625-row master and the 190,494-row training CSV after label precedence and conflict quarantine, respectively. Both must be recounted on the same freshly built table and the measurement table stated for each. The same figure pair appears in §1.1 and §4.4.

### 3.3 The independent audit

`scripts/audit_extended_dataset.py` re-derives the entire merge from scratch as a **separate implementation** and runs 12 checks: key uniqueness, wild-type residue validation, label-precedence re-derivation, provenance-token coherence, cross-source conflict detection, and feature-coverage accounting. The methodological point is explicit: a builder that validates itself only proves that it is self-consistent. An audit that shares no code with the builder can catch an error that is baked into both the build and its self-check.

### 3.4 Feature construction

For a variant `(i, wt → mut)` in sequence `X`, the protein-language-model feature block is

```
z = [ h_wt ‖ h_mut ‖ (h_mut − h_wt) ‖ |h_mut − h_wt| ‖ PLLR_i ]
PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)
```

where `h_wt` and `h_mut` are last-layer ESM-2 hidden states at residue `i` for the wild-type and the in-silico-mutated sequence. Both log-probabilities are conditioned on the *same* masked context, so both are read from a **single forward pass** over the wild-type sequence (Meier et al. 2021). Sequences beyond the 1,022-residue positional capacity are handled with overlapping sliding windows: hidden states are averaged across covering windows and log-probabilities are averaged in log space. Mixed-precision (fp16/bf16) autocast is used on CUDA, and all features are cached to disk keyed by the variant, not by row position.

Approximately 22 external prior columns are optionally appended, each accompanied by an `is_missing_*` indicator so that "not scored" is distinguishable from a real median prediction. The leakage-safe prior set is defined once, in `src.transfer.TRANSFER_PRIOR_COLS`; DMS-derived and functional-assay validation-only columns are deliberately absent, and a regression test fails if a newly joined source is not wired through.

### 3.5 Models

| Component | Description |
|---|---|
| Residual MLP head | `Linear(d→h)` → *N* × (`Linear → LayerNorm → GELU → Dropout` + residual) → `Linear(h→1)`; 256 hidden units, dropout 0.15 |
| Branch / concat / gated fusion | Frozen-branch projections to a shared dimension; a sigmoid branch gate with softmax per-feature gating, GLU, and a residual connection, adapted from the MVmamba† recipe |
| ESM-2 fine-tune, Siamese | ProPath† recipe: separate wild-type and variant forward passes, pooled at the mutated residue, with a head over `[h_wt ‖ h_vt ‖ Δ ‖ |Δ|]` |
| ESM-2 fine-tune, token classifier | CSBJ† recipe: per-residue classification at the mutated site |
| MVmamba features | Global plus local pooled wild-type/variant states with a ±3-residue window, mutation-centred for long chains |

The backbone freeze depth is a parameter: `n_unfrozen_layers = −1` is a full fine-tune, `0` is a frozen backbone (the ablation floor), and `N > 0` unfreezes the last *N* transformer layers. Gradient checkpointing is available.

### 3.6 Optimisation

Binary cross-entropy is the default loss. Weighted BCE uses a **fixed** class ratio computed from the fitting partition rather than recomputed per mini-batch. Focal loss is retained for comparison with a neutral `alpha = 0.5`. Training uses AdamW (Loshchilov and Hutter 2019) with cosine annealing, early stopping with best-weight restoration, patience 10, a maximum of 60 epochs, and a learning rate of 3 × 10⁻⁴. Clinical rows carry a configurable loss weight (default 5.0), and early stopping is monitored on **clinical-only** ROC-AUC whenever both classes are present in the validation slice.

For the ESM-2 backbone fine-tune (Stage 2b), the backbone learning rate is 1 × 10⁻⁵, the head learning rate 3 × 10⁻⁴, the effective batch size 8, and training runs for 10 epochs under mixed precision with gradient accumulation and gradient checkpointing. Both learning rates follow a linear warmup over the first 10% of optimiser steps and then a cosine decay to zero, and the loss carries a `pos_weight` computed from the fitting partition's class ratio; per-gene positive prevalence in this panel ranges from 40% to 82%, so a single unweighted objective is not appropriate across folds. The zero-shot term log P(mut | X) − log P(wt | X) enters as a fixed-scale, zero-initialised residual on the logit, which makes the untrained model exactly the zero-shot predictor (ROC-AUC ≈ 0.834 pooled over these genes) and every deviation from it attributable to training.

**Stage-2b ablation grid.** The Stage-2b result is reported against a pre-registered three-axis ablation on the identical leave-one-gene-out splits (§7.2, Table 6): the **feature branch** (`esm`, the ESM feature block and the zero-shot term alone, versus `esm+priors`, which additionally routes the same published-prior columns the frozen probe reads — AlphaMissense, the ProteinGym zero-shot scores and their rank transforms, allele frequency and its ACMG flags, structural and domain features — through the fusion head, with the five gene-level gnomAD constraint columns dropped on both sides under leave-one-gene-out because across genes they encode gene identity rather than variant evidence) crossed with **freeze depth** (`n_unfrozen_layers ∈ {0, 2, −1}`) and the **zero-shot term** (residual, concatenated, or absent), giving 16 cells in five tiers with three seeds on the two headline cells. The branch axis exists because without it the comparison is not interpretable: Stage 2b as first implemented read none of the prior columns, so any gap against the frozen probe confounds freeze depth with feature set. Two decision rules are fixed before the runs. First, if the frozen floor's held-out mean falls inside the full fine-tune's bootstrap confidence interval *at the same branch*, backbone gradients are declared to have added nothing at this training-set size and the frozen configuration is reported as the model. Second, if `esm+priors` does not clear the priors-only probe, ESM-2 is reported as adding nothing on top of published priors for unseen-gene prediction in this panel. Tiers are ordered by scientific value so that an interrupted run degrades gracefully, and the driver is resumable: every artefact is tagged with its cell identity, and a cell is re-run only if any of its three outputs is missing.

### 3.7 The evaluation protocol

**Table 3 — Leakage controls.**

| Control | Implementation | What it prevents |
|---|---|---|
| Residue-group disjointness | `StratifiedGroupKFold` on `"{uniprot_id}:{position}"`, asserted at runtime | Different substitutions at one residue on both sides of a split |
| Protein-level holdout | Leave-one-protein-out; leave-one-gene-out for mismatch-repair genes | A new-residue-in-known-protein number reported as a new-gene number |
| Sequence-cluster disjointness | MMseqs2, 20% identity / 20% coverage; whole clusters assigned to one side | Paralogues and near-duplicate isoforms straddling a split |
| Nested calibration | Outer validation fold untouched; outer-training data group-split into fit / early-stop / calibration partitions | Calibrators fitted on the fold they score |
| Clinical-only slices | Metrics reported separately for all labels and for ClinVar ∪ ProteinGym-clinical | DMS assay-fitness performance quoted as clinical performance |
| Target-leakage diagnostic | `scripts/diagnose_overfitting.py` | Features that encode the label by construction |
| Bootstrap CIs and tuned thresholds | 10,000-iteration percentile confidence intervals; MCC-optimal threshold tuned on an inner slice only | Point estimates on a few-thousand-row set read as precise |
| Mandatory branch ablations | ESM-only, priors-only, pretrained-fused, concat, gated fusion — identical splits | A fusion architecture credited for a single branch's gain |
| Schema-drift abort | Checkpoints store the prior-column order and the ESM block dimension; a mismatch aborts the run | A transfer run silently invalidating itself |
| Persisted imputation constants | Imputation medians stored in the checkpoint and reused across stages | Warm-starting a head onto differently centred inputs |

Reported metrics are ROC-AUC, PR-AUC, MCC (at an MCC-optimal threshold tuned on an inner slice), Brier score, expected and maximum calibration error, and reliability diagrams. The pipeline default of 10,000 bootstrap iterations is honoured across all stages; standalone scripts default to 2,000 and must be passed the larger value explicitly.

### 3.8 The Lynch-syndrome instantiation

Five safeguards, each with its justification.

1. **Pinned canonical references with hard-fail length checks** — *MLH1* P40692 / 756 aa, *MSH2* P43246 / 934 aa, *MSH6* P52701 / 1,360 aa, *PMS2* P54278 / 862 aa. A silent reference update cannot shift numbering underneath a build.
2. **Fail-closed PMS2 exon 11–15 pseudogene gate.** Short-read sequencing calls inside the *PMS2CL* homology region are untrustworthy. The build **refuses to run** without one of: a homology CSV with orthogonally confirmed flags, an explicitly verified codon range, or an instruction to exclude PMS2 entirely. Unconfirmed homology-region labels are withheld, never guessed. The general principle — *no supervision is better than supervision you cannot defend* — is the paper's ethos in one line.
3. **InSiGHT / ClinGen expert-panel tiering** recorded as an explicit evidence tier and training weight.
4. **Orthogonal functional evidence held out by construction.** MaveDB score sets (an *MSH2* loss-of-function screen; an *MLH1* cellular-abundance assay — a different evidence class) and CIMRA OddsPath values live in a validation-only column set, never in the training prior set. CIMRA rows with a hypothesised splicing mechanism are excluded automatically, because a cell-free assay cannot detect splicing.
5. **Leave-one-mismatch-repair-gene-out evaluation** with bootstrap confidence intervals, because the clinically relevant question for a rare-disease gene is transfer to an unseen gene, not interpolation within a known one.

Training proceeds in two stages plus an optional third. Stage 1 pretrains the head on frozen ESM-2 embeddings over all 80 panel genes, excluding every mismatch-repair gene for a true transfer estimate. Stage 2 warm-starts on mismatch-repair clinical labels only, still with frozen embeddings. Stage 2b is the true ESM-2 backbone gradient fine-tune.

### 3.9 Compute and reproducibility

The build machine is Linux with 12 CPU cores, 14 GB RAM, and no GPU; Python 3.14.7; `torch` 2.13.0+cpu, `transformers` 5.15.1, `pandas` 3.0.5, `numpy` 2.5.2, `scikit-learn` 1.9.0, `scipy` 1.18.1. The `+cpu` build is stated explicitly because the Stage-2b numbers in §4.7 were produced on a *different* machine (Windows, CUDA, ~15 GiB); a reproducibility section that conflates the two would be wrong. Total downloads are approximately 1.27 GB. The AlphaMissense streaming filter scanned 216,175,352 lines (about 7 minutes, cached thereafter). A pytest suite (149 tests at the latest run) covers parsing, splitting, API adapters, fine-tuning logic, the ablation-grid definition, and the allele-frequency label/feature quarantine.

We include a **GPU-memory budget analysis** as a practical contribution. The default Stage-2b configuration (650M checkpoint, Siamese recipe, full unfreeze, micro-batch 8, 1,022-residue crops) requires approximately **51 GiB in fp32** — about 10 GiB of weights, gradients, and AdamW state, plus about 41 GiB of stored activations, with the Siamese design doubling the activation term. A closed-form preflight refuses an impossible configuration *before* the multi-hour download stages rather than failing in the backward pass at the end. On a ~15 GiB card the full Siamese fine-tune fits at micro-batch 1 with gradient accumulation 8 and gradient checkpointing (about 10 GiB, effective batch still 8); below about 11 GiB a full fine-tune is impossible because the static AdamW state alone is about 9.7 GiB.

---

## 4. Results

The central results are five integrity findings (§4.1–4.5), a leakage-clean supervised benchmark (§4.6), and a leave-one-gene-out protein-language-model fine-tune on the Lynch-syndrome panel (§4.7). Experiments not yet complete are listed in §7.2 and are not reported as results.

### 4.1 Finding 1 — Label inversion: approximately 185,000 labels were backwards, and AUC did not move

ProteinGym's `DMS_score_bin = 1` marks the *top half of assay fitness* — the tolerated variants. The builder mapped that value directly to `label = 1` ("pathogenic"), inverting approximately 185,000 DMS-derived labels. Two independent anchors confirmed the inversion:

| Anchor | Evidence |
|---|---|
| AlphaMissense | `P(AM ≥ 0.9 | bin = 1) = 0.217` versus `P(AM ≥ 0.9 | bin = 0) = 0.506` → `bin = 1` is benign |
| ProteinGym clinical | mean DMS score **−0.8** for pathogenic versus **+1.9** for benign |

The fix (`label = 1 − dms_bin`) was mirrored in the auditor, the dataset was rebuilt, and the audit re-passed 12/12. The finding worth stating in the abstract is that **cross-validated AUC was unchanged**: AUC is invariant to a global label flip once the classification head re-learns the mapping. Only the semantics — and any prospective risk ranking — were wrong. This is precisely why the bug class survives ordinary metric monitoring, and it argues for external semantic anchors as a standard integrity check.

### 4.2 Finding 2 — Target leakage: ROC-AUC 0.9987 → 0.716

The first full run reported ROC-AUC **0.9987** on all labels. The target-leakage diagnostic established four things.

1. On single-assay rows — **97.3% of labelled data** — the feature `dms_bin_median` equals `1 − label` **by construction**. The answer was in the feature matrix.
2. With leaky features removed, the honest all-labels number is **0.716**.
3. **There was no classic overfitting.** The per-fold train-versus-validation AUC gap is approximately 0.000 (range −0.001 to +0.011), and folds early-stop within 2–16 epochs. The pathology was target leakage, not model capacity, and a standard overfitting diagnostic would have found nothing.
4. Using `−DMS_score` as a baseline is likewise circular on the DMS slice, because the label is a median split of that same score; only its clinical-slice value is interpretable.

### 4.3 Finding 3 — In-fold calibration

Temperature and isotonic calibrators were fitted on each outer validation fold and then scored on that same fold. The tell was an isotonic expected calibration error of **exactly zero** — an impossible out-of-sample value. Calibration is now nested: the outer validation fold is left untouched, and both calibrators are fitted on a dedicated calibration partition carved by residue group from the outer-training data. This costs fitting data per fold and buys valid out-of-fold calibration, Brier, ECE, and threshold metrics.

### 4.4 Finding 4 — The objective was optimising the wrong target

Only **5,152 of 190,494** labelled rows (**2.7%**) carry a clinical label; approximately 188,000 are DMS aggregates. An unweighted loss therefore optimises assay fitness, not pathogenicity. Two corrections follow: a configurable clinical loss weight (default 5.0) and early stopping monitored on clinical-only ROC-AUC. Separately, the prior default focal-loss `alpha = 0.25` was inappropriate — positives are 43% of all labels and about 52% of clinical labels — so it was down-weighting the positive class; it is replaced by a neutral 0.5, with plain BCE as the default.

### 4.5 Finding 5 — Silent-failure classes found by reading code, not by watching metrics

None of the following produced a bad metric.

| Failure | Mechanism | Fix |
|---|---|---|
| Fine-tuned model could not see the zero-shot ESM score | The fine-tuner was built on the encoder alone, with no masked-LM head, so `log P(mut) − log P(wt)` — the roughly 0.834-AUC untrained signal — was structurally uncomputable. A 650M-parameter backbone on a few hundred labels was asked to rediscover it and early-stopped first | Load the masked-LM model; read PLLR off the same wild-type forward pass; a flag disables it for the ablation. Verified identical to the standalone PLLR extractor to 2.4 × 10⁻⁶ |
| Siamese head fed unnormalised ESM states | A `LayerNorm` sat *after* the first `Linear`; raw ESM hidden states have large position-dependent norms and half the Siamese vector is near-duplicate, so the head spent capacity on scale rather than on the substitution | `LayerNorm` before the head; PLLR scaled by a fixed registered constant, not a batch statistic |
| One gene silently carried a different feature space inside a leave-one-gene-out design | For chains over the positional cap, the "variant-type" local window was sliced from the *wild-type* sequence, so two of eight fusion feature blocks were identically zero — only for the one gene (*MSH6*, 1,360 aa) that exceeds the cap | Window taken from the substituted sequence; a regression test on the wild-type/variant contrast |
| Backbone comparison reported `1 − AUC` | Raw PLLR is negative for damaging variants but was scored against a pathogenic-equals-1 label, so every cell came out below 0.5 — output that reads as "the language model is useless" | A signed pathogenicity score; any earlier reading of that script is discarded |
| Vacuous mutant-residue check | A vectorised string-length comparison truncated every value to one character, so the check always passed | Explicit per-row amino-acid validation |
| Feature cache validated only row count | A rebuilt table with the same row count in a different order returned another variant's embeddings, with no error and no metric change | Cache hits validated by the variant key |
| Missingness indistinguishable from prediction | Most zero-shot prior columns are absent for approximately 99% of rows; median imputation made "not scored" look like a real median prediction | An `is_missing_*` indicator for every numerical prior |
| Imputation constants recomputed per stage | Stage 1 used 80-gene medians, Stage 2 used 4-gene medians; since the imputed constant *is* the feature for approximately 99% of rows, warm-starting carried weights onto differently centred inputs | Constants persisted in the checkpoint and reused |
| Cross-protein group collapse | The grouping key was the raw residue position, collapsing equal positions of different proteins into one leakage group | Group key is `"{uniprot_id}:{position}"` |
| Half-built virtual environment counted as ready | Environment creation finishes in a second; the dependency install takes minutes, so an interrupted install was silently adopted | Readiness gated on a post-install stamp file |

### 4.6 Table 4 — Leakage-clean performance (priors mode, 80 genes, 3-fold residue-group-disjoint CV)

| Slice | Model | ROC-AUC | MCC |
|---|---|---|---|
| All labels (190,494) | MLP + isotonic | 0.716 | 0.32 |
| All labels | AlphaMissense | 0.709 | 0.30 |
| **Clinical only (5,152)** | **MLP + isotonic** | **0.963** | **0.78** |
| Clinical only | AlphaMissense | 0.945 | 0.75 |

Four points must accompany this table.

- **0.716 is the honest all-labels number.** Predicting assay fitness without seeing the assay is intrinsically hard; the earlier 0.9987 was circularity, not learning.
- **The clinically meaningful headline is 0.963 / 0.78**, a **+1.8-point AUC** gain over AlphaMissense from combining it with 17 published zero-shot scores and domain flags — without DMS features and without ESM.
- **Calibration.** Temperature scaling moved ECE from 23% to 22%; isotonic regression to 11%. Fixed-threshold metrics are sensitive to score shift, so the AUC and the calibrated variants are the ones to trust.
- **Scope caveat, stated in the caption itself.** Folds group by `uniprot_id:position`, not by protein, so this is a *new-residue-in-a-known-protein* estimate. It is **not** an unseen-gene number, and because AlphaMissense was trained on ClinVar the comparison is plausibly contaminated in AlphaMissense's favour, which makes the +1.8-point gain conservative but uncontrolled.

### 4.7 Table 5 — Lynch-syndrome leave-one-gene-out, ESM-2 Siamese fine-tune

Each mismatch-repair gene is held out entirely while Stage 2b fits the head and the ESM-2 backbone on the other three, so these are genuine unseen-gene estimates, unlike §4.6. Two Stage-2b runs exist; we report the second and keep the first only as the delta that isolates the PLLR fix (Finding 5).

**Run 2 (current).** Incorporates the Finding 5 corrections — chiefly that the PLLR term is now computable by the fine-tuned model — and includes PMS2 for the first time, via a verified codon range, so the 21 held-out PMS2 rows are homology-checked.

| Held-out gene | ROC-AUC | 95% CI | PR-AUC | MCC | Threshold | *n* | Best epoch |
|---|---|---|---|---|---|---|---|
| MLH1 | **0.903** | 0.851–0.947 | 0.976 | 0.528 | 0.932 | 208 | 5 |
| MSH2 | **0.850** | 0.806–0.891 | 0.767 | 0.458 | 0.896 | 335 | 2 |
| MSH6 | **0.887** | 0.820–0.945 | 0.912 | 0.644 | 0.513 | 119 | 1 |
| PMS2 | 0.956 | 0.838–1.000 | 0.990 | 0.691 | 0.068 | 21 | 3 |

Mean ROC-AUC **0.880** over the three genes with adequate sample size (MLH1, MSH2, MSH6; sample-size-weighted 0.873), or 0.899 including PMS2. Mean MCC 0.543 over the three genes.

**Run 1 (superseded).** With the PLLR term structurally uncomputable and PMS2 absent: MLH1 0.899, MSH2 0.878, MSH6 0.907; mean ROC-AUC 0.895, mean MCC 0.492. Run 1 → Run 2 moved the mean ROC-AUC by −0.015 and the mean MCC by +0.051, almost all of the MCC gain coming from MSH2 (0.297 → 0.458) as its MCC-optimal threshold moved from 0.057 to 0.896.

Four points must accompany this table.

1. **The fine-tuning is not yet attributed.** The best epoch is 1–5; the frozen-backbone ablation floor (§7.2) has not been run; and a frozen linear probe over published priors on the same leave-one-gene-out splits currently *leads* Stage 2b on ROC-AUC by roughly 4–6 points (mean 0.923 for the warm-started probe, 0.945 for the from-scratch probe). Until the ablation lands, this is an ESM-2-*representation* result, not a fine-tuning result.
2. **Thresholds do not transfer.** The MCC-optimal threshold spans 0.068 (PMS2) to 0.932 (MLH1). Any clinical use requires a per-gene threshold or per-gene recalibration — a finding in its own right.
3. **Per-gene base rates differ sharply.** PR-AUC moves opposite to ROC-AUC between MLH1 (0.976 versus 0.903) and MSH2 (0.767 versus 0.850). Positive-class prevalence must be reported per gene.
4. **PMS2 rests on 21 held-out rows** (CI 0.838–1.000). The point value carries almost no information; the row is kept for completeness and excluded from every headline mean.

Both runs are single-seed. The provenance of Run 2 (exact freeze depth, seed, and memory-fit flags) is to be reconciled against the run's summary file before submission.

---

## 5. Discussion

**What the corrections mean generally.** Three of the five findings changed what the model was learning; one changed nothing measurable and everything semantic. In multi-source variant-effect datasets, the dangerous errors are *well-formed but wrong values in places no metric looks*. This motivates audit-first construction: an integrity audit implemented independently of the builder, external semantic anchors for label orientation, and target-leakage diagnostics as routine.

**What the numbers do and do not license.** The 0.963 clinical-slice result is a real, leakage-clean, residue-disjoint number, and it exceeds AlphaMissense by 1.8 points. It is *not* an unseen-gene result, and AlphaMissense's own ClinVar training makes even that comparison contaminated in AlphaMissense's favour. Both points belong in the results text, not only in the limitations.

**Why 0.716 is the interesting number.** The all-labels task asks a model to predict assay fitness without seeing the assay. That it is hard is informative: DMS bins and clinical labels are genuinely different targets, which is the empirical justification for the clinical-weighted objective and the separate evaluation slices.

**On whether to fine-tune at all.** With only a few hundred clinical training variants per unseen-gene fold, Stage-2b backbone fine-tuning of a 650M-parameter model early-stops within one to five epochs and, on current evidence, does not clear a frozen-feature baseline. The pre-registered ablation (§7.2) tests whether a linear probe on the same frozen ESM-2 features matches the full fine-tune. If it does — the pre-registered expectation — the finding is that at this label budget the pretrained representation is the asset and the gradient signal is too thin to improve it without overfitting. That is the empirical justification for the frozen-backbone two-stage design on rare-disease genes, and a useful counterweight for a subfield that increasingly treats full fine-tuning as the default. A negative result, pre-registered and honestly reported, is a contribution.

**Clinical translation.** Calibrated scores map to ACMG/AMP evidence strengths through the Tavtigian/Brnich Bayesian framework; expert-panel tiers give a gold-standard subset; and the PMS2 gate reflects the reality that a sequencing technology's blind spot must propagate into a model's refusal to opine. The immediate application is VUS triage: once ESM features are cached, all roughly 966,000 unlabelled substitutions — including approximately 25,000 ClinVar VUS — can be scored into calibrated risk tiers.

**A recommendation to the field.** Benchmark papers should report the split rung (residue, protein, or cluster), state whether any baseline was trained on the evaluation labels, and publish a target-leakage diagnostic. We offer this pipeline's audit as a template.

---

## 6. Limitations

1. **No unseen-gene estimate on the broad panel.** The 80-gene headline numbers are residue-disjoint within known proteins. An unseen-gene estimate exists for the mismatch-repair panel (§4.7, mean ROC-AUC 0.880 over three genes) but covers four genes and roughly 683 held-out rows.
2. **AlphaMissense contamination.** AlphaMissense was trained on ClinVar; the clinical slice may be temporally contaminated, biasing the baseline upward. This makes the +1.8-point gain conservative but uncontrolled. Mitigation: a temporal ClinVar holdout of variants first submitted after the AlphaMissense cutoff.
3. **DMS binarisation is a proxy target.** The recommended fix — multitask learning with within-assay-normalised continuous fitness as one task and clinical pathogenicity as another — needs a new model head and data interface and is not implemented.
4. **Sparse priors.** Most zero-shot columns are absent for approximately 99% of rows; the imputed constant *is* the feature for most variants. Missingness indicators mitigate but do not solve this.
5. **The fine-tuning is unattributed.** Stage 2b reports (§4.7), but with early stopping at epoch 1–5, no frozen-backbone ablation floor, and a frozen probe currently ahead on ROC-AUC by 4–6 points, the honest framing until the ablation lands is that fine-tuning did not beat the probe.
6. **PMS2 rests on 21 held-out rows.** It is included via a verified codon range but excluded from every headline mean; without a verified span the fail-closed gate drops it entirely and the mismatch-repair result is three genes.
7. **Single-run results.** Reported numbers are single runs, not seed-averaged. Mean ± standard deviation over at least three seeds is planned for every reported configuration.
8. **Mismatch-repair functional-assay coverage is uneven.** MaveDB has *MSH2* and *MLH1* score sets; *MSH6* and *PMS2* have none, so external validation is partial.
9. **Provenance-weighted labels are heuristic.** The weights (review stars, 0.75, 0.20) are reasoned but not tuned or validated.

---

## 7. Conclusion and planned experiments

### 7.1 Conclusion

Circularity is the default outcome when a missense variant-effect dataset is assembled from heterogeneous public sources, and the resulting inflation is largely invisible to ordinary metric monitoring. We have described a leakage-audited pipeline, five reproducible failure modes it exposes, and the honest numbers that remain after correction: ROC-AUC 0.963 on the clinical slice under residue-disjoint cross-validation, 0.716 on the harder all-labels task, and a mean 0.880 unseen-gene ROC-AUC on the Lynch-syndrome panel that does not yet beat a frozen-feature baseline. The contribution is the instrumentation and the honesty of the evaluation, not a new state-of-the-art model.

### 7.2 Planned experiments before submission

The following are specified and scripted but not yet run; each either fills a slot in §4 or removes a claim.

1. **Leave-one-protein-out cross-validation** on the broad 80-gene panel — the single most important gap; without it the paper has no unseen-gene claim outside the mismatch-repair panel. A substantial drop from 0.963 is expected and reporting it honestly is the stronger result.
2. **MMseqs2 cluster-disjoint split** (20% identity / 20% coverage) — completes the three-rung difficulty ladder (residue, protein, cluster).
3. **Seed averaging** (≥ 3 seeds) on every reported configuration.
4. **ESM feature contribution** — priors-only versus ESM-only versus ESM-plus-priors on identical splits, to test whether protein-language-model features add clinical signal beyond published scores.
5. **Fine-tuning strategy benchmark** — all four recipes (frozen probe, Siamese fine-tune, token classifier, gated fusion) on identical mismatch-repair splits, plus ESM-1b versus ESM-2 masked-marginal zero-shot.
6. **Stage-2b branch × freeze-depth × zero-shot-term ablation (Table 6).** Sixteen cells over the feature branch (`esm`, `esm+priors`), freeze depth (`n_unfrozen_layers ∈ {0, 2, −1}`), and the zero-shot term (residual, concatenated, absent), with three seeds on the two headline cells, on the leave-one-gene-out splits; approximately 31 GPU-hours on a 15 GiB card, of which tier 1 (about 16 hours) alone settles the headline. This is the experiment that attributes — or retracts — the §4.7 fine-tuning claim, and the branch axis is what separates that claim from the feature set. Pre-registered reading: if the frozen floor's held-out mean sits inside the full fine-tune's 95% confidence interval at the same branch, report the frozen probe as the model and the fine-tune as a negative result; and if `esm+priors` does not clear the priors-only probe, report ESM-2 as adding nothing on top of published priors here.

**Table 6 — Stage-2b branch × freeze-depth × zero-shot-term ablation (planned).** Sixteen cells; the seed column gives the seeds run for each. Cell names are the identifiers carried by every artefact of that run.

| Cell | Branch | `n_unfrozen` | Zero-shot term | Seeds | ROC-AUC MLH1 | MSH2 | MSH6 | PMS2 | Mean (3-gene) |
|---|---|---|---|---|---|---|---|---|---|
| Frozen priors probe (comparator) | priors only, no ESM | — | — | 1 | 0.964 | 0.901 | 0.969 | 1.000 | 0.945 |
| `esmpri_concat_full_pllr-residual` | esm+priors | −1 | residual | 42/43/44 | | | | | |
| `esmpri_concat_frozen_pllr-residual` | esm+priors | 0 | residual | 42/43/44 | | | | | |
| `esm_full_pllr-residual` | esm | −1 | residual | 42 | | | | | |
| `esm_frozen_pllr-residual` | esm | 0 | residual | 42 | | | | | |
| `esmpri_concat_frozen_pllr-off` | esm+priors | 0 | absent | 42 | | | | | |
| `esm_frozen_pllr-off` | esm | 0 | absent | 42 | | | | | |
| `esmpri_concat_frozen_pllr-concat` | esm+priors | 0 | concatenated | 42 | | | | | |
| `esm_frozen_pllr-concat` | esm | 0 | concatenated | 42 | | | | | |
| `esmpri_gatewave_frozen_pllr-residual` | esm+priors, GateWave fusion | 0 | residual | 42 | | | | | |
| `esmpri_concat_last2_pllr-residual` | esm+priors | 2 | residual | 42 | | | | | |
| `esm_last2_pllr-residual` | esm | 2 | residual | 42 | | | | | |
| `esmpri_concat_full_pllr-off` | esm+priors | −1 | absent | 42 | | | | | |
| §4.7 Run 2 (historical, not re-run) | esm | −1 | concatenated | 1 | 0.903 | 0.850 | 0.887 | 0.956 | 0.880 |

The §4.7 Run 2 row is carried for continuity only: its zero-shot term entered concatenated with the ESM features, so its nearest grid cell is `esm_full_pllr-residual`, which differs in how that term enters. Report each cell as mean ± standard deviation over the seeds available for it, and keep PMS2 out of every headline mean.

7. **Temporal ClinVar holdout** — the only way to defuse the AlphaMissense contamination objection.
8. **MaveDB / CIMRA external validation** — correlate model scores against the withheld *MSH2* loss-of-function and *MLH1* abundance assays.

---

## 8. Data and code availability

All code is available at `[repository URL]` under `[licence]`. The pipeline is reproducible with a single command on Linux, macOS, and Windows. Every remote artefact is recorded with URL, version, byte count, retrieval date, and SHA-256 checksum in the provenance manifest. Source data are public: ClinVar (NCBI; snapshot date `[pin before submission]`), ProteinGym v1.3, AlphaMissense (Google DeepMind, CC BY-NC-SA 4.0 — verify this non-commercial licence against the target venue's terms), UniProt (CC BY 4.0), gnomAD v4, AlphaFold DB (CC BY 4.0), InterPro, and MaveDB. CIMRA OddsPath values are extracted from published supplementary tables and are not redistributed. Per-source licences are documented in the repository.

---

## 9. Author contributions

[To be completed. Suggested structure: pipeline and experiments — [names]; data integration and audit — [names]; clinical framing and Lynch-syndrome safeguards — [names]; supervision — [Supervisor].]

## 10. Acknowledgements

[To be completed — funding, compute, and any curated-data acknowledgements. The Stage-2b runs used a CUDA workstation provided by [source].]

---

## 11. References

*Verified and safe to cite as listed:*

1. Richards S. et al. (2015). Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. *Genetics in Medicine* 17:405–424.
2. Tavtigian S. V. et al. (2018). Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework. *Genetics in Medicine* 20:1054–1060.
3. Brnich S. E. et al. (2020). Recommendations for application of the functional evidence PS3/BS3 criterion using the ACMG/AMP sequence variant interpretation framework. *Genome Medicine* 12:3.
4. Grimm D. G. et al. (2015). The evaluation of tools used to predict the impact of missense variants is hindered by two types of circularity. *Human Mutation* 36:513–523.
5. Landrum M. J. et al. (2018). ClinVar: improving access to variant interpretations and supporting evidence. *Nucleic Acids Research* 46:D1062–D1067.
6. Rives A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118:e2016239118.
7. Meier J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *Advances in Neural Information Processing Systems* 34.
8. Lin Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379:1123–1130.
9. Cheng J. et al. (2023). Accurate proteome-wide missense variant effect prediction with AlphaMissense. *Science* 381:eadg7492.
10. Frazer J. et al. (2021). Disease variant prediction with deep generative models of evolutionary data. *Nature* 599:91–95.
11. Laine E., Karami Y., Carbone A. (2019). GEMME: a simple and fast global epistatic model predicting mutational effects. *Molecular Biology and Evolution* 36:2604–2619.
12. Brandes N. et al. (2023). Genome-wide prediction of disease variant effects with a deep protein language model. *Nature Genetics* 55:1512–1522.
13. Notin P. et al. (2023). ProteinGym: large-scale benchmarks for protein fitness prediction and design. *Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track*.
14. Chen S. et al. (2024). A genomic mutational constraint map using variation in 76,156 human genomes. *Nature* 625:92–100.
15. Karczewski K. J. et al. (2020). The mutational constraint spectrum quantified from variation in 141,456 humans. *Nature* 581:434–443.
16. Jumper J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596:583–589.
17. Varadi M. et al. (2022). AlphaFold Protein Structure Database. *Nucleic Acids Research* 50:D439–D444.
18. Esposito D. et al. (2019). MaveDB: an open-source platform to distribute and interpret data from multiplexed assays of variant effect. *Genome Biology* 20:223.
19. Steinegger M., Söding J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology* 35:1026–1028.
20. Guo C. et al. (2017). On calibration of modern neural networks. *International Conference on Machine Learning*.
21. Zadrozny B., Elkan C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *ACM SIGKDD*.
22. Lin T.-Y. et al. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision*.
23. Loshchilov I., Hutter F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*.
24. Thompson B. A. et al. (2014). Application of a 5-tiered scheme for standardized classification of 2,360 unique mismatch repair gene variants in the InSiGHT locus-specific database. *Nature Genetics* 46:107–115.
25. Jia X. et al. (2021). Massively parallel functional testing of MSH2 missense variants conferring Lynch syndrome risk. *American Journal of Human Genetics* 108:163–175.

*† Cited in the text but not yet verified against the primary source — confirm authors, year, venue, and any hyperparameters attributed to them before submission:* VariPred; ProPath (the Siamese wild-type/variant fine-tune, including the backbone learning rate 1 × 10⁻⁵ / batch 8 / 10 epochs attributed to it); the CSBJ per-residue token-classifier paper; MVmamba; and the CIMRA / PMS2 OddsPath source.

---

## 12. Pre-submission checklist

- [ ] Every number traced to the run log or a fresh run — no number carried over from a pre-fix run
- [ ] Table 4's caption states "residue-disjoint, not protein-disjoint"
- [ ] AlphaMissense ClinVar contamination stated in Results *and* Discussion, not only Limitations
- [ ] Seeds, split counts, and library versions in Methods
- [ ] "State-of-the-art" used only where a baseline was run on our split
- [ ] Every planned experiment in §7.2 either completed and moved into §4, or its claim removed
- [ ] ClinVar snapshot date pinned
- [ ] AlphaMissense non-commercial licence checked against the venue
- [ ] The five unverified references (†) confirmed against originals
- [ ] Abstract numbers match the final tables exactly
- [ ] The 5,466 / 5,152 clinical-count discrepancy (§3.2) resolved and each figure's source table stated
- [ ] Stage-2b Run 2 provenance (freeze depth, seed, memory flags) reconciled against its summary file
- [ ] Author names, roll numbers, supervisor, affiliation, and funding filled in
