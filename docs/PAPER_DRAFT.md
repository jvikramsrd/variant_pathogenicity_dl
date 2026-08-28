# Paper content pack — Variant Pathogenicity DL

*Written 2026-08-28. Everything below is grounded in this repository's own docs
(`PROJECT_DOCUMENTATION.md`, `TRAINING_NOTES.md`, `FINE_TUNING_FINDINGS.md`,
`RUN_REPORT.md`, `RUNLOG.md`, `MMR_TRANSFER_WORKFLOW.md`) and code. Numbers
quoted here are real past-run results from those files. Slots marked
**[PENDING]** are experiments that must be run before the claim can be made —
they are marked, not invented.*

---

## 0. Read this first: what paper you actually have

Be clear-eyed about which paper the current evidence supports, because it
determines every framing choice below.

**What you can defend today**

- A six-plus-source, leakage-audited missense variant table under one validated
  key, with SHA-256 provenance, explicit label precedence, quarantined
  cross-source conflicts, and an *independent* 12-check audit (12/12 pass).
- Four concrete, reproducible **circularity and silent-failure findings**, each
  with the diagnostic that caught it and the metric change it did (or
  conspicuously did *not*) produce. This is your strongest and most novel
  material.
- A leakage-clean supervised result on the clinical slice: **ROC-AUC 0.963 /
  MCC 0.78** vs AlphaMissense's **0.945 / 0.75** — priors-only, no ESM, no DMS
  features, 80 genes, 3-fold residue-group-disjoint CV.
- A domain-specific Lynch-syndrome (MMR) pipeline with safeguards a generic
  pipeline has no reason to have: pinned references with hard-fail length
  checks, a fail-closed PMS2 pseudogene gate, VCEP evidence tiering, and
  orthogonal functional evidence (MaveDB / CIMRA) withheld from training by
  construction.
- A complete, benchmarked implementation of four fine-tuning strategies and two
  backbones on identical splits.
- **A true unseen-gene result on the MMR panel** (new, 2026-08-28). Stage 2b —
  ESM-2 backbone gradient fine-tuning, siamese/ProPath recipe — ran to
  completion and produced leave-one-*gene*-out numbers: MLH1 **0.8993**, MSH2
  **0.8776**, MSH6 **0.9073** ROC-AUC, mean **0.895** over 662 held-out clinical
  rows. Read §6.10 before quoting these: `best_epoch` is 1–3 and the frozen
  ablation floor has not been run, so *that the fine-tune produced them* is not
  yet established.

**What you cannot defend today**

- That the *fine-tuning* is what produced the MMR numbers. Stage 2b now runs and
  reports (2026-08-28), but it early-stops at epoch 1–3 and the frozen
  `--n_unfrozen_layers 0` ablation floor has not been run. Until it is, the
  honest statement is "an ESM-2 siamese model trained under Stage 2b reaches
  0.895 mean unseen-gene AUC" — not "backbone fine-tuning beat the probe".
- Any unseen-gene claim **on the broad 80-gene panel**. The 0.963 is a
  **new-residue-in-a-known-protein** estimate (folds group by
  `uniprot_id:position`, not by protein). The new 0.895 is unseen-*gene* but
  only across three MMR genes with 119–335 rows each. Broad-panel LOPO and the
  MMseqs2 cluster-disjoint split are implemented but still not reported.
- The full-model `esm+priors` run on the 80-gene panel, still a TODO in
  `RUNLOG.md`.
- Any claim of beating AlphaMissense cleanly, because AlphaMissense was itself
  trained on ClinVar — the clinical slice is plausibly temporally contaminated.

### The recommended framing

> **Not:** "A new deep-learning model that beats AlphaMissense."
> **Instead:** "Circularity is the default outcome in multi-source missense
> variant-effect benchmarks — here is a leakage-audited pipeline, four
> reproducible failure modes it exposes, and what honest numbers look like
> afterwards."

That framing is (a) fully supported by evidence you already have, (b) genuinely
novel — the field cites Grimm et al. 2015 on circularity constantly but almost
nobody instruments for it, and (c) *strengthened* rather than weakened by the
0.716 all-labels number. You are reporting the drop from 0.9987 to 0.716 as a
finding, which is the whole point.

If/when the GPU runs land, the ESM and LOPO results slot into Section 6 as an
additional results subsection without changing the paper's spine.

### Venue options

| Venue | Fit | Framing to use |
|---|---|---|
| **Bioinformatics** / **Briefings in Bioinformatics** | Strong | Methods + benchmark; emphasise the audit and the pipeline |
| **Genome Biology** / **Genome Medicine** | Strong if MMR/Lynch is the spine | Clinical-translation angle: VUS burden, ACMG, VCEP tiers, PMS2 gate |
| **NeurIPS / ICLR Datasets & Benchmarks track** | Strong | Benchmark-integrity angle; ProteinGym is the direct comparator |
| **BMC Bioinformatics** / **PLOS Comp Biol** | Safe | Software + methods |
| **Human Mutation / Genetics in Medicine** | Only with LOPO + MMR results | Clinical classification angle |

Recommendation: **Bioinformatics or a Datasets & Benchmarks track.** Both accept
"the benchmark was broken and here is the instrumentation" as a primary
contribution.

---

## 1. Title

Pick one; the first is the recommendation.

1. **Circularity by Default: A Leakage-Audited Multi-Source Benchmark for
   Missense Variant Pathogenicity Prediction**
2. From 0.9987 to 0.716: Diagnosing Target Leakage, Label Inversion and In-Fold
   Calibration in Multi-Source Variant-Effect Datasets
3. A Provenance-Tracked, Audit-First Pipeline for Missense Pathogenicity
   Prediction, with Application to Lynch Syndrome
4. Honest Evaluation of Protein Language Model Features for Clinical Variant
   Interpretation

**Running title:** Leakage-audited missense variant benchmarking

---

## 2. Abstract (≈250 words — written out, edit freely)

> Computational prediction of missense variant pathogenicity increasingly relies
> on datasets assembled from heterogeneous public sources — clinical archives,
> deep mutational scanning (DMS) assays, published predictor scores and
> proteome-wide priors. We show that such assembly is a systematic source of
> circular evaluation, and that the resulting inflation is invisible to ordinary
> metric monitoring.
>
> We construct a unified table of 1,156,625 missense substitutions across 80
> human proteins, integrating ClinVar, ProteinGym v1.3 (DMS, clinical benchmark,
> and 17 published zero-shot scores), AlphaMissense, UniProt, gnomAD v4,
> AlphaFold DB and InterPro under a single validated key
> `(uniprot_id, position, wt_aa, mut_aa)`, with SHA-256 provenance for every
> artefact and an independent 12-check integrity audit. Wild-type residues are
> validated against canonical UniProt sequences before joining; 50,304
> isoform-mismatched rows are discarded rather than merged.
>
> Instrumenting this pipeline exposes four failure modes. (i) A DMS
> label-orientation inversion mislabelled ~185,000 variants while leaving
> cross-validated AUC *unchanged*. (ii) A DMS-derived feature equalled the
> inverted label by construction on 97.3% of labelled rows, producing a
> spurious ROC-AUC of 0.9987; the leakage-clean value is 0.716. (iii)
> Calibrators fitted on the folds they scored yielded an impossible isotonic
> expected calibration error of exactly zero. (iv) With 2.7% of rows clinically
> labelled, an unweighted objective optimises assay fitness rather than
> pathogenicity.
>
> After correction, a fused head over published priors attains ROC-AUC 0.963 and
> MCC 0.78 on the clinical slice, versus 0.945 and 0.75 for AlphaMissense alone.
> We release the pipeline, the audit, and a Lynch-syndrome (MLH1/MSH2/MSH6/PMS2)
> instantiation with a fail-closed PMS2 pseudogene gate and orthogonal
> functional evidence withheld from training.

**Keywords:** missense variant interpretation · variants of uncertain
significance · data leakage · circularity · protein language models · ClinVar ·
ProteinGym · AlphaMissense · Lynch syndrome · benchmark integrity

---

## 3. Introduction — content, paragraph by paragraph

Write five paragraphs. Content for each:

**¶1 — The clinical problem.**
A missense variant substitutes one amino acid. Clinical genetics must classify
it pathogenic or benign under ACMG/AMP guidelines (Richards et al. 2015). Most
observed missense variants receive neither call: they are **Variants of
Uncertain Significance (VUS)**, and the VUS backlog is the single largest
bottleneck in returning actionable results to patients. Quantify it: ClinVar
holds millions of submissions but only a few thousand ≥2-star missense
assertions per gene panel; this repository's own 80-gene build yields 5,466
**[RECOUNT — see Table 1]**
ClinVar-labelled rows against 25,000 ClinVar VUS. State the stakes concretely
for a hereditary cancer syndrome — a VUS in MLH1 leaves a family without
surveillance guidance.

**¶2 — Why the problem is computationally hard.** Three properties, stated as
the design constraints they are (this table is already in
`PROJECT_DOCUMENTATION.md` §1 and should be reused):

| Property | Consequence |
|---|---|
| Clinical labels are scarce and unevenly distributed | Supervision must be pooled across proteins and supplemented; single-gene deep models have too few labels |
| The plentiful labels are proxies | DMS measures *assay fitness*, not clinical consequence; published predictor scores were themselves trained on ClinVar |
| Leakage is the default outcome, not the exception | Same-residue variants across folds, paralogues, assay-derived features encoding the label, calibrators fitted on the fold they score |

**¶3 — What the field currently does, and where it breaks.** Protein language
models (ESM-1b/ESM-2, Meier et al. 2021; Lin et al. 2023) give strong zero-shot
scores from pseudo-log-likelihood ratios. Supervised models (AlphaMissense,
Cheng et al. 2023; EVE, Frazer et al. 2021) push further. ProteinGym (Notin et
al. 2023) standardised benchmarking. But the standard recipe — pool ClinVar +
DMS + published scores, cross-validate, report AUC — reintroduces exactly the
circularity Grimm et al. (2015) identified a decade ago, in a new form: **type-1
circularity** (variants from the same protein/residue across the split) and
**type-2 circularity** (the label and a feature share a source). Nobody
instruments for it; the metric looks fine.

**¶4 — Our position.** State it plainly: *a variant-effect pipeline should be
judged on the defensibility of its evidence as much as on its AUC.* We therefore
treat leakage control as a first-class subsystem rather than a preprocessing
footnote, and we report the corrections that changed our own numbers as primary
results.

**¶5 — Contributions.** Enumerate exactly:

1. A **leakage-audited multi-source variant table** (1,156,625 substitutions, 80
   proteins) with a single validated key, sequence-level wild-type validation,
   SHA-256 provenance, explicit label precedence, cross-source conflict
   quarantine, and a 12-check audit implemented *independently of the builder*.
2. **Four reproducible circularity/silent-failure findings**, each with the
   diagnostic that detected it — including one (label inversion) that left AUC
   unchanged and one (target leakage) that cost 0.28 AUC.
3. An **evaluation protocol** with residue-group-disjoint, protein-disjoint and
   MMseqs2 cluster-disjoint splits, nested calibration, clinical-only slices,
   10,000-iteration bootstrap CIs, and mandatory branch/fusion ablations on
   identical splits.
4. A **Lynch-syndrome instantiation** with pinned references, a fail-closed
   PMS2 pseudogene gate, InSiGHT/ClinGen VCEP tiering, and MaveDB/CIMRA
   functional evidence held out for validation only.
5. **Open, one-command reproducibility** across Linux/macOS/Windows with a
   closed-form VRAM preflight, 64 unit tests, and a complete run log including
   every build failure and its root cause.

---

## 4. Related work — with the comparison table

### 4.1 Narrative content

Organise into five threads. For each, say what it does, then what you do
differently — never just summarise.

**(a) Zero-shot protein language models.**
ESM-1b/ESM-2 (Rives et al. 2021; Lin et al. 2023) score variants by
pseudo-log-likelihood ratio, `PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)`
(Meier et al. 2021). No labels required, so no ClinVar circularity — but also no
clinical calibration. EVE (Frazer et al. 2021) and GEMME (Laine et al. 2019) use
MSA-based generative models instead. *Our difference:* we ingest all 17
ProteinGym zero-shot scores as features **and** retain them as baselines on
identical splits, and we do not assume ESM-2 beats ESM-1b — `compare_backbones.py`
tests it with true masked-marginal scoring.

**(b) Supervised proteome-wide predictors.** AlphaMissense (Cheng et al. 2023)
is the current general-purpose prior. *Our difference:* it is used as both a
feature and the baseline to beat, and — critically — we state explicitly that
because AlphaMissense was trained on ClinVar, our clinical-slice comparison is
plausibly temporally contaminated in AlphaMissense's favour. Most papers using
AlphaMissense as a baseline do not say this.

**(c) Fine-tuning recipes for PLM-based variant effect prediction.** Four
distinct recipes exist in the literature and are usually adopted rather than
compared: a frozen linear probe (VariPred), a Siamese WT/VT backbone fine-tune
(ProPath), a per-residue token classifier (CSBJ), and frozen global+local pooled
WT/VT features with gated fusion (MVmamba). *Our difference:* all four are
implemented in one codebase and benchmarked head-to-head on **identical MMR
splits** by `scripts/compare_finetune_strategies.py`, so the recipe choice is an
experimental result rather than an inherited assumption.

**(d) Benchmarks and circularity.** Grimm et al. (2015) established the two
circularity types. ProteinGym (Notin et al. 2023) standardised DMS-based
benchmarking. *Our difference:* we instrument for circularity inside the
pipeline — a target-leakage diagnostic, an independent audit, a nested
calibration protocol — and we report the resulting metric collapse (0.9987 →
0.716) as a finding rather than quietly fixing it.

**(e) Clinical variant curation for Lynch syndrome.** InSiGHT/ClinGen VCEPs
(Thompson et al. 2014 and successors) provide expert-panel MMR classifications.
Functional assays are calibrated to ACMG evidence strengths via OddsPath
(Brnich et al. 2020; Tavtigian et al. 2018). *Our difference:* VCEP tier is an
explicit weight; MaveDB (Esposito et al. 2019) and CIMRA evidence is withheld
from training by construction so that agreement with it is a genuine external
check; and PMS2 exon 11–15 pseudogene-region variants are gated fail-closed.

### 4.2 The comparison table — put this in the paper

| Method / resource | Supervision | Label sources | Split discipline reported | Circularity instrumentation | Calibration | Functional-assay use | What we do differently |
|---|---|---|---|---|---|---|---|
| **ESM-1v / ESM-2 PLLR** (Meier 2021; Lin 2023) | none (zero-shot) | — | n/a | n/a | none | — | Used as feature + baseline; PLLR from a single WT forward pass; ESM-1b vs ESM-2 tested, not assumed |
| **EVE** (Frazer 2021) | unsupervised (MSA) | — | n/a | n/a | GMM-based | — | Ingested as a `zs_*` prior and retained as a baseline on identical splits |
| **GEMME** (Laine 2019) | unsupervised (MSA) | — | n/a | n/a | none | — | Same |
| **AlphaMissense** (Cheng 2023) | weakly supervised | population + ClinVar-adjacent | proteome-level | none reported | none | — | Both feature and baseline; ClinVar contamination of our clinical slice stated explicitly |
| **ProteinGym v1.3** (Notin 2023) | benchmark suite | DMS + clinical | assay-level | partial (DMS/clinical separated) | none | DMS is the target | We separate DMS-fitness from clinical evaluation into distinct reported slices, and drop 50,304 isoform-mismatched DMS rows |
| **VariPred** | supervised probe | ClinVar-family | residue-level (typical) | none reported | none | — | Implemented as one of four benchmarked recipes, not the assumed one |
| **ProPath** (Siamese) | supervised fine-tune | ClinVar-family | residue-level (typical) | none reported | none | — | Implemented (`mode="siamese"`, backbone LR 1e-5, batch 8, 10 epochs) and benchmarked |
| **CSBJ token classifier** | supervised fine-tune | ClinVar-family | residue-level (typical) | none reported | none | — | Implemented (`mode="wt_site"`) and benchmarked |
| **MVmamba** | supervised fusion | multi-source | residue-level | none reported | none | — | Frozen WT/VT feature recipe + GateWave gated fusion reimplemented and ablated vs plain concat |
| **This work** | supervised, provenance-weighted | ClinVar (≥2★) > PG-clinical > DMS, with conflict quarantine | **residue-disjoint + protein-disjoint + MMseqs2 cluster-disjoint** | **12-check independent audit + target-leakage diagnostic** | **nested** temperature + isotonic | **withheld from training; validation only** | — |

*(Verify the four fine-tuning-recipe citations against the originals before
submission — see §12. The rest are standard and safe.)*

---

## 5. Methods — the content, in the order it should be written

### 5.1 Problem formulation

Given a protein sequence `X = x_1 … x_L` and a substitution `(i, wt → mut)` with
`x_i = wt`, predict `P(pathogenic)`. Define the clinical target explicitly as
the ACMG/AMP P/LP vs B/LB dichotomy, and state up front that DMS assay fitness
is a **proxy**, deliberately kept as a separate evaluation slice.

### 5.2 Data sources and integration

State the sources and the reason for each (reuse the `PROJECT_DOCUMENTATION.md`
§3.1 table). Then the four integration rules — these are the methodological
substance:

1. **One key.** Every source is normalised to
   `(uniprot_id, position, wt_aa, mut_aa)`.
2. **Sequence-level validation before joining.** Rows whose wild-type residue
   disagrees with the canonical UniProt sequence are **dropped, not merged**
   (50,304 DMS + 152 AlphaMissense rows on the 80-gene build). RefSeq `NP_`
   accessions are matched by **exact whole-protein sequence equality**, never by
   identifier string. Justify: silent isoform/numbering mismatch is the most
   common way a multi-source variant table becomes quietly wrong; a dropped row
   is recoverable, a wrongly-joined row is invisible.
3. **Label precedence and weighting.** ClinVar (≥2 review stars by default) >
   ProteinGym-clinical > single-assay DMS bin. Every label carries
   `label_source` and `label_weight` (ClinVar weighted by stars, PG-clinical
   0.75, DMS 0.20).
4. **Conflict quarantine.** Cross-source disagreement leaves `label` blank and
   `label_conflict=1`; raw evidence is retained but the row cannot enter
   training. Justify: heterogeneous public evidence cannot establish clinical
   truth; making disagreement explicit and unusable is more honest than picking
   a winner.

Also document: SHA-256 provenance manifest with URL/version/bytes/date; atomic
`.part` staging so a server ignoring HTTP Range cannot corrupt a resumed
download; ZIP integrity checks before parsing. Sources **rejected** and why
(dbNSFP: ~30 GB and redundant with the `zs_*` columns; EVE repo: ships MSAs
only) — recording rejections is what stops the same evaluation being redone.

**Table 1 — dataset composition.**

| | 10-gene build | 80-gene build |
|---|---|---|
| Proteins | 10 | 80 (every human protein with a ProteinGym v1.3 DMS assay) |
| Master table rows | 110,124 | **1,156,625** unique substitutions |
| Labelled rows | 14,131 | **190,494** (82,149 P/LP · 108,345 B/LB) |
| ClinVar-labelled | 2,067 | 5,466 (+ ~25,000 VUS) **[RECOUNT]** |
| Clinical-slice rows | — | 5,152 (**2.7%** of labelled) |
| AlphaMissense coverage | 10 proteomes | 80 proteomes (~1.16M scored) |
| Rows dropped by wt-validation | — | 50,304 DMS + 152 AM |
| ClinVar ∩ PG-clinical | — | 1,919, agreement **100%** |
| Audit | — | **12/12 pass** |

> **[RECOUNT] — unresolved before submission.** The 5,466 ClinVar-labelled count
> (`TRAINING_NOTES.md`) and the 5,152 clinical-slice count
> (`FINE_TUNING_FINDINGS.md`) cannot both describe the same table: the clinical
> slice is defined in `scripts/train_extended.py:227` as
> `clinvar_label.notna() | clinical_label.notna()` — a **union** that must be
> ≥ the ClinVar-only count. The likely explanation is that 5,466 is measured on
> the 1,156,625-row master table and 5,152 on the 190,494-row train CSV after
> label precedence and conflict quarantine, but that is not established. Recount
> both on the same table from the fresh build and state which table each is
> measured on. The same pair appears in §3 ¶1 and §6.4.

### 5.3 The independent audit

`scripts/audit_extended_dataset.py` re-derives the merge from scratch as a
**separate implementation** and runs 12 checks: key uniqueness, wt-residue
validation, label-precedence re-derivation, provenance-token coherence,
cross-source conflicts, feature coverage. Make the methodological point
explicitly: *a builder that validates itself only proves it is self-consistent.*

### 5.4 Feature construction

For a variant `(i, wt → mut)` in sequence `X`:

```
z = [ h_wt ‖ h_mut ‖ (h_mut − h_wt) ‖ |h_mut − h_wt| ‖ PLLR_i ]
PLLR_i = log P(x_mut | X_\i) − log P(x_wt | X_\i)
```

`h_wt`, `h_mut` are last-layer ESM-2 hidden states at residue `i` for the
wild-type and in-silico-mutated sequences. Note the efficiency point: both
log-probabilities are conditioned on the *same* masked context, so both come
from a **single forward pass** over the wild-type sequence. Sequences beyond the
1,022-residue positional capacity use overlapping sliding windows — hidden
states averaged across covering windows, log-probabilities averaged in log
space. fp16 autocast on CUDA (≈2–3× throughput); all features cached.

~22 external prior columns are optionally appended, each with an
`is_missing_*` indicator. The leakage-safe prior set is defined once, in
`src.transfer.TRANSFER_PRIOR_COLS`; DMS-derived and functional-assay
validation-only columns are deliberately absent, and a regression test fails if
a newly joined source is not wired through.

### 5.5 Models

| Component | Description |
|---|---|
| Residual MLP head | `Linear(d→h)` → N × (`Linear → LayerNorm → GELU → Dropout` + residual) → `Linear(h→1)`; 256 hidden, dropout 0.15 |
| Branch / concat / GateWave fusion | Frozen-branch projections to a shared dim; GateWave = sigmoid branch gate + softmax per-feature gating + GLU + residual (adapted from MVmamba) |
| ESM-2 fine-tune, Siamese | ProPath recipe: separate WT/VT forward passes, pool at the mutated residue, head over `[h_wt ‖ h_vt ‖ Δ ‖ |Δ|]` |
| ESM-2 fine-tune, token classifier | CSBJ recipe: per-residue classification at the mutated site |
| MVmamba features | Global + local pooled WT/VT with a ±3-residue window, mutation-centred for long chains (e.g. MSH6, 1,360 aa) |

`n_unfrozen_layers`: −1 = full fine-tune, 0 = frozen (ablation floor), N = last N
layers. Gradient checkpointing available.

### 5.6 Optimisation

BCE by default (not focal — see Finding 4). Weighted BCE uses a **fixed** class
ratio from the fitting partition, not recomputed per mini-batch. Focal loss
retained for comparison with neutral `alpha=0.5`. AdamW + cosine annealing,
early stopping with best-weight restoration, patience 10, max 60 epochs, LR
3e-4. Clinical rows carry `--clinical_weight` (default 5.0); early stopping is
monitored on **clinical-only** ROC-AUC whenever both classes are present.

For Stage 2b: backbone LR 1e-5, batch 8, 10 epochs, AMP (bf16 where supported,
else fp16 + GradScaler), gradient accumulation, gradient checkpointing.

### 5.7 The evaluation protocol — this is a contribution, write it as one

**Table 2 — leakage controls.**

| Control | Implementation | What it prevents |
|---|---|---|
| Residue-group disjointness | `StratifiedGroupKFold` on `"{uniprot_id}:{position}"`, asserted at runtime | Different substitutions at one residue on both sides of a split |
| Protein-level holdout | Leave-one-protein-out; leave-one-gene-out for MMR | A *new-residue-in-known-protein* number reported as a *new-gene* number |
| Sequence-cluster disjointness | MMseqs2, 20% identity / 20% coverage; whole clusters assigned to a side | Paralogues and near-duplicate isoforms straddling a split |
| Nested calibration | Outer validation fold untouched; outer-training group-split into fit / early-stop / calibration partitions | Calibrators fitted on the fold they score |
| Clinical-only slices | Metrics reported separately for all labels and for ClinVar ∪ PG-clinical | DMS assay-fitness performance quoted as clinical performance |
| Target-leakage diagnostic | `scripts/diagnose_overfitting.py` | Features that encode the label by construction |
| Bootstrap CIs + tuned thresholds | 10,000-iteration percentile CIs; MCC-optimal threshold tuned on an inner slice only | Point estimates on a few-thousand-row set read as precise |

**10,000 is the pipeline default and is honoured**: `run_mmr_pipeline.py` sets `--n_bootstrap 10000` and passes it to stages 2 and 2b, so the §6.10 numbers already carry 10,000-iteration CIs. The *standalone* scripts do not — `eval_leave_one_protein_out.py:87` defaults to 2,000 — so pass `--n_bootstrap 10000` explicitly whenever you run LOPO or the comparison scripts outside the pipeline.
| Mandatory ablations | ESM-only, priors-only, pretrained-fused, concat, GateWave — identical splits | A fusion architecture credited for a single branch's gain |
| Schema-drift abort | Checkpoints store prior-column order + ESM block dim; mismatch aborts | A transfer run silently invalidating itself |
| Persisted imputation constants | Imputation medians stored in the checkpoint and reused across stages | Warm-starting a head onto differently-centred inputs |

Metrics: ROC-AUC, PR-AUC, MCC (at an MCC-optimal threshold tuned on an inner
slice), Brier, ECE, MCE, plus reliability diagrams.

### 5.8 The Lynch-syndrome instantiation

Five safeguards, each with its justification:

1. **Pinned canonical references with hard-fail length checks** — MLH1 P40692 /
   756 aa, MSH2 P43246 / 934 aa, MSH6 P52701 / 1,360 aa, PMS2 P54278 / 862 aa.
   A silent reference update cannot shift numbering underneath a build.
2. **Fail-closed PMS2 exon 11–15 pseudogene gate.** Short-read NGS calls inside
   the PMS2CL homology region are untrustworthy. The build **refuses to run**
   without one of: a homology CSV with `orthogonally_confirmed` flags, an
   explicitly verified `--pms2_codon_range`, or `--exclude_pms2`. Unconfirmed
   homology-region labels are withheld, never guessed. State the general
   principle here — *no supervision is better than supervision you cannot
   defend* — it is the paper's ethos in one line.
3. **InSiGHT / ClinGen VCEP tiering** (`expert_panel`, `evidence_tier`,
   `tier_weight`).
4. **Orthogonal functional evidence held out by construction.** MaveDB (MSH2
   Jia et al. 2021 LOF screen, `urn:mavedb:00000050-a-1`, 17,746 variants; MLH1
   2025 cellular-abundance assay, `urn:mavedb:00001218-a-1` — a *different*
   evidence class) and CIMRA OddsPath live in
   `FUNCTIONAL_ASSAY_VALIDATION_ONLY_COLS`, never in `TRANSFER_PRIOR_COLS`.
   CIMRA rows with a hypothesised splicing mechanism are excluded automatically
   (CIMRA is cell-free and cannot detect splicing).
5. **Leave-one-MMR-gene-out evaluation** with bootstrap CIs, because the
   clinically relevant question for a rare-disease gene is transfer to an unseen
   gene, not interpolation within a known one.

Two-stage transfer: Stage 1 pretrains the head on frozen ESM-2 embeddings over
all 80 panel genes (`--mode leave_gene_out` excludes every MMR gene for a true
transfer estimate; `--mode practical` allows them, with no unseen-gene claim).
Stage 2 warm-starts on MMR clinical labels only. Stage 2b is the true backbone
gradient fine-tune.

### 5.9 Compute and reproducibility

Report honestly. Build machine: Linux, 12 cores, 14 GB RAM, no GPU; Python
3.14.7; torch 2.13.0**+cpu**, transformers 5.15.1, pandas 3.0.5, numpy 2.5.2,
scikit-learn 1.9.0, scipy 1.18.1. State the `+cpu` build explicitly: the
Stage-2b numbers in §6.10 came from a *different* machine (Windows, CUDA), and a
reproducibility section that conflates the two is wrong.
Downloads ≈1.27 GB. AlphaMissense streaming filter scanned 216,175,352 lines
(~7 min, cached thereafter). 64 unit tests.

Include the **VRAM budget analysis** — it is a genuinely useful practical
contribution and reviewers of reproducibility-minded venues like it: the default
Stage-2b config (650M checkpoint, Siamese, full unfreeze, micro-batch 8,
1,022-residue crops) requires **≈51 GiB in fp32** — ~10 GiB weights/gradients/
AdamW state plus ~41 GiB stored activations, with the Siamese design doubling
the activation term. A closed-form preflight (`estimate_finetune_vram_gib`)
refuses an impossible configuration *before* the multi-hour download stages
rather than OOM-ing in the backward pass at the end. On a ~15 GiB card the full
Siamese fine-tune fits at batch 1 / grad-accum 8 / gradient checkpointing
(~10 GiB, effective batch still 8); below ~11 GiB a full fine-tune is impossible
because the static AdamW state alone is ~9.7 GiB.

---

## 6. Results — content and the exact tables

### 6.1 Finding 1 — Label inversion: ~185,000 labels were backwards, and AUC did not move

ProteinGym's `DMS_score_bin = 1` marks the *top half of assay fitness*
(**tolerated**); the builder mapped it directly to `label = 1` ("pathogenic"),
inverting ~185,000 DMS-derived labels.

Two independent anchors confirmed the inversion:

| Anchor | Evidence |
|---|---|
| AlphaMissense | `P(AM ≥ 0.9 \| bin=1) = 0.217` vs `P(AM ≥ 0.9 \| bin=0) = 0.506` → bin=1 is benign |
| ProteinGym clinical | mean DMS score **−0.8** for pathogenic vs **+1.9** for benign |

Fixed (`label = 1 − dms_bin`), mirrored in the auditor, dataset rebuilt,
re-audited 12/12.

**The finding worth stating in the abstract:** cross-validated AUC was
**unchanged**, because AUC is flip-invariant once the head re-learns the mapping.
Only the *semantics* and any prospective risk ranking were wrong. This is
precisely why this bug class survives ordinary metric-watching, and it argues
for external semantic anchors as a standard integrity check.

### 6.2 Finding 2 — Target leakage: ROC-AUC 0.9987 → 0.716

The first run reported ROC-AUC **0.9987** on all labels.
`scripts/diagnose_overfitting.py` established:

1. On single-assay rows — **97.3% of labelled data** — the feature
   `dms_bin_median` equals `1 − label` **by construction**. The answer was in
   the feature matrix.
2. With leaky features removed, the honest all-labels number is **0.716**.
3. **There was no classic overfitting.** Per-fold train-vs-val AUC gap ≈ 0.000
   (range −0.001 … +0.011); folds early-stop within 2–16 epochs. Make this
   point explicitly: the pathology was *target leakage*, not capacity — and
   standard overfitting diagnostics would have found nothing.
4. `-DMS_score` as a baseline is likewise circular on the DMS slice, since the
   label is a median split of that same score; only its clinical-slice value is
   interpretable.

### 6.3 Finding 3 — In-fold calibration

Temperature and isotonic calibrators were fitted on each outer validation fold
and then scored on that fold. The tell was an isotonic ECE of **exactly zero** —
an impossible out-of-sample value. Calibration is now nested: the outer
validation fold is untouched and both calibrators are fitted on a dedicated
calibration partition carved by residue group from the outer-training data. This
costs fitting data per fold and buys valid out-of-fold calibration, Brier, ECE
and threshold metrics.

### 6.4 Finding 4 — The objective was optimising the wrong target

Only **5,152 of 190,494** labelled rows (**2.7%**) carry a clinical label; ~188k
are DMS aggregates. An unweighted loss therefore optimises assay fitness, not
pathogenicity. Two corrections: a configurable `--clinical_weight` (default 5.0)
and early stopping monitored on clinical-only ROC-AUC. Separately, the prior
default `focal_alpha = 0.25` was inappropriate — positives are 43% of all labels
and ~52% of clinical labels — so it was down-weighting the positive class;
neutral 0.5, with plain BCE as the default.

### 6.5 Finding 5 — Silent-failure classes found by reading code, not by watching metrics

Group these; the framing ("none of them would have shown up as a bad number") is
the point.

| Failure | Mechanism | Fix |
|---|---|---|
| Vacuous mutant-residue check | `np.char.str_len(np.asarray(df["mut_aa"], dtype="U1")) == 1` truncates every value to ≤1 char, so the check always passed | Explicit per-row `len(m)==1 and m in VALID_AA` |
| Feature cache validated only row count | A rebuilt table with the same row count in a different order returned another variant's embeddings — no error, no metric change | Cache hits validated by variant key |
| `esm+priors` appended zero priors | The extraction pool retained only metadata columns, so the advertised priors never reached the model | Pool carries raw prior columns; cache width change forces a new cache key |
| gnomAD features joined but invisible | AF columns merged onto the table but never added to `TRANSFER_PRIOR_COLS` | Wired through + a regression test that fails if a new source is joined without wiring |
| Missingness indistinguishable from prediction | `zs_*` columns absent for ~99% of rows; median imputation made "not scored" look like a real median prediction | `is_missing_*` indicator for every numerical prior |
| Imputation constants recomputed per stage | Stage 1 used 80-gene medians, Stage 2 used 4-gene medians; since the imputed constant *is* the feature for ~99% of rows, warm-starting carried weights onto differently-centred inputs | Constants persisted in the checkpoint and reused |
| Cross-protein group collapse | Grouping key was raw `position`, collapsing equal positions of *different* proteins into one leakage group when pooling | Group key is `"{uniprot_id}:{position}"` |
| Per-batch `pos_weight` | Recomputed per mini-batch → noisy gradients on small final batches | Fixed ratio from the fitting partition |
| Half-built virtualenv counted as ready | `venv` creation finishes in a second; the dependency install takes minutes, so interrupted installs were silently adopted | Readiness gated on a post-install stamp file |

### 6.6 Table 3 — Leakage-clean performance (priors mode, 80 genes, 3-fold residue-group-disjoint CV)

| Slice | Model | ROC-AUC | MCC |
|---|---|---|---|
| All labels (190,494) | MLP + isotonic | 0.716 | 0.32 |
| All labels | AlphaMissense | 0.709 | 0.30 |
| **Clinical only (5,152)** | **MLP + isotonic** | **0.963** | **0.78** |
| Clinical only | AlphaMissense | 0.945 | 0.75 |

How to write the reading of this table — do not skip any of these four points:

- **0.716 is the honest all-labels number.** The task "predict assay fitness
  bins without seeing the assay" is intrinsically hard; the previous 0.9987 was
  circularity, not learning.
- **The clinically meaningful headline is 0.963 / 0.78**, a **+1.8 pt AUC** gain
  over AlphaMissense from combining AM with 17 published zero-shot scores and
  domain flags — **without DMS features and without ESM**.
- **Calibration:** temperature scaling moved ECE 23% → 22%; isotonic → 11%.
  Fixed-threshold metrics (MCC@0.5) are sensitive to score shift, so trust AUC
  and the calibrated variants.
- **Scope caveat, stated in the caption itself:** this is a
  *new-residue-in-a-known-protein* estimate. Folds group by
  `uniprot_id:position`, not by protein. It is **not** an unseen-gene number.

### 6.7 [PENDING] Table 4 — protein-disjoint evaluation

Run `scripts/eval_leave_one_protein_out.py` and report ROC-AUC / PR-AUC / MCC
with 10,000-iteration bootstrap CIs per held-out protein (pass
`--n_bootstrap 10000`; the default is 2,000), plus the pooled
estimate. **You should expect a substantial drop from 0.963**, and reporting
that drop honestly is a stronger result than hiding it. This is the single most
important missing experiment — the paper is materially weaker without it.

Also report the MMseqs2 cluster-disjoint split (20% id / 20% cov) as a third
rung on the difficulty ladder. The three-rung table — residue-disjoint,
protein-disjoint, cluster-disjoint — is itself a nice figure.

### 6.8 [PENDING] Table 5 — ESM feature contribution

`train_extended.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D
--no_dms_features`. Report priors-only vs ESM-only vs ESM+priors on identical
splits. The hypothesis to state: the biggest gain should appear on the clinical
slice, because PLLR/embeddings add protein-context information no prior score
contains.

### 6.9 [PENDING] Table 6 — fine-tuning strategy benchmark

`scripts/compare_finetune_strategies.py --holdout_gene MSH2` — all four recipes
(MVmamba frozen pooled / VariPred linear probe / ProPath Siamese / CSBJ token
classifier) on one split. Report the frozen (`n_unfrozen_layers=0`) row as the
ablation floor. Plus `compare_backbones.py` for ESM-1b vs ESM-2 masked-marginal
zero-shot — and report it whichever way it comes out.

### 6.10 Table 7 — MMR leave-one-gene-out, ESM-2 siamese fine-tune

**Landed 2026-08-28** — the first Stage-2b run to complete. Leave-one-*gene*-out
over the MMR panel, so these are genuine unseen-gene estimates, unlike the 0.963
in §6.6.

| Held-out gene | Mode | ROC-AUC | 95% CI | PR-AUC | MCC | Threshold | n | Best epoch |
|---|---|---|---|---|---|---|---|---|
| MLH1 | siamese | **0.8993** | 0.8446–0.9445 | 0.9757 | 0.5275 | 0.4344 | 208 | 1 |
| MSH2 | siamese | **0.8776** | 0.8350–0.9162 | 0.7931 | 0.2972 | 0.0567 | 335 | 3 |
| MSH6 | siamese | **0.9073** | 0.8508–0.9536 | 0.9195 | 0.6519 | 0.2674 | 119 | 1 |

Mean ROC-AUC **0.895** (n-weighted 0.890) over 662 held-out clinical rows. PMS2
is absent because the fail-closed pseudogene gate excludes it by default — say
so in the caption rather than letting a reader wonder.

**Four things must be written alongside this table. Each is a reviewer question
you cannot afford to leave open:**

1. **The ablation floor is missing.** `best_epoch` is 1, 3, 1 — the run
   early-stops almost immediately. Nothing here yet separates "backbone
   fine-tuning works" from "a frozen probe on ESM-2 embeddings works and the
   backbone gradients were irrelevant". Rerun with `--n_unfrozen_layers 0` and
   report it as the floor. **Until that exists, do not attribute the result to
   fine-tuning.**
2. **Thresholds do not transfer.** The MCC-optimal threshold ranges 0.0567
   (MSH2) to 0.4344 (MLH1) — an eightfold spread across three genes. MSH2 gets
   MCC 0.2972 despite ROC-AUC 0.8776, which is the clearest symptom. Any
   clinical deployment needs a per-gene threshold or per-gene recalibration.
   This is a finding in its own right and belongs in the Discussion.
3. **Per-gene base rates differ sharply.** PR-AUC moves opposite to ROC-AUC
   between MLH1 (0.9757 vs 0.8993) and MSH2 (0.7931 vs 0.8776). Report class
   prevalence per gene in the caption or the table is misreadable.
4. **Single seed.** Report mean ± sd over ≥3 seeds before submission. Holdouts
   are small (119–335) and the three CIs overlap heavily, so no gene is
   significantly different from another — say that explicitly rather than
   ranking the genes. The CIs themselves are fine: 10,000 iterations, since
   `run_mmr_pipeline.py` defaults `--n_bootstrap` to 10,000 and passes it to
   stage 2b.

**Still owed for this subsection:** the five mandated fusion ablations (ESM-only,
priors-only, pretrained-fused, concat, GateWave) via
`scripts/run_mmr_transfer.py --eval lopo`. The ablation battery is what licenses
any claim that fusion, rather than one branch, produced the gain.

### 6.11 [PENDING] External validation against withheld functional evidence

Correlate model scores against MaveDB MSH2 LOF scores and the MLH1 abundance
assay — never used in training. This is the closest thing to an unbiased
external check the project has, and it is worth its own subsection.

---

## 7. Discussion — content

**¶1 — What the corrections mean generally.** Three of the five findings changed
what the model was learning; one changed nothing measurable and everything
semantic. Generalise: in multi-source variant-effect datasets, the dangerous
errors are *well-formed but wrong values in places no metric looks*. Argue that
this motivates audit-first construction — an integrity audit implemented
independently of the builder, external semantic anchors for label orientation,
and target-leakage diagnostics as routine.

**¶2 — What the numbers do and do not license.** 0.963 on the clinical slice is
a real, leakage-clean, residue-disjoint result and it beats AlphaMissense by 1.8
points. It is *not* an unseen-gene result, and AlphaMissense's own ClinVar
training makes even that comparison contaminated in AlphaMissense's favour. Say
this in the discussion, not only in the limitations — reviewers reward it.

**¶3 — Why 0.716 is the interesting number.** The all-labels task asks for assay
fitness without seeing the assay. That it is hard is informative: DMS bins and
clinical labels are genuinely different targets, which is the empirical
justification for the clinical-weighted objective and the separate slices.

**¶4 — Clinical translation.** Where this fits ACMG/AMP: calibrated scores map
to evidence strengths via the Tavtigian/Brnich framework; VCEP tiers give a
gold-standard subset; the PMS2 gate reflects the reality that a sequencing
technology's blind spot must propagate into the model's refusal to opine.
Emphasise VUS triage — once ESM features are cached, all ~966,000 unlabelled
substitutions (including 25,000 ClinVar VUS) can be scored into calibrated risk
tiers.

**¶5 — A recommendation to the field.** Propose concretely: benchmark papers
should report the split rung (residue / protein / cluster), state whether any
baseline was trained on the evaluation labels, and publish a target-leakage
diagnostic. Offer this pipeline's audit as a template.

---

## 8. Limitations — write these; do not let a reviewer find them first

1. **No unseen-gene estimate on the broad panel.** The 80-gene headline numbers
   are residue-disjoint within known proteins. An unseen-gene estimate now
   exists for MMR (§6.10, mean 0.895) but covers three genes and 662 rows.
   [Remove once §6.7 lands.]
2. **AlphaMissense contamination.** AlphaMissense was trained on ClinVar; the
   clinical slice may be temporally contaminated, biasing the baseline upward —
   which makes our +1.8 pt a conservative gain, but an uncontrolled one.
   Mitigation: a temporal ClinVar holdout (variants first submitted after the
   AlphaMissense cutoff).
3. **DMS binarisation is a proxy target.** The recommended fix — multitask
   learning with within-assay-normalised continuous fitness as one task and
   clinical pathogenicity as another — needs a new head and data interface and
   is not implemented.
4. **Sparse priors.** Most `zs_*` columns are absent for ~99% of rows; the
   imputed constant *is* the feature for most variants. Missingness indicators
   mitigate but do not solve this.
5. **The fine-tuning is unattributed.** Stage 2b now reports (§6.10), but with
   `best_epoch` 1–3 and no frozen-backbone ablation floor, the gain cannot be
   attributed to backbone gradients rather than to the ESM-2 representation
   itself. The 650M Siamese config needs ~51 GiB fp32 (~10 GiB with AMP +
   accumulation + checkpointing). §6.8's broad-panel `esm+priors` run is still
   owed. [Rewrite once the ablation floor and §6.8 land.]
6. **PMS2 is excluded by default**, so the MMR result is effectively three
   genes unless a verified homology span is supplied.
7. **Single-run results.** Numbers are from single runs, not seed-averaged.
   Report mean ± sd over ≥3 seeds before submission — cheap and expected.
8. **MMR functional-assay coverage is uneven.** MaveDB has MSH2 and MLH1 score
   sets; MSH6 and PMS2 have none, so external validation is partial.
9. **Provenance-weighted labels are heuristic.** Weights (stars, 0.75, 0.20)
   are reasoned but not tuned or validated.

---

## 9. Figures and tables — the shot list

| # | Type | Content | Status |
|---|---|---|---|
| **Fig 1** | Schematic | Pipeline: sources → validated key + provenance → audit → features → grouped CV → nested calibration → outputs | Draw from the `README.md` architecture block |
| **Fig 2** | Bar / annotated | **The circularity waterfall**: 0.9987 (leaky) → 0.716 (clean, all labels) → 0.963 (clinical slice), with the cause annotated on each step | **Make this the headline figure** |
| **Fig 3** | Two panels | Label-inversion evidence: AM ≥ 0.9 rate by DMS bin; DMS score distribution by PG-clinical label | Data in hand |
| **Fig 4** | Scatter / line | Train vs val AUC per fold, showing gap ≈ 0.000 — "no classic overfitting" | Data in hand |
| **Fig 5** | Reliability diagram | Uncalibrated vs temperature vs isotonic, nested; ECE annotated (23% → 22% → 11%) | `ext80_reliability_diagram.png` exists |
| **Fig 6** | Grouped bars | The three-rung split ladder: residue-disjoint / protein-disjoint / cluster-disjoint | **[PENDING]** |
| **Fig 7** | Forest plot | MMR leave-one-gene-out ROC-AUC with bootstrap CIs, per gene × ablation | Siamese row **has data** (§6.10); ablation rows **[PENDING]** |
| **Table 1** | Dataset composition | §5.2 | Ready |
| **Table 2** | Leakage controls | §5.7 | Ready |
| **Table 3** | Leakage-clean performance | §6.6 | Ready |
| **Table 4** | Method comparison | §4.2 | Ready |
| **Table 7** | MMR leave-one-gene-out, siamese | §6.10 | **Landed 2026-08-28** (ablation floor still owed) |
| **Tables 5–6** | ESM contribution / recipe benchmark | §6.8–6.9 | **[PENDING]** |
| **Supp** | Audit report, provenance manifest, build-failure table (12 failures + root causes), VRAM budget derivation | `RUN_REPORT.md` §5 | Ready |

---

## 10. Experiments to run before submission — priority order

1. **Leave-one-protein-out CV** (`eval_leave_one_protein_out.py`) — the single
   most important gap. Without it the paper has no unseen-gene claim.
2. **Seed averaging** (≥3 seeds) on every reported configuration. Cheap, and its
   absence is a common desk-reject.
3. **MMseqs2 cluster-disjoint split** (`build_cluster_split.py`) — completes the
   three-rung ladder and Fig 6.
4. **ESM+priors run** (`--features esm+priors --no_dms_features`, 650M) —
   supports the "PLM features add clinical signal" claim.
5. **Temporal ClinVar holdout** — the only way to defuse the AlphaMissense
   contamination objection.
6. **Fine-tuning strategy benchmark** (`compare_finetune_strategies.py`) and
   **backbone comparison** (`compare_backbones.py`).
7. **MMR LOPO + fusion ablation battery** (`run_mmr_transfer.py --eval lopo`).
8. **MaveDB/CIMRA external validation** correlation.
9. **Ablation grid**: AM-only / all priors / priors minus each source family /
   ESM-only / ESM+priors, with feature availability reported alongside
   performance.

Items 1–3 are mandatory. Items 4–7 determine whether this is a benchmark paper
or a benchmark-plus-model paper.

---

## 11. Data and code availability statement (draft)

> All code is available at `<repo URL>` under `<licence>`. The pipeline is
> reproducible with a single command on Linux, macOS and Windows
> (`run_mmr_pipeline.py`). Every remote artefact is recorded with URL, version,
> byte count, retrieval date and SHA-256 checksum in
> `data/processed/extended/manifest.json`. Source data are public: ClinVar
> (NCBI), ProteinGym v1.3 (Marks lab), AlphaMissense (Google DeepMind, CC BY-NC-SA
> 4.0 — **check whether this restricts your venue's licence terms**), UniProt (CC
> BY 4.0), gnomAD v4, AlphaFold DB (CC BY 4.0), InterPro, MaveDB. CIMRA OddsPath
> values are extracted from published supplementary tables and are not
> redistributed. Per-source licences are documented in `docs/DATASETS.md`.

Two things to check before submission: (a) AlphaMissense's **non-commercial**
licence and what it permits in a published derivative; (b) whether ClinVar
snapshot date must be pinned in the paper — it should be.

---

## 12. Reference list — verify the starred ones

Safe to cite as-is:

- Richards S. et al. (2015). *Standards and guidelines for the interpretation of
  sequence variants.* Genetics in Medicine 17:405–424.
- Tavtigian S.V. et al. (2018). *Modeling the ACMG/AMP variant classification
  guidelines as a Bayesian classification framework.* Genetics in Medicine.
- Brnich S.E. et al. (2020). *Recommendations for application of the functional
  evidence PS3/BS3 criterion.* Genome Medicine 12:3.
- Grimm D.G. et al. (2015). *The evaluation of tools used to predict the impact
  of missense variants is hindered by two types of circularity.* Human Mutation
  36:513–523. **← the key citation for your framing; cite it early and often.**
- Landrum M.J. et al. (2018). *ClinVar: improving access to variant
  interpretations.* Nucleic Acids Research 46:D1062.
- Rives A. et al. (2021). *Biological structure and function emerge from scaling
  unsupervised learning to 250 million protein sequences.* PNAS 118:e2016239118.
- Meier J. et al. (2021). *Language models enable zero-shot prediction of the
  effects of mutations on protein function.* NeurIPS. **← the PLLR /
  masked-marginal method.**
- Lin Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein
  structure with a language model.* Science 379:1123–1130. **← ESM-2.**
- Cheng J. et al. (2023). *Accurate proteome-wide missense variant effect
  prediction with AlphaMissense.* Science 381:eadg7492.
- Frazer J. et al. (2021). *Disease variant prediction with deep generative
  models of evolutionary data.* Nature 599:91–95. **← EVE.**
- Laine E., Karami Y., Carbone A. (2019). *GEMME: a simple and fast global
  epistatic model predicting mutational effects.* Molecular Biology and
  Evolution 36:2604–2619.
- Brandes N. et al. (2023). *Genome-wide prediction of disease variant effects
  with a deep protein language model.* Nature Genetics 55:1512–1522.
- Notin P. et al. (2023). *ProteinGym: large-scale benchmarks for protein
  fitness prediction and design.* NeurIPS Datasets and Benchmarks.
- Chen S. et al. (2024). *A genomic mutational constraint map using variation in
  76,156 human genomes.* Nature. **← gnomAD v4.**
- Karczewski K.J. et al. (2020). *The mutational constraint spectrum quantified
  from variation in 141,456 humans.* Nature 581:434–443.
- Jumper J. et al. (2021). *Highly accurate protein structure prediction with
  AlphaFold.* Nature 596:583–589.
- Varadi M. et al. (2022). *AlphaFold Protein Structure Database.* Nucleic Acids
  Research 50:D439.
- Esposito D. et al. (2019). *MaveDB: an open-source platform to distribute and
  interpret data from multiplexed assays of variant effect.* Genome Biology
  20:223.
- Steinegger M., Söding J. (2017). *MMseqs2 enables sensitive protein sequence
  searching for the analysis of massive data sets.* Nature Biotechnology
  35:1026–1028.
- Guo C. et al. (2017). *On calibration of modern neural networks.* ICML. **←
  temperature scaling.**
- Zadrozny B., Elkan C. (2002). *Transforming classifier scores into accurate
  multiclass probability estimates.* KDD. **← isotonic regression.**
- Lin T.-Y. et al. (2017). *Focal loss for dense object detection.* ICCV.
- Loshchilov I., Hutter F. (2019). *Decoupled weight decay regularization.*
  ICLR. **← AdamW.**
- Thompson B.A. et al. (2014). *Application of a 5-tiered scheme for
  standardized classification of 2,360 unique mismatch repair gene variants.*
  Nature Genetics 46:107–115. **← InSiGHT.**
- Jia X. et al. (2021). *Massively parallel functional testing of MSH2 missense
  variants conferring Lynch syndrome risk.* American Journal of Human Genetics
  108:163–175.

**Verify before citing — the repo names these but I have not confirmed the exact
reference:** VariPred; ProPath; the CSBJ per-residue token-classifier paper;
MVmamba; Rayner et al. 2022 (CIMRA/PMS2). Pull each from the original source and
confirm authors, year, venue and the specific hyperparameters you attribute to
them (e.g. ProPath's backbone LR 1e-5 / batch 8 / 10 epochs). Attributing a
recipe to a paper that did not specify it is an easy and embarrassing reviewer
catch.

---

## 13. Writing checklist

- [ ] Every number in the paper traced to `RUNLOG.md` or a fresh run — no number
      carried over from a pre-fix run
- [ ] Table 3's caption states "residue-disjoint, not protein-disjoint"
- [ ] AlphaMissense ClinVar contamination stated in Results *and* Discussion, not
      only Limitations
- [ ] Seeds, split counts, and library versions in Methods
- [ ] The word "state-of-the-art" used only where a baseline was run on your split
- [ ] Every **[PENDING]** either filled or its claim removed
- [ ] ClinVar snapshot date pinned
- [ ] AlphaMissense non-commercial licence checked against the venue
- [ ] The four uncertain citations verified against originals
- [ ] Abstract numbers match the final tables exactly
