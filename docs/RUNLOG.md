# Running Log

Append one entry per build / training run. Newest at the top.
Format: date · what ran (command) · outcome · artifacts.

---

## 2026-09-23 (Windows dev PC, CPU) — broad genomic SLM branch built; nothing trained

`vpdl-slm` (new; `vpdl` and the DL branch untouched). Audit first:
`docs/slm/LLM_CODEBASE_AUDIT.md`. Everything expensive is **NOT RUN — REQUIRES
DGX SPARK**; the runbook is `docs/slm/GENOMIC_SLM_DGX_RUNBOOK.md`.

Measured here, on the ClinVar file already on this PC (`vpdl-slm inventory`,
sha256 `7e5f0c79…`, 10 minutes):

- **4,558,681 variants, 35,013 gene symbols, 22,111 distinct conditions** — but
  **2,520,821 variants (55%) name no specific condition at all**, only "not
  provided", "not specified" or a laboratory umbrella term.
- Classes: VUS 2,372,680 · LB 1,098,869 · P 215,659 · B 214,051 · LP 126,098;
  conflicting 166,121; paired terms 107,712.
- Consequence: missense 55%, then synonymous 762k, intronic 499k, splice region
  165k, frameshift 150k, nonsense 97k, splice site 96k, CNV 76k — the corpus is
  **not** missense-only.
- Expert-panel variants 22,390 (3 stars); practice guideline 63.
- MMR (MLH1/MSH2/MSH6/PMS2): **31,725 variants = 0.70%** of ClinVar. MMR is a
  downstream specialisation here, not the training domain.

Three findings worth remembering:
1. **The narratives were never downloaded.** ClinVar's free text lives in
   `submission_summary.txt.gz` (`Description`), not in `variant_summary`. Every
   earlier plan that said "ClinVar free-text summaries" was planning against a
   file nobody had fetched. Readers are written and tested against synthetic
   files in ClinVar's layout; the corpus itself is TBD.
2. **ClinVar now carries VUS sub-tiers** (`VUS-high` 48, `VUS-mid` 42) — a
   native ranking signal for the VUS task, now parsed and used as a check.
3. **Umbrella and catch-all conditions cover most of ClinVar** ("Inborn genetic diseases" 362,633,
   "Hereditary cancer-predisposing syndrome" 202,567, "Cardiovascular
   phenotype" 93,976) would have made one "disease" of most of a lab's
   submissions; they are excluded from disease grouping.

Validation actually executed: `pytest tests/slm tests/kb tests/dl
tests/regression` → **401 passed, 3 skipped** (tests/slm alone 180: 22 existing
+ 158 new). `vpdl-slm smoke` — records → stats → clusters → split → roles →
examples → leakage → pretraining corpus → packing → continued pretraining (2
steps) → fine-tuning (1 epoch) → baselines → export → explanation — **ok in
42 s** on synthetic data with random weights. Both dry runs pass.

Two bugs the tests caught in the new code, both in guards that would otherwise
have failed silently: the tiny test tokenizer was trained (non-deterministic
merge order) so the tokenizer-fingerprint guard fired on meaningless
differences — it is now built from a fixed vocabulary; and the fingerprint
included the tokenizer's padding state, which a first call mutates, so the
token cache would never have hit.

No number here is a genomics result: synthetic data, random weights.

---

## 2026-09-22 (DGX Spark) — small language model: training speed measured

Trial corpus: 5 PubMed baseline files + GeneReviews passages. `vpdl slm-train
--size small --benchmark 20` (109.5M parameters, 2,048-token context,
524,288 tokens per step, random weights).

| | eager | `--compile` |
|---|---|---|
| tokens / s | 23,039 | **34,710** |
| achieved TFLOPS (BF16) | 20.4 | 30.7 |
| share of measured peak (90.1) | 23% | 34% |
| projected, 2.5B tokens | 30.1 h | **20.0 h** |
| loss at step 10 / 20; val at 20 | 9.656 / 9.016; 8.979 | identical |

- Training works on the GPU; loss falls from ~10.4 (uniform over 32k tokens).
- Identical losses show compilation changed speed, not the computation.
  `--compile` is now the default.
- First attempt failed inside Triton: `Python.h` missing. PyTorch routes the
  rotary-embedding outer product through a Triton kernel even without
  compile. Fixed with `sudo apt install python3.12-dev`; `slm-train` now
  checks for the header up front.
- Medium (~340M, 7B tokens) by the same FLOP arithmetic: ~7 days at 30.7
  TFLOPS, likely less (larger matrices use the GPU better). To be benchmarked.

---

## 2026-09-22 (DGX Spark) — knowledge base: second evaluation (development set); medgemma:27b chosen

`vpdl kb-eval --models qwen3:32b medgemma:27b llama3.1:8b`, after the prompt and
citation-reader changes below. Same index and questions as the first run —
**a development result, not a clean test** (the changes were made after reading
the first run's answers).

| | qwen3:32b | medgemma:27b | llama3.1:8b |
|---|---|---|---|
| unanswerable questions answered "not found" | **8/8** | **8/8** | **8/8** |
| answerable answered, key facts present | 22/24 | 22/24 | 21/24 |
| answered but key fact missing | 0 | 0 | 0 |
| withheld by the citation check | 1 | 0 | 2 |
| "not found" on answerable questions | 1 | 2 | 1 |
| seconds per question | 35.3 | **6.2** | 1.6 |

(first run: qwen3 18/24, llama 13/24 with one misleading answer)

- The prompt change removed nearly all citation-format withholdings.
- **medgemma:27b becomes the default**: same score as qwen3:32b, nothing
  withheld, ~6x faster. It answered the MLH1/PMS2 immunohistochemistry
  question "not found" rather than risk it — a safe failure.
- `ashkenazi-founder` is "not found" for every model: the one search miss
  (MSH2 c.1906G>C sits in a large variant table), handled safely.
- llama3.1:8b now passes the keyword check on the IHC question, but it stays
  disqualified: one prompt change is not evidence the failure mode is gone.
- Owed before any accuracy claim: (1) read medgemma's 22 answers against their
  cited passages; (2) a fresh question set from other chapters, written before
  looking at any answers.

---

## 2026-09-22 (DGX Spark) — knowledge base: first build and evaluation; safety bar met

`vpdl kb-build` — 892 GeneReviews chapters -> 44,476 passages (122 archive files
not in NCBI's chapter list skipped, 0 unreadable); bge-m3 embeddings 651 s, 100%
on GPU; ClinVar lookup 32,827 Lynch-gene records.
`vpdl kb-eval --models llama3.1:8b qwen3:32b` — 24 answerable, 8 unanswerable,
5 variant-lookup questions (`docs/kb/eval_questions.jsonl`). Both models 100% on GPU.

| | llama3.1:8b | qwen3:32b |
|---|---|---|
| right section found (search, all 892 chapters) | 23/24 | 23/24 |
| unanswerable questions answered "not found" | **8/8** | **8/8** |
| answerable questions answered (valid citations) | 14/24 | 18/24 |
| answered AND key facts present | 13/24 | 18/24 |
| withheld by the citation check | 9 | 5 |
| variant lookups correct | 5/5 | 5/5 |
| seconds per question | 1.7 | 34.6 |

Reading it:
- **Safety bar met by both:** no unanswerable question was answered.
- **qwen3:32b: every answer it gave carried the key facts** (18/18). The six
  it did not give were withheld or "not found" — safe failures.
- **llama3.1:8b: one answered question lacks its key fact** — either a wrong
  answer that passed the citation check or a correct one in other words.
  Must be read before this model is used at all.
- **Three llama withholdings were a bug in the check, not the model:** answers
  that were just a number ("48% [S1].") were treated as empty. Fixed the same
  day (a sentence with a number or citation is a claim); re-run owed.
- Both models answered `ashkenazi-founder` "not found" — consistent with the
  one search miss.
- The remaining withholdings (6 llama, 5 qwen) are sentences without a citation.
  The offending sentences are in `runs/kb/answers_<model>.jsonl` (`detail`)
  and decide the fix: prompt format vs. citation rule.
- qwen3 is 20x slower, most likely its thinking mode.

**Answer review (same day).** Every answer that failed or was withheld was read:
- **llama3.1:8b gave one clinically misleading answer that passed every
  check** (`ihc-mlh1-pms2-loss`). The source's first step for MLH1/PMS2 loss is
  BRAF and/or MLH1 promoter methylation testing on the tumour; llama answered
  with the step that comes after it ("germline or paired germline/tumor
  testing"). Verbatim words, valid citations, wrong for the question.
  **llama3.1:8b is disqualified** for this use. qwen3:32b answered it correctly.
- llama's 5 "uncited" answers were correct but copied GeneReviews' literature
  references ("[Pearlman et al 2017, ...]") instead of passage numbers; its
  `pms2-difficult` answer cited as "[S1 Note: 9]", which the check did not read.
- qwen3's 5 "uncited" answers cited once per paragraph or added a closing
  sentence. One closing sentence ("emphasized across multiple guidelines") is
  not in any passage — the strict rule caught a real embellishment, so it stays.
- The three llama "empty" answers ("48% [S1]", "38% [S2]", "66-71 years
  [S2][S5]") were correct.

Changes: the citation check reads "[S1 Note: 9]" as S1; the prompt now forbids
copying literature references and closing remarks, requires a citation on
every sentence, asks for steps in order, and shows a format example.
**These changes were made after reading the evaluation's answers, so the next
run on the same 32 questions is a development result, not a clean test.** A
fresh question set, written before looking at answers, is owed before any
accuracy claim.

---

## 2026-09-21 (DGX Spark) — ProteinGym combined vs individual: combining HURTS, 212 of 217 assays

`vpdl pg-combined` (ridge, CPU, `fold_random_5`) — 217 assays, 696,311 single
substitutions, 95 zero-shot score features, sibling assays excluded via
ProteinGym's reference file (55 assays affected). 532 s.

**Reading check passed first.** 79 of the 95 score columns matched a published
zero-shot column by name; for every one, our raw per-assay Spearman equals
ProteinGym's (Pearson 1.0 across 217 assays, mean |diff| ≤ 0.0003). The features
are the right columns, the right way up.

| | |
|---|---|
| mean Spearman, individual (one model per assay) | **0.708** |
| mean Spearman, combined (one model, all assays) | **0.597** |
| mean / median delta (combined − individual) | **−0.110 / −0.101** |
| combined better in | **5 / 217** assays (Wilcoxon p = 1.8e-36) |
| zero-shot TranceptEVE_L alone, same variants | 0.450 |

By assay size (mean delta): <1k variants −0.129 (57 assays), 1k–5k −0.118 (112),
>5k −0.071 (48); Spearman(delta, size) = +0.29.

Worst: TADBP_HUMAN_Bolognesi_2019 −0.386, RASK_HUMAN_Weng_2022_abundance −0.381,
SOX30_HUMAN_Tsuboyama_2023_7JJK −0.351, RAD_ANTMA_Tsuboyama_2023_2CJJ −0.335,
KCNE1_HUMAN_Muhammad_2023_expression −0.318. Best: AICDA_HUMAN_Gajula_2014_3cycles
+0.101 (209 variants); the other four gains are ≤ +0.015.

What this does and does not show:
- **Both arms learn.** Combined beats the no-training reference by +0.15, so one
  shared weighting of the 95 predictors is useful — just much less useful than a
  per-assay one. Same direction as the MMR result: pooling heterogeneous labels
  cost accuracy there too.
- **Small assays lose MORE, not less** — the opposite of the usual case for
  pooling. Untested explanation: the combined loss is row-weighted, so the shared
  weights are fitted mostly to the large assays. Test: weight each assay equally.
- **Several of the worst assays measure stability, abundance or expression**
  rather than activity. Untested explanation: which predictors work depends on
  what an assay measures, and the combined model is not told. Test: delta by
  ProteinGym's `coarse_selection_type`, or a combined model given it.
- **Random folds favour the individual arm** (it trains on other substitutions at
  the same positions). Owed: `--scheme fold_modulo_5` and `fold_contiguous_5`
  before any claim about pooling in general.
- One model class (linear). Owed: `--model gbm`.

Code change after this run: ridge moved from sklearn `RidgeCV` to an equivalent
Gram-matrix implementation that runs on the GPU (`--device`, L21 checks it
matches sklearn). A `--device cuda` re-run should reproduce these numbers.

---

## 2026-09-21 (DGX Spark) — ProteinGym: published per-assay results REPRODUCED

`vpdl pg-reproduce` — One-Hot baseline on all 217 substitution assays, ProteinGym's
own `fold_random_5` folds, one Spearman per assay over pooled out-of-fold
predictions, compared with ProteinGym's published "One-Hot Encodings" column.
5.4 s on CPU. Regression suite: 57 passed.

| | |
|---|---|
| correlation with published, across 217 assays | **Pearson 0.970, Spearman 0.959** |
| within 0.02 / within 0.05 of published | 71% / **86%** |
| mean ours / mean published | 0.619 / 0.598 |
| median difference | +0.009 |

**Where we differ, we are higher** — largest gaps: ZIKV polyprotein +0.309, POLG_CXB3N
+0.278, HMDH +0.180, POLG_DEN26 +0.180, HIV env +0.172, MSH2 +0.099.

Checked before believing it:
- **Not leakage.** None of the six largest-gap assays contains a duplicated
  mutant, so no variant can sit in both train and test.
- **The pattern is protein length.** All six are 852–3,423 residues against a
  median of 245 across the 217.
- **Consistent with an optimisation budget, not a protocol mismatch.** Both are
  the same linear model on the same features. ProteinGym's is trained by AdamW for
  a fixed 10,000 steps, batch 64, learning rate peaking at 3e-4 and decaying to
  1e-5 (ProteinNPT `scripts/train.py`); on a 3,423-residue protein a given position
  appears in ~2% of batches, so its weight sees ~200 small updates. This solves
  the ridge problem exactly. `pg-reproduce` now reports
  `spearman_gap_vs_length`; a clearly positive value confirms it.

**Verdict:** reproduction passes. The pipeline computes what ProteinGym computes,
and its individual-assay baseline is at least as strong as the published one.

**Next — the combined model.** One-hot features are protein-specific (they encode
positions), so they cannot be pooled across proteins. The combined arm needs
features that mean the same thing for every protein: ProteinGym's precomputed
zero-shot scores (1.9 GB, no GPU) or ESM-1v embeddings computed on the DGX
(696,311 variants, 51,854 of them on proteins longer than ESM's 1,022-residue
window). Individual and combined arms must use the same features.

---

## 2026-09-21 (DGX Spark, GB10 + PyTorch) — the effect replicates across GBM, MLP, BiLSTM

Same table, same held-out ClinVar variants, seeds 42/43/44. `vpdl paired`,
10,000 paired resamples. Stratified mean ΔAUC against each model's own
ClinVar-only arm, over informative scoreable genes (MLH1 + MSH6 for DMS arms —
MSH2 is structurally identical and excluded; MLH1 + MSH2 + MSH6 for AlphaMissense):

| arm vs ClinVar-only | GBM | MLP | BiLSTM |
|---|---|---|---|
| + all DMS (pooled) | **−0.071 [−0.116, −0.025]** | −0.035 [−0.074, +0.001] | −0.036 [−0.074, +0.003] |
| DMS only | **−0.048 [−0.088, −0.007]** | **−0.038 [−0.079, −0.001]** | **−0.051 [−0.099, −0.006]** |
| AlphaMissense alone | −0.029 [−0.062, +0.002] | **−0.045 [−0.077, −0.016]** | **−0.045 [−0.079, −0.013]** |

Bold = CI excludes zero. MLH1 on its own: pooling degrades it significantly in
all three models (−0.092 / −0.051 / −0.054).

ClinVar-only ROC-AUC by model:

| | GBM | MLP | BiLSTM |
|---|---|---|---|
| MLH1 | 0.951 | 0.970 | 0.975 |
| MSH2 | 0.914 | 0.951 | 0.959 |
| MSH6 | 0.977 | 0.967 | 0.953 |
| mean, 3 scoreable | 0.947 | 0.963 | 0.962 |

**Reading.**

1. **Pooling hurts in every model family.** Same direction everywhere, largest
   for GBM. Significant for GBM; for MLP and BiLSTM the upper CI bound sits at
   +0.001 / +0.003, so borderline — but MLH1 alone is significant in all three.
2. **Training only on the MSH2 assay is significantly worse on clinical labels
   than training on ClinVar, in all three models.** The most robust result here.
3. **Trained models now beat AlphaMissense significantly** — for MLP and
   BiLSTM, not GBM. This answers the power concern in the dose-response entry:
   the stratified mean has enough power where single genes did not.
4. **Two predictions in `docs/v2/ARCHITECTURE.md` were wrong.** GBM was expected
   to be the strong arm and the BiLSTM a baseline to beat. Measured: MLP ≈ BiLSTM
   > GBM. BiLSTM ≈ MLP means the residue windows add nothing measurable over the
   six tabular features — the recurrence is not doing the work, the features are.
5. **GPU determinism holds.** MSH2 pooled-vs-ClinVar is exactly 0.000 [0, 0] for
   MLP and BiLSTM as well, trained with torch on the GB10.

**The model comparison is not yet fair — do not claim neural > GBM.** MLP and
BiLSTM early-stop on inner validation; GBM fits all 600 trees with no early
stopping on ~270 rows, which plausibly explains its deficit. Within-model
comparisons (the paper's actual question) are unaffected, since every arm of a
model uses the same configuration. Before any cross-model claim: give GBM early
stopping, re-run its arms, then `vpdl paired --reference-model gbm --cross-model`.

---

## 2026-09-21 (DGX Spark) — dose-response: the harm scales with volume

`--train-cap pg_dms=N`, N = 100 / 300 / 1000 / 3000, GBM, seeds 42/43/44, same
table. `vpdl paired` against ClinVar-only, 10,000 paired resamples:

| DMS labels added | MLH1 ΔAUC [95% CI] | MSH6 ΔAUC [95% CI] |
|---|---|---|
| 100 | +0.001 [−0.041, +0.055] | −0.001 [−0.012, +0.014] |
| 300 | −0.017 [−0.074, +0.049] | +0.002 [−0.017, +0.025] |
| 1,000 | −0.041 [−0.104, +0.029] | −0.006 [−0.028, +0.019] |
| 3,000 | −0.041 [−0.103, +0.031] | −0.007 [−0.029, +0.016] |
| **16,749 (all)** | **−0.092 [−0.163, −0.015]** | −0.049 [−0.107, +0.000] |
| DMS only | −0.044 [−0.106, +0.023] | **−0.051 [−0.108, −0.002]** |
| *AlphaMissense alone* | *−0.044 [−0.121, +0.032]* | *−0.008 [−0.043, +0.024]* |

MSH2: exactly 0.000 [0, 0] for every DMS arm — the design constant, again.

**Reading.** Small additions are harmless; harm grows with the DMS:clinical
ratio and is significant at full volume on MLH1. At full volume the pooled
model scores **below AlphaMissense with no training at all** (0.859 vs 0.907).

**Correction to the entry below.** It said the dose-response would decide
between label semantics and volume. It cannot: both predict harm that grows
with volume, and "no harm at N = 100" does not rule out a per-label effect too
small to see at that size. The experiment settles the *practical* question —
naive pooling at natural volumes hurts, balanced pooling does not — and leaves
the mechanism open. Leading candidate: DMS covers every possible substitution,
almost none observed in gnomAD, so gnomAD features carry no signal there; in
ClinVar they are what separates benign from pathogenic. Swamping training with
DMS may teach the model to ignore them. Testable via feature importance.

**Power, stated plainly.** ClinVar-only beats AlphaMissense alone by +0.044 /
+0.036 / +0.008 (MLH1/MSH2/MSH6) and no CI excludes zero. With 31-57 benign
variants per gene, this panel cannot show a trained model beats AlphaMissense
gene by gene. `vpdl paired` now adds a stratified mean ΔAUC across informative
scoreable genes (structurally identical folds excluded) for a headline number
with a tighter interval.

---

## 2026-09-21 (DGX Spark) — first grid: pooling ClinVar with DMS HURTS

Three arms on one table (`0884e5dfcf63…`), GBM, seeds 42/43/44, all scored on
the same 450 held-out ClinVar variants. Mean ROC-AUC ± SD over seeds:

| gene | ClinVar only | ClinVar + DMS | DMS only |
|---|---|---|---|
| MLH1 (174) | **0.951** ± 0.006 | 0.859 ± 0.026 | 0.907 ± 0.023 |
| MSH2 (177) | 0.914 ± 0.031 | 0.914 ± 0.031 | skipped (all DMS is MSH2) |
| MSH6 (78) | **0.977** ± 0.003 | 0.927 ± 0.016 | 0.925 ± 0.010 |
| PMS2 (21, 4 benign) | 1.000 | 0.858 ± 0.052 | 0.672 ± 0.098 |

**Design check passed:** MSH2 is identical in the first two arms. Holding MSH2
out removes every DMS label, so they train on the same data — and they came out
the same to three decimals. The arm mechanics are sound; every other difference
is caused by the training data.

**Finding, preliminary:** adding the MSH2 DMS assay to ClinVar training cost
0.092 on MLH1 and 0.050 on MSH6, roughly 5-6 seed-SDs each. DMS alone beat the
pooled arm on MLH1.

**Not yet publishable — one confound is unresolved.** The pooled arm trains on
16,749 MSH2 assay rows against ~276 clinical rows (60:1). Two explanations fit:

1. *Label semantics* — "damaging in a cell assay" is not "pathogenic in a
   clinic", and mixing them corrupts the clinical signal.
2. *Volume* — 60:1 MSH2 rows turn a clinical model into an MSH2-assay model.

Only (1) is a scientific finding. Next: dose-response with `--train-cap
pg_dms=N` at N = 100/300/1000/3000. If small N already hurts, it is (1).

**Caveat on `vpdl compare`:** its DMS-only mean (0.916) averages MLH1 and MSH6
only, since MSH2 was skipped. Not comparable with the other arms' means;
compare per gene, or use `vpdl paired`.

Added for the follow-up: `vpdl paired` (paired bootstrap ΔAUC on identical
variants, plus a zero-training AlphaMissense baseline) and `--train-cap`.
`roc_auc` tie-ranking vectorised via scipy for the ~10^6 calls the paired
bootstrap makes. Regression suite: 44 tests / 19 categories.

---

## 2026-09-21 (DGX Spark `spark-5472`, GB10, aarch64) — first v2 build and first v2 result

**Regression suite:** 41 passed, 1 failed (the torch-dependent MLP seeding
test; torch deliberately not yet installed). First time the suite ever
executed. No test failed on its own account.

**Build** — `vpdl build --sources clinvar pg_dms alphamissense gnomad --out data/built/mmr.csv`,
dataset sha256 `0884e5dfcf63…`. Every oracle check against v1's manifests passed:

| | v1 | v2 |
|---|---|---|
| rows | 74,328 | **74,328** (exact) |
| ClinVar labelled | 464 | 450 (MLH1 174 / MSH2 177 / MSH6 78 / PMS2 21) |
| DMS labels | 16,420, MSH2 | 16,749, MSH2 only |
| gnomAD observed | 6,492 | 6,795 of 74,328; the rest filled as PM2 |

Orientation checks ran and passed: ClinVar **0.927**, DMS **0.871** against
AlphaMissense. Before the 2026-09-21 assembly fix this check could never
execute. 20 MSH2 variants where ClinVar and the DMS assay disagree were
quarantined. PMS2 homology gate withheld 641 of 1,312 PMS2 rows; gene retained.

**First result** — `vpdl train ... --train-sources clinvar --model gbm --seeds 42 --n-bootstrap 1000`:

| gene | v2 ROC-AUC | v1 curated-features head | Δ |
|---|---|---|---|
| MLH1 (n=174) | 0.957 | 0.965 | −0.008 |
| MSH2 (n=177) | 0.896 | 0.908 | −0.012 |
| MSH6 (n=78) | 0.980 | 0.979 | +0.001 |
| **mean, scoreable** | **0.944** | **0.951** | −0.007 |

PMS2 1.000 on 17 pathogenic / 4 benign — uninformative, excluded from the mean.

Within 0.012 of v1 on every gene despite 6 features against 27 and 450 labels
against 683. That is the differential check the 2026-09-20 council asked for,
and it passes: the rebuild reproduces the old pipeline's magnitude. **Not a
scientific comparison** — the label sets differ — but strong evidence nothing
is broken.

Per-gene seeds were identical to those computed on the Windows dev box
(`798404085` for MLH1), confirming derive_seed is stable across x86 and ARM.

**Provenance bugs found in this run's own summary, fixed before the grid:**
`scikit-learn` recorded as null (looked up by import name, not distribution
name); `dirty: true` caused by untracked outputs rather than code (v1's
failure mode — every v1 summary said dirty). Dirty now means modified tracked
files or untracked files under `vpdl/`, and the paths are recorded. A
porcelain-parsing bug that ate the first character of the first path was
caught in testing before it shipped.

---

## 2026-09-15 (dev box + Linux CUDA box `DESKTOP-UJ3ATL6`) — dataset re-pinned, main grid re-run; ablations still owed

**What ran.** `.venv/bin/python scripts/run_stage2b_grid.py --tiers 1 2 3 4 5
--esm_model facebook/esm2_t33_650M_UR50D --mode siamese --eval lopo --batch_size 1
--grad_accum 8 --gradient_checkpointing --n_bootstrap 10000 --force` on the Linux CUDA
box, against a freshly rebuilt `data/mmr/processed/extended/extended_dataset.csv`
(`build_mmr_dataset.py`, `built_at_utc` 2026-09-13T03:18:59Z). Pushed as `6f871b7`,
pulled clean on the dev box (no dataset/manifest files outside `data/processed/stage2b_grid/`
were part of that push).

**Outcome — dataset identity moved, not just the model.** The rebuild picked up a
fresher ClinVar snapshot (one label moved: 464→463 labelled, 9589→9590 VUS), corrected
a stale `sources.gnomad.enabled` manifest flag (no new gnomAD data — same 6492 AF rows),
and added MaveDB as a populated source (17014 scored rows, previously no source block at
all). New dataset SHA-256 `79b68399…`, re-pinned as canonical over the previous
`78eb5d60860c…`. Full writeup, the MaveDB training-vs-validation decision (kept
validation-only, matching PROJECT_PLAN.md Phase 2 — `prepare_split()` already filters to
`label_source in {clinvar, pg_clinical}`, no code change needed), and the reopened-item
list are in `MISSING_EVIDENCE.md` item 14.

**Caught before it reached the manuscript.** The metric deltas between this run and the
previous one at identical (cell, seed) pairs were as large as 0.47 MCC — well outside the
paper's own seed-spread noise floor (SD up to 0.056). Checked
`provenance.dataset_sha256` before trusting any of it rather than after: that is what
surfaced the dataset change. Caught before any table was touched.

**Only the 16-cell main grid was retrained.** The 12 `ablate_*` feature-family ablation
cells on the branch still carry `dataset_sha256` starting `78eb5d60860c…` — the
superseded build. **Owed:** re-run all 12 ablation cells (`ablate_domains` /
`ablate_structure` / `ablate_prior_scores` / `ablate_gnomad_and_scores`, seeds 42/43/44)
against the now-current table, then error analysis and figures, in that order — but see
the blocker below first.

**Blocker found — PMS2 lost all clinical supervision in the rebuild.** Every one of the
16 new cells' summaries records `splits_skipped_no_rows: ["PMS2"]`; MLH1/MSH2/MSH6
holdout counts are unchanged (208/335/119) so this is not a proportional effect of the
ClinVar refresh (which only moved 1 record overall). PMS2 went from 21 usable holdout
variants to 0. Root cause not yet found — likely the homology gate or a coordinate
mapping regressed for PMS2 specifically. **Do not re-run the ablations or trust
`79b68399…` as a baseline until this is root-caused** (see `MISSING_EVIDENCE.md` item
14) — re-running against a build that silently dropped a gene would just be more work to
discard.

**Manuscript changes made in this pass.** Table 1 (source counts) updated to the new
manifest's numbers, MaveDB row corrected from "not enabled" to "17014 scored rows, held
out by design," AlphaMissense license corrected from CC BY-NC-SA 4.0 to the actually-true
CC BY 4.0 (verified against the source repository, not assumed) in two places, and the
MaveDB/CIMRA Table 1 `\todo` confirmations resolved from the manifest directly. **Not
touched:** Tables 3/4 and every headline AUROC/MCC number in the abstract and results —
those still describe the superseded-dataset run and must not be cited until items 1 and 3
are re-closed against `79b68399…`.

---

## 2026-09-12 (read-only dev box, round 2) — fresh-context review; one checkpoint-loading gap closed, inference pipeline added

No training, inference, dataset processing, or benchmark was run (same
environment as the entry below: no working Python install with
torch/pandas/pytest here). Scope was a second fresh-context static review of
this branch, followed by new implementation work for two gaps the review
found still open: an ECC-plugin-driven audit/hardening pass per the
project's standing "inspect -> map -> static review -> implement -> static
test/review -> document -> fresh-context review" loop.

**Found.** `src/esm_finetune.py::save_finetuned` has stamped a `format` tag
(`FINETUNE_CHECKPOINT_FORMAT`) on every checkpoint since it was introduced,
but `load_finetuned_model` never checked it back on load — the identical
defect class already fixed for `src.transfer.load_transfer_head`
(`TransferHeadFormatError`), just not ported to this sibling function. A
foreign or future-format `.pt` file would have been accepted silently and
failed later, deep inside model construction, with an error naming neither
the checkpoint nor its format.

**Fixed.** `src/esm_finetune.py` — added `FinetuneCheckpointFormatError` and
a format check in `load_finetuned_model`. Deliberately *not* a byte-for-byte
mirror of `TransferHeadFormatError`: real legacy checkpoints predating this
field already exist and are migrated elsewhere in the same function (the
`use_pllr` -> `pllr_mode` handling just below), so a missing tag is accepted
(logged as a warning) and only a *present*, *mismatched* tag is rejected.
Rejecting on missing would have silently broken loading of every
already-produced checkpoint that predates this session. Two new tests added
to `tests/test_new_data_sources.py`: a foreign-format rejection, and an
explicit no-format-tag-still-loads regression (so a future edit can't
reintroduce the stricter, breaking check by accident). Syntax-checked with
`python3 -m py_compile`; not run, for the same reason nothing else on this
branch has been.

**Reviewed, no defect found.** `scripts/run_mmr_transfer.py`'s own load path
for the stage-1 pretrain checkpoint (`ckpt.get("config")`'s `esm_dim` /
`feature_columns` check, lines 269-278) already rejects an incompatible
checkpoint via a targeted schema comparison rather than a version-string
tag — adequate for what it guards, and not the same gap as the two format-tag
cases above; left unchanged.

**Added (new capability, not a defect fix).** No inference entry point
existed anywhere in this codebase — every model-scoring codepath was fused
into a train-and-evaluate script. Added `src/inference.py` +
`scripts/predict.py`, scoped deliberately to `arch="priors"` stage-2
transfer-head checkpoints only (a single named-column feature view; scoring
an `esm`/`concat`/`gatewave` checkpoint needs an ESM-2 backbone forward pass,
which this pass does not implement — `UnsupportedArchitectureError` names
the gap rather than guessing at it). Reuses `load_transfer_head`'s existing
format versioning and stored per-view scaler statistics; never fits
anything on the inference input. `tests/test_inference.py` (16 tests,
synthetic-only: hand-built `torch.nn.Module` stub, no real checkpoint) covers
malformed/reordered/missing columns, unknown/unseen genes, an unsupported
architecture, a propagated format error, NaN features, single-row-vs-batch
consistency, an explicit threshold override, and CPU-only device selection
with no CUDA visible. See `docs/INFERENCE.md`. Syntax-checked, not run — no
real `.pt` checkpoint has been loaded through this path in any environment
yet; see that doc's "Remaining risk".

**Added (portable smoke test, not run here).**
`scripts/sanity_check_overfit.py` — a tiny synthetic (numpy-generated,
perfectly-separable) overfit check for `src.transfer.build_model` /
`fit_head`, for the other PC to run before trusting any real-data result
from the same training code path.

---

## 2026-09-12 (read-only dev box) — static review of the uncommitted audit branch; one doc/deliverable gap closed

No training, inference, dataset processing, or benchmark was run (no working
Python environment with pytest/pandas exists on this machine; confirmed by
import failure in both the system interpreter and `.venv`). Scope was a
fresh-context static review of the already-uncommitted changes on this
branch (`docs/PIPELINE_MAP.md`, `docs/HOLDOUT_PROTOCOL.md`,
`src/data_validation.py`, `src/gene_aliases.py`, `src/split_manifest.py`,
and the provenance/figure/seeding diffs), plus the `git diff` against `HEAD`.

**Found:** `docs/HOLDOUT_PROTOCOL.md` (written earlier this branch) claims
`tests/test_split_manifest.py` exists, covers `src/split_manifest.py` and
`src/gene_aliases.py` with synthetic data, and has been syntax-checked but
not run. The file did not exist (confirmed via `ls tests/` and `git status`).

**Fixed.** Added `tests/test_split_manifest.py` — 15 tests, synthetic data
only (a 2-line temp CSV stands in for a dataset file so `sha256_file` has
real bytes to hash; real MMR gene symbols/accessions from `CANONICAL_GENE_IDS`
so `resolve_gene_id` exercises the actual alias table). Covers: symbol/
accession/case resolution, unknown-gene rejection, same-spelling and
cross-spelling disjointness violations, manifest round-trip through JSON,
manifest-version rejection, and dataset-checksum verification (match and
mismatch). Syntax-checked with `python3 -m py_compile`; not run, for the
same reason nothing else on this branch has been (no pytest/pandas here).

**Reviewed, no defect found.** The seeding fixes in
`scripts/eval_leave_one_protein_out.py` and `scripts/run_mmr_transfer.py`,
the SSRF-style hardening in `src/interpro.py`/`src/structure.py`
(host-prefix checks, page caps), the `download_file` `max_bytes` cap, the
`refresh_manifest` label-count recomputation, and the `figure2`/`figure4`
additions in `scripts/make_figures.py` were cross-checked against their
declared column/schema producers (`src/metrics.py::evaluation_report`,
`scripts/recalibrate_grid.py`) and found internally consistent.
`prepare_split` in `scripts/finetune_esm_mmr.py` was checked against
`assert_disjoint_gene_sets`: not wired in, and correctly so — its two
partitions are a boolean mask and its exact complement over one column, so
they are disjoint by construction and an alias-collision check adds no
protection there; it remains valuable for a future caller that builds
partitions from independently-specified gene lists.

---

## 2026-09-07 (CUDA box + CPU dev box) — post-seed-fix re-run completed, 28 cells

Two pushes closed the re-run item 12 left open, and the paper's model numbers now
all come from one initialisation regime.

- **16 grid cells** (`b9a47c3`, run at `aec4333`) — the full freeze-depth x PLLR x
  branch grid, re-run with the per-split seeding fix.
- **12 feature-family ablation cells** (`472d20b`, run at `a3acf30`, a descendant
  of the fix) — re-run after the first push deleted the originals from the branch;
  see `MISSING_EVIDENCE.md` item 13.

**Provenance holds across the whole set.** All 28 summaries carry a native block,
one `dataset_sha256` (`78eb5d60860c...`) and one `split_definition`
(`6f87ecd06877ff66`). `make_figures.py --strict_provenance` passes. No backfill was
needed, so item 7 closes. Caveat: every summary records `git dirty: true`; what was
uncommitted on the CUDA box at run time is not recoverable from the artifacts.

**The fix moved all four genes, as item 12 predicted** — 58 of 64 grid rows changed.
Mean |dAUROC| vs the pre-fix run: MLH1 0.011, MSH2 0.006, MSH6 0.014, PMS2 0.022.
Largest single shift 0.066 (PMS2, `esm_frozen_pllr-concat`), which is what a
21-variant holdout with 4 negatives does.

**The pre-registered reading survives.** Mean AUROC over the scoreable genes,
`pllr=residual`, seed 42:

| branch | frozen | last2 | full |
|---|---|---|---|
| `esm+priors` | **0.9449** | 0.9358 | 0.9409 |
| `esm` only | 0.9042 | 0.9321 | 0.9270 |

Backbone gradients buy nothing once the priors are there. The branch axis added on
2026-09-02 was the right correction: the 0.880-vs-0.945 gap §6.12 was built around
was the feature set, not the freeze depth.

**Error analysis (item 9, new).** `scripts/error_analysis.py`, run on both headline
arms. 85 consensus errors of 683 for the frozen arm, 97 for the full fine-tune. Every
false positive of the frozen arm is *MSH2*. The covariates that separate errors from
correct calls are `label_source` and `evidence_tier` (BH p = 1.0e-4), **neither a
model input**: 44 of 56 false positives are `evidence_tier == unreviewed`
ProteinGym-clinical benign calls. That is a label-quality finding and it explains the
*MSH2* MCC anomaly logged on 2026-08-28. 193 of 683 variants flip between seeds of one
arm — the item-12 spread, measured per variant.

**Figures.** `scripts/make_figures.py` now produces Figures 2-6. Figure 2 (dataset
composition) is new; panel (c) shows why *MSH2* behaves differently — it carries 144
of the panel's 180 PG-clinical benign labels against *MLH1*'s 6.

---

## 2026-09-06 (CPU dev box) — reproducibility check FAILED, root-caused, fixed

The 2026-09-05 check on the CUDA box (`10fdc98`) ran the same cell twice at one
commit — `repro_a` / `repro_b`, frozen backbone, `esm+priors`, seed 42. It does
not reproduce, and the way it fails names the cause.

| holdout | order | predictions differing | max abs prob delta | ROC-AUC a / b |
|---|---|---|---|---|
| MLH1 | 1st | 208/208 holdout + 95/95 inner-val | 0.942 | 0.9485 / 0.9276 |
| MSH2 | 2nd | 0/335 | 0.0 | bit-identical |
| MSH6 | 3rd | 0/119 | 0.0 | bit-identical |
| PMS2 | 4th | 0/21 | 0.0 | bit-identical |

**Root cause.** `run_one_split` built `ESMFineTuneClassifier` — which
initialises the head — before anything seeded the RNG; the only `set_seed` was
inside `fit_esm_finetune`, several statements later. `main()` never seeded
either, so the *first* split of a process drew its head from the entropy
PyTorch seeds its default generator with at import. Every later split inherited
a state that split one's `set_seed(42)` had already made deterministic, which is
exactly why three of four folds matched to the last bit. `MMR_GENES` is ordered
`(MLH1, MSH2, MSH6, PMS2)`, so the unseeded split has always been MLH1.

**Folds 2-4 matching was contingent, not safe.** They inherit the RNG state
*after* split one's training, so their initialisation depends on how many draws
split one consumed — which depends on the epoch it early-stops at. Both runs
here stopped MLH1 at epoch 3. Had they stopped at different epochs, all four
folds would have moved.

**Fix.** `set_seed(args.seed)` at the top of `run_one_split`, before the model is
constructed. Per split rather than once in `main()` on purpose: it also makes a
single gene re-run with `--eval holdout` reproduce what the full sweep computed
for it, which chained-from-fold-one seeding would not.
`scripts/compare_finetune_strategies.py` had the same defect in
`run_frozen_probe` and `run_esm_finetune` — there the unseeded model is whichever
strategy runs first, so a head-to-head comparison depended on loop order — and is
fixed the same way. No other driver is affected: `run_mmr_transfer.py`,
`train_extended.py`, `eval_leave_one_protein_out.py`, `pretrain_esm_80.py` and
`main.py` all call `set_global_seed` at the top of `main()`.

**Tests:** 1 new in `tests/test_finetune_grid.py`, which aborts `run_one_split` at
model construction and asserts the head's first RNG draw is the same under two
different ambient seeds. It fails on the old code in 4 s on CPU. Suite 214 -> 215,
all passing.

**What this does to the results already in hand.** All 28 grid cells have an
unseeded MLH1. The measured spread from initialisation alone is 0.021 AUROC on
MLH1, against a seed-to-seed spread of 0.019-0.049 on the same gene in the
three-seed arms — so most of what was read as seed variance on MLH1 is
uncontrolled initialisation, and **no MLH1 difference below about 0.02 AUROC
separates two cells.** MSH2/MSH6/PMS2 are unaffected as *measurements*. The
numbers stay valid draws; they are not replayable, and the fix changes every
fold's initialisation, so restoring replayability means re-running.

---

## 2026-09-04 (CPU dev box) — feature-family ablation mechanism; no training run

Code and documentation only. **No model was trained and no number in the paper
changed.**

- **Feature-family ablation** (`MISSING_EVIDENCE.md` item 3, Table 4 rows 4–7).
  `src/transfer.py` names four families in `PRIOR_FEATURE_GROUPS` — `structure`,
  `gnomad`, `domains`, `prior_scores` — and `scripts/finetune_esm_mmr.py`
  exposes `--drop_prior_groups`. The groups partition `TRANSFER_PRIOR_COLS`
  exactly; a prior column added later without a group now fails a test rather
  than being silently unablatable.
- **The proxy guard is the point.** `--drop_prior_groups gnomad` alone *raises*,
  naming the AlphaMissense/zero-shot columns that survive: those models were
  trained on population data, so a "without gnomAD" arm that keeps them has
  removed the legible copy of the signal and nothing else. `--allow_proxy_leak`
  overrides it for the comparison where the proxy is the subject, and is
  recorded in the summary JSON.
- **Provenance.** Each run's summary JSON now records `drop_prior_groups`,
  `allow_proxy_leak` and the resolved `prior_columns` list — "27 prior columns"
  in the paper becomes checkable against the artifact instead of asserted.
- **Tests:** 6 new in `tests/test_mmr_modules.py`; suite 190 → 196, all passing.
- **Manuscript bibliography.** `docs/manuscript/references.bib` was rewritten
  from the verified list in `docs/PAPER.md` §11 plus `docs/DATASETS.md`, and the
  eight now-dangling `\cite` keys in `main.tex` (`acmg`, `clinvar`,
  `proteingym`, `alphamissense`, `gnomad`, `alphafold`, `mavedb`, `esm2`) were
  repointed at the new author-year keys; four citations were added where the
  text names a method the bibliography already covers (PLLR, temperature
  scaling, isotonic regression, circularity). Every `\cite` key now resolves.
  **Not verified by a build** — no TeX toolchain on this box; `latexmk -pdf
  main.tex` still has to run somewhere before submission. Author lists are
  still truncated to `and others`, DOIs are still absent, and `uniprot`,
  `interpro` and `mane` are still `TODO` placeholders.

**Still outstanding:** the four ablation runs themselves. Commands are in
`MISSING_EVIDENCE.md` item 3, pinned to the `esmpri_concat_frozen_pllr-residual_seed42`
grid cell so only the feature set moves.

---

## 2026-09-02 (CPU dev box) — Stage-2b ablation grid: plumbing smoke run

End-to-end check of the new grid path before it leaves for the CUDA box. **8M
parameters, two epochs, one held-out gene — a plumbing test, not a result.**
None of these numbers is comparable to a 650M run and none belongs in the paper.

- **Command:**

  ```bash
  .venv/bin/python scripts/run_stage2b_grid.py --tiers 3 \
    --esm_model facebook/esm2_t6_8M_UR50D --mode siamese \
    --eval holdout --holdout_gene MSH2 \
    --mmr_csv data/mmr/processed/extended/extended_dataset.csv \
    --panel_json data/mmr/processed/extended/panel_sequences.json \
    --epochs 2 --n_bootstrap 200 --out_dir <scratch>/grid_smoke
  ```

- **5/5 tier-3 cells completed, 0 failed.** Split sizes 277 fine-tune / 71
  inner-val / 335 holdout — the clinical-only partition, with PMS2's 21 rows in
  the fine-tune pool.
- Every cell wrote all three artefacts (results CSV, per-variant predictions
  CSV, summary JSON), plus the combined `stage2b_grid_results.csv` and
  `stage2b_grid_manifest.json`.
- **The predictions are seed-ensemblable**, which is the point of the change:
  all five cells return the same 335 variants keyed by
  `gene:position:wt_aa:mut_aa`, each row carrying `label`, `prob`, `threshold`,
  `seed`, `cell_slug`, `branch`, `n_unfrozen_layers`, `pllr_mode`. Runs 1 and 2
  of Stage 2b returned metrics only and cannot be compared variant-by-variant.
- **Resumability verified:** re-running the identical command logs
  `5 already complete | 0 to run` and executes nothing.

**Two things the run itself turned up:**

1. **Resume was broken under `--eval holdout`** — fixed in this commit.
   `finetune_esm_mmr.py` writes `..._siamese_holdout_MSH2_<slug>.csv`, while the
   driver's completeness check looked for `..._siamese_holdout_<slug>.csv`, so no
   holdout cell was ever recognised as complete and a resume silently re-ran the
   whole tier. Both sides now derive the name from
   `src.finetune_grid.output_tag`, with a regression test that creates files
   using the *script's* function and asserts the *driver* finds them. `--eval
   lopo` — what the GPU run uses — was unaffected: the two constructions
   happened to agree there.
2. **Ignore the wall-clock in that manifest.** One cell reports 172.8 min against
   ~2.8 min for its four siblings. The box suspended (s2idle) at 09:39:17 and
   resumed at 12:29:07 mid-cell; `journalctl` confirms it. Profiling the three
   fusion/PLLR configurations directly gives 0.43–0.46 s per example with no
   difference between them, so a tier-3 cell here really costs ~3 min.

Grid driver and `docs/GPU_RUN_PLAYBOOK.md` are ready for the CUDA box; 149 tests
pass.

## 2026-08-30 (Windows CUDA box) — Stage 2b re-run: PLLR fix + PMS2 included

Second Stage-2b run to complete ("Run 2" in `docs/PAPER_DRAFT.md` §6.10).
Incorporates the four ESM-branch corrections from 2026-08-28 (night) — chiefly
the PLLR term `log P(mut|X) − log P(wt|X)` now being structurally computable by
the fine-tuned model — and includes **PMS2 for the first time**: the fail-closed
pseudogene gate was opened with a verified `--pms2_codon_range`, so the 21
held-out PMS2 rows are homology-checked.

- **Command:** `run_mmr_pipeline.py`, siamese / ProPath recipe, `--eval lopo`.
  Pipeline defaults imply full unfreeze (`--n_unfrozen_layers -1`), `--use_pllr`,
  seed 42, 10,000-iteration CIs.
- **Not captured in the pasted console output:** the micro-batch / grad-accum /
  gradient-checkpointing values, the summary JSON, and the exact PMS2 codon
  range. Read `data/processed/esm_finetune/esm_finetune_summary_siamese_lopo.json`
  on the CUDA box and reconcile before quoting a config in the paper.

| holdout | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n | best_epoch |
|---|---|---|---|---|---|---|---|
| MLH1 | 0.9029 | 0.8506–0.9474 | 0.9764 | 0.5275 | 0.9316 | 208 | 5 |
| MSH2 | 0.8503 | 0.8057–0.8913 | 0.7669 | 0.4575 | 0.8963 | 335 | 2 |
| MSH6 | 0.8874 | 0.8200–0.9450 | 0.9121 | 0.6444 | 0.5131 | 119 | 1 |
| PMS2 | 0.9559 | 0.8382–1.0000 | 0.9903 | 0.6912 | 0.0678 | 21  | 3 |

Mean ROC-AUC **0.880** over MLH1/MSH2/MSH6 (n-weighted 0.873); 0.899 with PMS2.
Mean MCC **0.543** (three genes). Artifacts:
`data/processed/esm_finetune/esm_finetune_results_siamese_lopo.csv` (that box).

**Change vs the 2026-08-28 run** (PLLR structurally uncomputable, PMS2 absent,
MLH1 0.8993 / MSH2 0.8776 / MSH6 0.9073, mean 0.895 / MCC 0.492):

- ROC-AUC: MLH1 +0.004, MSH2 −0.027, MSH6 −0.020 → mean **−0.015**.
- MCC (the plan's primary metric): MSH2 0.297 → **0.458** (+0.161); MSH6
  0.652 → 0.644 (−0.008); MLH1 0.528 unchanged → mean **+0.051**, essentially
  all from MSH2.
- Thresholds tightened: MSH2's MCC-optimal threshold moved 0.057 → 0.896, so
  MLH1/MSH2 now agree within 0.04 (MSH6 0.51, PMS2 0.068 remain outliers).
- `best_epoch` 1/3/1 → 5/2/1/3: MLH1 now trains a meaningful number of epochs.

Net: the fix trades a fraction of a ranking point for a better operating point.

**Still owed (unchanged):** `--n_unfrozen_layers 0` ablation floor,
`--no-use_pllr` ablation, ≥3-seed averaging — now specified as a single grid in
`docs/PAPER_DRAFT.md` §6.12 (Table 8). Until the floor runs, these numbers
cannot be attributed to backbone fine-tuning: the frozen priors-probe (Stage 2,
2026-08-28: MLH1 0.9425 / MSH2 0.8527 / MSH6 0.9734, mean 0.923; from-scratch
27-feat variant mean 0.945) is still **ahead of Stage 2b by ~4–6 ROC-AUC
points**.

## 2026-08-28 (night, CPU box) — fixing the ESM branch

The ESM branch was the weakest part of the project (siamese LOGO mean ROC-AUC
0.895 against the priors probe's 0.944, early-stopping at epoch 1/3/1). Four
defects found, all in the branch itself rather than in the data.

### 1. The fine-tuned model could not see PLLR *(root cause)*

`ESMFineTuneClassifier` was built on `AutoModel` — the encoder alone, no
masked-LM head — so `log P(mut|X) - log P(wt|X)` was structurally
uncomputable. That term is the zero-shot ESM score, which on this panel
reaches **ROC-AUC 0.834 pooled with no training at all**. A 650M-parameter
backbone on 662 labels was being asked to rediscover it from scratch, and
early-stopped before it could.

Now loads `AutoModelForMaskedLM`, keeps `.esm` + `.lm_head`, and reads PLLR off
the **same** wild-type forward pass that already produces `site_wt` (Meier et
al. 2021 — one shared context, so no extra compute). Verified identical to
`src/esm_extractor.py`'s existing PLLR to **2.4e-06**. `--no-use_pllr` runs the
ablation. The LM head follows the backbone's freeze state.

### 2. Unnormalised head input

`feat` went straight into `nn.Linear`; the LayerNorm sat *after* it. Raw ESM
hidden states have large position-dependent norms, and in siamese mode half the
concatenated vector (`site_wt`, `site_mut`) is near-duplicate while the
informative part — their difference at one substituted residue — is small. A
`LayerNorm` now precedes the head. PLLR is scaled by a fixed constant (10.0,
a registered buffer, not a batch statistic) so it stays identical between
training and single-variant inference.

### 3. MVmamba's variant-type local window was the *wild-type* window

For chains over the positional capacity the branch sliced the centred window
out of `sequence` (wild type) and assigned it to `l_vt`. So `l_vt == l_wt`
exactly, and `l_vt - l_wt` / `|l_vt - l_wt|` were **identically zero** — two of
eight feature blocks dead, and precisely the windowed WT/VT contrast the
MVmamba recipe is built around. On this panel only **MSH6** (1360 aa) exceeds
the limit, so one gene silently had a different feature space from the other
three, inside a leave-one-gene-out design.

### 4. `compare_backbones.py` reported 1 - AUC

Raw PLLR is *negative* for damaging variants, but it was compared directly
against a pathogenic=1 label. Every cell came out below 0.5 (ROC-AUC
0.03-0.18 for two strong 650M models) — output that reads as "the protein
language model is useless on our data". Added
`MaskedMarginalScorer.pathogenicity_score()` (negated, higher = more
pathogenic) and switched the script to it. **Any earlier reading of that
script's output should be discarded.**

### Also fixed

The MVmamba extractor materialised the full `[N, L, d]` hidden tensor:
**25.2 GiB** for MSH6's 3,886 VUS at 1360 aa x 1280 dims — an immediate OOM on
a 14 GiB box, on exactly the VUS-scoring task the pipeline exists to perform.
Both outputs wanted from that pass are reductions, so it now embeds in chunks
(`vt_chunk_size`, default 64) and reduces on arrival: peak ~0.4 GiB. Verified
bit-identical to the old implementation on the short-sequence path at chunk
sizes 1/3/64; the long-sequence path differs only by defect 3 above.

### Verification

- In-model vs extractor PLLR: max abs diff 2.4e-06 across 6 variants.
- Chunked vs original MVmamba features: max diff 0.00e+00 (short path).
- End-to-end `finetune_esm_mmr.py` on CPU (esm2_t6_8M, frozen backbone,
  holdout MSH6): validation AUC now **rises across epochs** — 0.8813, 0.8829,
  0.8895 — instead of peaking at epoch 1. Holdout ROC-AUC 0.847. That is an
  8M-parameter smoke test, **not** a result to compare against the 650M runs.
- 86 tests pass under pytest; 37/37 in `test_mmr_modules.py` standalone. Three
  regression tests added (PLLR orientation, WT/VT local contrast, chunking).

### Still owed on the GPU box

- ~~Re-run stage 2b with PLLR on 650M~~ — done 2026-08-30 (see the entry at the
  top of this log). ROC-AUC mean −0.015, MCC mean +0.051 vs the pre-fix run.
- `--no-use_pllr` ablation, `--n_unfrozen_layers 0` ablation floor, and seed
  averaging are now a single grid — see the top-of-log TODO and
  `docs/PAPER_DRAFT.md` §6.12.

## 2026-08-28 (evening, CPU box) — closing the gap to the published bar

**Result: mean ROC-AUC 0.9229 -> 0.9445, mean MCC 0.4626 -> 0.6894** over the
three MMR genes with meaningful n. MSH2 now exceeds the best published single
predictor. The change is a one-flag configuration fix, not a new model.

### What was actually wrong

The broad panel was built with `include_gnomad: false`, so the stage-1
checkpoint's prior schema has **19** columns. Stage 2 pins its feature order to
the checkpoint's (correctly — weight transfer requires it), which silently
discarded the **8** richer features the MMR table does have:

    gnomad_log10_af  acmg_ba1  acmg_bs1  acmg_pm2
    af_plddt  af_disordered  in_interpro_domain  is_functional_site

`gnomad_log10_af` is the one PROJECT_PLAN.md Phase 3 step 1 explicitly requires
as an *input* feature, citing MVmamba's own 0.895->0.901 ablation. It was being
thrown away by the warm start.

### Measured (leave-one-gene-out, priors mode, 10,000-iteration CIs, seed 42)

| holdout | warm-start (19 feats) | scratch (27 feats) | delta | best published |
|---|---|---|---|---|
| MLH1 | 0.9425 | **0.9641** | +0.0217 | 0.984 (BayesDel) |
| MSH2 | 0.8527 | **0.9007** | +0.0480 | 0.896 (TranceptEVE-L) **beaten** |
| MSH6 | 0.9734 | 0.9686 | -0.0048 | 0.991 (MetaRNN) |
| PMS2 | 0.9559 | 1.0000 | +0.0441 | 0.956 (AlphaMissense) |

MCC gains are larger and uniform: +0.262 / +0.212 / +0.206 / +0.103. MCC is the
plan's primary metric.

Against PROJECT_PLAN.md Phase 3 step 7's MVmamba anchor (AUC 0.901 / AUPR 0.848
/ MCC 0.656): now 0.9445 / 0.9389 / 0.6894 — ahead on all three. **This is not
a like-for-like claim**: MVmamba reports on 18,731 variants across many genes,
this is 662 clinical variants across three. Report both denominators.

### New guard: `--gene_constant_priors {auto,drop,keep}`

The five gnomAD *gene-level* constraint columns (pLI, oe_lof, oe_mis, mis_z,
syn_z) hold one value per gene, so under leave-one-gene-out they are a
5-dimensional gene identifier, not evidence. Scratch-trained with them in,
**MLH1 collapses to ROC-AUC 0.500 / MCC 0.000 in every seed tried** (n=3);
dropped, 0.965 +/- 0.007. `auto` drops them for `--eval lopo`, keeps them for a
single `--eval holdout`, and `run_mmr_pipeline.py` resolves it once and passes
the same literal to both stages so warm-start schemas still match.

This currently only bites in `--scratch` mode, because the checkpoint schema has
no gnomAD columns at all. **It becomes live the moment the broad panel is
rebuilt with `--all_sources`** — which is exactly what the README recommends for
the full GPU run. Rebuild both stages together.

### Rank normalisation: implemented, not recommended

`--rank_normalize {off,add,replace}` converts each published score to a
within-gene percentile (+ a skip-NaN consensus mean). Motivation was strong: a
*zero-training* rank-mean beats the trained head on every gene, and on MSH2
beats the best published predictor (0.916 vs 0.896). But as head input it did
not reproduce that: `replace` gave no gain over dropping the gene-constant
columns, and `add` helped MSH6 (0.971 -> 0.985) while destabilising MLH1's
threshold (MCC 0.345 +/- 0.375). Left off by default; the flag and the finding
are worth keeping — a plain rank-mean consensus is a strong, honest baseline
the learned head still has to justify itself against.

### Caveats

- Four configurations were compared and the holdout was read each time. The two
  that shipped are defect fixes with an independent rationale (a discarded
  feature schema; a gene-identifier feature), not hyperparameters tuned on the
  test set — but seed-average and confirm via inner validation before writing up.
- Scratch beats warm-start here only because the checkpoint is
  feature-impoverished *and* 98.9% DMS-labelled. The right fix is to rebuild the
  broad panel with `--all_sources` and re-pretrain, so pretraining and
  fine-tuning share the 27-feature schema. Then re-test warm-start vs scratch.
- MLH1 and MSH6 remain ~0.02 ROC-AUC below the best single published predictor.
- PMS2's 1.000 is n=21 with 4 benign. Not a result.
- Artifacts: `data/processed/mmr_transfer_scratch/`.

## 2026-08-28 (later, CPU box) — audit, PMS2 inclusion, published-predictor benchmark

- **PMS2 is now trainable instead of dropped.** `scripts/derive_pms2_homology_range.py`
  derives the PMS2CL homology span from the Ensembl exon table for MANE Select
  `ENST00000265849`: exons 11-15 = c.1145-2589 = **protein codons 382-862**. The
  derivation self-validates (2,589 bp CDS -> 862 aa, matching the pinned P54278).
  Recorded as `src.mmr_dataset.PMS2_PSEUDOGENE_CODON_RANGE`; the gate itself stays
  fail-closed, the range must still be passed explicitly.
  - `scripts/build_mmr_dataset.py --min_stars 2 --pms2_codon_range 382 862`
    -> 74,328 rows; PMS2 contributes **21 clinical labels (17 P / 4 B), all at
    codon <=301**, plus 1,118 in-region-free VUS. 9,139 PMS2 rows inside the
    region keep their data and lose their labels, as intended.
  - Four-gene leave-one-gene-out, frozen priors probe, warm-started from
    `mmr_pipeline_pretrain.pt`, 10,000-iteration CIs:

    | holdout | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n |
    |---|---|---|---|---|---|---|
    | MLH1 | 0.9425 | 0.8822-0.9872 | 0.9787 | 0.4871 | 0.8123 | 208 |
    | MSH2 | 0.8527 | 0.8106-0.8903 | 0.7549 | 0.3986 | 0.4346 | 335 |
    | MSH6 | 0.9734 | 0.9477-0.9910 | 0.9740 | 0.5020 | 0.7753 | 119 |
    | PMS2 | 0.9559 | 0.8382-1.0000 | 0.9903 | 0.5830 | 0.3172 |  21 |

  - **Do not quote the PMS2 row as a performance result.** n=21 with 4 benign;
    the CI reaches 1.0. It demonstrates the gene runs end to end, nothing more.
  - The other three genes' numbers shifted slightly from the earlier three-gene
    run because PMS2 now contributes to their fine-tuning pool.

- **Published-predictor benchmark** — new `scripts/benchmark_published_predictors.py`
  scores all 17 ProteinGym clinical-benchmark predictors + AlphaMissense on this
  repo's own MMR clinical slice. Score orientation is fitted on the broad panel
  with the MMR genes removed, never on the evaluation data. Writes
  `docs/PUBLISHED_COMPARISON.md` + `data/processed/benchmark/`.
  - **Not yet run to completion on committed data.** A provisional pass
    (2,000 bootstrap iterations, three genes, before PMS2 was added) produced
    the numbers below; the definitive 10,000-iteration four-gene run is still
    owed and no benchmark artefact is checked in. Treat these as indicative.
  - Provisional: **the model did not clear the published bar on any Lynch
    gene.** Per-gene ROC-AUC, ours vs best published: MLH1 0.940 vs 0.984
    (BayesDel), MSH2 0.853 vs 0.896 (TranceptEVE-L), MSH6 0.973 vs 0.991
    (MetaRNN). Same ordering on MCC. CIs overlapped in every individual
    comparison, but the gap was consistent across all three genes and all
    three metrics.
  - The unsupervised methods (GEMME, EVE, TranceptEVE-L, PoET) also beat it,
    so the gap cannot be explained away as ClinVar circularity in the
    baselines' favour. GEMME was the strongest single comparator.
  - **PMS2 has zero ProteinGym zero-shot coverage** — AlphaMissense is its only
    published comparator, and its prior feature vector is correspondingly thin.
  - This is PROJECT_PLAN.md Phase 3 step 7's bar, and it is currently unmet.

- **Fixes.** MVmamba feature cache rejected itself permanently once any variant
  failed alignment (compared post-alignment `meta` against the raw table);
  `run_mmr_transfer.py`'s summary JSON reported requested rather than evaluated
  splits, so it claimed PMS2 was evaluated on every `--exclude_pms2` build. Two
  regression tests added for the PMS2 range (82 passing).

## 2026-08-28

- **Stage 2b: first successful ESM-2 backbone fine-tune** — `run_mmr_pipeline.py`
  (siamese / ProPath recipe, leave-one-gene-out), on the Windows CUDA box
  - **First Stage-2b run ever to complete.** The 2026-08-27 attempt OOM'd in the
    backward pass; the AMP + grad-accum + checkpointing knobs and the VRAM
    preflight added that day are what made it fit.
  - Leave-one-**gene**-out over the MMR panel — these are *unseen-gene* numbers,
    not residue-disjoint ones:

    | holdout | mode | ROC-AUC | 95% CI | PR-AUC | MCC | thr | n | best_epoch |
    |---|---|---|---|---|---|---|---|---|
    | MLH1 | siamese | 0.8993 | 0.8446–0.9445 | 0.9757 | 0.5275 | 0.4344 | 208 | 1 |
    | MSH2 | siamese | 0.8776 | 0.8350–0.9162 | 0.7931 | 0.2972 | 0.0567 | 335 | 3 |
    | MSH6 | siamese | 0.9073 | 0.8508–0.9536 | 0.9195 | 0.6519 | 0.2674 | 119 | 1 |

  - Mean ROC-AUC **0.895** (n-weighted 0.890) over 662 held-out clinical rows.
    PMS2 absent, as expected — the fail-closed pseudogene gate excludes it by default.
  - Artifacts: `data/processed/esm_finetune/esm_finetune_results_siamese_lopo.csv`

- **Cautions on the above — do not write these into the paper unqualified**
  - `best_epoch` is 1, 3, 1. The run early-stops almost immediately, so it is not
    yet established that backbone gradients bought anything over the frozen
    probe. **Run `--n_unfrozen_layers 0` as the ablation floor before claiming
    the fine-tune is what produced these numbers.**
  - MCC-optimal thresholds span 0.0567–0.4344 across three genes. Scores do not
    transfer on an absolute scale; MSH2 reaches MCC 0.297 despite ROC-AUC 0.878.
    Any deployed threshold must be per-gene or recalibrated.
  - PR-AUC moves opposite to ROC-AUC between MLH1 (0.976 vs 0.899) and MSH2
    (0.793 vs 0.878) → per-gene base rates differ sharply. Report prevalence
    per gene alongside these.
  - Single seed. Holdouts are small (119–335) and all three CIs overlap
    heavily, so no gene is significantly different from another. CIs are
    10,000-iteration (`run_mmr_pipeline.py` defaults `--n_bootstrap` to 10,000
    and passes it through to stage 2b), matching the paper's protocol section.
    Note the *standalone* scripts default to 2,000 — pass it explicitly there.

## 2026-08-23

- **Leakage-clean full training** — `scripts/train_extended.py --no_dms_features`
  - 80 genes, 190,494 labelled rows, group-disjoint CV, DMS-derived features removed
  - Train-vs-val AUC gap ≈ 0.000 → no classic overfitting
  - Clinical-only slice: **ROC-AUC 0.963 / MCC 0.78** (AM baseline: 0.945 / 0.75)
  - All-labels: 0.716 (honest number; earlier 0.9987 was circular via `dms_bin_median`)
  - Artifacts: `data/processed/extended_train/ext80_*`, checkpoints

- **Overfitting diagnostic added** — `scripts/diagnose_overfitting.py`
  - Proved `dms_bin_median` == flipped label on 97.3% of rows (target leakage)

- **Label-inversion bug fixed** in `src/extended_builder.py`
  - ProteinGym `DMS_score_bin=1` = top fitness half (tolerated); builder had
    mapped it straight to "pathogenic" → ~185k labels were inverted
  - Verified vs AlphaMissense anchor + PG-clinical anchor; dataset rebuilt, re-audited 12/12

- **Expanded dataset build** — `make_expanded_panel.py` + `build_extended_dataset.py --panel_file`
  - Panel 10 → 80 genes; master table 110,124 → 1,156,625 rows
  - 50,304 isoform-mismatched DMS + 152 AlphaMissense rows dropped by wt-validation
  - Audit: 12/12 passed (`audit_report.json`); ClinVar↔PG-clinical agree on all 1,919 overlaps
  - Train CSV: `extended_dataset_train.csv` (190,494 rows: 82,149 P/LP · 108,345 B/LB)

## TODO / next runs

- [x] GPU box: install `requirements-cuda.txt`, verify `torch.cuda.is_available()` — done 2026-08-28
- [ ] Definitive benchmark: `python scripts/benchmark_published_predictors.py
      --n_bootstrap 10000 --model_results data/processed/mmr_transfer/mmr_transfer_results_lopo.csv`
      (~90 min CPU; run after rebuilding the MMR table with `--pms2_codon_range 382 862`)
- [ ] Stage 2b ablation grid on the CUDA box — freeze depth × PLLR × **branch**
      (`esm` vs `esm+priors`), 16 cells in 5 tiers, ~31 h. One resumable command;
      see `docs/GPU_RUN_PLAYBOOK.md`. Every filename is cell-tagged, so the old
      "move the results CSV aside between cells" step is gone. Subsumes the old
      "ablation floor" and "seed averaging" line items. `PAPER_DRAFT.md` §6.12
      and `PAPER.md` §7.2/Table 6 now describe *this* grid; the earlier
      freeze-depth-only design they carried could not separate freeze depth from
      feature set.
- [ ] Remaining recipes on the same splits: `wt_site`, VariPred probe, MVmamba pooled
- [ ] Full model: `train_extended.py --features esm+priors --esm_model facebook/esm2_t33_650M_UR50D --no_dms_features`
- [ ] Leave-one-**protein**-out CV on the broad 80-gene panel (still owed; distinct from the MMR gene-out above)
- [ ] Score VUS risk tiers from ESM-mode run (`ext80_vus_predictions.csv`)
