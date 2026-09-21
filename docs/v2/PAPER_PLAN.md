# Paper plan — do pooled supervision sources beat individual ones?

Status: **design only. No results exist.** Every table below is a shape to be
filled, not a finding. Written 2026-09-20.

## The claim

> Variant-effect predictors are routinely trained on whatever supervision is
> available — clinical assertions, deep mutational scans, benchmark labels —
> pooled into one table. We ask whether that pooling is justified: for a
> four-gene Lynch-syndrome panel under leave-one-gene-out evaluation, does a
> model trained on all sources outperform the best model trained on any single
> source, and does the answer depend on the model class?

Note what is *not* claimed: not a new architecture, not state of the art. The
contribution is a measurement nobody appears to have made directly, plus the
protocol that makes it trustworthy. See
[LITERATURE.md](LITERATURE.md) §3 for how this differs from feature-category
ablation, which is already done.

## Why the question is live

Two pieces of published evidence point in opposite directions, which is what
makes it worth measuring rather than assuming:

- Fine-tuning on DMS improves *clinical* prediction, with 0.78 correlation
  between the two benchmarks' gains (arXiv 2405.06729) — pooling should help.
- Clinically-trained tools contribute little over and above broader annotation
  features, and PLM-derived scores contribute negligibly
  (AnnotateMissense, arXiv 2605.24520) — pooling may add mostly redundancy.

A correlation of 0.78 is high enough to expect transfer and low enough that the
sources plainly disagree somewhere. Where they disagree is the paper.

## Design

**Unit of comparison:** one cell = (source set × model × seed), evaluated
leave-one-gene-out over MLH1 / MSH2 / MSH6 / PMS2.

**Source sets.** Each individual source, then the pooled set:

| arm | supervision | notes |
|---|---|---|
| `clinvar` | ClinVar ≥2 star | the conventional choice |
| `pg_dms` | ProteinGym DMS | **confounded on this panel** — the entire DMS pool is one MSH2 assay, so it contributes nothing when MSH2 is held out. Must be reported as single-gene, not as a general DMS arm. |
| `pg_clinical` | ProteinGym clinical benchmark | independent clinical labels |
| `all` | everything pooled | the arm under test |

**Models.** `gbm` (expected strong), `mlp` (continuity with v1), `bilstm`
(baseline to beat, see `vpdl/models/bilstm.py`).

**Seeds.** 42/43/44 minimum. v1 measured MCC standard deviation up to 0.056
across seeds of one arm; any difference below that is not a finding.

**Primary metric** MCC; **secondary** ROC-AUC and PR-AUC, all with
10,000-resample bootstrap CIs. Thresholds selected on inner validation carved
from the training genes — never on the held-out gene.

## Tables to fill

**Table 1 — source composition.** Rows retained per source, overlap, label
counts, licence. Generated from the assembly report, never typed by hand.

**Table 2 — the main result.** Mean over scoreable genes, ± SD over three seeds.

| source set | gbm | mlp | bilstm |
|---|---|---|---|
| clinvar | · | · | · |
| pg_dms | · | · | · |
| pg_clinical | · | · | · |
| **all** | · | · | · |

**Table 3 — per gene.** The same cells broken out by held-out gene, with cohort
sizes. PMS2's fold is small enough that it may not be scoreable; report `n`
beside every number so a reader can see that rather than infer it.

**Table 4 — Δ(pooled − best individual)** with a bootstrap CI on the
difference. This single column is the paper's answer.

## Figures

1. Pipeline schematic — sources, assembly, split, evaluation. Drawn, not plotted.
2. Source composition and overlap (UpSet or stacked bar), from the assembly report.
3. Main comparison with CIs — the Table 2 shape, plotted.
4. Per-gene breakdown with cohort sizes.
5. Agreement between source-specific models: where do ClinVar-trained and
   DMS-trained models disagree, and is the disagreement systematic? This is the
   most likely home for the paper's genuinely interesting result.

## Threats to validity, stated up front

1. **The DMS arm is confounded with gene identity** on this panel. Unavoidable
   with one assay; must be stated wherever the arm appears, not buried.
2. **Four genes is a small panel.** Leave-one-gene-out over four folds gives
   wide intervals. The honest framing is a case study with a reusable protocol,
   not a general claim about variant effect prediction.
3. **PMS2 is partly unsupervisable.** The PMS2CL homology gate withholds labels
   in codons 382–862; the retained fold may be too small to score. Report the
   gate's effect explicitly — v1 once lost the entire gene to a build flag and
   trained sixteen cells before noticing.
4. **Circularity.** Several prior-score features derive from models trained on
   ClinVar. Ablations that drop a family while a proxy remains are refused by
   `vpdl.features.resolve_ablation`; arms run with `allow_proxy_leak=True` must
   be reported as joint bounds.
5. **ACMG frequency thresholds come from v1, not from the VCEP.**
   `vpdl.sources.gnomad` uses BA1 > 5%, BS1 > 0.1%, PM2 < 1e-5 — v1's values,
   chosen for an autosomal-dominant early-onset cancer syndrome. (A draft used
   generic BS1 1% / PM2 1e-4, ten times too permissive; corrected 2026-09-21.)
   The InSiGHT MMR expert panel publishes gene-specific thresholds. **Confirm
   against the VCEP specification before a headline run**, pass any
   differences via `thresholds=`, and record which set was used.
6. **A negative result is a result.** If pooling does not help, the paper says
   so. The protocol is designed so that outcome is publishable rather than a
   dead end.

## Sequence

1. Build each single-source table and the pooled table; commit assembly reports.
2. `gbm` across all four source sets, three seeds — this alone fills Table 2's
   first column and answers the headline question.
3. Add `mlp`, then `bilstm`.
4. Figure 5's agreement analysis — where sources disagree.
5. Draft, with Tables 1–4 generated from artefacts rather than transcribed.

## Non-goals

- Beating published state of the art. Different question, and a four-gene panel
  cannot support the claim.
- A new architecture. The models are instruments here, not contributions.
- Genome-wide generalisation. The protocol generalises; these numbers do not.
