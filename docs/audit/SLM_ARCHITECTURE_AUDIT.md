# SLM architecture audit — local code audit, 2026-09-24

Scope: `vpdl/slm/{schema,labels,variants,interface}.py`, `vpdl/slm/modeling/**`,
`vpdl/slm/build/{features,examples}.py`, `vpdl/slm/text/acmg.py`,
`vpdl/slm/evaluation/explain.py`. Tiny random-weight models only; nothing trained on real data.

The SLM is a **broad genomic** model: no MMR gene is hard-coded in the model or data paths
(MMR appears only as a split reservation, stats tag, and one of two ACMG gene
specifications, marked `verified=False`).

## Variant-class coverage (ran `vpdl.slm.variants` on real-looking HGVS)

| Class | Parsed | Featurised | Modelled | Example checked |
|---|---|---|---|---|
| missense | yes | yes | yes | `p.Gly67Arg` |
| nonsense | yes | yes | yes | `p.Arg100Ter` |
| synonymous | yes | yes | yes | `p.Leu5=` |
| frameshift | yes | yes | yes | `p.Val42fs` |
| in-frame indel | yes | yes | yes | `p.Lys618del` |
| start-loss | yes | yes | yes | `p.Met1?` |
| stop-loss | yes | yes | yes | `p.Ter756Trpext*?` |
| splice site (±1/2) | yes | yes | yes | `c.790+1G>A`, `c.117-2A>G` |
| splice region (±3–8) | yes | yes | yes | `c.790+5G>A` |
| intronic (>8) | yes | yes | yes | `c.790+60G>A` |
| 5′ UTR / upstream | yes | yes | yes | `c.-28A>G` |
| 3′ UTR / downstream | yes | yes | yes | `c.*15T>C` |
| non-coding transcript (`n.`) | yes | yes | yes | read only |
| CNV / structural | yes (ClinVar Type / length) | yes | yes | ran |
| regulatory / promoter | folded into `utr5_or_upstream` (documented) | — | — | — |
| mitochondrial (`m.`), genomic `g.` | **no** → `other` | `other` | `other` | ran (C-4) |

## The reasoning flow

Implemented: backbone → pooled text + structured fields + optional DL embedding (masked)
→ fused representation → five-class head, ACMG head (gene-masked), and per-sentence
evidence-type/polarity heads trained in parallel; calibration, uncertainty, and a
deterministic grounded explainer after the fact.

Answer to "does it collapse to text → label?": the classifier itself is text (+ structured)
→ five classes. Evidence extraction, type and polarity are **parallel heads sharing the
backbone**, not inputs to the class head, and at inference their outputs are not consumed
(finding C-3). Evidence strength is parsed into the tables but has no head (C-5). This is
honest in the docs (the architecture diagram draws the heads in parallel); it is recorded
here so no one reads the flow as a chained reasoning model.

## Findings

| ID | Severity | Where | Finding | Status |
|---|---|---|---|---|
| C-2 | HIGH | `vpdl/slm/modeling/predict.py:32` | The ACMG gene-specification mask was applied to the training loss only; at inference, codes a gene's specification excludes (e.g. PM1/PP2/BP1 for MLH1, PP5/BP6 everywhere) came out as live probabilities (0.37–0.60 on a random model). | **FIXED** — `predict_documents` multiplies by the same applicability mask; test `test_acmg_codes_a_gene_specification_excludes_are_never_predicted` |
| C-1 | HIGH | `vpdl/slm/cli.py` `cmd_embed`; `finetune._evaluate_and_save` | The only export path (`vpdl-slm embed`) never fills `acmg`, `evidence`, `explanation` or `evidence_embedding`; `explain()` is exercised only by the smoke run with a hard-coded code list. Empty fields were indistinguishable from "no evidence found". | **MITIGATED** — every exported record now carries `quality_flags` `acmg_not_exported`, `evidence_not_exported`, `explanation_not_exported`. Wiring the explainer into the export is OPEN (feature work). |
| C-3 | MEDIUM | `vpdl/slm/modeling/multitask.py` | Evidence heads are parallel and their predictions are unused downstream. | OPEN — design decision |
| C-4 | LOW | `vpdl/slm/variants.py:109-146` | `m.` and `g.` notation fall to `other` (≈0.8% of ClinVar is `other` in total). | OPEN |
| C-5 | INFO | `vpdl/slm/schema.py` | `evidence_strength` is parsed, never modelled. | Documented |

## Verified correct

- Batch size 1 forward (class `[1,5]`, ACMG `[1,28]`); consistent dtype casts at each fusion
  point; attention masks respected in pooling.
- A missing DL record is a zero vector with `dl_mask = 0`, and the model's output is
  identical whatever the masked vector contains (not imputed).
- `StructuredEncoder` refuses forbidden features (review status, stars, submitter count,
  submitter, anything in `NEVER_INPUT`) and is fitted on training rows only.
- The five-class objective (P, LP, VUS, LB, B) with soft targets for "P/LP"-style pairs.
