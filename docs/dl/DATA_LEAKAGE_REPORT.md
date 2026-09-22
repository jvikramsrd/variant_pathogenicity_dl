# Data leakage report (DL branch)

Two parts. **Part A** was measured on this PC from the four UniProt sequences
alone (read from the cached AlphaFold metadata; six pairwise alignments, 0.3 s).
**Part B** needs the canonical table and is generated on the DGX by
`vpdl-dl leakage` — its tables are pasted in below when produced, never typed.

The training pipeline enforces the critical checks itself: `run_cell` calls
`vpdl.dl.leakage.leakage_gate` before the first model is built and raises
`LeakageError` on any critical finding.

## Checks

| check | critical when | otherwise |
|---|---|---|
| exact duplicates | any duplicate variant row | — |
| HGVS duplicates | one protein change under two variant ids | multi-record ClinVar changes counted (info) |
| protein duplicates | two accessions with an identical sequence | — |
| same variant across splits | a variant key on both sides of a fold | — |
| residue group across splits | a `uniprot:position` group on both sides (L3) | — |
| homology straddle | aligned paralog residues on both sides under `logo_purged` / `family` | **warning** under `logo` (by design) |
| sequence similarity | — | warning: held-out protein ≥ 20 % identical to a training protein; homologous-twin counts |
| functional overlap | a functional value used as a feature; functional validation under a non-gene-disjoint split; a validation variant trained on in its fold | — |
| feature leakage | a feature agrees with the label at ≥ 0.99 (L2) | warning at ≥ 0.95 |
| derived features | a feature derived from a training label source (e.g. `dms` columns when training on `pg_dms`) | gene-constant features listed (dropped, L15); circular priors flagged (AlphaMissense thresholds were calibrated on ClinVar) |
| pretraining exposure | a per-fold (strict) backbone whose corpus did not hold out that fold's gene | — |

## Part A — sequence similarity on this panel (measured)

Global alignment, BLOSUM62, gap 11/1, free end gaps (`vpdl.dl.homology.align`):

| pair | identity / shorter | identity / aligned | coverage |
|---|---|---|---|
| MLH1 – PMS2 (MutL) | **0.259** | 0.281 | 0.922 |
| MSH2 – MSH6 (MutS) | **0.263** | 0.287 | 0.916 |
| MLH1 – MSH2 | 0.008 | 0.400 | 0.020 |
| MLH1 – MSH6 | 0.015 | 0.177 | 0.082 |
| MSH2 – PMS2 | 0.030 | 0.205 | 0.147 |
| MSH6 – PMS2 | 0.009 | 0.190 | 0.049 |

Clustering at 20 % identity recovers exactly the known MutL / MutS families,
which is the `family` split. Residues sharing a homology cluster with the
paralog:

| protein | residues aligned to the paralog |
|---|---|
| MLH1 | 697 / 756 (92.2 %) |
| MSH2 | 856 / 934 (91.6 %) |
| MSH6 | 856 / 1360 (62.9 %) |
| PMS2 | 697 / 862 (80.9 %) |

196 aligned MLH1–PMS2 positions and 246 aligned MSH2–MSH6 positions carry the
same wild-type residue, so an identical substitution can sit on both sides of a
leave-one-gene-out fold.

**What this licenses.** Leave-one-gene-out is a test of transfer to an unseen
*gene*, not to an unseen *sequence family*: when PMS2 is held out, 92 % of
MLH1's residues are homologous positions in training. That is not a bug in
LOGO — it is what the design measures — but it means LOGO results must be read
alongside `family` (no paralog in training) and `logo_purged` (paralog present,
homologous positions removed). Because 81–92 % of residues are homologous,
`logo_purged` removes most of the paralog's training rows; expect it to sit
close to `family`. What this does **not** license: a claim that LOGO numbers
are inflated. Whether paralog information helps is the LOGO vs family
comparison, measured on the DGX.

## Part B — per split scheme (to be generated on the DGX)

```bash
for s in logo logo_purged family random_debug; do
  vpdl-dl leakage --data data/built/canonical_full.csv --split $s --out runs/dl/leakage
done
```

Paste each `runs/dl/leakage/leakage_<scheme>.md` below.

### logo

_pending DGX run_

### logo_purged

_pending DGX run_

### family

_pending DGX run_

### random_debug

_pending DGX run (expected: critical-free but warnings; never a reported result)_
