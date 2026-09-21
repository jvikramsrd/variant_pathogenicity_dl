# vpdl v2 — architecture

Status: **design agreed, implementation not started.** Written 2026-09-19.

## What this is for

One claim, and the whole design serves it:

> Does pooling heterogeneous variant-effect data sources into one training
> table beat training on each source alone — and for which sources?

Every prior version of this project built one merged table and trained on it.
That makes the central question of the new paper *unaskable*, because the
sources are fused before any model sees them. So the organising constraint of
v2 is:

**Every data source must be independently loadable, independently trainable,
and independently evaluable, under one shared evaluation protocol.**

That is not a nice-to-have refactor. It is the experiment.

## Layers

Four, with one direction of dependency. No layer reaches back upwards.

```
sources/     one module per source. Each emits the same record schema and
             knows nothing about any other source. Independently testable,
             independently runnable.
                       |
assemble.py  composes 1..N sources into a training table. Label precedence,
             coordinate validation, provenance manifest. Selecting ONE source
             is the normal case, not a degenerate one.
                       |
features.py  table -> matrices. Tabular priors, ESM-2 embeddings, PLLR,
             local WT/VT windows. Ablation resolution lives here.
                       |
models/      fit/predict/save/load behind one protocol. mlp | gbm | bilstm.
evaluate.py  one metric path for every cell, so any two cells are comparable.
```

`provenance.py`, `splits.py` and `device.py` are cross-cutting and depended on
by everything; they contain no source- or model-specific logic.

### Source contract

Each `sources/<name>.py` exposes:

```python
def fetch(cache_dir: Path) -> Path      # download, checksum, resume
def load(path: Path) -> pd.DataFrame    # -> the standard record schema
def provides() -> SourceCapabilities    # labels? features? which columns?
```

Standard record schema — every source emits exactly these, nothing else:

| column | meaning |
|---|---|
| `uniprot_id`, `position`, `wt_aa`, `mut_aa` | the key. UniProt is the coordinate authority. |
| `gene` | resolved from `uniprot_id`, never trusted from the source |
| `label` | 1 = pathogenic, 0 = benign, NaN = unlabelled |
| `label_source` | which source supplied the label |
| `evidence_tier` | review confidence, source-specific vocabulary |
| `feature_*` | zero or more numeric feature columns, prefixed |

A source that cannot map a row onto the canonical sequence **drops it and
counts it**. Silent coordinate drift is how ~50k isoform-mismatched rows got
into v1 before wt-validation was added.

## Experiment matrix

This is the paper's results section, and the CLI exists to fill it in.

```
sources:  clinvar | proteingym_dms | proteingym_clinical | alphamissense
          | gnomad | mavedb | structure(alphafold+interpro) | ALL
models:   mlp | gbm | bilstm
splits:   leave-one-gene-out (primary), group-disjoint CV (secondary)
seeds:    42, 43, 44 — three minimum, always. Single-seed deltas are not
          interpretable on this data (observed MCC SD up to 0.056).
```

Every cell writes one run directory with metrics, per-variant predictions,
inner-validation predictions (so calibration is possible without a leak), and a
provenance block. `vpdl compare` pools cells only after `assert_comparable`
passes.

## Models, and why these three

The data is one vector per variant — tabular priors plus pooled ESM-2
embeddings — with 683 clinically labelled examples across four genes. That
shape dictates the shortlist.

- **`gbm`** (XGBoost/LightGBM) — the right tool for small-n heterogeneous
  tabular data, and the likely winner. v1's own result already points here: a
  curated-feature head with no ESM input beat every cell in the ESM grid.
- **`mlp`** — residual MLP head. Carried over conceptually from v1 because it
  is the arm all existing numbers use; keeps continuity with prior results.
- **`bilstm`** — included as an explicit **baseline to beat**, not a
  contender. ESM-2 is already the sequence model: a 650M-parameter transformer
  pretrained on UniRef50. A BiLSTM trained over windowed sequence context from
  four genes (3,912 residues) is radically undertrained by comparison, and the
  field moved from LSTM protein models (UniRep, early TAPE) to transformers for
  exactly this reason. Reporting it with a number answers "why not an RNN?"
  with evidence instead of an assertion, which is worth the small cost of
  running it.

**Measured 2026-09-21 — both predictions above were wrong.** ClinVar-only mean
ROC-AUC over the three scoreable genes: MLP 0.963, BiLSTM 0.962, GBM 0.947. The
BiLSTM tied the MLP rather than losing, and since it sees the same six tabular
features plus residue windows, the tie says the windows add nothing measurable —
the features carry the signal. GBM's deficit is likely configuration (no early
stopping, unlike the other two) and must be fixed before any cross-model claim.
See `docs/RUNLOG.md`.

Adding a model means implementing `models/base.py::Model` and registering it.
No other layer changes.

## DGX Spark

Target: GB10 Grace-Blackwell, unified CPU/GPU memory, aarch64 Linux.

What actually changes versus the current CUDA box:

1. **aarch64 wheels.** x86-64 PyTorch wheels do not apply. Requirements need a
   separate ARM path; use NVIDIA's published builds for the platform rather
   than the default PyPI index.
2. **Unified memory removes the contortions.** The current grid runs
   `--batch_size 1 --grad_accum 8 --gradient_checkpointing` because it was
   squeezed onto a ~15 GiB card. With a large unified pool those become
   *configuration*, not baked-in defaults. Expect a substantial wall-clock win
   from honest batch sizes alone.
3. **bf16 is native.** Existing `amp_dtype_for()` logic already prefers bf16
   when `torch.cuda.is_bf16_supported()`, which is the correct behaviour here.

`device.py` **detects and reports; it does not assume.** It queries capability,
available memory and dtype support at runtime and logs what it found, so a
wrong assumption surfaces as a log line rather than an OOM three hours into a
grid. Concrete SM version, driver and wheel index must be confirmed on the
machine — record what the box actually reports in `docs/v2/HARDWARE.md` on
first run rather than trusting this document.

## What carries forward from v1

**`tests/regression/test_landmines.py`** — 23 tests across 14 defect classes
(L1-L14), encoding every bug already paid for: the ProteinGym label inversion, the `dms_bin_median` target
leak, unqualified leakage groups, vacuous amino-acid validation, schema-blind
caches, PLLR sign, unseeded first splits, silent feature-schema truncation, the
PMS2 fail-closed gate and the 3-gene panel that got through it, stale
manifests, proxy-leaking ablations, the dead WT/VT window contrast,
incomparable pooling, and unvalidated checkpoint tags.

These are written against the v2 API *before it exists*. They are red until
implemented, and they are the definition of done for each layer. Nothing else
from v1 is carried over as code.

## Build order and status

Each step must end green on its slice of the regression suite. "Verified" below
means *executed*, not merely written — this dev box has no numpy/pandas, so only
the stdlib-only paths could be run here; the rest needs the Ubuntu box.

| # | Layer | Landmines | Status |
|---|---|---|---|
| 1 | `provenance.py`, `device.py`, `splits.py` | L3, L10, L13 | written; **L10, L13 verified**, L3 needs pandas |
| 2 | `sources/base.py`, `sources/clinvar.py` | L4, L9 | written; needs pandas |
| 3 | `sources/` — proteingym_dms, alphamissense, gnomad, uniprot | L1, L15 | written; **gnomAD group-matching verified**, rest needs pandas |
| 4 | `assemble.py` | L1, L9 | written; needs pandas |
| 5 | `features.py` | L2, L5, L6, L11, L12, L15 | written; needs pandas |
| 6 | `models/`, `evaluate.py` | L6, L7, L8, L14 | written; **L7 (seed derivation), L8, L14 verified** |
| 7 | `experiment.py`, `cli.py` | — | written; needs pandas |
| 8 | First experiment: single-source vs pooled, `gbm`, three seeds | — | **not started** |

### A note on gnomAD

It was nearly deferred alongside the other feature sources. That was wrong, on
this project's own evidence: un-dropping these columns produced v1's largest
single result improvement (mean ROC-AUC 0.9229 -> 0.9445, MCC 0.4626 -> 0.6894),
`ablate_gnomad_and_scores` was the largest ablation effect measured at -0.0526
AUROC, and allele frequency's increment over the external prior scores was
0.0211 against a standard error of 0.0052 — so it is not redundant with
AlphaMissense. Deferring it would also have left `PRIOR_GROUPS["gnomad"]` and
`PROXY_FOR` referring to a family no source produced, making
`resolve_ablation(drop_groups=["gnomad"])` a silent no-op that reported success.

It brings landmine **L15** with it: gene-level constraint (pLI, o/e missense,
missense Z) is constant within a gene, so under leave-one-gene-out it is a
gene-identity label and nothing else. `features.drop_gene_constant()` removes
it, and `experiment.run_cell` calls that on every cell.

Still absent by design: MaveDB, AlphaFold and InterPro sources; the ESM-2
embedding path; calibration. None blocks the headline experiment.

## Companion documents

- [LITERATURE.md](LITERATURE.md) — how these datasets have been used, the
  closest prior work, and what remains unasked. Read before claiming novelty:
  feature-category ablation is already published.
- [PAPER_PLAN.md](PAPER_PLAN.md) — the experimental design, the tables to fill,
  and the threats to validity stated up front.

## Decisions on the record

- **Full rewrite over consolidation** — chosen 2026-09-19 against the
  recommendation to consolidate. Mitigation is the regression suite above,
  written first.
- **The v1 Lynch/MMR stage-2b paper is abandoned**, and `results/stage2b-grid`
  is its archive branch. The PMS2 dataset bug is still fixed here, because any
  table trained on must be correct regardless of which paper consumes it.
- **MaveDB stays validation-only** unless deliberately moved, which is then an
  experimental arm with its own row in the matrix — not a build-flag side
  effect. See MISSING_EVIDENCE item 14.
