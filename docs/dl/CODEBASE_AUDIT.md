# Codebase audit — DL branch

Written 2026-09-22, before any DL-branch code was changed. Read-only inspection
plus the existing test suites; no data was downloaded and nothing was trained.

## 1. Scope map

| Area | Paths | Status for this work |
|---|---|---|
| **DL — live (v2)** | `vpdl/sources/`, `vpdl/assemble.py`, `vpdl/features.py`, `vpdl/splits.py`, `vpdl/models/`, `vpdl/evaluate.py`, `vpdl/experiment.py`, `vpdl/analysis.py`, `vpdl/device.py`, `vpdl/provenance.py` | In scope. Extended, not rewritten. |
| **DL — ProteinGym study** | `vpdl/proteingym/` | In scope but not modified (separate study, results in `docs/RUNLOG.md`). |
| **DL — new** | `vpdl/dl/` (this work), `configs/dl/`, `scripts/check_hardware.py`, `tests/dl/`, `docs/dl/` | Created here. |
| **DL — v1 archive** | `src/`, `scripts/` (except `check_hardware.py`), `tests/test_*.py`, `run_*.py`, `main.py` | Frozen. Retained as the historical "existing DL model" (stage-2b siamese ESM-2 fine-tune) and as a source of ported logic. Not modified. |
| **LLM / reasoning — DO NOT TOUCH** | `vpdl/kb/`, `vpdl/slm/`, `tests/kb/`, `tests/slm/`, `docs/kb/`, `docs/slm/` | Out of scope. Not modified, refactored, retrained or re-run. |
| **Shared interfaces** | `vpdl/cli.py` (hosts `kb-*` and `slm-*` commands), `vpdl/sources/clinvar.py` (`_THREE_TO_ONE`, `_review_stars` are imported by `vpdl/kb/variants.py`), the assembled table `data/built/mmr.csv` (columns `gene, position, wt_aa, mut_aa, feature_alphamissense_score, feature_gnomad_log10_af, feature_gnomad_observed` are read by `kb-ask --evidence`), `pyproject.toml`, `.gitignore`, `docs/RUNLOG.md` | `vpdl/cli.py` **not modified** (the DL commands live in `vpdl/dl/cli.py`). `clinvar.py` / `gnomad.py`: record iterators added, `load()` output unchanged, both helpers kept. `pyproject.toml` / `.gitignore`: additive lines only. The assembled table is not rewritten. |

The LLM branch's only dependencies on DL code are the two ClinVar helpers and
the three table columns above. All three are kept byte-compatible.

## 2. What exists

### Data
- **Sources (v2)**: ClinVar (`variant_summary.txt.gz`, ≥2 stars, PMS2 gate),
  ProteinGym DMS (MSH2 Jia 2020 only on this panel), AlphaMissense, gnomAD v4
  (GraphQL API, AF + ACMG BA1/BS1/PM2 flags + gene constraint), UniProt
  (coordinate authority, pinned lengths 756/934/1360/862).
- **Assembly**: one row per `(uniprot_id, position, wt_aa, mut_aa)`, per-source
  labels `label__<source>`, contradictions quarantined, gnomAD absence → PM2,
  orientation check against AlphaMissense after the merge.
- **Last real build** (DGX, 2026-09-21): 74,328 rows, sha256 `0884e5dfcf63…`,
  450 ClinVar labels (MLH1 174 / MSH2 177 / MSH6 78 / PMS2 21), 16,749 DMS
  labels (MSH2), 641 of 1,312 PMS2 rows withheld by the homology gate.
- **v1 archive only**: CIMRA loader (`src/cimra.py`, user-supplied CSV,
  Tavtigian OddsPath bins), MaveDB loader (`src/mavedb.py`), AlphaFold pLDDT
  (`src/structure.py`), InterPro, MMseqs2 cluster split
  (`scripts/build_cluster_split.py`), PMS2 exon→codon derivation from Ensembl
  (`scripts/derive_pms2_homology_range.py`).
- **Local cache on this PC**: AlphaFold v6 PDB + metadata for all four MMR
  proteins (`data/mmr/raw/alphafold/`), v1 per-gene CSVs. No v2 built table.

### Models, training, evaluation (v2)
- `gbm` (xgboost/lightgbm/sklearn), `mlp` (residual MLP), `bilstm` (WT/VT
  residue windows ±7 + tabular) behind one registry; seeds derived per held-out
  gene (L7).
- `run_cell`: leave-one-gene-out only; inner validation carved from training
  genes; threshold chosen by MCC on inner validation; bootstrap CIs; per-variant
  predictions + inner-val predictions + provenance written per cell.
- `vpdl paired`: paired bootstrap ΔAUC on identical held-out variants.
- Metrics: ROC-AUC, PR-AUC, MCC, sensitivity, specificity, precision.

### ESM / PLM
- **v2 has no protein language model path at all** (`docs/v2/ARCHITECTURE.md`:
  "the ESM-2 embedding path; calibration" listed as still absent).
- **v1 archive** has a substantial one: `src/esm_extractor.py` (site embeddings,
  sliding windows for >1022 aa), `src/mvmamba_features.py` (global+local WT/VT,
  VariPred centred windows), `src/esm_finetune.py` (siamese / wt_site
  fine-tuning, partial unfreezing, bf16, grad accumulation, warmup+cosine,
  PLLR residual), `src/fusion.py` (concat and GateWave heads).

### Checkpoints
- v2: pickle envelope with a validated format tag (L14); models are not saved
  by `run_cell`.
- v1: `save_finetuned` / `load_finetuned_model` for the fine-tuned ESM.
- No resumable DL training checkpoint (optimizer/scheduler/RNG) exists outside
  `vpdl/slm/` (LLM branch, out of scope).

### Tests (baseline run on this PC, before any change)
- `tests/regression` (the v2 spec): **62 passed, 1 skipped (needs CUDA),
  4 failed**. All four failures are this PC's Application Control policy
  blocking scikit-learn's compiled `_loss` module (`DLL load failed … An
  Application Control policy has blocked this file`); none is a code fault.
  torch 2.14 CPU and transformers 5.17 **do** load here (an older note said
  torch was blocked; that is no longer true).
- `tests/test_*.py` target the v1 archive.

## 3. Assessment

### Works, and is kept
- The whole v2 protocol: record schema, assembly, label precedence, conflict
  quarantine, orientation anchor, PMS2 fail-closed gate, gene-constant guard,
  proxy-ablation guard, leak guard, provenance gate, paired analysis. This is
  the evaluation protocol every new DL model is plugged into, unchanged.
- The regression suite (`tests/regression/test_landmines.py`) as the spec.
- The existing `mlp` / `bilstm` / `gbm` arms and their published numbers
  (ClinVar-only mean ROC-AUC 0.963 / 0.962 / 0.947) as the **existing-model
  regression baseline**.

### Incomplete
- No PLM embeddings, no zero-shot scores, no fine-tuning, no pretraining.
- No structure, genomic, CIMRA or MaveDB source in v2.
- Only leave-one-gene-out; no cluster, family or debug split; no leakage report.
- No F1 / Brier / ECE / calibration slope; no calibration; no uncertainty.
- No canonical variant table (genomic coordinates, HGVS, review stars, QC
  status are not carried).
- No DL output schema for later integration.

### Inefficient
- v1 ESM extraction runs fp16 autocast only, one variant set at a time, and
  re-extracts per gene/model into `{gene}_{model}_features.npz` caches with no
  version metadata.
- v1 fine-tuning ran `batch 1 × accumulation 8 + gradient checkpointing`
  because of a 15 GiB card; those are configuration, not defaults, on the DGX.

### Scientifically unsafe or mislabelled (new findings of this audit)
1. **v1's "masked-marginal" scorer does not mask.**
   `src/mvmamba_features.py::MaskedMarginalScorer` and the PLLR term in
   `src/esm_extractor.py` / `src/esm_finetune.py` read log-probabilities from
   one *unmasked* wild-type forward pass. That is the **wild-type marginal**
   (Meier et al. 2021 define both, separately); the docstrings say otherwise.
   Every v1 number described as masked-marginal or PLLR (e.g. "zero-shot ESM
   ROC-AUC 0.834") is a wild-type-marginal number. v2 implements the true
   masked marginal and keeps the wild-type marginal as a separately named,
   cheaper baseline.
2. **ClinVar records sharing one protein change are collapsed by file order.**
   `vpdl/sources/clinvar.py::load` does `drop_duplicates(..., keep="first")`
   on the protein key. Two ClinVar records (different nucleotide changes, or
   the GRCh37/GRCh38 copies) mapping to one substitution with *discordant*
   classifications are resolved silently. The canonical builder now detects
   and quarantines these; `load()` keeps its behaviour by default so the
   existing dataset hash stays reproducible.
3. **Leave-one-gene-out is not a sequence-cluster split on this panel.**
   MLH1/PMS2 (MutL homologs) and MSH2/MSH6 (MutS homologs) are paralogs.
   Holding out PMS2 while training on MLH1 leaves homologous positions in
   training. A family-level split and homologous-position ("twin") counts are
   added.
4. **gnomAD frequencies are unreliable in the PMS2CL homology region.** The
   gate withholds ClinVar labels there, but population features for the same
   residues come from the same short-read mapping problem. The canonical table
   marks them `population_reliable = False`.
5. **`features.PRIOR_GROUPS["prior_scores"]` matches any column containing
   `_score`.** A new feature named e.g. `feature_contact_score` would silently
   join the prior-score ablation group. New DL features are named to avoid it,
   and explicit `structure` / `genomic` keywords were added (no existing column
   changes group).
6. **GBM ignores the inner-validation set** (already recorded in RUNLOG).
   Early stopping is added as an opt-in so historical GBM numbers stay
   reproducible.
7. **v1 stage-2b grid results are not a valid baseline**: they were computed on
   dataset `79b68399…`, which lost all PMS2 supervision (MISSING_EVIDENCE item
   14). The valid existing-model baseline is the v2 `train-clinvar` arms on
   `0884e5dfcf63…`.

## 4. Reuse plan

| Existing component | Decision |
|---|---|
| v2 sources, assembly, features guards, splits, provenance, evaluate, analysis | **Retained**; extended in place where noted. |
| `run_cell` | **Extended**: split schemes, embedding blocks (incl. per-fold), new model families, extra-row scoring/export, leakage gate. Leave-one-gene-out predictions and inner-validation predictions verified byte-identical to a pre-change snapshot (`mlp`, `bilstm`); results CSVs gain `f1`, `brier`, `ece` columns. |
| `mlp`, `bilstm`, `gbm` | **Retained unchanged** (GBM gains opt-in early stopping). They are the regression baseline. |
| v1 `_sliding_spans`, `centered_window_bounds` | **Ported** into `vpdl/dl/context.py` with tests. |
| v1 siamese site pooling, LayerNorm-before-head, warmup+cosine schedule, bf16/fp16 autocast rules | **Ported** into `vpdl/dl/plm/` and `vpdl/dl/trainer.py`. |
| v1 concat / GateWave fusion | **Generalised** to N modalities in `vpdl/dl/fusion.py`. |
| v1 CIMRA loader and Tavtigian bins | **Ported** into `vpdl/dl/functional.py` as validation-only data. |
| v1 AlphaFold pLDDT parsing | **Extended** into `vpdl/dl/structure.py` (adds RSA, SS proxy, contacts, half-sphere exposure, geometry; refuses a model whose residues differ from UniProt). Download is a `curl` step in the runbook. |
| v1 MMseqs2 cluster script | **Not ported** — with four proteins a built-in aligner is enough; MMseqs2 stays optional. |
| v1 masked-marginal scorer | **Replaced** by a true masked marginal (finding 1); wild-type marginal kept, correctly named. |
| `vpdl/kb/`, `vpdl/slm/` | **Untouched.** |

## 5. After the work — validation actually executed on this PC

| check | result |
|---|---|
| `tests/dl` (new; CPU, tiny random-weight models, synthetic data) | 107 passed |
| `tests/regression` | 62 passed, 1 skipped (needs CUDA), 4 failed — the same four scikit-learn `_loss` DLL blocks as the baseline; no new failure |
| `tests/kb`, `tests/slm` (LLM branch, run read-only as a compatibility check) | 47 passed, 22 passed |
| `run_cell` leave-one-gene-out equivalence | predictions + valpreds byte-identical to the pre-change snapshot |
| `vpdl-dl` CLI dry run on a 281-row synthetic table (real sequences, real AlphaFold files): canonical → leakage → structure (no SASA) → fusion train → calibrate → failure → export | all steps ran; random labels gave near-chance AUC (no leak signal) |
| four real AlphaFold models parse to exactly the UniProt sequences | yes (MLH1, MSH2, MSH6, PMS2) |
| paralog identity/homology on the four real sequences | see DATA_LEAKAGE_REPORT.md Part A |
| `git status` on LLM paths (`vpdl/kb`, `vpdl/slm`, their tests and docs) and `vpdl/cli.py` | no changes |

Not executed here: any training on real data, any pretraining, any PLM
download or inference with real weights, embedding generation, structure
prediction, hyperparameter search, or benchmarking.
