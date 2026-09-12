# Inference pipeline

Static design + implementation document. Nothing in this file, `src/inference.py`,
`scripts/predict.py`, or `tests/test_inference.py` has been executed against a
real checkpoint or dataset in this environment — this dev box has no working
Python install with `torch`/`pandas` (see `docs/RUNLOG.md`). Every command
below is for **the other PC** (the one with the trained checkpoint and a
working environment).

## Why this exists

Before this change, every script in this repo that scores a model
(`scripts/run_mmr_transfer.py`, `scripts/eval_leave_one_protein_out.py`,
`scripts/finetune_esm_mmr.py`) does so as one step inside a larger
train-and-evaluate loop. There was no standalone "load a saved checkpoint,
score a CSV of new variants" entry point — a real gap against "design a
correct and efficient inference pipeline." `src/inference.py` +
`scripts/predict.py` fill it.

## Scope

Scores **`arch="priors"` stage-2 transfer-head checkpoints only**
(`src/transfer.py::save_transfer_head` output) — a single named-column
feature view, no ESM-2 embedding computation required. This is deliberate,
not an oversight:

* `arch in ("esm", "concat", "gatewave")` checkpoints need an ESM-2 backbone
  forward pass over a raw sequence to produce their embedding feature view.
  That is a materially larger dependency (downloading/running a
  multi-hundred-MB-to-multi-GB transformer) that this pass did not implement
  or test. `scripts/predict.py` / `load_inference_model` raise
  `UnsupportedArchitectureError` naming the architecture rather than
  attempting a silent partial score — use the training-time scripts to score
  those checkpoints until an embedding path is added and tested against a
  real ESM-2 model.
* Calibration (`src/calibration.py`'s temperature/isotonic fitting) is **not**
  applied — `probability` is the model's raw sigmoid output. Calibrating a
  held-out probability is a separate, not-yet-audited step (see "Remaining
  risks" in this audit's final report).

## Design (implementation claims — verifiable by reading the code, not run here)

| Requirement | How it's met | Where |
|---|---|---|
| Explicit validated input schema | `validate_input_frame` names every missing identifying or feature column; reorders to the checkpoint's own column order regardless of input order | `src/inference.py::validate_input_frame` |
| Preprocessing + model versioned together | Delegates to `load_transfer_head`, which already rejects an incompatible checkpoint format (`TransferHeadFormatError`) | `src/transfer.py::load_transfer_head` |
| Gene/feature alignment via canonical IDs | Every `uniprot_id` resolved through `src.gene_aliases.resolve_gene_id`; unresolvable/misspelled gene raises `UnknownGeneError` | `src/inference.py::align_genes` |
| Only train-time-fitted preprocessing applied | Uses the checkpoint's own stored per-view `StandardScaler` statistics (`scale_views`); never fits on the inference input | `src/inference.py::predict` |
| Documented missing-feature behaviour | A NaN in any required feature column after alignment raises `InferenceInputError` naming the offending rows; inference does not impute | `src/inference.py::predict` |
| Prediction output schema | `uniprot_id, position, wt_aa, mut_aa, probability, predicted_label, threshold_used, checkpoint` | `src/inference.py::predict` → `PredictionBatch` |
| Batching | `predict_logits`'s internal batch loop (`batch_size`, default 1024) | `src/transfer.py::predict_logits` |
| CPU/GPU selection | `--device {cpu,cuda,auto}`; `auto` picks CUDA if visible, else CPU; explicit `--device cuda` with no CUDA fails loudly instead of silently falling back | `scripts/predict.py::resolve_device` |
| Model/version metadata in output | `PredictionBatch.checkpoint_path`, `.model_config`, `.threshold`; `--print_schema` dumps the checkpoint's declared schema without scoring | `src/inference.py`, `scripts/predict.py` |
| Useful error messages | Every raised exception names the exact missing column(s)/gene(s)/format string, never a bare `KeyError`/`AttributeError` | throughout |

## Portable commands (run on the other PC — not executed here)

Inspect a checkpoint's required schema without scoring anything:

```bash
python scripts/predict.py \
  --checkpoint data/mmr/processed/transfer/priors_MLH1_holdout_head.pt \
  --input_csv /dev/null --output_csv /dev/null \
  --print_schema
```

Dry run — validate an input CSV against the checkpoint's schema, report the
row count that would be scored, write nothing:

```bash
python scripts/predict.py \
  --checkpoint data/mmr/processed/transfer/priors_MLH1_holdout_head.pt \
  --input_csv my_variants.csv \
  --output_csv predictions.csv \
  --dry_run
```

Score, on CPU:

```bash
python scripts/predict.py \
  --checkpoint data/mmr/processed/transfer/priors_MLH1_holdout_head.pt \
  --input_csv my_variants.csv \
  --output_csv predictions.csv \
  --device cpu
```

Score, restricting to a known gene set and overriding the stored threshold:

```bash
python scripts/predict.py \
  --checkpoint data/mmr/processed/transfer/priors_MLH1_holdout_head.pt \
  --input_csv my_variants.csv \
  --output_csv predictions.csv \
  --known_genes MLH1,MSH2,MSH6,PMS2 \
  --threshold 0.4
```

`my_variants.csv` must contain `uniprot_id, position, wt_aa, mut_aa` plus
every column the checkpoint's own schema names (see `--print_schema`'s
`required_feature_columns`). Column order does not matter; extra columns are
ignored.

## Non-data-processing unit tests (synthetic only)

`tests/test_inference.py` — every model is a hand-built `torch.nn.Module`
stub, every input frame is a handful of in-memory rows. Not run in this
environment; run elsewhere with:

```bash
python -m pytest tests/test_inference.py -v
```

Covers: missing identifying/feature columns (named in the error), reordered
input columns, unknown/unaliased genes, genes outside a declared known set,
an unsupported checkpoint architecture, a propagated checkpoint-format
error, CPU-only device selection with no CUDA visible, NaN feature values,
single-row vs. batch scoring consistency, and an explicit threshold override.

## Remaining risk

No test here loads a *real* `.pt` checkpoint (this environment cannot run
one) — the synthetic tests exercise the schema/alignment/error-handling
logic, not the actual tensor shapes a real `save_transfer_head` payload
produces. Run `tests/test_inference.py` **and** a real `--dry_run` /
`--print_schema` call against an actual checkpoint on the other PC before
trusting this path for anything beyond the logic it was designed against.
