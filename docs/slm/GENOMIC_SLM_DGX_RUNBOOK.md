# Genomic SLM — running it on the DGX Spark

Every command below exists today (`vpdl-slm --help`). Nothing here has been
run on real data: **the whole pipeline is NOT RUN — REQUIRES DGX SPARK.**

Do the phases in order. Each one is cheap until Phase 7; stop at the first
surprise rather than working around it.

## Phase 0 — install and prove the stack (minutes)

```bash
sudo apt install python3.12-dev        # Triton's C helper; already needed by `vpdl slm-train`
```
```bash
pip install -e ".[genomic-slm]"
```
```bash
pytest tests/slm -q
```
Expect **188 passed** (22 from the from-scratch pipeline, 166 from this branch).

```bash
vpdl-slm hardware --smoke --out runs/slm_genomic/hardware.json
```
Read three fields: `running_on_intended_target` (should be `true` here),
`checks.bf16`, and `suggested_start.micro_batch` — a starting batch size
computed from free memory, which `autotune` (Phase 7) then measures properly.
If bf16 is false the report says why, and every later figure is a
development-machine figure.

```bash
vpdl-slm smoke
```
The whole pipeline on synthetic data with random weights, ~1 minute. `ok: true`
means the stages fit together. It measures nothing about genomics.

## Phase 1 — see what is already here (10 minutes)

```bash
vpdl-slm inventory --out runs/slm_genomic/inventory.json
```
```bash
vpdl-slm catalog --markdown --out docs/slm/catalog.md
```
The inventory streams the whole ClinVar file; on this developer's PC that took
10 minutes and found 4,558,681 variants across 35,013 genes.

## Phase 2 — download what is missing (one command each)

```bash
mkdir -p data/raw/clinvar && cd data/raw/clinvar
```
```bash
wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz
```
```bash
wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/var_citations.txt
```
```bash
wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
```
`submission_summary.txt.gz` carries the narratives — the text this branch
exists to read. Take `variant_summary` from the same release so the two agree.

Optional but valuable:
* **ClinGen Evidence Repository** — download the tab-delimited export from
  <https://erepo.clinicalgenome.org/evrepo/>. Its column names in
  `vpdl/slm/erepo.py` are **unverified**; the reader prints the file's real
  header and refuses to guess, so the first run tells you what to fix.
* **An older ClinVar release** (`.../tab_delimited/archive/`) for the VUS
  reclassification evaluation.
* **PubMed baseline** — already covered by docs/slm/RUNBOOK.md §2.

## Phase 3 — build the records (hours, one pass per file)

```bash
vpdl-slm build-records --variant-summary data/raw/clinvar/variant_summary.txt.gz \
  --submission-summary data/raw/clinvar/submission_summary.txt.gz \
  --citations data/raw/clinvar/var_citations.txt --workers 8 --out data/slm_genomic
```
Add `--erepo data/raw/clinvar/erepo.tsv` when you have it, and
`--dl-canonical data/built/canonical_full.csv` to fill the DL join key.

Try it small first: `--limit-documents 50000` finishes in minutes and produces
the same tables in miniature.

Read `data/slm_genomic/manifest.json`: input hashes, row counts, and every
reader's skip counts.

## Phase 4 — measure the corpus before believing in it

```bash
vpdl-slm stats --records data/slm_genomic --tokenizer pubmedbert \
  --out runs/slm_genomic/corpus_stats.json --markdown docs/slm/corpus_stats.md
```
This is the answer to "is there enough clinical text?". Until it has run, the
corpus is not described as sufficient anywhere.

```bash
vpdl-slm dedup --records data/slm_genomic --out data/slm_genomic/clusters.parquet --workers 16
```
Near-duplicate and laboratory-template clusters — the heaviest CPU step, and
the one that uses the machine's cores. `--workers` defaults to every core;
the result is identical whatever the number (each worker seeds its own
permutations the same way, which `tests/slm` checks), so this is speed only.

## Phase 5 — splits, roles, examples, and the audit

```bash
vpdl-slm splits --records data/slm_genomic --scheme variant \
  --clusters data/slm_genomic/clusters.parquet \
  --out data/slm_genomic/splits/variant.parquet --summary runs/slm_genomic/split_variant.json
```
Repeat per scheme (`text`, `laboratory`, `gene`, `disease`, `mmr`,
`temporal --cutoff 2025-01-01`, `functional --functional-variants …`).

```bash
vpdl-slm roles --records data/slm_genomic \
  --splits data/slm_genomic/splits/mmr.parquet data/slm_genomic/splits/temporal.parquet \
           data/slm_genomic/splits/gene.parquet \
  --out data/slm_genomic/pretrain_exclusions.json
```
Fix the evaluation sets **now**: this is what pretraining must not see.

```bash
vpdl-slm examples --records data/slm_genomic --split data/slm_genomic/splits/variant.parquet \
  --task classify --out data/slm_genomic/examples/classify_variant.parquet
```
```bash
vpdl-slm examples --records data/slm_genomic --split data/slm_genomic/splits/variant.parquet \
  --task evidence_type --out data/slm_genomic/examples/evidence_type_variant.parquet
```

```bash
vpdl-slm leakage --records data/slm_genomic \
  --examples data/slm_genomic/examples/classify_variant.parquet --scheme variant \
  --clusters data/slm_genomic/clusters.parquet --features consequence variant_type \
  --pretraining data/slm_genomic/pretrain_exclusions.json \
  --out docs/slm/leakage_variant.md
```
**Exit code 3 means critical leakage: stop.** Read the report, fix the cause,
re-run. Note the `gene_shortcut` line: that ROC-AUC is the bar a model must
beat to claim it reads evidence.

## Phase 6 — the baselines, before any neural arm

```bash
vpdl-slm baselines --examples data/slm_genomic/examples/classify_variant.parquet \
  --records data/slm_genomic --kinds majority tfidf_lr tfidf_svm structured_lr \
  --backend sklearn --out runs/slm_genomic/baselines
```

## Phase 7 — continued pretraining (days)

```bash
vpdl-slm pretrain-corpus --kb data/kb --pubmed data/raw/pubmed --records data/slm_genomic \
  --exclusions data/slm_genomic/pretrain_exclusions.json \
  --out data/slm_genomic/pretrain_corpus
```
Add `--include-narratives --split data/slm_genomic/splits/variant.parquet` only
if you want training-role narratives (conclusion-masked) in the corpus; it is
off by default.

```bash
vpdl-slm pretrain-pack --corpus data/slm_genomic/pretrain_corpus \
  --backbone pubmedbert --out data/slm_genomic/pretrain_tokens
```
```bash
vpdl-slm sizing --corpus-tokens <tokens from data_meta.json> \
  --examples <rows from Phase 5> --out runs/slm_genomic/sizing.json
```
```bash
vpdl-slm pretrain --config configs/slm/pretrain_dgx.toml --dry-run
```
→ check the `accelerator` block: `bf16_supported: true`,
`float32_matmul_precision: "high"`, `cudnn_benchmark: true`, and the memory
this machine actually has. Then measure the batch size rather than guessing it:

```bash
vpdl-slm autotune --config configs/slm/pretrain_dgx.toml --tokens-per-step 32768 --out runs/slm_genomic/autotune.json
```
→ it runs a few short benchmarks at increasing micro-batch, with and without
`torch.compile`, stops when memory runs out, and prints `put_in_config`. Copy
those three values into `configs/slm/pretrain_dgx.toml`. `--tokens-per-step`
keeps the optimiser step the same size while the micro-batch grows, so tuning
changes speed, not what is computed.

Sanity check on the number it reports: the from-scratch small model reached
30.7 TFLOPS on this machine against a measured 90.1 TFLOPS peak. If autotune's
best is under ~20% of peak, something other than the GPU is the bottleneck —
check `accelerator` in the dry run before spending days on the real run.

```bash
vpdl-slm pretrain --config configs/slm/pretrain_dgx.toml --benchmark 20
```
Confirms the chosen setting and projects the hours. Set `total_steps` from it
and from `sizing`, then:

```bash
tmux new -s slm-pretrain
```
```bash
vpdl-slm pretrain --config configs/slm/pretrain_dgx.toml
```
Detach with `Ctrl-b d`. To stop: `Ctrl-C` once — it finishes the step, saves
and exits. To continue: the same command. Checkpoints every 250 steps and at
least every 30 minutes; the last 3 are kept.

Watch it:
```bash
tail -f runs/slm_genomic/pretrain/log.jsonl
```
`loss` and `val_loss` falling steadily, `grad_norm` small. A non-finite loss
stops the run on purpose, with the last checkpoint intact.

## Phase 8 — fine-tuning and the ablations

```bash
vpdl-slm finetune --config configs/slm/exp005_pubmedbert.toml --dry-run
```
Then the arms, three seeds each (edit `seed`, or copy the config):

```bash
vpdl-slm finetune --config configs/slm/exp009_custom_slm.toml
```
```bash
vpdl-slm finetune --config configs/slm/exp010_genomic_pretrained.toml
```
…through `exp020_final.toml`. Each writes `metrics.json`,
`predictions.parquet`, `embeddings_<split>.npy`, a model file, and one line in
`runs/slm_genomic/registry.jsonl`.

Three seeds per arm, without copying configs (each gets its own run directory):

```bash
for s in 42 43 44; do vpdl-slm finetune --config configs/slm/exp010_genomic_pretrained.toml --seed $s; done
```

The configs ship with `batch_size = 16`, which is small for this machine.
Raise it once you know what fits — `--batch-size` overrides the config without
editing it:

```bash
vpdl-slm finetune --config configs/slm/exp010_genomic_pretrained.toml --batch-size 64
```

## Phase 9 — evaluation, calibration, VUS

Calibration is fitted inside each run (validation only) and reported before and
after. To re-score saved predictions, or to add the VUS reclassification
evaluation:

```bash
vpdl-slm evaluate --predictions runs/slm_genomic/exp020-final/predictions.parquet \
  --examples data/slm_genomic/examples/classify_variant.parquet \
  --vus-old data/slm_genomic_old/variants.parquet --vus-new data/slm_genomic/variants.parquet \
  --out runs/slm_genomic/exp020-final/evaluate.json
```
(`--vus-old` is a `variants.parquet` built from an archived release by running
Phase 3 against it into another directory.)

## Phase 10 — holdouts and MMR

```bash
vpdl-slm finetune --config configs/slm/exp021_gene_holdout.toml
```
```bash
vpdl-slm finetune --config configs/slm/exp023_temporal.toml
```
```bash
vpdl-slm finetune --config configs/slm/exp024_functional.toml
```
```bash
vpdl-slm finetune --config configs/slm/exp019_mmr_adapter.toml
```
```bash
vpdl-slm finetune --config configs/slm/exp025_mmr.toml
```

## Phase 11 — export for the future fusion

```bash
vpdl-slm embed --predictions runs/slm_genomic/exp020-final/predictions.parquet \
  --embeddings runs/slm_genomic/exp020-final/embeddings_test.npy \
  --examples data/slm_genomic/examples/classify_variant.parquet --split test \
  --model-version exp020-final --feature-version slm-genomic-prep/1 \
  --out runs/slm_genomic/export
```
This writes `SLMRepresentation` records. **Fusion itself is out of scope for
this phase and is not implemented.**

## What gets written

| Path | What | Committed? |
|---|---|---|
| `data/slm_genomic/*.parquet`, `manifest.json` | the tables | no (regenerable; manifest is small — commit it if useful) |
| `data/slm_genomic/{splits,examples,clusters,token_cache,pretrain_*}` | derived data | no |
| `runs/slm_genomic/**/metrics.json`, `registry.jsonl`, `*_stats.json`, `sizing.json` | the evidence for the paper | **yes** |
| `runs/slm_genomic/**/checkpoints`, `model/`, `*.npy`, `predictions.parquet` | large artefacts | no |
| `docs/slm/leakage_*.md`, `corpus_stats.md` | reports | **yes** |

## If something goes wrong

* `leakage` exits 3 — read the critical findings; they are the audit doing its
  job, not an obstacle to route around.
* "token files were packed with a different tokenizer" — re-run
  `pretrain-pack` with the backbone you are about to train.
* "was written with a different configuration" — a resume against changed
  settings or data; use a new `out_dir` or restore the settings.
* Loss became NaN — the run stopped with its last checkpoint intact; lower the
  learning rate and resume, or roll back to an earlier checkpoint.
* ERepo columns not found — the message prints the file's real header; set them
  and record the true names in GENOMIC_SLM_KNOWLEDGE_BASE.md.
