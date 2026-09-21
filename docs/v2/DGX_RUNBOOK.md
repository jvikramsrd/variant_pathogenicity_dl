# DGX Spark runbook

From an empty DGX Spark to the first comparison table. Written 2026-09-21.

Each phase ends with a **check**. If a check fails, stop there and send the
output back rather than continuing — every phase after it depends on it, and
the whole history of this project is plausible numbers built on a broken step.

The expected values in the checks come from v1's build manifests. v1 is the
only thing that has ever run these sources end to end, so it is the oracle.

---

## The experiment in one paragraph

One table is built from all sources. Each **arm** trains on a different subset
of label sources — ClinVar alone, ClinVar + DMS, DMS alone — and **every arm is
scored on the same held-out ClinVar variants**. That is what makes the arms
comparable: they differ in what they learned from, never in what they were
tested on. Evaluation is leave-one-gene-out over MLH1 / MSH2 / MSH6 / PMS2.
Compare arms on **ROC-AUC**: each arm picks its threshold on its own training
labels, so MCC partly measures threshold transfer, while ROC-AUC needs no
threshold.

---

## Phase 0 — ship the latest code (on the Windows box)

`v2/rebuild` is already on GitHub (`2354f38 v2_progress`), but that commit
predates the 2026-09-21 fixes — without them the build drops every feature from
feature-only sources, scores each arm on a different test set, and imputes
PM2 away. Push them before cloning on the DGX.

```bash
git add -A -- vpdl tests/regression docs pyproject.toml .gitignore
git status
```

**Check:** only `vpdl/`, `tests/regression/`, `docs/v2/`, `pyproject.toml` and
`.gitignore` appear. Nothing under `src/`, `scripts/` or `data/`.

```bash
git commit -m "v2: fix assembly feature loss, one-table arm design, gnomAD PM2; add DGX runbook"
git push
```

---

## Phase 1 — confirm the machine

```bash
nvidia-smi
uname -m
python3 --version
free -g
df -h ~
```

**Check:** `uname -m` prints `aarch64`. `nvidia-smi` shows the GB10. Python is
3.11 or newer. Leave at least ~20 GB free for downloads and outputs.

---

## Phase 2 — get the code

```bash
git clone https://github.com/jvikramsrd/variant_pathogenicity_dl.git
cd variant_pathogenicity_dl
git checkout v2/rebuild
git config user.name "Jyothi Vikrama Simha Reddy Dirisinapu"
git config user.email "jvikramsrd@gmail.com"
```

If the repository is private, authenticate first (`gh auth login`, or add an
SSH key and clone the `git@github.com:` URL).

**Check:** `ls vpdl` lists `assemble.py`, `cli.py`, `sources/`, `models/`.

---

## Phase 3 — environment, deliberately without PyTorch

Torch is not needed for the headline arm (gradient boosting), and ARM64 CUDA
torch is the one step most likely to cost a day. It comes in Phase 9, after
there is already a result.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip unzip wget
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,gbm]"
```

**Check:** `vpdl --help` lists five subcommands: `device`, `sources`, `build`,
`train`, `compare`.

---

## Phase 4 — prove the code before trusting it

```bash
python -m pytest tests/regression -v
```

**Check:** everything passes **except one expected failure** —
`test_every_split_is_seeded_including_the_first`, which constructs an MLP and
needs torch. Any other failure: stop and send the output. Some failures may be
the test's fault rather than the code's; that still gets fixed before any data
is touched, because this suite is what stands between you and every bug v1
shipped.

Record what the hardware actually reports — the code detects rather than
assumes, and this file is where the detection gets checked against reality:

```bash
{ echo '# DGX Spark — as reported'; echo; echo '```'; nvidia-smi; uname -a; vpdl device; echo '```'; } > docs/v2/HARDWARE.md
```

---

## Phase 5 — download the sources

```bash
mkdir -p data/raw && cd data/raw
wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
wget -c https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz
wget -c https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
sha256sum * > SHA256SUMS
date -u +%Y-%m-%dT%H:%MZ > DOWNLOADED_AT
cd ../..
```

ClinVar changes weekly and v1 never recorded which snapshot it used.
`SHA256SUMS` and `DOWNLOADED_AT` exist so that can't happen again.

If the ProteinGym URL fails, the same archive is on Zenodo record `15293562`.

**Check:** the MSH2 assay is inside the ProteinGym archive:

```bash
unzip -l data/raw/DMS_ProteinGym_substitutions.zip | grep -i msh2
```

---

## Phase 6 — build ONE table

```bash
vpdl build --sources clinvar pg_dms alphamissense gnomad --out data/built/mmr.csv
```

This also fetches UniProt sequences and gnomAD over the network, so it takes a
few minutes. It **fails loudly** if any of the four genes is missing — the guard
added after v1 once trained sixteen cells on a silently three-gene table.

Always include `alphamissense`: it is the independent anchor the build uses to
check each label source's orientation, which is how v1's 185,000 inverted
ProteinGym labels were eventually caught.

**Check** the JSON report the build prints:

- `orientation_checks` has entries for `clinvar` and `pg_dms`, both well above
  0.5. A value below 0.5 aborts the build — that is an inverted label mapping.
- `dropped_per_source.pg_dms.sequence_mismatch` may be non-zero. That is
  expected: assay coordinates that disagree with the canonical UniProt sequence
  are dropped, not repositioned.

---

## Phase 7 — oracle check against v1

```bash
python - <<'EOF'
import pandas as pd
t = pd.read_csv("data/built/mmr.csv", low_memory=False)
print("rows:", len(t))
for column in [c for c in t.columns if c.startswith("label__")]:
    labelled = t[t[column].notna()]
    print(f"\n{column}: {len(labelled)} labelled")
    print(labelled.groupby("gene")[column].agg(n="count", pathogenic="sum").to_string())
print("\nfeature coverage on ClinVar-labelled rows (fraction non-null):")
print(t[t["label__clinvar"].notna()].filter(like="feature_").notna().mean().round(3).to_string())
EOF
```

**Check** against v1's manifest:

| quantity | v1 | expect now |
|---|---|---|
| total rows | 74,328 | **exactly 74,328** — every substitution in the four canonical proteins |
| `label__clinvar` labelled | 464 | roughly 440–470; ClinVar drifts between snapshots |
| genes with ClinVar labels | 4 | **all four**, PMS2 small |
| `label__pg_dms` | 16,420, all MSH2 | ~16,000, **MSH2 only** |
| `feature_alphamissense_score` coverage | — | ~1.0 |
| `feature_gnomad_observed` | 6,492 of 74,328 observed | the fraction equal to 1 should be small; every other variant is PM2 |
| every gnomAD column coverage | — | **1.0** — absence is filled as PM2 evidence, never left NaN |

**Feature coverage near zero on the ClinVar rows means the assembly is dropping
features** (regression landmine L16). Stop.

---

## Phase 8 — the first real number

One arm, one model, one seed, fast bootstrap. The point is to see the pipeline
run end to end before spending time on a grid.

```bash
vpdl train --data data/built/mmr.csv --sources clinvar pg_dms alphamissense gnomad \
  --train-sources clinvar --model gbm --seeds 42 --n-bootstrap 1000 --out runs/smoke
```

**Read the log top to bottom.** Before any metric it prints the dataset hash
and, per gene, how many training and evaluation labels it has. Check those
first — a metric computed on the wrong table looks exactly like one computed on
the right one.

**Sanity band:** v1's closest comparator (curated features, no ESM) reached mean
ROC-AUC 0.951 over the three scoreable genes (MLH1 0.965, MSH2 0.908, MSH6
0.979). v2 has fewer prior features and trains on ClinVar alone, so expect
somewhat lower, roughly **0.85–0.97**.

- **Below ~0.7:** a bug. Check score orientation and feature coverage.
- **Above ~0.99:** almost certainly a leak. v1's fake 0.9987 was a feature that
  was the label in disguise.

Either way, stop before the grid.

---

## Phase 9 — the main experiment

Three arms, three seeds, one table.

```bash
for arm in "clinvar" "clinvar pg_dms" "pg_dms"; do
  vpdl train --data data/built/mmr.csv --sources clinvar pg_dms alphamissense gnomad \
    --train-sources $arm --model gbm --seeds 42 43 44 --out runs/main
done
vpdl compare --runs runs/main
```

`$arm` is unquoted on purpose so `"clinvar pg_dms"` becomes two arguments.

What to expect:

- The **DMS-only arm skips MSH2** — every DMS label is MSH2, so holding it out
  leaves nothing to train on. It is recorded in `genes_skipped`, not hidden. On
  MLH1, MSH6 and PMS2 that arm is a genuine *cross-protein transfer* test: a
  model trained only on an MSH2 functional assay, scored on clinical labels for
  a gene it never saw.
- Because that arm has one fewer fold, **compare arms per gene**, not only on
  the headline mean.
- `compare` refuses to pool if the arms don't share a table and a test set. With
  one build that holds by construction.

---

## Phase 10 — PyTorch, then the MLP and BiLSTM arms

Only after Phase 9 has produced a table.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x = torch.randn(1024, 1024, device='cuda'); print('matmul ok', float((x @ x).sum()))"
```

**Check:** CUDA is available, the device name is the GB10, the matmul runs, and
there is **no warning that the GPU's compute capability is unsupported** by the
installed build. If there is one, or CUDA is unavailable, use NVIDIA's
container instead — it is built for this GPU:

```bash
docker run --gpus all -it --rm --ipc=host -v "$PWD":/workspace -w /workspace nvcr.io/nvidia/pytorch:<TAG>-py3
pip install -e ".[dev,gbm]"
```

Take `<TAG>` as the newest `YY.MM-py3` listed on the NGC PyTorch catalogue page.
Whichever route works, add the verification output to `docs/v2/HARDWARE.md`.

Then the suite should be fully green, and the other two model arms can run:

```bash
python -m pytest tests/regression -v
for arm in "clinvar" "clinvar pg_dms" "pg_dms"; do
  for model in mlp bilstm; do
    vpdl train --data data/built/mmr.csv --sources clinvar pg_dms alphamissense gnomad \
      --train-sources $arm --model $model --seeds 42 43 44 --out runs/main
  done
done
vpdl compare --runs runs/main
```

The BiLSTM is a baseline to beat, not a contender — see `vpdl/models/bilstm.py`
for why. If it wins, check for leakage before believing it.

### Optional: feature-family ablation

```bash
vpdl train --data data/built/mmr.csv --sources clinvar pg_dms alphamissense gnomad \
  --train-sources clinvar --model gbm --seeds 42 43 44 \
  --drop-groups gnomad prior_scores --out runs/main
```

Dropping `gnomad` alone is **refused on purpose**: AlphaMissense was trained on
population data, so removing gnomAD while it stays in removes only the legible
copy of the signal. Drop both and report it as a joint bound.

---

## Phase 11 — send the results back

```bash
git add runs/main data/built/mmr.manifest.json docs/v2/HARDWARE.md
git add -f data/raw/SHA256SUMS data/raw/DOWNLOADED_AT
git status
```

**Check:** only additions. A `deleted:` line means something that exists on the
branch is missing on this machine — the exact way v1 once lost 48 result files
in a single push. Do not commit until you know why.

```bash
git commit -m "results: first v2 grid on DGX Spark"
git push
```

`data/built/mmr.csv` itself is deliberately not committed: it regenerates from
the raw files, and its manifest records the hash that proves which table the
results came from.

---

## ProteinGym benchmark — each dataset reproduced against its published result

The study: train a model on each of ProteinGym's 217 substitution assays on its
own, check its score against the number ProteinGym publishes for that assay,
then train one model on all of them combined and compare, assay by assay.

Step 1 is reproduction, with ProteinGym's simplest published baseline
("One-Hot Encodings"). It runs on CPU in minutes. Until our numbers match
theirs, nothing we report about combining datasets should be trusted.

**Download** (13 MB of data plus the published results table):
```bash
cd data/raw
wget -c https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip
wget -O pg_supervised_spearman_fold_random_5.csv https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv
cd ../..
```

**One assay first** — the MSH2 assay, published One-Hot score 0.513:
```bash
vpdl pg-reproduce --only MSH2_HUMAN_Jia_2020
```

**Then all 217:**
```bash
vpdl pg-reproduce
```

**What "reproduced" means here.** Same model class, same folds, same metric;
the optimiser differs (closed-form ridge here, AdamW for 10,000 steps in
ProteinGym), so exact equality is not expected. Most assays within ~0.05 and a
correlation across assays above ~0.9 means the pipeline reproduces ProteinGym.
A systematic gap means a protocol difference to find before going further.

---

## What not to do

- **Don't build separate tables per arm.** Separate tables mean separate test
  sets, and then the arms are not comparable — `compare` will refuse them.
- **Don't read a single-seed difference.** v1 measured MCC standard deviation up
  to 0.056 across seeds of one arm.
- **Don't drop a source from a build to "try without it".** Use
  `--train-sources` to change what an arm learns from; keep the table fixed.
- **Don't trust a number whose log you didn't read** down to the per-gene counts.
