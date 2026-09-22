# Small language model — running it on the DGX

Plan and reasoning: [PLAN.md](PLAN.md). Code: `vpdl/slm/`. Tests: `tests/slm/`.

Always do the **trial run** (5 PubMed files, a few minutes) before the full
run (all 1,334 files, days). The trial proves every stage works on real data;
the full run then only repeats it at scale.

## 0. Once: install and test

PyTorch runs some operations through Triton kernels, which need the Python
development headers to build on first use (without them the first training
step fails with "Python.h: No such file or directory"):
```bash
sudo apt install python3.12-dev
```
```bash
pip install -e ".[slm]"
```
```bash
pytest tests/slm -q
```
Expect **22 passed**. Among them: a run interrupted halfway and resumed must end
with exactly the same weights as one that never stopped.

## 1. Trial run (minutes)

Download 5 of the 1,334 PubMed files (~200 MB):
```bash
for n in 0001 0002 0003 0004 0005; do wget -c -P data/raw/pubmed https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n$n.xml.gz; done
```
Build the text (PubMed + the GeneReviews passages already in `data/kb`):
```bash
vpdl slm-corpus --pubmed data/raw/pubmed --kb data/kb
```
Train the vocabulary:
```bash
vpdl slm-tokenizer
```
Turn the text into token files:
```bash
vpdl slm-pack
```
Measure the real training speed (20 steps; saves nothing):
```bash
vpdl slm-train --size small --benchmark 20
```
It prints `tokens_per_s`, achieved `tflops`, and `projected_hours` for the
full small run. `torch.compile` is on by default (measured 2026-09-22: 34,700
vs 23,000 tokens/s, identical losses); if compilation fails on some machine,
add `--no-compile`.

## 2. Full run (days)

Download all 1,334 files (51.8 GB), each by its exact name, 8 at a time, from
inside the project folder. `-c` resumes a partly downloaded file; re-running
the command skips finished ones. NCBI publishes no connection limit we could
find; 8 is a moderate level for a public server. If lines with `429` or `503`
appear, NCBI is throttling: stop and re-run with `-P 4`. (A recursive
`wget -r` does not work: NCBI's robots.txt tells link-following tools to stay
out, and wget obeys it.)
```bash
cd ~/variant_pathogenicity_dl && seq -f "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n%04g.xml.gz" 1 1334 | xargs -n 1 -P 8 wget -c -nv --tries=5 -P data/raw/pubmed
```
Each finished file prints one line. Never run two download commands at once:
two `wget`s writing the same file corrupt it.

Measured speed (MB per minute; 51.8 GB takes ~9 minutes at 100 MB/s):
```bash
a=$(du -sm data/raw/pubmed | cut -f1); sleep 60; b=$(du -sm data/raw/pubmed | cut -f1); echo "$((b-a)) MB per minute"
```
A file cut short by a dropped connection is skipped by `slm-corpus` and listed
under `unreadable_files` in `data/slm/corpus/stats.json`: re-run the loop, then
rebuild.

Rebuild corpus, vocabulary and token files on everything (same three commands
as the trial; the corpus step replaces the trial corpus):
```bash
vpdl slm-corpus --pubmed data/raw/pubmed --kb data/kb
```
```bash
vpdl slm-tokenizer
```
```bash
vpdl slm-pack
```
Train the small model (~1 day). Run it inside `tmux` so closing the terminal
does not stop it:
```bash
tmux new -s slm
```
```bash
vpdl slm-train --size small
```
Detach with `Ctrl-b d`; reattach with `tmux attach -t slm`.

**Checkpoints.** A full checkpoint is saved every 250 steps and at least every
30 minutes; the last 3 are kept in `runs/slm/<size>/checkpoints/`.

- **To stop:** press `Ctrl-C` once (or `kill`, or shut the machine down). It
  finishes the current step, saves a checkpoint, and exits — nothing is lost.
  A second `Ctrl-C` aborts immediately without saving.
- **To continue:** run the same `slm-train` command. It resumes from the newest
  checkpoint, exactly as if it had never stopped.
- **After a crash or power cut:** same command; at most ~30 minutes of work is lost.
- **To roll back** (e.g. the loss spiked): pick an earlier checkpoint and
```bash
vpdl slm-train --size small --resume-from runs/slm/small/checkpoints/step-0001000.pt
```
  Newer checkpoints are moved to `checkpoints/abandoned-<time>/`, not deleted.
- **Milestones:** every 1,000 steps the weights alone are saved to
  `milestones/step-NNNNNNN/` — loadable for evaluation while training goes on.

Disk: a full small checkpoint is ~1.3 GB (weights + optimiser state), a
milestone ~0.45 GB; the small run needs ~6 GB, the medium ~35 GB.

Only if the small run is healthy, the medium model (~5–6 days):
```bash
vpdl slm-train --size medium
```

## Is it healthy?

Watch the log:
```bash
tail -f runs/slm/small/log.jsonl
```
- `loss` and `val_loss` fall steadily. Starting near ~10.4 (random guessing
  over 32,000 tokens), the small model should reach roughly 3 or below; if
  `val_loss` rises while `loss` keeps falling, it is memorising.
- `grad_norm` stays small (mostly below ~1–2). Spikes that grow are the
  warning before divergence.
- A `nan` loss stops the run by design. Resume with a lower rate:
  `vpdl slm-train --size small --lr 3e-4 --out runs/slm/small-lr3e-4`.
- `tflops` near the benchmark's figure means the GPU is doing the work.

## What gets written

| Path | What |
|---|---|
| `data/raw/pubmed/` | PubMed baseline files (not committed) |
| `data/slm/corpus/` | text shards + `stats.json` (every document kept or dropped, and why) |
| `data/slm/tokenizer.json`, `tokenizer_meta.json` | the vocabulary, and how it splits clinical text |
| `data/slm/train.bin`, `val.bin`, `data_meta.json` | token files and their counts |
| `runs/slm/<size>/run.json` | settings, parameter count, GPU, git commit, data counts |
| `runs/slm/<size>/log.jsonl` | the training record |
| `runs/slm/<size>/checkpoints/` | last 3 full checkpoints (not committed) |
| `runs/slm/<size>/milestones/` | weights every 1,000 steps (not committed) |
| `runs/slm/<size>/final/` | the trained model in Hugging Face format (not committed) |

`run.json` and `log.jsonl` are small and **are** the evidence for the paper:
commit them.
