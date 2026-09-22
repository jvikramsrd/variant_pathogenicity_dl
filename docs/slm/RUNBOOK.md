# Small language model — running it on the DGX

Plan and reasoning: [PLAN.md](PLAN.md). Code: `vpdl/slm/`. Tests: `tests/slm/`.

Always do the **trial run** (5 PubMed files, a few minutes) before the full
run (all 1,334 files, days). The trial proves every stage works on real data;
the full run then only repeats it at scale.

## 0. Once: install and test

```bash
pip install -e ".[slm]"
```
```bash
pytest tests/slm -q
```
Expect **13 passed**. Among them: a run interrupted halfway and resumed must end
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
full small run. Try once more with `--compile`; keep whichever is faster.

## 2. Full run (days)

Download all 1,334 files (51.8 GB; `-c` resumes if interrupted):
```bash
wget -c -r -np -nd -A "pubmed26n*.xml.gz" -P data/raw/pubmed https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
```
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
Detach with `Ctrl-b d`; reattach with `tmux attach -t slm`. If anything stops
it, run the same `slm-train` command again: it resumes from the last
checkpoint (every 250 steps) exactly where it left off.

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
| `runs/slm/<size>/checkpoint.pt` | latest checkpoint (not committed) |
| `runs/slm/<size>/final/` | the trained model in Hugging Face format (not committed) |

`run.json` and `log.jsonl` are small and **are** the evidence for the paper:
commit them.
