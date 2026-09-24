# Paper: feature sources, label sources and architectures in the MMR genes

`main.tex` is the manuscript. Every number in it is a macro from `generated/numbers.tex`,
and every table and figure is a file in `generated/`. All of them are written by one script
from the saved DGX runs, so nothing is typed by hand.

## Rebuild the numbers, tables and figures (about 3 minutes, CPU)

```bash
python paper/analysis.py
```

Inputs: `runs/dl/{main,ablation_features,ablation_labels}/`, `results/dl/` and
`data/built/canonical_full.csv` (+ `canonical_zs.csv` for the ESM baselines). The script stops
and writes nothing if any of these checks fails:

- `results/dl/checks.json` is not citable;
- the dataset hash differs from the one the runs recorded;
- a recomputed per-gene AUROC or MCC differs from `results/dl/cells.csv`;
- two arms scored different variants;
- a reconstructed feature list differs from the schema hash a run recorded.

`generated/MANIFEST.json` records the input hashes and the hash of every output.

## Check the LaTeX without a TeX installation

```bash
python paper/check_tex.py
```

This checks that every macro is defined, every citation key exists, every `\ref` has a
label, every `\input` and figure file exists, and that braces and environments balance. It
does not replace compiling.

## Compile

On Overleaf, upload `main.tex`, `references.bib` and the `generated/` folder, and compile
with pdfLaTeX. On Linux (for example the DGX):

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-science latexmk
```

```bash
cd paper && latexmk -pdf main.tex
```

## Still to fill in by the author

These are marked in red in the PDF (`\todo{...}`): affiliation, correspondence e-mail,
archive DOI (for example Zenodo), acknowledgements, funding, and conflict of interest.

## What is pre-specified and what is post hoc

Pre-specified: the design (`docs/dl/ABLATION.md`, `scripts/dl_ablation.sh`), models, seeds,
evaluation, scoreable-gene rule, AUROC as the cross-arm comparison, and the zero-training
baselines.

Post hoc, and labelled as such in the paper: the rank-mean baseline, the gnomAD-observed
subset, and the B-1 sensitivity analysis.

## Tests

```bash
python -m pytest tests/paper
```
