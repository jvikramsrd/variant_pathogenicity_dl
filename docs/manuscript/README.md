# Manuscript source

LaTeX source for the *Bioinformatics* (Oxford University Press) submission.
`docs/MANUSCRIPT.md` remains the readable Markdown version; this directory is
the submission artifact. The two must be kept in step — if you change a number
in one, change it in the other.

## Files

| File | What it is |
|---|---|
| `main.tex` | The manuscript |
| `references.bib` | Bibliography, transcribed from the verified list in `../PAPER.md` §11 (see below) |

## Building

```bash
latexmk -pdf main.tex
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**`bioinfo.cls` is not in this repository.** It is distributed by OUP with their
author templates and is not redistributable here. `main.tex` detects this: with
the class present it typesets in the journal's layout; without it, the preamble
falls back to `article` and defines no-op stand-ins for the class-specific
macros (`\firstpage`, `\access`, `\corresp`, `\history`, `\editor`, …). The
fallback builds and reads correctly but is **not** the journal's layout — use it
for review, and drop `bioinfo.cls` alongside `main.tex` before generating the
submission PDF.

Get the class from the *Bioinformatics* author guidelines page.

## Conventions in this source

- **`\todo{...}` renders in bold red.** That is deliberate. A placeholder that
  reads like ordinary prose is how an unverified number reaches a reviewer.
  Every one is enumerated in `../../MISSING_EVIDENCE.md`. **Do not replace a
  `\todo` with a plausible value** — replace it with a measured one, or leave it.
- **`\gene{}`** italicises gene symbols. Protein products are not italicised.
- **`\num{}`** (siunitx) formats counts with thin-space grouping.
- Tables use `booktabs` and are `table*` (full width, two-column layout).

## Abstract structure

*Bioinformatics* requires **Motivation / Results / Availability and
implementation / Contact / Supplementary information** — not the
Background/Methods/Results/Conclusions structure used in the Markdown draft.
`main.tex` follows the journal's structure; the Markdown version does not. This
is the one place the two documents legitimately differ.

## Before submission

1. Complete `references.bib`. Entries are transcribed from the list in
   `../PAPER.md` §11, headed "Verified and safe to cite as listed", plus the
   citation blocks in `../DATASETS.md` — not from recollection. Two things are
   still owed, both stated in the file's header comment: **author lists are
   truncated** (`Richards, S. and others` is what the repository verifies;
   *Bioinformatics* wants them in full) and **DOIs are absent**. Three entries
   — `uniprot`, `interpro`, `mane` — remain full `TODO` placeholders because no
   verified citation for them exists anywhere in the repository. `grep -c TODO
   references.bib` counts what is left. The predictors listed under a dagger in
   `../PAPER.md` (VariPred, ProPath, the CSBJ token classifier, MVmamba, the
   CIMRA/PMS2 OddsPath source) are deliberately absent; verify before citing
   any of them. Every `\cite` key in `main.tex` resolves to an entry here —
   `bibtex` warns on any that stops doing so.
2. Resolve every `\todo`. `grep -c 'todo{' main.tex` gives the count.
3. Add figures. None exist yet; `MISSING_EVIDENCE.md` item 10 lists all six with
   captions, required input artifacts and output paths.
4. Check the length. *Bioinformatics* Original Papers run to about seven pages;
   this draft will need trimming once figures are added, and the per-gene and
   feature-definition tables should move to Supplementary.
5. Re-verify the numbers against artifacts after any re-run. The numbers in
   §3 were cross-checked against the source files at the time of writing; a new
   dataset build or grid run invalidates them.
