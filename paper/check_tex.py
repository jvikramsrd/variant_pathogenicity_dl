"""Static checks on paper/main.tex for machines without a TeX installation.

    python paper/check_tex.py

Catches the failures that usually break a build or silently corrupt a paper: a number
macro used but never generated, a citation key missing from references.bib, a \\ref to
an undefined label, a missing \\input or figure file, unbalanced braces or environments,
and characters outside ASCII in the sources. It does not replace compiling the paper.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent

# Commands the manuscript may use that are not generated macros.
LATEX = set("""
documentclass usepackage input newcommand renewcommand title author date begin end maketitle
noindent textbf textit emph url citep cite ref label caption centering includegraphics
resizebox textwidth section subsection appendix clearpage setcounter arabic thetable
thefigure bibliographystyle bibliography item small todo gene log geq times pm Delta
toprule midrule bottomrule cmidrule multicolumn endfirsthead endhead textcolor ldots
texttt thanks footnote hline linewidth vspace hspace par ne leq alpha beta mathrm
""".split())


def expand(path: Path, seen: list[Path]) -> str:
    text = path.read_text(encoding="utf-8")
    seen.append(path)

    def include(match: re.Match) -> str:
        target = PAPER / match.group(1)
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        if not target.exists():
            raise SystemExit(f"\\input target missing: {target}")
        return expand(target, seen)

    return re.sub(r"\\input\{([^}]+)\}", include, text)


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def main() -> int:
    problems: list[str] = []
    files: list[Path] = []
    source = strip_comments(expand(PAPER / "main.tex", files))

    for path in files + [PAPER / "references.bib"]:
        raw = path.read_bytes()
        bad = sorted({ch for ch in raw.decode("utf-8") if ord(ch) > 127})
        if bad:
            problems.append(f"{path.name}: non-ASCII characters {bad}")

    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", source))
    used = set(re.findall(r"\\([A-Za-z]+)", source))
    undefined = sorted(used - defined - LATEX)
    if undefined:
        problems.append(f"undefined commands: {undefined}")

    bib = (PAPER / "references.bib").read_text(encoding="ascii")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", source) for k in group.split(",")}
    if missing := sorted(cited - keys):
        problems.append(f"citation keys not in references.bib: {missing}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", source))
    if missing := sorted(refs - labels):
        problems.append(f"\\ref to undefined labels: {missing}")
    if duplicate := sorted({l for l in labels if len(re.findall(r"\\label\{" + re.escape(l) + r"\}", source)) > 1}):
        problems.append(f"labels defined twice: {duplicate}")

    for figure in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        if not (PAPER / figure).exists():
            problems.append(f"figure missing: {figure}")

    depth = 0
    for index, ch in enumerate(re.sub(r"\\[{}]", "", source)):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            problems.append(f"unbalanced '}}' near character {index}")
            break
    if depth > 0:
        problems.append(f"{depth} unclosed '{{'")

    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", source):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            problems.append(f"environment mismatch at \\end{{{name}}}")
            break
    if stack:
        problems.append(f"unclosed environments: {stack}")

    body = source.split("\\begin{document}", 1)[1]
    text_only = re.sub(r"\$[^$]*\$", "", body)
    text_only = re.sub(r"\\(url|label|ref|cite[pt]?|includegraphics|input|texttt)(\[[^]]*\])?\{[^}]*\}",
                       "", text_only)
    if bare := re.findall(r"[^\\]_", text_only):
        problems.append(f"{len(bare)} unescaped '_' outside math, e.g. {bare[:3]}")
    stray_amp = [line for line in text_only.splitlines()
                 if re.search(r"(?<!\\)&", line) and not re.search(r"\\\\\s*$|&.*&", line)]
    if stray_amp:
        problems.append(f"possible stray '&' outside a table: {stray_amp[:2]}")

    todos = re.findall(r"\\todo\{([^}]+)\}", body)
    for problem in problems:
        print("PROBLEM:", problem)
    print(f"checked {len(files)} files, {len(used & defined)} number macros, {len(cited)} citations, "
          f"{len(refs)} references; {len(todos)} author TODOs: {todos}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
