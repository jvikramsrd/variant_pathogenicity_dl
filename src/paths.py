"""Turn a missing generated artifact into an actionable error.

Every dataset/model file this project reads from disk -- panel JSONs, the
audited extended-dataset CSVs, pretrain/fine-tune checkpoints -- is a build
artifact, not something checked into git (see .gitignore: data/raw/,
data/processed/**, data/mmr/**, *.pt are all excluded). On a fresh clone or a
new machine, reading one of these before it has been built raises a bare
FileNotFoundError deep inside pandas/json/torch with no hint about which
command produces it. ``require_exists`` front-loads that check with a message
naming the fix.
"""
from __future__ import annotations

from pathlib import Path


def require_exists(path: Path, generator_hint: str) -> Path:
    """Return ``path`` if it exists, else raise a FileNotFoundError naming how to build it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. It is a generated artifact (gitignored, "
            f"not part of the repo) -- run `{generator_hint}` first to build it.")
    return path
