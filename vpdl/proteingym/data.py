"""Read ProteinGym's per-assay fold files."""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FOLD_SCHEMES", "AMINO_ACIDS", "Assay", "parse_assay", "iter_assays"]

FOLD_SCHEMES = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Assay:
    dms_id: str
    frame: pd.DataFrame      # mutant, wt, position (1-based), mut, DMS_score, fold_*
    skipped: int             # rows that were not a single standard-residue substitution


def parse_assay(dms_id: str, frame: pd.DataFrame) -> Assay:
    """Keep single substitutions between standard residues; check the numbering.

    The numbering check matters more than it looks. If positions were read one
    off, every feature would land on the wrong residue and the model would still
    train and still report a Spearman — just a meaningless one. So each row's
    mutant residue is confirmed against its own ``mutated_sequence``, and a
    systematic mismatch stops the run.
    """
    parsed = frame["mutant"].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
    ok = (parsed.notna().all(axis=1)
          & parsed[0].isin(list(AMINO_ACIDS))
          & parsed[2].isin(list(AMINO_ACIDS)))

    out = frame.loc[ok].copy()
    out["wt"] = parsed.loc[ok, 0].to_numpy()
    out["position"] = parsed.loc[ok, 1].astype(int).to_numpy()
    out["mut"] = parsed.loc[ok, 2].to_numpy()
    out = out.reset_index(drop=True)

    if "mutated_sequence" in out.columns and len(out):
        observed = [
            sequence[position - 1] if 0 < position <= len(sequence) else None
            for sequence, position in zip(out["mutated_sequence"], out["position"])
        ]
        mismatched = sum(o != m for o, m in zip(observed, out["mut"]))
        if mismatched:
            raise ValueError(
                f"{dms_id}: {mismatched} of {len(out)} mutants disagree with their "
                "own mutated_sequence at the stated position — the numbering is "
                "off. Refusing to fit features to the wrong residues."
            )

    return Assay(dms_id=dms_id, frame=out, skipped=int((~ok).sum()))


def iter_assays(
    zip_path: Path | str,
    only: Iterable[str] | None = None,
) -> Iterator[Assay]:
    """Yield every assay in ``cv_folds_singles_substitutions.zip``, by DMS_id order."""
    wanted = set(only) if only else None
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".csv"))
        for name in names:
            dms_id = Path(name).stem
            if wanted is not None and dms_id not in wanted:
                continue
            with archive.open(name) as handle:
                frame = pd.read_csv(handle)
            yield parse_assay(dms_id, frame)
