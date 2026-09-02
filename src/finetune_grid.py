"""Stage-2b ablation grid: cell identity, tiers, and output naming.

The grid answers a question ``docs/PAPER_DRAFT.md`` §6.12 asks but cannot
currently settle. That section varies freeze depth only, and compares the
result against a frozen *priors* probe -- but Stage 2b reads no prior
features at all, so the 6.5-point gap it measures confounds freeze depth
with feature set. Adding the ``branch`` axis makes freeze depth measurable
with the feature set held constant.

Cells are named by a deterministic slug so every artefact is
self-identifying. Before this existed, ``RUNLOG.md`` instructed the operator
to "move each ``esm_finetune_results_siamese_lopo.csv`` aside before the next
run" across a dozen cells -- an error-prone manual step in the middle of a
multi-day run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

#: ``esm`` reads the ESM feature block alone (Stage 2b as it shipped);
#: ``esm+priors`` additionally reads the leakage-safe prior columns that the
#: frozen Stage-2 probe already reads.
BRANCHES = ("esm", "esm+priors")


@dataclass(frozen=True)
class GridCell:
    """One configuration of the Stage-2b ablation grid."""

    branch: str
    n_unfrozen_layers: int
    pllr_mode: str
    seed: int = 42
    fusion: str = "concat"
    tier: str = ""

    def __post_init__(self) -> None:
        if self.branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}; got {self.branch!r}")

    def slug(self) -> str:
        """Deterministic, filesystem-safe identifier for this cell."""
        branch = "esmpri" if self.branch == "esm+priors" else "esm"
        nuf = {-1: "full", 0: "frozen"}.get(
            self.n_unfrozen_layers, f"last{self.n_unfrozen_layers}")
        parts = [branch, nuf, f"pllr-{self.pllr_mode}", f"seed{self.seed}"]
        if self.branch == "esm+priors":
            parts.insert(1, self.fusion)
        return "_".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {"branch": self.branch,
                "n_unfrozen_layers": self.n_unfrozen_layers,
                "pllr_mode": self.pllr_mode, "seed": self.seed,
                "fusion": self.fusion, "tier": self.tier, "slug": self.slug()}


def _tier(name: str, cells: Sequence[GridCell]) -> List[GridCell]:
    """Stamp *name* onto each cell (GridCell is frozen, so rebuild)."""
    return [GridCell(branch=c.branch, n_unfrozen_layers=c.n_unfrozen_layers,
                     pllr_mode=c.pllr_mode, seed=c.seed, fusion=c.fusion,
                     tier=name)
            for c in cells]


#: Tiers are ordered by scientific value so an interrupted run degrades
#: gracefully: tier 1 alone settles the paper's headline claim.
TIERS: Dict[str, List[GridCell]] = {
    # 1 -- the headline fair fight: does ESM add anything on top of priors?
    "1": _tier("1", [
        GridCell("esm+priors", nuf, "residual", seed)
        for nuf in (-1, 0) for seed in (42, 43, 44)
    ]),
    # 2 -- branch attribution: how much of any gain is priors vs backbone?
    "2": _tier("2", [
        GridCell("esm", -1, "residual", 42),
        GridCell("esm", 0, "residual", 42),
    ]),
    # 3 -- the PLLR axis, measured where it is clean: at the frozen floor the
    # backbone cannot relearn the term, so the on/off gap is attributable.
    "3": _tier("3", [
        GridCell("esm+priors", 0, "off", 42),
        GridCell("esm", 0, "off", 42),
        GridCell("esm+priors", 0, "concat", 42),
        GridCell("esm", 0, "concat", 42),
        GridCell("esm+priors", 0, "residual", 42, fusion="gatewave"),
    ]),
    # 4 -- freeze-depth middle ground.
    "4": _tier("4", [
        GridCell("esm+priors", 2, "residual", 42),
        GridCell("esm", 2, "residual", 42),
    ]),
    # 5 -- PLLR at full fine-tune. Confounded with the backbone relearning the
    # term during fine-tuning, which is exactly why tier 3 exists as well.
    "5": _tier("5", [GridCell("esm+priors", -1, "off", 42)]),
}


def output_tag(mode: str, eval_mode: str, cell_slug: str,
               holdout_gene: str | None = None) -> str:
    """Filename stem shared by the fine-tune script and the grid driver.

    Both sides must derive the name here rather than build their own. When
    they drifted -- the driver omitted the held-out gene the script puts in
    the name -- ``cell_is_complete`` looked for files that never existed, so
    every ``--eval holdout`` cell re-ran on restart instead of being skipped.
    """
    if eval_mode == "holdout":
        if not holdout_gene:
            raise ValueError("holdout_gene is required when eval_mode='holdout'")
        split = f"holdout_{holdout_gene}"
    else:
        split = eval_mode
    return f"{mode}_{split}_{cell_slug}"


def cells_for(tiers: Sequence[str]) -> List[GridCell]:
    """Cells for *tiers*, in tier order, de-duplicated by slug."""
    seen, out = set(), []
    for name in tiers:
        if name not in TIERS:
            raise ValueError(f"unknown tier {name!r}; expected {sorted(TIERS)}")
        for cell in TIERS[name]:
            if cell.slug() not in seen:
                seen.add(cell.slug())
                out.append(cell)
    return out


__all__ = ["BRANCHES", "GridCell", "TIERS", "cells_for", "output_tag"]
