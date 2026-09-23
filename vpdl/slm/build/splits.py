"""Split framework: nine ways to hold data out, each with a stated guarantee.

Every scheme returns one row per DOCUMENT with ``split`` in
{train, val, test, excluded} (the MMR scheme adds its own names, below), and
every scheme except ``random`` keeps a VARIANT GROUP on one side. A variant
group joins documents that share any of: VariationID, GRCh38 genomic change
(one change named on two transcripts), or gene + protein change (two
nucleotide changes giving one substitution). Without that, "variant-isolated"
would still let p.Arg123His from c.367C>T train a model tested on c.368G>A.

    random        per document — the optimistic baseline; variants straddle folds (A)
    variant       variant groups (B)
    text          variant groups joined through near-duplicate text clusters (C)
    laboratory    submitters held out; test variants' training-lab documents excluded (C')
    gene          genes held out (D); `reserve_genes` (MMR) never enter the gene test
    disease       specific-disease groups held out (E); catch-all / umbrella conditions
                  never define a group, their documents only train
    temporal      train < cutoff <= test (F). mode "novel": test variants have no document
                  before the cutoff; "reinterpretation": test = post-cutoff documents of
                  variants already interpreted before it. Undated documents are excluded.
    functional    variants with independent functional data are test only (G)
    mmr           non-MMR train/val/broad_test; MMR split into mmr_train / mmr_val (for an
                  adapter) and test (H). Zero-shot transfer trains on `train` only.

Assignment is by a salted SHA-1 of the group key, so it is identical on every
machine and in every process, and changes only when the seed does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.slm.clinvar_text import specific_condition
from vpdl.slm.text.dedup import UnionFind
from vpdl.slm.variants import MMR_GENES

__all__ = ["SCHEMES", "SplitConfig", "split_frame", "variant_groups", "make_split", "split_summary",
           "SplitError", "bucket"]

SCHEMES = ("random", "variant", "text", "laboratory", "gene", "disease", "temporal",
           "functional", "mmr")


class SplitError(ValueError):
    pass


@dataclass
class SplitConfig:
    scheme: str
    seed: int = 13
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    cutoff: str | None = None                 # temporal: ISO date
    val_cutoff: str | None = None             # temporal: default cutoff minus one year
    temporal_mode: str = "novel"              # novel | reinterpretation
    reserve_genes: tuple[str, ...] = MMR_GENES
    mmr_genes: tuple[str, ...] = MMR_GENES
    mmr_adapt_fractions: tuple[float, float] = (0.6, 0.2)   # mmr_train, mmr_val; rest test
    functional_variants: tuple[str, ...] = ()  # protein_variant_ids with holdout assay data
    max_group_share: float = 0.3              # a group bigger than this cannot be split sensibly
    use_template_clusters: bool = False       # text scheme: also join template families

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bucket(key: str, seed: int) -> float:
    """Deterministic uniform [0, 1) from a key."""
    digest = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:13], 16) / float(1 << 52)


def split_frame(tables: Mapping[str, pd.DataFrame], clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Documents with the variant fields every scheme needs."""
    documents = tables["documents"]
    variants = tables["variants"][["variant_id", "genomic_key", "hgvs_p", "consequence",
                                   "protein_variant_id", "gene"]].rename(columns={"gene": "variant_gene"})
    frame = documents[["document_id", "variant_id", "gene", "submitter", "date", "disease_ids",
                       "disease_names", "label5", "label_category", "source_id", "restricted"]].merge(
        variants, on="variant_id", how="left")
    frame["gene"] = frame["gene"].fillna("")
    frame["gene"] = frame["gene"].where(frame["gene"] != "", frame["variant_gene"].fillna(""))
    if clusters is not None:
        frame = frame.merge(clusters, on="document_id", how="left")
    return frame.reset_index(drop=True)


def _connected(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Row groups: rows sharing a non-missing value in ANY of `columns` are joined."""
    finder = UnionFind(len(frame))
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        present = values.notna() & values.astype(str).ne("") & values.astype(str).ne("None")
        first: dict[Any, int] = {}
        for row, value in zip(np.flatnonzero(present.to_numpy()), values[present]):
            key = value if not isinstance(value, list) else tuple(value)
            if key in first:
                finder.union(first[key], int(row))
            else:
                first[key] = int(row)
    return finder.labels()


def variant_groups(frame: pd.DataFrame) -> np.ndarray:
    protein = np.where(frame["hgvs_p"].notna() & frame["hgvs_p"].astype(str).str.len().gt(3),
                       frame["gene"].astype(str) + "|" + frame["hgvs_p"].astype(str), None)
    work = frame.assign(_protein=protein)
    return _connected(work, ["variant_id", "genomic_key", "_protein", "protein_variant_id"])


def _assign_by_group(groups: np.ndarray, keys: Sequence[str], seed: int, val: float,
                     test: float) -> np.ndarray:
    group_key = {}
    for group, key in zip(groups, keys):
        group_key.setdefault(group, key)
    draw = {g: bucket(str(k), seed) for g, k in group_key.items()}
    out = np.empty(len(groups), dtype=object)
    for i, group in enumerate(groups):
        u = draw[group]
        out[i] = "test" if u < test else "val" if u < test + val else "train"
    return out


def _check_group_sizes(groups: np.ndarray, config: SplitConfig, what: str) -> None:
    if len(groups) == 0:
        return
    share = np.bincount(np.unique(groups, return_inverse=True)[1]).max() / len(groups)
    if share > config.max_group_share:
        raise SplitError(f"{config.scheme}: the largest {what} group holds {share:.0%} of documents "
                         f"(> {config.max_group_share:.0%}); the split would be one group. Raise the "
                         "near-duplicate threshold or split by another unit.")


def make_split(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    if config.scheme not in SCHEMES:
        raise SplitError(f"unknown scheme {config.scheme!r}; known {SCHEMES}")
    n = len(frame)
    split = np.full(n, "train", dtype=object)
    groups = variant_groups(frame) if n else np.zeros(0, dtype=int)
    variant_key = frame["variant_id"].astype(str).to_numpy()
    s = config.scheme

    if s == "random":
        split = np.array(["test" if (u := bucket(d, config.seed)) < config.test_fraction
                          else "val" if u < config.test_fraction + config.val_fraction else "train"
                          for d in frame["document_id"].astype(str)], dtype=object)
    elif s == "variant":
        split = _assign_by_group(groups, variant_key, config.seed, config.val_fraction, config.test_fraction)
    elif s == "text":
        columns = ["_vg", "near_dup_cluster"] + (["template_cluster"] if config.use_template_clusters else [])
        missing = [c for c in columns[1:] if c not in frame.columns]
        if missing:
            raise SplitError(f"text split needs {missing}: run `vpdl-slm dedup` first")
        joined = _connected(frame.assign(_vg=groups), columns)
        _check_group_sizes(joined, config, "text-similarity")
        split = _assign_by_group(joined, variant_key, config.seed, config.val_fraction, config.test_fraction)
    elif s == "laboratory":
        labs = frame["submitter"].astype(str).to_numpy()
        lab_split = _assign_by_group(np.unique(labs, return_inverse=True)[1], labs, config.seed,
                                     config.val_fraction, config.test_fraction)
        split = lab_split.copy()
        # The same variant interpreted by a lab on another side: keep it on the
        # most held-out side only (test > val > train); its other documents go.
        in_test = np.isin(groups, pd.Series(groups[lab_split == "test"]).unique())
        in_val = np.isin(groups, pd.Series(groups[lab_split == "val"]).unique())
        split[in_test & (lab_split != "test")] = "excluded"
        split[~in_test & in_val & (lab_split == "train")] = "excluded"
    elif s == "gene":
        genes = frame["gene"].astype(str).to_numpy()
        reserved = np.isin(genes, list(config.reserve_genes))
        gene_split = np.array([("test" if (u := bucket(g, config.seed)) < config.test_fraction else
                                "val" if u < config.test_fraction + config.val_fraction else "train")
                               for g in genes], dtype=object)
        gene_split[reserved] = "train"
        split = _spread_group_side(groups, gene_split)
    elif s == "disease":
        specific = frame["disease_names"].map(lambda names: [n for n in names if specific_condition(n)])
        keys = specific.map(lambda names: names[0] if names else None)
        joined = _connected(frame.assign(_vg=groups, _d=keys,
                                         _dall=specific.map(lambda n: "|".join(sorted(n)) or None)),
                            ["_vg", "_d", "_dall"])
        has_disease = keys.notna().to_numpy()
        _check_group_sizes(joined[has_disease], config, "disease")
        split = _assign_by_group(joined, keys.fillna("").astype(str).to_numpy(), config.seed,
                                 config.val_fraction, config.test_fraction)
        no_specific = ~has_disease
        split[no_specific & (split != "train")] = "train"
        split = _spread_group_side(joined, split)
    elif s == "temporal":
        split = _temporal(frame, groups, config)
    elif s == "functional":
        if not config.functional_variants:
            raise SplitError("functional split needs functional_variants (protein ids with holdout data)")
        held = frame["protein_variant_id"].isin(list(config.functional_variants)).to_numpy()
        held_groups = pd.Series(groups[held]).unique()
        is_test = np.isin(groups, held_groups)
        rest = _assign_by_group(groups, variant_key, config.seed, config.val_fraction, 0.0)
        split = np.where(is_test, "test", rest).astype(object)
    elif s == "mmr":
        mmr = frame["gene"].isin(list(config.mmr_genes)).to_numpy()
        mmr_groups = pd.Series(groups[mmr]).unique()
        mmr_any = np.isin(groups, mmr_groups)          # a variant group touching an MMR gene
        broad = _assign_by_group(groups, variant_key, config.seed, config.val_fraction, config.test_fraction)
        broad = np.where(broad == "test", "broad_test", broad).astype(object)
        adapt_train, adapt_val = config.mmr_adapt_fractions
        inner = _assign_by_group(groups, variant_key, config.seed + 1, adapt_val, 1 - adapt_train - adapt_val)
        inner = np.select([inner == "test", inner == "val"], ["test", "mmr_val"], "mmr_train").astype(object)
        split = np.where(mmr_any, inner, broad).astype(object)

    restricted = frame["restricted"].fillna(False).astype(bool).to_numpy() if "restricted" in frame else np.zeros(n, bool)
    split[restricted] = "excluded"
    out = pd.DataFrame({"document_id": frame["document_id"].to_numpy(),
                        "variant_id": frame["variant_id"].to_numpy(),
                        "variant_group": groups, "split": split})
    out.attrs["config"] = config.as_dict()
    return out


def _spread_group_side(groups: np.ndarray, split: np.ndarray) -> np.ndarray:
    """A variant group spanning two sides goes wholly to the most held-out one (test > val > train)."""
    rank = {"train": 0, "val": 1, "test": 2, "excluded": -1}
    worst: dict[int, str] = {}
    for group, value in zip(groups, split):
        if rank.get(value, 0) > rank.get(worst.get(group, "train"), 0):
            worst[group] = value
    return np.array([worst.get(g, v) if v != "excluded" else v for g, v in zip(groups, split)], dtype=object)


def _temporal(frame: pd.DataFrame, groups: np.ndarray, config: SplitConfig) -> np.ndarray:
    if not config.cutoff:
        raise SplitError("temporal split needs cutoff (ISO date)")
    cutoff = config.cutoff
    val_cutoff = config.val_cutoff or f"{int(cutoff[:4]) - 1}{cutoff[4:]}"
    dates = frame["date"].fillna("").astype(str).to_numpy()
    dated = dates != ""
    first = pd.Series(np.where(dated, dates, "9999"), index=groups).groupby(level=0).min()
    group_first = first.reindex(groups).to_numpy()
    split = np.full(len(frame), "excluded", dtype=object)
    before = dated & (dates < cutoff)
    after = dated & (dates >= cutoff)
    if config.temporal_mode == "novel":
        novel = group_first >= cutoff
        split[after & novel] = "test"
        split[before] = "train"
        in_val_window = before & (group_first >= val_cutoff)
        split[in_val_window] = "val"
    elif config.temporal_mode == "reinterpretation":
        seen_before = group_first < cutoff
        split[after & seen_before] = "test"
        split[before] = "train"
        split[before & (dates >= val_cutoff) & (group_first >= val_cutoff)] = "val"
    else:
        raise SplitError(f"unknown temporal_mode {config.temporal_mode!r}")
    return split


def split_summary(split: pd.DataFrame, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    counts = split["split"].value_counts().to_dict()
    variants = split.groupby("split")["variant_id"].nunique().to_dict()
    summary = {"config": split.attrs.get("config"), "documents": counts, "variants": variants,
               "split_sha256": hashlib.sha256(json.dumps(
                   sorted(zip(split["document_id"].astype(str), split["split"].astype(str))))
                   .encode()).hexdigest()}
    if frame is not None and "label5" in frame.columns:
        merged = split.merge(frame[["document_id", "label5", "gene"]], on="document_id", how="left")
        summary["labels"] = {k: v["label5"].fillna("none").value_counts().to_dict()
                             for k, v in merged.groupby("split")}
        summary["genes"] = merged.groupby("split")["gene"].nunique().to_dict()
    return summary
