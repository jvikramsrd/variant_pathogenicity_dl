"""TOML configuration for the expensive pipelines — read strictly, never half-understood.

One file per experiment (``configs/slm/*.toml``). Unknown keys are an error,
not a silent no-op: a typo in ``max_lenght`` must not quietly train at the
default length. Values are checked against the dataclass fields of
``FinetuneConfig`` / ``PretrainConfig``.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

__all__ = ["load_toml", "load_finetune_config", "load_pretrain_config", "config_problems"]

_TUPLE_FIELDS = {"tasks", "features"}


def load_toml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _build(cls, data: Mapping[str, Any], path: Path | str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {unknown} for {cls.__name__}; "
                         f"known: {sorted(known)}")
    values = dict(data)
    for name in _TUPLE_FIELDS & set(values):
        values[name] = tuple(values[name])
    return cls(**values)


def load_finetune_config(path: Path | str, overrides: Mapping[str, Any] | None = None):
    from vpdl.slm.modeling.finetune import FinetuneConfig
    data = load_toml(path)
    data = data.get("finetune", data)
    data.update(dict(overrides or {}))
    return _build(FinetuneConfig, data, path)


def load_pretrain_config(path: Path | str, overrides: Mapping[str, Any] | None = None):
    from vpdl.slm.modeling.continued import PretrainConfig
    data = load_toml(path)
    data = data.get("pretrain", data)
    data.update(dict(overrides or {}))
    return _build(PretrainConfig, data, path)


def config_problems(config: Any) -> list[str]:
    """Paths and settings a run needs, checked before anything expensive starts."""
    problems = []
    for name in ("examples", "unit_examples", "records", "dl_outputs", "corpus_dir", "token_dir"):
        value = getattr(config, name, None)
        if value and not Path(value).exists():
            problems.append(f"{name}: {value} does not exist")
    tasks = getattr(config, "tasks", ())
    if "evidence" in tasks and not getattr(config, "unit_examples", None):
        problems.append("tasks include 'evidence' but unit_examples is not set")
    if getattr(config, "select_metric", "auto") not in ("auto", "binary_auc", "macro_f1", "neg_loss"):
        problems.append(f"select_metric {config.select_metric!r} is not one of "
                        "auto | binary_auc | macro_f1 | neg_loss")
    train = getattr(config, "train", {})
    if isinstance(train, dict):
        from vpdl.dl.trainer import TrainConfig
        unknown = sorted(set(train) - {f.name for f in fields(TrainConfig)})
        if unknown:
            problems.append(f"train: unknown setting(s) {unknown}")
    return problems
