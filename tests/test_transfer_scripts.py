"""Regression tests: does the 2026-09-06 reproducibility fix reach every
script that builds a model head inside a per-split/per-architecture loop?

docs/RUNLOG.md (2026-09-06) found that scripts/finetune_esm_mmr.py's
run_one_split built ``ESMFineTuneClassifier`` -- which initialises the
head -- before anything seeded the RNG, so the *first* split of a process
drew its head from whatever entropy PyTorch's default generator carried at
that point rather than from ``--seed``. The fix was to call ``set_seed``
immediately before construction, inside the loop, not once in ``main()``.
``scripts/compare_finetune_strategies.py`` had the identical defect and got
the identical fix; see tests/test_finetune_grid.py for the original
regression test this file mirrors.

That fix was never ported to two sibling scripts with the same shape:

* ``scripts/run_mmr_transfer.py`` -- ``run_one_split`` calls ``build_model``
  for each of 5 architectures before ``fit_head``'s own ``set_global_seed``
  runs; only ``main()`` seeds once, before the whole gene x architecture loop.
* ``scripts/eval_leave_one_protein_out.py`` -- the held-out-gene loop in
  ``main()`` calls ``build_model`` before ``fit_head``'s seed call, with the
  same one-time seeding in ``main()`` and nothing per iteration.

Both are now fixed the same way (seed immediately before ``build_model``,
inside the loop). These tests reproduce the ambient-entropy trick from
tests/test_finetune_grid.py's
``test_first_split_head_init_depends_on_seed_not_process_entropy`` against
each script's own first ``build_model`` call: two different
``torch.manual_seed`` values, standing in for the differing process-start
entropy of two real runs, must yield the same head-initialisation draw.

Scope note: this checks that the loop's *first* iteration no longer reads
ambient state -- the exact defect that was fixed. It does not additionally
simulate a later iteration inheriting RNG state consumed by an earlier
iteration's training (tests/test_finetune_grid.py doesn't either); the fix
applied is unconditional per iteration, so the first iteration is
representative of every iteration in the same loop body.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_script(name: str):
    """Import a `scripts/` file as a module (see tests/test_finetune_grid.py).

    The module must be registered in ``sys.modules`` *before* exec_module:
    @dataclass resolves its own module through sys.modules, and fails with an
    opaque AttributeError if it is not there yet.
    """
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mmr_frame(n_per_gene: int = 12) -> pd.DataFrame:
    """Two-gene clinical-label frame, enough for a 5-fold position-group
    split within either gene alone (12 distinct positions/groups)."""
    rng = np.random.default_rng(0)
    rows = []
    for gene in ("MLH1", "MSH2"):
        for i in range(n_per_gene):
            rows.append({
                "gene": gene,
                "uniprot_id": f"U_{gene}",
                "position": i + 1,
                "label": float(i % 2),
                "label_source": "clinvar",
            })
    return pd.DataFrame(rows).sample(
        frac=1.0, random_state=int(rng.integers(1 << 30)))


class _SentinelStop(Exception):
    """Aborts the caller once the first architecture's head has been built."""


def test_run_mmr_transfer_first_arch_head_init_depends_on_seed_not_ambient_entropy():
    """scripts/run_mmr_transfer.py never got the docs/RUNLOG.md 2026-09-06 fix.

    ``build_model`` was called before ``fit_head``'s own ``set_global_seed``,
    so the first architecture benchmarked for a gene drew its head from
    whatever RNG state the process happened to be in -- the same defect
    class as the MLH1 bug that motivated the fix in
    scripts/finetune_esm_mmr.py, here spanning architecture order within one
    gene rather than gene order within one sweep.
    """
    mod = load_script("run_mmr_transfer")
    captured: dict = {}

    def fake_assemble_features(df, sequence_by_gene, model_name, processed_dir,
                               device, features_mode="esm+priors", batch_size=8,
                               overwrite_cache=False, fixed_prior_columns=None,
                               fixed_impute_values=None):
        meta = df.reset_index(drop=True)
        rng = np.random.default_rng(0)
        n = len(meta)
        return mod.FeatureBundle(
            X_esm=rng.standard_normal((n, 4)).astype(np.float32),
            X_prior=rng.standard_normal((n, 3)).astype(np.float32),
            prior_cols=["p0", "p1", "p2"], meta=meta, impute_values={})

    def recorder(*a, **kw):
        captured["draw"] = float(torch.randn(1).item())
        raise _SentinelStop

    real_assemble, real_build = mod.assemble_features, mod.build_model
    mod.assemble_features = fake_assemble_features
    mod.build_model = recorder
    try:
        def draw_for(ambient_seed: int) -> float:
            captured.clear()
            torch.manual_seed(ambient_seed)
            args = mod.parse_args(["--seed", "42", "--n_bootstrap", "10",
                                   "--no-save_checkpoints"])
            try:
                mod.run_one_split(args, _mmr_frame(), {}, torch.device("cpu"),
                                  "MLH1", ckpt=None)
            except _SentinelStop:
                pass
            return captured["draw"]

        a = draw_for(ambient_seed=1)
        b = draw_for(ambient_seed=999_983)
        assert a == b, (
            "first architecture's head initialisation differs between two "
            f"runs of the same command ({a} vs {b}): build_model is still "
            "reading process-start entropy instead of --seed")
    finally:
        mod.assemble_features = real_assemble
        mod.build_model = real_build


def test_eval_leave_one_protein_out_head_init_depends_on_seed_not_ambient_entropy():
    """scripts/eval_leave_one_protein_out.py has the identical unfixed defect
    at its own ``build_model`` call, one gene-loop iteration earlier than
    ``fit_head``'s ``set_global_seed``.
    """
    mod = load_script("eval_leave_one_protein_out")
    captured: dict = {}

    def fake_assemble_features(df, sequence_by_gene, model_name, processed_dir,
                               device, features_mode="priors", batch_size=8,
                               overwrite_cache=False):
        meta = df.reset_index(drop=True)
        rng = np.random.default_rng(0)
        n = len(meta)
        return mod.FeatureBundle(
            X_esm=None, X_prior=rng.standard_normal((n, 3)).astype(np.float32),
            prior_cols=["p0", "p1", "p2"], meta=meta, impute_values={})

    def recorder(*a, **kw):
        captured["draw"] = float(torch.randn(1).item())
        raise _SentinelStop

    real_assemble, real_build = mod.assemble_features, mod.build_model
    mod.assemble_features = fake_assemble_features
    mod.build_model = recorder
    tmp_csv = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False) as fh:
            _mmr_frame().to_csv(fh.name, index=False)
            tmp_csv = fh.name

        def draw_for(ambient_seed: int) -> float:
            captured.clear()
            torch.manual_seed(ambient_seed)
            sys.argv = ["eval_leave_one_protein_out.py",
                       "--train_csv", tmp_csv, "--genes", "MLH1,MSH2",
                       "--min_variants", "1", "--n_bootstrap", "10"]
            try:
                mod.main()
            except _SentinelStop:
                pass
            return captured["draw"]

        a = draw_for(ambient_seed=1)
        b = draw_for(ambient_seed=999_983)
        assert a == b, (
            "held-out-gene loop's head initialisation differs between two "
            f"runs of the same command ({a} vs {b}): build_model is still "
            "reading process-start entropy instead of --seed")
    finally:
        mod.assemble_features = real_assemble
        mod.build_model = real_build
        if tmp_csv:
            Path(tmp_csv).unlink(missing_ok=True)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
