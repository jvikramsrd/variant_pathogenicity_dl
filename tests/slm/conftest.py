"""Shared fixtures for tests/slm."""

import pytest


@pytest.fixture(autouse=True)
def _private_run_registry(tmp_path, monkeypatch):
    """Tests never append to the real runs/slm_genomic/registry.jsonl.

    That file is the record of real runs; a test or smoke run landing in it
    would sit next to DGX results as if it were one.
    """
    try:
        import vpdl.slm.modeling.finetune as finetune
    except ImportError:                      # the module is optional for the older tests
        return
    monkeypatch.setattr(finetune, "REGISTRY", tmp_path / "registry.jsonl")
