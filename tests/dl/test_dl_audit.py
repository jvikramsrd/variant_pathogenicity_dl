"""Regression tests for the local code audit of 2026-09-24 (DL_ARCHITECTURE_AUDIT.md and friends).

Each test pins one defect the audit fixed in the DL branch or the shared core. Synthetic data only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dl_helpers import assembled_table


# -- the per-cell leakage gate ---------------------------------------------------------------

def test_the_cell_gate_refuses_a_feature_that_is_the_label(table):
    from vpdl.dl.leakage import LeakageError, leakage_gate
    from vpdl.dl.splits import make_folds

    work = table.assign(_train=table["label__clinvar"], _eval=table["label__clinvar"])
    folds = make_folds(work, "logo")
    columns = [c for c in work.columns if c.startswith("feature_")]
    leakage_gate(work, folds, "logo", columns, ["clinvar"])             # honest features pass
    proxy = work.assign(feature_stability_delta=1.0 - work["_train"])   # the label, flipped (L2)
    with pytest.raises(LeakageError, match="feature_stability_delta"):
        leakage_gate(proxy, folds, "logo", columns + ["feature_stability_delta"], ["clinvar"])


# -- trainer configuration ------------------------------------------------------------------

@pytest.mark.parametrize("field", ["epochs", "patience", "batch_size", "grad_accum", "keep_last"])
def test_a_non_positive_training_setting_is_refused_not_clamped(field):
    from vpdl.dl.trainer import TrainConfig

    TrainConfig(**{field: 1})
    with pytest.raises(ValueError, match=field):
        TrainConfig(**{field: 0})


def test_plm_finetune_checkpoints_hold_only_what_trains(tmp_path):
    torch = pytest.importorskip("torch")
    from vpdl.dl.plm.backbones import tiny_backbone
    from vpdl.dl.plm.finetune import PLMFinetuneClassifier

    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 60))
    frame = pd.DataFrame([{"uniprot_id": "X", "position": p, "wt_aa": sequence[p - 1],
                           "mut_aa": "A" if sequence[p - 1] != "A" else "C"} for p in range(1, 21)])
    y = np.array([i % 2 for i in range(20)])
    model = PLMFinetuneClassifier(backbone=tiny_backbone(), strategy={"kind": "lora", "lora_rank": 2},
                                  seed=1, epochs=1, batch_size=8, head_hidden=8,
                                  gradient_checkpointing=False, checkpoint_dir=str(tmp_path))
    model.fit_frames(frame, y, frame.iloc[:6], y[:6], sequences={"X": sequence})
    saved = torch.load(sorted(tmp_path.glob("epoch-*.pt"))[-1], weights_only=False)
    trainable = {n for n, p in model.model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.model.named_parameters() if not p.requires_grad}
    assert frozen and trainable
    assert not frozen & set(saved["model"]), "frozen backbone weights were checkpointed"
    assert trainable <= set(saved["model"])


# -- failure analysis under the family split -------------------------------------------------

def test_overfitting_is_measured_per_fold_under_the_family_split():
    from vpdl.dl.failure import overfitting

    y = np.array([0, 1] * 10)
    predictions = pd.DataFrame({"gene": ["MLH1"] * 10 + ["PMS2"] * 10, "fold": "family:MutL",
                                "label": y, "score": y * 0.8 + 0.1})
    valpreds = pd.DataFrame({"gene": ["MSH2"] * 20, "fold": "family:MutL", "label": y,
                             "score": y * 0.6 + 0.2})
    table = overfitting(predictions, valpreds)
    assert list(table["fold"]) == ["family:MutL"]
    assert np.isfinite(table["inner_val_auc"]).all()


# -- provenance --------------------------------------------------------------------------------

def test_a_run_record_carries_its_command_line():
    from vpdl.dl.tracking import run_record

    record = run_record("test", {"a": 1}, seed=0)
    assert record["command"] == list(sys.argv)
    assert record["experiment_id"] == run_record("test", {"a": 1}, seed=0)["experiment_id"]


def test_manifest_paths_use_forward_slashes_on_every_os(tmp_path):
    from vpdl.provenance import verify_manifest, write_manifest

    artefact = tmp_path / "sub" / "table.csv"
    artefact.parent.mkdir()
    artefact.write_text("a,b\n1,2\n")
    write_manifest(tmp_path / "manifest.json", [artefact])
    recorded = json.loads((tmp_path / "manifest.json").read_text())["artefacts"]["table.csv"]["path"]
    assert "\\" not in recorded
    verify_manifest(tmp_path / "manifest.json")


# -- the DL export contract --------------------------------------------------------------------

def _representations():
    from vpdl.dl.interface import DLRepresentation
    return [DLRepresentation(f"P40692:{i}:G>R", np.full(4, float(i), dtype=np.float32), 0.5, 0.1,
                             {"gene": "MLH1", "fold": "MLH1", "model_version": "m",
                              "feature_version": "f"}) for i in (1, 2)]


def test_referenced_dl_embeddings_are_checked_for_non_finite_values(tmp_path):
    from vpdl.dl.interface import read_dl_outputs, write_dl_outputs

    paths = write_dl_outputs(_representations(), tmp_path)
    assert [r.embedding[0] for r in read_dl_outputs(paths["jsonl"])] == [1.0, 2.0]
    corrupt = np.load(paths["embeddings"])
    corrupt[1, 2] = np.nan
    np.save(paths["embeddings"], corrupt)
    with pytest.raises(ValueError, match="non-finite"):
        read_dl_outputs(paths["jsonl"])


def test_reading_an_export_does_not_hold_its_file_open(tmp_path):
    from vpdl.dl.interface import read_dl_outputs, write_dl_outputs

    paths = write_dl_outputs(_representations(), tmp_path)
    held = read_dl_outputs(paths["jsonl"])                  # still referenced below
    write_dl_outputs(_representations(), tmp_path)           # rewrite in place (Windows: locked before)
    paths["embeddings"].unlink()
    assert held[0].embedding.shape == (4,)


# -- vpdl-dl train --dry-run ---------------------------------------------------------------------

def _canonical(tmp_path, sequences):
    from vpdl.dl.canonical import build_canonical
    table, _, _ = build_canonical(assembled_table(sequences, per_position=1), sequences)
    data = tmp_path / "canonical.csv"
    table.to_csv(data, index=False)
    return table, data


def test_a_training_dry_run_checks_everything_and_writes_nothing(tmp_path, sequences, capsys):
    from vpdl.dl.cli import main

    _, data = _canonical(tmp_path, sequences)
    out = tmp_path / "runs"
    code = main(["train", "--data", str(data), "--sources", "clinvar", "--model", "mlp", "cnn",
                 "--seeds", "42", "--out", str(out), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["trained"] is False
    assert report["leakage"]["critical"] == 0
    assert [c["constructed"] for c in report["cells"]] == [True, True]
    assert len(report["folds"]) == 4
    assert not out.exists()


def test_a_training_dry_run_stops_on_critical_leakage(tmp_path, sequences, capsys):
    from vpdl.dl.cli import main

    table, data = _canonical(tmp_path, sequences)
    table.assign(feature_proxy=table["label__clinvar"]).to_csv(data, index=False)
    code = main(["train", "--data", str(data), "--sources", "clinvar", "--model", "mlp",
                 "--seeds", "42", "--out", str(tmp_path / "runs"), "--dry-run"])
    assert code == 3 and "feature_proxy" in capsys.readouterr().out
