"""Model construction, forward passes, losses, determinism and checkpoint resumption.

All on CPU with tiny, randomly initialised networks: the numbers mean nothing,
the shapes, plumbing and invariants do. No pretrained weights are downloaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


# -- sequence-window baselines -----------------------------------------------------------

def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    out = []
    for _ in range(n):
        wt = "".join(rng.choice(aa, 15))
        vt = wt[:7] + rng.choice([a for a in aa if a != wt[7]]) + wt[8:]
        out.append((wt, vt))
    return out


@pytest.mark.parametrize("encoder", ["aa_mlp", "cnn", "bilstm_attn", "transformer"])
def test_window_baselines_construct_fit_and_predict(encoder):
    from vpdl.models import build_model

    windows = _windows(64)
    y = np.array([i % 2 for i in range(64)])
    tab = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)
    model = build_model(encoder, n_tabular_features=3, seed=42, split_index="MLH1",
                        epochs=2, batch_size=16)
    model.fit(windows, y, tabular=tab, windows_val=windows[:16], y_val=y[:16],
              tabular_val=tab[:16])
    p = model.predict_proba(windows, tabular=tab)
    assert p.shape == (64,) and np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("encoder", ["aa_mlp", "cnn", "bilstm_attn", "transformer"])
def test_window_baselines_are_seeded_per_split(encoder):
    from vpdl.models import build_model

    init = lambda split: build_model(encoder, seed=42, split_index=split).initial_weights()
    np.testing.assert_array_equal(init("MLH1"), init("MLH1"))
    assert not np.array_equal(init("MLH1"), init("MSH2"))


# -- fusion ------------------------------------------------------------------------------

def test_fusion_modalities_masks_uncertainty_and_representation():
    from vpdl.dl.fusion import FusionClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 7)).astype(np.float32)
    X[:, 6] = (rng.random(120) > 0.3)                 # availability mask for "structure"
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int)
    model = FusionClassifier(n_features=7, seed=42, split_index="MSH6",
                             modalities={"population": [0, 1], "structure": [2, 3],
                                         "sequence_plm": [4, 5]},
                             masks={"structure": 6}, epochs=3, batch_size=32, mc_samples=5)
    model.fit(X, y, X_val=X[:30], y_val=y[:30])
    p = model.predict_proba(X)
    mean, std = model.predict_with_uncertainty(X)
    rep = model.represent(X)
    assert p.shape == (120,) and mean.shape == (120,) and np.all(std >= 0)
    assert rep.shape == (120, 128)
    np.testing.assert_allclose(model.predict_proba(X), p)       # deterministic in eval


def test_fusion_gated_mode_and_missing_modality_is_not_zero_filled_by_value():
    from vpdl.dl.fusion import build_fusion_net

    net = build_fusion_net({"a": 3, "b": 2}, mode="gated", modality_dropout=0.0).eval()
    parts = {"a": torch.randn(4, 3), "b": torch.randn(4, 2)}
    masked = torch.tensor([[1.0, 0.0]] * 4)
    changed = {"a": parts["a"], "b": parts["b"] * 100}
    # With modality b masked out, its values must not matter.
    torch.testing.assert_close(net(parts, masked), net(changed, masked))


def test_fusion_is_seeded_before_construction():
    from vpdl.dl.fusion import FusionClassifier

    init = lambda split: FusionClassifier(4, seed=42, split_index=split).initial_weights()
    np.testing.assert_array_equal(init("PMS2"), init("PMS2"))
    assert not np.array_equal(init("PMS2"), init("MLH1"))


# -- PEFT ----------------------------------------------------------------------------------

def _tiny(positional="rotary"):
    pytest.importorskip("transformers")
    from vpdl.dl.plm.backbones import tiny_backbone
    return tiny_backbone(positional=positional)


def test_lora_trains_only_adapters_and_starts_as_identity():
    from vpdl.dl.plm.forward import hidden_states
    from vpdl.dl.plm.peft import apply_strategy, merge_lora, trainable_state_dict

    backbone = _tiny()
    before = hidden_states(backbone, ["MKTAYIAKQRQISFVKSHFSRQ"])[0]
    summary = apply_strategy(backbone.model, {"kind": "lora", "lora_rank": 4})
    after = hidden_states(backbone, ["MKTAYIAKQRQISFVKSHFSRQ"])[0]
    np.testing.assert_allclose(before, after, atol=1e-5)          # B zero-init
    names = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert 0 < summary["trainable_fraction"] < 0.5
    assert set(trainable_state_dict(backbone.model)) == set(names)
    assert merge_lora(backbone.model) == 2 * 2                     # query+value x 2 layers


def test_adapters_start_as_identity_and_last_n_unfreezes_the_top():
    from vpdl.dl.plm.forward import hidden_states
    from vpdl.dl.plm.peft import apply_strategy

    backbone = _tiny()
    before = hidden_states(backbone, ["MKTAYIAKQR"])[0]
    apply_strategy(backbone.model, {"kind": "adapters", "adapter_dim": 8})
    np.testing.assert_allclose(before, hidden_states(backbone, ["MKTAYIAKQR"])[0], atol=1e-5)

    other = _tiny()
    apply_strategy(other.model, {"kind": "last_n", "last_n": 1})
    trainable = {n.split(".")[3] for n, p in other.model.named_parameters()
                 if p.requires_grad and ".layer." in n}
    assert trainable == {"1"}                                      # only the top layer


def test_full_finetune_of_a_large_backbone_is_refused_by_default():
    from vpdl.dl.plm.peft import apply_strategy

    backbone = _tiny()
    with pytest.raises(ValueError, match="refused"):
        apply_strategy(backbone.model, {"kind": "full", "full_max_params_m": 0})
    apply_strategy(backbone.model, {"kind": "full", "full_max_params_m": 0, "allow_full": True})


# -- embeddings and zero-shot ---------------------------------------------------------------

@pytest.mark.parametrize("policy", ["full", "centered", "asymmetric", "sliding", "hierarchical"])
def test_embedding_blocks_have_the_right_shape_and_a_live_wt_vt_contrast(policy):
    from vpdl.dl.plm.embed import RAW_BLOCKS, EmbeddingExtractor, derive

    backbone = _tiny()                         # max_residues 62: a 90-residue chain overflows
    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 90))
    variants = [(5, sequence[4], "W" if sequence[4] != "W" else "A"),
                (80, sequence[79], "G" if sequence[79] != "G" else "A")]
    blocks = EmbeddingExtractor(backbone, policy, chunk=1).extract(sequence, variants)
    assert set(blocks) == set(RAW_BLOCKS)
    assert all(b.shape == (2, 32) for b in blocks.values())
    assert not np.allclose(blocks["site_wt"], blocks["site_vt"])   # landmine L12
    assert derive(blocks, "site.concat4").shape == (2, 128)
    np.testing.assert_allclose(derive(blocks, "local.abs_diff"),
                               np.abs(blocks["local_vt"] - blocks["local_wt"]))


def test_sliding_policy_matches_a_brute_force_recompute():
    from vpdl.dl.context import sliding_spans
    from vpdl.dl.plm.embed import EmbeddingExtractor
    from vpdl.dl.plm.forward import hidden_states

    backbone = _tiny()
    rng = np.random.default_rng(1)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 150))
    mutated = sequence[:99] + ("A" if sequence[99] != "A" else "C") + sequence[100:]
    blocks = EmbeddingExtractor(backbone, "sliding").extract(sequence, [(100, sequence[99],
                                                                        mutated[99])])
    spans = sliding_spans(150, 62)
    total, counts = np.zeros((150, 32)), np.zeros(150)
    for (s, e), h in zip(spans, hidden_states(backbone, [mutated[s:e] for s, e in spans])):
        total[s:e] += h
        counts[s:e] += 1
    np.testing.assert_allclose(blocks["global_vt"][0], (total / counts[:, None]).mean(0),
                               atol=1e-4)


def test_esm1b_style_backbone_refuses_full_context_on_a_long_chain():
    from vpdl.dl.context import ContextError
    from vpdl.dl.plm.embed import EmbeddingExtractor

    backbone = _tiny("absolute")
    sequence = "A" * 90
    with pytest.raises(ContextError):
        EmbeddingExtractor(backbone, "full").extract(sequence, [(5, "A", "V")])


def test_masked_marginal_masks_and_differs_from_the_wildtype_marginal():
    from vpdl.dl.plm.forward import site_log_probs
    from vpdl.dl.plm.zeroshot import position_log_probs, score_variants, zeroshot_frame

    backbone = _tiny()
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    masked = position_log_probs(backbone, sequence, [4], "masked_marginal")[4]
    unmasked = position_log_probs(backbone, sequence, [4], "wt_marginal")[4]
    assert not np.allclose(masked, unmasked)
    # Manual masked pass: token at the site replaced by <mask>.
    ids, att = backbone.alphabet.encode([sequence], mask_positions=[3])
    assert ids[0, 4] == backbone.alphabet.mask
    manual = site_log_probs(backbone, [sequence], [3], masked=True)[0]
    np.testing.assert_allclose(masked, manual, atol=1e-5)
    raw = score_variants({4: masked}, [(4, "A", "V")])
    frame = pd.DataFrame({"uniprot_id": ["X"], "position": [4], "wt_aa": ["A"], "mut_aa": ["V"]})
    scored = zeroshot_frame(backbone, frame, {"X": sequence})
    assert scored["raw_llr"].iloc[0] == pytest.approx(raw[0], abs=1e-5)
    assert scored["pathogenicity"].iloc[0] == pytest.approx(-raw[0], abs=1e-5)   # L6


def test_zeroshot_refuses_a_wildtype_mismatch():
    from vpdl.dl.plm.zeroshot import zeroshot_frame

    frame = pd.DataFrame({"uniprot_id": ["X"], "position": [4], "wt_aa": ["W"], "mut_aa": ["V"]})
    with pytest.raises(ValueError, match="mismatch"):
        zeroshot_frame(_tiny(), frame, {"X": "MKTAYIAKQR"})


# -- PLM fine-tuning ---------------------------------------------------------------------------

def test_plm_finetune_fits_and_starts_as_the_zero_shot_predictor():
    from vpdl.dl.plm.finetune import PLMFinetuneClassifier

    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 90))
    rows = [{"uniprot_id": "X", "position": p, "wt_aa": sequence[p - 1],
             "mut_aa": "A" if sequence[p - 1] != "A" else "C"} for p in range(1, 41)]
    frame = pd.DataFrame(rows)
    y = np.array([i % 2 for i in range(40)])
    model = PLMFinetuneClassifier(backbone=_tiny(), strategy={"kind": "lora", "lora_rank": 2},
                                  context="centered", seed=42, split_index="MLH1", epochs=1,
                                  batch_size=8, head_hidden=16, gradient_checkpointing=False)
    before = model.predict_frames(frame, sequences={"X": sequence})
    assert before.shape == (40,)
    model.fit_frames(frame, y, frame.iloc[:10], y[:10], sequences={"X": sequence})
    after = model.predict_frames(frame, sequences={"X": sequence})
    assert after.shape == (40,) and not np.allclose(before, after)
    assert model.represent_frames(frame.iloc[:3], sequences={"X": sequence}).shape == (3, 16)


# -- trainer: checkpoint resumption --------------------------------------------------------

def _toy_problem():
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(64, 5)), dtype=torch.float32)
    y = (X[:, 0] > 0).float()
    return X, y


def _fit(config, epochs_limit=None, seed=7):
    from torch import nn

    from vpdl.dl.trainer import Trainer, seed_everything

    seed_everything(seed)
    X, y = _toy_problem()
    model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1))
    trainer = Trainer(model, config, seed=seed, run_name="toy")
    loss = lambda m, b: nn.functional.binary_cross_entropy_with_logits(m(b[0]).squeeze(-1), b[1])
    result = trainer.fit(64, lambda i: (X[i], y[i]), loss)
    return model, result


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    from vpdl.dl.trainer import TrainConfig

    full, _ = _fit(TrainConfig(epochs=4, batch_size=16, lr=1e-2,
                               checkpoint_dir=str(tmp_path / "a")))
    # Same run, "killed" after epoch 2 (a 2-epoch run's checkpoints), then resumed.
    import shutil
    first = TrainConfig(epochs=4, batch_size=16, lr=1e-2, checkpoint_dir=str(tmp_path / "b"),
                        keep_last=10)
    _fit(first)
    checkpoints = sorted((tmp_path / "b").glob("epoch-*.pt"))
    for later in checkpoints:                  # keep only epoch 1 (0-based) as the resume point
        if later.name > "epoch-0001.pt":
            later.unlink()
    resumed, result = _fit(TrainConfig(epochs=4, batch_size=16, lr=1e-2, keep_last=10,
                                       checkpoint_dir=str(tmp_path / "b"), resume=True))
    assert result.resumed_from and result.resumed_from.endswith("epoch-0001.pt")
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resume_refuses_a_different_configuration(tmp_path):
    from vpdl.dl.trainer import TrainConfig

    _fit(TrainConfig(epochs=2, batch_size=16, checkpoint_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="different configuration"):
        _fit(TrainConfig(epochs=2, batch_size=8, checkpoint_dir=str(tmp_path), resume=True))


def test_checkpoint_carries_everything_resumption_needs(tmp_path):
    from vpdl.dl.trainer import TRAIN_CHECKPOINT_FORMAT, TrainConfig

    _fit(TrainConfig(epochs=1, batch_size=16, checkpoint_dir=str(tmp_path)))
    payload = torch.load(next(tmp_path.glob("epoch-*.pt")), weights_only=False)
    assert payload["format"] == TRAIN_CHECKPOINT_FORMAT
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "step", "rng", "config",
                "best_state", "best_score", "stalled"):
        assert key in payload, key
    assert {"python", "numpy", "torch"} <= set(payload["rng"])


# -- pretraining objectives -------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["P1", "P2", "P3", "P4"])
def test_each_pretraining_arm_runs_one_epoch_and_writes_a_mergeable_delta(tmp_path, arm):
    from vpdl.dl.plm.finetune import load_adapted_backbone
    from vpdl.dl.pretrain.corpus import build_corpus, write_fasta
    from vpdl.dl.pretrain.run import PretrainConfig, pretrain

    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    homologs = {f"h{i}": "".join(rng.choice(aa, 60)) for i in range(30)}
    write_fasta(homologs, tmp_path / "raw.fasta")
    panel = {"P40692": "".join(rng.choice(aa, 60)), "P43246": "".join(rng.choice(aa, 60))}
    build_corpus([tmp_path / "raw.fasta"], panel, tmp_path / "corpus", mode="strict",
                 holdout="P40692", val_fraction=0.2)
    config = PretrainConfig(arm=arm, backbone="tiny", corpus_dir=str(tmp_path / "corpus"),
                            strategy={"kind": "lora", "lora_rank": 2}, crop=40, epochs=1,
                            batch_size=8, grad_accum=1, gradient_checkpointing=False)
    backbone = _tiny()
    summary = pretrain(config, tmp_path / arm, backbone=backbone)
    assert np.isfinite(summary["best_val_mlm_loss"])
    fresh = _tiny()
    fresh.spec = type(fresh.spec)(**{**fresh.spec.__dict__, "name": "tiny"})
    info = load_adapted_backbone(fresh, tmp_path / arm)
    assert info["pretrain_arm"] == arm
    assert not any(hasattr(m, "lora_a") for m in fresh.model.modules())   # merged + unwrapped


def test_p0_trains_nothing(tmp_path):
    from vpdl.dl.pretrain.run import PretrainConfig, pretrain

    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "corpus_manifest.json").write_text('{"holdout": null}')
    summary = pretrain(PretrainConfig(arm="P0", corpus_dir=str(tmp_path / "corpus")),
                       tmp_path / "P0")
    assert "nothing trained" in summary["note"]
    assert not (tmp_path / "P0" / "backbone_delta.pt").exists()


def test_strict_corpus_excludes_the_held_out_gene_and_its_orthologs(tmp_path):
    from vpdl.dl.pretrain.corpus import build_corpus, read_fasta, write_fasta

    rng = np.random.default_rng(3)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    target = "".join(rng.choice(aa, 120))
    ortholog = "".join(c if rng.random() > 0.1 else rng.choice(aa) for c in target)
    unrelated = "".join(rng.choice(aa, 120))
    write_fasta({"orth": ortholog, "other": unrelated}, tmp_path / "raw.fasta")
    report = build_corpus([tmp_path / "raw.fasta"], {"P40692": target}, tmp_path / "c",
                          mode="strict", holdout="P40692", val_fraction=0.0)
    kept = set(read_fasta(tmp_path / "c" / "train.fasta"))
    assert kept == {"other"} and report.n_excluded_homologs == 2      # the gene + ortholog
