"""Backbone registry and loading: ESM-1b, the ESM-2 family, and "existing".

"existing" is ESM-2 650M, the backbone v1's stage-2b grid fine-tuned, so the
comparison the plan asks for — existing backbone vs ESM-1b vs ESM-2 — is three
registry names under one protocol. Two independent papers found ESM-1b better
than ESM-2 for clinical pathogenicity specifically; nothing here assumes either.

Tokenisation uses the fixed 33-token ESM alphabet and is checked against the
checkpoint's own tokenizer at load time, so a vocabulary mismatch fails loudly
instead of scoring the wrong residue (the PLLR sign bug's cousin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = ["ESM_ALPHABET", "AA20", "BackboneSpec", "BACKBONES", "EXISTING_BACKBONE",
           "resolve_spec", "EsmAlphabet", "LoadedBackbone", "load_backbone",
           "tiny_backbone"]

ESM_ALPHABET: tuple[str, ...] = (
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I",
    "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
)
AA20 = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    params_m: int
    layers: int
    hidden: int
    positional: str                  # absolute | rotary
    max_residues: int = 1022         # training crop; ESM-1b's is also a hard limit
    note: str = ""

    @property
    def supports_full_length(self) -> bool:
        return self.positional == "rotary"


BACKBONES: dict[str, BackboneSpec] = {
    "esm1b": BackboneSpec("esm1b", "facebook/esm1b_t33_650M_UR50S", 650, 33, 1280,
                          "absolute", note="UR50/S; learned positions, hard 1,022 limit"),
    "esm2_8m": BackboneSpec("esm2_8m", "facebook/esm2_t6_8M_UR50D", 8, 6, 320, "rotary",
                            note="smoke runs only"),
    "esm2_35m": BackboneSpec("esm2_35m", "facebook/esm2_t12_35M_UR50D", 35, 12, 480, "rotary"),
    "esm2_150m": BackboneSpec("esm2_150m", "facebook/esm2_t30_150M_UR50D", 150, 30, 640,
                              "rotary"),
    "esm2_650m": BackboneSpec("esm2_650m", "facebook/esm2_t33_650M_UR50D", 650, 33, 1280,
                              "rotary"),
    "esm2_3b": BackboneSpec("esm2_3b", "facebook/esm2_t36_3B_UR50D", 3000, 36, 2560, "rotary",
                            note="only with evidence the 650M model underfits"),
}
EXISTING_BACKBONE = "esm2_650m"


def resolve_spec(name: str) -> BackboneSpec:
    key = EXISTING_BACKBONE if name == "existing" else name
    if key not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; known: {sorted(BACKBONES)} + 'existing'")
    return BACKBONES[key]


class EsmAlphabet:
    """The ESM vocabulary; ``<cls> residues <eos>`` per sequence, right-padded."""

    def __init__(self, tokens: Sequence[str] = ESM_ALPHABET) -> None:
        self.tokens = tuple(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.cls, self.pad = self.index["<cls>"], self.index["<pad>"]
        self.eos, self.mask = self.index["<eos>"], self.index["<mask>"]
        self.unk = self.index["<unk>"]
        self.aa_ids = [self.index[a] for a in AA20]

    def check_tokenizer(self, tokenizer: Any) -> None:
        """Raise unless a HF tokenizer maps every token to the same id."""
        wrong = [(t, i, tokenizer.convert_tokens_to_ids(t)) for i, t in enumerate(self.tokens)
                 if tokenizer.convert_tokens_to_ids(t) != i]
        if wrong:
            raise ValueError(f"checkpoint tokenizer disagrees with the ESM alphabet: {wrong[:5]}")

    def encode(self, sequences: Sequence[str], mask_positions: Sequence[int] | None = None):
        """``(input_ids, attention_mask)`` LongTensors; optional per-sequence mask."""
        import torch

        width = max(len(s) for s in sequences) + 2
        ids = torch.full((len(sequences), width), self.pad, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            codes = [self.cls] + [self.index.get(c, self.unk) for c in sequence] + [self.eos]
            ids[row, :len(codes)] = torch.tensor(codes)
            attention[row, :len(codes)] = 1
            if mask_positions is not None and mask_positions[row] is not None:
                ids[row, 1 + int(mask_positions[row])] = self.mask
        return ids, attention


@dataclass
class LoadedBackbone:
    model: Any                       # transformers EsmForMaskedLM
    alphabet: EsmAlphabet
    spec: BackboneSpec
    version: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def encoder(self):
        """The bare encoder (EsmModel), whatever head wraps it."""
        return getattr(self.model, "esm", self.model)


def load_backbone(name: str, device=None, dtype: str = "auto",
                  attn_implementation: str = "sdpa", gradient_checkpointing: bool = False,
                  revision: str | None = None) -> LoadedBackbone:
    """Load a registry backbone from the Hugging Face hub (or its local cache).

    ``dtype="auto"`` keeps fp32 master weights; autocast decides compute
    precision. ``attn_implementation="sdpa"`` routes attention through
    PyTorch's scaled-dot-product kernels (FlashAttention / memory-efficient
    back ends where the GPU and shape allow); an unsupported value falls back
    to eager, logged.
    """
    import torch
    import transformers
    from transformers import AutoTokenizer, EsmForMaskedLM

    spec = resolve_spec(name)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=revision)
    alphabet = EsmAlphabet()
    alphabet.check_tokenizer(tokenizer)
    torch_dtype = {"auto": None, "fp32": torch.float32, "bf16": torch.bfloat16,
                   "fp16": torch.float16}[dtype]
    kwargs: dict[str, Any] = dict(revision=revision)
    if torch_dtype is not None:
        # `dtype=` from transformers 4.56; older releases call it `torch_dtype`.
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = torch_dtype
    try:
        model = EsmForMaskedLM.from_pretrained(spec.hf_id, attn_implementation=attn_implementation,
                                               **kwargs)
        attention = attn_implementation
    except (ValueError, ImportError, TypeError) as error:
        logger.warning("%s: attn_implementation=%s unavailable (%s); using eager.",
                       spec.hf_id, attn_implementation, error)
        model = EsmForMaskedLM.from_pretrained(spec.hf_id, **kwargs)
        attention = "eager"
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if device is not None:
        model.to(device)
    version = {"hf_id": spec.hf_id,
               "revision": revision or getattr(model.config, "_commit_hash", None),
               "transformers": transformers.__version__, "attention": attention,
               "positional": spec.positional}
    logger.info("backbone %s (%s) loaded: %s", spec.name, spec.hf_id, version)
    return LoadedBackbone(model=model, alphabet=alphabet, spec=spec, version=version)


def tiny_backbone(positional: str = "rotary", layers: int = 2, hidden: int = 32,
                  heads: int = 2, max_residues: int = 62, seed: int = 0) -> LoadedBackbone:
    """A randomly initialised ESM with the real alphabet — tests and dry runs only.

    Same classes, same tokenisation, same forward signature as the 650M
    model, so shape and plumbing errors surface on a CPU in seconds. Its
    outputs mean nothing.
    """
    import torch
    from transformers import EsmConfig, EsmForMaskedLM

    torch.manual_seed(seed)
    config = EsmConfig(vocab_size=len(ESM_ALPHABET), hidden_size=hidden,
                       num_hidden_layers=layers, num_attention_heads=heads,
                       intermediate_size=2 * hidden, max_position_embeddings=max_residues + 4,
                       pad_token_id=1, mask_token_id=32, position_embedding_type=positional,
                       token_dropout=False, emb_layer_norm_before=False)
    model = EsmForMaskedLM(config)
    spec = BackboneSpec(f"tiny_{positional}", "random-init", 0, layers, hidden, positional,
                        max_residues=max_residues, note="tests only")
    return LoadedBackbone(model=model, alphabet=EsmAlphabet(), spec=spec,
                          version={"hf_id": "random-init", "seed": seed})
