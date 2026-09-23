"""Text backbones: pretrained biomedical encoders, our own from-scratch decoder, tiny test models.

A backbone spec is one string:

    hf:<model id or local dir>   any Hugging Face encoder (BERT family) — the baselines and
                                 the start of continued pretraining
    slm:<dir>                    a model saved by ``vpdl slm-train`` (Llama-style, random
                                 init, trained from scratch) or by ``vpdl-slm pretrain``
    tiny-bert / tiny-llama       random-weight models a few thousand parameters large, with
                                 a tokenizer built in memory — tests and dry runs only

Named baselines (ids and licence tags checked on Hugging Face 2026-09-22):

    biobert           dmis-lab/biobert-base-cased-v1.2        no licence tag on the model card
    pubmedbert        microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext   MIT
    bioclinicalbert   emilyalsentzer/Bio_ClinicalBERT         MIT

Nothing here downloads on import; ``load_backbone`` fetches a hub model only
when asked, on the machine that runs it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = ["NAMED_BACKBONES", "BackboneSpec", "Backbone", "load_backbone", "tiny_tokenizer",
           "resolve_spec", "tokenizer_fingerprint"]

NAMED_BACKBONES: dict[str, dict[str, str]] = {
    "biobert": {"id": "dmis-lab/biobert-base-cased-v1.2",
                "licence": "no licence tag on the Hugging Face card (checked 2026-09-22) — confirm before release",
                "lineage": "BERT-base (general) -> PubMed/PMC continued pretraining"},
    "pubmedbert": {"id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                   "licence": "MIT", "lineage": "trained from scratch on PubMed abstracts + PMC full text"},
    "bioclinicalbert": {"id": "emilyalsentzer/Bio_ClinicalBERT", "licence": "MIT",
                        "lineage": "BioBERT -> MIMIC-III clinical notes"},
}

_TINY_TEXT = [
    "The variant was absent from population databases such as gnomAD.",
    "It segregated with disease in three affected relatives.",
    "A functional assay showed reduced mismatch repair activity.",
    "In silico tools predict a damaging effect on MLH1 protein function.",
    "The c.199G>A (p.Gly67Arg) change replaces glycine with arginine.",
    "Loss of MSH2 expression and microsatellite instability were observed in the tumour.",
    "The allele frequency is 1.2% in the general population, which is too common.",
    "This missense change is predicted to be tolerated and is poorly conserved.",
    "PM2_Supporting PP3 PS3_Moderate BS1 BP4 BA1 PVS1",
    "Lynch syndrome is an autosomal dominant cancer predisposition.",
]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    pooling: str = "auto"          # auto | cls | mean | last
    max_length: int = 512
    attn_implementation: str | None = "sdpa"
    revision: str | None = None


def resolve_spec(name: str) -> str:
    """'pubmedbert' -> 'hf:microsoft/...'; anything with a scheme is returned as is."""
    if name in NAMED_BACKBONES:
        return f"hf:{NAMED_BACKBONES[name]['id']}"
    return name


class Backbone:
    def __init__(self, model, tokenizer, kind: str, spec: BackboneSpec, version: dict[str, Any]):
        self.model, self.tokenizer, self.kind, self.spec = model, tokenizer, kind, spec
        self.version = version
        self.hidden_size = int(model.config.hidden_size)
        pooling = spec.pooling
        if pooling == "auto":
            pooling = "cls" if kind == "encoder" else "mean"
        self.pooling = pooling

    def pool(self, hidden, attention_mask):
        import torch
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        if self.pooling == "last":
            last = attention_mask.sum(1).long().clamp(min=1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        raise ValueError(f"unknown pooling {self.pooling!r}")

    def encode(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        return self.pool(hidden, attention_mask)

    def tokenize(self, texts: Sequence[str], max_length: int | None = None) -> dict[str, Any]:
        return self.tokenizer(list(texts), truncation=True, padding="max_length",
                              max_length=max_length or self.spec.max_length, return_tensors="np")


def tiny_tokenizer():
    """A tiny WordPiece tokenizer with a FIXED vocabulary — identical in every process.

    Built, not trained: ``tokenizers``' WordPiece trainer orders its merges
    non-deterministically, and two tokenizers that differ by one id would make
    the tokenizer-fingerprint guard (which protects cached tokens and packed
    token files) fire on a difference that means nothing. Characters and their
    continuations are all present, so any lower-case text tokenises without
    [UNK]; the whole words of a few clinical sentences are added on top so the
    ids stay readable in tests.
    """
    import string

    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    characters = list(string.ascii_lowercase + string.digits) + list(".,:;()[]%<>+-*/'\"=#&?!")
    words = sorted({word for line in _TINY_TEXT for word in
                    "".join(c if c.isalnum() else " " for c in line.lower()).split()})
    tokens = (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + characters
              + [f"##{c}" for c in characters] + words)
    vocabulary = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    tokenizer = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]", max_input_chars_per_word=64))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def _tiny(kind: str, seed: int = 0):
    import torch
    tokenizer = tiny_tokenizer()
    torch.manual_seed(seed)
    if kind == "tiny-bert":
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=64, max_position_embeddings=256,
                            pad_token_id=tokenizer.pad_token_id)
        return BertModel(config), tokenizer, "encoder"
    from transformers import LlamaConfig, LlamaModel
    config = LlamaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64, max_position_embeddings=256,
                         pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.cls_token_id,
                         eos_token_id=tokenizer.sep_token_id)
    return LlamaModel(config), tokenizer, "decoder"


def _weights_digest(path: Path) -> str | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        file = path / name
        if file.exists():
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()[:16]
    return None


def load_backbone(spec: BackboneSpec | str, seed: int = 0) -> Backbone:
    spec = spec if isinstance(spec, BackboneSpec) else BackboneSpec(spec)
    name = resolve_spec(spec.name)
    if name in ("tiny-bert", "tiny-llama"):
        model, tokenizer, kind = _tiny(name, seed)
        return Backbone(model, tokenizer, kind, spec, {"backbone": name, "seed": seed,
                                                       "weights": "random (test only)"})
    scheme, _, location = name.partition(":")
    if scheme not in ("hf", "slm"):
        raise ValueError(f"backbone spec {spec.name!r}: use hf:<id>, slm:<dir>, tiny-bert, tiny-llama "
                         f"or one of {sorted(NAMED_BACKBONES)}")
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    kwargs = {"revision": spec.revision} if spec.revision else {}
    config = AutoConfig.from_pretrained(location, **kwargs)
    kind = "decoder" if getattr(config, "is_decoder", False) or config.model_type in (
        "llama", "gpt2", "mistral", "qwen2", "gemma", "gemma2", "phi3") else "encoder"
    load = dict(kwargs)
    if spec.attn_implementation:
        load["attn_implementation"] = spec.attn_implementation
    try:
        model = AutoModel.from_pretrained(location, **load)
    except (ValueError, ImportError, TypeError):
        load.pop("attn_implementation", None)       # older architectures without SDPA
        model = AutoModel.from_pretrained(location, **load)
    tokenizer = AutoTokenizer.from_pretrained(location, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    local = Path(location)
    version = {"backbone": name, "revision": spec.revision,
               "weights_sha256_16": _weights_digest(local) if local.is_dir() else None,
               "commit": getattr(config, "_commit_hash", None),
               "model_type": config.model_type, "vocab_size": tokenizer.vocab_size}
    return Backbone(model, tokenizer, kind, spec, version)


def tokenizer_fingerprint(tokenizer) -> str:
    """Identity of a tokenizer's vocabulary and rules, for cache keys.

    Padding and truncation are deliberately excluded: a fast tokenizer records
    them in its own state, and calling it once with ``padding="max_length"``
    changes them — which would give the same tokenizer two fingerprints and
    make every cache lookup after the first one miss.
    """
    try:
        payload = json.loads(tokenizer.backend_tokenizer.to_str())
        for key in ("padding", "truncation"):
            payload.pop(key, None)
        payload = json.dumps(payload, sort_keys=True)
    except (AttributeError, ValueError):
        payload = json.dumps(sorted(tokenizer.get_vocab().items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
