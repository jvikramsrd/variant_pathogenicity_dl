"""Our own byte-pair vocabulary, trained on the corpus.

Byte-level, so any text round-trips exactly (no unknown tokens; Greek letters,
"≥", accented names all survive). Pre-tokenisation follows GPT-2: letters,
digits and punctuation are split apart before merging, so "c.199G>A" becomes
"c", ".", "199", "G", ">", "A" — every identifier splits at the same kind of
boundary instead of at arbitrary learned points, and digits stay digits.

Special tokens are fixed now, before pretraining, so that the task-tuning
stage can use a chat format without resizing the model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from vpdl.slm.corpus import iter_texts

__all__ = ["SPECIAL_TOKENS", "EOS", "train_tokenizer", "load_tokenizer", "PROBES"]

EOS = "<|endoftext|>"
SPECIAL_TOKENS = [EOS, "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

# Written into tokenizer_meta.json: how the vocabulary splits text we care about.
PROBES = ["MLH1 c.199G>A (p.Gly67Arg)", "Lynch syndrome", "autosomal dominant",
          "colonoscopy every 1-2 years", "microsatellite instability", "BRCA1"]


def train_tokenizer(corpus_dir: Path | str, out_path: Path | str,
                    vocab_size: int = 32_000, sample_bytes: float = 2e9) -> dict:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    stats = json.loads((Path(corpus_dir) / "stats.json").read_text())
    total = sum(s.get("train_characters", 0) for s in stats["sources"].values())
    every = max(1, math.ceil(total / sample_bytes))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(iter_texts(corpus_dir, "train", every=every), trainer=trainer)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "sampled_every_nth_document": every,
        "special_tokens": {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "probes": {text: tokenizer.encode(text).tokens for text in PROBES},
    }
    # Byte-level tokens contain symbols like "Ġ"; never rely on the platform encoding.
    out.with_name("tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_tokenizer(path: Path | str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))
