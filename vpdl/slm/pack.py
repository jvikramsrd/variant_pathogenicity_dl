"""Text -> one flat array of token ids per split (``train.bin``, ``val.bin``).

Every document is followed by the end-of-text token, then documents are laid
end to end. Training reads random windows straight from these files through a
memory map, so a 6-billion-token corpus never has to fit in RAM.

uint16 holds ids below 65,536, which covers the 32k vocabulary with room to
spare; a larger vocabulary is refused rather than silently wrapped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from vpdl.slm.corpus import iter_texts
from vpdl.slm.tokenizer import EOS, load_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["pack"]

BATCH_DOCS = 10_000


def pack(corpus_dir: Path | str, tokenizer_path: Path | str, out_dir: Path | str) -> dict:
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() >= 2 ** 16:
        raise ValueError("Vocabulary too large for uint16 token files.")
    eos = tokenizer.token_to_id(EOS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    counts = {}
    for split in ("train", "val"):
        tokens = documents = 0
        with (out / f"{split}.bin").open("wb") as handle:
            batch: list[str] = []

            def flush() -> None:
                nonlocal tokens, documents
                for encoding in tokenizer.encode_batch(batch):
                    ids = np.asarray(encoding.ids + [eos], dtype=np.uint16)
                    handle.write(ids.tobytes())
                    tokens += len(ids)
                documents += len(batch)
                batch.clear()

            for text in iter_texts(corpus_dir, split):
                batch.append(text)
                if len(batch) >= BATCH_DOCS:
                    flush()
                    if split == "train" and documents % 1_000_000 == 0:
                        logger.info("packed %d documents, %.2fB tokens (%.0fs)",
                                    documents, tokens / 1e9, time.time() - started)
            if batch:
                flush()
        counts[split] = {"documents": documents, "tokens": tokens}
        logger.info("%s: %d documents -> %d tokens", split, documents, tokens)

    meta = {
        "splits": counts,
        "dtype": "uint16",
        "eos_id": eos,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "corpus_stats": json.loads((Path(corpus_dir) / "stats.json").read_text()),
        "seconds": round(time.time() - started, 1),
    }
    (out / "data_meta.json").write_text(json.dumps(meta, indent=2))
    # The token files are only meaningful with this exact tokenizer: keep them together.
    if Path(tokenizer_path).resolve() != (out / "tokenizer.json").resolve():
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    return meta
