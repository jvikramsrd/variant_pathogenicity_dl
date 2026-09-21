"""The local model server (Ollama). Talks to this machine only.

Rule 5 of docs/kb/DESIGN.md: questions, passages and answers never leave the DGX.
The client enforces it rather than trusting configuration — a URL pointing
anywhere but loopback is refused before any request is made.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Sequence
from urllib.parse import urlparse

import numpy as np

__all__ = ["LocalOllama", "DEFAULT_URL", "CONTEXT_TOKENS"]

DEFAULT_URL = "http://127.0.0.1:11434"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

# Ollama's default context window is a few thousand tokens and it drops what
# does not fit FROM THE START — the instructions go first, silently. Six
# passages of ~350 words plus instructions need ~4,000 tokens; this leaves room.
CONTEXT_TOKENS = 16384

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalOllama:
    def __init__(self, url: str | None = None, timeout: float = 600.0):
        url = (url or os.environ.get("VPDL_OLLAMA_URL") or DEFAULT_URL).rstrip("/")
        host = urlparse(url).hostname
        if host not in _LOOPBACK:
            raise ValueError(
                f"Refusing to use a model server at {url}: the knowledge base only "
                "talks to this machine (docs/kb/DESIGN.md, rule 5).")
        self.url, self.timeout = url, timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code == 404 and "not found" in detail:
                raise RuntimeError(f"Model {payload.get('model')!r} is not installed: "
                                   f"run `ollama pull {payload.get('model')}`.") from error
            raise RuntimeError(f"Ollama error {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Ollama is not reachable at {self.url} ({error.reason}). "
                               "Is it running? `ollama serve` or "
                               "`sudo systemctl start ollama`.") from error

    def embed(self, texts: Sequence[str], model: str) -> np.ndarray:
        response = self._post("/api/embed", {"model": model, "input": list(texts)})
        return np.asarray(response["embeddings"], dtype=np.float32)

    def chat(self, messages: list[dict], model: str, context_tokens: int = CONTEXT_TOKENS) -> str:
        response = self._post("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            # Deterministic: the same question gets the same answer, so an
            # evaluation run measures the model rather than the dice.
            "options": {"temperature": 0, "seed": 0, "num_ctx": context_tokens},
        })
        content = response.get("message", {}).get("content", "")
        return _THINKING.sub("", content).strip()
