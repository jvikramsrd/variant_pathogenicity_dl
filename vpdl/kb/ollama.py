"""The local model server (Ollama). Talks to this machine only, and checks the GPU.

Rule 5 of docs/kb/DESIGN.md: questions, passages and answers never leave the DGX.
The client enforces it rather than trusting configuration — a URL pointing
anywhere but loopback is refused before any request is made.

GPU: Ollama decides for itself where a model runs, and when it cannot use the
GPU it runs on the CPU without an error — answers still come, 10-50x slower.
So after each model's first use the client asks Ollama where it actually
loaded (``/api/ps``: ``size_vram`` of ``size``) and stops if any of it is on the
CPU. ``require_gpu=False`` (CLI: ``--allow-cpu``) turns the stop into a warning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Sequence
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["LocalOllama", "DEFAULT_URL", "CONTEXT_TOKENS", "GPU_SHARE_REQUIRED"]

DEFAULT_URL = "http://127.0.0.1:11434"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

# Ollama's default context window is a few thousand tokens and it drops what
# does not fit FROM THE START — the instructions go first, silently. Six
# passages of ~350 words plus instructions need ~4,000 tokens; this leaves room.
CONTEXT_TOKENS = 16384

# `ollama ps` prints "100% GPU" when the whole model is in GPU memory; a split
# such as "20%/80% CPU/GPU" means some layers run on the CPU.
GPU_SHARE_REQUIRED = 0.99

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def _tagged(model: str) -> str:
    """Ollama lists "bge-m3" as "bge-m3:latest"."""
    return model if ":" in model else f"{model}:latest"


class LocalOllama:
    def __init__(self, url: str | None = None, timeout: float = 600.0,
                 require_gpu: bool = True):
        url = (url or os.environ.get("VPDL_OLLAMA_URL") or DEFAULT_URL).rstrip("/")
        host = urlparse(url).hostname
        if host not in _LOOPBACK:
            raise ValueError(
                f"Refusing to use a model server at {url}: the knowledge base only "
                "talks to this machine (docs/kb/DESIGN.md, rule 5).")
        self.url, self.timeout, self.require_gpu = url, timeout, require_gpu
        self.gpu_share: dict[str, float] = {}      # model -> share checked this run

    def _request(self, path: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            model = (payload or {}).get("model")
            if error.code == 404 and "not found" in detail:
                raise RuntimeError(f"Model {model!r} is not installed: "
                                   f"run `ollama pull {model}`.") from error
            raise RuntimeError(f"Ollama error {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Ollama is not reachable at {self.url} ({error.reason}). "
                               "Is it running? `ollama serve` or "
                               "`sudo systemctl start ollama`.") from error

    # -- where is it running? ------------------------------------------------
    def placement(self, model: str) -> float | None:
        """Share of a loaded model held in GPU memory (1.0 = all); None if not loaded."""
        for entry in self._request("/api/ps").get("models", []):
            if _tagged(model) in (_tagged(entry.get("name", "")), _tagged(entry.get("model", ""))):
                size = entry.get("size") or 0
                return float(entry.get("size_vram", 0)) / size if size else None
        return None

    def _check_gpu(self, model: str) -> None:
        if model in self.gpu_share:
            return
        share = self.placement(model)
        if share is None:
            logger.warning("%s: Ollama does not list it as loaded, so where it ran "
                           "cannot be checked.", model)
            return
        self.gpu_share[model] = share
        if share >= GPU_SHARE_REQUIRED:
            logger.info("%s: %.0f%% on GPU", model, 100 * share)
            return
        message = (f"{model} is running {100 * (1 - share):.0f}% on the CPU, not the GPU "
                   "(check with `ollama ps`). Usually Ollama cannot see the GPU: run "
                   "`nvidia-smi`, then `sudo systemctl restart ollama`; see "
                   "docs/kb/RUNBOOK.md, 'Is it on the GPU?'.")
        if self.require_gpu:
            raise RuntimeError(message + " To run on the CPU anyway: --allow-cpu.")
        logger.warning(message)

    # -- the two things the knowledge base asks for ----------------------------
    def embed(self, texts: Sequence[str], model: str) -> np.ndarray:
        response = self._request("/api/embed", {"model": model, "input": list(texts)})
        self._check_gpu(model)
        return np.asarray(response["embeddings"], dtype=np.float32)

    def chat(self, messages: list[dict], model: str, context_tokens: int = CONTEXT_TOKENS) -> str:
        response = self._request("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            # Deterministic: the same question gets the same answer, so an
            # evaluation run measures the model rather than the dice.
            "options": {"temperature": 0, "seed": 0, "num_ctx": context_tokens},
        })
        self._check_gpu(model)
        content = response.get("message", {}).get("content", "")
        return _THINKING.sub("", content).strip()
