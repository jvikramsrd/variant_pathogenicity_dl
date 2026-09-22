"""The model: a Llama-style decoder with random weights — trained from scratch.

Built as a Hugging Face ``LlamaForCausalLM`` only for its standard layout
(RoPE positions, RMSNorm, SwiGLU feed-forward) and because that layout exports
to GGUF, which Ollama runs — so the finished model is evaluated by the same
``vpdl kb-eval`` as medgemma, GPU check included. No pretrained weights are
loaded anywhere: ``build`` seeds and initialises every parameter itself.

Sizes (with the 32k vocabulary; input and output embeddings are shared):

    tiny     ~0.1M   tests only
    small    ~110M   12 layers x 768   — the pipeline check (~20 hours on the DGX, measured)
    medium   ~340M   24 layers x 1024  — the real run (~7 days, estimated)
"""

from __future__ import annotations

__all__ = ["SIZES", "DEFAULTS", "build", "count_parameters", "flops_per_token"]

# Per-size defaults: peak learning rate as in GPT-3's table for similar sizes, and
# training tokens at ~20 per parameter (the Chinchilla rule).
DEFAULTS = {
    "tiny": dict(lr=3e-3, tokens=1e6),
    "small": dict(lr=6e-4, tokens=2.5e9),
    "medium": dict(lr=3e-4, tokens=7e9),
}

SIZES = {
    "tiny": dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=176),
    "small": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                  intermediate_size=2048),
    "medium": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                   intermediate_size=2816),
}


def _config(size: str, vocab_size: int, context: int, special: dict[str, int]):
    from transformers import LlamaConfig
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; use one of {sorted(SIZES)}.")
    return LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=context,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        bos_token_id=special.get("<|endoftext|>"),
        eos_token_id=special.get("<|endoftext|>"),
        pad_token_id=special.get("<|pad|>"),
        attention_bias=False,
        mlp_bias=False,
        **SIZES[size],
    )


def build(size: str, vocab_size: int, context: int, special: dict[str, int] | None = None,
          seed: int = 0):
    """A freshly initialised model. Same seed, same starting weights."""
    import torch
    from transformers import LlamaForCausalLM

    config = _config(size, vocab_size, context, special or {})
    try:
        config._attn_implementation = "sdpa"  # PyTorch's fused attention; recent versions default to it
    except (AttributeError, ValueError):
        pass
    torch.manual_seed(seed)
    return LlamaForCausalLM(config)


def count_parameters(size: str, vocab_size: int = 32_000, context: int = 4096) -> int:
    """Parameter count without allocating the weights (built on the meta device)."""
    import torch
    from transformers import LlamaForCausalLM

    with torch.device("meta"):
        model = LlamaForCausalLM(_config(size, vocab_size, context, {}))
    return sum(p.numel() for p in model.parameters())


def flops_per_token(size: str, vocab_size: int, context: int) -> float:
    """Training FLOPs per token: 6 x parameters, plus attention over the context.

    The attention term (12 x layers x width x context, as in the PaLM paper's
    appendix, not halved for causal masking) matters at this size: for the
    small model it adds ~35% at a 2,048-token context and ~70% at 4,096, which
    the plain 6 x parameters rule hides. That is why pretraining runs at 2,048
    and the context is extended to 4,096 during task tuning, where the reader
    actually needs it.
    """
    shape = SIZES[size]
    return (6 * count_parameters(size, vocab_size, context)
            + 12 * shape["num_hidden_layers"] * shape["hidden_size"] * context)
