"""The genomic SLM: a text backbone, structured features, an optional DL representation, many heads.

    narrative text ──> backbone ──> pooled text vector ─┐
    structured fields ──> embeddings + scaled numbers ──┼─> fuse ──> slm_embedding ──> class head (5)
    DLRepresentation (optional, masked) ─────────────────┘                        └─> ACMG head (28, multi-label)
    one evidence sentence ──> backbone ──> unit vector ──> evidence-type head (multi-label)
                                                        └─> polarity head (4)

Heads are switched on per experiment (``HeadConfig.tasks``): classification
only (EXP-009/010), + evidence (EXP-012), + ACMG (EXP-013), all (EXP-014).
The DL representation enters as its own modality with a presence bit — a
variant without one (every non-missense variant) gets zeros and bit 0, never
an imputed embedding — and can be dropped at random during training
(``dl_dropout``) so the text path cannot come to depend on it.

This module builds the model; it does NOT fuse DL and SLM as the project's
final multimodal model (docs/slm/GENOMIC_SLM_DL_INTERFACE.md): the DL input
here is an ablation arm (EXP-018) of the SLM, gated off by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from vpdl.slm.schema import CLASSES
from vpdl.slm.text.acmg import CODES, GENE_SPECS, applicable_codes
from vpdl.slm.text.evidence import EVIDENCE_TYPES, POLARITIES

__all__ = ["HeadConfig", "GenomicSLM", "applicability_mask", "TASK_HEADS"]

TASK_HEADS = ("classify", "acmg", "evidence")


@dataclass
class HeadConfig:
    tasks: tuple[str, ...] = ("classify",)
    embedding_dim: int = 256
    dropout: float = 0.1
    cat_cardinalities: tuple[int, ...] = ()
    cat_dim: int = 16
    n_numeric: int = 0
    dl_dim: int | None = None
    dl_dropout: float = 0.0
    n_classes: int = len(CLASSES)
    acmg_codes: tuple[str, ...] = CODES
    evidence_types: tuple[str, ...] = EVIDENCE_TYPES
    polarities: tuple[str, ...] = POLARITIES
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [t for t in self.tasks if t not in TASK_HEADS]
        if unknown:
            raise ValueError(f"unknown head(s) {unknown}; known {TASK_HEADS}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def applicability_mask(genes: Sequence[str], codes: Sequence[str] = CODES):
    """[B, C] float mask: 1 where a code may be predicted for that gene (GENE_SPECS)."""
    import torch
    cache: dict[str, tuple[str, ...]] = {}
    rows = []
    for gene in genes:
        if gene not in cache:
            cache[gene] = applicable_codes(gene, GENE_SPECS)
        allowed = set(cache[gene])
        rows.append([1.0 if code in allowed else 0.0 for code in codes])
    return torch.tensor(rows, dtype=torch.float32)


def _module_class():
    import torch
    from torch import nn

    class GenomicSLM(nn.Module):
        def __init__(self, backbone, config: HeadConfig):
            super().__init__()
            self.backbone_wrapper = backbone
            self.backbone = backbone.model            # registered, so .to()/state_dict see it
            self.config = config
            text_dim = backbone.hidden_size
            self.categorical = nn.ModuleList(nn.Embedding(n, config.cat_dim)
                                             for n in config.cat_cardinalities)
            width = text_dim + config.cat_dim * len(config.cat_cardinalities) + 2 * config.n_numeric
            if config.dl_dim:
                width += config.dl_dim + 1
                self.dl_norm = nn.LayerNorm(config.dl_dim)
            self.fuse = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.embedding_dim),
                                      nn.GELU(), nn.Dropout(config.dropout))
            self.class_head = nn.Linear(config.embedding_dim, config.n_classes)
            if "acmg" in config.tasks:
                self.acmg_head = nn.Linear(config.embedding_dim, len(config.acmg_codes))
            if "evidence" in config.tasks:
                self.unit_proj = nn.Sequential(nn.LayerNorm(text_dim),
                                               nn.Linear(text_dim, config.embedding_dim), nn.GELU(),
                                               nn.Dropout(config.dropout))
                self.type_head = nn.Linear(config.embedding_dim, len(config.evidence_types))
                self.polarity_head = nn.Linear(config.embedding_dim, len(config.polarities))

        def encode_text(self, input_ids, attention_mask):
            return self.backbone_wrapper.encode(input_ids, attention_mask)

        def forward(self, input_ids, attention_mask, categorical=None, numeric=None,
                    numeric_mask=None, dl_embedding=None, dl_mask=None, **_):
            text = self.encode_text(input_ids, attention_mask).float()
            parts = [text]
            for j, embedding in enumerate(self.categorical):
                parts.append(embedding(categorical[:, j]))
            if self.config.n_numeric:
                parts += [numeric.float() * numeric_mask.float(), numeric_mask.float()]
            if self.config.dl_dim:
                present = dl_mask.float().unsqueeze(-1)
                if self.training and self.config.dl_dropout > 0:
                    keep = (torch.rand_like(present) >= self.config.dl_dropout).float()
                    present = present * keep
                parts += [self.dl_norm(dl_embedding.float()) * present, present]
            embedding = self.fuse(torch.cat(parts, dim=-1))
            out = {"embedding": embedding, "text_embedding": text,
                   "class_logits": self.class_head(embedding)}
            if "acmg" in self.config.tasks:
                out["acmg_logits"] = self.acmg_head(embedding)
            return out

        def forward_units(self, input_ids, attention_mask):
            if "evidence" not in self.config.tasks:
                raise RuntimeError("evidence heads are not enabled in this model")
            unit = self.unit_proj(self.encode_text(input_ids, attention_mask).float())
            return {"unit_embedding": unit, "type_logits": self.type_head(unit),
                    "polarity_logits": self.polarity_head(unit)}

    return GenomicSLM


def GenomicSLM(backbone, config: HeadConfig):         # noqa: N802 — a factory named as the class
    return _module_class()(backbone, config)
