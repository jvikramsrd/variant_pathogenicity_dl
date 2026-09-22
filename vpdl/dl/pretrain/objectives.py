"""Label-free pretraining objectives, each switched on by its own weight.

    mlm                 masked residue prediction on wild-type crops (ESM's
                        15% / 80-10-10 recipe)
    mutation_position   given a crop with ONE synthetic substitution, find it
                        (a per-token logit, softmax over the crop's residues)
    substitution        at the substituted position, predict the ORIGINAL
                        residue from the variant context
    contrastive         WT/VT triplet on site embeddings: a conservative
                        substitution (BLOSUM62 >= 1) must stay closer to the
                        wild type than a radical one (<= -2), by a margin
    paired              WT/VT paired regression: predict the substitution's
                        BLOSUM62 score from [h_wt, h_vt, h_vt - h_wt]

Leakage rules, enforced by construction:

* **No clinical or assay label enters any objective.** Substitutions are
  synthetic and uniform over positions; severity comes from BLOSUM62, a generic
  matrix with no knowledge of these genes' variants.
* **Positions are uniform, never ClinVar positions.** Choosing where to mutate
  from clinical data would leak where pathogenic variants cluster.

The contrastive and paired objectives inject BLOSUM62's view of severity;
whether that helps, hurts, or merely re-learns what the model knew is exactly
what the P2/P3/P4 ablation measures against P0/P1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from vpdl.dl.homology import BLOSUM62
from vpdl.dl.plm.backbones import AA20

__all__ = ["OBJECTIVES", "ObjectiveWeights", "mlm_mask", "sample_substitutions",
           "build_pretrain_model"]

OBJECTIVES = ("mlm", "mutation_position", "substitution", "contrastive", "paired")


@dataclass
class ObjectiveWeights:
    # Every objective defaults OFF, MLM included: an arm is exactly the
    # objectives it names (P0 names none and trains nothing).
    mlm: float = 0.0
    mutation_position: float = 0.0
    substitution: float = 0.0
    contrastive: float = 0.0
    paired: float = 0.0
    mask_rate: float = 0.15
    contrastive_margin: float = 0.2

    def active(self) -> list[str]:
        return [name for name in OBJECTIVES if getattr(self, name) > 0]

    @classmethod
    def from_dict(cls, value: Mapping | None) -> "ObjectiveWeights":
        value = dict(value or {})
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown objective settings {sorted(unknown)}; objectives are "
                             f"{OBJECTIVES}")
        return cls(**value)


_CONSERVATIVE = {a: [b for b in AA20 if b != a and BLOSUM62[(a, b)] >= 1] for a in AA20}
_RADICAL = {a: [b for b in AA20 if BLOSUM62[(a, b)] <= -2] for a in AA20}


def mlm_mask(ids, attention, alphabet, rate: float, generator):
    """ESM/BERT masking: of the chosen residues 80% <mask>, 10% random, 10% kept."""
    import torch

    special = (ids == alphabet.cls) | (ids == alphabet.eos) | (ids == alphabet.pad)
    chosen = (torch.rand(ids.shape, generator=generator) < rate) & ~special
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    roll = torch.rand(ids.shape, generator=generator)
    inputs = ids.clone()
    inputs[chosen & (roll < 0.8)] = alphabet.mask
    random_ids = torch.tensor(alphabet.aa_ids)[
        torch.randint(0, 20, ids.shape, generator=generator)]
    swap = chosen & (roll >= 0.8) & (roll < 0.9)
    inputs[swap] = random_ids[swap]
    return inputs, labels


def sample_substitutions(sequences: Sequence[str], rng: np.random.Generator,
                         kind: str = "uniform") -> list[tuple[int, str, str]]:
    """One ``(pos0, wt, mut)`` per sequence at a uniform standard residue.

    ``kind`` = uniform | conservative | radical (BLOSUM62 classes).
    """
    out = []
    for sequence in sequences:
        candidates = [i for i, c in enumerate(sequence) if c in AA20]
        pos0 = int(rng.choice(candidates))
        wt = sequence[pos0]
        pool = {"uniform": [a for a in AA20 if a != wt], "conservative": _CONSERVATIVE[wt],
                "radical": _RADICAL[wt]}[kind] or [a for a in AA20 if a != wt]
        out.append((pos0, wt, str(rng.choice(pool))))
    return out


def build_pretrain_model(backbone, weights: ObjectiveWeights):
    """Backbone (with PEFT applied) + the heads the active objectives need."""
    import torch
    from torch import nn

    d = backbone.model.config.hidden_size
    alphabet = backbone.alphabet
    aa_index = {a: i for i, a in enumerate(AA20)}

    class PretrainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm = backbone.model
            self.position_head = nn.Linear(d, 1) if weights.mutation_position else None
            self.substitution_head = nn.Linear(d, 20) if weights.substitution else None
            self.paired_head = (nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 1))
                                if weights.paired else None)

        def _hidden(self, ids, attention):
            return self.lm.esm(input_ids=ids, attention_mask=attention).last_hidden_state

        def losses(self, crops: Sequence[str], rng: np.random.Generator,
                   generator) -> dict[str, "torch.Tensor"]:
            device = next(self.parameters()).device
            out: dict[str, torch.Tensor] = {}
            ids, attention = alphabet.encode(crops)
            ids, attention = ids.to(device), attention.to(device)
            if weights.mlm:
                inputs, labels = mlm_mask(ids.cpu(), attention.cpu(), alphabet,
                                          weights.mask_rate, generator)
                logits = self.lm(input_ids=inputs.to(device), attention_mask=attention).logits
                out["mlm"] = nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.to(device).reshape(-1), ignore_index=-100)
            rows = torch.arange(len(crops), device=device)
            if weights.mutation_position or weights.substitution or weights.paired:
                subs = sample_substitutions(crops, rng)
                mutated = [c[:p] + m + c[p + 1:] for c, (p, _, m) in zip(crops, subs)]
                vt_ids, vt_att = alphabet.encode(mutated)
                h_vt = self._hidden(vt_ids.to(device), vt_att.to(device))
                sites = torch.tensor([p + 1 for p, _, _ in subs], device=device)
                if weights.mutation_position:
                    token_logits = self.position_head(h_vt).squeeze(-1).float()
                    residue = vt_att.to(device).bool() & (vt_ids.to(device) != alphabet.cls) \
                        & (vt_ids.to(device) != alphabet.eos)
                    token_logits = token_logits.masked_fill(~residue, -1e4)
                    out["mutation_position"] = nn.functional.cross_entropy(token_logits, sites)
                if weights.substitution:
                    wt_targets = torch.tensor([aa_index[w] for _, w, _ in subs], device=device)
                    out["substitution"] = nn.functional.cross_entropy(
                        self.substitution_head(h_vt[rows, sites]).float(), wt_targets)
                if weights.paired:
                    h_wt = self._hidden(ids, attention)[rows, sites]
                    pair = torch.cat([h_wt, h_vt[rows, sites], h_vt[rows, sites] - h_wt], -1)
                    target = torch.tensor([BLOSUM62[(w, m)] / 4.0 for _, w, m in subs],
                                          device=device)
                    out["paired"] = nn.functional.mse_loss(
                        self.paired_head(pair.float()).squeeze(-1), target)
            if weights.contrastive:
                conservative = sample_substitutions(crops, rng, "conservative")
                radical = [(p, w, str(rng.choice(_RADICAL[w] or [a for a in AA20 if a != w])))
                           for p, w, _ in conservative]
                sites = torch.tensor([p + 1 for p, _, _ in conservative], device=device)

                def site_embedding(subs):
                    seqs = [c[:p] + m + c[p + 1:] for c, (p, _, m) in zip(crops, subs)]
                    vids, vatt = alphabet.encode(seqs)
                    return self._hidden(vids.to(device), vatt.to(device))[rows, sites]

                anchor = self._hidden(ids, attention)[rows, sites]
                close = 1 - nn.functional.cosine_similarity(anchor, site_embedding(conservative))
                far = 1 - nn.functional.cosine_similarity(anchor, site_embedding(radical))
                out["contrastive"] = torch.relu(weights.contrastive_margin + close - far).mean()
            return out

        def weighted(self, parts: Mapping[str, "torch.Tensor"]) -> "torch.Tensor":
            return sum(getattr(weights, name) * value.float() for name, value in parts.items())

    return PretrainModel()
