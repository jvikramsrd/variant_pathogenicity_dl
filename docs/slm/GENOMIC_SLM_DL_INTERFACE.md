# DL ↔ SLM interface

Code: `vpdl/slm/interface.py`. The DL branch's side (`vpdl/dl/interface.py`)
was **not modified** — not its schema, its datasets, its evaluation or its
models.

## 1. What the DL branch already froze

It defined one shape per variant (`ModalityRepresentation`), its own instance
(`DLRepresentation`), a JSON Schema with a dependency-free validator, and — by
name, with no implementation — `ReasoningRepresentation`, "the SAME shape,
named, so a future fusion layer can accept both without special cases".

This branch is that reasoning branch, so **`SLMRepresentation` subclasses
`ReasoningRepresentation`** and keeps `modality = "reasoning"`. Nothing had to
change on the DL side for the two to meet.

## 2. Reading the DL branch's output

```python
dl = DLInput.from_outputs("runs/dl/export/dl_outputs_<cell>.jsonl")
embeddings, mask = dl.matrix(protein_variant_ids)   # [N, D], [N]
dl.fold_check(protein_variant_ids, genes)
```

* **Dimensions come from the records**, never hard-coded (the shipped fusion
  config happens to give 128; the reader takes what the file says, and refuses
  a file whose embeddings differ in width).
* **A variant with no DL record gets zeros and `mask = 0`** — never an imputed
  embedding. Most ClinVar variants are not protein substitutions on the DL
  panel, so this is the normal case.
* **The join key is the DL branch's own identity.** `protein_variant_id`
  (`UniProt:pos:wt>mut`) is filled from the DL canonical table's
  `clinvar_variation_ids` column, so the two branches agree by construction
  rather than by re-deriving coordinates here.
* **Out-of-fold is checked, not assumed.** A DL score is out-of-gene for its
  own gene only. `fold_check` counts how many joined variants have a DL record
  whose fold equals their gene (`out_of_fold`) and how many do not
  (`in_fold_or_other`) — a number the leakage report carries.

## 3. What the SLM emits

`write_slm_outputs` writes JSONL plus `.npy` arrays, mirroring the DL branch's
own writer, and `SLM_OUTPUT_SCHEMA` is validated on every record:

```json
{
  "schema_version": "slm-genomic/1.0",
  "variant_id": "clinvar:12345",
  "protein_variant_id": "P40692:67:G>R",
  "gene": "MLH1",
  "slm_embedding": {"file": "slm_outputs.embeddings.npy", "row": 0},
  "slm_embedding_dim": 256,
  "evidence_embedding": null,
  "class_probabilities": {"pathogenic": …, "likely_pathogenic": …, "vus": …,
                          "likely_benign": …, "benign": …},
  "pathogenic_probability": 0.71, "benign_probability": 0.08, "vus_probability": 0.21,
  "uncertainty": 0.93, "uncertainty_method": "predictive_entropy",
  "abstained": false,
  "evidence": [ … ], "acmg": [ … ], "explanation": "… [E1].",
  "model_version": "…", "feature_version": "…", "dataset_version": "…",
  "split": "test", "calibration": "temperature", "quality_flags": []
}
```

Semantics, in the DL contract's terms: `score` = `pathogenic_probability` =
p(P) + p(LP) after calibration, **not** renormalised against VUS (a variant the
model calls uncertain has a low score on purpose); `uncertainty` = predictive
entropy in nats, or an ensemble/MC-dropout standard deviation when
`uncertainty_method` says so; `embedding` = the fused representation.
Probabilities must name all five classes and sum to 1, or the record is
refused.

## 4. What this interface deliberately does not do

It does not fuse anything. No model in this repository takes a
`DLRepresentation` and an `SLMRepresentation` and produces a joint prediction.
The DL representation enters the SLM only as one optional input modality
(EXP-018, an ablation of the SLM, off by default).

The final multimodal fusion — DL + SLM + structured features — is **out of
scope for this phase and is not implemented**. When it is built, the DL
branch's design says how: register each modality with a name and a width in
`vpdl.dl.fusion.build_fusion_net`, which needs no change to either encoder.

## 5. Checks that protect the join

* embedding width consistent within a file, and taken from the file;
* `model_version`, `feature_version`, `dataset_version` carried from the DL
  export into the SLM run record, so a mismatch is visible in the registry;
* missing DL records masked, never imputed (unit-tested: random values behind
  a zero mask cannot change the logits);
* fold compatibility reported for every joined variant.
