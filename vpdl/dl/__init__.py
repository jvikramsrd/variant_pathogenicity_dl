"""vpdl.dl — the deep-learning branch, built on the v2 protocol.

Everything here plugs into the existing layers rather than replacing them:
labels, splits, thresholds, metrics and provenance still come from
``vpdl.assemble`` / ``vpdl.experiment`` / ``vpdl.evaluate``, so a new DL model
is scored on exactly the held-out variants the existing ``mlp`` / ``bilstm`` /
``gbm`` arms were scored on.

Layout (see docs/dl/DL_ARCHITECTURE.md):

    hgvs, canonical, homology, functional     data: identity, QC, validation sets
    splits, leakage                           partitions and the leakage gate
    context, feature_store, plm/              protein language models, cached
    structure, genomic                        other DL modalities
    fusion, trainer, peft (plm/), pretrain/   models and how they are trained
    calibration, failure, tracking            evaluation beyond discrimination
    interface                                 the frozen output contract

Out of scope, deliberately: ``vpdl.kb`` and ``vpdl.slm`` (the LLM branch).
Nothing in this package imports them.
"""

DL_SCHEMA_VERSION = "dl-output/1.0"

__all__ = ["DL_SCHEMA_VERSION"]
