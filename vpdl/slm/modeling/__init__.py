"""The genomic SLM model and how it is trained (docs/slm/GENOMIC_SLM_ARCHITECTURE.md).

    backbone    pretrained biomedical encoders, our from-scratch decoder, tiny test models
    peft        frozen / LoRA / adapters / last-n / full, reusing the DL branch's strategies
    multitask   text + structured features + optional DL representation -> heads
    losses      soft-target classification, ordinal, ACMG, evidence type, polarity
    data        tokenisation cache and batch assembly
    finetune    supervised (multi-task) training on vpdl.dl.trainer.Trainer
    continued   continued (domain-adaptive) pretraining: MLM for encoders, CLM for decoders
    predict     predictions, MC-dropout samples, embeddings
    sizing      data-driven model-size candidates and memory / time estimates
"""
