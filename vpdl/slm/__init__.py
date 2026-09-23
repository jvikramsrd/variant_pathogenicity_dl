"""The LLM / SLM branch: a small language model, and the genomic reasoning system around it.

**From-scratch pretraining** (built 2026-09-22, runs on the DGX; plan:
``docs/slm/PLAN.md``, runbook ``docs/slm/RUNBOOK.md``; CLI: ``vpdl slm-*``):

    pubmed.py      PubMed abstracts from NLM's baseline (retractions removed)
    corpus.py      abstracts + GeneReviews -> train/validation text shards
    tokenizer.py   our own byte-pair vocabulary
    pack.py        text -> one flat array of token ids per split
    model.py       Llama-style decoder, random weights (small ~110M, medium ~340M)
    train.py       pretraining loop: resumable, logged, measured

**The broad genomic SLM** (built 2026-09-23; docs: ``docs/slm/GENOMIC_SLM_*.md``;
CLI: ``vpdl-slm``). A model of variant evidence, not of one gene panel — MMR /
Lynch syndrome is a downstream specialisation and evaluation domain:

    schema, labels, variants, clinvar_text, erepo, catalog   identity, terms, readers
    text/        sentences, conclusion masking, ACMG criteria, evidence units, dedup, tiers
    build/       inventory, records, stats, clusters, splits, roles, leakage, examples,
                 features, pretraining corpus, synthetic data
    modeling/    backbones, PEFT, multi-task model, losses, data, continued pretraining,
                 fine-tuning, prediction, sizing
    evaluation/  metrics, calibration, uncertainty, VUS, grounded explanations
    retrieval, teacher, baselines, interface, experiments, hardware, config, cli, smoke

The knowledge base (``vpdl.kb``) is a separate system with its own rules: it
retrieves and cites, and never classifies a variant. This branch's classifier
is a research model; it is not wired into ``vpdl kb-ask``.
"""
