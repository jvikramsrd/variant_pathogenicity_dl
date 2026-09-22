"""A small language model trained from scratch. Plan: ``docs/slm/PLAN.md``.

    pubmed.py      PubMed abstracts from NLM's baseline (retractions removed)
    corpus.py      abstracts + GeneReviews -> train/validation text shards
    tokenizer.py   our own byte-pair vocabulary
    pack.py        text -> one flat array of token ids per split
    model.py       Llama-style decoder, random weights (small ~110M, medium ~340M)
    train.py       pretraining loop: resumable, logged, measured

How to run it: ``docs/slm/RUNBOOK.md``.
"""
