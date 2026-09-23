"""Clinical text processing for the genomic SLM (docs/slm/GENOMIC_SLM_REASONING.md).

    sentences    sentence spans with exact character offsets
    conclusion   find and remove sentences that state the verdict
    acmg         ACMG/AMP criteria: catalogue, parsing, masking, gene specs, task policies
    evidence     sentences -> evidence units (role, types, polarity, codes) by rules
    dedup        exact / near-duplicate / laboratory-template clusters (MinHash + Jaccard)
    quality      context tiers and third-party text restrictions
"""
