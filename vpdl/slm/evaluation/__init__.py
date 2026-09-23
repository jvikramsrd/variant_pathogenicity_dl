"""Evaluation of the genomic SLM (docs/slm/GENOMIC_SLM_EXPERIMENT_PLAN.md).

    metrics       five-class, pathogenic-vs-benign, evidence multi-label, span F1
    calibration   temperature / vector scaling (validation only), ECE, Brier, per-group
    uncertainty   entropy, mutual information, abstention, risk-coverage
    vus           VUS ranking against later reclassification, enrichment, sub-tiers
    explain       structured grounded explanations and the grounding check
"""
