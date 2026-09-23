"""Building the genomic SLM's data: measured, provenance-kept, leakage-audited.

    inventory      what already exists on this machine, measured
    records        ClinVar (+ ERepo) files -> variants / documents / evidence_units / citations
    stats          corpus quantification (docs/slm/GENOMIC_SLM_DATASET.md)
    roles          data roles, reserved evaluation sets, independent validation
    splits         random / variant / text-similarity / laboratory / gene / disease /
                   temporal / functional / MMR splits
    leakage        the 15-check leakage audit and the gate training passes through
    examples       task examples under each task's leakage policy
    features       structured genomic features, fitted on training rows only
    pretrain_corpus  the continued-pretraining text, with evaluation exclusions
"""
