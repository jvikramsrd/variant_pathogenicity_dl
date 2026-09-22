"""Continued and variant-aware pretraining of an EXISTING protein language model.

Nothing here trains a protein model from scratch. The pipeline is

    pretrained ESM (P0)  ->  MMR-domain continued MLM (P1)
                         ->  + variant-aware objectives (P2, P3, P4)
                         ->  supervised fine-tuning / probing (vpdl.experiment)

and every arm is scored by the same downstream protocol. See
docs/dl/PRETRAINING_DESIGN.md for the objectives and their leakage rules.
"""
