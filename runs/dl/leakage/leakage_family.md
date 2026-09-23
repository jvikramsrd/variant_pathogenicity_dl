### Split scheme: `family`

Critical findings: **0**

| check | severity | count | finding |
|---|---|---|---|
| derived_feature_leakage | warning | 1 | ['feature_alphamissense_score']: AlphaMissense thresholds were calibrated on ClinVar; it was trained on population frequency and structure, not ClinVar labels. |
| derived_feature_leakage | info | 0 | gene-constant features removed before training (they encode gene identity): none |
| derived_feature_leakage | info | 0 | no feature is derived from a training label source |
| exact_duplicates | info | 0 | no duplicate variant rows |
| feature_leakage | info | 0 | no feature agrees with the label at >= 0.95 |
| functional_overlap | info | 0 | no functional-assay value is a feature |
| hgvs_duplicates | info | 0 | every normalised HGVS p. maps to exactly one variant id |
| hgvs_duplicates | info | 47 | 47 protein changes are backed by several ClinVar records (resolved order-independently in the canonical table) |
| homology_straddle | info | 0 | 0 homology cluster(s) have aligned paralog residues on both sides |
| protein_duplicates | info | 0 | all protein sequences are distinct |
| sequence_similarity | info | 6 | pairwise identity (identical / shorter length) |
| sequence_similarity | info | 0 | fold family:MutL: training proteins >= 20% identical to the held-out protein: none; 0 test variants have a labelled training variant at the aligned paralog residue (0 with the same substitution) |
| sequence_similarity | info | 0 | fold family:MutS: training proteins >= 20% identical to the held-out protein: none; 0 test variants have a labelled training variant at the aligned paralog residue (0 with the same substitution) |
| split_isolation | info | 0 | no variant appears on both sides of any fold |
| split_isolation | info | 0 | no (protein, residue) group straddles any fold |
