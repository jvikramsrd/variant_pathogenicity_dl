### Split scheme: `random_debug`

Critical findings: **0**

| check | severity | count | finding |
|---|---|---|---|
| derived_feature_leakage | warning | 1 | ['feature_alphamissense_score']: AlphaMissense thresholds were calibrated on ClinVar; it was trained on population frequency and structure, not ClinVar labels. |
| sequence_similarity | warning | 1 | fold random_debug:0: training proteins >= 20% identical to the held-out protein: {'P40692-P54278': 0.259, 'P43246-P52701': 0.263}; 4 test variants have a labelled training variant at the aligned paralog residue (1 with the same substitution) |
| sequence_similarity | warning | 3 | fold random_debug:1: training proteins >= 20% identical to the held-out protein: {'P40692-P54278': 0.259, 'P43246-P52701': 0.263}; 9 test variants have a labelled training variant at the aligned paralog residue (3 with the same substitution) |
| sequence_similarity | warning | 3 | fold random_debug:2: training proteins >= 20% identical to the held-out protein: {'P40692-P54278': 0.259, 'P43246-P52701': 0.263}; 10 test variants have a labelled training variant at the aligned paralog residue (3 with the same substitution) |
| sequence_similarity | warning | 0 | fold random_debug:3: training proteins >= 20% identical to the held-out protein: {'P40692-P54278': 0.259, 'P43246-P52701': 0.263}; 6 test variants have a labelled training variant at the aligned paralog residue (0 with the same substitution) |
| sequence_similarity | warning | 5 | fold random_debug:4: training proteins >= 20% identical to the held-out protein: {'P40692-P54278': 0.259, 'P43246-P52701': 0.263}; 19 test variants have a labelled training variant at the aligned paralog residue (5 with the same substitution) |
| derived_feature_leakage | info | 0 | no feature is derived from a training label source |
| exact_duplicates | info | 0 | no duplicate variant rows |
| feature_leakage | info | 0 | no feature agrees with the label at >= 0.95 |
| functional_overlap | info | 0 | no functional-assay value is a feature |
| hgvs_duplicates | info | 0 | every normalised HGVS p. maps to exactly one variant id |
| hgvs_duplicates | info | 47 | 47 protein changes are backed by several ClinVar records (resolved order-independently in the canonical table) |
| homology_straddle | info | 30 | 30 homology cluster(s) have aligned paralog residues on both sides |
| protein_duplicates | info | 0 | all protein sequences are distinct |
| sequence_similarity | info | 6 | pairwise identity (identical / shorter length) |
| split_isolation | info | 0 | no variant appears on both sides of any fold |
| split_isolation | info | 0 | no (protein, residue) group straddles any fold |
