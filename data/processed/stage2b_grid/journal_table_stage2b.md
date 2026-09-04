# Stage-2b ablation grid - full metric panel

Mean over scoreable LOPO holdout genes (MLH1, MSH2, MSH6). PMS2 excluded: n=21
with only 4 negatives, failing the minority-class>=5 availability rule.
Thresholds selected on each fold validation split only; never on holdout.

| cell | n_genes_scored | roc_auc | pr_auc | accuracy | f1_pathogenic | f1_macro | balanced_accuracy | precision | recall | specificity | mcc | brier | ece_uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| esm_frozen_pllr-concat_seed42 | 3 | 0.885 | 0.895 | 0.796 | 0.793 | 0.772 | 0.801 | 0.838 | 0.782 | 0.82 | 0.578 | 0.193 | 0.199 |
| esm_frozen_pllr-off_seed42 | 3 | 0.897 | 0.902 | 0.779 | 0.768 | 0.75 | 0.787 | 0.859 | 0.74 | 0.834 | 0.558 | 0.182 | 0.188 |
| esm_frozen_pllr-residual_seed42 | 3 | 0.912 | 0.914 | 0.752 | 0.763 | 0.727 | 0.772 | 0.806 | 0.777 | 0.767 | 0.522 | 0.201 | 0.231 |
| esm_full_pllr-residual_seed42 | 3 | 0.913 | 0.921 | 0.763 | 0.812 | 0.715 | 0.747 | 0.76 | 0.936 | 0.557 | 0.555 | 0.173 | 0.209 |
| esm_last2_pllr-residual_seed42 | 3 | 0.91 | 0.909 | 0.805 | 0.817 | 0.783 | 0.821 | 0.849 | 0.809 | 0.834 | 0.602 | 0.199 | 0.21 |
| esmpri_concat_frozen_pllr-concat_seed42 | 3 | 0.935 | 0.944 | 0.824 | 0.824 | 0.793 | 0.797 | 0.84 | 0.849 | 0.744 | 0.624 | 0.136 | 0.166 |
| esmpri_concat_frozen_pllr-off_seed42 | 3 | 0.935 | 0.945 | 0.848 | 0.836 | 0.822 | 0.831 | 0.899 | 0.81 | 0.853 | 0.67 | 0.128 | 0.159 |
| esmpri_concat_frozen_pllr-residual_seed42 | 3 | 0.934 | 0.943 | 0.777 | 0.796 | 0.756 | 0.806 | 0.847 | 0.798 | 0.814 | 0.582 | 0.169 | 0.212 |
| esmpri_concat_frozen_pllr-residual_seed43 | 3 | 0.915 | 0.921 | 0.792 | 0.788 | 0.763 | 0.783 | 0.829 | 0.803 | 0.763 | 0.573 | 0.197 | 0.203 |
| esmpri_concat_frozen_pllr-residual_seed44 | 3 | 0.927 | 0.933 | 0.63 | 0.621 | 0.626 | 0.722 | 0.861 | 0.595 | 0.849 | 0.465 | 0.181 | 0.19 |
| esmpri_concat_full_pllr-off_seed42 | 3 | 0.904 | 0.92 | 0.829 | 0.837 | 0.792 | 0.793 | 0.829 | 0.87 | 0.717 | 0.608 | 0.176 | 0.19 |
| esmpri_concat_full_pllr-residual_seed42 | 3 | 0.932 | 0.932 | 0.845 | 0.859 | 0.819 | 0.824 | 0.827 | 0.922 | 0.726 | 0.673 | 0.151 | 0.227 |
| esmpri_concat_full_pllr-residual_seed43 | 3 | 0.91 | 0.926 | 0.812 | 0.812 | 0.786 | 0.805 | 0.851 | 0.826 | 0.783 | 0.619 | 0.126 | 0.138 |
| esmpri_concat_full_pllr-residual_seed44 | 3 | 0.917 | 0.933 | 0.863 | 0.863 | 0.838 | 0.841 | 0.877 | 0.866 | 0.816 | 0.691 | 0.174 | 0.202 |
| esmpri_concat_last2_pllr-residual_seed42 | 3 | 0.919 | 0.924 | 0.846 | 0.836 | 0.814 | 0.818 | 0.892 | 0.803 | 0.833 | 0.646 | 0.203 | 0.243 |
| esmpri_gatewave_frozen_pllr-residual_seed42 | 3 | 0.943 | 0.947 | 0.826 | 0.829 | 0.797 | 0.804 | 0.842 | 0.856 | 0.751 | 0.63 | 0.212 | 0.267 |
