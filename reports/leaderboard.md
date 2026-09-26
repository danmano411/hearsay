# Leaderboard

Lower minDCF is better (0 = perfect, 1 = trivial). Headline = `test_internal_testlike`.

| model | set | n real/fake | minDCF official | minDCF brief | minDCF combined | EER % | notes |
|---|---|---|---|---|---|---|---|
| R0_raw_tree | val | 6274/11697 | 1.0000 | 0.7961 | 0.8981 | 35.24 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_tree | val_testlike | 6274/2689 | 1.0000 | 0.9461 | 0.9730 | 51.80 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_tree | test_internal_testlike | 6823/2924 | 1.0000 | 0.8399 | 0.9200 | 47.06 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_lgbm | val | 6274/11697 | 0.5163 | 0.3672 | 0.4418 | 12.39 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_raw_lgbm | val_testlike | 6274/2689 | 0.6220 | 0.5257 | 0.5739 | 18.74 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_raw_lgbm | test_internal_testlike | 6823/2924 | 0.6344 | 0.5097 | 0.5720 | 18.78 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_prep_tree | val | 6274/11697 | 1.0000 | 0.9571 | 0.9786 | 47.64 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_tree | val_testlike | 6274/2689 | 1.0000 | 0.9662 | 0.9831 | 51.32 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_tree | test_internal_testlike | 6823/2924 | 1.0000 | 0.9771 | 0.9886 | 53.45 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_lgbm | val | 6274/11697 | 0.9529 | 0.9472 | 0.9501 | 35.59 | LightGBM {'num_leaves': 15, 'min_child_samples': 100}, 9 trivial stats, prep |
| R0_prep_lgbm | val_testlike | 6274/2689 | 0.9526 | 0.9612 | 0.9569 | 38.28 | LightGBM {'num_leaves': 15, 'min_child_samples': 100}, 9 trivial stats, prep |
| R0_prep_lgbm | test_internal_testlike | 6823/2924 | 0.9385 | 0.9721 | 0.9553 | 38.68 | LightGBM {'num_leaves': 15, 'min_child_samples': 100}, 9 trivial stats, prep |
| R1_lgbm_all | val | 6274/11697 | 0.3020 | 0.2612 | 0.2816 | 7.56 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=799, 228 feats |
| R1_lgbm_all | val_testlike | 6274/2689 | 0.2822 | 0.2820 | 0.2821 | 7.93 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=799, 228 feats |
| R1_lgbm_all | test_internal_testlike | 6823/2924 | 0.2933 | 0.3162 | 0.3048 | 7.97 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=799, 228 feats |
| R1_lgbm_spec | val | 6274/11697 | 0.3882 | 0.3130 | 0.3506 | 9.83 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=883, 180 feats |
| R1_lgbm_spec | val_testlike | 6274/2689 | 0.3344 | 0.3450 | 0.3397 | 9.59 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=883, 180 feats |
| R1_lgbm_spec | test_internal_testlike | 6823/2924 | 0.3319 | 0.3593 | 0.3456 | 9.85 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=883, 180 feats |
| R1_lgbm_bio | val | 6274/11697 | 0.6961 | 0.6592 | 0.6777 | 20.05 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=934, 48 feats |
| R1_lgbm_bio | val_testlike | 6274/2689 | 0.7031 | 0.7041 | 0.7036 | 21.16 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=934, 48 feats |
| R1_lgbm_bio | test_internal_testlike | 6823/2924 | 0.7031 | 0.7228 | 0.7129 | 20.65 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=934, 48 feats |
| R1_logreg_all | val | 6274/11697 | 0.6408 | 0.5919 | 0.6164 | 17.26 | standardized LR C=0.01, 228 feats |
| R1_logreg_all | val_testlike | 6274/2689 | 0.6083 | 0.5569 | 0.5826 | 15.77 | standardized LR C=0.01, 228 feats |
| R1_logreg_all | test_internal_testlike | 6823/2924 | 0.5827 | 0.6122 | 0.5975 | 17.03 | standardized LR C=0.01, 228 feats |
| R1_svm_all | val | 6274/11697 | 0.4389 | 0.3841 | 0.4115 | 10.96 | SVM-RBF C=1.0, 15k balanced train subsample |
| R1_svm_all | val_testlike | 6274/2689 | 0.4010 | 0.3737 | 0.3874 | 10.30 | SVM-RBF C=1.0, 15k balanced train subsample |
| R1_svm_all | test_internal_testlike | 6823/2924 | 0.4064 | 0.4195 | 0.4130 | 11.90 | SVM-RBF C=1.0, 15k balanced train subsample |
| R1_lgbm_nolow | val | 6274/11697 | 0.3186 | 0.2644 | 0.2915 | 8.10 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=975, 226 feats (no 0-200 Hz contrast) |
| R1_lgbm_nolow | val_testlike | 6274/2689 | 0.2939 | 0.2782 | 0.2860 | 8.26 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=975, 226 feats (no 0-200 Hz contrast) |
| R1_lgbm_nolow | test_internal_testlike | 6823/2924 | 0.3177 | 0.3284 | 0.3230 | 8.66 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=975, 226 feats (no 0-200 Hz contrast) |
| R0_raw_tree_full | val | 6274/12284 | 0.9296 | 0.6482 | 0.7889 | 50.35 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_tree_full | val_testlike | 6274/2689 | 0.7844 | 0.7693 | 0.7769 | 65.89 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_tree_full | test_internal_testlike | 6823/2924 | 0.8196 | 0.7446 | 0.7821 | 63.24 | depth-3 tree, 9 trivial stats, raw |
| R0_raw_lgbm_full | val | 6274/12284 | 0.4834 | 0.3516 | 0.4175 | 11.72 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_raw_lgbm_full | val_testlike | 6274/2689 | 0.5089 | 0.4848 | 0.4969 | 14.65 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_raw_lgbm_full | test_internal_testlike | 6823/2924 | 0.5279 | 0.5571 | 0.5425 | 16.59 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, raw |
| R0_prep_tree_full | val | 6274/12284 | 0.9751 | 0.9780 | 0.9766 | 91.46 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_tree_full | val_testlike | 6274/2689 | 0.9399 | 0.9705 | 0.9552 | 90.93 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_tree_full | test_internal_testlike | 6823/2924 | 0.9297 | 0.9736 | 0.9516 | 85.98 | depth-3 tree, 9 trivial stats, prep |
| R0_prep_lgbm_full | val | 6274/12284 | 0.9343 | 0.9137 | 0.9240 | 31.48 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, prep |
| R0_prep_lgbm_full | val_testlike | 6274/2689 | 0.9274 | 0.9392 | 0.9333 | 32.61 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, prep |
| R0_prep_lgbm_full | test_internal_testlike | 6823/2924 | 0.9146 | 0.9280 | 0.9213 | 34.02 | LightGBM {'num_leaves': 63, 'min_child_samples': 50}, 9 trivial stats, prep |
| R1_lgbm_all_full | val | 6274/12284 | 0.2895 | 0.2057 | 0.2476 | 6.75 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1065, 228 feats |
| R1_lgbm_all_full | val_testlike | 6274/2689 | 0.2505 | 0.2060 | 0.2283 | 6.21 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1065, 228 feats |
| R1_lgbm_all_full | test_internal_testlike | 6823/2924 | 0.2624 | 0.2434 | 0.2529 | 6.57 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1065, 228 feats |
| R1_lgbm_spec_full | val | 6274/12284 | 0.4118 | 0.2569 | 0.3344 | 9.47 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=851, 180 feats |
| R1_lgbm_spec_full | val_testlike | 6274/2689 | 0.3117 | 0.2528 | 0.2823 | 7.44 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=851, 180 feats |
| R1_lgbm_spec_full | test_internal_testlike | 6823/2924 | 0.2909 | 0.2790 | 0.2849 | 7.93 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=851, 180 feats |
| R1_lgbm_bio_full | val | 6274/12284 | 0.6730 | 0.5328 | 0.6029 | 17.74 | LightGBM {'num_leaves': 63, 'min_child_samples': 50} it=674, 48 feats |
| R1_lgbm_bio_full | val_testlike | 6274/2689 | 0.6414 | 0.5292 | 0.5853 | 16.51 | LightGBM {'num_leaves': 63, 'min_child_samples': 50} it=674, 48 feats |
| R1_lgbm_bio_full | test_internal_testlike | 6823/2924 | 0.6383 | 0.5608 | 0.5996 | 16.65 | LightGBM {'num_leaves': 63, 'min_child_samples': 50} it=674, 48 feats |
| R1_lgbm_nolow_full | val | 6274/12284 | 0.3257 | 0.2210 | 0.2734 | 7.33 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1028, 226 feats |
| R1_lgbm_nolow_full | val_testlike | 6274/2689 | 0.2754 | 0.2171 | 0.2462 | 6.76 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1028, 226 feats |
| R1_lgbm_nolow_full | test_internal_testlike | 6823/2924 | 0.2732 | 0.2665 | 0.2699 | 7.32 | LightGBM {'num_leaves': 255, 'min_child_samples': 20} it=1028, 226 feats |
| R1_logreg_all_full | val | 6274/12284 | 0.6492 | 0.4854 | 0.5673 | 15.84 | standardized LR C=1.0, 228 feats |
| R1_logreg_all_full | val_testlike | 6274/2689 | 0.5608 | 0.4444 | 0.5026 | 13.94 | standardized LR C=1.0, 228 feats |
| R1_logreg_all_full | test_internal_testlike | 6823/2924 | 0.5610 | 0.5019 | 0.5315 | 14.87 | standardized LR C=1.0, 228 feats |
| R5_smoke_spec_bio | val | 6274/12284 | 0.3582 | 0.2358 | 0.2970 | 8.17 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (5-fold cross-fit shown there) |
| R5_smoke_spec_bio | val_testlike | 6274/2689 | 0.2830 | 0.2189 | 0.2510 | 6.99 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (5-fold cross-fit shown there) |
| R5_smoke_spec_bio | test_internal_testlike | 6823/2924 | 0.2734 | 0.2474 | 0.2604 | 7.04 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (5-fold cross-fit shown there) |
| R5_smoke_spec_bio_v2 | val | 6274/12284 | 0.3577 | 0.2392 | 0.2985 | 8.16 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (group 5-fold cross-fit shown there; 'val' mixes cross-fit and full-fit rows; base models early-stopped on val_testlike) |
| R5_smoke_spec_bio_v2 | val_testlike | 6274/2689 | 0.2831 | 0.2218 | 0.2524 | 6.99 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (group 5-fold cross-fit shown there; 'val' mixes cross-fit and full-fit rows; base models early-stopped on val_testlike) |
| R5_smoke_spec_bio_v2 | test_internal_testlike | 6823/2924 | 0.2734 | 0.2474 | 0.2604 | 7.04 | LR fusion of R1_lgbm_spec_full+R1_lgbm_bio_full; fit on val_testlike (group 5-fold cross-fit shown there; 'val' mixes cross-fit and full-fit rows; base models early-stopped on val_testlike) |
