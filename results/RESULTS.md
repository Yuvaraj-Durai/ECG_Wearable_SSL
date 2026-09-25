# Results: reproduction of Lai et al., Nat. Commun. 14:3741 (2023) on public data

Chapman-Shaoxing/Ningbo: 62 SNOMED-CT terms with ≥ 50 positives; split 36120/4515/4515 (train/val/test). Pre-training: CODE-15% + Chapman training split. See README.md for every deviation from the paper.

## Table 1. Per-term diagnostic performance of MSDNN with PW&Aug (seed 0)

Clean = Chapman test split (n = 4515); noisy = the same ECGs with real NSTDB artifacts (the stand-in for the paper's online wearable test). Thresholds: per-class F1-optimal on the validation split.

| Term | Name | Samples (all) | Test pos. | AUROC | AUPRC | F1 | TP | TN | FP | FN | Sen | Spe | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SB | sinus bradycardia | 16559 | 1677 | 0.999 | 0.998 | 0.984 | 1634 | 2809 | 29 | 43 | 0.974 | 0.990 | 0.978 |
| NSR | sinus rhythm | 8125 | 810 | 0.984 | 0.954 | 0.880 | 660 | 3634 | 71 | 150 | 0.815 | 0.981 | 0.857 |
| AFL | atrial flutter | 8060 | 798 | 0.971 | 0.818 | 0.845 | 759 | 3469 | 248 | 39 | 0.951 | 0.933 | 0.841 |
| STach | sinus tachycardia | 7254 | 736 | 0.996 | 0.980 | 0.937 | 695 | 3725 | 54 | 41 | 0.944 | 0.986 | 0.936 |
| TAb | t wave abnormal | 7043 | 675 | 0.906 | 0.611 | 0.614 | 382 | 3571 | 269 | 293 | 0.566 | 0.930 | 0.576 |
| LVHV | left ventricular high voltage | 5401 | 575 | 0.963 | 0.802 | 0.718 | 480 | 3601 | 339 | 95 | 0.835 | 0.914 | 0.689 |
| STC | s t changes | 4232 | 448 | 0.895 | 0.449 | 0.494 | 262 | 3682 | 385 | 186 | 0.585 | 0.905 | 0.479 |
| TInv | t wave inversion | 2877 | 287 | 0.946 | 0.516 | 0.503 | 187 | 3962 | 266 | 100 | 0.652 | 0.937 | 0.505 |
| SA | sinus arrhythmia | 2550 | 243 | 0.988 | 0.852 | 0.773 | 181 | 4222 | 50 | 62 | 0.745 | 0.988 | 0.764 |
| AF | atrial fibrillation | 1780 | 172 | 0.923 | 0.203 | 0.327 | 102 | 3928 | 415 | 70 | 0.593 | 0.904 | 0.296 |
| STD | st depression | 1668 | 165 | 0.944 | 0.399 | 0.463 | 89 | 4203 | 147 | 76 | 0.539 | 0.966 | 0.444 |
| LAD | left axis deviation | 1545 | 167 | 0.985 | 0.707 | 0.631 | 129 | 4249 | 99 | 38 | 0.772 | 0.977 | 0.653 |
| PAC | premature atrial contraction | 1312 | 129 | 0.975 | 0.770 | 0.712 | 91 | 4360 | 26 | 38 | 0.705 | 0.994 | 0.740 |
| PR | pacing rhythm | 1181 | 116 | 1.000 | 0.981 | 0.941 | 108 | 4387 | 12 | 8 | 0.931 | 0.997 | 0.915 |
| NSSTTA | nonspecific st t abnormality | 1158 | 122 | 0.894 | 0.151 | 0.241 | 42 | 4202 | 191 | 80 | 0.344 | 0.957 | 0.237 |
| IAVB | 1st degree av block | 1140 | 110 | 0.983 | 0.732 | 0.648 | 59 | 4385 | 20 | 51 | 0.536 | 0.995 | 0.624 |
| CRBBB | complete right bundle branch block | 1096 | 121 | 0.991 | 0.613 | 0.741 | 109 | 4329 | 65 | 12 | 0.901 | 0.985 | 0.739 |
| PVC | premature ventricular contractions | 1091 | 129 | 0.989 | 0.678 | 0.701 | 87 | 4340 | 46 | 42 | 0.674 | 0.990 | 0.664 |
| QAb | qwave abnormal | 1063 | 111 | 0.966 | 0.518 | 0.487 | 51 | 4359 | 45 | 60 | 0.459 | 0.990 | 0.493 |
| LQRSV | low qrs voltages | 1043 | 109 | 0.960 | 0.412 | 0.443 | 36 | 4380 | 26 | 73 | 0.330 | 0.994 | 0.421 |
| RAD | right axis deviation | 853 | 82 | 0.985 | 0.513 | 0.497 | 32 | 4412 | 21 | 50 | 0.390 | 0.995 | 0.474 |
| STIAb | st interval abnormal | 801 | 76 | 0.912 | 0.302 | 0.349 | 32 | 4334 | 105 | 44 | 0.421 | 0.976 | 0.300 |
| NSIVCB | nonspecific intraventricular conduction disorder | 771 | 67 | 0.936 | 0.317 | 0.413 | 32 | 4381 | 67 | 35 | 0.478 | 0.985 | 0.386 |
| SVT | supraventricular tachycardia | 724 | 64 | 0.996 | 0.777 | 0.756 | 47 | 4436 | 15 | 17 | 0.734 | 0.997 | 0.746 |
| CVCL/CCVCL | clockwise or counterclockwise vectorcardiographic loop | 653 | 61 | 0.920 | 0.233 | 0.291 | 11 | 4441 | 13 | 50 | 0.180 | 0.997 | 0.259 |
| RBBB | right bundle branch block | 649 | 57 | 0.987 | 0.407 | 0.483 | 50 | 4362 | 96 | 7 | 0.877 | 0.978 | 0.493 |
| LVH | left ventricular hypertrophy | 647 | 65 | 0.978 | 0.473 | 0.497 | 43 | 4384 | 66 | 22 | 0.662 | 0.985 | 0.494 |
| PRWP | poor R wave Progression | 638 | 57 | 0.963 | 0.204 | 0.289 | 22 | 4385 | 73 | 35 | 0.386 | 0.984 | 0.289 |
| LQT | prolonged qt interval | 394 | 38 | 0.901 | 0.278 | 0.293 | 12 | 4443 | 34 | 26 | 0.316 | 0.992 | 0.286 |
| BBB | bundle branch block | 385 | 38 | 0.958 | 0.303 | 0.292 | 13 | 4435 | 42 | 25 | 0.342 | 0.991 | 0.280 |
| LAnFB | left anterior fascicular block | 380 | 33 | 0.989 | 0.316 | 0.439 | 18 | 4449 | 33 | 15 | 0.545 | 0.993 | 0.429 |
| ERe | early repolarization | 366 | 38 | 0.931 | 0.199 | 0.087 | 1 | 4471 | 6 | 37 | 0.026 | 0.999 | 0.044 |
| ATach | atrial tachycardia | 297 | 35 | 0.931 | 0.102 | 0.211 | 8 | 4430 | 50 | 27 | 0.229 | 0.989 | 0.172 |
| VPB | ventricular premature beats | 294 | 39 | 0.982 | 0.212 | 0.337 | 28 | 4397 | 79 | 11 | 0.718 | 0.982 | 0.384 |
| IRBBB | incomplete right bundle branch block | 246 | 26 | 0.905 | 0.128 | 0.211 | 11 | 4452 | 37 | 15 | 0.423 | 0.992 | 0.297 |
| AVB | av block | 244 | 21 | 0.930 | 0.045 | 0.083 | 6 | 4323 | 171 | 15 | 0.286 | 0.962 | 0.061 |
| LBBB | left bundle branch block | 240 | 25 | 0.986 | 0.235 | 0.150 | 4 | 4484 | 6 | 21 | 0.160 | 0.999 | 0.229 |
| TPW | tall p wave | 215 | 20 | 0.983 | 0.402 | 0.400 | 14 | 4404 | 91 | 6 | 0.700 | 0.980 | 0.224 |
| ARH | atrial rhythm | 215 | 25 | 0.992 | 0.703 | 0.605 | 8 | 4487 | 3 | 17 | 0.320 | 0.999 | 0.444 |
| CLBBB | complete left bundle branch block | 213 | 18 | 0.998 | 0.568 | 0.680 | 16 | 4481 | 16 | 2 | 0.889 | 0.996 | 0.640 |
| STE | st elevation | 176 | 17 | 0.901 | 0.046 | 0.106 | 6 | 4347 | 151 | 11 | 0.353 | 0.966 | 0.069 |
| CCR | countercolockwise rotation | 162 | 7 | 0.927 | 0.027 | 0.103 | 0 | 4489 | 19 | 7 | 0.000 | 0.996 | 0.000 |
| PWC | p wave change | 142 | 9 | 0.861 | 0.061 | 0.111 | 1 | 4482 | 24 | 8 | 0.111 | 0.995 | 0.059 |
| AVJR | atrioventricular junctional rhythm | 139 | 19 | 0.989 | 0.489 | 0.438 | 5 | 4494 | 2 | 14 | 0.263 | 1.000 | 0.385 |
| UAb | u wave abnormal | 136 | 10 | 0.933 | 0.028 | 0.033 | 0 | 4470 | 35 | 10 | 0.000 | 0.992 | 0.000 |
| MI | myocardial infarction | 123 | 13 | 0.978 | 0.494 | 0.452 | 5 | 4493 | 9 | 8 | 0.385 | 0.998 | 0.370 |
| FB | fusion beats | 116 | 11 | 0.973 | 0.091 | 0.165 | 4 | 4459 | 45 | 7 | 0.364 | 0.990 | 0.133 |
| RVH | right ventricular hypertrophy | 110 | 8 | 0.990 | 0.228 | 0.308 | 1 | 4504 | 3 | 7 | 0.125 | 0.999 | 0.167 |
| PVT | paroxysmal ventricular tachycardia | 109 | 16 | 0.991 | 0.453 | 0.414 | 5 | 4493 | 6 | 11 | 0.312 | 0.999 | 0.370 |
| PPW | prolonged P wave | 106 | 12 | 0.717 | 0.067 | 0.148 | 2 | 4489 | 14 | 10 | 0.167 | 0.997 | 0.143 |
| VEsR | ventricular escape rhythm | 96 | 6 | 0.998 | 0.425 | 0.400 | 3 | 4503 | 6 | 3 | 0.500 | 0.999 | 0.400 |
| CR | clockwise rotation | 76 | 8 | 0.929 | 0.126 | 0.167 | 2 | 4495 | 12 | 6 | 0.250 | 0.997 | 0.182 |
| CHB | complete heart block | 76 | 9 | 0.995 | 0.229 | 0.286 | 4 | 4499 | 7 | 5 | 0.444 | 0.998 | 0.400 |
| JE | junctional escape | 75 | 8 | 0.953 | 0.047 | 0.060 | 2 | 4460 | 47 | 6 | 0.250 | 0.990 | 0.070 |
| WPW | wolff parkinson white pattern | 72 | 4 | 0.948 | 0.555 | 0.500 | 1 | 4509 | 2 | 3 | 0.250 | 1.000 | 0.286 |
| IIAVB | 2nd degree av block | 66 | 6 | 0.928 | 0.057 | 0.000 | 0 | 4498 | 11 | 6 | 0.000 | 0.998 | 0.000 |
| BPAC | blocked premature atrial contraction | 62 | 2 | 0.955 | 0.007 | 0.000 | 0 | 4503 | 10 | 2 | 0.000 | 0.998 | 0.000 |
| AVD | atrioventricular dissociation | 59 | 3 | 0.978 | 0.035 | 0.000 | 0 | 4498 | 14 | 3 | 0.000 | 0.997 | 0.000 |
| VF | ventricular fibrillation | 59 | 3 | 1.000 | 0.700 | 0.571 | 3 | 4511 | 1 | 0 | 1.000 | 1.000 | 0.857 |
| AnMI | anterior myocardial infarction | 57 | 6 | 0.993 | 0.260 | 0.000 | 0 | 4509 | 0 | 6 | 0.000 | 1.000 | 0.000 |
| VEsB | ventricular escape beat | 56 | 3 | 0.996 | 0.121 | 0.122 | 2 | 4486 | 26 | 1 | 0.667 | 0.994 | 0.129 |
| LPR | prolonged pr interval | 52 | 5 | 0.947 | 0.023 | 0.030 | 2 | 4419 | 91 | 3 | 0.400 | 0.980 | 0.041 |
| **Average** | | 1411.7 | | **0.956** | **0.394** | **0.401** | 106.4 | 4303.2 | 70.4 | 35.0 | **0.464** | **0.983** | **0.385** |

Columns TP-F1 (right block) are on the noisy test set.

## Table 2. Example-based performance (per-ECG averages)

| Test set | TP | TN | FP | FN | Sen | Spe | F1 | Acc |
|---|---|---|---|---|---|---|---|---|
| Clean (n = 4515) | 1.52 | 59.0 | 1.03 | 0.43 | 0.854 | 0.983 | 0.742 | 0.976 |
| Noisy (n = 4515) | 1.46 | 59.1 | 0.97 | 0.48 | 0.831 | 0.984 | 0.730 | 0.977 |

## Table 3. Atrial fibrillation at different thresholds (test split)

Bold: high-sensitivity point (validation sensitivity ≥ 0.95, threshold 0.110), break-even point (0.494), optimal-F1 point (0.153).

| Threshold | TP | TN | FP | FN | F1 | Sensitivity | Precision | Specificity |
|---|---|---|---|---|---|---|---|---|
| 0.004 | 172 | 3352 | 991 | 0 | 0.258 | 1.000 | 0.148 | 0.772 |
| 0.100 | 166 | 3719 | 624 | 6 | 0.345 | 0.965 | 0.210 | 0.856 |
| **0.110** | **164** | **3731** | **612** | **8** | **0.346** | **0.953** | **0.211** | **0.859** |
| **0.153** | **159** | **3789** | **554** | **13** | **0.359** | **0.924** | **0.223** | **0.872** |
| 0.200 | 137 | 3855 | 488 | 35 | 0.344 | 0.797 | 0.219 | 0.888 |
| 0.300 | 92 | 3972 | 371 | 80 | 0.290 | 0.535 | 0.199 | 0.915 |
| 0.400 | 55 | 4105 | 238 | 117 | 0.237 | 0.320 | 0.188 | 0.945 |
| **0.494** | **37** | **4208** | **135** | **135** | **0.215** | **0.215** | **0.215** | **0.969** |
| 0.600 | 12 | 4287 | 56 | 160 | 0.100 | 0.070 | 0.176 | 0.987 |
| 0.700 | 2 | 4325 | 18 | 170 | 0.021 | 0.012 | 0.100 | 0.996 |
| 0.800 | 0 | 4342 | 1 | 172 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.900 | 0 | 4343 | 0 | 172 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.970 | 0 | 4343 | 0 | 172 | 0.000 | 0.000 | 0.000 | 1.000 |
| 0.998 | 0 | 4343 | 0 | 172 | 0.000 | 0.000 | 0.000 | 1.000 |

## Table 4. Ablation (Fig. 2a): macro metrics on the Chapman test split, mean ± s.d. over seeds

| Model | Params | Seeds | AUROC | AUPRC | F1 | AUPRC (noisy test) | ΔAUPRC clean→noisy |
|---|---|---|---|---|---|---|---|
| DNN (34-layer, Hannun) | 11.23M | 3 | 0.921 ± 0.006 | 0.330 ± 0.010 | 0.332 ± 0.008 | 0.327 ± 0.009 | -0.003 |
| Single-scale DNN-18 (k=17) | 4.20M | 3 | 0.941 ± 0.005 | 0.348 ± 0.008 | 0.354 ± 0.009 | 0.336 ± 0.008 | -0.012 |
| MSDNN | 2.17M | 3 | 0.941 ± 0.004 | 0.343 ± 0.017 | 0.351 ± 0.018 | 0.335 ± 0.017 | -0.008 |
| MSDNN with PW | 2.17M | 3 | 0.950 ± 0.002 | 0.380 ± 0.005 | 0.379 ± 0.005 | 0.359 ± 0.005 | -0.020 |
| MSDNN with Aug | 2.17M | 3 | 0.953 ± 0.001 | 0.391 ± 0.006 | 0.399 ± 0.007 | 0.383 ± 0.008 | -0.008 |
| MSDNN with PW&Aug | 2.17M | 3 | 0.957 ± 0.000 | 0.395 ± 0.001 | 0.400 ± 0.004 | 0.388 ± 0.002 | -0.007 |

Paired-sample t-tests on per-class test AUPRC (seed-averaged), as in the paper's Statistical analysis:

| Comparison | mean ΔAUPRC | p-value |
|---|---|---|
| DNN (34-layer, Hannun) vs MSDNN | +0.0130 | 0.0682 |
| Single-scale DNN-18 (k=17) vs MSDNN | -0.0050 | 0.33 |
| MSDNN vs MSDNN with PW | +0.0369 | 1.71e-05 |
| MSDNN vs MSDNN with Aug | +0.0485 | 1.82e-06 |
| MSDNN vs MSDNN with PW&Aug | +0.0521 | 8.89e-07 |
| MSDNN with Aug vs MSDNN with PW&Aug | +0.0036 | 0.544 |
| MSDNN vs MSDNN with PW (10% of training set) | +0.0409 | 0.000415 |

## Table 5. Each augmentation alone (Fig. 2c): test macro AUPRC, mean ± s.d. over seeds

| Augmentation | AUROC | AUPRC | AUPRC (noisy test) |
|---|---|---|---|
| None | 0.941 ± 0.004 | 0.343 ± 0.017 | 0.335 ± 0.017 |
| Frequency dropout | 0.939 ± 0.005 | 0.341 ± 0.007 | 0.336 ± 0.005 |
| Crop resize | 0.948 ± 0.002 | 0.378 ± 0.006 | 0.366 ± 0.003 |
| Cycle mask | 0.945 ± 0.001 | 0.368 ± 0.008 | 0.358 ± 0.008 |
| Channel mask | 0.947 ± 0.004 | 0.359 ± 0.002 | 0.350 ± 0.001 |
| All four combined | 0.953 ± 0.001 | 0.391 ± 0.006 | 0.383 ± 0.008 |

## Table 6. Training-set size (Fig. 2b): test macro AUPRC / AUROC, random vs pretrained initialisation (seed 0)

| Fraction | Training ECGs | MSDNN AUPRC | MSDNN with PW AUPRC | Δ | MSDNN AUROC | MSDNN with PW AUROC |
|---|---|---|---|---|---|---|
| 10% | 3612 | 0.192 | 0.233 | +0.041 | 0.837 | 0.858 |
| 20% | 7224 | 0.215 | 0.293 | +0.078 | 0.856 | 0.897 |
| 30% | 10836 | 0.240 | 0.314 | +0.073 | 0.892 | 0.911 |
| 40% | 14448 | 0.282 | 0.324 | +0.042 | 0.909 | 0.922 |
| 50% | 18060 | 0.286 | 0.339 | +0.053 | 0.911 | 0.931 |
| 60% | 21672 | 0.307 | 0.365 | +0.057 | 0.925 | 0.938 |
| 70% | 25284 | 0.302 | 0.361 | +0.058 | 0.924 | 0.944 |
| 80% | 28896 | 0.326 | 0.366 | +0.041 | 0.935 | 0.946 |
| 90% | 32508 | 0.341 | 0.372 | +0.031 | 0.942 | 0.948 |
| 100% | 36120 | 0.334 | 0.378 | +0.044 | 0.938 | 0.948 |

## Table 7. Robustness (Fig. 2d): test macro AUPRC on clean and noisy ECGs

| Model | Clean | Noisy | Drop |
|---|---|---|---|
| MSDNN | 0.343 ± 0.017 | 0.335 ± 0.017 | -0.008 |
| MSDNN with Aug | 0.391 ± 0.006 | 0.383 ± 0.008 | -0.008 |
| MSDNN with PW&Aug | 0.395 ± 0.001 | 0.388 ± 0.002 | -0.007 |

Gain from augmentation: clean +0.048, noisy +0.049 AUPRC.

## Table 8. Reduced-lead devices simulated from the 12 leads (Fig. 2e), MSDNN with PW&Aug

| Leads | AUROC | AUPRC | F1 |
|---|---|---|---|
| 1 (lead I) | 0.907 ± 0.000 | 0.298 ± 0.001 | 0.310 ± 0.001 |
| 3 (Holter: II, V1, V5) | 0.952 ± 0.001 | 0.373 ± 0.002 | 0.379 ± 0.007 |
| 3 (Frank: I, aVF, V2) | 0.948 ± 0.000 | 0.371 ± 0.001 | 0.378 ± 0.004 |
| 12 | 0.957 ± 0.000 | 0.395 ± 0.001 | 0.400 ± 0.004 |

## Table 9. CPSC2018 (9 classes): 10 cross-validation models, simple voting, held-out test split

The paper reports F1 0.839 on the hidden CPSC2018 test set (challenge best 0.837). That set is not public; our held-out 15% split is scored with the same metric (mean per-class F1).

| Initialisation | Vote F1 | Mean-probability F1 | Single model F1 | Mean AUROC | Normal | AF | I-AVB | LBBB | RBBB | PAC | PVC | STD | STE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random + Aug | **0.820** | 0.821 | 0.786 ± 0.018 | 0.974 | 0.763 | 0.918 | 0.898 | 0.885 | 0.926 | 0.770 | 0.809 | 0.833 | 0.571 |
| PW (MoCo) + Aug | **0.806** | 0.808 | 0.782 ± 0.010 | 0.968 | 0.724 | 0.920 | 0.867 | 0.857 | 0.922 | 0.715 | 0.795 | 0.810 | 0.641 |

## Table 10. Model size and single-thread CPU inference time per 10 s, 12-lead ECG

| Model | Params | ms / ECG |
|---|---|---|
| DNN | 11.23M | 22.7 |
| Single-scale DNN-18 | 4.20M | 12.9 |
| MSDNN | 2.17M | 9.9 |

The paper: MSDNN 2.13M parameters, < 0.08 s per 15 s recording.

