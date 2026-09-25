# What the paper did and what this reproduction did

**Paper:** Lai J., Tan H., Wang J. *et al.* Practical intelligent diagnostic algorithm for wearable 12-lead ECG via
self-supervised learning on large-scale dataset. *Nat. Commun.* 14, 3741 (2023).
https://doi.org/10.1038/s41467-023-39472-8

**This reproduction:** the paper's method rebuilt from its Methods section, with every experiment in its tables and
figures re-run on public data. The paper's data are private and its code is no longer online.

**Summary.** The paper's main conclusions hold on public data: the small MSDNN matches a network five times its size,
and both self-supervised pre-training and ECG augmentation give significant gains. Absolute scores are lower
(AUPRC 0.395 vs 0.646) because the data differ. The 2-month clinical test cannot be repeated.

---

## 1. Goal

| | Paper | This reproduction |
|---|---|---|
| Question | Can a small network, pre-trained on unlabelled ECGs, diagnose wearable 12-lead ECGs accurately and in real time? | Do the paper's method and conclusions hold when rebuilt independently on public data? |
| Output | 60 ECG diagnostic terms (multi-label) | 62 ECG diagnostic terms (multi-label) |

## 2. Data

| | Paper | This reproduction |
|---|---|---|
| Labelled ECGs | 164,538 wearable ECGs (Cardiocloud, China), each labelled by 2 cardiologists and reviewed by 3 senior ones | **Chapman-Shaoxing/Ningbo** (PhysioNet): 45,150 resting hospital ECGs |
| Unlabelled ECGs (pre-training) | 493,948 wearable ECGs + the training set = 639,708 | **CODE-15%** (Brazil): 341,512 ECGs, labels ignored + the Chapman training set = 377,632 |
| Lead placement | Mason–Likar (wearable, electrodes on the torso) | Standard Wilson (resting, electrodes on the limbs) |
| Record length, sampling | 15 s, 500 Hz | 10 s, 500 Hz (CODE-15% resampled from 400 Hz) |
| Diagnoses | 60 terms (paper's own list) | 62 SNOMED-CT terms with at least 50 positive ECGs |
| Split | 157,538 train / 7,000 offline test, split by patient | 36,120 train / 4,515 validation / 4,515 test, one ECG per patient |
| Bad-quality ECGs | real inferior-quality wearable ECGs (1,791 in the test set) | the test set with **real recorded noise** (MIT-BIH Noise Stress Test Database) added |
| Clinical test | 12,521 ECGs over 2 months, live in a hospital | not possible |
| External dataset | CPSC2018, hidden official test set | CPSC2018, 15% held out of the public set (the official test set is not public) |

## 3. Method

The same method in both, as far as the paper describes it.

| Component | Paper | This reproduction |
|---|---|---|
| Preprocessing | 5th-order Butterworth high-pass, 0.5 Hz | same |
| Network (MSDNN) | multiscale ResNet-18: 4 parallel kernels (3, 5, 9, 17), 8 residual modules, SE blocks, 2.13M parameters | same design, **2.17M parameters** |
| Baseline (DNN) | Hannun et al. 34-layer network, kernel 17, SE blocks | same (11.2M parameters) |
| Extra baseline | – | **single-scale DNN-18**: MSDNN with only kernel 17, to isolate the effect of the multiscale kernels |
| Pre-training | MoCo + distributional divergence loss; queue 72,000, batch 360, τ 0.07, momentum 0.9, 128-d, β 0.15, 3 extra views; Adam 1e-3; **100 epochs** | same settings, **50 epochs** (smaller pool) |
| Augmentations | frequency dropout, crop resize, cycle mask, channel mask; **strength not given** | the same four; strength chosen by us (see README) |
| Fine-tuning | SGD (momentum 0.9, weight decay 5e-4, batch 128), one-cycle schedule, 100 epochs | same |
| Loss | binary cross-entropy + pairwise ranking, weighted by inverse class frequency | same |
| Model selection | lowest loss on the **test** set | lowest loss on the **validation** set (the test set is never used for selection) |
| Decision thresholds | chosen by senior cardiologists | F1-optimal on validation, plus a high-sensitivity rule (sensitivity ≥ 0.95) for Table 3 |
| Repetitions | not stated | **3 random seeds** for every main experiment, with paired t-tests |

## 4. Results, experiment by experiment

✅ same conclusion · ⚠️ partly · ❌ different

### Overall performance (Tables 1–2)

| Metric | Paper (offline test) | Ours (clean test) | |
|---|---|---|---|
| Mean AUROC | 0.975 | 0.957 | ⚠️ lower |
| Mean AUPRC | 0.646 | 0.395 | ⚠️ lower (different data) |
| Mean F1 | 0.575 | 0.400 | ⚠️ lower |
| Per-ECG sensitivity / specificity | 0.873 / 0.975 | 0.854 / 0.983 | ✅ close |
| Per-ECG F1 / accuracy | 0.756 / 0.969 | 0.742 / 0.976 | ✅ close |
| Noisy / online test: per-ECG sensitivity / specificity | 0.857 / 0.956 (online) | 0.831 / 0.984 (noisy) | ✅ close |

### Ablation (Fig. 2a), mean AUPRC

| Model | Paper | Ours (3 seeds) | |
|---|---|---|---|
| DNN (34-layer) | 0.578 | 0.330 | |
| Single-scale DNN-18 | – | 0.348 | (extra check) |
| MSDNN | 0.582 | 0.343 | ✅ same accuracy with far fewer parameters |
| MSDNN with PW | 0.593 | 0.380 | ✅ pre-training helps |
| MSDNN with Aug | 0.637 | 0.391 | ✅ augmentation helps |
| MSDNN with PW&Aug | 0.646 | 0.395 | ✅ best |

| Significance test | Paper p-value | Ours p-value | |
|---|---|---|---|
| DNN vs MSDNN | 0.749 (not significant) | 0.068 (not significant) | ✅ |
| MSDNN vs + PW | 0.202 | 1.7 × 10⁻⁵ | ⚠️ significant here, not in the paper |
| MSDNN vs + Aug | < 0.001 | 1.8 × 10⁻⁶ | ✅ |
| + Aug vs + PW&Aug | "not significantly improved" | 0.544 | ✅ |
| MSDNN vs + PW, 10% of data | 0.015 | 4.2 × 10⁻⁴ | ✅ |

### Other experiments

| Experiment | Paper | Ours | |
|---|---|---|---|
| **Fig. 2b** pre-training gain vs training-set size | +0.031 at 10%, shrinking to +0.011 at 100% | +0.041 at 10%, peak +0.078 at 20%, +0.044 at 100% | ⚠️ always helps, but the gain does not shrink |
| **Fig. 2c** best single augmentation | crop resize | crop resize (0.378) | ✅ |
| **Fig. 2c** all four together vs none | +0.055 | +0.048 | ✅ |
| **Fig. 2c** every augmentation helps | yes | frequency dropout gives no gain (0.341 vs 0.343) | ⚠️ |
| **Fig. 2d** augmentation gain on bad-quality ECGs | larger: +0.071 vs +0.049 on good quality | same: +0.049 noisy vs +0.048 clean | ❌ our added noise only lowers AUPRC by 0.008 |
| **Fig. 2e** number of leads (1 / 3 Holter / 3 Frank / 12) | 0.464 / 0.586 / 0.584 / 0.646 | 0.298 / 0.373 / 0.371 / 0.395 | ✅ same order and pattern |
| **Fig. 2f** PR curves (PVC, STD, PRBBB) | full method best | PVC 0.678 and STD 0.399 best with the full method; IRBBB used instead of PRBBB (not in Chapman) | ✅ |
| **Fig. 2g / Table 3** AF threshold trade-off | a low, doctor-chosen threshold raises sensitivity to 0.90 | a low threshold (0.11) raises sensitivity to 0.95 | ✅ trade-off reproduced; ⚠️ AF itself is weak here (AUPRC 0.20) |
| **Fig. 3c** attention maps (CAM) | the model focuses on the abnormal beats | PVC attention falls on the wide abnormal beats | ✅ |
| **CPSC2018** F1 | 0.839 (challenge best 0.837) | 0.820 | ✅ close |
| **CPSC2018** pre-training helps | not reported | 0.806 with PW vs 0.820 without | ❌ pre-training did not transfer |
| Model size / speed | 2.13M, < 80 ms per ECG | 2.17M, 10 ms per ECG on one CPU core | ✅ |
| **Online clinical test** | sensitivity 0.736, specificity 0.954 over 2 months | – | not reproducible |

## 5. Why the absolute numbers are lower

1. **Different data.** Resting hospital ECGs with standard lead placement, not wearable ECGs; 10 s instead of 15 s.
2. **Four times fewer labels** (36k vs 157k training ECGs) and many rare classes: 34 of the 62 terms have fewer
   than 400 examples in the whole dataset.
3. **Label quality.** The paper had every ECG reviewed by five cardiologists. Chapman's labels are noisier; for
   example, AF and atrial flutter overlap heavily, which is why AF scores only 0.20 AUPRC.
4. **Shorter pre-training** (50 vs 100 epochs, 377k vs 640k ECGs).

## 6. What is still missing

| Missing | Reason | Can it be added? |
|---|---|---|
| The paper's own wearable data | private (Cardiocloud) | only with a data-sharing agreement |
| The 2-month clinical test | needs the hospital system | no |
| Evaluation on real wearable ECGs | not done yet | **yes: the paper's 7,000-ECG offline test set is public on ScienceDB (doi 10.57760/sciencedb.07677)** |
| Cardiologist-chosen thresholds | no cardiologists involved | replaced by automatic rules |
| 3 seeds for Fig. 2b | 1 seed was run | yes |

## 7. Conclusion

| Paper's claim | Reproduced? |
|---|---|
| A multiscale network keeps accuracy with far fewer parameters | ✅ yes |
| Self-supervised pre-training improves diagnosis, most when labels are scarce | ✅ yes (gain stays about +0.04 at all sizes) |
| ECG augmentation improves diagnosis | ✅ yes |
| Augmentation makes the model more robust to artifacts | ⚠️ not shown; our noise test is too mild |
| More leads give better diagnosis | ✅ yes |
| The method generalises to CPSC2018 | ✅ yes (F1 0.820 vs 0.839) |
| The model works in a real clinical setting | not testable |

**Open problems suggested by these results:** pre-training that transfers across datasets (it hurt on CPSC2018),
rare diagnoses (many terms below 0.1 AUPRC), realistic wearable-noise testing, and label noise in public ECG datasets.

---

Full tables: [results/RESULTS.md](results/RESULTS.md) · Figures: [figures/](figures/) (all Fig. 2 panels on one page:
[fig2_all.png](figures/fig2_all.png)) · Code and setup: [README.md](README.md)
