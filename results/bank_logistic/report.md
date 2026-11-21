# DriftLens experiment report

## Run settings

- Rows: 41188 (positive rate 0.113)
- Model: logistic, seed 42, bootstrap resamples 200
- Excluded leakage columns: duration
- All windows leakage-free (temporal audit): True
- Source checksum (SHA-256): 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
- Versions: {'python': '3.15.0', 'numpy': '2.5.5', 'pandas': '3.1.1', 'scikitLearn': '1.9.1'}
- Runtime: 17.0 s

## Split manifest

| segment | start | end | size |
| --- | --- | --- | --- |
| train | 0 | 20594 | 20594 |
| calibration | 20594 | 24712 | 4118 |
| eval0 | 24712 | 28831 | 4119 |
| eval1 | 28831 | 32950 | 4119 |
| eval2 | 32950 | 37069 | 4119 |
| eval3 | 37069 | 41188 | 4119 |

## RQ1/RQ2: mean metrics across evaluation blocks

| strategy | prAuc | brier | logLoss | ece | precisionTop | updateSeconds | newLabelsUsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.2481 | 0.2316 | 0.6473 | 0.2619 | 0.2500 | 0.0000 | 0.0000 |
| recalibrated | 0.1818 | 0.1741 | 0.6136 | 0.1310 | 0.1256 | 0.0394 | 4118.7500 |
| rolling | 0.3007 | 0.1456 | 0.5825 | 0.0563 | 0.3076 | 0.7008 | 4118.7500 |

## RQ1/RQ2: per-window metrics

| strategy | block | start | end | positiveRate | prAuc | brier | logLoss | ece | precisionTop | updateSeconds | newLabelsUsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 24712 | 28831 | 0.1015 | 0.1775 | 0.1165 | 0.3769 | 0.1297 | 0.1845 | 0.0000 | 0 |
| frozen | 1 | 28831 | 32950 | 0.1199 | 0.1174 | 0.2773 | 0.7505 | 0.4010 | 0.1019 | 0.0000 | 0 |
| frozen | 2 | 32950 | 37069 | 0.1573 | 0.1488 | 0.2657 | 0.7261 | 0.3466 | 0.1408 | 0.0000 | 0 |
| frozen | 3 | 37069 | 41188 | 0.4593 | 0.5488 | 0.2670 | 0.7358 | 0.1702 | 0.5728 | 0.0000 | 0 |
| recalibrated | 0 | 24712 | 28831 | 0.1015 | 0.0707 | 0.1004 | 0.6467 | 0.0854 | 0.0388 | 0.0387 | 4118 |
| recalibrated | 1 | 28831 | 32950 | 0.1199 | 0.1174 | 0.1168 | 0.4015 | 0.0926 | 0.1019 | 0.0403 | 4119 |
| recalibrated | 2 | 32950 | 37069 | 0.1573 | 0.1631 | 0.1339 | 0.4411 | 0.0371 | 0.1214 | 0.0447 | 4119 |
| recalibrated | 3 | 37069 | 41188 | 0.4593 | 0.3759 | 0.3454 | 0.9650 | 0.3089 | 0.2403 | 0.0342 | 4119 |
| rolling | 0 | 24712 | 28831 | 0.1015 | 0.0720 | 0.0998 | 0.8327 | 0.0721 | 0.0437 | 0.4238 | 4118 |
| rolling | 1 | 28831 | 32950 | 0.1199 | 0.2335 | 0.1011 | 0.3470 | 0.0110 | 0.2573 | 0.5560 | 4119 |
| rolling | 2 | 32950 | 37069 | 0.1573 | 0.3477 | 0.1208 | 0.3998 | 0.0398 | 0.4248 | 0.7036 | 4119 |
| rolling | 3 | 37069 | 41188 | 0.4593 | 0.5495 | 0.2609 | 0.7504 | 0.1023 | 0.5049 | 1.1198 | 4119 |

## RQ2: paired bootstrap 95% intervals (strategy minus frozen)

For PR-AUC higher is better; for Brier lower is better.

| strategy | block | metric | observedDiff | ciLow | ciHigh | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| recalibrated | 0 | prAuc | -0.1068 | -0.1285 | -0.0819 | below frozen |
| recalibrated | 0 | brier | -0.0161 | -0.0230 | -0.0088 | below frozen |
| recalibrated | 1 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 1 | brier | -0.1605 | -0.1662 | -0.1538 | below frozen |
| recalibrated | 2 | prAuc | 0.0143 | -0.0046 | 0.0304 | no clear difference |
| recalibrated | 2 | brier | -0.1318 | -0.1405 | -0.1228 | below frozen |
| recalibrated | 3 | prAuc | -0.1729 | -0.1998 | -0.1430 | below frozen |
| recalibrated | 3 | brier | 0.0784 | 0.0643 | 0.0937 | above frozen |
| rolling | 0 | prAuc | -0.1056 | -0.1272 | -0.0863 | below frozen |
| rolling | 0 | brier | -0.0167 | -0.0234 | -0.0095 | below frozen |
| rolling | 1 | prAuc | 0.1161 | 0.0929 | 0.1430 | above frozen |
| rolling | 1 | brier | -0.1762 | -0.1833 | -0.1675 | below frozen |
| rolling | 2 | prAuc | 0.1990 | 0.1698 | 0.2314 | above frozen |
| rolling | 2 | brier | -0.1449 | -0.1529 | -0.1356 | below frozen |
| rolling | 3 | prAuc | 0.0007 | -0.0248 | 0.0276 | no clear difference |
| rolling | 3 | brier | -0.0061 | -0.0144 | 0.0030 | no clear difference |

## RQ3: risk-coverage (mean error among accepted cases over blocks)

| coverage | frozen | recalibrated | rolling |
| --- | --- | --- | --- |
| 0.5000 | 0.4165 | 0.2419 | 0.1700 |
| 0.6000 | 0.4013 | 0.2365 | 0.1680 |
| 0.7000 | 0.3894 | 0.2327 | 0.1669 |
| 0.8000 | 0.3970 | 0.2241 | 0.1692 |
| 0.9000 | 0.4077 | 0.2188 | 0.1795 |
| 1.0000 | 0.4141 | 0.2095 | 0.1910 |

## Calibration (reliability, pooled over evaluation blocks)

| strategy | bin | binLow | binHigh | count | meanPredicted | observedRate |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 0.0000 | 0.1000 | 813 | 0.0793 | 0.0443 |
| frozen | 1 | 0.1000 | 0.2000 | 1950 | 0.1305 | 0.0538 |
| frozen | 2 | 0.2000 | 0.3000 | 610 | 0.2485 | 0.0984 |
| frozen | 3 | 0.3000 | 0.4000 | 883 | 0.3556 | 0.2095 |
| frozen | 4 | 0.4000 | 0.5000 | 4063 | 0.4589 | 0.1656 |
| frozen | 5 | 0.5000 | 0.6000 | 3519 | 0.5430 | 0.2259 |
| frozen | 6 | 0.6000 | 0.7000 | 2930 | 0.6494 | 0.2614 |
| frozen | 7 | 0.7000 | 0.8000 | 1292 | 0.7426 | 0.4582 |
| frozen | 8 | 0.8000 | 0.9000 | 416 | 0.8254 | 0.5769 |
| recalibrated | 0 | 0.0000 | 0.1000 | 4111 | 0.0163 | 0.1012 |
| recalibrated | 1 | 0.1000 | 0.2000 | 10180 | 0.1425 | 0.2701 |
| recalibrated | 2 | 0.2000 | 0.3000 | 1977 | 0.2404 | 0.1361 |
| recalibrated | 3 | 0.3000 | 0.4000 | 208 | 0.3179 | 0.0817 |
| rolling | 0 | 0.0000 | 0.1000 | 8788 | 0.0506 | 0.0919 |
| rolling | 1 | 0.1000 | 0.2000 | 2861 | 0.1400 | 0.1653 |
| rolling | 2 | 0.2000 | 0.3000 | 1295 | 0.2490 | 0.3344 |
| rolling | 3 | 0.3000 | 0.4000 | 967 | 0.3442 | 0.3454 |
| rolling | 4 | 0.4000 | 0.5000 | 524 | 0.4473 | 0.4408 |
| rolling | 5 | 0.5000 | 0.6000 | 437 | 0.5509 | 0.5195 |
| rolling | 6 | 0.6000 | 0.7000 | 427 | 0.6472 | 0.6370 |
| rolling | 7 | 0.7000 | 0.8000 | 383 | 0.7576 | 0.6815 |
| rolling | 8 | 0.8000 | 0.9000 | 516 | 0.8528 | 0.5000 |
| rolling | 9 | 0.9000 | 1.0000 | 278 | 0.9408 | 0.5576 |

## RQ4: drift vs. frozen-model quality

| block | meanKs | meanJs | frozenPrAuc | frozenBrier |
| --- | --- | --- | --- | --- |
| 0 | 0.5232 | 0.1531 | 0.1775 | 0.1165 |
| 1 | 0.6164 | 0.1057 | 0.1174 | 0.2773 |
| 2 | 0.5954 | 0.0709 | 0.1488 | 0.2657 |
| 3 | 0.5913 | 0.1314 | 0.5488 | 0.2670 |

Correlation(meanKS, frozen PR-AUC) = 0.046 (only a few blocks: interpret with caution).

## Synthetic stress tests (frozen model, all evaluation rows)

| perturbation | severity | prAuc | brier | ece | precisionTop |
| --- | --- | --- | --- | --- | --- |
| numericShift | 0.0000 | 0.3777 | 0.2316 | 0.2618 | 0.4939 |
| numericShift | 0.2500 | 0.3777 | 0.2091 | 0.2193 | 0.4939 |
| numericShift | 0.5000 | 0.3777 | 0.1904 | 0.1773 | 0.4939 |
| numericShift | 1.0000 | 0.3777 | 0.1647 | 0.0974 | 0.4939 |
| missingValues | 0.0000 | 0.3777 | 0.2316 | 0.2618 | 0.4939 |
| missingValues | 0.2500 | 0.3316 | 0.2500 | 0.2794 | 0.4260 |
| missingValues | 0.5000 | 0.3079 | 0.2636 | 0.2916 | 0.3774 |
| missingValues | 1.0000 | 0.2797 | 0.2834 | 0.3047 | 0.3350 |
| unseenCategories | 0.0000 | 0.3777 | 0.2316 | 0.2618 | 0.4939 |
| unseenCategories | 0.2500 | 0.3737 | 0.2424 | 0.2791 | 0.4836 |
| unseenCategories | 0.5000 | 0.3660 | 0.2544 | 0.2962 | 0.4703 |
| unseenCategories | 1.0000 | 0.3622 | 0.2783 | 0.3303 | 0.4648 |

## Limitations

- Row order is a proxy for time, not a record-level timestamp.
- Stress tests are synthetic perturbations, not observed drift.
- Few evaluation blocks: correlations and trends are indicative only.
- Educational prototype; not a validated banking decision system.