# DriftLens experiment report

## Run settings

- Rows: 41188 (positive rate 0.113)
- Model: histGradientBoosting, seed 42, bootstrap resamples 200
- Excluded leakage columns: duration
- All windows leakage-free (temporal audit): True
- Source checksum (SHA-256): 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
- Versions: {'python': '3.15.0', 'numpy': '2.5.5', 'pandas': '3.1.1', 'scikitLearn': '1.9.1'}
- Runtime: 21.1 s

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
| frozen | 0.2252 | 0.1830 | 0.6057 | 0.1330 | 0.2391 | 0.0000 | 0.0000 |
| recalibrated | 0.2199 | 0.1683 | 0.5284 | 0.1051 | 0.2245 | 0.0543 | 4118.7500 |
| rolling | 0.2529 | 0.1546 | 0.4984 | 0.0793 | 0.2858 | 1.5449 | 4118.7500 |

## RQ1/RQ2: per-window metrics

| strategy | block | start | end | positiveRate | prAuc | brier | logLoss | ece | precisionTop | updateSeconds | newLabelsUsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 24712 | 28831 | 0.1015 | 0.1128 | 0.0918 | 0.3330 | 0.0250 | 0.1238 | 0.0000 | 0 |
| frozen | 1 | 28831 | 32950 | 0.1199 | 0.1284 | 0.1074 | 0.3800 | 0.0423 | 0.1214 | 0.0000 | 0 |
| frozen | 2 | 32950 | 37069 | 0.1573 | 0.1609 | 0.1392 | 0.4751 | 0.0808 | 0.1578 | 0.0000 | 0 |
| frozen | 3 | 37069 | 41188 | 0.4593 | 0.4988 | 0.3934 | 1.2348 | 0.3837 | 0.5534 | 0.0000 | 0 |
| recalibrated | 0 | 24712 | 28831 | 0.1015 | 0.0913 | 0.0952 | 0.3659 | 0.0607 | 0.0655 | 0.0738 | 4118 |
| recalibrated | 1 | 28831 | 32950 | 0.1199 | 0.1284 | 0.1058 | 0.3680 | 0.0177 | 0.1214 | 0.0529 | 4119 |
| recalibrated | 2 | 32950 | 37069 | 0.1573 | 0.1609 | 0.1338 | 0.4407 | 0.0391 | 0.1578 | 0.0455 | 4119 |
| recalibrated | 3 | 37069 | 41188 | 0.4593 | 0.4988 | 0.3384 | 0.9390 | 0.3029 | 0.5534 | 0.0449 | 4119 |
| rolling | 0 | 24712 | 28831 | 0.1015 | 0.1006 | 0.0945 | 0.3518 | 0.0495 | 0.0728 | 1.0810 | 4118 |
| rolling | 1 | 28831 | 32950 | 0.1199 | 0.1653 | 0.1049 | 0.3639 | 0.0198 | 0.1748 | 1.3094 | 4119 |
| rolling | 2 | 32950 | 37069 | 0.1573 | 0.2191 | 0.1391 | 0.4884 | 0.0941 | 0.2646 | 1.5836 | 4119 |
| rolling | 3 | 37069 | 41188 | 0.4593 | 0.5267 | 0.2798 | 0.7894 | 0.1538 | 0.6311 | 2.2057 | 4119 |

## RQ2: paired bootstrap 95% intervals (strategy minus frozen)

For PR-AUC higher is better; for Brier lower is better.

| strategy | block | metric | observedDiff | ciLow | ciHigh | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| recalibrated | 0 | prAuc | -0.0216 | -0.0358 | -0.0065 | below frozen |
| recalibrated | 0 | brier | 0.0034 | 0.0025 | 0.0042 | above frozen |
| recalibrated | 1 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 1 | brier | -0.0017 | -0.0021 | -0.0012 | below frozen |
| recalibrated | 2 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 2 | brier | -0.0054 | -0.0063 | -0.0044 | below frozen |
| recalibrated | 3 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 3 | brier | -0.0550 | -0.0573 | -0.0527 | below frozen |
| rolling | 0 | prAuc | -0.0123 | -0.0247 | -0.0003 | below frozen |
| rolling | 0 | brier | 0.0027 | 0.0018 | 0.0037 | above frozen |
| rolling | 1 | prAuc | 0.0369 | 0.0148 | 0.0617 | above frozen |
| rolling | 1 | brier | -0.0026 | -0.0040 | -0.0011 | below frozen |
| rolling | 2 | prAuc | 0.0582 | 0.0382 | 0.0780 | above frozen |
| rolling | 2 | brier | -0.0001 | -0.0012 | 0.0011 | no clear difference |
| rolling | 3 | prAuc | 0.0279 | 0.0023 | 0.0529 | above frozen |
| rolling | 3 | brier | -0.1136 | -0.1228 | -0.1061 | below frozen |

## RQ3: risk-coverage (mean error among accepted cases over blocks)

| coverage | frozen | recalibrated | rolling |
| --- | --- | --- | --- |
| 0.5000 | 0.1968 | 0.1999 | 0.1791 |
| 0.6000 | 0.2012 | 0.2037 | 0.1865 |
| 0.7000 | 0.2032 | 0.2059 | 0.1916 |
| 0.8000 | 0.2042 | 0.2068 | 0.1939 |
| 0.9000 | 0.2060 | 0.2079 | 0.1977 |
| 1.0000 | 0.2095 | 0.2095 | 0.2030 |

## Calibration (reliability, pooled over evaluation blocks)

| strategy | bin | binLow | binHigh | count | meanPredicted | observedRate |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 0.0000 | 0.1000 | 14050 | 0.0696 | 0.2027 |
| frozen | 1 | 0.1000 | 0.2000 | 2420 | 0.1176 | 0.2496 |
| frozen | 2 | 0.2000 | 0.3000 | 6 | 0.2193 | 0.0000 |
| recalibrated | 0 | 0.0000 | 0.1000 | 5768 | 0.0590 | 0.1066 |
| recalibrated | 1 | 0.1000 | 0.2000 | 10708 | 0.1321 | 0.2649 |
| rolling | 0 | 0.0000 | 0.1000 | 8317 | 0.0578 | 0.1321 |
| rolling | 1 | 0.1000 | 0.2000 | 4801 | 0.1335 | 0.1904 |
| rolling | 2 | 0.2000 | 0.3000 | 1173 | 0.2443 | 0.3163 |
| rolling | 3 | 0.3000 | 0.4000 | 920 | 0.3496 | 0.4489 |
| rolling | 4 | 0.4000 | 0.5000 | 792 | 0.4415 | 0.4609 |
| rolling | 5 | 0.5000 | 0.6000 | 201 | 0.5452 | 0.5622 |
| rolling | 6 | 0.6000 | 0.7000 | 196 | 0.6494 | 0.6327 |
| rolling | 7 | 0.7000 | 0.8000 | 74 | 0.7377 | 0.7027 |
| rolling | 8 | 0.8000 | 0.9000 | 2 | 0.8172 | 0.5000 |

## RQ4: drift vs. frozen-model quality

| block | meanKs | meanJs | frozenPrAuc | frozenBrier |
| --- | --- | --- | --- | --- |
| 0 | 0.5232 | 0.1531 | 0.1128 | 0.0918 |
| 1 | 0.6164 | 0.1057 | 0.1284 | 0.1074 |
| 2 | 0.5954 | 0.0709 | 0.1609 | 0.1392 |
| 3 | 0.5913 | 0.1314 | 0.4988 | 0.3934 |

Correlation(meanKS, frozen PR-AUC) = 0.223 (only a few blocks: interpret with caution).

## Synthetic stress tests (frozen model, all evaluation rows)

| perturbation | severity | prAuc | brier | ece | precisionTop |
| --- | --- | --- | --- | --- | --- |
| numericShift | 0.0000 | 0.2247 | 0.1830 | 0.1330 | 0.2542 |
| numericShift | 0.2500 | 0.2256 | 0.1825 | 0.1314 | 0.2621 |
| numericShift | 0.5000 | 0.2140 | 0.1826 | 0.1299 | 0.2124 |
| numericShift | 1.0000 | 0.2189 | 0.1831 | 0.1333 | 0.1657 |
| missingValues | 0.0000 | 0.2247 | 0.1830 | 0.1330 | 0.2542 |
| missingValues | 0.2500 | 0.2219 | 0.1842 | 0.1364 | 0.2573 |
| missingValues | 0.5000 | 0.2203 | 0.1852 | 0.1392 | 0.2421 |
| missingValues | 1.0000 | 0.2241 | 0.1867 | 0.1452 | 0.2458 |
| unseenCategories | 0.0000 | 0.2247 | 0.1830 | 0.1330 | 0.2542 |
| unseenCategories | 0.2500 | 0.2253 | 0.1834 | 0.1347 | 0.2579 |
| unseenCategories | 0.5000 | 0.2229 | 0.1841 | 0.1366 | 0.2464 |
| unseenCategories | 1.0000 | 0.2215 | 0.1853 | 0.1398 | 0.2470 |

## Limitations

- Row order is a proxy for time, not a record-level timestamp.
- Stress tests are synthetic perturbations, not observed drift.
- Few evaluation blocks: correlations and trends are indicative only.
- Educational prototype; not a validated banking decision system.