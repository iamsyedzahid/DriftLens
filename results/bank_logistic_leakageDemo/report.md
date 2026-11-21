# DriftLens experiment report

> **LEAKAGE DEMONSTRATION.** This run keeps post-outcome columns (duration). Its numbers are NOT valid primary results.

## Run settings

- Rows: 41188 (positive rate 0.113)
- Model: logistic, seed 42, bootstrap resamples 200
- Excluded leakage columns: none
- All windows leakage-free (temporal audit): True
- Source checksum (SHA-256): 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8
- Versions: {'python': '3.15.0', 'numpy': '2.5.5', 'pandas': '3.1.1', 'scikitLearn': '1.9.1'}
- Runtime: 19.1 s

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
| frozen | 0.4519 | 0.1831 | 0.7571 | 0.1727 | 0.5358 | 0.0000 | 0.0000 |
| recalibrated | 0.4519 | 0.1474 | 0.4654 | 0.1073 | 0.5358 | 0.0279 | 4118.7500 |
| rolling | 0.4573 | 0.1318 | 0.6018 | 0.0753 | 0.5237 | 0.5379 | 4118.7500 |

## RQ1/RQ2: per-window metrics

| strategy | block | start | end | positiveRate | prAuc | brier | logLoss | ece | precisionTop | updateSeconds | newLabelsUsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 24712 | 28831 | 0.1015 | 0.3215 | 0.0853 | 0.3560 | 0.0642 | 0.3835 | 0.0000 | 0 |
| frozen | 1 | 28831 | 32950 | 0.1199 | 0.4054 | 0.0982 | 0.3820 | 0.0808 | 0.4733 | 0.0000 | 0 |
| frozen | 2 | 32950 | 37069 | 0.1573 | 0.4179 | 0.1355 | 0.5831 | 0.1281 | 0.5534 | 0.0000 | 0 |
| frozen | 3 | 37069 | 41188 | 0.4593 | 0.6626 | 0.4133 | 1.7074 | 0.4179 | 0.7330 | 0.0000 | 0 |
| recalibrated | 0 | 24712 | 28831 | 0.1015 | 0.3215 | 0.0844 | 0.3173 | 0.0528 | 0.3835 | 0.0290 | 4118 |
| recalibrated | 1 | 28831 | 32950 | 0.1199 | 0.4054 | 0.0913 | 0.3106 | 0.0381 | 0.4733 | 0.0248 | 4119 |
| recalibrated | 2 | 32950 | 37069 | 0.1573 | 0.4179 | 0.1179 | 0.4009 | 0.0646 | 0.5534 | 0.0275 | 4119 |
| recalibrated | 3 | 37069 | 41188 | 0.4593 | 0.6626 | 0.2959 | 0.8326 | 0.2737 | 0.7330 | 0.0302 | 4119 |
| rolling | 0 | 24712 | 28831 | 0.1015 | 0.2193 | 0.0920 | 1.0316 | 0.0771 | 0.3155 | 0.2754 | 4118 |
| rolling | 1 | 28831 | 32950 | 0.1199 | 0.4611 | 0.0844 | 0.2809 | 0.0294 | 0.5146 | 0.5255 | 4119 |
| rolling | 2 | 32950 | 37069 | 0.1573 | 0.5513 | 0.1011 | 0.3236 | 0.0571 | 0.5898 | 0.6048 | 4119 |
| rolling | 3 | 37069 | 41188 | 0.4593 | 0.5977 | 0.2495 | 0.7711 | 0.1377 | 0.6748 | 0.7460 | 4119 |

## RQ2: paired bootstrap 95% intervals (strategy minus frozen)

For PR-AUC higher is better; for Brier lower is better.

| strategy | block | metric | observedDiff | ciLow | ciHigh | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| recalibrated | 0 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 0 | brier | -0.0009 | -0.0026 | 0.0005 | no clear difference |
| recalibrated | 1 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 1 | brier | -0.0068 | -0.0083 | -0.0054 | below frozen |
| recalibrated | 2 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 2 | brier | -0.0176 | -0.0200 | -0.0150 | below frozen |
| recalibrated | 3 | prAuc | 0.0000 | 0.0000 | 0.0000 | no clear difference |
| recalibrated | 3 | brier | -0.1174 | -0.1226 | -0.1125 | below frozen |
| rolling | 0 | prAuc | -0.1022 | -0.1387 | -0.0715 | below frozen |
| rolling | 0 | brier | 0.0067 | 0.0039 | 0.0098 | above frozen |
| rolling | 1 | prAuc | 0.0557 | 0.0418 | 0.0745 | above frozen |
| rolling | 1 | brier | -0.0138 | -0.0169 | -0.0108 | below frozen |
| rolling | 2 | prAuc | 0.1334 | 0.1051 | 0.1639 | above frozen |
| rolling | 2 | brier | -0.0344 | -0.0390 | -0.0300 | below frozen |
| rolling | 3 | prAuc | -0.0649 | -0.0832 | -0.0417 | below frozen |
| rolling | 3 | brier | -0.1638 | -0.1827 | -0.1496 | below frozen |

## RQ3: risk-coverage (mean error among accepted cases over blocks)

| coverage | frozen | recalibrated | rolling |
| --- | --- | --- | --- |
| 0.5000 | 0.1163 | 0.1152 | 0.1176 |
| 0.6000 | 0.1291 | 0.1279 | 0.1172 |
| 0.7000 | 0.1383 | 0.1375 | 0.1241 |
| 0.8000 | 0.1531 | 0.1518 | 0.1336 |
| 0.9000 | 0.1731 | 0.1712 | 0.1514 |
| 1.0000 | 0.2041 | 0.2001 | 0.1790 |

## Calibration (reliability, pooled over evaluation blocks)

| strategy | bin | binLow | binHigh | count | meanPredicted | observedRate |
| --- | --- | --- | --- | --- | --- | --- |
| frozen | 0 | 0.0000 | 0.1000 | 15086 | 0.0156 | 0.1784 |
| frozen | 1 | 0.1000 | 0.2000 | 574 | 0.1390 | 0.4739 |
| frozen | 2 | 0.2000 | 0.3000 | 255 | 0.2445 | 0.5451 |
| frozen | 3 | 0.3000 | 0.4000 | 118 | 0.3497 | 0.5932 |
| frozen | 4 | 0.4000 | 0.5000 | 88 | 0.4534 | 0.6477 |
| frozen | 5 | 0.5000 | 0.6000 | 84 | 0.5445 | 0.6310 |
| frozen | 6 | 0.6000 | 0.7000 | 56 | 0.6511 | 0.6071 |
| frozen | 7 | 0.7000 | 0.8000 | 48 | 0.7503 | 0.6042 |
| frozen | 8 | 0.8000 | 0.9000 | 64 | 0.8453 | 0.6562 |
| frozen | 9 | 0.9000 | 1.0000 | 103 | 0.9648 | 0.6214 |
| recalibrated | 0 | 0.0000 | 0.1000 | 10957 | 0.0484 | 0.0833 |
| recalibrated | 1 | 0.1000 | 0.2000 | 3322 | 0.1404 | 0.3441 |
| recalibrated | 2 | 0.2000 | 0.3000 | 1125 | 0.2415 | 0.6000 |
| recalibrated | 3 | 0.3000 | 0.4000 | 408 | 0.3453 | 0.6397 |
| recalibrated | 4 | 0.4000 | 0.5000 | 251 | 0.4457 | 0.7012 |
| recalibrated | 5 | 0.5000 | 0.6000 | 152 | 0.5484 | 0.7039 |
| recalibrated | 6 | 0.6000 | 0.7000 | 91 | 0.6472 | 0.7473 |
| recalibrated | 7 | 0.7000 | 0.8000 | 59 | 0.7431 | 0.6780 |
| recalibrated | 8 | 0.8000 | 0.9000 | 53 | 0.8485 | 0.6415 |
| recalibrated | 9 | 0.9000 | 1.0000 | 58 | 0.9533 | 0.6034 |
| rolling | 0 | 0.0000 | 0.1000 | 10258 | 0.0266 | 0.0596 |
| rolling | 1 | 0.1000 | 0.2000 | 1691 | 0.1452 | 0.2620 |
| rolling | 2 | 0.2000 | 0.3000 | 1004 | 0.2465 | 0.3964 |
| rolling | 3 | 0.3000 | 0.4000 | 656 | 0.3452 | 0.4497 |
| rolling | 4 | 0.4000 | 0.5000 | 529 | 0.4499 | 0.5388 |
| rolling | 5 | 0.5000 | 0.6000 | 447 | 0.5484 | 0.5794 |
| rolling | 6 | 0.6000 | 0.7000 | 379 | 0.6491 | 0.6491 |
| rolling | 7 | 0.7000 | 0.8000 | 367 | 0.7487 | 0.6403 |
| rolling | 8 | 0.8000 | 0.9000 | 447 | 0.8524 | 0.5235 |
| rolling | 9 | 0.9000 | 1.0000 | 698 | 0.9569 | 0.6390 |

## RQ4: drift vs. frozen-model quality

| block | meanKs | meanJs | frozenPrAuc | frozenBrier |
| --- | --- | --- | --- | --- |
| 0 | 0.4744 | 0.1531 | 0.3215 | 0.0853 |
| 1 | 0.5610 | 0.1057 | 0.4054 | 0.0982 |
| 2 | 0.5403 | 0.0709 | 0.4179 | 0.1355 |
| 3 | 0.5434 | 0.1314 | 0.6626 | 0.4133 |

Correlation(meanKS, frozen PR-AUC) = 0.494 (only a few blocks: interpret with caution).

## Synthetic stress tests (frozen model, all evaluation rows)

| perturbation | severity | prAuc | brier | ece | precisionTop |
| --- | --- | --- | --- | --- | --- |
| numericShift | 0.0000 | 0.4307 | 0.1831 | 0.1721 | 0.5309 |
| numericShift | 0.2500 | 0.4307 | 0.1772 | 0.1601 | 0.5309 |
| numericShift | 0.5000 | 0.4307 | 0.1707 | 0.1447 | 0.5309 |
| numericShift | 1.0000 | 0.4307 | 0.1576 | 0.1022 | 0.5309 |
| missingValues | 0.0000 | 0.4307 | 0.1831 | 0.1721 | 0.5309 |
| missingValues | 0.2500 | 0.3843 | 0.1824 | 0.1657 | 0.4879 |
| missingValues | 0.5000 | 0.3595 | 0.1811 | 0.1579 | 0.4624 |
| missingValues | 1.0000 | 0.3315 | 0.1808 | 0.1483 | 0.4399 |
| unseenCategories | 0.0000 | 0.4307 | 0.1831 | 0.1721 | 0.5309 |
| unseenCategories | 0.2500 | 0.4275 | 0.1815 | 0.1693 | 0.5322 |
| unseenCategories | 0.5000 | 0.4237 | 0.1803 | 0.1659 | 0.5279 |
| unseenCategories | 1.0000 | 0.4197 | 0.1767 | 0.1577 | 0.5237 |

## Limitations

- Row order is a proxy for time, not a record-level timestamp.
- Stress tests are synthetic perturbations, not observed drift.
- Few evaluation blocks: correlations and trends are indicative only.
- Educational prototype; not a validated banking decision system.