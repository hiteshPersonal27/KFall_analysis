# KFall Baseline Reproduction -- Comparison

| Algorithm | FN | FP | Sensitivity | Sensitivity (paper) | Specificity | Specificity (paper) | Lead time (ms) | Lead time (paper, ms) |
|---|---|---|---|---|---|---|---|---|
| Threshold | 55 | 86 | 87.47% | 95.50% | 83.52% | 83.43% | 347 +/- 134 | 333 +/- 160 |
| SVM | 43 | 105 | 89.51% | 99.77% | 79.89% | 94.87% | 225 +/- 124 | 385 +/- 159 |
| ConvLSTM | 3 | 46 | 99.28% | 99.32% | 91.19% | 99.01% | 224 +/- 135 | 403 +/- 163 |

## Reproduction success criteria

- [x] SVM outperforms Threshold (sensitivity)
- [x] ConvLSTM outperforms Threshold (sensitivity)
- [x] ConvLSTM most balanced sensitivity/specificity
- [ ] Specificity ordering Threshold < SVM < ConvLSTM
