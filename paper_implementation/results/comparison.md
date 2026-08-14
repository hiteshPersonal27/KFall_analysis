# KFall Baseline Reproduction -- Comparison

| Algorithm | FN | FP | Sensitivity | Sensitivity (paper) | Specificity | Specificity (paper) | Lead time (ms) | Lead time (paper, ms) |
|---|---|---|---|---|---|---|---|---|
| Threshold | 55 | 86 | 87.47% | 95.50% | 83.52% | 83.43% | 347 +/- 135 | 333 +/- 160 |
| SVM | 72 | 120 | 83.60% | 99.77% | 77.01% | 94.87% | 226 +/- 123 | 385 +/- 159 |
| ConvLSTM | 24 | 33 | 94.53% | 99.32% | 93.68% | 99.01% | 224 +/- 136 | 403 +/- 163 |

## Reproduction success criteria

- [ ] SVM outperforms Threshold (sensitivity)
- [x] ConvLSTM outperforms Threshold (sensitivity)
- [x] ConvLSTM most balanced sensitivity/specificity
- [ ] Specificity ordering Threshold < SVM < ConvLSTM
