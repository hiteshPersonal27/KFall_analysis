# CNN + Transformer, No Pooling vs. Pooled vs. ConvLSTM Baselines

Training scale for this run: full scale (stride=1, ADL uncapped), mean-pooled classification head.

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| **No-pooling transformer (this run, full scale)** | 94.53% | 87.16% | 224+/-137 |
| Pooled transformer (mean-pooled head, full scale) | 93.17% | 81.03% | 227+/-136 |
| ConvLSTM baseline (full scale) | 94.53% | 93.68% | 224+/-136 |

## Success criterion: did removing pooling move specificity toward/past the ConvLSTM baseline?

vs. pooled transformer (81.03%): +6.13pp.
vs. ConvLSTM baseline (93.68%): -6.52pp.

**Specificity moved meaningfully TOWARD the ConvLSTM baseline -- supports "pooling destroyed information attention needed," though a real gap to the ConvLSTM baseline remains.**
