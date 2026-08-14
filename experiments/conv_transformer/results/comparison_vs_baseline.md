# CNN + Transformer vs. Baseline ConvLSTM (minimal LSTM->attention swap)

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| CNN + Transformer | 93.17% | 81.03% | 227+/-136 |
| Baseline ConvLSTM | 94.53% | 93.68% | 224+/-136 |

Specificity delta: -12.6pp.

**Result: attention performed WORSE than the LSTM on this direct swap.**
