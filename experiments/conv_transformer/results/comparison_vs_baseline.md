# CNN + Transformer vs. Baseline ConvLSTM (minimal LSTM->attention swap)

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| CNN + Transformer | 99.28% | 80.08% | 225+/-137 |
| Baseline ConvLSTM | 99.28% | 91.19% | 224+/-135 |

Specificity delta: -11.1pp.

**Result: attention performed WORSE than the LSTM on this direct swap.**
