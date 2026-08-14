# CWT + 2D-CNN vs. Baseline Raw-1D ConvLSTM

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| CWT + 2D-CNN + LSTM | 83.83% | 73.75% | 228+/-126 |
| Baseline raw-1D ConvLSTM | 94.53% | 93.68% | 224+/-136 |

Baseline specificity: 93.68%. CWT specificity: 73.75%. Delta: -19.9pp.

**Result: specificity did NOT improve meaningfully over the baseline -- this rules out "missing frequency information" as the explanation for the plateau.**
