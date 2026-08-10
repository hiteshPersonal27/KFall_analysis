# CWT + 2D-CNN vs. Baseline Raw-1D ConvLSTM

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| CWT + 2D-CNN + LSTM | 89.76% | 73.75% | 228+/-126 |
| Baseline raw-1D ConvLSTM | 99.28% | 91.19% | 224+/-135 |

Baseline specificity: 91.19%. CWT specificity: 73.75%. Delta: -17.4pp.

**Result: specificity did NOT improve meaningfully over the baseline -- this rules out "missing frequency information" as the explanation for the plateau.**
