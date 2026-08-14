# GAF 2D-CNN vs. ConvLSTM Baseline vs. cwt_lstm (matched 150K scale)

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| **GAF 2D-CNN (this run)** | 80.64% | 9.20% | 219+/-118 |
| ConvLSTM baseline (corrected test set) | 94.53% | 93.68% | 224+/-136 |
| cwt_lstm (2D CWT scalogram, matched 150K scale) | 83.83% | 73.75% | 228+/-126 |

## Success criterion: does GAF specificity meaningfully improve over, meet, or fall short of the matched-scale ConvLSTM baseline?

vs. ConvLSTM baseline (93.68%): -84.5pp.
vs. cwt_lstm at matched 150K scale (73.75%): -64.6pp.

**Result: GAF specificity falls short of the ConvLSTM baseline -- consistent with the CWT precedent. Evidence that the specificity plateau is not explained by "signal shape isn't explicit enough"; points back toward the LSTM's role, aggregation rule, or training-recipe details as the more likely explanations.**
