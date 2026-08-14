# GAF Global Normalization vs. Original (Per-Window) vs. ConvLSTM Baselines

| | Sensitivity | Specificity | Lead time (ms) |
|---|---|---|---|
| **GAF, GLOBAL norm (this fix, full scale)** | 94.53% | 66.48% | 223+/-131 |
| GAF, per-window norm (original, 150K scale) | 80.64% | 9.20% | 219+/-118 |
| ConvLSTM baseline (full scale) | 94.53% | 93.68% | 224+/-136 |

## CONSEC_WINDOWS sweep (diagnostic: is the flicker pattern reduced?)

| CONSEC_WINDOWS | Sensitivity | Specificity |
|---|---|---|
| 1 | 100.00% | 65.13% |
| 2 | 94.53% | 66.48% |
| 3 | 93.62% | 68.01% |
| 5 | 90.66% | 72.61% |
| 8 | 85.65% | 79.12% |

## Success criterion

vs. ConvLSTM at full scale (93.68%): -27.2pp.
vs. original per-window GAF (9.20%): +57.28pp.

**Result: global-normalization GAF specificity still falls short of the ConvLSTM baseline, though (see the before/after row above) the fix's actual effect on the instability problem should be judged from the CONSEC_WINDOWS sweep and the original 9.20% comparison, not this number alone.**
