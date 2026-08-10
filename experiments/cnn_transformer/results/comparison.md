# 1D-CNN + Transformer vs. Baseline ConvLSTM vs. PreFallKD's published numbers

| Model | Accuracy | Precision | Recall (Sens.) | Specificity | F1 | Lead time (ms) |
|---|---|---|---|---|---|---|
| **This repro (transformer)** | 92.77% | 86.46% | 99.28% | 87.55% | 92.43% | 225+/-136 |
| CNNLSTM (PreFallKD's repro) (paper) | 97.67% | 88.35% | 94.58% | 98.13% | 91.36% | 493.5 |
| ViT-tiny (PreFallKD's teacher) (paper) | 98.36% | 92.02% | 95.73% | 99.36% | 93.84% | 235.4 |
| PreFallKD (KD-distilled student) (paper) | 98.05% | 90.62% | 94.79% | 98.53% | 92.66% | 551.3 |
| Our ConvLSTM baseline | -- | -- | 99.28% | 91.19% | -- | 224+/-135 |

## Correctness check (primary success criterion)

Distance from PreFallKD's ViT-tiny: accuracy delta +5.6pp, specificity delta -11.8pp.
**Reproduction is NOT close to PreFallKD's published range -- treat this as a signal to debug the implementation before drawing conclusions about attention vs LSTM.**

## Research question: does attention improve specificity over our ConvLSTM baseline?

Baseline specificity: 91.19%. Transformer specificity: 87.55%. Delta: -3.6pp.
