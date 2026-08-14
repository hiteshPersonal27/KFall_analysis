# ViT-style Transformer vs. Baseline ConvLSTM vs. PreFallKD's published numbers

| Model | Accuracy | Precision | Recall (Sens.) | Specificity | F1 | Lead time (ms) |
|---|---|---|---|---|---|---|
| **This repro (transformer)** | -- | -- | 94.53% | 87.55% | -- | 225+/-136 |
| CNNLSTM (PreFallKD's repro) (paper) | 97.67% | 88.35% | 94.58% | 98.13% | 91.36% | 493.5 |
| ViT-tiny (PreFallKD's teacher) (paper) | 98.36% | 92.02% | 95.73% | 99.36% | 93.84% | 235.4 |
| PreFallKD (KD-distilled student) (paper) | 98.05% | 90.62% | 94.79% | 98.53% | 92.66% | 551.3 |
| Our ConvLSTM baseline | -- | -- | 94.53% | 93.68% | -- | 224+/-136 |

FN=24, FP=65, TP=415, TN=457 (corrected 439-fall/522-ADL test set, post window-labeling fix).

## Correctness check (primary success criterion)

Distance from PreFallKD's ViT-tiny: specificity delta -11.8pp.
**Reproduction is NOT close to PreFallKD's published range -- treat this as a signal to debug the implementation before drawing conclusions about attention vs LSTM.**

## Research question: does attention improve specificity over our ConvLSTM baseline?

Baseline specificity: 93.68%. Transformer specificity: 87.55%. Delta: -6.1pp.
