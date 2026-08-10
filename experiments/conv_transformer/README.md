# CNN + Transformer — Minimal LSTM-vs-Attention Ablation

Tests the LSTM-vs-attention question as cleanly as possible: takes the exact
same 1D-CNN feature extractor as the baseline ConvLSTM
(`paper_implementation/convlstm_model.py`) and swaps **only** the LSTM stage
for a Transformer encoder. Everything else — conv filters, kernel size,
dropout, classification-head convention ("last timestep" from the recurrent/
attention stage), training recipe (seed, batch size, epochs, LR, optimizer,
window stride, ADL cap, per-trial aggregation rule) — is kept identical on
purpose, so any specificity difference isolates the recurrent-vs-attention
architecture choice specifically.

This is a **different, complementary experiment** to `experiments/cnn_transformer/`,
which reproduces PreFallKD's own ViT-style patch-tokenization architecture
(no conv layers, different loss, different training recipe) — that
experiment changes too many things at once to cleanly answer "does attention
beat LSTM," which is why this one exists as a separate, tightly-controlled
ablation. See that experiment's README for why it was built that way (it
follows an external reference paper) and its own, different result.

## Methodology

**Conv stack**: identical to the baseline — `Conv1d(9→32) → BatchNorm1d →
ReLU → MaxPool1d`, `×3` (filters 32/64/128, kernel size 3), producing a
`(128 channels, 6 time-steps)` sequence per window (verified via a shape
dry-run — matches the baseline's own measured post-conv sequence length
exactly).

**Transformer stage** (replaces the LSTM): operates directly on the conv
stack's `128`-channel output (no down-projection to a smaller hidden size,
unlike `experiments/cnn_transformer/`'s PreFallKD-style 63-dim model) — a
transformer has no separate "hidden size" the way an LSTM does, so it uses
its input dimension throughout. 2 encoder layers (matching the baseline's
`LSTM_LAYERS=2`), 4 attention heads (the natural divisor of 128 closest to
a "small" head count), dropout 0.5 (matching the baseline), learned
position embeddings added to the 6-step sequence (attention has no built-in
order sense, unlike an LSTM). Classification uses the same "last timestep"
convention as the baseline's `out[:, -1, :]`, not a CLS token.

**Training**: `Adam` (not `AdamW` — kept identical to the baseline
deliberately, unlike `experiments/cnn_transformer/`'s `AdamW` per PreFallKD's
own recipe), same seed/batch size/epochs/LR, same `stride=1` uncapped
full-scale window loading, class-weighted cross-entropy. A 4-subject smoke
test confirmed the pipeline before committing to the ~19-minute full run.

**Evaluation**: identical `CONSEC_WINDOWS=2` persistence aggregation and
`stride=1` test resolution as the baseline, for a directly comparable
result.

## Result

| | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| CNN + Transformer (LSTM swapped) | 99.28% | **80.08%** | 225±137 ms |
| Baseline ConvLSTM | 99.28% | **91.19%** | 224±135 ms |

**Verdict: in this clean, isolated swap, attention performed clearly worse
than the LSTM.** Specificity dropped 11.1pp (80.08% vs. 91.19%) while
sensitivity tied exactly (99.28% both — both models miss the identical 3
fall trials, the same pattern seen across every experiment in this project,
suggesting sensitivity is close to a practical ceiling here regardless of
architecture, and specificity is where architectures actually differentiate).

Unlike `experiments/cnn_transformer/`'s result (which came with a caveat,
since that reproduction failed its own correctness check against PreFallKD's
published numbers), **this result has no such caveat** — the only variable
that changed between this and the baseline is the recurrent/attention stage
itself, everything else is identical by construction. This is a clean
negative result: for this specific architecture (a small conv feature
extractor feeding a 6-step sequence), the LSTM generalizes better to
held-out subjects than a small transformer does.

**Plausible reason**: with only 6 time-steps per window and ~1.4M training
windows, this is a small-sequence, high-data regime — exactly where LSTMs'
built-in recency/order bias is a helpful inductive prior, and where a
transformer's lack of that prior (position embeddings are learned from
scratch, not built in) may need more data or more careful regularization to
match, not less. This is speculative, not verified further in this pass.

## Files

```
experiments/conv_transformer/
├── model.py                       # ConvTransformer: baseline conv stack + Transformer (not LSTM)
├── train.py                        # identical recipe to convlstm_model.py, only the model differs
├── evaluate.py                      # test-set evaluation + comparison
└── results/
    ├── conv_transformer.csv           # per-trial predictions
    ├── conv_transformer_best.pt       # best checkpoint
    ├── norm_stats.npz                 # per-channel normalization stats
    └── comparison_vs_baseline.md      # generated comparison table
```

## Run commands

```bash
# 1. Model shape/param check
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer/model.py

# 2. Train (full scale directly -- raw window storage, same memory cost as the baseline)
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer/train.py

# 3. Evaluate + compare against the baseline ConvLSTM
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer/evaluate.py
```
