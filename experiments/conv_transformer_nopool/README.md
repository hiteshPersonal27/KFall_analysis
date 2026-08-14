# CNN + Transformer, No Pooling — Testing the Pooling-Information-Loss Hypothesis

Direct follow-up to `experiments/conv_transformer/`'s result (attention
performed worse than LSTM: 81.03% vs. ConvLSTM's 93.68% specificity). That
experiment's README identified reason #1 for the regression as: the conv
stack's 3x `MaxPool1d` compressed the 50-frame window down to just 6
time-steps before the sequence stage, putting attention in a regime where
its long-range advantage doesn't apply but its cost (learning order from
scratch) still does.

**Classification-head fix carried over:** like `experiments/conv_transformer/`,
this model uses global average pooling (`out.mean(dim=1)`) over the sequence
for classification, not the LSTM baseline's "last timestep" convention —
non-causal self-attention has no reason to treat the last position as a
cumulative summary. The numbers below are from the retrained mean-pooled
model.

This experiment tests that specific hypothesis directly: **remove all
pooling**, so the transformer sees the (near-)full 50-step sequence instead
of a compressed 6-step summary, and check whether specificity moves toward
the ConvLSTM baseline.

**Result: the hypothesis is confirmed.** After the classification-head fix
(mean pooling instead of the LSTM baseline's "last timestep" convention) and
a re-run at full training scale, removing pooling raised specificity by
+6.13pp (87.16% vs. 81.03%) — see "Final result" below. (An earlier draft of
this section, based on a reduced 150K-scale run using the stale
"last-timestep" head, found the opposite; that result is superseded.)

## Methodology

**Model** (`model.py`): identical conv config to `experiments/conv_transformer/`
(filters 32/64/128, kernel size 3) but with `padding=1` (preserves sequence
length) and **no `MaxPool1d` anywhere**. Verified via shape dry-run: the
transformer now operates on a **50-step sequence** (vs. 6 for the pooled
version). Same position-embedding approach, same head-count logic (4 heads,
nearest divisor of 128), same global-average-pooling classification
convention (`out.mean(dim=1)`) — only the conv-stack pooling changed, per
the task's isolation requirement.

**Training** (`train.py`): same recipe as the pooled version (`Adam`, same
LR/batch/epochs, `dropout=0.5`, `CONSEC_WINDOWS=2` at eval). **Training
scale was deliberately reduced** for this first run — `stride=3`,
`MAX_ADL_TRAIN_WINDOWS=150000` (matching `experiments/cwt_lstm/`'s own
reduced-scale approach) — as a compute-cost precaution: self-attention cost
scales with the *square* of sequence length, so going from 6 to 50 tokens
is a theoretical ~70x increase in attention compute per window. Wall-clock
time was logged explicitly to check whether full-scale would even be
practical.

**Evaluation** (`evaluate.py`): full-resolution test set (`stride=1`),
`CONSEC_WINDOWS=2` aggregation — identical to every other experiment.

## Wall-clock result: full-scale WOULD have been practical

| | Time |
|---|---|
| Data build (150K scale) | 107s |
| Training (25 epochs, 150K scale) | 261s (10.4s/epoch) |
| **Total** | **368s (~6 min)** |

The theoretical 70x attention-cost increase did **not** materialize in
practice — per-batch time was only ~2x the pooled version's (attention is
just one part of total compute; conv layers, MLP layers, and fixed
per-batch overhead don't scale with sequence length). Extrapolating this
run's per-batch cost to the full 1.41M-window uncapped scale suggests
roughly ~35-40 minutes total — comparable to the other full-scale runs in
this project, not prohibitive.

Since full-scale was confirmed practical here, and the pooled version was
subsequently retrained at full scale anyway (for the classification-head
fix), this model was also retrained at full scale for a fair, scale-matched
final comparison — see "Final result" below, which supersedes the original
150K-scale numbers.

## Final result: specificity improved, and now beats the pooled version

**Update (post mean-pooling and window-labeling fixes):** the numbers below
supersede an earlier draft of this section (99.28%/77.20% at 150K training
scale, run before the classification-head fix and the window-labeling data
fix). The model was retrained with global-average-pooling classification and
re-evaluated on the corrected full 439-fall/522-ADL test set, at full scale:

| | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| **No-pooling transformer (mean-pooled head, full scale, corrected test set)** | **94.53%** | **87.16%** | 224±137 ms |
| Pooled transformer (mean-pooled head, corrected test set) | 93.17% | 81.03% | 227±136 ms |
| ConvLSTM baseline (corrected test set) | 94.53% | 93.68% | 224±136 ms |

FN=24, FP=67, TP=415, TN=455 on the corrected 439-fall/522-ADL test set.

- vs. the pooled transformer: **+6.13pp** specificity (87.16% vs. 81.03%)
- vs. ConvLSTM baseline: **-6.52pp** specificity (87.16% vs. 93.68%)

**Success criterion (per the task): did removing pooling move specificity
meaningfully toward the ConvLSTM baseline compared to the pooled version?
Yes.** No-pooling closes about half the gap between the pooled transformer
and the ConvLSTM baseline, without costing sensitivity — it ties the
baseline's sensitivity exactly (94.53%, same 24 missed falls), while the
pooled version sits slightly lower (93.17%, 30 FN).

## What this means for the original hypothesis

**Update: this section's original conclusion (pooling-induced information
loss refuted) was based on pre-fix numbers and does not hold up.** The
original run compared a no-pooling transformer using the stale "last
timestep" classification head against a pooled version using the same stale
head, both at mismatched training scales (150K vs. full), and found
no-pooling *worse*. After the classification-head fix (mean pooling instead
of last-timestep, see the top of this file) and the window-labeling data fix
were applied and both models were retrained at full scale on the corrected
test set, **no-pooling is clearly better than pooling**: 87.16% vs. 81.03%
specificity, a **+6.13pp** improvement, closing roughly half the remaining
gap to the ConvLSTM baseline (93.68%).

This is consistent with the original hypothesis after all: compressing the
50-frame window down to 6 tokens before the attention stage was destroying
information attention could otherwise use, and removing that compression
recovers a meaningful chunk of the specificity gap. The earlier "hypothesis
refuted" conclusion was an artifact of the classification-head bug (and the
scale mismatch), not a genuine finding about pooling.

The remaining gap to the ConvLSTM baseline (87.16% vs. 93.68%, -6.52pp) is
still real and unexplained by pooling alone — the other candidate reasons in
`experiments/conv_transformer/README.md` (generalization gap, effective data
diversity, training-recipe mismatch) remain live explanations for what's
left.

## Files

```
experiments/conv_transformer_nopool/
├── model.py                            # same conv config, MaxPool1d removed -> 50-token sequence
├── train.py                             # same recipe as conv_transformer, full scale (stride=1, uncapped)
├── evaluate.py                           # test-set evaluation + 4-way comparison
└── results/
    ├── conv_transformer_nopool.csv          # per-trial predictions
    ├── conv_transformer_nopool_best.pt      # best checkpoint
    ├── norm_stats.npz                       # per-channel normalization stats
    └── comparison_vs_baseline.md            # generated comparison table
```

## Run commands

```bash
# 1. Model shape/param check (confirms 50-token sequence length)
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer_nopool/model.py

# 2. Train (full scale: stride=1, ADL uncapped)
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer_nopool/train.py

# 3. Evaluate + compare against pooled version and ConvLSTM baselines
/mnt/d/KFall/venv/bin/python3 experiments/conv_transformer_nopool/evaluate.py
```
