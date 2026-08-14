# CNN + Transformer — Minimal LSTM-vs-Attention Ablation

Tests the LSTM-vs-attention question as cleanly as possible: takes the exact
same 1D-CNN feature extractor as the baseline ConvLSTM
(`paper_implementation/convlstm_model.py`) and swaps **only** the LSTM stage
for a Transformer encoder. Everything else — conv filters, kernel size,
dropout, training recipe (seed, batch size, epochs, LR, optimizer,
window stride, ADL cap, per-trial aggregation rule) — is kept identical on
purpose, so any specificity difference isolates the recurrent-vs-attention
architecture choice specifically.

**Classification-head fix (found during validation):** this model originally
copied the LSTM baseline's `out[:, -1, :]` ("last timestep") classification
convention. That convention only makes sense for the LSTM, whose recurrence
is causal — the last timestep's hidden state is a genuine cumulative summary
of everything before it. This transformer's self-attention is *not*
causally masked (bidirectional), so "last position" carries no such special
meaning; it's just one arbitrary token like any other. **Fixed to global
average pooling (`out.mean(dim=1)`)** over all sequence positions instead,
and the model was retrained from scratch with this change. All numbers in
this README reflect the mean-pooled architecture.

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
order sense, unlike an LSTM). Classification uses global average pooling
over the sequence (`out.mean(dim=1)`), not the baseline's "last timestep"
convention and not a CLS token — see the classification-head fix noted
above.

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
| CNN + Transformer (LSTM swapped, mean-pooled) | 93.17% | **81.03%** | 227±136 ms |
| Baseline ConvLSTM | 94.53% | **93.68%** | 224±136 ms |

FN=30, FP=99, TP=409, TN=423 on the corrected 439-fall/522-ADL test set.

**Verdict: in this clean, isolated swap, attention performed clearly worse
than the LSTM.** Specificity dropped 12.65pp (81.03% vs. 93.68%). Sensitivity
also dropped slightly (93.17% vs. 94.53%, 30 FN vs. 24 FN) — unlike some of
this project's other transformer variants, this one does not tie the
baseline's miss set exactly, though the gap is still small next to the
specificity gap, and specificity remains where architectures differentiate
most.

**Note on the numbers above:** these reflect two fixes applied after this
experiment's initial run: (1) the mean-pooling classification-head fix
described above (previously last-timestep, which produced 99.28%/80.08% —
that number is stale and superseded), and (2) the window-labeling bug fix in
`paper_implementation/data.py` (see that project's README), which corrected
the baseline ConvLSTM's own number from 99.28%/91.19% to 94.53%/93.68%. Both
this model and the baseline were retrained/re-evaluated on the corrected
439-fall/522-ADL test set.

Unlike `experiments/cnn_transformer/`'s result (which came with a caveat,
since that reproduction failed its own correctness check against PreFallKD's
published numbers), **this result has no such caveat** — the only variable
that changed between this and the baseline is the recurrent/attention stage
itself, everything else is identical by construction. This is a clean
negative result: for this specific architecture (a small conv feature
extractor feeding a 6-step sequence), the LSTM generalizes better to
held-out subjects than a small transformer does.

## Why attention underperformed here (analysis, not further verified in this pass)

Four contributing factors, in order of how well-evidenced each is by what
was actually measured:

**1. Sequence length (6 steps) is exactly the regime where attention's core
advantage doesn't apply, but its cost still does.** Self-attention exists to
solve the long-range-dependency problem: an RNN carrying information from
step 1 to step 200 has to survive 199 update steps, which is genuinely hard
(vanishing gradients). Attention lets any position reach any other position
in one hop, fixing that. But this sequence is only 6 steps long — an LSTM
surviving 5 update steps is a non-issue, so attention's core benefit isn't
being exercised. Meanwhile its cost is real: attention has no inherent
sense of order (it would treat the sequence identically in any permutation
unless told otherwise), so it needs learned position embeddings — starting
from random noise — to recover something an LSTM gets for free from its
step-by-step, causal processing. For a task that's fundamentally about
temporal order (accel drop, then rotation, then impact spike), that's a
real handicap with no offsetting benefit at this sequence length.

**2. The result pattern is a classic generalization gap, not underfitting.**
Validation balanced accuracy was ~99.5% for BOTH models during training —
essentially tied. The divergence (81.03% vs. 93.68% specificity) only shows
up on the held-out TEST subjects, a completely different group of people
from training/validation. If the transformer were simply "worse," it should
have also scored worse on validation; it didn't. This specific
pattern — equal on data resembling training, unequal on genuinely new
people — means the transformer learned *something* that worked on the
training subjects but doesn't transfer, e.g. subject-specific movement
quirks that happen to correlate with the label in the training set, rather
than the subject-invariant physical signature of a fall. A model with fewer
built-in constraints (no recency/order bias forcing it toward the physically
correct signal) has more freedom to lean on this kind of pattern.

**3. "1.4M training windows" overstates real data diversity.** Windows were
built with `stride=1` -- consecutive windows share 49 of 50 frames, so
they're near-duplicates of each other, not independent examples. The real
diversity is bounded by the actual number of distinct fall events, subjects
(21 in training), and task types (~36), not the inflated window count.
Position embeddings, learned entirely from data with no prior, are only as
good as how *diverse* (not how *numerous*) that data is -- this is the same
phenomenon documented in the original ViT paper, where transformers
underperform CNNs until trained on datasets orders of magnitude larger and
more diverse than what compensates a CNN's built-in translation-invariance
bias. The LSTM's order bias didn't need that diversity to already be
correct; the transformer's did, and may not have gotten enough.

**4. A genuine confound: the training recipe was deliberately NOT
transformer-tuned.** To isolate the architecture change as the only
variable, this experiment kept the LSTM baseline's exact recipe (`Adam`,
`dropout=0.5`, no LR warmup, no label smoothing). Transformers are commonly
more sensitive to training recipe than LSTMs (warmup schedules, `AdamW`,
different dropout placement are typical). So part of this gap could be
"attention needs different training conditions than an LSTM-tuned recipe
provides," not purely "attention is worse at this task" -- this experiment's
design can't distinguish between those two explanations, and doing so would
need a second run with a transformer-appropriate recipe (not done in this
pass).

## Candidate next steps (not implemented in this pass)

- **Deepen/restructure the conv front-end** (e.g. more `Conv1d → MaxPool`
  stages before the sequence stage, or a slower downsampling schedule that
  leaves more than 6 time-steps for the recurrent/attention stage to work
  with) -- directly targets reason #1 by giving attention an actual
  sequence-length regime where its long-range advantage could matter, rather
  than compressing away almost all of the temporal structure before the
  attention stage ever sees it.
- Transformer-appropriate training recipe (LR warmup, `AdamW`, tuned
  dropout) to isolate reason #4 from reasons #1-3.
- Lower stride (less window overlap) or synthetic augmentation to test
  reason #3's data-diversity hypothesis directly.

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
