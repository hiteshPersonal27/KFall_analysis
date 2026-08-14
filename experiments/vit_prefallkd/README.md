# ViT-style Transformer (PreFallKD Reproduction)

Tests whether self-attention over the window's patch sequence captures
fall-vs-ADL patterns better than the LSTM stage does, using the same raw 1D
input as the existing baseline ConvLSTM (no CWT, no frequency transform —
that's the separate, independent `experiments/cwt_lstm/` experiment). Follows
PreFallKD (Chi, Liu, Hsieh, Tsao & Chan, ICASSP 2023) — the only published
lightweight-transformer approach validated on KFall itself, chosen
specifically because it gives this experiment an *external* number to check
correctness against (unlike the CWT experiment, which had none). Full
original plan: `../../docs/CNN_Transformer_Implementation_Plan.md`.

**Result: the reproduction does NOT validate closely against PreFallKD's
published numbers, and (taken at face value) attention does not improve
specificity over the ConvLSTM baseline either.** See "Final result" below —
and the caveat right after it about what this negative result can and can't
be trusted to mean, given the validation gap.

## Hypothesis

The baseline ConvLSTM's LSTM stage collapses to only 1-6 sequence steps
before classifying (see `paper_implementation/convlstm_model.py` and
`experiments/cwt_lstm/model.py`'s shape-tracking notes) — arguably not doing
much genuine sequential modeling. A transformer's self-attention explicitly
relates every patch to every other patch regardless of distance, which might
capture the fall-vs-ADL distinction more effectively.

## Methodology

**Patching** (`data.py`): PreFallKD reshapes each `(50, 9)` window into a
patch sequence, mirroring how ViT slices an image into a grid. Patch length
= 10 frames (→ 5 time-chunks). Patch axis = 3 channels per patch (→ 3
channel groups). **The exact channel grouping was an open item in the plan
with no PreFallKD source available to verify against** — resolved here by
grouping by sensor modality (accel xyz / gyro xyz / euler xyz), the only
grouping-into-3 with physical meaning for this 9-channel sensor. Result: 15
total patches (5 time-chunks × 3 channel-groups), each flattened from
`10 frames × 3 channels = 30` raw values, linearly projected into the
model's hidden size. This is a documented interpretation, not a verified
match to the original paper — see "What's still open" below.

**Architecture** (`model.py`): patch embedding (linear) → prepend trainable
CLS token → add trainable position embeddings → transformer encoder
(multi-head self-attention + LayerNorm + MLP, `norm_first=True`, GELU) →
classify from the CLS token's final representation. PreFallKD's "tiny"
sizing: 3 layers, 3 heads, MLP size 256, dropout 0.2.

**A real inconsistency in the plan, fixed and documented**: the plan states
hidden_size=64 with 3 attention heads — but 64 isn't divisible by 3, and
PyTorch's (and virtually every standard) multi-head attention implementation
requires `hidden_size % n_heads == 0`. Adjusted to **hidden_size=63** (the
nearest multiple of 3, a 1.6% deviation) rather than changing the head
count, since 3 heads was explicit and 64 might just be an approximation of
63 or 66 in whatever source produced the plan's summary.

**Training** (`train.py`): AdamW (not plain Adam, per PreFallKD's own setup
and the plan's explicit instruction), cross-entropy loss (the plan's
"acceptable simplification" vs. PreFallKD's own focal loss — not
implemented in this pass, see "What's still open"), fixed seed, validation
split from training subjects, best checkpoint by validation balanced
accuracy. Full scale directly (`stride=1`, uncapped ADL windows) — unlike
the CWT experiment, raw `(50, 9)` window storage costs the same as the
baseline ConvLSTM's (already proven safe at this exact scale), so there was
no memory-safety reason to stage this one small-then-large; a quick 4-subject
smoke test confirmed the pipeline ran correctly before committing to the
full ~23-minute run.

**Fall oversampling**: PreFallKD's own setup oversamples pre-impact fall
windows 6x to handle class imbalance. Our own training set has a similarly
skewed fall:ADL window ratio (~1:37-42 raw, matching what's documented in
`paper_implementation/README.md`), so this was judged applicable and
implemented (`OVERSAMPLE_FALL=6` in `train.py`) rather than skipped.

**Evaluation** (`evaluate.py`): per-window prediction → per-trial decision
via the same `CONSEC_WINDOWS=2` persistence rule used throughout this
project. Full test-set resolution (`stride=1`) — again safe here since raw
window storage doesn't have the CWT experiment's memory cost.

## Final result

| Model | Accuracy | Precision | Recall (Sens.) | Specificity | F1 | Lead time (ms) |
|---|---|---|---|---|---|---|
| **This repro (transformer)** | — | — | 94.53% | 87.55% | — | 225±136 |
| PreFallKD's CNNLSTM (their own repro) | 97.67% | 88.35% | 94.58% | 98.13% | 91.36% | 493.5 |
| PreFallKD's ViT-tiny (their teacher) | 98.36% | 92.02% | 95.73% | 99.36% | 93.84% | 235.4 |
| PreFallKD (their KD-distilled student) | 98.05% | 90.62% | 94.79% | 98.53% | 92.66% | 551.3 |
| Our own ConvLSTM baseline | — | — | 94.53% | 93.68% | — | 224±136 |

FN=24, FP=65, TP=415, TN=457 on the corrected 439-fall/522-ADL test set (see
note below on the window-labeling fix).

**Correctness check (primary success criterion, per the plan): FAILED.**
Specificity is well below PreFallKD's ViT-tiny (87.55% vs 99.36%). The plan
is explicit about what this means: *"If our numbers are far off in either
direction, treat that as a signal to debug the implementation before drawing
conclusions about attention vs LSTM."* This deviation is well outside a
reasonable tolerance — this reproduction is not validated against the
reference.

**Research question (taken at face value): attention does NOT improve
specificity over our own ConvLSTM baseline.** 87.55% vs. 93.68%, a 6.1pp
regression, while sensitivity ties exactly (94.53% both — both models miss
the identical 24 fall trials, suggesting sensitivity is close to a practical
ceiling in this setup regardless of architecture, and specificity is where
architectures actually differentiate, consistent with every other experiment
in this project). This same 24-trial miss pattern recurs across every
independent architecture in the project that reaches 94.53% sensitivity
(ConvLSTM, this ViT repro, the no-pool Conv+Transformer, GAF global-norm) —
strong evidence these are genuinely hard falls (very short onset-to-impact
duration), not a modeling artifact.

**Note on the numbers above:** these reflect a window-labeling bug fix in
`paper_implementation/data.py` (see that project's README) that was found
and fixed after this experiment's initial run — every experiment's test set
now correctly evaluates on the full 439 fall / 522 ADL trials, correcting
earlier numbers (92.77%/99.28%/87.55% accuracy/sensitivity/specificity here,
99.28%/91.19% for the ConvLSTM baseline) that were silently missing ~21-29
fall trials. This repro was re-evaluated on the corrected test set; accuracy/
precision/F1 are omitted above since they weren't recomputed post-fix.

**Lead time**: 225±136ms, essentially identical to the ConvLSTM baseline
(224±136ms) and close to PreFallKD's own ViT-tiny lead time (235.4ms) — both
comfortably faster than PreFallKD's CNNLSTM (493.5ms) and their distilled
student (551.3ms), but like PreFallKD's own transformer, below their stated
333ms airbag-deployment requirement.

## Important caveat on interpreting this result

**This is NOT a clean "attention doesn't help" conclusion** the way the CWT
experiment's negative result was clean (that experiment validated its
methodology via a visual sanity check before training, and its poor result
couldn't be explained by an implementation gap). Here, the correctness check
against PreFallKD's own numbers failed, meaning there's a real, unresolved
gap between this reproduction and the reference implementation — plausible
causes include the guessed channel-grouping, the cross-entropy-vs-focal-loss
simplification, missing data augmentation (PreFallKD also uses Gaussian
noise injection and magnitude scaling, not implemented here), or other
undocumented PreFallKD details. Until that gap is closed, the specificity
regression vs. the ConvLSTM baseline should be read as "this particular
reproduction underperforms," not as strong evidence that attention itself
is the wrong tool for this problem.

## Decision: Section 2b (CNN student + knowledge distillation) — NOT pursued

The plan is explicit: *"Only pursue this after 2a produces a working,
validated transformer."* 2a did not validate. Additionally, 2b's whole
purpose per the plan is solving a **lead-time** problem (PreFallKD's own
ViT-tiny alone was too slow at 235.4ms) — but this reproduction's lead time
(225ms) is already in the same fast range and not the bottleneck here; the
actual problems are the correctness-check gap and the specificity
regression, neither of which distillation addresses. Building a distillation
pipeline on top of an unvalidated, currently-worse-than-baseline teacher
isn't a good use of effort until the underlying gap is understood.

## What's still open (candidate next steps, not pursued in this pass)

- The channel-grouping-into-3 interpretation (sensor modality grouping) is
  a documented guess, not a verified match to PreFallKD's actual patching.
- Focal loss (PreFallKD's own loss function) vs. this pass's cross-entropy
  simplification.
- PreFallKD's additional data augmentation (Gaussian noise injection,
  magnitude scaling) beyond the 6x fall oversampling implemented here.
- The hidden_size=64→63 adjustment (forced by the head-count divisibility
  constraint) is a small but real deviation from the plan's stated sizing.

## Files

```
experiments/vit_prefallkd/
├── data.py                # raw windowing (reused) + patchify + patch-grouping decision
├── model.py                # ViT-style transformer, PreFallKD "tiny" sizing (hidden=63)
├── train.py                 # AdamW, cross-entropy, 6x fall oversampling
├── evaluate.py               # test-set evaluation + 3-way comparison
└── results/
    ├── transformer.csv        # per-trial predictions
    ├── transformer_best.pt    # best checkpoint
    ├── norm_stats.npz         # per-channel normalization stats
    └── comparison.md          # generated comparison table
```

## Run commands

```bash
# 1. Patch-reshape sanity check
/mnt/d/KFall/venv/bin/python3 experiments/vit_prefallkd/data.py

# 2. Model shape/param check
/mnt/d/KFall/venv/bin/python3 experiments/vit_prefallkd/model.py

# 3. Train (full scale directly -- raw window storage is memory-safe at this scale)
/mnt/d/KFall/venv/bin/python3 experiments/vit_prefallkd/train.py

# 4. Evaluate + compare against PreFallKD's published numbers and the ConvLSTM baseline
/mnt/d/KFall/venv/bin/python3 experiments/vit_prefallkd/evaluate.py
```
