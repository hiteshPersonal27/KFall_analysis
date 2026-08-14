# CWT + 2D-CNN Experiment

Tests whether adding explicit frequency information (via Continuous Wavelet
Transform) closes the specificity gap that persisted in the baseline raw-1D
ConvLSTM (`paper_implementation/convlstm_model.py`) even after training on
its full 1.4M-window dataset. This is a genuinely open experiment — no
published KFall result exists for this exact approach, so there's no
external number to validate against, only an internal comparison against the
baseline. Full original plan: `../../docs/CWT_2DCNN_Implementation_Plan.md`.

**Result: the hypothesis is refuted.** CWT+2D-CNN performs *worse* than the
raw-1D baseline on both metrics, not better. See "Final result" below.

## Hypothesis

Vigorous ADLs (jogging, jumping, quick sit-downs) and real falls produce
similar raw-amplitude spikes but might differ in frequency character — ADLs
are repetitive/rhythmic, a fall onset is a one-off, low-frequency event. Raw
1D amplitude alone (what the baseline ConvLSTM sees) may not expose this
difference clearly; an explicit frequency decomposition might.

## Methodology

**Wavelet & scales**: complex Morlet (standard w0=6 formulation), 24
log-spaced scales covering ~1-45 Hz. Each `(50, 9)` raw window (same 50-frame
windowing as the baseline, reusing `paper_implementation/data.py`'s split and
window logic) becomes a `(9, 24, 50)` scalogram — one 2D time-frequency map
per sensor channel.

**Visual sanity check (go/no-go per the plan, run before any model code)**:
compared scalograms of a fall window (near impact) vs. a jog window.
Confirmed real structural differences — see `sanity_check_scalograms.png`.
AccX: the fall shows a localized transient blob emerging late in the window;
the jog shows a persistent narrow band sustained across the *entire* window
— exactly the "one-off event vs. rhythmic" distinction the hypothesis
predicted. This was a legitimate "go" signal.

**Architecture** (`model.py`): mirrors the baseline's structure so the
comparison isolates the 1D-vs-2D/frequency change specifically — 3x
(Conv2D→BatchNorm2D→ReLU→MaxPool2D) blocks with the same 32/64/128 filter
progression, then an LSTM, then dense+softmax. The conv stack's actual
pre-LSTM sequence length was checked (per the plan's explicit concern that
the original 1D model collapsed to a sequence length of 1, making its LSTM
just an extra nonlinear layer, not real sequential modeling): here it comes
out to **6**, so this version does retain genuine multi-step sequential
information into the LSTM — one structural advantage over the baseline that
didn't translate into better results (see below).

**Training** (`train.py`): CrossEntropyLoss with inverse-frequency class
weights, Adam, fixed seed, validation split from training subjects, best
checkpoint by validation balanced accuracy — same conventions as the
baseline. Per-channel standardization fit on training data only.

**Evaluation** (`evaluate.py`): per-window prediction → per-trial decision
via the same `CONSEC_WINDOWS=2` consecutive-window persistence rule used
throughout this project (`paper_implementation/svm_model.py`'s comment has
the rationale), so results are directly comparable.

## GPU-accelerated CWT (a significant implementation change from the plan)

The plan's original build order assumed `pywt.cwt()` (Python, CPU,
one-window-at-a-time). Measured: **~14ms/window** — the initial small-scale
(20K-window) training run took the CWT step alone ~8 minutes, and a
full-dataset test-set pass was projected at ~20 minutes.

CWT is fundamentally "convolve the signal with a bank of scaled wavelet
kernels" — exactly what `torch.conv1d` already does, and it batches
trivially across thousands of windows at once on a GPU. Reimplemented from
scratch as a batched PyTorch op (`gpu_cwt.py`): **~0.037ms/window, a ~380x
speedup**. Verified to produce the same qualitative scalogram patterns as
the pywt version (re-ran the sanity check after switching — same transient
blob vs. persistent band distinction, see `gpu_cwt.py`'s docstring for why
exact numerical parity with pywt isn't necessary here: same wavelet family,
this experiment trains and evaluates entirely on its own output).

## A real mistake, and the fix (worth documenting honestly)

Reasoning that GPU CWT removed the compute bottleneck, the first attempt set
`TRAIN_WINDOW_STRIDE=1, MAX_ADL_TRAIN_WINDOWS=None` — matching the baseline's
best (uncapped) settings directly, without checking memory first. **This was
wrong**: each CWT scalogram is `(9, 24, 50)` — **24x bigger** per window than
the baseline's raw `(9, 50)` window (24 scales × the raw channel count). The
full ~1.76M-window dataset needed **~74GB** just for the training array,
which exceeded the machine's 61GB RAM and crashed the system (killed VS Code
along with the training process).

**Fix applied**:
- `data.py` now estimates memory before allocating anything and raises
  `MemoryError` if a request would exceed a configurable cap
  (`DEFAULT_MAX_MEMORY_GB=8.0`), rather than silently trying and OOMing.
- Scalograms are stored as `float16` (halves memory vs. float32); cast to
  float32 per-batch during training/inference, where precision actually
  matters.
- Training scale was set back to `TRAIN_WINDOW_STRIDE=3,
  MAX_ADL_TRAIN_WINDOWS=150000` (~3.3GB) — deliberately matched to the
  baseline ConvLSTM's own intermediate 150K-window checkpoint, so the final
  comparison below is apples-to-apples on training data volume, not just
  compared against the baseline's best (full-scale) number.
- Test set uses `stride=5` (~1.7GB) rather than the baseline's `stride=1`
  (~8.6GB, right at the memory cap) for the same reason.

The re-run trained and evaluated cleanly with memory headroom to spare
throughout (confirmed via `free -h` during the run).

## Final result

| | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| **CWT + 2D-CNN + LSTM** (150K training scale) | **83.83%** | **73.75%** | 228±126 ms |
| Baseline raw-1D ConvLSTM, full-scale (corrected test set) | 94.53% | 93.68% | 224±136 ms |
| Paper (ConvLSTM) | 99.32% | 99.01% | 403±163 ms |

**Note:** the ConvLSTM baseline row above (94.53%/93.68%) reflects the
window-labeling bug fix in `paper_implementation/data.py` (see that
project's README) — every experiment's test set now correctly evaluates on
the full 439 fall / 522 ADL trials, correcting an earlier baseline number of
99.28%/91.19% that was silently missing ~21-29 fall trials. CWT+2D-CNN was
re-evaluated on the same corrected test set.

At a matched training scale, CWT+2D-CNN underperforms the raw-1D baseline on
**both** sensitivity (83.83% vs 94.53%, -10.7pp) and specificity (73.75% vs
93.68%, -19.9pp). This is despite validation balanced accuracy reaching
~99% during training — a sizeable train/val-to-test gap, suggesting the
richer (9, 24, 50) input let the model fit the training subjects more
precisely without that fit generalizing to held-out subjects as well as the
simpler raw-1D representation does.

**Conclusion, per the plan's stated success criteria**: this result **rules
out "missing frequency information" as the explanation** for the baseline
ConvLSTM's specificity plateau. The visual sanity check showed real,
interpretable frequency-domain differences between falls and vigorous ADLs
— so the frequency information genuinely exists in the signal — but giving
the model explicit access to it did not help, and by this evidence actively
hurt generalization. This points back toward the other candidate
explanations noted in the original plan: the LSTM's role, the persistence
aggregation rule, or training-recipe details the source paper doesn't
disclose, rather than a missing input representation.

## Files

```
experiments/cwt_lstm/
├── data.py                          # windowing + CWT transform + memory-safety guard
├── gpu_cwt.py                       # GPU-batched Morlet CWT (torch.conv1d)
├── model.py                         # 2D-CNN + LSTM architecture
├── train.py                         # training loop
├── evaluate.py                      # test-set evaluation + comparison
├── sanity_check_scalograms.png      # go/no-go visual check (fall vs. jog)
├── cache/                           # disk-cached CWT tensors (gitignored via docs/-style pattern -- large)
└── results/
    ├── cwt_lstm.csv                 # per-trial predictions
    ├── cwt_lstm_best.pt             # best checkpoint
    ├── norm_stats.npz               # per-channel normalization stats
    └── comparison_vs_baseline.md    # generated comparison table
```

## Run commands

```bash
# 1. Visual sanity check (go/no-go, run before anything else)
/mnt/d/KFall/venv/bin/python3 experiments/cwt_lstm/data.py --sanity-check

# 2. Train (stride=3, ADL capped at 150K -- see the memory-safety note above
#    before raising these; check `free -h` first)
/mnt/d/KFall/venv/bin/python3 experiments/cwt_lstm/train.py

# 3. Evaluate + compare against the baseline
/mnt/d/KFall/venv/bin/python3 experiments/cwt_lstm/evaluate.py
```
