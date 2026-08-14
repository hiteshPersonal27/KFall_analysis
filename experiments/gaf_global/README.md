# GAF Global-Normalization Fix

Direct follow-up to `experiments/gaf_mtf/`'s catastrophic result (9.20%
specificity, ~93% validation accuracy ceiling). Diagnosis there: GASF's
normalization is inherently *per-window* — each window rescaled to `[-1,1]`
using its own min/max, not a fixed statistic. Two overlapping windows
sharing 90% of their frames could still get different local min/max
references, discontinuously shifting the entire pairwise-angle-sum image —
plausibly explaining both the noisy training (near-identical inputs mapped
to visually different images, same label) and the unstable evaluation
(`CONSEC_WINDOWS` persistence collapsing).

**This experiment tests the fix: global normalization** — per-channel
min/max computed once from training data, applied identically to every
window at train and test time. Trained at **full scale** (not the 150K
matched scale `gaf_mtf` used), per explicit direction to prioritize the
best achievable result.

**Result: the fix worked dramatically, but a real gap to the ConvLSTM
baseline remains.** See "Final result" below.

## Two real memory bugs, found and fixed during this experiment

Worth documenting honestly, since both crashed/failed the run before a
working version was reached — a second, distinct memory lesson beyond
`experiments/cwt_lstm/`'s original OOM incident:

**Bug 1 — full float32 copies.** The first `train.py` attempt computed
per-channel mean/std for the post-encoding normalization step with
`X_fit.astype(np.float32)` directly on the full ~20GB float16 array. This
creates a full ~40GB float32 *copy* as a temporary — done twice (mean and
std) — and OOM'd the machine (killed VS Code), the same class of mistake
`cwt_lstm` made once already, just recurring in a new file. Fixed by
reshaping (a view, not a copy, for a contiguous array) and using
`dtype=np.float64` as the reduction accumulator for the **mean** — numpy
computes reductions with an explicit dtype in a streaming fashion, no full
upcast copy needed.

**Bug 2 — `.std()` is not actually streaming.** The mean fix above was
correct, but applying the same `dtype=np.float64` approach to `.std()`
failed differently: numpy's variance computation internally computes
`array - mean` as a **full temporary array** before squaring — this tried
to allocate ~79GB and failed with a clean Python `MemoryError` (no system
crash this time, but still wrong). Fixed with an explicit **two-pass
chunked** standard deviation: mean first (the streaming reduction, which is
genuinely safe), then sum of squared deviations accumulated chunk-by-chunk
for the variance — never materializing more than one chunk's temporary at
once. Verified via a memory-monitored test showing zero RSS delta before
committing to the real full-scale run again.

Both fixes are in `train.py`, commented in place with the specific numbers
that failed, so this doesn't get silently reintroduced later.

## Methodology

**Global stats** (`data.py`'s `compute_global_stats`): per-channel (ACC_M,
GYR_M, VV) min/max computed from the **fit subjects'** full, unwindowed
trial signals (not just windows) — fit-on-train-only, same convention as
every other normalization step in this project. Saved to
`results/global_stats.npz`, reused identically for validation and test
windows.

**`gasf_global`**: same GASF math as `gaf_mtf`'s `gasf()` (min-max rescale
→ `arccos` → pairwise cosine-sum), but rescaling uses the fixed
`(ch_min, ch_max)` passed in, not the window's own min/max.

**Model**: identical to `gaf_mtf/model.py` (`GAF_CNN`, plain 2D CNN → global
average pool → dense → softmax) — unchanged, since the model was never the
hypothesized problem.

**Training scale**: full (`stride=1`, uncapped ADL windows) — 1.41M fit +
345K val windows. Stored as **`float16`** (not `float32`) specifically to
keep this safely under available RAM (~20GB for the full fit+val set vs.
~40GB at float32) — see the memory-bug section above for why this still
needed care. `DEFAULT_MAX_MEMORY_GB` raised to 28 (from `gaf_mtf`'s 8),
deliberate and documented, not a removed safety check — the guard itself
stayed active throughout.

**Wall-clock**: data build 238s, training 2778s (111s/epoch, 25 epochs) —
**total ~50 minutes**. Evaluation (full `stride=1` test set, ~428K windows,
~6GB) took ~105s.

## Visual sanity checks (go/no-go, before any model code)

Two checks, per the plan:

1. **Fall vs. ADL structural difference** (`sanity_check_gaf_global.png`):
   still shows a real difference (fall has localized blob structure, jog is
   much flatter) — though the jog images now look visually more
   compressed/uniform than `gaf_mtf`'s originals, likely because the global
   range is dominated by rare extreme fall values (e.g. `gyr_m`'s global max
   of 482 deg/s), compressing ordinary ADL dynamics into a narrow slice of
   the `[-1,1]` range. Worth flagging as a real trade-off of this fix, not
   ignored.
2. **Overlap-stability, the direct fix confirmation**
   (`sanity_check_overlap_stability.png`): two windows 5 frames apart,
   before/after. Mean absolute pixel difference dropped from **0.397**
   (per-window norm) to **0.050** (global norm) — an **~8x reduction** in
   exactly the instability diagnosed as the root cause. Strong, direct
   evidence the fix targets the right mechanism.

## Training result: validation accuracy ceiling recovered

| | Validation balanced accuracy (best epoch) |
|---|---|
| `gaf_mtf` (per-window norm, 150K scale) | ~93% |
| `gaf_global` (this fix, full scale) | **~99.1%** |

This matches the training-quality prediction from the diagnosis: with
stable, non-flickering inputs, the model reaches the same ~99% ceiling
every other experiment in this project reaches — not stuck below it like
the original GAF attempt.

## Final result

| | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| **GAF, global norm (this fix, full scale)** | **94.53%** | **66.48%** | 223±131 ms |
| GAF, per-window norm (original, 150K scale) | 80.64% | 9.20% | 219±118 ms |
| ConvLSTM baseline (full scale) | 94.53% | 93.68% | 224±136 ms |

FN=24, FP=175, TP=415, TN=347 on the corrected 439-fall/522-ADL test set.

**The fix produced a large, genuine improvement on both metrics** —
sensitivity +13.89pp (80.64%→94.53%), specificity **+57.28pp** (9.20%→66.48%)
— not a trade-off where one improved at the other's expense. Sensitivity
now exactly matches the ConvLSTM baseline (same 24 missed falls).

**`CONSEC_WINDOWS` sweep** (diagnostic: is the flicker pattern actually
reduced, not just the headline number better?):

| CONSEC_WINDOWS | Sensitivity | Specificity |
|---|---|---|
| 1 (no persistence) | 100.00% | 65.13% |
| 2 (project default) | 94.53% | 66.48% |
| 3 | 93.62% | 68.01% |
| 5 | 90.66% | 72.61% |
| 8 | 85.65% | 79.12% |

Compare to `gaf_mtf`'s original sweep (consec=1: 1.15% specificity;
sensitivity collapsing to 9.27% by consec=8). Here, specificity climbs
*smoothly and substantially* even at `consec=1` (65.13%, vs. the original's
1.15%), and sensitivity degrades gracefully rather than collapsing — direct
confirmation the aggregation-instability mechanism is fixed, not just
patched over by a different number.

**What's NOT fully fixed**: a real ~27pp specificity gap to the ConvLSTM
baseline remains (66.48% vs. 93.68%) even after the normalization fix and
full-scale training. This means the original diagnosis was correct and the
fix worked as intended, but per-window normalization instability was not
the *only* factor limiting GAF's performance — something about the GASF
representation itself (or, per the visual sanity check's flagged trade-off,
the way global normalization compresses ordinary-ADL dynamic range when the
global range is dominated by rare extreme fall values) still leaves it
behind the raw 1D ConvLSTM baseline. Consistent with this project's broader
pattern (`experiments/conv_transformer/README.md`'s analysis): richer 2D
representations keep underperforming the simpler raw-1D approach, for
different specific reasons each time, which is itself informative about
where the real bottleneck likely is (not "the input representation isn't
rich enough").

## Files

```
experiments/gaf_global/
├── data.py                             # global stats + gasf_global + float16 + raised mem cap
├── model.py                             # identical to gaf_mtf's GAF_CNN
├── train.py                              # full-scale training (both memory bugs documented + fixed)
├── evaluate.py                            # test-set evaluation + CONSEC_WINDOWS sweep + comparison
├── sanity_check_gaf_global.png            # go/no-go: fall vs. ADL (global norm)
├── sanity_check_overlap_stability.png     # THE fix confirmation: before/after overlap stability
└── results/
    ├── gaf_global.csv                        # per-trial predictions
    ├── gaf_global_best.pt                    # best checkpoint
    ├── norm_stats.npz                        # per-channel image standardization stats
    ├── global_stats.npz                      # per-channel GASF rescaling min/max (the fix itself)
    └── comparison_vs_baseline.md             # generated comparison table + sweep
```

## Run commands

```bash
# 1. Visual sanity checks (go/no-go + fix confirmation)
/mnt/d/KFall/venv/bin/python3 experiments/gaf_global/data.py --sanity-check

# 2. Train (full scale -- ~50 min; watch `free -h` before/during, see memory-bug notes above)
/mnt/d/KFall/venv/bin/python3 experiments/gaf_global/train.py

# 3. Evaluate + compare against gaf_mtf and ConvLSTM baselines + CONSEC_WINDOWS sweep
/mnt/d/KFall/venv/bin/python3 experiments/gaf_global/evaluate.py
```
