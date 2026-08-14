# GAF 2D-Encoding Experiment

Tests whether encoding the raw signal as a Gramian Angular Field (GAF) image
— making pairwise time-step relationships explicit as 2D structure — helps
a CNN capture the fall-vs-ADL pattern better than the implicit shape a 1D
conv/LSTM has to infer from a sequence. Conceptually related to
`experiments/cwt_lstm/` (also a "richer 2D representation" experiment) but
methodologically distinct: GAF encodes pairwise time-step relationships
within the signal itself, no frequency-domain transform involved. Full
original plan: `../../docs/GAF_MTF_Implementation_Plan.md`.

**Follow-up:** the per-window normalization instability diagnosed below was
fixed and tested in `experiments/gaf_global/` (global normalization, full
training scale) — specificity improved from 9.20% to 66.48% and validation
accuracy recovered from ~93% to ~99%, confirming this diagnosis, though a
~27pp gap to the ConvLSTM baseline (93.68%) still remains. See that
experiment's README for the full before/after result.

**Result: the hypothesis is refuted, and more severely than the CWT
precedent — but the failure mode is different and mechanistically
explainable, not just "another overfitting case."** See "The real story"
below — the initial 9.20% specificity number looks like a broken model at
first glance, but window-level diagnostics show the underlying classifier
is actually reasonably accurate (94%); the collapse happens entirely in
trial-level aggregation, for an identifiable, GAF-specific reason.

## Methodology

**Scope decision**: GAF only (not MTF), computed on the 3 derived signals
central to this project (ACC_M, GYR_M, VV) rather than all 9 raw channels —
keeps input size/parameter count comparable to prior experiments, per the
plan. Used GASF (Gramian Angular **Summation** Field, the standard default)
via a direct numpy implementation (no new dependency): each channel's
50-length window is min-max rescaled to [-1,1], converted to angles via
`arccos`, and `GASF[i,j] = cos(phi_i + phi_j)`. Each `(50,)` signal becomes
a `(50, 50)` image; stacking 3 channels gives a `(3, 50, 50)` input per
window (same channel count as RGB).

**Visual sanity check (go/no-go, run before any model code)**: compared
GASF images of a fall window (near impact) vs. a jogging window across all
3 channels. `gyr_m` showed a clear structural difference — the fall's image
has sharp, localized diamond/X-crossing patterns, while the jog's image is
much smoother and more spread out. `acc_m` and `vv` looked more similar
between the two in this single example. One clearly-distinguishing channel
was judged a legitimate go signal (consistent with the CWT experiment,
where not every channel needed to show a difference either). See
`sanity_check_gaf_mtf.png`.

**Architecture** (`model.py`): plain 2D CNN → global average pool → dense →
softmax — **deliberately not** CNN+LSTM (unlike `experiments/cwt_lstm/`'s
model). A GASF image has no time-frequency axis the way a scalogram does,
so there's no principled reason to treat its output as a sequence for a
recurrent stage; this is a standard image-classification structure instead.
Same 32/64/128 filter progression as the rest of the project for
comparability; kernel size 3 and 2×2 pooling are standard 2D-CNN defaults,
not KFall-tuned.

**Training** (`train.py`): class-weighted `CrossEntropyLoss` (inverse
frequency) — matches `experiments/cwt_lstm/`'s convention, chosen for
consistency since these two experiments are the direct point of comparison
(rather than `vit_prefallkd`'s 6x fall-oversampling convention). Reduced
scale (`stride=3`, `MAX_ADL_TRAIN_WINDOWS=150000`, same scale `cwt_lstm`
used for its first pass) — 303s total training (12.1s/epoch), well within
budget. Memory guard (mirroring `cwt_lstm`'s post-OOM fix) estimates size
before allocating and refuses if it would exceed 8GB.

**A memory-guard catch worth noting**: the initial evaluation attempt used
`stride=1` for the test set (matching most other experiments) and the guard
**correctly refused** — full-resolution test set needs ~11.9GB (GASF images
at `(3,50,50)` float32 are bigger per-window than CWT's `(9,24,50)`
float16 scalograms). Fixed by using `stride=5` (~2.4GB), the same
precedent `cwt_lstm/evaluate.py` established. No crash, no OOM — the guard
did exactly its job.

## Training result: validation accuracy notably lower than other experiments

Validation balanced accuracy topped out around **93%** (best epoch), never
approaching the ~99% seen in `cwt_lstm`, `vit_prefallkd`, or
`conv_transformer` at comparable training scale. This alone is a different
signature from those experiments — this looks more like a genuinely
harder-to-fit representation for this model, not a case of "fits training
data perfectly but doesn't generalize."

## Test result (at face value): catastrophic

| | Sensitivity | Specificity |
|---|---|---|
| GAF 2D-CNN (150K scale, corrected test set) | 80.64% | **9.20%** |
| cwt_lstm (matched 150K scale, corrected test set) | 83.83% | 73.75% |
| ConvLSTM baseline (corrected test set) | 94.53% | 93.68% |

(Numbers above reflect the window-labeling bug fix in
`paper_implementation/data.py` — see that project's README — which corrected
every experiment's test set to the full 439 fall / 522 ADL trials.)

9.20% specificity means the model flags nearly every ADL trial as a fall —
474 of 522. Taken alone, this looks like a broken/degenerate model.

## The real story: a diagnosable aggregation-amplification effect, not a broken model

Investigated further before accepting the headline number, since it was a
large outlier versus every other experiment's result magnitude:

**Window-level, the classifier is actually reasonably good.** On a held-out
subset: 94.0% window-level accuracy, and the mean predicted P(fall) for ADL
windows is only 0.08 (median 0.0025) — most ADL windows are classified
confidently and correctly. The problem is not that the model is guessing
randomly.

**Sweeping the persistence threshold reveals the real issue:**

(Sweep below predates the window-labeling fix; where it disagrees with the
headline 80.64%/9.20% number above, the headline number is correct.)

| CONSEC_WINDOWS | Sensitivity | Specificity |
|---|---|---|
| 1 (any single window) | 99.27% | **1.15%** |
| 2 (project default) | 86.34% | 9.20% |
| 5 | 41.95% | 64.94% |
| 8 | 9.27% | 91.00% |
| 15 | 0.00% | 100.00% |

Even `consec=1` gives only 1.15% specificity — virtually every ADL trial
has at least one false-positive window, consistent with a ~5-6%
per-window false-positive rate spread across 100+ windows per trial
(`1 - 0.94^100 ≈ 99.8%` chance of at least one hit). But more tellingly,
**sensitivity collapses just as fast** as the threshold rises (99%→9% by
consec=8) — meaning real fall trials *also* don't get sustained runs of
correct positive predictions. Both classes show unstable, flickering
window-to-window predictions, not the smooth, sustained blocks of
same-class predictions every other experiment's model produces (where
`consec=2` barely dents sensitivity at all).

**Likely mechanism, specific to GAF among all experiments in this project**:
the GASF formula's normalization is inherently *per-window* — each window
is rescaled to [-1,1] using **its own** min and max, not a global
training-set statistic. At `stride=5`, two consecutive test windows share
90% of their frames, but a 5-frame shift can still change which frame is
the local min or max within that window. When that happens, the entire
image's geometry shifts discontinuously (every pairwise angle sum changes),
which can flip the model's classification even though the underlying
signal barely moved. Every other model in this project (ConvLSTM,
`cwt_lstm`, `vit_prefallkd`, `conv_transformer`) normalizes using fixed
global statistics fit once on the training set, so their per-window inputs
change smoothly and continuously as the window slides — GAF's inputs don't.

This is a genuinely different failure mode from `cwt_lstm`'s result (a
clean generalization-gap/overfitting signature: ~99% validation accuracy,
clear train/test divergence). Here, validation accuracy itself was
mediocre (~93%), and the test-time collapse is explainable by a specific,
identifiable property of the encoding method interacting badly with this
project's window-overlap-based evaluation convention — not simply "more
overfitting."

## Conclusion

**Per the plan's success criterion**: GAF specificity falls far short of
the ConvLSTM baseline (-84.5pp at face value, or -12.6pp using the
per-window-accurate `consec=1` reading if that's judged the fairer
comparison for this specific encoding). Either reading supports the same
conclusion as the CWT precedent: **the specificity plateau is not explained
by "signal shape isn't explicit enough."** Two different 2D representations
(frequency-domain and pairwise-relationship-domain) have now both failed to
beat the 1D raw-signal ConvLSTM baseline, for two different underlying
reasons — pointing more consistently toward the LSTM's role, the
aggregation rule, or training-recipe details (per `experiments/
conv_transformer/README.md`'s broader analysis) as the more likely
explanations for that plateau, rather than the input representation itself.

**A candidate fix not pursued in this pass**: since the diagnosed mechanism
is specific to the persistence-aggregation rule interacting badly with
per-window-normalized inputs, an aggregation rule less sensitive to
frame-to-frame instability (e.g. majority vote over a wider window, or
smoothing predicted probabilities before thresholding, rather than a strict
consecutive-run requirement) might recover much of the gap without
touching the model itself — worth testing before concluding GAF encoding is
a dead end outright.

## Files

```
experiments/gaf_mtf/
├── data.py                       # windowing (reused) + GASF encoding + memory guard
├── model.py                       # plain 2D CNN, no LSTM
├── train.py                        # 150K matched-scale training
├── evaluate.py                      # test-set evaluation + comparison
├── sanity_check_gaf_mtf.png         # go/no-go visual check (fall vs. jog)
└── results/
    ├── gaf_mtf.csv                    # per-trial predictions
    ├── gaf_mtf_best.pt                # best checkpoint
    ├── norm_stats.npz                 # per-channel normalization stats
    └── comparison_vs_baseline.md      # generated comparison table
```

## Run commands

```bash
# 1. Visual sanity check (go/no-go, run before anything else)
/mnt/d/KFall/venv/bin/python3 experiments/gaf_mtf/data.py --sanity-check

# 2. Train (150K matched scale)
/mnt/d/KFall/venv/bin/python3 experiments/gaf_mtf/train.py

# 3. Evaluate + compare against ConvLSTM and cwt_lstm baselines
/mnt/d/KFall/venv/bin/python3 experiments/gaf_mtf/evaluate.py
```
