# KFall Pre-Impact Fall Pattern: Implementation Walkthrough

This document explains **what was built, why, and what each output shows**, across
the full project history. It is the technical/implementation companion to
`KFall_Pattern_Analysis_Report.docx` (the formal report, which covers Sections 1–3
below only). Read this one for the complete story and code paths; read the `.docx`
for a precise, formal summary of the core dataset-validation findings.

**Migration note**: every script in this project resolves its paths relative to its
own file location (not the current working directory) and reuses
`paper_threshold_validation/analyze_pattern.py` as a shared pipeline — nothing is
hardcoded to this machine. See `../../README.md` for setup and run commands.

---

## 1. Grounding the method in the source paper

`docs/paper.pdf` — Yu, Jang & Xiong (2021), *"A Large-Scale Open Motion Dataset
(KFall) and Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using
Wearable Inertial Sensors,"* Frontiers in Aging Neuroscience — describes the
originating dataset's own threshold-based benchmark algorithm (Fig. 6 of the paper):

- `ACC_M(T) < 0.8g` **and** `VV(T) > 0.3 m/s` (vertical velocity)
- then, within the next 10 frames, `|Pitch| > 25°` or `|Roll| > 25°`
- all computed on a 5 Hz low-pass filtered signal
- thresholds obtained by grid search, not guessed
- reported benchmark: **95.50% sensitivity, 83.43% specificity, 333±160 ms lead time**

The project's original question was whether acceleration magnitude alone is enough
to detect a fall. The paper's own design — combining three signals, not one — was
the first evidence that it isn't; the sections below test that directly.

---

## 2. `paper_threshold_validation/analyze_pattern.py` — paper-faithful reimplementation

Computes, per trial:

- **ACC_M** — `sqrt(AccX²+AccY²+AccZ²)`, 5 Hz low-pass filtered (`lowpass_filter`).
- **VV (vertical velocity)** — body-frame acceleration is rotated into the world
  frame using the fused Euler-angle orientation (`scipy.spatial.transform.Rotation`,
  intrinsic Z-Y-X convention), the vertical component is isolated, and integrated.
  Two bugs were found and fixed during development, both caught by a sanity check
  (VV should be ≈0 during a 30-second quiet-standing trial):
  1. **Drift** — naive full-trial cumulative integration accumulated tiny sensor
     bias into multiple m/s of spurious velocity over a long recording. Fixed with
     a **1-second windowed integral** (`VV(t) = cumulative(t) − cumulative(t−1s)`),
     which bounds drift regardless of trial length.
  2. **Sign convention** — the up-positive integral went *negative* during an
     actual fall's descent (checked directly against SA06/T32). Flipped the sign so
     `VV > 0.3` means "falling downward," matching the paper's threshold direction.
- **Tilt** — `max(|Pitch|, |Roll|)` from the Euler angles.

Three detection rules, applied identically to every trial:

| Rule | Definition |
|---|---|
| **A** | `ACC_M < 0.8g` at any point in the trial (accel-only) |
| **B** | Paper's full algorithm: `ACC_M<0.8g AND VV>0.3` then tilt confirmation within 10 frames |
| **C** | Gyro-based: `max(\|GyrX\|,\|GyrZ\|) > max(3×standing-baseline, 30°/s)` — fainting-sitting subset only (F06–F08 / tasks 25–27), the one subset the paper explicitly flags as weak on acceleration |

Each rule was run on **both** fall trials (T20–T34, sensitivity) and ADL trials
(T01–T21, specificity).

### Verified results (`pattern_results.csv`, 2,319 fall + 2,717 ADL trials)

| Rule | Sensitivity | Specificity |
|---|---|---|
| A (accel only) | 100.0% | 17.4% |
| B (accel+VV+tilt) | 95.6% | 34.9% |
| Paper's own Threshold algorithm (reference) | 95.50% | 83.43% |

Rule B's sensitivity matching the paper's reported 95.50% almost exactly is the key
validation signal that the VV/rotation implementation is correct.

Fainting-sitting subset (F06–F08, n=461): Rule A 100.0%, Rule B 87.4%, **Rule C
100.0%** (Rule C was not tested for specificity).

Plots (`paper_threshold_validation/plots/`):
- `sensitivity_specificity_comparison.png` — Rule A vs. B, sensitivity/specificity bars
- `heatmap_ruleB.png` — subject × task grid of Rule B detection rate
- `adl_false_positive_by_task.png` — per-ADL-task false-positive rate, A vs. B
- `fainting_subset_axis_comparison.png` — A vs. B vs. C on the F06–F08 subset

**Interpretation**: accel magnitude alone (Rule A) has no real discriminating power
— it fires on both falls and ordinary activity almost indiscriminately. Adding
vertical velocity and tilt (Rule B) roughly doubles specificity for a negligible
sensitivity cost, confirming the paper's own design choice was necessary, not
arbitrary. The remaining sensitivity/specificity gap to the paper's own numbers is
attributed to evaluation-protocol differences (this analysis checks "fired anywhere
in the whole file"; a real detector would use a streaming, debounced evaluation).

---

## 3. Phase-aligned signal visualization (event-locked averaging)

Threshold-crossing rates say *whether* a rule fired, not what the signal actually
looks like. To visualize shape directly, every fall trial is time-normalized onto a
common **phase axis**: 0% = labeled onset frame, 100% = labeled impact frame,
extended to −50%…+150% for context. This is the standard event-locked averaging
technique from signal processing (the same idea as ERP averaging), and it lets
trials of very different absolute duration (a ~70-frame sitting-fainting fall vs. a
~500-frame walking-with-hands fall) be overlaid and averaged meaningfully.

Implemented as two parallel, self-contained subfolders:

- **`analysis_3panel/visualize_pattern_3panel.py`** — ACC_M, VV, Tilt.
- **`analysis_4panel/visualize_pattern_4panel.py`** — same, plus **Gyro
  magnitude** (`sqrt(GyrX²+GyrY²+GyrZ²)`, 5 Hz low-pass filtered), added because
  tilt (a fused/integrated orientation estimate) is not the same signal as raw
  angular velocity, and an earlier baseline exploration had already suggested gyro
  has a sharp, clean signature the tilt-based check doesn't capture on its own.

Each produces 4 plots in its own `plots/` subfolder:

1. **`pattern_grand_average.png`** — mean ± 1 std band per signal, across all 2,319
   fall trials, on the phase axis. Answers "is there one consistent global shape."
2. **`pattern_intersubject_overlay.png`** — one line per subject (32 lines) plus
   the global mean — inter-subject consistency.
3. **`pattern_intrasubject_small_multiples.png`** — a panel per subject, each
   subject's individual trials (thin) plus their mean (bold), ACC_M only — intra-
   subject consistency. (Kept ACC_M-only in both the 3- and 4-panel versions: a
   32-panel grid repeated per signal would clutter rather than clarify.)
4. **`pattern_trial_heatmap.png`** — every trial as one row, phase on the x-axis,
   color = ACC_M — the single-trial data underlying the averages, checked to
   confirm the average isn't hiding inconsistency. (Also kept ACC_M-only, same
   reasoning.)

### Findings

- **ACC_M**: near-flat ~1g baseline, dips to ~0.6g mean, sharp spike to ~3.0g
  precisely at phase 100 (the labeled impact frame). Clean, sharp, well-timed.
- **VV**: rises smoothly from ~0, crosses the 0.3 m/s threshold around phase
  15–20%, peaks ~2.2 m/s just before impact. Clean, sharp, well-timed.
- **Gyro magnitude** (4-panel only): smooth, monotonic rise from a ~30–50°/s
  baseline starting near onset, sharp peak (~200°/s) right at impact, then drops.
  As clean as ACC_M/VV — this revised an earlier (Rule C-derived) impression that
  gyro was a weak signal; the weakness was in the ad hoc Rule C threshold test,
  not the raw signal.
- **Tilt**: the outlier. Its spread is large even before onset (±20–100° band at
  phase −50%, before any fall-related motion), and it shows no sharp localized
  feature — just a gentle post-impact upward drift. This is consistent with, and
  explains, Rule B's weaker performance on the two "slow, sinking" fall types
  (T20/F01 forward-fall-sitting-down, T25/F06 forward-sitting-fainting).
- Inter-subject and intra-subject views both show tight agreement for ACC_M, VV,
  and gyro magnitude across all 32 subjects and both within- and across-subject
  trial sets; tilt shows visibly wider spread in both views.

**Net conclusion**: three of the four raw signal channels on this single IMU —
accel magnitude, vertical velocity, and gyro magnitude — show a consistent,
sharply-timed, generalizable fall signature across the entire dataset, confirmed
down to the individual-trial level (not just as an averaging artifact). Tilt
(orientation angle) is comparatively weak and noisy as a standalone marker. This
is why tilt was **excluded** from the rolling-regression work (§5) — only
ACC_M/VV/GYR_M carried forward.

This does not mean the pattern is a ready-to-use detector on its own — §2's
specificity results show simple thresholds on these same signals still
over-trigger on fast ADLs (jogging, jumping, stumbling) — but it confirms the
underlying physical signature genuinely generalizes across subjects and fall types.

---

## 4. `signal_quality/` — interactive Plotly dashboards

Two self-contained interactive HTML tools (Plotly embedded, no external
dependencies) for exploring the same phase-aligned signals interactively instead
of via static PNGs:

- **`build_signal_dashboard.py` → `signal_dashboard.html`** — the main tool.
  Per-sensor dropdown (Accelerometer / Gyroscope / Orientation) × 3 view modes:
  1. **Subject × Task** — any of the 15 fall or 21 ADL tasks for one subject;
     individual trials + mean, magnitude + raw-axis panels.
  2. **Subject, all tasks** — overlay every task's mean for one subject
     (Falls/ADLs group toggle).
  3. **Task, all subjects** — overlay all 32 subjects' means for one task (fall
     or ADL).

  ADL tasks have no fall onset/impact labels, so their phase axis means something
  different (0%=trial start, 100%=trial end vs. falls' 0%=onset, 100%=impact) —
  the dashboard auto-switches the axis label and shows a warning note whenever an
  ADL task is selected, and overlay modes never mix the two families on one plot.

- **`build_signal_quality.py` → `signal_quality.html`** — simpler grand-average-only
  view (mean ± 1 std across all fall trials), no selectors.

Both reuse the validated pipeline from `paper_threshold_validation/analyze_pattern.py`
exactly, so results stay consistent with §2 (Rule B still matches the paper's 95.50%).

---

## 5. `rolling_regression/` — causal Savitzky–Golay β signals

### The idea

Instead of thresholding the raw signal value, watch its **local shape**: at every
frame, fit a small sliding window of *past-only* samples with a low-order
polynomial. This is formally a **causal Savitzky–Golay derivative filter**:

- **β1 (slope)** — linear fit's slope. Is the signal rising or falling, how fast?
- **β2 (curvature)** — quadratic fit's leading coefficient. How sharply is it bending?

Both are stamped at the window's *last* sample (one-sided/causal, not the textbook
centered/symmetric SG convention) so the method is real-time deployable — it only
ever uses data a device would already have. Runs on the three **validated strong
channels only**: ACC_M, VV, GYR_M (tilt excluded — see §3's finding).

Implementation detail: since the window's x-values are always `0..W-1`, the
regression reduces to one fixed pseudo-inverse matrix, computed once per window
size and reused via `sliding_window_view` + matmul — avoiding millions of
individual `polyfit` calls (there are ~1.65 million window-fits per β-signal
across all fall trials alone).

### Files

- **`visualize_beta_signals.py`** → 3 static PNGs (`plots/beta_ACC_M.png`, etc.),
  one window size (25 frames), falls vs. 4 "risky" ADLs (stumble/sit-down/quick-sit/
  jump) plus an all-ADL mean, on the same phase axis as §3.
- **`build_beta_dashboard.py`** → **`beta_dashboard.html`** — the interactive,
  much more capable version. Global controls: Channel (ACC_M/VV/GYR_M) × Derivative
  (β1/β2) × **SG window (15/25/35 frames)**, so the smoothing/reaction-speed
  trade-off is explorable live. Three modes:
  1. **Trial explorer** — one subject+task, raw signal + β1/β2, individual trials + mean.
  2. **Per-task means** — every fall/ADL task as its own line (deliberately *not*
     blended into one average — see "Known limitations" below).
  3. **Separability** — box-plot distributions of β at a chosen pre-impact lead
     time (0–300 ms) for falls vs. risky ADLs — the view that most directly
     predicts detector performance (distribution overlap ≈ expected false-alarm rate).

### Findings

- **VV β1 separates earliest** — its slope rises steadily and positively from
  onset onward, distinguishable from flat ADLs well before impact (~phase 25–50%).
  **This is the best pre-impact discriminator found so far.**
- **ACC_M β spikes largest but latest** — mostly right at/after impact (~phase
  95–110%), too late for much pre-impact lead.
- **GYR_M β1 separates in the mean but is noisier** (wider spread per trial).
- The β signals do separate falls from risky ADLs on all three channels, but the
  spread (±1 std) is wide enough that a real detector faces a genuine
  sensitivity/specificity trade-off — not a clean guillotine cut.

### Known limitations (not yet fixed)

- **Mode 3 (separability)** currently blends **all 15 fall types into one "All
  falls" box** — the same blending problem Mode 2 was built to avoid, but it
  crept back in here. It can't yet show whether specific fall types (e.g. a slow
  sitting-fall) are harder to separate from specific ADLs (e.g. a sit-down) than
  others — arguably the most useful question for detector design. Only 4 ADL
  types are broken out individually (the rest are folded into "All ADLs").
  **Planned fix** (not yet implemented): replace the hardcoded groups with
  fall-task and ADL-task selectors so any specific pair can be compared.

Full detail: `rolling_regression/README.md`.

---

## 6. `ensemble_trigger/` — voting-ensemble detector

**Status: implemented and evaluated.** A **voting ensemble** combining three
independent per-frame binary detectors (fire / don't fire, causal, using only past
frames), pooled by majority vote:

| Detector | Signal | Fires when |
|---|---|---|
| Threshold | ACC_M (raw) | crosses a fixed cutoff at the current frame |
| CUSUM | VV's β1 (from §5) | accumulated departure from baseline passes a limit |
| Shapelet | ACC_M (raw) | recent window matches a learned fall shape (z-normalized distance) |

CUSUM runs on **VV's β1** (§5's earliest pre-impact discriminator) rather than a
raw signal, since the slope flattens routine gait oscillation and keeps the CUSUM
baseline steadier — confirming the methodology doc's expectation.

### Methodology
A proper **by-subject train/test split** (26/6, matching the source paper's own
ratio) was used — the first place in this project where that's been necessary,
since Threshold's cutoff, CUSUM's `k`/`h`, and the Shapelet's match cutoff are all
tuned on this project's own data (unlike `paper_threshold_validation/`, which
reused the paper's already-published thresholds, so no leakage risk existed
there). All tuning used train subjects only; all numbers below are test-subject-only.

### Results (test set, never used for tuning)

| Method | Sensitivity | Specificity | Lead time |
|---|---|---|---|
| Threshold alone | 95.0% | 65.3% | 419 ms |
| CUSUM alone | 91.8% | 57.8% | 250 ms |
| Shapelet alone | 51.9% | 63.7% | 1012 ms |
| ANY (≥1) | 99.8% | 32.1% | 793 ms |
| **MAJORITY (≥2, default)** | **94.5%** | **62.2%** | **302 ms** |
| UNANIMOUS (=3) | 44.4% | 92.5% | 172 ms |
| *Paper's Threshold (reference)* | 95.50% | 83.43% | 333 ms |
| *Paper's SVM (reference)* | 99.77% | 94.87% | 385 ms |
| *Paper's ConvLSTM (reference)* | 99.32% | 99.01% | 403 ms |

### The key finding: partial, not full, complementarity

The methodology doc's open question — do the three detectors make different
mistakes, or the same ones? — has a real, mixed answer, visible in the
per-activity false-alarm heatmap (`plots/false_alarm_breakdown.png`):

- **Threshold and CUSUM are strongly correlated**: both false-alarm at ~100% on
  jumping/jogging (D04, D08, D09) and stumbling (D10). Unsurprising — ACC_M and
  VV are physically related, so both react to genuinely fast, dynamic movement.
- **Shapelet fails on a different set of activities** — mostly static/lying ones
  (D01, D02, D05, D12, D17, D18) — genuine complementarity with the other two.
- One activity, D17 (lie down to bed and get up quickly), fools all three.

So voting helps *some* but not fully: MAJORITY's specificity (62.2%) doesn't
clear Threshold's alone (65.3%) by much, because two of the three members share a
real, physical source of correlated error rather than being independent — exactly
the failure mode the methodology doc warned about. **MAJORITY does not beat the
paper's own three benchmarks** here; it lands between the paper's Threshold and
its trained classifiers, while remaining fully explainable (every firing traces to
one of three simple, interpretable rules).

### Two real bugs found and fixed during this build

1. **Un-normalized shapelet distance** — plain sum-of-squared-differences is
   dominated by ACC_M's shared ~1g baseline, not the dip/bump shape itself; the
   shapelet never fired on anything, including a known clean fall. Fixed with
   z-normalization (standard practice in the shapelet literature, e.g. Ye & Keogh).
2. **Naive "fall trial" sampling for shapelet scoring** — most of a fall trial is
   calm activity before the event, so a uniform random sample of "fall" windows is
   mostly indistinguishable from ADL windows after normalization. Confirmed
   empirically: the *true* dip+spike shape scored *worse* against this naive
   sample than against ADL windows. Fixed by scoring each candidate via its
   minimum distance to any position within a whole sampled trial (the standard
   shapelet-discovery statistic) instead of a mean distance to pre-cut snippets.

Files: `ensemble_trigger/ensemble_trigger_plan.md` (methodology + build plan —
read this first), `detectors.py`, `build_ensemble.py`, `ensemble_results.csv`,
`plots/`. Full detail: `ensemble_trigger/README.md`.

---

## 7. Current file map

```
KFall/
├── README.md                                    # project overview, setup, findings summary
├── requirements.txt
├── .gitignore
├── sensor_data/, label_data/, *.zip              # raw dataset (gitignored)
└── fall_pattern_analysis/
    ├── paper_threshold_validation/               # §2 -- also the shared pipeline
    │   ├── analyze_pattern.py
    │   ├── pattern_results.csv
    │   ├── plots/
    │   └── README.md
    ├── analysis_3panel/                          # §3 -- ACC_M, VV, Tilt
    │   ├── visualize_pattern_3panel.py
    │   ├── plots/
    │   └── README.md
    ├── analysis_4panel/                          # §3 -- + Gyro magnitude
    │   ├── visualize_pattern_4panel.py
    │   ├── plots/
    │   └── README.md
    ├── signal_quality/                           # §4 -- interactive dashboards
    │   ├── build_signal_dashboard.py
    │   ├── build_signal_quality.py
    │   └── README.md
    ├── rolling_regression/                       # §5 -- causal SG beta1/beta2
    │   ├── visualize_beta_signals.py
    │   ├── build_beta_dashboard.py
    │   ├── plots/
    │   └── README.md
    ├── ensemble_trigger/                         # §6 -- voting ensemble (implemented)
    │   ├── ensemble_trigger_plan.md
    │   ├── detectors.py
    │   ├── build_ensemble.py
    │   ├── ensemble_results.csv
    │   ├── plots/
    │   └── README.md
    └── docs/
        ├── pattern_analysis.md                   # this file
        ├── KFall_Pattern_Analysis_Report.docx    # formal report (covers §1-3 only)
        ├── build_report.py                       # regenerates the .docx
        └── paper.pdf                             # source publication
```

Generated HTML dashboards (`signal_dashboard.html`, `signal_quality.html`,
`beta_dashboard.html`) and all `plots/` folders are gitignored — regenerate by
running the corresponding script (see each folder's README, or the root
`README.md`'s run-commands section).
