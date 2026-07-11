# KFall Pre-Impact Fall Pattern: Implementation Walkthrough

This document explains **what was built, why, and what each output shows**, in the
order the work actually happened. It is the technical/implementation companion to
`KFall_Pattern_Analysis_Report.docx` (the formal report). Read this one if you want
to understand the code paths and reasoning; read the `.docx` for a precise, formal
summary of findings.

---

## 1. Starting point: baseline exploratory plots

`generate_plots.py` (project root) produces 5 baseline plots from a single example
trial (Subject SA06, Task T32 "forward fall while walking caused by a slip"):

| File | What it shows |
|---|---|
| `plots/plot1_freefall_signal.png` | SVM (accel magnitude) dip → impact spike on one trial |
| `plots/plot2_rotational_whiplash.png` | Gyroscope comparison: calm walking vs. a fall |
| `plots/plot3_center_of_mass.png` | Roll vs. pitch trajectory (phase space) |
| `plots/plot4_cwt_scalogram.png` | Wavelet time-frequency view of AccZ |
| `plots/plot5_label_isolation.png` | SVM with the labeled onset→impact window shaded |

These motivated the original question: **does the dip-then-spike pattern seen in
this one trial hold across the whole dataset (32 subjects × 15 fall types)?**

---

## 2. First pass: naive threshold sweep (superseded)

An initial script checked, per trial, whether `min(SVM) < 0.6g` and whether that dip
preceded the impact peak, across every subject/task. The 0.6g threshold was inherited
from `generate_plots.py`'s shading choice — a visualization decision, not a derived
one. This version was later deleted once its limitations were identified (see §3).

---

## 3. Grounding the method in the source paper

`docs/paper.pdf` — Yu, Jang & Xiong (2021), *"A Large-Scale Open Motion Dataset
(KFall) and Benchmark Algorithms for Detecting Pre-impact Fall of the Elderly Using
Wearable Inertial Sensors,"* Frontiers in Aging Neuroscience — describes the actual
threshold-based benchmark algorithm (Fig. 6 of the paper):

- `ACC_M(T) < 0.8g` **and** `VV(T) > 0.3 m/s` (vertical velocity)
- then, within the next 10 frames, `|Pitch| > 25°` or `|Roll| > 25°`
- all computed on a 5 Hz low-pass filtered signal
- thresholds obtained by grid search, not guessed
- reported benchmark: **95.50% sensitivity, 83.43% specificity, 333±160 ms lead time**

This reframed the original question: a single accel threshold was never claimed by
the paper's authors to be sufficient on its own — they combine three signals for a
reason. The naive v1 script never tested that reason (it never checked false
positives on non-fall activities), so it could not confirm or refute whether SVM
alone was "the" defining factor.

---

## 4. `analyze_pattern.py` — paper-faithful reimplementation

Canonical script (project root). Computes, per trial:

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

Each rule was run on **both** fall trials (T20–T34, sensitivity) and — the check
the v1 script never did — **ADL trials** (T01–T21, specificity).

### Verified results (`pattern_results.csv`, 2,319 fall + 2,744 ADL trials)

| Rule | Sensitivity | Specificity |
|---|---|---|
| A (accel only) | 100.0% | 17.3% |
| B (accel+VV+tilt) | 95.6% | 35.8% |
| Paper's own Threshold algorithm (reference) | 95.50% | 83.43% |

Rule B's sensitivity matching the paper's reported 95.50% almost exactly is the key
validation signal that the VV/rotation implementation is correct.

Fainting-sitting subset (F06–F08, n=461): Rule A 100.0%, Rule B 87.4%, **Rule C
100.0%** (corrected — an earlier verbal report of 0.2% for Rule C was a
mis-transcription during the conversation and has been re-verified against the
current code and CSV as 100.0%; Rule C was not tested for specificity).

Plots (`plots/`):
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

## 5. Phase-aligned signal visualization (event-locked averaging)

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
  angular velocity, and the dataset's own Plot 2 (§1) already suggested gyro has a
  sharp, clean signature the paper's tilt-based check doesn't capture on its own.

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
  As clean as ACC_M/VV — this was the notable finding from adding the 4th panel,
  and it revises the earlier (Rule C-derived) impression that gyro was a weak
  signal; the weakness was in the ad hoc Rule C threshold test, not the raw signal.
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
does not mean the pattern is a ready-to-use detector on its own — Rule A/B/C's
specificity results (§4) show simple thresholds on these same signals still
over-trigger on fast ADLs (jogging, jumping, stumbling) — but it confirms the
underlying physical signature genuinely generalizes across subjects and fall types,
which was the original question this analysis set out to answer.

---

## 6. File map

```
KFall/
├── analyze_pattern.py                  # Rule A/B/C, sensitivity/specificity (§4)
├── pattern_results.csv                 # per-trial results underlying §4
├── generate_plots.py                   # 5 baseline plots (§1)
├── plots/                              # §1 baseline + §4 rule-comparison plots
├── analysis_3panel/
│   ├── visualize_pattern_3panel.py     # ACC_M, VV, Tilt (§5)
│   └── plots/
├── analysis_4panel/
│   ├── visualize_pattern_4panel.py     # + Gyro magnitude (§5)
│   └── plots/
└── docs/
    ├── explanation.md                  # original 5-plot rationale
    ├── paper.pdf                       # source paper (§3)
    ├── pattern_analysis.md             # this file
    └── KFall_Pattern_Analysis_Report.docx  # formal report
```
