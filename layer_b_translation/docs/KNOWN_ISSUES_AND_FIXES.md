# Layer B — Known Issues, Fixes Applied, and Open Problems

This is the authoritative record of what was found wrong with Layer B's Phase
1 implementation, what was fixed, how each fix was validated, and what is
still genuinely open. It supersedes the narrative in `FIXES_TRACKER.md` (kept
as the working session log) as the polished reference. Read alongside
`LayerB_Implementation_Plan.md` (the original spec) and `Fix_Plan_Steps_0-6.md`
(the approved fix plan this document reports against).

## Methodology used throughout

Every fix below follows one discipline: **calibrate constants on train
subjects only, evaluate accuracy/correlation on held-out test subjects only,
exactly once.** This reuses the existing seeded 80/20 subject-level split
already used elsewhere in this repo (`paper_implementation/data.py::make_subject_split`,
seed 42 → 26 train subjects / 6 test subjects, see `layer_b_translation/split_utils.py`).
The original Phase 1 implementation did not do this — every threshold was fit
and graded on the same full 2,319-trial dataset, which is why its reported
accuracy numbers looked better than they actually generalized.

32 subjects total (26 train / 6 test) is a small base for subject-level
splitting. Every held-out number below should be read with that in mind.

---

## Problem-by-problem record

### 1.1 Direction Classifier

**Original defect:** ACC_M bump-vs-dip heuristic scored only 35.2% accuracy,
and on inspection was predicting "forward" for nearly every trial — ACC_M is
an unsigned magnitude and structurally can't encode "which way."

**Fix:** Replaced with a signed rule using pitch/roll deviation at the instant
of peak rotational rate (max GYR_M in the fall window). Calibration run on
train subjects (`tests/calibrate_direction.py`) revealed the sign convention
is *opposite* to the naive anatomical guess for this device's Euler output:
**roll** (not pitch) separates forward vs. backward strongly; pitch is what
separates lateral falls. This is flagged in `direction_classifier.py` as
likely a sensor-mounting/axis-convention artifact, not a universal rule —
re-run the calibration check before reusing this on different hardware.
`ROLL_DOMINANCE_RATIO=1.9` frozen on train data before touching test subjects.

**Result:** 88.0% train accuracy → **90.7% held-out test accuracy** (398/439).
Gate (≥80%) cleared comfortably; the pre-planned ML fallback (logistic
regression/linear SVM, per explicit instruction if the rule-based approach
failed) was not needed.

### 1.2 CoM Estimation

**Status:** Unchanged, and correctly so. This is a single-segment
approximation — the lower-back IMU's integrated vertical velocity standing in
for true multi-segment CoM — because a single IMU genuinely cannot provide a
real multi-segment biomechanical CoM model. That needs more IMUs or vision.
This is a hardware constraint, not a bug, and the fix pass deliberately did
not fake a fix here.

### 1.3 Time-to-Impact Estimate

**Original defect:** A single global constant (`fall_height_m = 2.61`),
reverse-solved from this dataset's own mean lead time, matched the *mean*
(710ms vs. 729ms empirical) but had only r=0.32 per-trial correlation on the
same full dataset it was fit on — it explained almost none of the real
per-trial variance, and the leakage made even that number unreliable.

**Fix:** Replaced with a causal 3-parameter linear regression:
`t_impact_ms = A + B·v0 + C·gyr_early`, where `v0` is CoM velocity at onset
(available instantly) and `gyr_early` is max GYR_M over the first 150ms past
onset. A version using GYR_M peak over the *whole* onset→impact window scored
even higher (held-out r=0.473) but was rejected — it peeks at the future and
could never be computed by a real-time controller. The causal 150ms-delayed
version actually scored *better* (r=0.534) once accepted. Fit on train
subjects only.

**Downstream consequence, preserved in code:** because `gyr_early` needs
150ms of post-onset data, the estimator itself isn't available until
onset+150ms. `usable_reaction_time_ms()` accounts for this — the torque
profile (1.6) is sized to the real remaining window, not the full predicted
duration from onset.

**Result:** held-out r=0.534, MAE=101.8ms, vs. a constant baseline's r≈0.
Gate (≥0.1 improvement) cleared.

### 1.4 Capture Point / XCoM / Base of Support

**Original defect:** deficit was only checked at the single onset frame, so
89% of trials showed exactly zero deficit (the stubbed base-of-support
boundary hadn't been breached yet at that single frame) — torque/strategy
defaulted to "no correction needed" for almost every real fall.

**Fix (part 1 — already applied before this fix pass):** deficit now
evaluated across the whole onset→impact window, taking the peak.

**Fix (part 2 — this pass):** base-of-support half-width is now
direction-conditioned (0.20m for forward falls, assuming a bracing/stepping
response is available; 0.12m for backward/lateral, assuming closer-to-pure
freefall) instead of one fixed 0.15m symmetric constant. **Important
honesty note:** this is NOT spatial asymmetry — this pipeline's CoM proxy is
derived from *vertical* velocity, not a true horizontal coordinate, so there
is no valid spatial "forward/backward" sign to apply to it. The values are
illustrative order-of-magnitude placeholders, not fit to this dataset's
deficit distribution, and not sourced from a specific citation.

**Fix (part 3 — causal re-evaluation, this pass):** added
`find_first_crossing_frame`, which walks the (already per-frame-causal)
deficit series and reports the *first* frame a real-time controller could
have detected a breach — vs. the post-hoc `.max()` over the whole window,
which implicitly assumes knowledge of the future. This produces a genuine
"lead time before impact" metric: median ~530-540ms across the dataset,
always ≤ the true onset→impact duration (sanity-checked).

### 1.5 Torque Magnitude

**Original defect:** `mass * deficit * gain`, where `gain=0.4` was picked
*after* seeing it saturate the 40Nm ceiling on 86% of trials with `gain=1.0`
— i.e. tuned against the exact thing it was later checked against.

**Fix attempt 1 (rejected before touching the dataset):** an impulse/latency
formula `mass*deficit/time_to_impact/actuator_latency` was tried, sanity-
checked with a representative toy deficit (0.05m, not this dataset's
statistics) — and gave physically implausible torques (hundreds of Nm) at
that toy scale already. Rejected on dimensional/physical grounds before ever
running it against real data: dividing by two independent time factors
double-penalizes a short window in a way real F=ma kinematics doesn't.

**Fix attempt 2 (adopted):** a kinematic formula matching the same template
already used by `time_to_impact.py`'s own free-fall model (`h = 0.5·a·t²`):
`F = 2·mass·deficit / effective_T²`, `torque = F · lever_arm`, where
`effective_T` subtracts a frozen `ACTUATOR_LATENCY_S = 0.10s` from the
available reaction window. This constant was picked once from a generic
motor-response-time ballpark and sanity-checked only against representative
toy deficits (0.1m/0.2m/0.5m) — never adjusted after seeing the real dataset's
saturation rate.

**Result — a genuine finding, not a defect to fix further:** given this
dataset's deficits (mean ~1.1-1.15m) and the frozen constants, **100% of
trials** (train and test) require more torque than the 40Nm ceiling allows.
One extreme outlier (max ~1.1 million Nm on train) traces to a single trial
with a near-zero reaction time, where the `1/effective_T²` term blows up — an
expected (if extreme) consequence of the formula for a genuinely
near-zero-reaction-time fall, not a computation bug.

### 1.6 Torque Profile Shape

No formula change. Now sized to `reaction_time_ms` (the real usable window
after the TTI estimator's own 150ms decision delay) instead of the raw
predicted `time_to_impact_ms`, consistent with 1.3's fix. Still a trapezoidal
ramp, still clamped to the 40Nm ceiling.

### 1.7 Recovery Strategy Selection

**Original defect:** the "always step" threshold (`xcom_deficit_high_m=1.54`)
was set to this dataset's own 90th percentile of deficit values — an
independently-fit, data-leaked number, not grounded in anything physical.

**Fix:** the step-vs-torque decision is now derived directly from whether the
*unclamped* required torque (1.5) exceeds the safety ceiling — i.e., step
when torque-only correction genuinely cannot deliver enough force. This ties
1.5 and 1.7 to the same physical constraint instead of two independently
calibrated numbers. The config file's percentile threshold was removed
entirely (direction-based lookup for forward/backward/lateral remains).

**Result:** because 1.5 now reports 100% of trials exceeding the ceiling,
1.7 now resolves 100% of trials to `"stepping"` — the direction-based
hip/ankle-torque branch never fires on this dataset. See "New problems
surfaced," below.

### 2.1 Stance-Leg Classifier (Phase 2)

Unchanged this pass. Still a rule-based gait-phase heuristic using
mediolateral roll sway, still returns `"unknown"` outside walking contexts.
**Cannot be scored for accuracy** — KFall has no ground-truth stance-leg
label. Left/right assignment uses an arbitrary, unvalidated sign convention,
clearly flagged in the module docstring.

### 2.2 Cause/Context Classifier (Phase 2, technically Layer A)

Unchanged this pass. 88.7%/90%+ accuracy range against task-ID-derived
ground truth (see `results/2.2_cause_context_classifier.txt` for the
train/test breakdown), not touched by this fix pass.

---

## New problems surfaced by these fixes

Fixing the original 7 problems did not produce a finished system — it
revealed deeper, more fundamental issues that were previously hidden by the
leaky calibration:

1. **1.7's strategy output is now degenerate in the opposite direction.**
   Before: 100% one thing because the threshold was mis-scaled. Now: 100%
   `"stepping"` because the (correctly derived) torque formula says
   torque-only correction is essentially never sufficient for this dataset's
   deficits. The direction-based hip/ankle branch of 1.7 — its actual
   purpose — never exercises on any trial.

2. **This points to 1.2 (CoM estimation) as the real bottleneck, not a
   fixable Phase 1 formula bug.** The single-segment vertical-velocity proxy
   produces deficits (~0.8-1.2m) that are plausible as *numbers* but
   unvalidated as *true horizontal CoM displacement*. Every downstream
   number (torque, strategy, the "100% stepping" finding) inherits this
   uncertainty. The fix pass made this visible for the first time; it does
   not resolve it, and cannot without additional sensing (multi-IMU/vision).

3. **The direction-aware BoS margins (0.20m/0.12m, 1.4) are still just
   plausible numbers, not measured** — explicitly flagged as such in code,
   since this model's CoM proxy can't support genuine spatial asymmetry.

## Still-open items, unchanged by this fix pass

4. Safety ceiling (40Nm) and actuator latency (0.10s) remain fabricated
   placeholders pending real physician/PT input and actual actuator hardware
   selection — now load-bearing inside a formula instead of an ad hoc gain,
   same external-dependency status either way.
5. Hip/knee torque split (70/30) — untouched, still a placeholder assumption.
6. Direction classifier's 90.7% held-out accuracy rests on only 439 test
   trials from 6 subjects — a different random subject split could plausibly
   move this by several points; treat as directionally solid, not precise.
7. Phase 2's stance-leg classifier (2.1) remains structurally unvalidatable
   with this dataset — no ground truth exists in KFall to check it against.

## What this means for next steps

The strategy component (1.7) cannot be meaningfully improved further without
either (a) accepting "always stepping" as the honest answer given current
single-IMU hardware, or (b) revisiting whether the CoM/XCoM formulation
itself — not just its calibration — is the right approach for a single
lower-back IMU. That is a sensing/design decision, not a code-level fix, and
is the natural next planning conversation once Layer C (patient calibration)
or Layer D (safety) work begins.
