# Layer B Phase 1 — Proper Fix Plan (Problems 1-7)

## Context

The full-dataset validation run against all 2,319 KFall fall trials surfaced 7
problems in `layer_b_translation/`. On inspection, most of them trace back to
one root cause: **every threshold/constant was calibrated and evaluated on the
same full dataset** (the earlier "fix" pass improved discrimination but did so
by tuning against the very numbers used to report success — classic leakage).
The direction classifier is additionally just a weak rule, independent of the
leakage issue. This plan fixes both: the methodology (so results are honest)
and the specific weak components (so results are actually better).

Reviewed and sharpened via a Plan-agent critique (see prior turn) — key
correction from the original draft: time-to-impact should NOT get a per-fall-
category constant (14-15 categories over ~26 train subjects is a bad n-to-
parameter ratio); it gets a 2-parameter linear regression instead. Torque
"derived from impulse-momentum" must have its free parameter (actuator
latency) frozen *before* ever looking at how often it saturates the ceiling,
or the fix just hides the same arbitrariness one level deeper.

## Step 0 — Ground-truth direction mapping (shared dependency)

New small module (e.g. `layer_b_translation/phase1_formulas/fall_direction_labels.py`)
with a single TASK_ID → true direction dict (forward/backward/lateral) sourced
from `docs/Fall_Prevention_Actuator_Input_Pipeline.md` Table 2. Used by Step 2
and Step 6. Gate: covers every task ID present in `layer_b_full_dataset.csv`.

## Step 1 — Adopt subject-level train/test split (fixes the leakage root cause)

Reuse `paper_implementation/data.py::make_subject_split` (already exists,
seeded 80/20 split by subject — same one `paper_implementation/` uses, so no
new methodology invented). Thread `train_subjects`/`test_subjects` through
`layer_b_translation/pipeline.py` and `layer_b_translation/tests/run_full_dataset.py`.
From here on: **every constant gets calibrated on train subjects only; every
accuracy/correlation number gets reported on held-out test subjects only.**
Gate: split reproduces deterministically (seed=42); ~80/20 by subject, not by
trial count.

## Step 2 — Direction classifier redesign (1.1)

Replace the ACC_M bump/dip heuristic in `direction_classifier.py`. Root cause
of the current 35% accuracy: bump/dip throws away sign information — ACC_M is
an unsigned magnitude, so it can't actually encode "which way." Pitch/roll are
signed and directly encode trunk tilt direction, which is what a clinician
would look at too.

- First, empirically verify the sign convention: on train subjects, plot/check
  `pitch[impact]-pitch[onset]` grouped by Step 0's known true direction —
  confirm it's separable before hardcoding a sign rule (don't assume from
  documentation).
- New rule: take the sign of pitch/roll at a physically meaningful instant —
  the frame of peak `gyr_m` (max rotational rate) rather than a fragile
  endpoint-to-endpoint delta — combined with a magnitude gate on `gyr_m` that
  bails out to `"uncertain"` when rotation is too weak to trust (rather than
  forcing a guess).
- Do not add raw AccX/Y/Z (would mean extending the shared `Trial` class in
  `paper_implementation/data.py`) unless pitch/roll/gyr_m demonstrably fails —
  reserve as fallback only.
- **Pre-registered gate, decided now, not after seeing results**: held-out
  accuracy ≥ 80%.
- **If the rule-based redesign doesn't clear 80%**: per your direction, don't
  stop at "document the limitation" — escalate to a lightweight ML fallback
  before giving up. Concretely: a small classifier (logistic regression or
  linear SVM — same low-inference-cost pattern already used in
  `paper_implementation/svm_model.py`, not a DL model; a single-IMU direction
  call doesn't need one) trained on hand-crafted features (peak `gyr_m`,
  pitch/roll sign and magnitude at peak rotation, timing-to-peak) using the
  **same Step 1 subject split** — fit only on train subjects, scored only on
  held-out test subjects. This keeps the exact discipline the rest of this
  plan is built on (no fitting and grading on the same data); switching to ML
  doesn't get a pass on that.
- **What "keep working until gates pass" means here, concretely**: iterate
  model choice/features on train subjects (cross-validating within train, not
  touching test) as much as needed. The held-out test set gets evaluated once
  the model is finalized — not repeatedly peeked at and adjusted against,
  which would just reintroduce leakage through a side door. If even the ML
  tier can't clear the bar, that becomes an honestly-reported finding about
  what's recoverable from single-IMU signals alone, not a reason to loosen the
  held-out evaluation itself.

## Step 3 — Time-to-impact recalibration (1.3)

In `time_to_impact.py`, replace the single global `CALIBRATED_FALL_HEIGHT_M`
constant with a 2-parameter linear regression `t_impact ~ a + b * v0` (v0 =
CoM velocity at onset — the one feature already causally available at decision
time), fit on train subjects only. **Explicitly not** a per-fall-category
constant (rejected: 14-15 categories over ~26 train subjects overfits badly).

- **Pre-registered gate**: held-out correlation must beat the constant-only
  baseline's held-out correlation by ≥ 0.1, or held-out MAE must improve by a
  stated margin. Same escalation principle as Step 2: if the linear regression
  doesn't clear this, try a slightly richer but still low-cost regression
  (e.g. adding `gyr_m` peak or direction as a second/third feature) before
  settling — still fit on train subjects only, still scored once on held-out
  test. If nothing recoverable from available features beats the constant,
  report that honestly as the real limitation of a single-IMU kinematic model.

## Step 4 — Direction-aware base-of-support + causal re-evaluation (1.2, 1.4)

- **CoM (1.2) stays as-is** — single-segment approximation is a genuine
  hardware constraint (needs more IMUs/vision), not a bug. No fake fix; the
  write-up must not claim CoM itself was "improved."
- **Base-of-support (1.4)**: make the footprint direction-aware/asymmetric
  (using Step 2's classified direction) instead of one fixed symmetric 0.15m
  circle — e.g. wider margin in the direction of the fall (a stated bracing-
  distance assumption, still flagged as an unvalidated stub, not fit to make
  the deficit distribution "look good").
- **Causal first-crossing re-evaluation** (time-boxed, ~60-90 min): today's
  deficit is computed causally per-frame already (each frame's CoM/XCoM only
  depends on past VV via integration) — what's non-causal is only the
  `.max()`-over-the-whole-window aggregation, which peeks at the future. Add a
  streaming scan: walk frames onset→impact using only past+current data, find
  the first frame deficit crosses the trigger threshold, report that as a real
  "lead time before impact" metric. If not finished cleanly in the time box,
  ship without it and flag plainly as "post-hoc max only, not yet real-time-
  validated" rather than a half-finished streaming mode.

## Step 5 — Torque from impulse-momentum, frozen before look (1.5, 1.6, 1.7)

- `torque_magnitude.py`: replace `mass * deficit * gain` with a real
  impulse-momentum argument: `mass * xcom_deficit_m / time_to_impact_s /
  ACTUATOR_LATENCY_S`. `ACTUATOR_LATENCY_S` is a new explicit placeholder
  constant, same status as the 40Nm ceiling — picked once from a plausible
  ballpark **and frozen before ever computing what fraction of trials it
  saturates**. This ordering is the whole point: if the latency constant gets
  adjusted after looking at the saturation rate, the circularity that made
  `gain=0.4` arbitrary just moves one level deeper and starts looking falsely
  principled.
- `recovery_strategy.py` / `config/recovery_strategy_thresholds.yaml`: drop
  the independently-fit `xcom_deficit_high_m` percentile threshold entirely.
  1.7's "step" branch becomes: step when the *required, unclamped* torque from
  1.5 would exceed the safety ceiling (torque-only correction genuinely can't
  deliver enough) — grounding 1.5 and 1.7 in the same physical constraint
  instead of two independently-calibrated numbers.
- Whatever saturation/step rate results is reported as a **finding**, not
  iterated against.
- Safety ceiling (40Nm) and hip/knee split (70/30) remain explicit flagged
  external TBDs pending real physician/PT/biomechanical input — config-driven,
  unchanged by this step.

## Step 6 — Revalidate and regenerate results (after every step, not just once)

Update `layer_b_translation/tests/run_full_dataset.py` to report **train vs.
held-out-test numbers side by side** for every recalibrated component (1.1
accuracy, 1.3 r/MAE, 1.5/1.7 saturation+step rate), with **train subject count
printed next to every number** as a standing small-n caveat (32 subjects, ~26
train, is a small base for subject-level splitting — don't let the results
read as more statistically solid than they are). Regenerate all
`layer_b_translation/results/*.txt` + `summary.txt`.

- Gate: full dataset still runs with 0 pipeline errors; every changed
  component's held-out number is present, and explicitly states whether its
  step's pre-registered gate was met or honestly not met.

## Ordering (hard dependencies)

Step 1 → Step 0 → Step 2 → (Step 3 can run in parallel with Step 2, both only
need Step 1) → Step 4 (needs Step 2's direction output for BoS asymmetry) →
Step 5 (needs Step 3's time-to-impact) → Step 6 (rerun after each prior step,
not only at the very end).

## Explicitly out of scope for this session

- Per-fall-category time-to-impact calibration (overfitting risk, would need
  materially more data — multi-session item if ever pursued).
- A fully causal, actuator-latency-validated real-time controller beyond the
  first-crossing lead-time metric in Step 4 (would need real hardware timing
  data, not available here).

## Verification

- `python3 -m pytest layer_b_translation/tests/` passes after each step.
- `python3 layer_b_translation/tests/run_full_dataset.py` runs clean (0
  pipeline errors) after each step; inspect the regenerated `results/*.txt`
  for that step's specific before/after numbers and gate outcome.
- Final end-to-end check: `python3 layer_b_translation/pipeline.py` on the
  existing sample trials, sanity-read by eye.
