# Layer B (Translation Layer) — Implementation Plan for Claude Code

## Context for Claude Code

This is part of **SthiraSangyaan**, a fall detection and prevention system for Parkinson's patients. Layer A (fall detection) is already implemented and validated in the repo `hiteshPersonal27/KFall_analysis` (Threshold, SVM, and ConvLSTM models on the KFall dataset — 100Hz lower-back IMU data, with derived signals ACC_M, GYR_M, VV already computed).

Layer B's job is to take a confirmed fall event (with its pre-impact signal window) and convert it into concrete actuator commands: which motor fires, how hard, how fast, and for how long. This plan implements Layer B in two phases: deterministic formula-based components first, then the one component that needs a learned/classified signal.

**Reference dataset:** KFall — 100Hz IMU trials, 32 subjects, with T_onset and T_impact temporal labels already validated as reliable.

**Existing validated signal:** direction classification rule — forward falls show a pre-impact ACC_M bump (reflexive bracing), backward/lateral falls show a pre-impact dip (unresisted freefall). This rule exists conceptually but has not yet been coded as a standalone function.

---

## Phase 1 — Deterministic / Formula-Based Components

Build these as pure functions, in dependency order. No training data needed — implement, then validate against known biomechanics conventions and sanity-check against KFall trial signals.

### 1.1 Direction Classifier (rule-based, not ML)
- **Input:** ACC_M and GYR_M signal window around T_onset (pre-impact segment)
- **Logic:** classify forward vs backward/lateral using the ACC_M bump-vs-dip signature; use trunk angle sign (integrated GYR_M or existing VV/orientation pipeline) to separate backward vs lateral
- **Output:** `direction ∈ {forward, backward, lateral}`
- **Note:** This should be coded as a standalone, callable function even though the rule is already known — it doesn't exist as code yet.

### 1.2 Center of Mass (CoM) Position & Velocity
- **Input:** body segment mass fractions (standard anthropometric table, e.g. de Leva 1996) × patient body mass (Layer C input, can stub with a default adult mass for now) + available joint/orientation angle estimates
- **Constraint:** current hardware is a single lower-back IMU — full multi-segment CoM is not physically derivable from this alone. Implement a **single-segment approximation**: treat the lower-back IMU position as a proxy for whole-body CoM, using its estimated position/velocity from the existing orientation+integration pipeline (the same one used for VV).
- **Output:** `CoM_position(t)`, `CoM_velocity(t)` — approximate, single-IMU-based
- **Flag clearly in code comments** that this is a simplification pending multi-IMU hardware, not the full biomechanical CoM model from the literature.

### 1.3 Time-to-Impact Estimate
- **Input:** `CoM_velocity` (from 1.2), estimated fall height (use patient leg length or a default anthropometric estimate as a stand-in for now)
- **Logic:** kinematic free-fall-style estimate: time remaining until CoM reaches ground-contact height, given current vertical velocity
- **Output:** `time_to_impact_ms`
- **Cross-check:** compare against empirical lead-time statistics already computed in `paper_implementation/results/` (the T_onset-to-T_impact lead times from the Threshold/SVM/ConvLSTM evaluation) as a sanity bound — flag if the formula output diverges wildly from observed empirical lead times.

### 1.4 Capture Point / XCoM
- **Input:** `CoM_position`, `CoM_velocity` (from 1.2), leg length
- **Formula:** Hof's linear inverted pendulum model — `XCoM = CoM_position + CoM_velocity / sqrt(g / leg_length)`
- **Output:** `XCoM_position`, and `XCoM_deficit` (distance of XCoM outside the base of support — use a stubbed/default base-of-support boundary for now, since true stance-foot position isn't available until Phase 2)

### 1.5 Torque Magnitude
- **Input:** `XCoM_deficit` (from 1.4), patient mass (Layer C, stub default for now)
- **Formula:** derive required corrective torque from momentum deficit implied by XCoM_deficit × mass; distribute across hip/knee per a simple fixed ratio (document the ratio assumption clearly, e.g. 70/30 hip/knee, pending literature-backed values)
- **Output:** `torque_hip_Nm`, `torque_knee_Nm`

### 1.6 Torque Profile Shape
- **Input:** `torque_hip_Nm`, `torque_knee_Nm` (from 1.5), `time_to_impact_ms` (from 1.3)
- **Logic:** generate a trapezoidal ramp profile (ramp-up, hold, ramp-down) scaled to fit within `time_to_impact_ms`, never instantaneous/square-wave
- **Output:** time-series torque command per actuator, ready to hand to a motor controller interface (stub the actual motor interface — out of scope here)

### 1.7 Recovery Strategy Selection (rule table, not ML)
- **Input:** `XCoM_deficit`, `direction` (from 1.1), `time_to_impact_ms`
- **Logic:** simple decision table — e.g. `if XCoM_deficit > threshold_high → stepping strategy; elif direction == forward → hip-torque strategy; else → ankle-torque strategy`. Implement as an editable config/lookup table, not hardcoded if/else, so thresholds can be tuned later without touching logic.
- **Output:** `strategy ∈ {hip_torque, ankle_torque, stepping}`

**Phase 1 deliverable:** an end-to-end callable pipeline — given a fall-window signal segment (already flagged as a fall by the existing Layer A models), produce `direction`, `time_to_impact_ms`, `CoM/XCoM` values, `strategy`, and a full torque profile per actuator. This is testable against KFall trial data even without real actuator hardware — it's a pure signal-processing pipeline at this stage.

---

## Phase 2 — Model/Classifier-Based Components

### 2.1 Stance-Leg Identification
- **Problem:** determine which foot is currently bearing weight, at the moment of fall onset, from a single lower-back IMU signal
- **Constraint:** true stance-leg identification normally needs foot-mounted sensors or multiple IMUs; a single lower-back IMU can only approximate this from gait-cycle phase inference
- **Approach:**
  - First attempt as a **rule-based gait-phase estimator**: use periodicity/step-timing detected in the pre-onset ACC_M/GYR_M signal (peak-to-peak intervals during walking-classified windows) to estimate gait phase and infer likely stance leg at fall onset
  - If the rule-based approach proves unreliable in validation, fall back to a lightweight trained classifier (e.g. small window-based model similar in spirit to the existing SVM approach in `paper_implementation/`) using KFall trip/slip trials (F11–F15) where stance leg is inferable from task description/video reference if available
- **Output:** `stance_leg ∈ {left, right, unknown}` — note that "unknown" should be a valid fallback output, not an error, since the current hardware genuinely cannot resolve this in all cases
- **Integration point:** feeds back into item 1.4 (XCoM_deficit base-of-support boundary) and item 1.7 (which specific actuator side fires), replacing the stubbed defaults used in Phase 1

### 2.2 Cause/Context Classification (Detection layer, listed here since it was raised in the same discussion but belongs to Layer A, not B)
- **Input:** pre-onset signal window (before T_onset), not the fall window itself
- **Logic:** classify into `{sit_stand_transition, static_fainting, dynamic_onset, trip_slip}` based on presence/absence of rhythmic pre-fall motion — flat/still signal suggests static fainting, periodic walking-like signal suggests dynamic onset/trip-slip, a short transition burst suggests sit-stand
- **Output:** `fall_cause_context` category
- **Use:** this determines whether item 2 (acceleration magnitude) in the Detection layer should be trusted or bypassed at inference time, per the fall-type requirement matrix already documented — implement this switch explicitly (e.g., a config flag per detected category) rather than leaving it as an implicit assumption
- **Note for Claude Code:** this task is technically Layer A, not Layer B, but is included here since it was designed in the same planning session and has direct data dependencies on the fall-type matrix already documented in the project.

**Phase 2 deliverable:** stance-leg output feeding back into Phase 1's stubbed defaults (1.4, 1.7), and a cause/context classifier that can gate Layer A's acceleration-based detection per the documented fall-type requirement matrix.

---

## Repo Placement

Suggest a new top-level folder: `layer_b_translation/` (parallel to existing `fall_pattern_analysis/` and `paper_implementation/`), with submodules:
```
layer_b_translation/
├── phase1_formulas/
│   ├── direction_classifier.py      # 1.1
│   ├── com_estimation.py            # 1.2
│   ├── time_to_impact.py            # 1.3
│   ├── capture_point.py             # 1.4
│   ├── torque_magnitude.py          # 1.5
│   ├── torque_profile.py            # 1.6
│   └── recovery_strategy.py         # 1.7 (config-driven lookup table)
├── phase2_models/
│   ├── stance_leg_classifier.py     # 2.1
│   └── cause_context_classifier.py  # 2.2 (Layer A, included for continuity)
├── config/
│   └── recovery_strategy_thresholds.yaml
├── tests/
│   └── (validate each formula against known KFall trials + sanity bounds)
└── README.md
```

## Explicit Stubs / Assumptions to Flag in Code

- Patient body mass: use a default (e.g. 70kg) until Layer C calibration input exists
- Leg length: use a default anthropometric estimate until Layer C calibration input exists
- Base-of-support boundary (for XCoM_deficit): use a fixed default footprint until stance-leg identification (2.1) is complete
- Hip/knee torque distribution ratio (1.5): document as a placeholder assumption, not literature-validated yet
- Actuator hardware interface: out of scope, stub the output as a torque-vs-time array only

## Success Criteria for Phase 1

- All formulas run end-to-end on a sample KFall fall trial without errors
- `time_to_impact_ms` output falls within a plausible range of the empirical lead-time statistics already in `paper_implementation/results/`
- Direction classifier output matches the physically validated ACC_M bump/dip rule on a held-out sample of labeled fall trials
- Torque profile never exceeds a placeholder safety ceiling (to be replaced by real Layer D value later)
