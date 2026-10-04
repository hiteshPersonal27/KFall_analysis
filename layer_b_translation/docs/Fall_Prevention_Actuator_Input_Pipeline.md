# Fall Prevention Actuator — Input Pipeline Reference

This document combines both source files your junior colleague prepared — the general variable/layer breakdown, and the fall-type-specific (F01–F15) requirement matrix — along with the explanations discussed, so it can be used as a single reference.

---

## Part 1: Why this pipeline exists (high-level explanation)

The wearable mechanical device (hip/knee actuator, soft-exoskeleton style) needs to physically push back against a fall **before** the person hits the ground. To do that safely and correctly, raw sensor data has to be translated step by step into a motor command. That translation happens in four layers:

| Layer | Question it answers |
|---|---|
| **A. Detection** | Is a fall happening, and roughly which way? |
| **B. Translation** | Given a fall is happening, how urgent is it, which leg needs to act, and how hard/how long should it push? |
| **C. Patient calibration** | How do we adjust all of the above for this specific patient's body and movement style? |
| **D. Safety** | No matter what the math says, what is the absolute ceiling we will never cross? |

Layer A already overlaps with your existing fall-detection work (KFall-based). Layer B is the actual new research gap — converting a detected fall into concrete numbers (time-to-impact, torque, direction) a motor controller can use. Layer C personalizes the pipeline per patient. Layer D is a hard, non-negotiable safety net.

**Two separate classification axes feed this pipeline, and both matter:**
- **Direction of fall** (forward / backward / lateral) — feeds Layer B, determines which actuator fires and which way it pushes. Physically validated: forward falls show a pre-impact acceleration-magnitude (ACC_M) bump from reflexive bracing; backward/lateral falls show a pre-impact dip (unresisted freefall).
- **Cause/context of fall** (sit-stand transition / static-onset fainting / dynamic-onset fainting / trip-slip) — feeds Layer A, determines which sensor inputs are even trustworthy right before the fall (e.g., fainting-while-seated has no pre-fall acceleration spike, so acceleration-based detection fails there and rotation must carry the signal instead).

---

## Part 2: Document 1 — Variable Layer Breakdown (general, layer-based)

### A. Detection-layer (already in your pipeline)

| # | Variable | Why It's Needed | Where/How You Get It |
|---|---|---|---|
| 1 | Fall confirmation flag/probability | This is the master trigger — nothing downstream fires until this crosses your decision threshold | Output of your DL classifier (video+IMU+speech+physio fusion), cross-checked against the rule-based threshold layer |
| 2 | Acceleration magnitude | Confirms a fall-like event is happening (vs. sitting/bending); its rate of rise also hints at fall severity | Computed from raw IMU accelerometer signal (√(ax²+ay²+az²)) over your 50-frame window |
| 3 | Angular velocity magnitude | Separates genuine falls from fast controlled movement (sitting quickly) — a key false-positive filter | Computed from raw IMU gyroscope signal, same window |
| 4 | Trunk angle change + direction | Direction (forward/back/lateral) is the first-pass signal for which side of the body needs support | Integrated from gyroscope data or estimated via a complementary/Kalman filter fusing accel+gyro; direction from per-axis sign |

### B. Translation-layer (your actual research gap)

| # | Variable | Why It's Needed | Where/How You Get It |
|---|---|---|---|
| 5 | Time-to-impact estimate | Sets the urgency of the actuator response — a 100ms-to-impact fall needs a much sharper torque ramp than a 400ms one | Modeled from vertical CoM velocity + estimated fall height (leg length), using kinematic equations (free-fall-like descent models) — this is not directly measured, it's computed |
| 6 | Fall direction relative to stance leg | Tells the controller which specific hip/knee actuator (left vs. right) needs to fire, not just "which direction" | Combine trunk angle direction (item 4) with gait-phase estimate (which foot is in stance) — gait phase usually comes from foot-pressure/IMU-based gait segmentation |
| 7 | CoM position & velocity | The foundational biomechanical state variable — nearly every balance-recovery controller in the literature is built on tracking CoM, not raw limb angles | Estimated from a body segment model (mass-weighted average of segment positions) using joint angles from IMUs/video pose estimation, differentiated for velocity |
| 8 | Capture point / XCoM | The actual "how close to falling" number used to decide whether and how hard to intervene — this is the standard bridge between kinematics and control in the literature | Computed formula: XCoM = CoM position + CoM velocity/√(g/leg length) (Hof's linear inverted pendulum model) — derived from item 7, not separately measured |
| 9 | Recovery strategy selection | Different fall types need different actuator behavior — a hip-stiffening response is wrong for a fall that needs foot-placement correction | Rule/logic layer that maps XCoM position + fall direction + time-to-impact into one of a small set of pre-defined strategies (hip torque / ankle torque / stepping) |
| 10 | Torque magnitude per joint | The actual number sent to the motor controller — how hard the actuator pushes | Computed from required change in CoM momentum (from XCoM deficit) × patient mass, distributed across hip/knee per the biomechanical models in your reading list |
| 11 | Torque profile shape | Prevents jerky, potentially injurious actuation — trapezoidal ramps avoid the instability of a sudden square-wave command | Design choice (not measured) — engineered based on actuator response characteristics and the ~10ms onset-delay convention from the literature |

### C. Patient/context parameters (calibration, set once or periodically)

| # | Variable | Why It's Needed | Where/How You Get It |
|---|---|---|---|
| 12 | Body mass / limb segment mass | Every torque calculation (item 10) is force × mass-dependent — without this, torque numbers are meaningless for a specific patient | Manually entered at setup, or estimated via standard anthropometric tables (% of body mass per segment) from patient height/weight |
| 13 | Baseline gait/mobility profile | Parkinson's patients have altered baseline movement (shuffling gait, freezing episodes) — using generic thresholds risks false triggers or missed detections | Recorded during a calibration session — walk the patient through normal ADLs and log baseline acceleration/angle/gait ranges before deployment |
| 14 | Actuator latency/response characteristics | Your time-to-impact budget (item 5) must subtract how long the actual motor takes to reach commanded torque — otherwise the command arrives too late | Measured from hardware datasheets/bench testing of the hip and knee actuators shown in your device image |

### D. System-level safety parameters

| # | Variable | Why It's Needed | Where/How You Get It |
|---|---|---|---|
| 15 | Maximum safe torque limit per actuator | Hard ceiling to prevent the exoskeleton itself from injuring a frail/osteoporotic patient during a false-positive or over-aggressive correction | Set by biomechanical safety literature + physician/PT input, hard-coded as a firmware-level cap, never exceeded regardless of computed torque |
| 16 | Fallback/failsafe trigger | If the DL model lags or fails, this simple threshold check still fires a basic protective response — a safety net under your main pipeline | Rule-based logic running in parallel (e.g., angular velocity > 200°/s AND acceleration > 0.9g → default safe response), independent of the neural network |

---

## Part 3: Document 2 — Fall-Type-Based Input Requirements (F01–F15)

### Table 2: Fall Type Reference (Task IDs)

| Fall ID | Task ID | Fall Name / Description |
|---|---|---|
| F01 | 20 | Forward fall when trying to sit down |
| F02 | 21 | Backward fall when trying to sit down |
| F03 | 22 | Lateral fall when trying to sit down |
| F04 | 23 | Forward fall when trying to get up |
| F05 | 24 | Lateral fall when trying to get up |
| F06 | 25 | Forward fall while sitting, caused by fainting |
| F07 | 26 | Lateral fall while sitting, caused by fainting |
| F08 | 27 | Backward fall while sitting, caused by fainting |
| F09 | 28 | Vertical (forward) fall while walking, caused by fainting |
| F10 | 29 | Fall while walking, use of hands to dampen fall, caused by fainting |
| F11 | 30 | Forward fall while walking, caused by a trip |
| F12 | 31 | Forward fall while jogging, caused by a trip |
| F13 | 32 | Forward fall while walking, caused by a slip |
| F14 | 33 | Lateral fall while walking, caused by a slip |
| F15 | 34 | Backward fall while walking, caused by a slip |

**Four fall-cause categories used for grouping:**
- **A. Sit/Stand-transition** (F01–F05)
- **B. Static-onset Fainting** (F06–F08)
- **C. Dynamic-onset Fainting** (F09–F10)
- **D. Trip/Slip** (F11–F15)

### Input Requirement by Category, With Reasoning

**Detection Layer**

| # | Input | Why Needed | Where It's From | A. Sit/Stand (F01–F05) | B. Static Fainting (F06–F08) | C. Dynamic Fainting (F09–F10) | D. Trip/Slip (F11–F15) |
|---|---|---|---|---|---|---|---|
| 1 | Fall confirmation flag/probability | Master trigger — nothing fires until this crosses threshold | DL classifier output (fusion) | Required — universal master trigger | Required — same | Required — same | Required — same |
| 2 | Acceleration magnitude | Confirms fall-like event; rate of rise hints at severity | IMU accelerometer, √(ax²+ay²+az²) | Required — dynamic sit/stand transition produces a clear Y-axis disruption | **Not Required** — no accel spike at onset; body is static before fainting | Required — walking is the preceding activity, so the general dynamic-accel rule applies despite the cause being fainting | Required — sharp accel spike is the primary detection signal |
| 3 | Angular velocity magnitude | Separates genuine falls from fast controlled movement | IMU gyroscope, same window | Required — filters fast-but-controlled movement from a real fall | Required — primary sensitive channel here since accel is unusable | Required — confirms rotational onset alongside accel | Required — confirms rotational onset alongside accel |
| 4 | Trunk angle change + direction | First-pass signal for which side needs support | Integrated gyro / Kalman fusion | Required — needed to pick actuator side | Required — only reliable direction source since accel is unusable | Required — same role as elsewhere | Required — same role as elsewhere |

**Translation Layer**

| # | Input | Why Needed | Where It's From | A. Sit/Stand (F01–F05) | B. Static Fainting (F06–F08) | C. Dynamic Fainting (F09–F10) | D. Trip/Slip (F11–F15) |
|---|---|---|---|---|---|---|---|
| 5 | Time-to-impact estimate | Sets urgency of actuator response | Modeled from CoM velocity + leg length | Required — needed to schedule torque even for slower onset | Required — longest lead time of all groups, but still must be quantified | Required — dynamic onset shortens the margin, so timing matters more | Required — shortest lead time of all groups, most time-critical use |
| 6 | Fall direction relative to stance leg | Tells controller which specific actuator (L/R) fires | Trunk angle + gait-phase/stance-foot | **Not Required** — subject is seated/standing, both feet grounded, no active stance leg to identify | **Not Required** — fall originates from static sitting, no gait cycle in progress | Required — subject was walking, so a stance leg exists and needs identifying | Required — critical; trip loads the swing leg, slip loads the stance leg |
| 7 | CoM position & velocity | Foundational biomechanical state variable | Body segment model from joint angles | Required — foundational for every balance calculation | Required — same | Required — same | Required — same |
| 8 | Capture point / XCoM | Standard "how close to falling" number | Derived from CoM via Hof's model | Required — universal bridge from kinematics to control | Required — same | Required — same | Required — same |
| 9 | Recovery strategy selection | Different fall types need different actuator behavior | Rule layer: XCoM + direction + time-to-impact | Required — hip-torque strategy must be explicitly chosen | Required — trunk-focused strategy must be actively selected, not defaulted | Required — same | Required — may need to switch to stepping/foot-placement strategy |
| 10 | Torque magnitude per joint | Actual number sent to motor controller | Computed from CoM momentum deficit × mass | Required — moderate torque still needs to be computed | Required — lower magnitude, but still computed, not assumed | Required — moderate-to-higher magnitude given dynamic onset | Required — higher magnitude given greater momentum, less time |
| 11 | Torque profile shape | Prevents jerky, injurious actuation | Trapezoidal ramp, engineered | Required — even slow falls need a controlled ramp | Required — ramp can be gentler, but still needed | Required — same | Required — ramp must be faster but still shaped, not a square wave |

**Medical Context**

| # | Input | Why Needed | Where It's From | A. Sit/Stand (F01–F05) | B. Static Fainting (F06–F08) | C. Dynamic Fainting (F09–F10) | D. Trip/Slip (F11–F15) |
|---|---|---|---|---|---|---|---|
| 12 | Body mass / limb segment mass | Torque = force × mass-dependent | Setup entry or anthropometric tables | Required — torque is meaningless without it | Required — same | Required — same | Required — same |
| 13 | Baseline gait/mobility profile | Prevents false triggers from patient-specific movement patterns | Calibration session, logged pre-deployment | Required — sit/stand thresholds vary meaningfully by patient baseline | **Not Required** — fainting onset is independent of gait pattern, adds little discriminative value here | Required — walking-based onset means gait baseline is relevant | Required — highly relevant; trip/slip false-positive risk is gait-baseline-sensitive |
| 14 | Actuator latency/response characteristics | Time-to-impact budget must subtract motor response delay | Hardware datasheets / bench testing | Required — needed to schedule actuation correctly, even unhurried | Required — same | Required — same | Required — most critical here, time budget is already shortest |

### Full Conditional Requirement Matrix (F01–F15, individual fall types)

✔ = Required, ✘ = Not Required

| # | Input | F01 | F02 | F03 | F04 | F05 | F06 | F07 | F08 | F09 | F10 | F11 | F12 | F13 | F14 | F15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Fall confirmation flag/probability | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 2 | Acceleration magnitude | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 3 | Angular velocity magnitude | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 4 | Trunk angle change + direction | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 5 | Time-to-impact estimate | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 6 | Fall direction relative to stance leg | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 7 | CoM position & velocity | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 8 | Capture point / XCoM | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 9 | Recovery strategy selection | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 10 | Torque magnitude per joint | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 11 | Torque profile shape | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 12 | Body mass / limb segment mass | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 13 | Baseline gait/mobility profile | ✔ | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| 14 | Actuator latency/response characteristics | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

---

## Part 4: Key Discussion Points (from working session)

**Why layers exist at all:** raw sensor numbers (e.g. "1.3g acceleration spike") cannot be sent to a motor directly. Each layer converts the previous layer's output into something more actionable, ending in a literal torque command.

**Two independent classification axes both feed the pipeline, at different layers:**
- Direction (forward/backward/lateral) → feeds **Layer B (Translation)**, decides which actuator fires and which way.
- Cause/context (sit-stand / static faint / dynamic faint / trip-slip) → feeds **Layer A (Detection)**, decides which sensor inputs are trustworthy right before the fall.

They are not competing — the fall-type categorization does not replace direction classification, it answers a different question and is equally necessary.

**Inference-time classification approach:**
- Direction: already solvable from the same pre-impact window used for fall detection, using the validated ACC_M bump (forward) vs dip (backward/lateral) signature, combined with gyroscope-derived trunk angle sign.
- Cause/context: needs a separate check on the window *before* fall onset — rhythmic/periodic motion beforehand suggests dynamic onset (trip/slip while walking); flat/still motion beforehand suggests static onset (fainting); a short sit-to-stand-like burst suggests a transition-related fall. This would require a small additional classifier or rule running in parallel with the direction classifier.

## Part 5: Detailed Completion Status — Item by Item

### Layer A — Detection

| Item | Status |
|---|---|
| Fall confirmation (Threshold/SVM/ConvLSTM) | ✅ Done — reproduced and validated against KFall paper |
| Acceleration magnitude, angular velocity | ✅ Done — core signals already in pipeline |
| Direction classification (forward/back/lateral) | ✅ Physically validated (ACC_M bump vs dip), not yet coded as a standalone live classifier but the rule is known |
| Cause/context classification (static-fainting vs dynamic vs trip/slip) | ❌ Not built — only proposed conceptually, no model or rule implemented yet |

### Layer B — Translation

| Item | Status |
|---|---|
| Time-to-impact estimate | ❌ Not built — needs a kinematic model from CoM velocity + leg length, not implemented |
| Fall direction relative to stance leg | ❌ Not built — needs gait-phase/stance-leg detection, which a single lower-back IMU cannot reliably give you |
| CoM position & velocity | ❌ Not built — needs a body segment model from joint angles (multi-IMU or video pose estimation), which you don't currently have |
| Capture Point / XCoM | ❌ Not built — this is derived from CoM, so it's blocked on CoM being available first |
| Recovery strategy selection | ❌ Not built — currently just a conceptual rule table (in the docx), not implemented logic |
| Torque magnitude / profile shape | ❌ Not built — this depends on XCoM deficit and patient mass, both upstream pieces missing |

### Layer C — Patient Calibration

| Item | Status |
|---|---|
| Body mass / segment mass entry | ❌ Not built — trivial to add (manual entry or anthropometric table), just not done |
| Baseline gait profile calibration | ❌ Not built — needs a defined calibration session protocol, not designed yet |
| Actuator latency measurement | ❌ Not built — depends on actual actuator hardware, which hasn't been selected/procured (only the IMU, BNO055, has been sourced) |

### Layer D — Safety

| Item | Status |
|---|---|
| Max torque ceiling | ❌ Not built — needs physician/PT input + biomechanical literature values, not sourced yet |
| Fallback rule-based trigger | ⚠️ Partially — a working rule-based threshold detector (Rule A / VV-calibrated thresholds) already exists and could double as this, just not formally wired in as the "failsafe under the DL model" |

### What This Means

**Done:** the entire detection layer for "is this a fall" and pre-impact timing (lead time), plus the physics behind direction classification.

**Missing — the real gap:** everything from "time-to-impact" onward. The whole translation layer (B) is conceptually designed on paper but not implemented, and a good chunk of it — CoM, XCoM, stance-leg identification — is **blocked on sensing you don't currently have**. A single lower-back IMU cannot give you joint angles or gait-phase/stance-leg information; that needs either multiple IMUs (on the legs) or a camera-based pose estimation system. Nothing in Layer C or D has started either — no actuator hardware, no calibration protocol, no safety limits defined.

### Can an MVP Be Built?

**Not with the full pipeline as documented** — the full XCoM/CoM-based approach genuinely needs more sensors than currently available, so completing everything as scoped isn't realistic without new hardware (extra IMUs on the legs at minimum, or vision).

**But a stripped-down MVP is realistic without waiting for all of it**, using only what's already available:
- Fall confirmation (done) + direction classification (done, physically validated) → decide which side/direction to actuate
- A simple time-to-impact proxy from lead-time statistics already measured in the fall detection results (rather than a full kinematic CoM model)
- A fixed, pre-engineered torque profile per direction (not computed per-patient from XCoM) — engineered once, not derived live
- Manually entered patient mass + a hard-coded safety torque ceiling

This gets a working, testable end-to-end trigger-to-actuation MVP using only the single IMU + existing detection pipeline, deferring the full CoM/XCoM/stance-leg biomechanical layer as a "v2" that would require additional sensing infrastructure.
