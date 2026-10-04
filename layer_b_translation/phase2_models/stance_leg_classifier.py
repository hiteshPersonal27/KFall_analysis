"""
2.1 -- Stance-Leg Identification (rule-based gait-phase estimator, Phase 2 per
the plan doc -- see docs/LayerB_Implementation_Plan.md).

CONSTRAINT, FLAGGED: a single lower-back IMU cannot truly resolve left vs.
right stance leg -- that needs foot-mounted sensors, multiple IMUs, or vision
(plan doc Phase 2, item 2.1). This module implements the plan's specified
first attempt -- a rule-based estimator using mediolateral trunk sway (a
known gait-analysis heuristic: during single-leg stance, the trunk/CoM shifts
laterally toward the stance leg) -- using roll deviation near fall onset as a
proxy for that sway, since this pipeline doesn't expose raw per-axis
acceleration (only ACC_M/GYR_M magnitudes + Euler angles, see
paper_implementation/data.py::Trial).

UNVALIDATED: the KFall dataset has no ground-truth stance-leg label, so this
cannot be scored for accuracy (unlike 1.1's direction or 2.2's cause/context,
both of which have task-ID-derived ground truth). Left vs. right is assigned
by an arbitrary sign convention documented below -- treat this as a structural
placeholder demonstrating the *mechanism*, not a validated result. "unknown"
is deliberately returned outside a walking/dynamic context, per the plan's
explicit requirement that unknown be a valid fallback, not an error.
"""

import numpy as np

ROLL_BASELINE_WINDOW_FRAMES = 100  # 1s @ 100Hz
ROLL_NEAR_ONSET_WINDOW_FRAMES = 30  # 0.3s @ 100Hz
APPLICABLE_CONTEXTS = {"dynamic_onset"}  # only meaningful when an active gait cycle exists


def classify_stance_leg(roll, onset, cause_context, fs=100):
    """
    roll: full-trial EulerX array (degrees).
    onset: fall onset frame index.
    cause_context: output of cause_context_classifier.classify_cause_context --
        stance leg is only inferable when the subject was walking (dynamic_onset);
        sit-stand/fainting trials have no active gait cycle / stance leg to identify
        (matches the plan's fall-type requirement matrix, item 6).

    Returns "left" | "right" | "unknown".

    Sign convention (arbitrary, undocumented in source literature for this exact
    sensor placement/orientation -- would need bench validation against a
    ground-truth stance-leg reference to confirm which sign maps to which side):
    positive roll deviation near onset (relative to the pre-onset baseline) ->
    "left"; negative -> "right".
    """
    if cause_context not in APPLICABLE_CONTEXTS:
        return "unknown"

    onset = int(onset)
    baseline_start = max(0, onset - ROLL_BASELINE_WINDOW_FRAMES)
    if onset <= baseline_start:
        return "unknown"
    baseline_roll = np.mean(roll[baseline_start:onset])

    near_start = max(0, onset - ROLL_NEAR_ONSET_WINDOW_FRAMES)
    near_onset_roll = np.mean(roll[near_start:onset])

    return "left" if (near_onset_roll - baseline_roll) > 0 else "right"


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_trial  # noqa: E402
    from cause_context_classifier import classify_cause_context  # noqa: E402

    samples = [(6, 30, 1), (6, 31, 1), (6, 20, 1)]
    for subject, task, trial_id in samples:
        t = load_trial(subject, task, trial_id)
        if t is None or t.onset is None:
            print(f"S{subject:02d} T{task} trial{trial_id}: could not load")
            continue
        context = classify_cause_context(t.acc_m, t.onset)
        stance = classify_stance_leg(t.roll, t.onset, context)
        print(f"S{subject:02d} T{task} trial{trial_id}: context={context} stance_leg={stance}")
