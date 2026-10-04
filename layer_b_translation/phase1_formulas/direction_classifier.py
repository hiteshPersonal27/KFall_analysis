"""
1.1 -- Direction Classifier (rule-based, not ML)

REDESIGNED (see docs/LayerB_Implementation_Plan.md fix plan, Step 2 -- the
original ACC_M bump/dip heuristic scored only 35.2% held-out accuracy because
ACC_M is an unsigned magnitude and can't actually encode "which way"). This
version uses signed Euler-angle deviation at the instant of peak rotational
rate (max GYR_M in the fall window), which does encode direction.

CALIBRATION NOTE (train-subjects-only, see split_utils.py -- empirical check
run BEFORE hardcoding this rule, per docs/LayerB_Implementation_Plan.md Step 2,
via layer_b_translation/tests/calibrate_direction.py):
  Sign convention turned out opposite to the naive anatomical guess -- ROLL
  (EulerX), not pitch, is what separates forward vs. backward strongly in this
  dataset (forward: roll_dev median -42 deg; backward: +52 deg), while PITCH
  (EulerY) is what separates out lateral falls (lateral: pitch_dev median +30
  deg, roll_dev median only +12 deg, i.e. comparatively flat in roll). This is
  presumably a sensor-mounting/axis-convention artifact specific to this
  device's Euler output, not a universal anatomical rule -- do not assume it
  transfers to a different IMU without re-running the same calibration check.

Rule: if |roll_dev| dominates |pitch_dev| (by ROLL_DOMINANCE_RATIO), the roll
SIGN picks forward vs. backward; otherwise it's lateral. RATIO swept 1.7-2.2 on
train subjects only (1880 trials); 1.9 sits in the middle of a flat ~87.7-88.0%
plateau (not the single noisy peak) -- frozen before ever touching test
subjects. Held-out (test-subject) accuracy is reported separately in
results/1.1_direction_classifier.txt by tests/run_full_dataset.py.

Run standalone for a sanity check against real KFall fall trials:
  python3 layer_b_translation/phase1_formulas/direction_classifier.py
"""

import numpy as np

BASELINE_WINDOW_FRAMES = 100     # 1s @ 100Hz, standing baseline just before onset
ROLL_DOMINANCE_RATIO = 1.9       # frozen on train subjects only, see calibration note above


def classify_direction(acc_m, gyr_m, pitch, roll, onset, impact, fs=100):
    """
    acc_m, gyr_m, pitch, roll: full-trial arrays (pitch/roll in degrees, EulerY/EulerX).
    onset, impact: frame indices (as loaded by paper_implementation.data.load_trial).

    Returns "forward" | "backward" | "lateral".
    """
    onset = int(onset)
    impact = int(impact)

    window = gyr_m[onset:impact + 1]
    peak_frame = onset + int(np.argmax(window)) if len(window) else onset

    baseline_start = max(0, onset - BASELINE_WINDOW_FRAMES)
    if onset > baseline_start:
        pitch_baseline = np.mean(pitch[baseline_start:onset])
        roll_baseline = np.mean(roll[baseline_start:onset])
    else:
        pitch_baseline = pitch[0]
        roll_baseline = roll[0]

    pitch_dev = pitch[peak_frame] - pitch_baseline
    roll_dev = roll[peak_frame] - roll_baseline

    if abs(roll_dev) > ROLL_DOMINANCE_RATIO * abs(pitch_dev):
        return "forward" if roll_dev < 0 else "backward"
    return "lateral"


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_trial  # noqa: E402
    from fall_direction_labels import TASK_TO_TRUE_DIRECTION  # noqa: E402

    samples = [(6, 20, 1), (6, 21, 1), (6, 22, 1)]
    for subject, task, trial_id in samples:
        t = load_trial(subject, task, trial_id)
        if t is None or t.onset is None:
            print(f"S{subject:02d} T{task} trial{trial_id}: could not load")
            continue
        expected = TASK_TO_TRUE_DIRECTION[task]
        pred = classify_direction(t.acc_m, t.gyr_m, t.pitch, t.roll, t.onset, t.impact)
        match = "OK" if pred == expected else "MISMATCH"
        print(f"S{subject:02d} T{task} trial{trial_id}: expected={expected} predicted={pred} [{match}]")
