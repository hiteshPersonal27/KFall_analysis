"""
2.2 -- Cause/Context Classification (Layer A, included here per the plan doc
since it was designed in the same session -- see docs/LayerB_Implementation_Plan.md).

Classifies the window BEFORE T_onset (not the fall window itself) into
{sit_stand_transition, static_fainting, dynamic_onset} using two features:
  - std(ACC_M) over the pre-onset window: near-zero for static fainting
    (subject was still before collapsing), higher for active movement.
  - peak normalized autocorrelation in the 0.4-1.2s lag range: high for
    periodic/rhythmic motion (walking), low for non-periodic motion
    (sit-stand transition, static fainting).

"dynamic_onset" merges the plan's "dynamic-onset fainting" (F09-F10) and
"trip-slip" (F11-F15) categories, since the plan itself notes both produce the
same periodic-walking pre-onset signature and can't be separated by this
signal alone -- distinguishing them would need additional context (e.g. a
detected trip/slip perturbation vs. a gradual faint), out of scope here.

Thresholds calibrated empirically against the KFall dataset's true fall-cause
labels (task ID -> category, see docs/Fall_Prevention_Actuator_Input_Pipeline.md
Table 2) -- see layer_b_translation/tests/run_full_dataset.py for the profiling
that produced them:
  sit_stand:         std ~0.070 (median), autocorr peak ~0.096 (median)
  static_fainting:   std ~0.013 (median), autocorr peak ~0.105 (median)
  dynamic_fainting:  std ~0.113 (median), autocorr peak ~0.501 (median)
  trip_slip:         std ~0.136 (median), autocorr peak ~0.522 (median)
"""

import numpy as np

WINDOW_S = 3.0            # pre-onset window length to analyze
FLAT_STD_THRESH_G = 0.03  # below this, pre-onset motion is "flat" (fainting-while-still)
PERIODIC_LAG_MIN_S = 0.4  # walking-cadence autocorrelation lag range
PERIODIC_LAG_MAX_S = 1.2
PERIODIC_THRESH = 0.30    # normalized autocorrelation peak above this = rhythmic/periodic


def classify_cause_context(acc_m, onset, fs=100, window_s=WINDOW_S):
    """
    acc_m: full-trial ACC_M array.
    onset: fall onset frame index.

    Returns "sit_stand_transition" | "static_fainting" | "dynamic_onset" | "unknown".
    "unknown" is a valid output when there isn't enough pre-onset history (e.g.
    onset near the start of the recording) to judge -- not an error.
    """
    onset = int(onset)
    start = max(0, onset - int(window_s * fs))
    pre = acc_m[start:onset]

    if len(pre) < 20:
        return "unknown"

    std = pre.std()
    if std < FLAT_STD_THRESH_G:
        return "static_fainting"

    pre_centered = pre - pre.mean()
    autocorr = np.correlate(pre_centered, pre_centered, mode="full")[len(pre_centered) - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-9)

    lag_min = int(PERIODIC_LAG_MIN_S * fs)
    lag_max = min(int(PERIODIC_LAG_MAX_S * fs), len(autocorr) - 1)
    peak_periodic = autocorr[lag_min:lag_max].max() if lag_max > lag_min else 0.0

    if peak_periodic > PERIODIC_THRESH:
        return "dynamic_onset"
    return "sit_stand_transition"


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    from data import load_trial  # noqa: E402

    samples = [(6, 20, 1, "sit_stand_transition"), (6, 25, 1, "static_fainting"),
               (6, 28, 1, "dynamic_onset"), (6, 30, 1, "dynamic_onset")]
    for subject, task, trial_id, expected in samples:
        t = load_trial(subject, task, trial_id)
        if t is None or t.onset is None:
            print(f"S{subject:02d} T{task} trial{trial_id}: could not load")
            continue
        pred = classify_cause_context(t.acc_m, t.onset)
        match = "OK" if pred == expected else "MISMATCH"
        print(f"S{subject:02d} T{task} trial{trial_id}: expected={expected} predicted={pred} [{match}]")
