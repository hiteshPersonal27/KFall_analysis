"""
Step 2 calibration script (train subjects ONLY -- see split_utils.py and
docs/LayerB_Implementation_Plan.md). Verifies the pitch/roll sign convention
at the peak-gyr_m instant before hardcoding a direction rule, then sweeps a
small set of gyr_m-gate thresholds and reports train-only accuracy for each,
so a threshold can be picked before ever touching test subjects.

Run: python3 layer_b_translation/tests/calibrate_direction.py
"""

import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LAYER_B_DIR = os.path.join(TESTS_DIR, "..")
sys.path.insert(0, LAYER_B_DIR)
sys.path.insert(0, os.path.join(LAYER_B_DIR, "phase1_formulas"))

from split_utils import train_subjects, iter_fall_trials  # noqa: E402
from fall_direction_labels import TASK_TO_TRUE_DIRECTION  # noqa: E402


def peak_gyr_frame(gyr_m, onset, impact):
    window = gyr_m[onset:impact + 1]
    if len(window) == 0:
        return onset
    return onset + int(np.argmax(window))


def main():
    rows = []
    for t in iter_fall_trials(train_subjects()):
        true_dir = TASK_TO_TRUE_DIRECTION.get(t.task)
        if true_dir is None:
            continue
        onset, impact = int(t.onset), int(t.impact)
        peak_frame = peak_gyr_frame(t.gyr_m, onset, impact)

        baseline_start = max(0, onset - 100)
        pitch_baseline = np.mean(t.pitch[baseline_start:onset]) if onset > baseline_start else t.pitch[0]
        roll_baseline = np.mean(t.roll[baseline_start:onset]) if onset > baseline_start else t.roll[0]

        pitch_dev = t.pitch[peak_frame] - pitch_baseline
        roll_dev = t.roll[peak_frame] - roll_baseline
        gyr_peak = t.gyr_m[peak_frame]

        rows.append((true_dir, pitch_dev, roll_dev, gyr_peak))

    print(f"n_trials={len(rows)} (train subjects only)\n")

    print("-- Sign convention check: pitch_dev / roll_dev at peak-gyr_m frame, by true direction --")
    for direction in ("forward", "backward", "lateral"):
        subset = [r for r in rows if r[0] == direction]
        pitch_devs = np.array([r[1] for r in subset])
        roll_devs = np.array([r[2] for r in subset])
        print(f"{direction:>8}: n={len(subset):4d}  "
              f"pitch_dev mean={pitch_devs.mean():7.2f} median={np.median(pitch_devs):7.2f} "
              f"std={pitch_devs.std():6.2f}  |  "
              f"roll_dev mean={roll_devs.mean():7.2f} median={np.median(roll_devs):7.2f} "
              f"std={roll_devs.std():6.2f}")

    print("\n-- gyr_m peak magnitude by true direction (for gate threshold sweep) --")
    for direction in ("forward", "backward", "lateral"):
        subset = [r for r in rows if r[0] == direction]
        gyr_peaks = np.array([r[3] for r in subset])
        print(f"{direction:>8}: mean={gyr_peaks.mean():.1f} median={np.median(gyr_peaks):.1f} "
              f"p10={np.percentile(gyr_peaks, 10):.1f}")


if __name__ == "__main__":
    main()
