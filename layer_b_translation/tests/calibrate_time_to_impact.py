"""
Step 3 calibration script (train subjects ONLY -- see split_utils.py). Fits
and compares three time-to-impact models, all evaluated once on held-out test
subjects: (1) constant baseline, (2) 2-param v0-only linear regression, (3)
3-param v0 + causal-early-window-gyr_m regression. Frozen coefficients from
this script's output are hardcoded into phase1_formulas/time_to_impact.py.

Run: python3 layer_b_translation/tests/calibrate_time_to_impact.py
"""

import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LAYER_B_DIR = os.path.join(TESTS_DIR, "..")
sys.path.insert(0, LAYER_B_DIR)
sys.path.insert(0, os.path.join(LAYER_B_DIR, "phase1_formulas"))

from split_utils import train_subjects, test_subjects, iter_fall_trials  # noqa: E402
from com_estimation import estimate_com  # noqa: E402
from time_to_impact import EARLY_WINDOW_FRAMES  # noqa: E402


def collect(subjects):
    rows = []
    for t in iter_fall_trials(subjects):
        com_pos, com_vel = estimate_com(t.vv)
        onset, impact = int(t.onset), int(t.impact)
        v0 = max(com_vel[onset], 0.0)
        window = t.gyr_m[onset:onset + EARLY_WINDOW_FRAMES]
        gyr_early = window.max() if len(window) else 0.0
        t_emp = (impact - onset) / 100.0 * 1000.0
        rows.append((v0, gyr_early, t_emp))
    return np.array(rows)


def main():
    train = collect(train_subjects())
    test = collect(test_subjects())

    baseline_const = train[:, 2].mean()

    A_train_2 = np.column_stack([np.ones(len(train)), train[:, 0]])
    coef2, *_ = np.linalg.lstsq(A_train_2, train[:, 2], rcond=None)

    A_train_3 = np.column_stack([np.ones(len(train)), train[:, 0], train[:, 1]])
    coef3, *_ = np.linalg.lstsq(A_train_3, train[:, 2], rcond=None)

    print(f"constant baseline (train mean): {baseline_const:.2f} ms")
    print(f"2-param fit (v0 only): a={coef2[0]:.2f} b={coef2[1]:.4f}")
    print(f"3-param fit (v0 + causal gyr_early): a={coef3[0]:.2f} b={coef3[1]:.4f} c={coef3[2]:.4f}")
    print()

    for name, data in [("TRAIN", train), ("TEST (held-out)", test)]:
        preds_const = np.full(len(data), baseline_const)
        preds_2 = coef2[0] + coef2[1] * data[:, 0]
        preds_3 = coef3[0] + coef3[1] * data[:, 0] + coef3[2] * data[:, 1]

        def stats(preds):
            r = np.corrcoef(preds, data[:, 2])[0, 1] if np.std(preds) > 0 else 0.0
            mae = np.abs(preds - data[:, 2]).mean()
            return r, mae

        r_c, mae_c = stats(preds_const)
        r_2, mae_2 = stats(preds_2)
        r_3, mae_3 = stats(preds_3)
        print(f"{name} (n={len(data)}): "
              f"constant r={r_c:.3f} MAE={mae_c:.1f}ms | "
              f"2-param r={r_2:.3f} MAE={mae_2:.1f}ms | "
              f"3-param(causal) r={r_3:.3f} MAE={mae_3:.1f}ms")


if __name__ == "__main__":
    main()
