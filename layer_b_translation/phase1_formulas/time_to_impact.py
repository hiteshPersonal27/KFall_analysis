"""
1.3 -- Time-to-Impact Estimate

REPLACED (see docs/LayerB_Implementation_Plan.md fix plan, Step 3 -- the
original single-constant kinematic free-fall model, calibrated AND evaluated
on the same full 2319-trial dataset, matched the mean lead time but had only
r=0.32 per-trial correlation -- it explained almost none of the real variance).

This version is a 2-feature linear regression, fit on TRAIN SUBJECTS ONLY (see
split_utils.py), evaluated once on held-out test subjects:

    t_impact_ms = A + B * v0 + C * gyr_early

  v0         = CoM velocity at onset (m/s) -- causally available at decision time.
  gyr_early  = max GYR_M over [onset, onset + EARLY_WINDOW_FRAMES) -- a short,
               CAUSAL early-window feature. IMPORTANT: this requires waiting
               EARLY_WINDOW_FRAMES (150ms) past onset before the estimate can be
               computed -- unlike v0, which is available instantly at onset. A
               3rd feature using GYR_M peak over the WHOLE onset->impact window
               was tried first and scored even higher (held-out r=0.473) but was
               REJECTED: computing it requires knowing data up to impact, i.e. it
               peeks at the future and could never be computed by a real-time
               controller. The 150ms-delayed causal version (r=0.534 held-out,
               beats even the leaky version) is what's used here.

Calibration results (see layer_b_translation/tests/calibrate_time_to_impact.py):
  constant-only baseline (train mean, no signal): held-out r ~ 0 (by definition)
  2-param (v0 only):                              held-out r=0.389, MAE=112.0ms
  3-param (v0 + gyr_early, CAUSAL):                held-out r=0.534, MAE=101.8ms  <- used
  Gate (>=0.1 r improvement over constant baseline, prefer richer causal model if
  it clears an additional >=0.1 over the 2-param version): CLEARED.

DOWNSTREAM CONSEQUENCE, MUST NOT BE DROPPED: because gyr_early needs 150ms of
data past onset, a real controller using this estimator cannot produce a
time-to-impact number until onset+150ms. The USABLE reaction window from that
decision point is `time_to_impact_ms - EARLY_WINDOW_MS`, not the full
`time_to_impact_ms` -- see `usable_reaction_time_ms()` below, used by
pipeline.py when sizing the torque profile (1.6).
"""

import numpy as np

G = 9.81                        # m/s^2, kept for anywhere true kinematics is needed
DEFAULT_LEG_LENGTH_M = 0.9       # adult-average stand-in, pending Layer C calibration (used by capture_point.py)

EARLY_WINDOW_FRAMES = 15         # 150ms @ 100Hz -- causal decision-delay window for gyr_early
EARLY_WINDOW_MS = EARLY_WINDOW_FRAMES * 10.0

# Frozen on TRAIN SUBJECTS ONLY (26 subjects, 1880 fall trials) -- see calibration
# note above and calibrate_time_to_impact.py. Not touched after seeing held-out numbers.
REGRESSION_A = 848.46
REGRESSION_B = -188.2722   # per m/s of v0
REGRESSION_C = -0.9630     # per deg/s of gyr_early


def estimate_time_to_impact_ms(v0, gyr_early):
    """
    v0: CoM velocity at onset (m/s, positive = descending), available instantly.
    gyr_early: max GYR_M over [onset, onset+EARLY_WINDOW_FRAMES) (deg/s), only
               available EARLY_WINDOW_MS after onset (see module docstring).

    Returns time_to_impact_ms (float, clamped to >= 0), measured from ONSET
    (not from the moment this can actually be computed -- see
    usable_reaction_time_ms for the number that matters for actuation).
    """
    t = REGRESSION_A + REGRESSION_B * max(v0, 0.0) + REGRESSION_C * gyr_early
    return max(t, 0.0)


def usable_reaction_time_ms(time_to_impact_ms):
    """
    The real window available to actuate, accounting for the causal decision
    delay baked into gyr_early (EARLY_WINDOW_MS). Clamped to >= 0 -- if the
    predicted time-to-impact is already less than the decision delay itself,
    there is no usable reaction time left (a genuine, honestly-reported finding
    for very fast falls, not an error).
    """
    return max(time_to_impact_ms - EARLY_WINDOW_MS, 0.0)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    from data import load_trial  # noqa: E402
    from com_estimation import estimate_com  # noqa: E402

    t = load_trial(6, 20, 1)
    com_pos, com_vel = estimate_com(t.vv)
    onset, impact = int(t.onset), int(t.impact)
    v0 = max(com_vel[onset], 0.0)
    window = t.gyr_m[onset:onset + EARLY_WINDOW_FRAMES]
    gyr_early = window.max() if len(window) else 0.0

    tti = estimate_time_to_impact_ms(v0, gyr_early)
    usable = usable_reaction_time_ms(tti)
    empirical_ms = (impact - onset) / 100.0 * 1000.0
    print(f"formula time_to_impact_ms={tti:.1f}  usable_reaction_ms={usable:.1f}  "
          f"empirical(onset->impact)={empirical_ms:.1f} ms")
