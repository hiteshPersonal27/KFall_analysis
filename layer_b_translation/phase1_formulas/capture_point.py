"""
1.4 -- Capture Point / XCoM (Hof's linear inverted pendulum model)

XCoM = CoM_position + CoM_velocity / sqrt(g / leg_length)

xcom_deficit = how far XCoM has moved outside the base of support (BoS). Since
true stance-foot position isn't available until Phase 2 (stance-leg
identification), BoS is stubbed as a fixed symmetric footprint around the
trial's starting CoM position -- see DEFAULT_BOS_HALF_WIDTH_M.
"""

import numpy as np

from time_to_impact import G, DEFAULT_LEG_LENGTH_M

DEFAULT_BOS_HALF_WIDTH_M = 0.15   # stubbed base-of-support half-width, pending 2.1 (used when direction is unknown)

# DIRECTION-AWARE BOS (see docs/LayerB_Implementation_Plan.md fix plan, Step 4).
# HONESTY NOTE: this pipeline's "com_position" is a single-segment proxy derived
# by integrating VERTICAL velocity (VV) -- see com_estimation.py -- not a true
# horizontal CoM coordinate. So there is no valid spatial "forward" vs "backward"
# SIGN to apply asymmetrically to xcom_position - bos_center (that would require
# an actual horizontal CoM model, which single-IMU hardware can't provide -- see
# 1.2's documented hardware constraint). What CAN be justified without fabricating
# spatial precision this model doesn't have: the OVERALL margin size varies by
# fall direction, reflecting that forward falls allow a reflexive bracing/stepping
# response (a wider effective margin before correction is needed) while
# backward/lateral falls are closer to unresisted freefall (a narrower margin).
# These are ILLUSTRATIVE, order-of-magnitude values, not sourced from a specific
# citation or fit to this dataset's deficit distribution -- same "flagged
# placeholder" status as DEFAULT_BOS_HALF_WIDTH_M itself, just direction-conditioned
# instead of one constant. Do not read the specific numbers as validated.
BOS_HALF_WIDTH_BY_DIRECTION_M = {
    "forward": 0.20,   # bracing/stepping response available -> more forgiving
    "backward": 0.12,  # least visual/proprioceptive warning -> least forgiving
    "lateral": 0.12,   # limited lateral step reach -> equally unforgiving
}


def bos_half_width_for_direction(direction):
    """Direction-conditioned BoS half-width; falls back to the flat default for
    an unrecognized/unknown direction rather than raising."""
    return BOS_HALF_WIDTH_BY_DIRECTION_M.get(direction, DEFAULT_BOS_HALF_WIDTH_M)


def compute_xcom(com_position, com_velocity, leg_length_m=DEFAULT_LEG_LENGTH_M):
    """
    com_position, com_velocity: arrays or scalars (m, m/s).
    Returns xcom_position (same shape as inputs).
    """
    omega0 = np.sqrt(G / leg_length_m)
    return com_position + com_velocity / omega0


def compute_xcom_deficit(xcom_position, bos_center=0.0, bos_half_width_m=DEFAULT_BOS_HALF_WIDTH_M):
    """
    Distance of XCoM outside the base of support, clamped to >= 0 (0 means
    XCoM is still within the stubbed footprint -- no corrective action needed
    by this measure alone).
    """
    distance_from_center = np.abs(xcom_position - bos_center)
    return np.maximum(distance_from_center - bos_half_width_m, 0.0)


def find_first_crossing_frame(xcom_deficit_series):
    """
    CAUSAL re-evaluation (Step 4): xcom_deficit at any given frame already only
    depends on past data (com_position/velocity up to that frame via integration)
    -- what was non-causal in the earlier "peak deficit over the whole window"
    approach was purely the .max() AGGREGATION peeking at the whole future window.
    This walks the already-causal per-frame series and returns the index of the
    FIRST frame where deficit becomes nonzero (i.e. the first real-time-detectable
    breach), or None if it never crosses within the given window.

    xcom_deficit_series: array, one value per frame (e.g. the onset->impact window).
    Returns the first index where deficit > 0, or None.
    """
    nonzero = np.nonzero(xcom_deficit_series > 1e-9)[0]
    return int(nonzero[0]) if len(nonzero) else None


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    from data import load_trial  # noqa: E402
    from com_estimation import estimate_com  # noqa: E402

    t = load_trial(6, 20, 1)
    com_pos, com_vel = estimate_com(t.vv)
    bos_center = com_pos[t.onset]
    xcom = compute_xcom(com_pos[t.impact], com_vel[t.impact])
    deficit = compute_xcom_deficit(xcom, bos_center=bos_center)
    print(f"XCoM at impact={xcom:.4f} m, deficit={deficit:.4f} m")
