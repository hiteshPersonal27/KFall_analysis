"""
1.2 -- Center of Mass (CoM) Position & Velocity

SIMPLIFICATION, FLAGGED: a single lower-back IMU cannot give a true
multi-segment CoM (that needs joint angles from multiple IMUs or video pose
estimation -- see docs/LayerB_Implementation_Plan.md Phase 2). This module
implements a single-segment approximation: the lower-back sensor position is
treated as a proxy for whole-body CoM, using the same world-frame vertical
velocity (VV) already computed and validated by the Layer A pipeline
(analyze_pattern.py::compute_signals), integrated once more for position.

This is a stand-in pending multi-IMU/vision hardware, not the de Leva (1996)
segment-weighted model referenced in the plan doc.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def estimate_com(vv, fs=100):
    """
    vv: world-frame vertical velocity array (m/s), positive = downward
        (matches the sign convention already used by the Layer A pipeline).

    Returns (com_position, com_velocity):
      com_velocity: alias of vv (m/s, positive = descending)
      com_position: cumulative vertical displacement from trial start (m,
                     positive = descended). Relative, not an absolute height --
                     only displacement/velocity are used downstream (1.3, 1.4).
    """
    com_velocity = np.asarray(vv, dtype=float)
    dt = 1.0 / fs
    com_position = cumulative_trapezoid(com_velocity, dx=dt, initial=0.0)
    return com_position, com_velocity


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper_implementation"))
    from data import load_trial  # noqa: E402

    t = load_trial(6, 20, 1)
    com_pos, com_vel = estimate_com(t.vv)
    print(f"trial n_frames={t.n_frames}, onset={t.onset}, impact={t.impact}")
    print(f"CoM velocity at onset: {com_vel[t.onset]:.3f} m/s, at impact: {com_vel[t.impact]:.3f} m/s")
    print(f"CoM displacement onset->impact: {com_pos[t.impact] - com_pos[t.onset]:.4f} m")
