"""
1.6 -- Torque Profile Shape

Generates a trapezoidal ramp (ramp-up, hold, ramp-down) scaled to fit within
time_to_impact_ms, never an instantaneous square wave. RAMP_FRACTION splits
the available time between ramp-up/hold/ramp-down (documented, tunable).

Also enforces a placeholder safety ceiling (real Layer D value not sourced
yet) -- clamps peak torque, never a functional replacement for real Layer D.
"""

import numpy as np

RAMP_UP_FRACTION = 0.3     # fraction of total profile time spent ramping up
RAMP_DOWN_FRACTION = 0.2   # fraction spent ramping down; remainder is "hold"
PLACEHOLDER_SAFETY_CEILING_NM = 40.0   # Layer D stub -- replace with real value
CONTROL_FS_HZ = 100        # actuator command rate, matches IMU sample rate for now


def generate_torque_profile(peak_torque_nm, time_to_impact_ms,
                              safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM,
                              fs=CONTROL_FS_HZ):
    """
    Returns (t_ms, torque_nm): 1D arrays describing torque commanded from now
    (t=0) until time_to_impact_ms, trapezoidal-shaped.

    peak_torque_nm is clamped to safety_ceiling_nm before shaping the ramp.
    """
    peak = min(peak_torque_nm, safety_ceiling_nm)

    n_samples = max(2, int(round(time_to_impact_ms / 1000.0 * fs)))
    t_ms = np.linspace(0.0, time_to_impact_ms, n_samples)

    ramp_up_end = RAMP_UP_FRACTION * time_to_impact_ms
    ramp_down_start = (1.0 - RAMP_DOWN_FRACTION) * time_to_impact_ms

    torque = np.empty(n_samples)
    for i, t in enumerate(t_ms):
        if t <= ramp_up_end and ramp_up_end > 0:
            torque[i] = peak * (t / ramp_up_end)
        elif t >= ramp_down_start and ramp_down_start < time_to_impact_ms:
            torque[i] = peak * (1.0 - (t - ramp_down_start) / (time_to_impact_ms - ramp_down_start))
        else:
            torque[i] = peak

    return t_ms, np.clip(torque, 0.0, safety_ceiling_nm)


if __name__ == "__main__":
    t_ms, torque = generate_torque_profile(peak_torque_nm=25.0, time_to_impact_ms=300.0)
    print(f"n_samples={len(t_ms)}, peak={torque.max():.2f} Nm, "
          f"starts_at_zero={torque[0] == 0.0}, ends_near_zero={torque[-1]:.2f}")
