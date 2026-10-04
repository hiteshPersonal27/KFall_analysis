"""
End-to-end Layer B pipeline (Phase 1): given a Layer-A-confirmed fall Trial
(see paper_implementation/data.py), produce direction, time-to-impact, CoM/XCoM,
strategy, and a torque profile per actuator.

Run standalone:
  python3 layer_b_translation/pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_formulas"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_implementation"))

from direction_classifier import classify_direction
from com_estimation import estimate_com
from time_to_impact import (estimate_time_to_impact_ms, usable_reaction_time_ms,
                              DEFAULT_LEG_LENGTH_M, EARLY_WINDOW_FRAMES)
from capture_point import (compute_xcom, compute_xcom_deficit, bos_half_width_for_direction,
                             find_first_crossing_frame)
from torque_magnitude import compute_torque, DEFAULT_PATIENT_MASS_KG
from torque_profile import generate_torque_profile, PLACEHOLDER_SAFETY_CEILING_NM
from recovery_strategy import select_strategy
from cause_context_classifier import classify_cause_context
from stance_leg_classifier import classify_stance_leg


def run_layer_b(trial, mass_kg=DEFAULT_PATIENT_MASS_KG, leg_length_m=DEFAULT_LEG_LENGTH_M):
    """
    trial: a paper_implementation.data.Trial with onset/impact set (fall trial).
    Returns a dict with every intermediate + final output, for inspection/testing.
    """
    if trial.onset is None or trial.impact is None:
        raise ValueError("run_layer_b requires a fall trial with onset/impact labels")

    onset, impact = int(trial.onset), int(trial.impact)

    direction = classify_direction(trial.acc_m, trial.gyr_m, trial.pitch, trial.roll, onset, impact)

    com_pos, com_vel = estimate_com(trial.vv)
    v0 = max(com_vel[onset], 0.0)
    gyr_early_window = trial.gyr_m[onset:onset + EARLY_WINDOW_FRAMES]
    gyr_early = gyr_early_window.max() if len(gyr_early_window) else 0.0
    time_to_impact_ms = estimate_time_to_impact_ms(v0, gyr_early)
    # Real usable actuation window: the estimator itself needs EARLY_WINDOW_FRAMES
    # (150ms) of causal data past onset before it can be computed -- see
    # time_to_impact.py docstring. Torque profile (1.6) is sized to this, not the
    # full time_to_impact_ms, since that's what a real controller would actually have.
    reaction_time_ms = usable_reaction_time_ms(time_to_impact_ms)

    # XCoM deficit is evaluated across the whole onset->impact window (fixed the
    # earlier onset-only bug where the stubbed BoS was almost never breached, see
    # results/1.4_capture_point_xcom.txt), using a direction-aware BoS half-width
    # (Step 4 -- see capture_point.py for why this is direction-CONDITIONED, not
    # spatially asymmetric, given this model's vertical-proxy CoM). bos_center is
    # anchored to the pre-fall CoM position.
    window = slice(onset, impact + 1)
    bos_center = com_pos[onset]
    bos_half_width_m = bos_half_width_for_direction(direction)
    xcom_position_series = compute_xcom(com_pos[window], com_vel[window], leg_length_m=leg_length_m)
    xcom_deficit_series = compute_xcom_deficit(xcom_position_series, bos_center=bos_center,
                                                 bos_half_width_m=bos_half_width_m)
    peak_idx = int(np.argmax(xcom_deficit_series))
    xcom_position = xcom_position_series[peak_idx]
    xcom_deficit = xcom_deficit_series[peak_idx]

    # Causal re-evaluation (Step 4): deficit at each frame is already only a
    # function of past data; find_first_crossing_frame reports the first frame a
    # real-time controller could actually have detected a breach at, vs. the
    # post-hoc peak above which requires knowing the whole window in advance.
    first_crossing_idx = find_first_crossing_frame(xcom_deficit_series)
    if first_crossing_idx is not None:
        first_crossing_lead_time_ms = (impact - (onset + first_crossing_idx)) / 100.0 * 1000.0
    else:
        first_crossing_lead_time_ms = None  # never breaches the BoS within this window

    # torque is UNCLAMPED here (impulse-momentum formula, Step 5) -- clamping happens
    # in generate_torque_profile below, and the unclamped total is what 1.7 needs to
    # decide whether torque-only correction is even sufficient. Uses reaction_time_ms
    # (already net of the TTI estimator's own 150ms decision delay), not the raw
    # time_to_impact_ms, since that's the actual correction window available.
    torque_hip, torque_knee = compute_torque(xcom_deficit, reaction_time_ms, mass_kg=mass_kg,
                                               lever_arm_m=leg_length_m)
    required_torque_total_nm = torque_hip + torque_knee
    t_ms_hip, torque_profile_hip = generate_torque_profile(torque_hip, reaction_time_ms)
    t_ms_knee, torque_profile_knee = generate_torque_profile(torque_knee, reaction_time_ms)

    strategy = select_strategy(required_torque_total_nm, direction,
                                safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM)

    # Phase 2 (2.2 feeds 2.1): cause/context uses the pre-onset window, stance leg
    # is only inferable when that context implies an active gait cycle.
    cause_context = classify_cause_context(trial.acc_m, onset)
    stance_leg = classify_stance_leg(trial.roll, onset, cause_context)

    return {
        "direction": direction,
        "cause_context": cause_context,
        "stance_leg": stance_leg,
        "time_to_impact_ms": time_to_impact_ms,
        "reaction_time_ms": reaction_time_ms,
        "com_position": com_pos[onset],
        "com_velocity": com_vel[onset],
        "xcom_position": xcom_position,
        "xcom_deficit": xcom_deficit,
        "bos_half_width_m": bos_half_width_m,
        "first_crossing_lead_time_ms": first_crossing_lead_time_ms,
        "torque_hip_Nm": torque_hip,
        "torque_knee_Nm": torque_knee,
        "required_torque_total_nm": required_torque_total_nm,
        "torque_profile_hip": (t_ms_hip, torque_profile_hip),
        "torque_profile_knee": (t_ms_knee, torque_profile_knee),
        "strategy": strategy,
    }


if __name__ == "__main__":
    from data import load_trial

    samples = [(6, 20, 1), (6, 21, 1), (6, 22, 1), (6, 30, 1), (6, 33, 1)]
    for subject, task, trial_id in samples:
        t = load_trial(subject, task, trial_id)
        if t is None or t.onset is None:
            print(f"S{subject:02d} T{task} trial{trial_id}: could not load, skipping")
            continue
        result = run_layer_b(t)
        print(f"S{subject:02d} T{task} trial{trial_id}: "
              f"direction={result['direction']:<8} "
              f"tti={result['time_to_impact_ms']:6.1f}ms "
              f"xcom_deficit={result['xcom_deficit']:.4f}m "
              f"strategy={result['strategy']:<12} "
              f"torque(hip/knee)={result['torque_hip_Nm']:.2f}/{result['torque_knee_Nm']:.2f} Nm")
