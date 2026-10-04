"""
1.5 -- Torque Magnitude per Joint

REPLACED (see docs/LayerB_Implementation_Plan.md fix plan, Step 5 -- the
original `mass * deficit * gain` had a free `gain` constant picked AFTER
looking at how often it saturated the safety ceiling -- leakage one level
deeper). A first replacement attempt (`mass*deficit/time/actuator_latency`,
two independent time divisors) was tried and REJECTED here before ever
touching the dataset: a toy sanity check with a representative deficit
(~0.5m, not fit to this dataset) gave torques in the hundreds of Nm --
physically implausible (human joint torques rarely exceed ~150-300Nm even at
maximal voluntary contraction) regardless of what the frozen constant is set
to, because dividing by two independent time factors double-penalizes a short
window in a way real F=ma kinematics doesn't.

Final formula -- same kinematic template already used by time_to_impact.py's
own free-fall model (`h = 0.5*a*t^2`), applied to closing the XCoM deficit
gap within the available correction window instead of a fall height:

    effective_time_s = max(available_time_ms/1000 - ACTUATOR_LATENCY_S, epsilon)
    required_force_N  = 2 * mass_kg * xcom_deficit_m / effective_time_s^2
    required_torque_Nm = required_force_N * lever_arm_m

`available_time_ms` is the caller-supplied usable reaction window (pipeline.py
passes `reaction_time_ms` from time_to_impact.py, which already accounts for
the 150ms causal decision delay). ACTUATOR_LATENCY_S further subtracts the
actuator's own physical ramp-up time from that budget -- a single,
dimensionally coherent time deduction, not a second independent divisor.

DISCIPLINE, DO NOT VIOLATE: ACTUATOR_LATENCY_S was picked ONCE, from a
plausible motor-response-time ballpark, sanity-checked only against a generic
representative toy deficit (not this dataset's actual statistics), and FROZEN
BEFORE this formula was ever run against the full dataset to see what
fraction of trials saturate the safety ceiling. Whatever that saturation rate
turns out to be is reported as a finding (results/1.5_torque_magnitude.txt),
not iterated against.
"""

DEFAULT_PATIENT_MASS_KG = 70.0   # Layer C stub, pending calibration entry
HIP_KNEE_SPLIT = (0.70, 0.30)    # placeholder assumption, pending biomech literature

# FROZEN placeholder, same external-dependency status as the 40Nm safety ceiling
# (torque_profile.py) -- a plausible electric-actuator response-time ballpark, not
# sourced from a specific datasheet (no real actuator hardware selected yet, per
# docs/Fall_Prevention_Actuator_Input_Pipeline.md Layer C status).
ACTUATOR_LATENCY_S = 0.10
MIN_EFFECTIVE_TIME_S = 0.02   # floor to avoid a divide-by-near-zero blowup, not a tuned value


def compute_torque(xcom_deficit_m, available_time_ms, mass_kg=DEFAULT_PATIENT_MASS_KG,
                    lever_arm_m=0.9, actuator_latency_s=ACTUATOR_LATENCY_S):
    """
    xcom_deficit_m: output of capture_point.compute_xcom_deficit (m, >= 0).
    available_time_ms: usable correction window (pass time_to_impact.usable_reaction_time_ms's
        output, i.e. already net of the direction/TTI estimator's own decision delay).
    mass_kg: patient body mass (Layer C stub for now).
    lever_arm_m: distance from pivot to CoM (pass DEFAULT_LEG_LENGTH_M from
        time_to_impact.py for consistency with capture_point.py's Hof model).

    Returns (torque_hip_Nm, torque_knee_Nm), both >= 0, UNCLAMPED -- the safety
    ceiling is applied downstream (torque_profile.py for the actual command,
    recovery_strategy.py for the step-vs-torque decision, which needs the
    unclamped value to know whether torque-only correction is even sufficient).
    """
    effective_time_s = max(available_time_ms / 1000.0 - actuator_latency_s, MIN_EFFECTIVE_TIME_S)
    required_force_n = 2.0 * mass_kg * xcom_deficit_m / (effective_time_s ** 2)
    total_torque = required_force_n * lever_arm_m
    hip_frac, knee_frac = HIP_KNEE_SPLIT
    return total_torque * hip_frac, total_torque * knee_frac


if __name__ == "__main__":
    # Representative toy sanity check (deficit ~0.5m, NOT this dataset's actual
    # statistics) -- run BEFORE ever evaluating against real KFall trials.
    for deficit in (0.1, 0.2, 0.5):
        torque_hip, torque_knee = compute_torque(xcom_deficit_m=deficit, available_time_ms=600.0)
        print(f"deficit={deficit}m, available=600ms -> "
              f"torque_hip={torque_hip:.2f} Nm, torque_knee={torque_knee:.2f} Nm, "
              f"total={torque_hip+torque_knee:.2f} Nm")
