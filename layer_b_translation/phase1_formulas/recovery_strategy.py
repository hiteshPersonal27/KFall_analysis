"""
1.7 -- Recovery Strategy Selection (rule table, not ML)

REDESIGNED (see docs/LayerB_Implementation_Plan.md fix plan, Step 5). The
"step vs. torque-only" decision no longer comes from an independently-fit
xcom_deficit percentile threshold (the old `xcom_deficit_high_m=1.54` was set
to this dataset's own 90th percentile, calibrated and evaluated on the same
data -- not grounded in anything physical). It now falls out of whether the
REQUIRED (unclamped) torque from 1.5 would exceed the safety ceiling: if
torque-only correction genuinely can't deliver enough force to close the XCoM
deficit within the safety limit, stepping is the only option left -- this ties
1.5 and 1.7 together via the same physical constraint instead of two
independently-calibrated numbers.

Direction-based selection (below the ceiling) still reads from
config/recovery_strategy_thresholds.yaml, per the plan doc's requirement that
this stay editable without touching logic.
"""

import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config",
                            "recovery_strategy_thresholds.yaml")

VALID_STRATEGIES = {"hip_torque", "ankle_torque", "stepping"}


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def select_strategy(required_torque_total_nm, direction, safety_ceiling_nm, config=None):
    """
    required_torque_total_nm: the UNCLAMPED total torque (hip+knee, before the
        70/30 split) that torque_magnitude.compute_torque would need to deliver
        to fully close the XCoM deficit -- pass the sum of its two return values.
    direction: "forward" | "backward" | "lateral" (from direction_classifier).
    safety_ceiling_nm: torque_profile.PLACEHOLDER_SAFETY_CEILING_NM (passed
        explicitly rather than imported, so this module doesn't reach across
        into torque_profile.py's namespace for a single constant).

    Returns strategy in VALID_STRATEGIES: "stepping" when torque-only
    correction can't deliver required_torque_total_nm within the safety
    ceiling; otherwise a direction-based torque strategy from config.
    """
    if config is None:
        config = load_config()

    if required_torque_total_nm > safety_ceiling_nm:
        return "stepping"

    strategy = config["direction_strategy"].get(direction)
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown direction '{direction}' or invalid strategy '{strategy}' in config")
    return strategy


if __name__ == "__main__":
    print(select_strategy(required_torque_total_nm=10.0, direction="forward", safety_ceiling_nm=40.0))
    print(select_strategy(required_torque_total_nm=10.0, direction="backward", safety_ceiling_nm=40.0))
    print(select_strategy(required_torque_total_nm=60.0, direction="forward", safety_ceiling_nm=40.0))
