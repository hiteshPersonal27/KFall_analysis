"""
Validates each Layer B Phase 1 formula against real KFall trials + the sanity
bounds documented in docs/LayerB_Implementation_Plan.md's Success Criteria.

Run: python3 -m pytest layer_b_translation/tests/
"""

import os
import sys

import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TESTS_DIR, ".."))
sys.path.insert(0, os.path.join(TESTS_DIR, "..", "phase1_formulas"))
sys.path.insert(0, os.path.join(TESTS_DIR, "..", "..", "paper_implementation"))

from pipeline import run_layer_b  # noqa: E402
from data import load_trial  # noqa: E402
from torque_profile import generate_torque_profile, PLACEHOLDER_SAFETY_CEILING_NM  # noqa: E402
from recovery_strategy import select_strategy, VALID_STRATEGIES  # noqa: E402

FALL_SAMPLES = [(6, 20, 1), (6, 21, 1), (6, 22, 1), (6, 30, 1), (6, 33, 1), (7, 20, 1)]


def _load_samples():
    trials = []
    for subject, task, trial_id in FALL_SAMPLES:
        t = load_trial(subject, task, trial_id)
        if t is not None and t.onset is not None:
            trials.append(t)
    return trials


@pytest.fixture(scope="module")
def fall_trials():
    trials = _load_samples()
    if not trials:
        pytest.skip("KFall sensor/label data not available in this environment")
    return trials


def test_pipeline_runs_end_to_end_without_errors(fall_trials):
    for t in fall_trials:
        result = run_layer_b(t)
        assert result["direction"] in {"forward", "backward", "lateral"}
        assert result["strategy"] in VALID_STRATEGIES
        assert np.isfinite(result["time_to_impact_ms"])
        assert result["time_to_impact_ms"] >= 0


def test_time_to_impact_within_plausible_range_of_empirical_lead_time(fall_trials):
    # Empirical lead times (paper_implementation/results/comparison.md) range
    # roughly 220-350ms mean, with std up to ~140ms across all three baselines.
    # Flag (not fail outright) if the formula diverges wildly -- here we assert
    # a generous bound (0-1500ms) since this is a coarse kinematic proxy, not a
    # calibrated fit; a tighter bound would need real leg-length/mass per subject.
    for t in fall_trials:
        result = run_layer_b(t)
        assert 0 <= result["time_to_impact_ms"] <= 1500


def test_torque_profile_never_exceeds_safety_ceiling(fall_trials):
    for t in fall_trials:
        result = run_layer_b(t)
        for _, torque in (result["torque_profile_hip"], result["torque_profile_knee"]):
            assert torque.max() <= PLACEHOLDER_SAFETY_CEILING_NM + 1e-9


def test_torque_profile_shape_is_trapezoidal_not_square_wave():
    t_ms, torque = generate_torque_profile(peak_torque_nm=20.0, time_to_impact_ms=300.0)
    assert torque[0] == 0.0                     # starts at zero, no instantaneous jump
    assert torque.max() == pytest.approx(20.0, abs=1e-6)
    assert torque[-1] < torque.max()             # ramps back down, doesn't stay pegged


def test_recovery_strategy_high_torque_always_steps():
    assert select_strategy(100.0, "forward", safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM) == "stepping"
    assert select_strategy(100.0, "backward", safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM) == "stepping"


def test_recovery_strategy_low_torque_uses_direction_table():
    assert select_strategy(10.0, "forward", safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM) == "hip_torque"
    assert select_strategy(10.0, "backward", safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM) == "ankle_torque"
    assert select_strategy(10.0, "lateral", safety_ceiling_nm=PLACEHOLDER_SAFETY_CEILING_NM) == "ankle_torque"
