"""
Step 1 -- Shared train/test subject split for layer_b_translation, reusing
paper_implementation's existing seeded 80/20 subject-level split (same
methodology already used to evaluate Threshold/SVM/ConvLSTM, so this isn't a
new convention -- see paper_implementation/data.py::make_subject_split).

From here on, every layer_b_translation calibration step must fit constants
on train_subjects() only and report accuracy/correlation on test_subjects()
only -- see docs/LayerB_Implementation_Plan.md fix plan.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "paper_implementation"))
sys.path.insert(0, os.path.join(_HERE, "..", "fall_pattern_analysis", "paper_threshold_validation"))
from data import make_subject_split, load_trial  # noqa: E402
from analyze_pattern import FALL_TASK_IDS  # noqa: E402


def get_split():
    return make_subject_split()


def train_subjects():
    return get_split()["train_subjects"]


def test_subjects():
    return get_split()["test_subjects"]


def iter_fall_trials(subjects):
    """Yield every loadable fall Trial (onset/impact labeled) for the given subjects."""
    for subject in subjects:
        for task in FALL_TASK_IDS:
            for trial_id in range(1, 6):
                t = load_trial(subject, task, trial_id)
                if t is not None and t.onset is not None:
                    yield t


if __name__ == "__main__":
    split = get_split()
    n_train = len(list(iter_fall_trials(split["train_subjects"])))
    n_test = len(list(iter_fall_trials(split["test_subjects"])))
    print(f"train_subjects={len(split['train_subjects'])} ({n_train} fall trials)")
    print(f"test_subjects={len(split['test_subjects'])} ({n_test} fall trials)")
