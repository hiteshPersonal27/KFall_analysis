"""
Shared data layer for the KFall paper baseline reproductions (threshold, SVM,
ConvLSTM). Reuses the validated pipeline from
../fall_pattern_analysis/paper_threshold_validation/analyze_pattern.py for
signal computation (5 Hz low-pass, ACC_M, world-frame VV) and label loading,
so all three algorithms operate on exactly the same signals already validated
elsewhere in this repo.

Run standalone for a sanity check:
  python3 paper_implementation/data.py
"""

import os
import sys
import json
import random

import numpy as np
import pandas as pd

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "fall_pattern_analysis", "paper_threshold_validation")
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, lowpass_filter, FALL_TASK_IDS, ADL_TASK_IDS, FS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_JSON = os.path.join(SCRIPT_DIR, "split.json")

WINDOW_WIDTH = 50   # frames (0.5 s @ 100 Hz)
WINDOW_STRIDE = 1
SPLIT_SEED = 42
TEST_FRACTION = 0.2  # ~20% of subjects held out for test


# ----------------------------------------------------------------------------- #
# Subject-level split
# ----------------------------------------------------------------------------- #
def make_subject_split(seed=SPLIT_SEED, test_fraction=TEST_FRACTION, force=False):
    """
    Deterministic 80/20 train/test split BY SUBJECT (a subject never appears in
    both sets). Persisted to split.json so threshold.py, svm_model.py and
    convlstm_model.py all evaluate on the identical held-out subjects.
    """
    if os.path.exists(SPLIT_JSON) and not force:
        with open(SPLIT_JSON) as f:
            return json.load(f)

    subjects = discover_subjects()
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    test_subjects = sorted(shuffled[:n_test])
    train_subjects = sorted(shuffled[n_test:])

    split = {"seed": seed, "train_subjects": train_subjects, "test_subjects": test_subjects}
    with open(SPLIT_JSON, "w") as f:
        json.dump(split, f, indent=2)
    return split


# ----------------------------------------------------------------------------- #
# Trial loading
# ----------------------------------------------------------------------------- #
class Trial:
    """One loaded, signal-computed trial (fall or ADL)."""

    __slots__ = ("subject", "task", "trial", "is_fall", "n_frames",
                 "acc_m", "gyr_m", "vv", "pitch", "roll", "yaw",
                 "onset", "impact")

    def __init__(self, subject, task, trial, is_fall, acc_m, gyr_m, vv,
                 pitch, roll, yaw, onset=None, impact=None):
        self.subject = subject
        self.task = task
        self.trial = trial
        self.is_fall = is_fall
        self.n_frames = len(acc_m)
        self.acc_m = acc_m
        self.gyr_m = gyr_m
        self.vv = vv
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.onset = onset
        self.impact = impact


def load_trial(subject, task, trial):
    """
    Load one trial and compute derived signals via the validated pipeline.
    Returns None if the sensor file or (for falls) label info is missing.
    """
    df = load_sensor_data(subject, task, trial)
    if df is None:
        return None

    is_fall = task in FALL_TASK_IDS
    onset = impact = None
    if is_fall:
        df_label = load_labels(subject)
        if df_label is None:
            return None
        info = get_fall_label_info(df_label, task, trial)
        if info is None:
            return None
        onset, impact = info["onset"], info["impact"]

    acc_m, vv = compute_signals(df)
    gyr_m = np.sqrt(
        lowpass_filter(df["GyrX"].values) ** 2
        + lowpass_filter(df["GyrY"].values) ** 2
        + lowpass_filter(df["GyrZ"].values) ** 2
    )
    pitch = df["EulerY"].values
    roll = df["EulerX"].values
    yaw = df["EulerZ"].values

    return Trial(subject, task, trial, is_fall, acc_m, gyr_m, vv, pitch, roll, yaw,
                 onset=onset, impact=impact)


def iter_subject_trials(subject):
    """Yield every loadable trial (fall + ADL, trials 1-5) for one subject."""
    for task in sorted(FALL_TASK_IDS + ADL_TASK_IDS):
        for trial_id in range(1, 6):
            t = load_trial(subject, task, trial_id)
            if t is not None:
                yield t


# ----------------------------------------------------------------------------- #
# Windowing (used by SVM / ConvLSTM)
# ----------------------------------------------------------------------------- #
class Window:
    """One labeled sliding window, tagged with the raw frame index it ends on."""

    __slots__ = ("trial", "start", "end", "label")

    def __init__(self, trial, start, end, label):
        self.trial = trial          # parent Trial
        self.start = start          # inclusive frame index (0-based, into trial arrays)
        self.end = end              # exclusive frame index
        self.label = label          # "fall" | "adl"


def iter_windows(trial, width=WINDOW_WIDTH, stride=WINDOW_STRIDE):
    """
    Slide a width-frame window across a trial.

    Fall trials: a window is labeled "fall" only if it lies fully within
    [onset, impact] (the paper's pre-impact framing) -- SAME rule as
    originally, for trials where that's possible. Windows before onset or
    after impact are excluded (not "fall", not "adl").

    EDGE CASES, surgically handled: two distinct ways the strict rule can
    yield ZERO windows for a fall trial instead of one:
      (a) impact-onset < width (< 500ms at 100Hz, width=50) -- no window
          CAN be fully contained, period. 21/439 fall trials (4.8%) hit this.
      (b) impact-onset >= width, but the (few) valid start positions don't
          happen to land on the stride grid -- e.g. duration=50 has exactly
          ONE valid start (start=onset), and if `onset` isn't a multiple of
          the test stride, a coarse stride (e.g. 5) skips it entirely even
          though width-wise a window fits. Found 8 additional trials hitting
          this at stride=5 (durations 50-52) that (a) alone didn't catch --
          discovered by re-checking actual yielded counts after the first
          fix, not assumed away.
    Both silently DROP the trial from every downstream window-based
    evaluation (aggregate_to_trials groupby) -- not counted as a missed
    detection, not counted at all -- inflating every reported sensitivity
    number, asymmetrically (only fall trials are affected; ADL trials have
    no such containment rule).

    Fix: run the strict rule first; if and ONLY if it yields zero windows,
    fall back to a single window anchored to end exactly at impact
    (capturing the most information-rich part -- everything up to and
    including impact). This is deliberately NOT a blanket loosening of the
    containment rule for every trial (an earlier, overly broad version of
    this fix relaxed "start>=onset" to "end>onset" for ALL trials, which
    roughly DOUBLED the window count even for normal, long-margin trials --
    e.g. one measured case went from 54 windows to 103 for the same trial --
    systematically adding many pre-onset-start windows dominated by
    near-impact motion, inflating sensitivity far beyond what the edge-case
    fix alone justifies). Checking the actual yielded count (not just
    comparing duration to width) catches both edge cases (a) and (b)
    uniformly, regardless of stride. Confirmed: min(impact) across all 2319
    fall trials in the dataset is 53 frames, so the fallback window's
    start = impact-width is never negative.

    ADL trials: every window is labeled "adl" (unaffected either way).
    """
    n = trial.n_frames
    if n < width:
        return

    if trial.is_fall:
        # Strict full-containment rule, exactly as originally implemented --
        # buffered (not yielded directly) so we can detect the zero-window
        # edge cases and fall back, regardless of WHY they're zero.
        strict_windows = []
        for start in range(0, n - width + 1, stride):
            end = start + width
            if start >= trial.onset and end <= trial.impact:
                strict_windows.append((start, end))

        if strict_windows:
            for start, end in strict_windows:
                yield Window(trial, start, end, "fall")
        else:
            # Edge case: strict rule found nothing (too-short duration, or a
            # coarse stride skipping the only valid position(s)). Single
            # fallback window anchored to end exactly at impact.
            end = trial.impact
            start = end - width
            if start >= 0:
                yield Window(trial, start, end, "fall")
    else:
        for start in range(0, n - width + 1, stride):
            end = start + width
            yield Window(trial, start, end, "adl")


def window_channel(window, channel):
    """Extract one named channel ('acc_m'|'gyr_m'|'vv'|'pitch'|'roll'|'yaw') as an array."""
    arr = getattr(window.trial, channel)
    return arr[window.start:window.end]


def window_raw9(window, df_cache=None):
    """
    Raw 9-channel (AccX/Y/Z, GyrX/Y/Z, EulerX/Y/Z) slice for ConvLSTM, shape (width, 9).
    Re-reads the sensor CSV (Trial only stores derived signals) -- cached per trial
    key to avoid re-reading the same file for every window in that trial.
    """
    if df_cache is None:
        df_cache = {}
    key = (window.trial.subject, window.trial.task, window.trial.trial)
    if key not in df_cache:
        df_cache[key] = load_sensor_data(*key)
    df = df_cache[key]
    cols = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]
    return df[cols].values[window.start:window.end]


# ----------------------------------------------------------------------------- #
# Sanity check
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    split = make_subject_split()
    print(f"Subject split (seed={split['seed']}): "
          f"{len(split['train_subjects'])} train, {len(split['test_subjects'])} test")
    print(f"  train: {split['train_subjects']}")
    print(f"  test:  {split['test_subjects']}")

    for label, subjects in (("TRAIN", split["train_subjects"]), ("TEST", split["test_subjects"])):
        n_fall_trials = n_adl_trials = n_fall_windows = n_adl_windows = 0
        for subj in subjects:
            for t in iter_subject_trials(subj):
                if t.is_fall:
                    n_fall_trials += 1
                else:
                    n_adl_trials += 1
                for w in iter_windows(t):
                    if w.label == "fall":
                        n_fall_windows += 1
                    else:
                        n_adl_windows += 1
        print(f"\n{label}: {len(subjects)} subjects")
        print(f"  fall trials: {n_fall_trials}   ADL trials: {n_adl_trials}")
        print(f"  fall windows: {n_fall_windows}   ADL windows: {n_adl_windows}   "
              f"(class balance fall:adl = 1:{n_adl_windows / max(n_fall_windows,1):.1f})")
        if label == "TEST":
            print(f"  (paper's test set: 444 fall files / 507 ADL files -- "
                  f"file counts, not window counts; compare trial counts above in ballpark)")
