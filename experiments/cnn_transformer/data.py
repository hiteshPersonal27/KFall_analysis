"""
Data layer for the 1D-CNN + Transformer experiment (see
../../docs/CNN_Transformer_Implementation_Plan.md). Follows PreFallKD (Chi,
Liu, Hsieh, Tsao & Chan, ICASSP 2023) -- the only published lightweight-
transformer approach validated on KFall itself.

Reuses paper_implementation/data.py's raw windowing (same 50-frame windows,
same subject-level split in paper_implementation/split.json) and the SAME
raw 9-channel input the baseline ConvLSTM uses -- no CWT, no derived
features. Patchifying is a cheap reshape (unlike CWT's scalogram, which is
24x bigger per window and OOM'd the machine once -- see
experiments/cwt_lstm/README.md), so windows are stored as plain raw (50, 9)
arrays, identical to convlstm_model.py's approach, and patchified on the fly
in the model/dataset -- no disk caching needed, no new memory-safety concern.

Patch-grouping decision (open item in the plan, no PreFallKD source
available to verify against -- documented here rather than guessed silently):
  - Patch length (time): 10 frames -> 50/10 = 5 temporal chunks.
  - Patch axis (channel grouping): 3 channels per patch -> 9/3 = 3 channel
    groups. Interpreted as grouping by SENSOR MODALITY (accel xyz, gyro xyz,
    euler xyz) rather than an arbitrary/contiguous grouping, since that's the
    only channel grouping in this dataset with physical meaning matching
    "groups of 3" (this is also the natural analogue of ViT's patch grid:
    one axis = time, one axis = "which sensor").
  - Total patches N = 5 time-chunks x 3 channel-groups = 15, each patch
    flattened from (10 frames x 3 channels) = 30 raw values, linearly
    projected to the model's hidden size.
"""

import os
import sys
import importlib.util

import numpy as np

# Load paper_implementation/data.py under a distinct module name (same
# workaround as experiments/cwt_lstm/data.py -- both files are named
# "data.py", which would otherwise self-collide on import).
PAPER_IMPL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "paper_implementation")
_spec = importlib.util.spec_from_file_location("_paper_impl_data", os.path.join(PAPER_IMPL_DIR, "data.py"))
_paper_data = importlib.util.module_from_spec(_spec)
sys.path.insert(0, PAPER_IMPL_DIR)
_spec.loader.exec_module(_paper_data)

make_subject_split = _paper_data.make_subject_split
iter_subject_trials = _paper_data.iter_subject_trials
iter_windows = _paper_data.iter_windows
window_raw9 = _paper_data.window_raw9
FS = _paper_data.FS

WINDOW_WIDTH = 50
RAW_CHANNELS = 9

PATCH_LEN = 10          # frames per patch
CHANNEL_GROUP_SIZE = 3   # channels per patch (grouped by sensor modality)
N_TIME_CHUNKS = WINDOW_WIDTH // PATCH_LEN         # 5
N_CHANNEL_GROUPS = RAW_CHANNELS // CHANNEL_GROUP_SIZE  # 3
N_PATCHES = N_TIME_CHUNKS * N_CHANNEL_GROUPS            # 15
PATCH_DIM = PATCH_LEN * CHANNEL_GROUP_SIZE               # 30

# Raw column order from window_raw9: AccX,AccY,AccZ, GyrX,GyrY,GyrZ, EulerX,EulerY,EulerZ
# -- already grouped by modality in groups of 3, so channel groups are simply
# contiguous slices [0:3], [3:6], [6:9].


def patchify(raw9):
    """
    raw9: (50, 9) array (one window). Returns (N_PATCHES, PATCH_DIM) = (15, 30).

    For time-chunk t in [0, N_TIME_CHUNKS) and channel-group g in
    [0, N_CHANNEL_GROUPS): patch[t*N_CHANNEL_GROUPS + g] = raw9[t*PATCH_LEN:(t+1)*PATCH_LEN,
                                                                  g*CHANNEL_GROUP_SIZE:(g+1)*CHANNEL_GROUP_SIZE].flatten()
    """
    # (5, 10, 3, 3): (time_chunk, frame_in_chunk, channel_group, channel_in_group)
    reshaped = raw9.reshape(N_TIME_CHUNKS, PATCH_LEN, N_CHANNEL_GROUPS, CHANNEL_GROUP_SIZE)
    # -> (time_chunk, channel_group, frame_in_chunk, channel_in_group) -> flatten last two dims
    patches = reshaped.transpose(0, 2, 1, 3).reshape(N_TIME_CHUNKS * N_CHANNEL_GROUPS, PATCH_DIM)
    return patches.astype(np.float32)


def patchify_batch(raw9_batch):
    """raw9_batch: (B, 50, 9) -> (B, N_PATCHES, PATCH_DIM)."""
    B = raw9_batch.shape[0]
    reshaped = raw9_batch.reshape(B, N_TIME_CHUNKS, PATCH_LEN, N_CHANNEL_GROUPS, CHANNEL_GROUP_SIZE)
    patches = reshaped.transpose(0, 1, 3, 2, 4).reshape(B, N_TIME_CHUNKS * N_CHANNEL_GROUPS, PATCH_DIM)
    return patches.astype(np.float32)


def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


def build_raw_dataset(subjects, stride, max_adl_windows=None, seed=42, oversample_fall=1):
    """
    Returns (X: (N, 50, 9) float32 RAW windows, y: (N,) int64, meta: DataFrame).
    Patchifying happens later (cheap, on the fly) -- this mirrors
    paper_implementation/convlstm_model.py's build_raw_dataset exactly.

    oversample_fall: if >1, duplicate fall windows this many times in the
    returned arrays (PreFallKD's own imbalance-handling approach, per the
    plan's open item -- see train.py for whether/how this is applied).
    """
    import pandas as pd

    fall_windows, adl_windows = collect_windows(subjects, stride)
    if max_adl_windows is not None and len(adl_windows) > max_adl_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adl_windows), size=max_adl_windows, replace=False)
        adl_windows = [adl_windows[i] for i in idx]

    windows = [(w, 1) for w in fall_windows for _ in range(oversample_fall)] + [(w, 0) for w in adl_windows]
    X = np.zeros((len(windows), WINDOW_WIDTH, RAW_CHANNELS), dtype=np.float32)
    y = np.zeros(len(windows), dtype=np.int64)
    meta = []
    df_cache = {}
    for i, (w, label) in enumerate(windows):
        X[i] = window_raw9(w, df_cache=df_cache)
        y[i] = label
        t = w.trial
        meta.append({"subject": t.subject, "task": t.task, "trial": t.trial,
                     "is_fall": t.is_fall, "end_frame": w.end,
                     "impact_frame": t.impact if t.is_fall else None})
    return X, y, pd.DataFrame(meta)


if __name__ == "__main__":
    print(f"Patch config: PATCH_LEN={PATCH_LEN} frames, CHANNEL_GROUP_SIZE={CHANNEL_GROUP_SIZE} channels")
    print(f"N_TIME_CHUNKS={N_TIME_CHUNKS}, N_CHANNEL_GROUPS={N_CHANNEL_GROUPS}, "
          f"N_PATCHES={N_PATCHES}, PATCH_DIM={PATCH_DIM}")

    # Sanity: patchify a synthetic window and confirm round-trip shape/content.
    fake = np.arange(WINDOW_WIDTH * RAW_CHANNELS, dtype=np.float32).reshape(WINDOW_WIDTH, RAW_CHANNELS)
    patches = patchify(fake)
    print(f"patchify({fake.shape}) -> {patches.shape}  (expected ({N_PATCHES}, {PATCH_DIM}))")
    assert patches.shape == (N_PATCHES, PATCH_DIM)
    # Spot-check: patch 0 = time-chunk 0, channel-group 0 (AccX,Y,Z) frames 0-9.
    expected_patch0 = fake[0:PATCH_LEN, 0:CHANNEL_GROUP_SIZE].flatten()
    assert np.allclose(patches[0], expected_patch0), "patch 0 content mismatch"
    print("Sanity check passed: patch 0 matches raw window's first 10 frames x AccX/Y/Z.")
