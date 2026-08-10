"""
Data layer for the CWT + 2D-CNN experiment (see ../../docs/CWT_2DCNN_Implementation_Plan.md).

Reuses paper_implementation/data.py's windowing (50-frame windows) and the
existing fixed subject split (paper_implementation/split.json) so results are
directly comparable to the baseline raw-1D ConvLSTM. Adds one new
responsibility: transforming each (50, 9) raw window into a (9, num_scales,
50) scalogram via the Continuous Wavelet Transform (CWT), applied
independently per channel.

CWT implementation: GPU-batched complex Morlet transform (gpu_cwt.py), not
pywt. pywt.cwt() processes one window at a time on CPU (~14ms/window
measured -- the test-set pass alone took ~20 minutes), but a CWT is
fundamentally a convolution with a bank of scaled wavelet kernels, which
torch.conv1d batches trivially across thousands of windows on GPU
(~0.037ms/window measured, ~380x faster). See gpu_cwt.py's docstring for
the wavelet math and why exact numerical parity with pywt doesn't matter
here (same wavelet family, this project trains/evaluates entirely on its
own output).

Scales: log-spaced to cover ~1-45 Hz (up to just under the 50 Hz Nyquist
limit for a 100 Hz signal), NUM_SCALES=24 (within the plan's suggested
16-32 range).

Precomputed CWT outputs are cached to disk (.npz) keyed by (subjects, stride,
max_adl_windows, wavelet params) so repeated epochs/runs don't recompute.

Run standalone for the visual sanity check (go/no-go checkpoint before any
model code):
  python3 experiments/cwt_lstm/data.py --sanity-check
"""

import os
import sys
import hashlib
import importlib.util

import numpy as np
import torch

# Load paper_implementation/data.py under a distinct module name (both files
# are named "data.py"; a plain `sys.path.insert` + `import data` would self-
# collide with this very module once it's imported as top-level "data").
PAPER_IMPL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "paper_implementation")
_spec = importlib.util.spec_from_file_location("_paper_impl_data", os.path.join(PAPER_IMPL_DIR, "data.py"))
_paper_data = importlib.util.module_from_spec(_spec)
sys.path.insert(0, PAPER_IMPL_DIR)  # paper_implementation/data.py itself imports from analyze_pattern.py
_spec.loader.exec_module(_paper_data)

make_subject_split = _paper_data.make_subject_split
iter_subject_trials = _paper_data.iter_subject_trials
iter_windows = _paper_data.iter_windows
window_raw9 = _paper_data.window_raw9
FS = _paper_data.FS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gpu_cwt import GPUCWT, scale_for_frequency, W0  # noqa: E402

CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

WAVELET = f"morlet-gpu-w0={W0}"
NUM_SCALES = 24
FREQ_MIN, FREQ_MAX = 1.0, 45.0   # Hz; stays under the 50 Hz Nyquist limit
WINDOW_WIDTH = 50
RAW_CHANNELS = 9
TRANSFORM_BATCH = 4096            # windows per GPU batch during CWT transform


def make_scales(num_scales=NUM_SCALES, freq_min=FREQ_MIN, freq_max=FREQ_MAX, fs=FS):
    """Log-spaced scales covering [freq_min, freq_max] Hz for our GPU Morlet kernel."""
    freqs = np.geomspace(freq_max, freq_min, num_scales)  # high freq -> low freq
    scales = np.array([scale_for_frequency(f, fs) for f in freqs])
    return scales, freqs


SCALES, SCALE_FREQS_HZ = make_scales()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cwt = GPUCWT(SCALES, fs=FS, device=_device)


def cwt_batch(raw9_batch):
    """raw9_batch: (N, 50, 9) array. Returns (N, 9, NUM_SCALES, 50) magnitude scalograms."""
    x = np.transpose(raw9_batch, (0, 2, 1))  # (N, 9, 50)
    return _cwt.transform_numpy(x)


def cwt_window(raw9):
    """Single-window convenience wrapper (used by the sanity check plot)."""
    return cwt_batch(raw9[None, :, :])[0]


def _cache_key(subjects, stride, max_adl_windows):
    raw = f"{sorted(subjects)}|{stride}|{max_adl_windows}|{WAVELET}|{NUM_SCALES}|{FREQ_MIN}-{FREQ_MAX}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


# Each scalogram is (9, NUM_SCALES, 50) -- NUM_SCALES=24x bigger per window
# than the baseline raw-1D ConvLSTM's (9, 50) window. Blindly matching the
# baseline's full uncapped stride=1 setting (~1.76M windows) here would need
# ~74GB just for the training array -- this OOM'd the machine once already.
# Guard against repeating that: refuse to allocate more than MAX_MEMORY_GB
# without an explicit override, and store as float16 (halves memory; the
# scalogram magnitudes don't need float32 precision for training).
STORAGE_DTYPE = np.float16
BYTES_PER_WINDOW = RAW_CHANNELS * NUM_SCALES * WINDOW_WIDTH * np.dtype(STORAGE_DTYPE).itemsize
DEFAULT_MAX_MEMORY_GB = 8.0


def estimate_memory_gb(n_windows):
    return n_windows * BYTES_PER_WINDOW / (1024 ** 3)


def build_cwt_dataset(subjects, stride, max_adl_windows=None, seed=42, use_cache=True, cache_tag="",
                       max_memory_gb=DEFAULT_MAX_MEMORY_GB):
    """
    Returns (X: (N, 9, NUM_SCALES, 50) float16, y: (N,) int64, meta: DataFrame).
    Precomputed and disk-cached; pass use_cache=False to force recompute.

    Raises MemoryError (before allocating anything) if the requested window
    count would exceed max_memory_gb -- lower max_adl_windows / raise stride,
    or pass a higher max_memory_gb explicitly if you've checked available RAM.
    """
    import pandas as pd

    key = _cache_key(subjects, stride, max_adl_windows) + (f"_{cache_tag}" if cache_tag else "")
    cache_path = os.path.join(CACHE_DIR, f"cwt_{key}.npz")
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        meta_df = pd.DataFrame(data["meta"].tolist())
        return data["X"], data["y"], meta_df

    fall_windows, adl_windows = collect_windows(subjects, stride)
    if max_adl_windows is not None and len(adl_windows) > max_adl_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adl_windows), size=max_adl_windows, replace=False)
        adl_windows = [adl_windows[i] for i in idx]

    windows = [(w, 1) for w in fall_windows] + [(w, 0) for w in adl_windows]
    n = len(windows)

    est_gb = estimate_memory_gb(n)
    if est_gb > max_memory_gb:
        raise MemoryError(
            f"build_cwt_dataset: {n} windows would need ~{est_gb:.1f} GB "
            f"(> max_memory_gb={max_memory_gb}). Lower max_adl_windows, raise "
            f"stride, or pass max_memory_gb explicitly if you've checked "
            f"available RAM first (e.g. via `free -h`)."
        )
    print(f"  ({n} windows, ~{est_gb:.2f} GB estimated)")

    X = np.empty((n, RAW_CHANNELS, NUM_SCALES, WINDOW_WIDTH), dtype=STORAGE_DTYPE)
    y = np.empty(n, dtype=np.int64)
    meta = []
    df_cache = {}

    # Two-pass: gather all raw9 arrays first (cheap), then transform in GPU
    # batches (the expensive step), rather than one window at a time.
    raw9_all = np.empty((n, WINDOW_WIDTH, RAW_CHANNELS), dtype=np.float32)
    for i, (w, label) in enumerate(windows):
        raw9_all[i] = window_raw9(w, df_cache=df_cache)
        y[i] = label
        t = w.trial
        meta.append({"subject": t.subject, "task": t.task, "trial": t.trial,
                     "is_fall": t.is_fall, "end_frame": w.end,
                     "impact_frame": t.impact if t.is_fall else None})

    for start in range(0, n, TRANSFORM_BATCH):
        end = min(start + TRANSFORM_BATCH, n)
        X[start:end] = cwt_batch(raw9_all[start:end]).astype(STORAGE_DTYPE)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, meta=np.array(meta, dtype=object))
    return X, y, pd.DataFrame(meta)


# ----------------------------------------------------------------------------- #
# Visual sanity check (go/no-go checkpoint per the plan -- run before any
# model code)
# ----------------------------------------------------------------------------- #
def sanity_check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = make_subject_split()
    train_subjects = split["train_subjects"]

    # Find one clear fall window (the one ending closest to impact -- the
    # window closest to impact within the valid [onset, impact] fall-labeled
    # range, since a full 50-frame window's midpoint is necessarily ~25
    # frames before its end, i.e. before impact) and one clear jog window
    # (D08 quick jog).
    fall_raw9 = fall_trial_desc = None
    jog_raw9 = jog_trial_desc = None
    for subj in train_subjects:
        for t in iter_subject_trials(subj):
            if t.is_fall and fall_raw9 is None:
                best_w = None
                for w in iter_windows(t, stride=1):
                    if best_w is None or w.end > best_w.end:
                        best_w = w
                if best_w is not None:
                    fall_raw9 = window_raw9(best_w)
                    fall_trial_desc = (f"SA{subj:02d} F{t.task-19:02d} trial{t.trial} "
                                        f"(window ending {t.impact - best_w.end} frames before impact)")
            if t.task == 8 and jog_raw9 is None:  # D08 jog quick
                for w in iter_windows(t, stride=1):
                    jog_raw9 = window_raw9(w)
                    jog_trial_desc = f"SA{subj:02d} D08 (jog quick)"
                    break
        if fall_raw9 is not None and jog_raw9 is not None:
            break

    fall_scalo = cwt_window(fall_raw9)
    jog_scalo = cwt_window(jog_raw9)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, (label, scalo, raw9, desc) in enumerate([
        ("FALL", fall_scalo, fall_raw9, fall_trial_desc),
        ("JOG (ADL)", jog_scalo, jog_raw9, jog_trial_desc),
    ]):
        for col, ch_idx, ch_name in [(0, 0, "AccX"), (1, 3, "GyrX")]:
            ax = axes[row, col]
            im = ax.imshow(scalo[ch_idx], aspect="auto", origin="lower", cmap="viridis",
                            extent=[0, WINDOW_WIDTH, SCALE_FREQS_HZ[-1], SCALE_FREQS_HZ[0]])
            ax.set_title(f"{label} -- {ch_name}\n{desc}", fontsize=9)
            ax.set_xlabel("frame")
            ax.set_ylabel("freq (Hz)")
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "sanity_check_scalograms.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity-check scalogram comparison to {out_path}")
    print("Inspect visually: does FALL show a broad, low-frequency-dominant blob,")
    print("while JOG shows a repeating narrow-band striped pattern? If not distinguishable,")
    print("treat this as a no-go signal before investing in model training.")


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        sanity_check()
    else:
        print(f"CWT config: {WAVELET}, num_scales={NUM_SCALES}, "
              f"freq range={FREQ_MIN}-{FREQ_MAX} Hz, device={_device}")
        print(f"Scale->freq mapping (Hz): {np.round(SCALE_FREQS_HZ, 1)}")
        print("Run with --sanity-check to generate the fall-vs-ADL scalogram comparison.")
