"""
Data layer for the GAF 2D-encoding experiment (see
../../docs/GAF_MTF_Implementation_Plan.md).

Reuses paper_implementation/data.py's raw windowing, subject split, and
derived-signal computation (ACC_M, GYR_M, VV -- already exposed per-Trial
via the validated pipeline used throughout this project) via its
window_channel() helper. Adds one new responsibility: encoding each
channel's (50,) window slice as a Gramian Angular Summation Field (GASF)
image.

Scope decision (per the plan): GAF only (not MTF), on the 3 derived signals
central to this project (ACC_M, GYR_M, VV), not all 9 raw channels -- keeps
input size/parameter count comparable to prior experiments. Each channel's
50-length signal becomes a (50, 50) image; stacking 3 channels gives a
(3, 50, 50) input per window (same channel count as RGB, unlike CWT's
(9, 24, 50)).

GASF formula (standard, e.g. Wang & Oates 2015):
  1. Min-max rescale the signal to [-1, 1]: x~ = (2x - max(x) - min(x)) / (max(x) - min(x))
  2. Treat x~ as cos(phi): phi = arccos(x~), phi in [0, pi]
  3. GASF[i,j] = cos(phi_i + phi_j) = x~_i*x~_j - sqrt(1-x~_i^2)*sqrt(1-x~_j^2)
Implemented directly via numpy (no new dependency) -- cheap enough (a 50x50
outer-product-style computation per channel) that no GPU batching or disk
caching is needed, unlike CWT's much more expensive per-window transform.

Memory-safety guard: mirrors experiments/cwt_lstm/data.py's fix after its
OOM incident -- estimate array size before allocating, raise MemoryError if
it would exceed a configurable cap.
"""

import os
import sys
import importlib.util

import numpy as np

# Load paper_implementation/data.py under a distinct module name (same
# workaround used by every other experiment -- both files are named
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
window_channel = _paper_data.window_channel
FS = _paper_data.FS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_WIDTH = 50
CHANNELS = ("acc_m", "gyr_m", "vv")   # 3 derived signals, per the plan's scope decision
N_CHANNELS = len(CHANNELS)

STORAGE_DTYPE = np.float32
BYTES_PER_WINDOW = N_CHANNELS * WINDOW_WIDTH * WINDOW_WIDTH * np.dtype(STORAGE_DTYPE).itemsize
DEFAULT_MAX_MEMORY_GB = 8.0


def estimate_memory_gb(n_windows):
    return n_windows * BYTES_PER_WINDOW / (1024 ** 3)


def gasf(x, eps=1e-6):
    """x: (50,) 1D signal. Returns (50, 50) GASF image."""
    x_min, x_max = x.min(), x.max()
    rng = x_max - x_min
    if rng < eps:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = (2 * x - x_max - x_min) / rng
    x_scaled = np.clip(x_scaled, -1.0, 1.0)  # guard arccos domain
    phi = np.arccos(x_scaled)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gasf_batch(x_batch):
    """x_batch: (B, 50) -> (B, 50, 50), vectorized (no explicit python loop over the batch)."""
    x_min = x_batch.min(axis=1, keepdims=True)
    x_max = x_batch.max(axis=1, keepdims=True)
    rng = x_max - x_min
    rng_safe = np.where(rng < 1e-6, 1.0, rng)
    x_scaled = (2 * x_batch - x_max - x_min) / rng_safe
    x_scaled = np.where(rng < 1e-6, 0.0, x_scaled)
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    phi = np.arccos(x_scaled)                                  # (B, 50)
    return np.cos(phi[:, :, None] + phi[:, None, :]).astype(np.float32)  # (B, 50, 50)


def encode_window(window):
    """window: a paper_implementation.data.Window. Returns (3, 50, 50) GASF stack."""
    out = np.empty((N_CHANNELS, WINDOW_WIDTH, WINDOW_WIDTH), dtype=np.float32)
    for c, ch in enumerate(CHANNELS):
        out[c] = gasf(window_channel(window, ch))
    return out


def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


def build_gaf_dataset(subjects, stride, max_adl_windows=None, seed=42, max_memory_gb=DEFAULT_MAX_MEMORY_GB):
    """
    Returns (X: (N, 3, 50, 50) float32, y: (N,) int64, meta: DataFrame).
    Raises MemoryError (before allocating anything) if the requested window
    count would exceed max_memory_gb.
    """
    import pandas as pd

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
            f"build_gaf_dataset: {n} windows would need ~{est_gb:.1f} GB "
            f"(> max_memory_gb={max_memory_gb}). Lower max_adl_windows, raise "
            f"stride, or pass max_memory_gb explicitly if you've checked "
            f"available RAM first (e.g. via `free -h`)."
        )
    print(f"  ({n} windows, ~{est_gb:.2f} GB estimated)")

    X = np.empty((n, N_CHANNELS, WINDOW_WIDTH, WINDOW_WIDTH), dtype=STORAGE_DTYPE)
    y = np.empty(n, dtype=np.int64)
    meta = []
    for i, (w, label) in enumerate(windows):
        X[i] = encode_window(w)
        y[i] = label
        t = w.trial
        meta.append({"subject": t.subject, "task": t.task, "trial": t.trial,
                     "is_fall": t.is_fall, "end_frame": w.end,
                     "impact_frame": t.impact if t.is_fall else None})
    return X, y, pd.DataFrame(meta)


# ----------------------------------------------------------------------------- #
# Visual sanity check (go/no-go per the plan -- run before any model code)
# ----------------------------------------------------------------------------- #
def sanity_check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = make_subject_split()
    train_subjects = split["train_subjects"]

    fall_window = fall_desc = None
    jog_window = jog_desc = None
    for subj in train_subjects:
        for t in iter_subject_trials(subj):
            if t.is_fall and fall_window is None:
                best_w = None
                for w in iter_windows(t, stride=1):
                    if best_w is None or w.end > best_w.end:
                        best_w = w
                if best_w is not None:
                    fall_window = best_w
                    fall_desc = (f"SA{subj:02d} F{t.task-19:02d} trial{t.trial} "
                                 f"(window ending {t.impact - best_w.end} frames before impact)")
            if t.task == 8 and jog_window is None:  # D08 jog quick
                for w in iter_windows(t, stride=1):
                    jog_window = w
                    jog_desc = f"SA{subj:02d} D08 (jog quick)"
                    break
        if fall_window is not None and jog_window is not None:
            break

    fall_gaf = encode_window(fall_window)
    jog_gaf = encode_window(jog_window)

    fig, axes = plt.subplots(2, N_CHANNELS, figsize=(4 * N_CHANNELS, 8))
    for row, (label, gaf_stack, desc) in enumerate([("FALL", fall_gaf, fall_desc), ("JOG (ADL)", jog_gaf, jog_desc)]):
        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            im = ax.imshow(gaf_stack[col], cmap="RdBu", vmin=-1, vmax=1, origin="lower")
            ax.set_title(f"{label} -- {ch}\n{desc}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "sanity_check_gaf_mtf.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity-check GASF comparison to {out_path}")
    print("Inspect visually: is there a structural difference between FALL and JOG GASF images?")
    print("If not distinguishable, treat this as a no-go signal before investing in model training.")


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        sanity_check()
    else:
        print(f"GAF config: channels={CHANNELS}, image size={WINDOW_WIDTH}x{WINDOW_WIDTH}")
        print(f"Bytes/window: {BYTES_PER_WINDOW} ({BYTES_PER_WINDOW/1024:.1f} KB)")
        print("Run with --sanity-check to generate the fall-vs-ADL GASF comparison.")
