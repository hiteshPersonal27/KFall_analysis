"""
Data layer for the GAF global-normalization fix (see the "GAF
Global-Normalization Fix" plan, previously saved at
~/.claude/plans/ok-so-how-many-sharded-cocoa.md, and
../gaf_mtf/README.md's diagnosis section).

experiments/gaf_mtf/ badly underperformed the ConvLSTM baseline (9.20% vs.
89.85% specificity at matched scale) and topped out at ~93% validation
accuracy (vs. ~99% elsewhere). Diagnosis: GASF's standard formula rescales
each window to [-1,1] using THAT WINDOW's OWN min/max. Two overlapping test
windows sharing 90% of their frames can still get different local min/max
references, discontinuously shifting the entire pairwise-angle-sum image --
plausibly explaining both the noisy training (near-identical inputs mapped
to visually different images but the same label) and the unstable
evaluation (CONSEC_WINDOWS persistence collapses because adjacent
overlapping windows don't agree).

THE FIX: global normalization. Per-channel min/max computed ONCE from the
training data (via full, unwindowed trial signals), then applied identically
to every window at train and test time -- deviates from the textbook GASF
formula, but removes the root cause instead of patching the symptom.

Also trained at FULL scale (stride=1, uncapped ADL), not the 150K matched
scale gaf_mtf used -- per explicit user direction. Memory note: GASF images
at (3,50,50) float32 would need ~50GB for the full ~1.76M-window fit+val
set, too close to available RAM (an OOM already crashed this machine once,
in experiments/cwt_lstm/'s early history). Stored as float16 instead
(~25GB), cast to float32 per-batch during training -- same fix cwt_lstm
adopted after its own incident.
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
CHANNELS = ("acc_m", "gyr_m", "vv")   # same 3 derived signals as gaf_mtf
N_CHANNELS = len(CHANNELS)

STORAGE_DTYPE = np.float16   # was float32 in gaf_mtf -- halves memory for the full-scale run
BYTES_PER_WINDOW = N_CHANNELS * WINDOW_WIDTH * WINDOW_WIDTH * np.dtype(STORAGE_DTYPE).itemsize
DEFAULT_MAX_MEMORY_GB = 28.0  # raised from gaf_mtf's 8.0 -- deliberate, documented (see module docstring)


def estimate_memory_gb(n_windows):
    return n_windows * BYTES_PER_WINDOW / (1024 ** 3)


# ----------------------------------------------------------------------------- #
# Global normalization stats (the fix)
# ----------------------------------------------------------------------------- #
def compute_global_stats(subjects):
    """
    Per-channel (min, max) computed from the FULL (unwindowed) trial signals
    of the given subjects -- fit on training subjects only, then reused
    identically for every window at both train and test time.
    Returns {channel: (min, max)}.
    """
    mins = {ch: np.inf for ch in CHANNELS}
    maxs = {ch: -np.inf for ch in CHANNELS}
    for subj in subjects:
        for t in iter_subject_trials(subj):
            for ch in CHANNELS:
                arr = getattr(t, ch)
                mins[ch] = min(mins[ch], float(np.min(arr)))
                maxs[ch] = max(maxs[ch], float(np.max(arr)))
    return {ch: (mins[ch], maxs[ch]) for ch in CHANNELS}


def save_global_stats(stats, path):
    np.savez(path, **{f"{ch}_min": v[0] for ch, v in stats.items()},
              **{f"{ch}_max": v[1] for ch, v in stats.items()})


def load_global_stats(path):
    data = np.load(path)
    return {ch: (float(data[f"{ch}_min"]), float(data[f"{ch}_max"])) for ch in CHANNELS}


# ----------------------------------------------------------------------------- #
# GASF, global-normalized version
# ----------------------------------------------------------------------------- #
def gasf_global(x, ch_min, ch_max, eps=1e-6):
    """x: (50,) 1D signal. Rescales using the FIXED (ch_min, ch_max), not x's own min/max."""
    rng = ch_max - ch_min
    if rng < eps:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = (2 * x - ch_max - ch_min) / rng
    x_scaled = np.clip(x_scaled, -1.0, 1.0)  # guard arccos domain (also catches out-of-range test values)
    phi = np.arccos(x_scaled)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gasf_global_batch(x_batch, ch_min, ch_max, eps=1e-6):
    """x_batch: (B, 50) -> (B, 50, 50), vectorized, fixed (ch_min, ch_max) for the whole batch."""
    rng = ch_max - ch_min
    if rng < eps:
        x_scaled = np.zeros_like(x_batch)
    else:
        x_scaled = (2 * x_batch - ch_max - ch_min) / rng
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    phi = np.arccos(x_scaled)                                  # (B, 50)
    return np.cos(phi[:, :, None] + phi[:, None, :]).astype(np.float32)  # (B, 50, 50)


def encode_window(window, global_stats):
    """window: a paper_implementation.data.Window. Returns (3, 50, 50) GASF stack."""
    out = np.empty((N_CHANNELS, WINDOW_WIDTH, WINDOW_WIDTH), dtype=np.float32)
    for c, ch in enumerate(CHANNELS):
        ch_min, ch_max = global_stats[ch]
        out[c] = gasf_global(window_channel(window, ch), ch_min, ch_max)
    return out


def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


def build_gaf_dataset(subjects, stride, global_stats, max_adl_windows=None, seed=42,
                       max_memory_gb=DEFAULT_MAX_MEMORY_GB):
    """
    Returns (X: (N, 3, 50, 50) float16, y: (N,) int64, meta: DataFrame).
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
        X[i] = encode_window(w, global_stats).astype(STORAGE_DTYPE)
        y[i] = label
        t = w.trial
        meta.append({"subject": t.subject, "task": t.task, "trial": t.trial,
                     "is_fall": t.is_fall, "end_frame": w.end,
                     "impact_frame": t.impact if t.is_fall else None})
    return X, y, pd.DataFrame(meta)


# ----------------------------------------------------------------------------- #
# Visual sanity checks (go/no-go per the plan -- run before any model code)
# ----------------------------------------------------------------------------- #
def _find_fall_and_jog_trial(subjects):
    for subj in subjects:
        for t in iter_subject_trials(subj):
            if t.is_fall:
                fall_trial = t
                break
        else:
            continue
        break
    for subj in subjects:
        for t in iter_subject_trials(subj):
            if t.task == 8:  # D08 jog quick
                return fall_trial, t
    return fall_trial, None


def sanity_check():
    """Original check: fall vs. jog GASF images (global-normalized), same layout as gaf_mtf's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = make_subject_split()
    train_subjects = split["train_subjects"]
    global_stats = compute_global_stats(train_subjects)
    print(f"Global stats (fit on {len(train_subjects)} training subjects): {global_stats}")

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
            if t.task == 8 and jog_window is None:
                for w in iter_windows(t, stride=1):
                    jog_window = w
                    jog_desc = f"SA{subj:02d} D08 (jog quick)"
                    break
        if fall_window is not None and jog_window is not None:
            break

    fall_gaf = encode_window(fall_window, global_stats)
    jog_gaf = encode_window(jog_window, global_stats)

    fig, axes = plt.subplots(2, N_CHANNELS, figsize=(4 * N_CHANNELS, 8))
    for row, (label, gaf_stack, desc) in enumerate([("FALL", fall_gaf, fall_desc), ("JOG (ADL)", jog_gaf, jog_desc)]):
        for col, ch in enumerate(CHANNELS):
            ax = axes[row, col]
            im = ax.imshow(gaf_stack[col], cmap="RdBu", vmin=-1, vmax=1, origin="lower")
            ax.set_title(f"{label} -- {ch} (global norm)\n{desc}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "sanity_check_gaf_global.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity-check GASF comparison to {out_path}")
    print("Inspect visually: is there a structural difference between FALL and JOG GASF images?")

    return global_stats


def sanity_check_overlap_stability(global_stats=None):
    """
    NEW check (the actual fix confirmation): plot the SAME two consecutive,
    heavily-overlapping windows (5-frame shift) under per-window
    normalization (gaf_mtf's original approach, reimplemented inline here
    for comparison only) vs. this experiment's global normalization. The
    per-window version should show a visible discontinuity between the two;
    the global version should look nearly identical (since the underlying
    signal barely changed) -- this is the direct before/after evidence the
    fix should produce.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = make_subject_split()
    train_subjects = split["train_subjects"]
    if global_stats is None:
        global_stats = compute_global_stats(train_subjects)

    # Find an ADL trial (jog) with at least two windows 5 frames apart.
    target_trial = None
    for subj in train_subjects:
        for t in iter_subject_trials(subj):
            if t.task == 8 and t.n_frames >= 60:
                target_trial = t
                break
        if target_trial is not None:
            break

    windows = list(iter_windows(target_trial, stride=5))
    w1, w2 = windows[2], windows[3]  # two windows 5 frames apart, past any edge effects

    def gasf_per_window(x, eps=1e-6):
        x_min, x_max = x.min(), x.max()
        rng = x_max - x_min
        x_scaled = np.zeros_like(x) if rng < eps else (2 * x - x_max - x_min) / rng
        x_scaled = np.clip(x_scaled, -1.0, 1.0)
        phi = np.arccos(x_scaled)
        return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)

    ch = "gyr_m"  # the channel that showed the clearest structure in gaf_mtf's original sanity check
    x1, x2 = window_channel(w1, ch), window_channel(w2, ch)
    per_window_1, per_window_2 = gasf_per_window(x1), gasf_per_window(x2)
    ch_min, ch_max = global_stats[ch]
    global_1 = gasf_global(x1, ch_min, ch_max)
    global_2 = gasf_global(x2, ch_min, ch_max)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (label, img1, img2) in enumerate([
        ("PER-WINDOW norm (gaf_mtf original)", per_window_1, per_window_2),
        ("GLOBAL norm (this fix)", global_1, global_2),
    ]):
        axes[row, 0].imshow(img1, cmap="RdBu", vmin=-1, vmax=1, origin="lower")
        axes[row, 0].set_title(f"{label}\nwindow @ frame {w1.start}", fontsize=9)
        axes[row, 1].imshow(img2, cmap="RdBu", vmin=-1, vmax=1, origin="lower")
        axes[row, 1].set_title(f"{label}\nwindow @ frame {w2.start} (5 frames later)", fontsize=9)
        diff = np.abs(img1 - img2)
        im = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=1, origin="lower")
        axes[row, 2].set_title(f"|difference|  (mean={diff.mean():.4f})", fontsize=9)
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
    plt.suptitle(f"Overlap-stability check: two windows 5 frames apart, channel={ch} "
                 f"({target_trial.subject}, D08 jog)")
    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "sanity_check_overlap_stability.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved overlap-stability comparison to {out_path}")
    print("Inspect visually: does GLOBAL norm's diff map look much smaller/smoother than PER-WINDOW's?")


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        gstats = sanity_check()
        sanity_check_overlap_stability(gstats)
    else:
        print(f"GAF (global norm) config: channels={CHANNELS}, image size={WINDOW_WIDTH}x{WINDOW_WIDTH}, "
              f"dtype={STORAGE_DTYPE}")
        print(f"Bytes/window: {BYTES_PER_WINDOW} ({BYTES_PER_WINDOW/1024:.1f} KB)")
        print("Run with --sanity-check to generate both go/no-go visual checks.")
