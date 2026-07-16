"""
Rolling-regression beta-signal visualization -- Phase 1 of the explainable
real-time pre-impact detector.

Idea: instead of thresholding raw ACC_M, watch the LOCAL SHAPE of the signal via a
causal rolling regression. At every frame t, fit a small window of past samples with
a linear model (slope beta1) and a quadratic (curvature beta2). beta1/beta2 are new
per-frame signals that should be ~flat during ordinary activity and spike distinctly
when a fall begins (steep negative slope for the freefall dip; strong curvature at
the impact bump).

This script is VISUALIZATION ONLY: it computes beta1/beta2 on all three validated
strong channels (ACC_M, VV, GYR_M) and plots them event-locked (phase axis) for
falls vs. the false-positive-prone ADLs, to confirm the beta signals actually
separate falls from daily activity BEFORE any detector/threshold is built.

Reuses the validated pipeline from ../analyze_pattern.py (5 Hz filter, ACC_M, VV,
gyro norm) -- see ../docs/pattern_analysis.md.

Run:  python3 fall_pattern_analysis/rolling_regression/visualize_beta_signals.py
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, lowpass_filter, FALL_TASK_IDS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

FS = 100.0
WINDOW = 25                       # frames (~0.25 s) -- small for fast local response
PHASE_MIN, PHASE_MAX, N_PHASE = -50, 150, 201
PHASE_GRID = np.linspace(PHASE_MIN, PHASE_MAX, N_PHASE)

CHANNELS = ["ACC_M", "VV", "GYR_M"]
CH_UNITS = {"ACC_M": "g/frame", "VV": "(m/s)/frame", "GYR_M": "(deg/s)/frame"}

# False-positive-prone ADLs to overlay against falls (task id -> label).
RISKY_ADLS = {10: "D10 stumble", 13: "D13 sit-down", 14: "D14 quick-sit", 4: "D04 jump"}
ADL_TASK_IDS = list(range(1, 20)) + [35, 36]

# Precompute the fixed regression operators. The window x-values are always
# 0..WINDOW-1, so the design matrices are constant and the fit reduces to one
# matmul (pinv @ y) per window.
_X = np.arange(WINDOW, dtype=float)
_P1 = np.linalg.pinv(np.vstack([np.ones(WINDOW), _X]).T)          # (2, W): rows = [b0, b1]
_P2 = np.linalg.pinv(np.vstack([np.ones(WINDOW), _X, _X**2]).T)   # (3, W): rows = [b0, b1, b2]


def rolling_beta(signal):
    """
    Causal rolling regression. Returns (beta1, beta2), each length len(signal),
    NaN for the first WINDOW-1 frames (window not yet full). beta is stamped at the
    window's LAST frame -> uses only past+current data (deployable in real time).
    """
    n = len(signal)
    beta1 = np.full(n, np.nan)
    beta2 = np.full(n, np.nan)
    if n < WINDOW:
        return beta1, beta2
    win = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)  # (n-W+1, W)
    beta1[WINDOW - 1:] = win @ _P1[1]        # linear-fit slope
    beta2[WINDOW - 1:] = win @ _P2[2]        # quadratic-fit curvature coefficient
    return beta1, beta2


def channels_of(df):
    acc_m, vv = compute_signals(df)
    gyr_m = np.sqrt(
        lowpass_filter(df["GyrX"].values) ** 2
        + lowpass_filter(df["GyrY"].values) ** 2
        + lowpass_filter(df["GyrZ"].values) ** 2
    )
    return {"ACC_M": acc_m, "VV": vv, "GYR_M": gyr_m}


def resample(frames, values, start, end, adl):
    duration = end - start
    if duration <= 0:
        return None
    target = start + PHASE_GRID / 100.0 * duration
    out = np.interp(target, frames, values)
    if adl:  # ADLs span only 0-100% of their own duration; no data outside
        out[(PHASE_GRID < 0) | (PHASE_GRID > 100)] = np.nan
    return out


def collect(task_ids, is_fall):
    """
    For each trial of the given tasks: compute channels, causal beta1/beta2 (frame-
    time), THEN phase-align. Returns {channel: {'beta1': [...], 'beta2': [...]}}
    lists of phase-aligned arrays, and a trial count.
    """
    subjects = discover_subjects()
    acc = {ch: {"beta1": [], "beta2": []} for ch in CHANNELS}
    n = 0
    for subject in subjects:
        df_label = load_labels(subject) if is_fall else None
        for task in task_ids:
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                frames = df["FrameCounter"].values
                if is_fall:
                    info = get_fall_label_info(df_label, task, trial) if df_label is not None else None
                    if info is None:
                        continue
                    start, end = info["onset"], info["impact"]
                    if start < frames.min() or end > frames.max():
                        continue
                else:
                    start, end = frames[0], frames[-1]

                chans = channels_of(df)
                ok = True
                staged = {}
                for ch in CHANNELS:
                    b1, b2 = rolling_beta(chans[ch])
                    a1 = resample(frames, b1, start, end, adl=not is_fall)
                    a2 = resample(frames, b2, start, end, adl=not is_fall)
                    if a1 is None:
                        ok = False
                        break
                    staged[ch] = (a1, a2)
                if not ok:
                    continue
                for ch in CHANNELS:
                    acc[ch]["beta1"].append(staged[ch][0])
                    acc[ch]["beta2"].append(staged[ch][1])
                n += 1
    return acc, n


def nanmean_stack(arrs):
    # All-NaN columns (ADL region outside 0-100%) intentionally yield NaN; silence
    # the expected "Mean of empty slice" warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.array(arrs), axis=0)


def nanstd_stack(arrs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(np.array(arrs), axis=0)


def sanity_checks():
    print("=== Sanity checks ===")
    subjects = discover_subjects()
    # 1) Quiet standing (D01): betas ~ 0
    df = load_sensor_data(subjects[0], 1, 1)
    ch = channels_of(df)
    for c in CHANNELS:
        b1, b2 = rolling_beta(ch[c])
        print(f"  Standing D01 {c}: mean|beta1|={np.nanmean(np.abs(b1)):.4f}, "
              f"mean|beta2|={np.nanmean(np.abs(b2)):.5f} (expect ~0)")
    # 2) Known slip fall (SA06/T32): ACC_M beta1 should dive negative in the descent
    df = load_sensor_data(6, 32, 1)
    lab = load_labels(6)
    info = get_fall_label_info(lab, 32, 1)
    frames = df["FrameCounter"].values
    b1, b2 = rolling_beta(channels_of(df)["ACC_M"])
    onset, impact = info["onset"], info["impact"]
    seg = (frames >= onset) & (frames <= impact)
    print(f"  Fall SA06/T32 ACC_M: min beta1 in onset->impact = {np.nanmin(b1[seg]):.3f} "
          f"(expect strongly negative = freefall dip)")
    print(f"                       max beta2 in onset->impact = {np.nanmax(b2[seg]):.4f} "
          f"(expect positive spike = impact bump)")
    print()


def plot_channel(ch, fall_acc, adl_by_task, adl_all, out_path):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    beta_specs = [("beta1", "β₁  (local slope)"), ("beta2", "β₂  (local curvature)")]
    adl_colors = plt.cm.tab10(np.linspace(0, 1, len(adl_by_task) + 1))

    for ax, (bkey, blabel) in zip(axes, beta_specs):
        fmean = nanmean_stack(fall_acc[ch][bkey])
        fstd = nanstd_stack(fall_acc[ch][bkey])
        ax.plot(PHASE_GRID, fmean, color="#c0392b", lw=2.4, label="Falls (mean, n≈2319)", zorder=5)
        ax.fill_between(PHASE_GRID, fmean - fstd, fmean + fstd, color="#c0392b", alpha=0.15,
                        label="Falls ±1 std", zorder=1)
        # risky ADLs
        for i, (tid, lbl) in enumerate(RISKY_ADLS.items()):
            if adl_by_task.get(tid) and adl_by_task[tid][ch][bkey]:
                ax.plot(PHASE_GRID, nanmean_stack(adl_by_task[tid][ch][bkey]),
                        color=adl_colors[i], lw=1.5, ls="--", label=lbl)
        # all-ADL mean
        if adl_all[ch][bkey]:
            ax.plot(PHASE_GRID, nanmean_stack(adl_all[ch][bkey]), color="#2c3e50",
                    lw=1.5, ls=":", label="All ADLs (mean)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.axvline(100, color="gray", ls="--", lw=1)
        ax.set_ylabel(f"{blabel}\n[{CH_UNITS[ch]}{'' if bkey=='beta1' else '/frame'}]")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[0].set_title(f"Rolling-regression β signals on {ch} — Falls vs. risky ADLs "
                      f"(window={WINDOW} frames, causal)\n"
                      "Fall phase: 0%=onset, 100%=impact  |  ADL phase: 0%=start, 100%=end")
    axes[-1].set_xlabel("Phase (% of onset→impact for falls / trial duration for ADLs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sanity_checks()

    print("Computing β signals for all fall trials (causal rolling regression)...")
    fall_acc, n_fall = collect(FALL_TASK_IDS, is_fall=True)
    print(f"  aggregated {n_fall} fall trials.")

    print("Computing β signals for risky ADL tasks...")
    adl_by_task = {}
    for tid in RISKY_ADLS:
        acc, n = collect([tid], is_fall=False)
        adl_by_task[tid] = acc
        print(f"  {RISKY_ADLS[tid]}: {n} trials.")

    print("Computing β signals for all ADL tasks (aggregate reference)...")
    adl_all, n_adl = collect(ADL_TASK_IDS, is_fall=False)
    print(f"  aggregated {n_adl} ADL trials.")

    print("\nPlotting...")
    for ch in CHANNELS:
        out = os.path.join(PLOTS_DIR, f"beta_{ch}.png")
        plot_channel(ch, fall_acc, adl_by_task, adl_all, out)
        print(f"  saved {out}")

    print(f"\nDone. {len(CHANNELS)} figures in {PLOTS_DIR}/")
