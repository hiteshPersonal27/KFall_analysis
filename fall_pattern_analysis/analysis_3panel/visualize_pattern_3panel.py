"""
3-panel version: ACC_M, Vertical Velocity, Tilt (pitch/roll).

Visualize the fall signature's shape (not just threshold-crossing rate) on a common,
event-locked time axis -- the direct signal-processing answer to "is the pattern
global": does the same shaped trace appear across subjects and tasks, not just
"did some threshold get crossed somewhere."

Method: for every fall trial, time-normalize onto a phase axis where 0% = the
labeled onset frame and 100% = the labeled impact frame (extended to -50%..+150%
for context before/after). This lets trials of very different durations (a 68-frame
sitting-fainting fall vs. a 500-frame walking-with-hands fall) be overlaid and
averaged meaningfully -- the same technique used for event-locked averaging in
neuroscience/signal processing (e.g. ERP analysis).

Run from the KFall project root: python3 analysis_3panel/visualize_pattern_3panel.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_threshold_validation"))
from analyze_pattern import (
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, FALL_TASK_IDS,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
PHASE_MIN, PHASE_MAX, N_PHASE = -50, 150, 201
PHASE_GRID = np.linspace(PHASE_MIN, PHASE_MAX, N_PHASE)


def resample_to_phase(frames, values, onset, impact):
    duration = impact - onset
    if duration <= 0:
        return None
    target_frames = onset + PHASE_GRID / 100.0 * duration
    return np.interp(target_frames, frames, values)


def build_phase_aligned_dataset():
    subjects = discover_subjects()
    records = []
    for subject in subjects:
        df_label = load_labels(subject)
        if df_label is None:
            continue
        for task in FALL_TASK_IDS:
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                label_info = get_fall_label_info(df_label, task, trial)
                if label_info is None:
                    continue
                onset, impact = label_info["onset"], label_info["impact"]
                frames = df["FrameCounter"].values
                if onset < frames.min() or impact > frames.max():
                    continue

                acc_m, vv = compute_signals(df)
                tilt = np.maximum(df["EulerY"].abs().values, df["EulerX"].abs().values)

                acc_m_p = resample_to_phase(frames, acc_m, onset, impact)
                vv_p = resample_to_phase(frames, vv, onset, impact)
                tilt_p = resample_to_phase(frames, tilt, onset, impact)
                if acc_m_p is None:
                    continue

                records.append({
                    "subject": subject, "task": task, "trial": trial,
                    "acc_m": acc_m_p, "vv": vv_p, "tilt": tilt_p,
                })
    return records


def plot_grand_average(records):
    acc_m = np.array([r["acc_m"] for r in records])
    vv = np.array([r["vv"] for r in records])
    tilt = np.array([r["tilt"] for r in records])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    specs = [
        (acc_m, "ACC_M (g)", "#1abc9c", 0.8, "0.8g dip threshold"),
        (vv, "Vertical Velocity (m/s)", "#3498db", 0.3, "0.3 m/s threshold"),
        (tilt, "Tilt = max(|Pitch|,|Roll|) (deg)", "#e74c3c", 25.0, "25 deg threshold"),
    ]
    for ax, (data, ylabel, color, thresh, thresh_label) in zip(axes, specs):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(PHASE_GRID, mean, color=color, linewidth=2, label="Mean across all fall trials")
        ax.fill_between(PHASE_GRID, mean - std, mean + std, color=color, alpha=0.2, label="±1 std")
        ax.axhline(thresh, color="black", linestyle=":", linewidth=1, label=thresh_label)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(100, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Grand-Average Fall Signature, Phase-Aligned (n={len(records)} trials)\n"
                       "0% = labeled onset, 100% = labeled impact  |  3-panel: ACC_M, VV, Tilt")
    axes[-1].set_xlabel("Phase (% of onset->impact duration)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pattern_grand_average.png"), dpi=300)
    plt.close()


def plot_intersubject_overlay(records):
    df_meta = pd.DataFrame([{"subject": r["subject"]} for r in records])
    acc_m = np.array([r["acc_m"] for r in records])
    vv = np.array([r["vv"] for r in records])
    tilt = np.array([r["tilt"] for r in records])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    specs = [(acc_m, "ACC_M (g)"), (vv, "Vertical Velocity (m/s)"), (tilt, "Tilt (deg)")]

    subjects = sorted(df_meta["subject"].unique())
    cmap = plt.get_cmap("viridis", len(subjects))

    for ax, (data, ylabel) in zip(axes, specs):
        for i, subj in enumerate(subjects):
            mask = (df_meta["subject"] == subj).values
            subj_mean = data[mask].mean(axis=0)
            ax.plot(PHASE_GRID, subj_mean, color=cmap(i), alpha=0.5, linewidth=1)
        ax.plot(PHASE_GRID, data.mean(axis=0), color="black", linewidth=2.5, label="Global mean")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(100, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Inter-Subject Overlay: each colored line = one subject's mean "
                       f"(n={len(subjects)} subjects)  |  3-panel: ACC_M, VV, Tilt")
    axes[-1].set_xlabel("Phase (% of onset->impact duration)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pattern_intersubject_overlay.png"), dpi=300)
    plt.close()


def plot_intrasubject_small_multiples(records):
    df_meta = pd.DataFrame([{"subject": r["subject"]} for r in records])
    acc_m = np.array([r["acc_m"] for r in records])
    subjects = sorted(df_meta["subject"].unique())

    ncols = 6
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, subj in enumerate(subjects):
        ax = axes[i]
        mask = (df_meta["subject"] == subj).values
        trials = acc_m[mask]
        for trace in trials:
            ax.plot(PHASE_GRID, trace, color="#95a5a6", alpha=0.3, linewidth=0.7)
        ax.plot(PHASE_GRID, trials.mean(axis=0), color="#1abc9c", linewidth=1.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.axvline(100, color="gray", linestyle="--", linewidth=0.6)
        ax.axhline(0.8, color="black", linestyle=":", linewidth=0.6)
        ax.set_title(f"SA{subj:02d} (n={len(trials)})", fontsize=9)

    for j in range(len(subjects), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Intra-Subject Consistency: individual fall trials (gray) + subject mean (teal), ACC_M",
                 fontsize=13)
    fig.text(0.5, 0.0, "Phase (% of onset->impact duration)", ha="center")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(os.path.join(PLOTS_DIR, "pattern_intrasubject_small_multiples.png"), dpi=300)
    plt.close()


def plot_trial_heatmap(records):
    order = sorted(range(len(records)), key=lambda i: (records[i]["task"], records[i]["subject"]))
    acc_m_sorted = np.array([records[i]["acc_m"] for i in order])
    tasks_sorted = [records[i]["task"] for i in order]

    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(acc_m_sorted, aspect="auto", cmap="inferno",
                    extent=[PHASE_MIN, PHASE_MAX, len(order), 0], vmin=0, vmax=2.0)
    ax.axvline(0, color="cyan", linestyle="--", linewidth=1)
    ax.axvline(100, color="cyan", linestyle="--", linewidth=1)

    prev_task = None
    for row, task in enumerate(tasks_sorted):
        if task != prev_task:
            ax.axhline(row, color="white", linewidth=0.5, alpha=0.6)
            ax.text(PHASE_MAX + 3, row + 1, f"T{task}", fontsize=7, va="top")
            prev_task = task

    ax.set_xlabel("Phase (% of onset->impact duration)")
    ax.set_ylabel("Trial (sorted by fall task, then subject)")
    ax.set_title(f"Single-Trial Raster: ACC_M across all {len(order)} fall trials\n"
                 "(the pattern underlying the grand average, trial by trial)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ACC_M (g)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pattern_trial_heatmap.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Building phase-aligned dataset from all fall trials...")
    records = build_phase_aligned_dataset()
    print(f"Aligned {len(records)} fall trials onto a common phase axis "
          f"({PHASE_MIN}% to {PHASE_MAX}%, 0=onset, 100=impact).")

    print("Plotting grand average...")
    plot_grand_average(records)
    print("Plotting inter-subject overlay...")
    plot_intersubject_overlay(records)
    print("Plotting intra-subject small multiples...")
    plot_intrasubject_small_multiples(records)
    print("Plotting trial-level heatmap...")
    plot_trial_heatmap(records)

    print(f"\nSaved 4 plots to {PLOTS_DIR}/")
