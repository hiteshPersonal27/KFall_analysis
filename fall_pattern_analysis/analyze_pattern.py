"""
Validate the fall signature using the KFall paper's actual detection logic
(Yu, Jang & Xiong 2021, Fig. 6), rather than an arbitrary SVM-only threshold.

Adds, specifically because the paper says they matter (not a blind multi-axis sweep):
  - Vertical velocity (VV), obtained by rotating body-frame acceleration into the
    world frame using the fused Euler-angle orientation, then integrating.
  - Pitch/Roll orientation check (paper's second-stage confirmation).
  - Gyroscope X/Z as an alternative sensitive axis, but ONLY for the sitting+fainting
    fall subset (F06-F08 / task IDs 25-27), which is the one subset the paper
    explicitly flags as having a weak acceleration signature.

Three rules are compared:
  Rule A: ACC_M-only threshold (SVM < 0.8g, the paper's threshold value, but a
          single-signal approach) -- the "is SVM alone enough" baseline.
  Rule B: the paper's full threshold algorithm (ACC_M < 0.8g AND VV > 0.3 m/s,
          followed within 10 frames by |Pitch|>25 deg or |Roll|>25 deg), 5 Hz
          low-pass filtered.
  Rule C: gyro-based alternative (fainting-sitting subset only).

Each rule is run on BOTH fall trials (T20-T34, sensitivity) and ADL trials
(T01-T21, specificity).
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # KFall/ -- makes this script cwd-independent
SENSOR_DIR = os.path.join(PROJECT_ROOT, "sensor_data")
LABEL_DIR = os.path.join(PROJECT_ROOT, "label_data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "pattern_results.csv")
FS = 100.0
LOWPASS_HZ = 5.0
ACC_M_THRESH = 0.8      # g, paper's grid-searched threshold
VV_THRESH = 0.3         # m/s, paper's threshold
ANGLE_THRESH = 25.0     # degrees, paper's threshold
LOOKAHEAD_FRAMES = 10   # paper's window for the orientation confirmation
GYRO_FAINT_MULT = 3.0   # multiple of standing baseline to flag gyro trigger
GYRO_FAINT_FLOOR = 30.0 # deg/s, absolute floor to avoid triggering on pure noise

FALL_TASK_IDS = list(range(20, 35))   # F01(20)..F15(34)
ADL_TASK_IDS = list(range(1, 22))     # D01(1)..D21(21)
FAINT_SITTING_TASKS = [25, 26, 27]    # F06, F07, F08


def discover_subjects():
    subjects = []
    for name in sorted(os.listdir(SENSOR_DIR)):
        m = re.match(r"SA(\d+)$", name)
        if m:
            subjects.append(int(m.group(1)))
    return subjects


def load_sensor_data(subject_id, task_id, trial_id):
    file_path = f"{SENSOR_DIR}/SA{subject_id:02d}/S{subject_id:02d}T{task_id:02d}R{trial_id:02d}.csv"
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)


def load_labels(subject_id):
    label_path = f"{LABEL_DIR}/SA{subject_id:02d}_label.xlsx"
    if not os.path.exists(label_path):
        return None
    df = pd.read_excel(label_path)
    df["Task Code (Task ID)"] = df["Task Code (Task ID)"].ffill()
    df["Description"] = df["Description"].ffill()
    return df


def get_fall_label_info(df_label, task_id, trial_id):
    pattern = f"({task_id})"
    match_rows = df_label[
        (df_label["Task Code (Task ID)"].str.contains(pattern, regex=False, na=False))
        & (df_label["Trial ID"] == trial_id)
    ]
    if match_rows.empty:
        return None
    row = match_rows.iloc[0]
    return {"onset": int(row["Fall_onset_frame"]), "impact": int(row["Fall_impact_frame"])}


def lowpass_filter(x, cutoff_hz=LOWPASS_HZ, fs=FS, order=4):
    n = len(x)
    padlen = 3 * order
    if n <= padlen:
        return x.copy()
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def compute_signals(df):
    """Returns filtered ACC_M and VV (vertical velocity, m/s) arrays."""
    accx = lowpass_filter(df["AccX"].values)
    accy = lowpass_filter(df["AccY"].values)
    accz = lowpass_filter(df["AccZ"].values)
    acc_body = np.column_stack([accx, accy, accz])
    acc_m = np.sqrt(accx**2 + accy**2 + accz**2)

    # "up" direction in body coordinates at t=0, from the raw (unfiltered) mean of
    # the first 10 samples, assuming the trial starts near-stationary.
    k = min(10, len(df))
    up_body0 = df[["AccX", "AccY", "AccZ"]].iloc[:k].mean().values
    norm = np.linalg.norm(up_body0)
    if norm < 1e-6:
        up_body0 = np.array([0.0, -1.0, 0.0])
    else:
        up_body0 = up_body0 / norm

    # Orientation as intrinsic Z-Y-X (yaw, pitch, roll) rotations, matching the
    # manufacturer's Euler angle convention (EulerX=roll, EulerY=pitch, EulerZ=yaw).
    angles = df[["EulerZ", "EulerY", "EulerX"]].values  # yaw, pitch, roll (deg)
    R_t = Rotation.from_euler("ZYX", angles, degrees=True)
    R_0 = Rotation.from_euler("ZYX", angles[0], degrees=True)

    # up_body(t) = R(t)^-1 * R(0) * up_body(0)  -- rotates the initial "up" direction
    # by the relative orientation change, so we can project acceleration onto the
    # (possibly rotated) vertical axis without needing an absolute world frame.
    rel = R_t.inv() * R_0
    up_body_t = rel.apply(up_body0)

    vertical_accel_g = np.einsum("ij,ij->i", acc_body, up_body_t) - 1.0
    time = df["TimeStamp(s)"].values
    vv_cumulative = cumulative_trapezoid(vertical_accel_g * 9.81, time, initial=0.0)

    # Naive full-trial integration drifts badly (a fraction-of-a-g sensor bias,
    # integrated over a 10-30s recording, produces multiple m/s of spurious
    # "velocity"). Falls are brief (~750ms per the paper), so use a short
    # windowed integral instead: VV(t) = cumulative(t) - cumulative(t - window),
    # which bounds drift to at most bias*window regardless of trial length.
    window_frames = int(round(1.0 * FS))  # 1s window
    vv_baseline = np.concatenate([np.zeros(window_frames), vv_cumulative[:-window_frames]])
    vv_up_positive = vv_cumulative - vv_baseline

    # Flip sign: paper's VV>0.3 threshold only makes physical sense as downward
    # speed (falling), confirmed by checking a real fall trial (SA06/T32) where
    # the up-positive integral goes strongly negative -- exactly during descent.
    vv = -vv_up_positive

    return acc_m, vv


def rule_a_fires(acc_m):
    idx = np.where(acc_m < ACC_M_THRESH)[0]
    return (idx[0] if len(idx) else None)


def rule_b_fires(acc_m, vv, pitch, roll):
    core = (acc_m < ACC_M_THRESH) & (vv > VV_THRESH)
    angle_exceed = (np.abs(pitch) > ANGLE_THRESH) | (np.abs(roll) > ANGLE_THRESH)
    # "within the next 10 frames from T" (inclusive of T)
    angle_exceed_series = pd.Series(angle_exceed[::-1])
    forward_any = angle_exceed_series.rolling(LOOKAHEAD_FRAMES, min_periods=1).max().astype(bool).values[::-1]
    fires = core & forward_any
    idx = np.where(fires)[0]
    return (idx[0] if len(idx) else None)


def get_gyro_baseline(subject_id, cache):
    if subject_id in cache:
        return cache[subject_id]
    df = load_sensor_data(subject_id, 1, 1)  # D01: stand for 30s
    if df is None:
        cache[subject_id] = 10.0
    else:
        cache[subject_id] = max(df["GyrX"].abs().max(), df["GyrZ"].abs().max())
    return cache[subject_id]


def rule_c_fires(df, baseline):
    peak = max(df["GyrX"].abs().max(), df["GyrZ"].abs().max())
    thresh = max(GYRO_FAINT_MULT * baseline, GYRO_FAINT_FLOOR)
    return peak > thresh


def sanity_check_vv():
    """VV should stay near 0 during quiet standing (D01) -- validates the rotation."""
    subjects = discover_subjects()
    df = load_sensor_data(subjects[0], 1, 1)
    _, vv = compute_signals(df)
    print(f"[sanity check] VV during standing (SA{subjects[0]:02d}, D01): "
          f"mean={vv.mean():.3f} m/s, max|VV|={np.abs(vv).max():.3f} m/s "
          f"(expect near 0)")


def process_fall_trials(subjects, gyro_baseline_cache):
    rows = []
    for subject in subjects:
        df_label = load_labels(subject)
        baseline = get_gyro_baseline(subject, gyro_baseline_cache)
        for task in FALL_TASK_IDS:
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                label_info = get_fall_label_info(df_label, task, trial) if df_label is not None else None
                if label_info is None:
                    continue
                acc_m, vv = compute_signals(df)
                pitch = df["EulerY"].values
                roll = df["EulerX"].values

                a_idx = rule_a_fires(acc_m)
                b_idx = rule_b_fires(acc_m, vv, pitch, roll)
                frames = df["FrameCounter"].values
                impact_frame = label_info["impact"]

                a_lead = (impact_frame - frames[a_idx]) * 10 if a_idx is not None else np.nan
                b_lead = (impact_frame - frames[b_idx]) * 10 if b_idx is not None else np.nan

                c_fires = np.nan
                if task in FAINT_SITTING_TASKS:
                    c_fires = rule_c_fires(df, baseline)

                rows.append({
                    "subject": subject, "task": task, "trial": trial, "is_fall": True,
                    "rule_a_fires": a_idx is not None, "rule_a_leadtime_ms": a_lead,
                    "rule_b_fires": b_idx is not None, "rule_b_leadtime_ms": b_lead,
                    "rule_c_fires": c_fires,
                })
    return rows


def process_adl_trials(subjects):
    rows = []
    for subject in subjects:
        for task in ADL_TASK_IDS:
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                acc_m, vv = compute_signals(df)
                pitch = df["EulerY"].values
                roll = df["EulerX"].values

                a_idx = rule_a_fires(acc_m)
                b_idx = rule_b_fires(acc_m, vv, pitch, roll)

                rows.append({
                    "subject": subject, "task": task, "trial": trial, "is_fall": False,
                    "rule_a_fires": a_idx is not None, "rule_a_leadtime_ms": np.nan,
                    "rule_b_fires": b_idx is not None, "rule_b_leadtime_ms": np.nan,
                    "rule_c_fires": np.nan,
                })
    return rows


def summarize(results):
    fall = results[results["is_fall"]]
    adl = results[~results["is_fall"]]

    sens_a = fall["rule_a_fires"].mean()
    sens_b = fall["rule_b_fires"].mean()
    spec_a = 1 - adl["rule_a_fires"].mean()
    spec_b = 1 - adl["rule_b_fires"].mean()

    print(f"\nFall trials: {len(fall)}  |  ADL trials: {len(adl)}")
    print(f"\n{'Rule':<10}{'Sensitivity':>14}{'Specificity':>14}")
    print(f"{'A (SVM only)':<10}{sens_a:>13.1%} {spec_a:>13.1%}")
    print(f"{'B (paper)':<10}{sens_b:>13.1%} {spec_b:>13.1%}")
    print(f"\nPaper's own Threshold algorithm (Table 3): sensitivity 95.50%, specificity 83.43%")

    lead_a = fall.loc[fall["rule_a_fires"], "rule_a_leadtime_ms"]
    lead_b = fall.loc[fall["rule_b_fires"], "rule_b_leadtime_ms"]
    print(f"\nMean lead time -- Rule A: {lead_a.mean():.0f} ms | Rule B: {lead_b.mean():.0f} ms "
          f"(paper's Threshold algorithm: 333±160 ms)")

    faint = fall[fall["task"].isin(FAINT_SITTING_TASKS)]
    print(f"\n=== Sitting+fainting subset (F06-F08, n={len(faint)}) ===")
    print(f"Rule A (accel) detection rate: {faint['rule_a_fires'].mean():.1%}")
    print(f"Rule B (accel+VV+angle) detection rate: {faint['rule_b_fires'].mean():.1%}")
    # rule_c_fires is NaN outside the fainting-sitting subset, which forces the
    # column to dtype=object (mixing numpy.bool_ with float NaN). .mean() on an
    # object-dtype column silently computes garbage -- cast to bool explicitly
    # after dropping the (already-excluded-by-filter) NaNs before aggregating.
    rule_c_rate = faint["rule_c_fires"].dropna().astype(bool).mean()
    print(f"Rule C (gyro) detection rate: {rule_c_rate:.1%}")

    return sens_a, sens_b, spec_a, spec_b


def make_plots(results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fall = results[results["is_fall"]]
    adl = results[~results["is_fall"]]

    # 1. Sensitivity/specificity comparison
    sens_a, sens_b = fall["rule_a_fires"].mean(), fall["rule_b_fires"].mean()
    spec_a, spec_b = 1 - adl["rule_a_fires"].mean(), 1 - adl["rule_b_fires"].mean()
    plt.figure(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width / 2, [sens_a, spec_a], width, label="Rule A (SVM only)", color="#e74c3c")
    plt.bar(x + width / 2, [sens_b, spec_b], width, label="Rule B (paper: ACC_M+VV+angle)", color="#2ecc71")
    plt.xticks(x, ["Sensitivity\n(fall trials correctly flagged)", "Specificity\n(ADL trials correctly ignored)"])
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title("Is SVM alone enough? Sensitivity/Specificity, Rule A vs Rule B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_specificity_comparison.png"), dpi=300)
    plt.close()

    # 2. Heatmap of Rule B detection rate, subject x task
    pivot = fall.pivot_table(index="subject", columns="task", values="rule_b_fires", aggfunc="mean")
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Rule B detection rate"},
                linewidths=0.3, linecolor="white")
    plt.title("Paper-faithful Detection Rule (B): Subject x Task Consistency")
    plt.xlabel("Fall Task ID")
    plt.ylabel("Subject ID")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap_ruleB.png"), dpi=300)
    plt.close()

    # 3. ADL false positive rate by task, Rule A vs Rule B
    fp = adl.groupby("task")[["rule_a_fires", "rule_b_fires"]].mean().sort_values("rule_a_fires", ascending=False)
    plt.figure(figsize=(12, 5))
    x = np.arange(len(fp))
    width = 0.35
    plt.bar(x - width / 2, fp["rule_a_fires"], width, label="Rule A (SVM only)", color="#e74c3c")
    plt.bar(x + width / 2, fp["rule_b_fires"], width, label="Rule B (paper)", color="#2ecc71")
    plt.xticks(x, [f"D{t:02d}" for t in fp.index], rotation=45)
    plt.ylabel("False positive rate")
    plt.title("ADL False-Positive Rate by Task: does SVM-only over-trigger on daily activities?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "adl_false_positive_by_task.png"), dpi=300)
    plt.close()

    # 4. Fainting-sitting subset: accel-based vs gyro-based detection
    faint = fall[fall["task"].isin(FAINT_SITTING_TASKS)]
    rates = {
        "Rule A\n(accel only)": faint["rule_a_fires"].mean(),
        "Rule B\n(accel+VV+angle)": faint["rule_b_fires"].mean(),
        "Rule C\n(gyro X/Z)": faint["rule_c_fires"].dropna().astype(bool).mean(),
    }
    plt.figure(figsize=(7, 5))
    plt.bar(rates.keys(), rates.values(), color=["#e74c3c", "#f39c12", "#3498db"])
    plt.ylim(0, 1.05)
    plt.ylabel("Detection rate")
    plt.title("Sitting+Fainting Falls (F06-F08): Accel- vs Gyro-based Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fainting_subset_axis_comparison.png"), dpi=300)
    plt.close()

    print(f"\nSaved 4 plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    print("Running VV sanity check...")
    sanity_check_vv()

    subjects = discover_subjects()
    gyro_baseline_cache = {}

    print(f"\nProcessing {len(subjects)} subjects x fall tasks (T20-T34)...")
    fall_rows = process_fall_trials(subjects, gyro_baseline_cache)

    print(f"Processing {len(subjects)} subjects x ADL tasks (T01-T21)...")
    adl_rows = process_adl_trials(subjects)

    results = pd.DataFrame(fall_rows + adl_rows)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(results)} trial results to {RESULTS_CSV}")

    summarize(results)
    make_plots(results)
