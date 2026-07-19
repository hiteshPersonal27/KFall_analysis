"""
Ensemble pre-impact fall trigger: Threshold + CUSUM + Shapelet, pooled by majority
vote. Implements ensemble_trigger_plan.md end to end.

Reuses the validated pipeline (paper_threshold_validation/analyze_pattern.py) and the
Savitzky-Golay construction (rolling_regression/) -- no signal math duplicated.

Methodology:
  - By-subject train/test split (26/6, matching the source paper's own ratio) --
    NEW for this project: unlike paper_threshold_validation (which used the paper's
    already-published thresholds), this ensemble's parameters (Threshold cutoff,
    CUSUM k/h, Shapelet match cutoff) are tuned on OUR data, so a held-out test set
    is required for honest reported performance. All tuning uses train subjects only.
  - Success criterion (per the plan): a detection counts only if it fires BEFORE the
    labeled impact frame. Lead time = (impact_frame - fire_frame) * 10 ms.
  - Per-activity false-alarm breakdown on the test set, to check whether the three
    detectors' errors are complementary (voting helps) or correlated (it won't).

Run:  python3 fall_pattern_analysis/ensemble_trigger/build_ensemble.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "paper_threshold_validation")
sys.path.insert(0, PIPELINE_DIR)
from analyze_pattern import (  # noqa: E402
    discover_subjects, load_sensor_data, load_labels, get_fall_label_info,
    compute_signals, FALL_TASK_IDS, ADL_TASK_IDS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from detectors import (  # noqa: E402
    rolling_beta1, threshold_fire, cusum_fire, discover_shapelet, shapelet_fire,
    window_distance, SHAPELET_LEN, SG_WINDOW,
)

PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "ensemble_results.csv")

FS = 100.0
N_TEST_SUBJECTS = 6  # matches the source paper's 26-train/6-test split (of 32 total)
RNG_SEED = 42

# Paper's own reference benchmarks (Table 3), for comparison.
PAPER_REFERENCE = {
    "Threshold (paper)": (0.9550, 0.8343, 333),
    "SVM (paper)": (0.9977, 0.9487, 385),
    "ConvLSTM (paper)": (0.9932, 0.9901, 403),
}


# ----------------------------------------------------------------------------- #
# Step 1: load every trial's signals once
# ----------------------------------------------------------------------------- #
def load_all_trials():
    subjects = discover_subjects()
    records = []
    for subject in subjects:
        df_label = load_labels(subject)
        for task in FALL_TASK_IDS + ADL_TASK_IDS:
            is_fall = task in FALL_TASK_IDS
            for trial in range(1, 6):
                df = load_sensor_data(subject, task, trial)
                if df is None:
                    continue
                frames = df["FrameCounter"].values
                onset = impact = None
                if is_fall:
                    info = get_fall_label_info(df_label, task, trial) if df_label is not None else None
                    if info is None:
                        continue
                    onset, impact = info["onset"], info["impact"]
                    if onset < frames.min() or impact > frames.max():
                        continue

                acc_m, vv = compute_signals(df)
                vv_beta1 = rolling_beta1(vv)

                records.append({
                    "subject": subject, "task": task, "trial": trial, "is_fall": is_fall,
                    "frames": frames, "acc_m": acc_m, "vv_beta1": vv_beta1,
                    "onset": onset, "impact": impact,
                })
    return subjects, records


def split_subjects(subjects, n_test=N_TEST_SUBJECTS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    test_subjects = sorted(shuffled[:n_test])
    train_subjects = sorted(shuffled[n_test:])
    return train_subjects, test_subjects


# ----------------------------------------------------------------------------- #
# Step 2: per-detector tuning (train only) + firing computation (all trials)
# ----------------------------------------------------------------------------- #
def sens_spec(fire_before_impact_flags_fall, fires_at_all_flags_adl):
    sens = np.mean(fire_before_impact_flags_fall) if len(fire_before_impact_flags_fall) else np.nan
    spec = 1 - np.mean(fires_at_all_flags_adl) if len(fires_at_all_flags_adl) else np.nan
    return sens, spec


def tune_threshold(train_records):
    falls = [r for r in train_records if r["is_fall"]]
    adls = [r for r in train_records if not r["is_fall"]]
    all_acc_m = np.concatenate([r["acc_m"] for r in falls])
    grid = np.linspace(np.nanpercentile(all_acc_m, 1), np.nanpercentile(all_acc_m, 60), 25)

    best_score, best_c = -np.inf, grid[0]
    for c in grid:
        fall_ok = []
        for r in falls:
            idx = threshold_fire(r["acc_m"], c)
            fire_frame = r["frames"][idx] if idx is not None else None
            fall_ok.append(fire_frame is not None and fire_frame < r["impact"])
        adl_fire = []
        for r in adls:
            idx = threshold_fire(r["acc_m"], c)
            adl_fire.append(idx is not None)
        sens, spec = sens_spec(fall_ok, adl_fire)
        score = (sens + spec) / 2
        if score > best_score:
            best_score, best_c = score, c
    return best_c, best_score


def tune_cusum(train_records):
    falls = [r for r in train_records if r["is_fall"]]
    adls = [r for r in train_records if not r["is_fall"]]

    all_beta1 = np.concatenate([r["vv_beta1"][~np.isnan(r["vv_beta1"])] for r in falls + adls])
    scale = np.nanstd(all_beta1)
    if scale == 0 or np.isnan(scale):
        scale = 1e-3

    k_grid = [0.25 * scale, 0.5 * scale, 1.0 * scale]
    h_grid = [m * scale for m in (2, 4, 6, 8, 12, 16, 24, 32)]

    best_score, best_k, best_h = -np.inf, k_grid[0], h_grid[0]
    for k in k_grid:
        # S+/S- depend only on k, not h -- compute once per k, reuse across all h.
        fall_series = [cusum_series(r["vv_beta1"], k) for r in falls]
        adl_series = [cusum_series(r["vv_beta1"], k) for r in adls]
        for h in h_grid:
            fall_ok = []
            for r, (s_plus, s_minus) in zip(falls, fall_series):
                t = first_cusum_fire(s_plus, s_minus, h)
                fire_frame = r["frames"][t] if t is not None else None
                fall_ok.append(fire_frame is not None and fire_frame < r["impact"])
            adl_fire = []
            for (s_plus, s_minus) in adl_series:
                t = first_cusum_fire(s_plus, s_minus, h)
                adl_fire.append(t is not None)
            sens, spec = sens_spec(fall_ok, adl_fire)
            score = (sens + spec) / 2
            if score > best_score:
                best_score, best_k, best_h = score, k, h
    return best_k, best_h, best_score


def cusum_series(beta1_vv, k, baseline_frames=100):
    valid = beta1_vv[~np.isnan(beta1_vv)]
    bf = baseline_frames if valid.size >= baseline_frames else max(valid.size // 2, 1)
    mu = valid[:bf].mean() if valid.size else 0.0
    x = np.nan_to_num(beta1_vv, nan=mu)
    dev = x - mu
    acc = np.frompyfunc(lambda a, b: a + b if a + b > 0 else 0.0, 2, 1)
    s_plus = acc.accumulate(dev - k, dtype=object).astype(float)
    s_minus = acc.accumulate(-dev - k, dtype=object).astype(float)
    return s_plus, s_minus


def first_cusum_fire(s_plus, s_minus, h):
    fp = np.where(s_plus > h)[0]
    fm = np.where(s_minus > h)[0]
    tp = int(fp[0]) if len(fp) else None
    tm = int(fm[0]) if len(fm) else None
    if tp is None and tm is None:
        return None
    if tm is None or (tp is not None and tp <= tm):
        return tp
    return tm


def tune_shapelet(train_records, rng):
    falls = [r for r in train_records if r["is_fall"]]
    adls = [r for r in train_records if not r["is_fall"]]

    # Candidate pool: onset-centered windows from train fall trials (where the
    # discriminative shape lives -- see plan). Give some pre/post margin.
    onset_windows = []
    for r in falls:
        frames, acc_m, onset = r["frames"], r["acc_m"], r["onset"]
        onset_idx = int(np.searchsorted(frames, onset))
        lo = max(0, onset_idx - SHAPELET_LEN)
        hi = min(len(acc_m), onset_idx + 3 * SHAPELET_LEN)
        if hi - lo >= SHAPELET_LEN:
            onset_windows.append(acc_m[lo:hi])

    # Scoring pools: WHOLE-TRIAL signals (not pre-cut windows) -- discover_shapelet
    # scores each candidate by its minimum distance to any position within each
    # sampled trial, which correctly finds wherever in that trial the candidate
    # actually fits best (see min_dist_to_trials' docstring for why a mean
    # distance to pre-cut windows is fragile here).
    def sample_trial_signals(records, n):
        idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
        return [records[i]["acc_m"] for i in idx]

    adl_trial_sample = sample_trial_signals(adls, 60)
    fall_trial_sample = sample_trial_signals(falls, 60)

    shapelet, score = discover_shapelet(onset_windows, adl_trial_sample, fall_trial_sample,
                                         n_candidates=150, rng=rng)

    # Tune c_match on train: grid over percentiles of observed distances.
    fall_min_dist = []
    for r in falls:
        if len(r["acc_m"]) < SHAPELET_LEN:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(r["acc_m"], SHAPELET_LEN)
        fall_min_dist.append(window_distance(shapelet, windows).min())
    grid = np.percentile(fall_min_dist, np.linspace(5, 95, 25))

    best_score, best_c = -np.inf, grid[len(grid) // 2]
    for c in grid:
        fall_ok = []
        for r in falls:
            idx = shapelet_fire(r["acc_m"], shapelet, c)
            fire_frame = r["frames"][idx] if idx is not None else None
            fall_ok.append(fire_frame is not None and fire_frame < r["impact"])
        adl_fire = []
        for r in adls:
            idx = shapelet_fire(r["acc_m"], shapelet, c)
            adl_fire.append(idx is not None)
        sens, spec = sens_spec(fall_ok, adl_fire)
        s = (sens + spec) / 2
        if s > best_score:
            best_score, best_c = s, c
    return shapelet, best_c, best_score


# ----------------------------------------------------------------------------- #
# Step 3: apply tuned detectors to every trial, build the results table
# ----------------------------------------------------------------------------- #
def evaluate_all(records, train_subjects, test_subjects, c_thresh, k_cusum, h_cusum,
                  shapelet, c_match):
    train_set, test_set = set(train_subjects), set(test_subjects)
    rows = []
    for r in records:
        split = "train" if r["subject"] in train_set else ("test" if r["subject"] in test_set else None)
        if split is None:
            continue
        frames, acc_m, vv_beta1 = r["frames"], r["acc_m"], r["vv_beta1"]

        idx_t = threshold_fire(acc_m, c_thresh)
        frame_t = int(frames[idx_t]) if idx_t is not None else None

        s_plus, s_minus = cusum_series(vv_beta1, k_cusum)
        idx_c = first_cusum_fire(s_plus, s_minus, h_cusum)
        frame_c = int(frames[idx_c]) if idx_c is not None else None
        side_c = None
        if idx_c is not None:
            side_c = "plus" if s_plus[idx_c] > h_cusum else "minus"

        idx_s = shapelet_fire(acc_m, shapelet, c_match)
        frame_s = int(frames[idx_s]) if idx_s is not None else None

        fire_frames = sorted(f for f in (frame_t, frame_c, frame_s) if f is not None)
        frame_any = fire_frames[0] if len(fire_frames) >= 1 else None
        frame_majority = fire_frames[1] if len(fire_frames) >= 2 else None
        frame_unanimous = fire_frames[2] if len(fire_frames) >= 3 else None

        rows.append({
            "subject": r["subject"], "task": r["task"], "trial": r["trial"],
            "is_fall": r["is_fall"], "split": split,
            "impact_frame": r["impact"],
            "threshold_fire_frame": frame_t, "cusum_fire_frame": frame_c,
            "cusum_side": side_c, "shapelet_fire_frame": frame_s,
            "any_fire_frame": frame_any, "majority_fire_frame": frame_majority,
            "unanimous_fire_frame": frame_unanimous,
        })
    return pd.DataFrame(rows)


def compute_metrics(df, fire_col, subset):
    d = df[df["split"] == subset]
    falls = d[d["is_fall"]]
    adls = d[~d["is_fall"]]

    fall_fired_before = (falls[fire_col].notna()) & (falls[fire_col] < falls["impact_frame"])
    sensitivity = fall_fired_before.mean() if len(falls) else np.nan
    specificity = 1 - adls[fire_col].notna().mean() if len(adls) else np.nan

    lead_ms = (falls.loc[fall_fired_before, "impact_frame"]
               - falls.loc[fall_fired_before, fire_col]) * 10
    mean_lead = lead_ms.mean() if len(lead_ms) else np.nan

    return sensitivity, specificity, mean_lead, len(falls), len(adls)


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def make_plots(df, shapelet, adl_task_labels):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rules = [("threshold_fire_frame", "Threshold"), ("cusum_fire_frame", "CUSUM"),
             ("shapelet_fire_frame", "Shapelet"), ("any_fire_frame", "ANY (≥1)"),
             ("majority_fire_frame", "MAJORITY (≥2)"), ("unanimous_fire_frame", "UNANIMOUS (=3)")]

    # 1. Sensitivity/specificity bars, test set, vs paper reference.
    sens_list, spec_list, labels = [], [], []
    for col, label in rules:
        s, p, _, _, _ = compute_metrics(df, col, "test")
        sens_list.append(s); spec_list.append(p); labels.append(label)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, sens_list, width, label="Sensitivity", color="#2ecc71")
    ax.bar(x + width / 2, spec_list, width, label="Specificity", color="#3498db")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Ensemble Trigger: Sensitivity/Specificity on held-out test subjects\n"
                  "(dashed lines: paper's own Threshold/SVM/ConvLSTM benchmarks)")
    colors_ref = {"Threshold (paper)": "#e74c3c", "SVM (paper)": "#9b59b6", "ConvLSTM (paper)": "#f39c12"}
    for name, (sens, spec, _) in PAPER_REFERENCE.items():
        ax.axhline(sens, color=colors_ref[name], linestyle="--", linewidth=1, alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_specificity.png"), dpi=300)
    plt.close()

    # 2. Per-activity false-alarm heatmap (test set ADLs only).
    test_adl = df[(df["split"] == "test") & (~df["is_fall"])]
    det_cols = ["threshold_fire_frame", "cusum_fire_frame", "shapelet_fire_frame"]
    det_names = ["Threshold", "CUSUM", "Shapelet"]
    tasks = sorted(test_adl["task"].unique())
    mat = np.zeros((len(tasks), 3))
    for i, t in enumerate(tasks):
        sub = test_adl[test_adl["task"] == t]
        for j, col in enumerate(det_cols):
            mat[i, j] = sub[col].notna().mean() if len(sub) else np.nan
    fig, ax = plt.subplots(figsize=(6, max(6, len(tasks) * 0.35)))
    sns.heatmap(mat, xticklabels=det_names,
                yticklabels=[adl_task_labels.get(t, str(t)) for t in tasks],
                cmap="RdYlGn_r", vmin=0, vmax=1, annot=True, fmt=".2f",
                cbar_kws={"label": "False-positive rate"}, ax=ax)
    ax.set_title("Per-Activity False-Alarm Rate by Detector (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "false_alarm_breakdown.png"), dpi=300)
    plt.close()

    # 3. The learned shapelet.
    plt.figure(figsize=(8, 5))
    t_ms = np.arange(len(shapelet)) * 10
    plt.plot(t_ms, shapelet, color="#c0392b", linewidth=2)
    plt.xlabel("Time within shapelet window (ms)")
    plt.ylabel("ACC_M (g)")
    plt.title(f"Learned Shapelet ({len(shapelet)} frames / {len(shapelet)*10} ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learned_shapelet.png"), dpi=300)
    plt.close()

    # 4. Lead time comparison.
    lead_means, lead_labels = [], []
    for col, label in rules:
        _, _, lead, _, _ = compute_metrics(df, col, "test")
        lead_means.append(lead if lead == lead else 0)
        lead_labels.append(label)
    plt.figure(figsize=(10, 5))
    plt.bar(lead_labels, lead_means, color="#1abc9c")
    for name, (_, _, lead) in PAPER_REFERENCE.items():
        plt.axhline(lead, linestyle="--", linewidth=1, alpha=0.6, label=f"{name} ({lead}ms)")
    plt.ylabel("Mean lead time (ms)")
    plt.xticks(rotation=20)
    plt.title("Mean Pre-Impact Lead Time (test set)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lead_time_comparison.png"), dpi=300)
    plt.close()

    print(f"\nSaved 4 plots to {PLOTS_DIR}/")


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Loading all trials (reusing paper_threshold_validation pipeline)...")
    subjects, records = load_all_trials()
    n_fall = sum(1 for r in records if r["is_fall"])
    n_adl = len(records) - n_fall
    print(f"Loaded {len(records)} trials ({n_fall} fall, {n_adl} ADL) across {len(subjects)} subjects.")

    train_subjects, test_subjects = split_subjects(subjects)
    print(f"\nTrain subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects  ({len(test_subjects)}): {test_subjects}")
    train_set = set(train_subjects)
    train_records = [r for r in records if r["subject"] in train_set]
    print(f"Train trials: {len(train_records)}")

    rng = np.random.default_rng(RNG_SEED)

    print("\nTuning Threshold (ACC_M) on train subjects...")
    c_thresh, score_t = tune_threshold(train_records)
    print(f"  best c = {c_thresh:.3f} g  (train score={score_t:.3f})")

    print("Tuning CUSUM (VV beta1) on train subjects...")
    k_cusum, h_cusum, score_c = tune_cusum(train_records)
    print(f"  best k={k_cusum:.5f}, h={h_cusum:.5f}  (train score={score_c:.3f})")

    print("Discovering + tuning Shapelet (ACC_M) on train subjects...")
    shapelet, c_match, score_s = tune_shapelet(train_records, rng)
    print(f"  best c_match={c_match:.4f}  (train score={score_s:.3f})")

    print("\n=== Sanity checks (tuned detectors on known real trials) ===")
    standing = next((r for r in records if r["subject"] == subjects[0] and r["task"] == 1), None)
    if standing is not None:
        t_idx = threshold_fire(standing["acc_m"], c_thresh)
        s_plus, s_minus = cusum_series(standing["vv_beta1"], k_cusum)
        c_idx = first_cusum_fire(s_plus, s_minus, h_cusum)
        sh_idx = shapelet_fire(standing["acc_m"], shapelet, c_match)
        print(f"  Standing SA{subjects[0]:02d}/D01: Threshold fires={t_idx is not None}, "
              f"CUSUM fires={c_idx is not None}, Shapelet fires={sh_idx is not None} (expect all False)")
    known_fall = next((r for r in records if r["subject"] == 6 and r["task"] == 32 and r["trial"] == 1), None)
    if known_fall is not None:
        t_idx = threshold_fire(known_fall["acc_m"], c_thresh)
        s_plus, s_minus = cusum_series(known_fall["vv_beta1"], k_cusum)
        c_idx = first_cusum_fire(s_plus, s_minus, h_cusum)
        sh_idx = shapelet_fire(known_fall["acc_m"], shapelet, c_match)
        impact = known_fall["impact"]
        t_ok = t_idx is not None and known_fall["frames"][t_idx] < impact
        c_ok = c_idx is not None and known_fall["frames"][c_idx] < impact
        sh_ok = sh_idx is not None and known_fall["frames"][sh_idx] < impact
        print(f"  Known slip fall SA06/T32: Threshold pre-impact fire={t_ok}, "
              f"CUSUM pre-impact fire={c_ok}, Shapelet pre-impact fire={sh_ok} (expect all True)")

    print("\nEvaluating all detectors + ensemble voting rules on every trial...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df = evaluate_all(records, train_subjects, test_subjects, c_thresh, k_cusum,
                           h_cusum, shapelet, c_match)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {len(df)} trial results to {RESULTS_CSV}")

    print("\n=== Test-set performance (held-out subjects, never used for tuning) ===")
    print(f"{'Method':<20}{'Sensitivity':>14}{'Specificity':>14}{'Lead(ms)':>12}")
    rules = [("threshold_fire_frame", "Threshold"), ("cusum_fire_frame", "CUSUM"),
             ("shapelet_fire_frame", "Shapelet"), ("any_fire_frame", "ANY (>=1)"),
             ("majority_fire_frame", "MAJORITY (>=2)"), ("unanimous_fire_frame", "UNANIMOUS (=3)")]
    for col, label in rules:
        sens, spec, lead, n_f, n_a = compute_metrics(df, col, "test")
        print(f"{label:<20}{sens:>13.1%} {spec:>13.1%} {lead:>11.0f}")
    print(f"\nTest set: {n_f} fall trials, {n_a} ADL trials")

    print("\n=== Paper's own reference benchmarks (Table 3) ===")
    for name, (sens, spec, lead) in PAPER_REFERENCE.items():
        print(f"{name:<20}{sens:>13.1%} {spec:>13.1%} {lead:>11.0f}")

    adl_task_labels = {1: "D01", 2: "D02", 3: "D03", 4: "D04", 5: "D05", 6: "D06", 7: "D07",
                       8: "D08", 9: "D09", 10: "D10", 11: "D11", 12: "D12", 13: "D13",
                       14: "D14", 15: "D15", 16: "D16", 17: "D17", 18: "D18", 19: "D19",
                       35: "D20", 36: "D21"}

    make_plots(df, shapelet, adl_task_labels)

    print("\n=== Per-activity false-alarm summary (test set) ===")
    test_adl = df[(df["split"] == "test") & (~df["is_fall"])]
    for col, label in [("threshold_fire_frame", "Threshold"), ("cusum_fire_frame", "CUSUM"),
                        ("shapelet_fire_frame", "Shapelet")]:
        by_task = test_adl.groupby("task")[col].apply(lambda s: s.notna().mean())
        worst = by_task.sort_values(ascending=False).head(3)
        print(f"  {label} worst 3 ADL tasks: " +
              ", ".join(f"{adl_task_labels.get(t,t)}={r:.0%}" for t, r in worst.items()))
