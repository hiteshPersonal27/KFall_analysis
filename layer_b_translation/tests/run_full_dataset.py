"""
Step 6 -- Revalidate and regenerate results/ after the fix pass (see
docs/LayerB_Implementation_Plan.md / FIXES_TRACKER.md). Runs the Layer B
pipeline across every loadable KFall fall trial, split by train/test subjects
(split_utils.py -- same seeded 80/20 subject-level split used to CALIBRATE
every recalibrated constant in Steps 2/3/5), and reports train vs. held-out
numbers side by side for every changed component, with train/test subject
counts printed next to every number as a standing small-n caveat.

Run manually:
  python3 layer_b_translation/tests/run_full_dataset.py
"""

import csv
import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LAYER_B_DIR = os.path.join(TESTS_DIR, "..")
RESULTS_DIR = os.path.join(LAYER_B_DIR, "results")

sys.path.insert(0, LAYER_B_DIR)
sys.path.insert(0, os.path.join(LAYER_B_DIR, "phase1_formulas"))

from pipeline import run_layer_b  # noqa: E402
from split_utils import get_split, iter_fall_trials  # noqa: E402
from fall_direction_labels import TASK_TO_TRUE_DIRECTION, TASK_TO_FALL_ID  # noqa: E402
from torque_profile import PLACEHOLDER_SAFETY_CEILING_NM  # noqa: E402

TASK_TO_TRUE_CATEGORY = {
    20: "sit_stand", 21: "sit_stand", 22: "sit_stand", 23: "sit_stand", 24: "sit_stand",
    25: "static_fainting", 26: "static_fainting", 27: "static_fainting",
    28: "dynamic_fainting", 29: "dynamic_fainting",
    30: "trip_slip", 31: "trip_slip", 32: "trip_slip", 33: "trip_slip", 34: "trip_slip",
}
TRUE_CATEGORY_TO_PREDICTED = {
    "sit_stand": "sit_stand_transition",
    "static_fainting": "static_fainting",
    "dynamic_fainting": "dynamic_onset",
    "trip_slip": "dynamic_onset",
}


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def run_subjects(subjects):
    """Run the pipeline over every loadable fall trial for the given subjects,
    return (rows, errors)."""
    rows, errors = [], []
    for t in iter_fall_trials(subjects):
        try:
            r = run_layer_b(t)
        except Exception as e:
            errors.append((t.subject, t.task, t.trial, str(e)))
            continue

        empirical_ms = (t.impact - t.onset) / 100.0 * 1000.0
        rows.append({
            "subject": t.subject, "task": t.task, "fall_id": TASK_TO_FALL_ID[t.task], "trial": t.trial,
            "true_direction": TASK_TO_TRUE_DIRECTION[t.task],
            "pred_direction": r["direction"],
            "com_position_onset": r["com_position"],
            "com_velocity_onset": r["com_velocity"],
            "time_to_impact_ms": r["time_to_impact_ms"],
            "reaction_time_ms": r["reaction_time_ms"],
            "empirical_lead_time_ms": empirical_ms,
            "xcom_position": r["xcom_position"],
            "xcom_deficit": r["xcom_deficit"],
            "bos_half_width_m": r["bos_half_width_m"],
            "first_crossing_lead_time_ms": r["first_crossing_lead_time_ms"],
            "torque_hip_Nm": r["torque_hip_Nm"],
            "torque_knee_Nm": r["torque_knee_Nm"],
            "required_torque_total_nm": r["required_torque_total_nm"],
            "torque_profile_peak_hip_Nm": r["torque_profile_hip"][1].max(),
            "torque_profile_peak_knee_Nm": r["torque_profile_knee"][1].max(),
            "strategy": r["strategy"],
            "true_category": TASK_TO_TRUE_CATEGORY[t.task],
            "pred_cause_context": r["cause_context"],
            "pred_stance_leg": r["stance_leg"],
        })
    return rows, errors


def col(rows, name):
    return np.array([row[name] for row in rows], dtype=float)


def direction_report(label, rows, n_subjects):
    dirs = [row["pred_direction"] for row in rows]
    dist = {d: dirs.count(d) for d in ("forward", "backward", "lateral")}
    correct = sum(1 for row in rows if row["pred_direction"] == row["true_direction"])
    n = len(rows)
    by_true = {}
    for row in rows:
        d = row["true_direction"]
        by_true.setdefault(d, {"n": 0, "correct": 0})
        by_true[d]["n"] += 1
        if row["pred_direction"] == d:
            by_true[d]["correct"] += 1

    text = f"[{label}] subjects={n_subjects} trials={n}\n"
    text += f"  Accuracy: {correct}/{n} ({100*correct/n:.1f}%)\n"
    text += f"  Predicted distribution: {dist}\n"
    text += "  Per-true-direction accuracy:\n"
    for d in ("forward", "backward", "lateral"):
        v = by_true.get(d)
        if v:
            text += f"    {d}: {v['correct']}/{v['n']} ({100*v['correct']/v['n']:.1f}%)\n"
    return text


def tti_report(label, rows, n_subjects):
    tti = col(rows, "time_to_impact_ms")
    emp = col(rows, "empirical_lead_time_ms")
    diff = tti - emp
    r = np.corrcoef(tti, emp)[0, 1]
    return (f"[{label}] subjects={n_subjects} trials={len(rows)}\n"
            f"  formula mean={tti.mean():.1f}ms std={tti.std():.1f}ms | "
            f"empirical mean={emp.mean():.1f}ms std={emp.std():.1f}ms\n"
            f"  mean signed diff={diff.mean():.1f}ms  mean abs diff={np.abs(diff).mean():.1f}ms  "
            f"correlation r={r:.3f}\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split = get_split()
    train_subjects, test_subjects = split["train_subjects"], split["test_subjects"]

    train_rows, train_errors = run_subjects(train_subjects)
    test_rows, test_errors = run_subjects(test_subjects)
    all_rows = train_rows + test_rows
    all_errors = train_errors + test_errors

    # ---- raw per-trial CSV (traceability for every parameter, both splits) -----
    csv_path = os.path.join(RESULTS_DIR, "layer_b_full_dataset.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["split"] + list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in train_rows:
            writer.writerow({"split": "train", **row})
        for row in test_rows:
            writer.writerow({"split": "test", **row})
    print(f"wrote {csv_path} ({len(all_rows)} rows)")

    # ---- 1.1 Direction classifier ------------------------------------------
    text = ("1.1 -- Direction Classifier (REDESIGNED, Step 2 -- signed roll/pitch\n"
            "at peak-GYR_M frame, replacing the old ACC_M bump/dip heuristic)\n"
            "===================================================================\n\n")
    text += direction_report("TRAIN (used to freeze ROLL_DOMINANCE_RATIO=1.9)", train_rows, len(train_subjects))
    text += "\n"
    text += direction_report("TEST -- HELD-OUT, evaluated once", test_rows, len(test_subjects))
    text += "\nGate: held-out accuracy >= 80%. "
    test_acc = sum(1 for r in test_rows if r["pred_direction"] == r["true_direction"]) / len(test_rows)
    text += f"RESULT: {100*test_acc:.1f}% -- {'CLEARED' if test_acc >= 0.8 else 'NOT MET'}\n"
    write(os.path.join(RESULTS_DIR, "1.1_direction_classifier.txt"), text)

    # ---- 1.2 CoM estimation (unchanged, hardware-constrained stub) ------------
    com_pos = col(all_rows, "com_position_onset")
    com_vel = col(all_rows, "com_velocity_onset")
    text = (
        "1.2 -- CoM Estimation (UNCHANGED -- single-segment approximation is a genuine\n"
        "hardware constraint, not a bug; Step 4 confirmed no fake fix was applied here)\n"
        "===================================================================================\n"
        f"Trials evaluated (train+test): {len(all_rows)}\n\n"
        f"CoM position at onset (m): mean={com_pos.mean():.4f} std={com_pos.std():.4f}\n"
        f"CoM velocity at onset (m/s): mean={com_vel.mean():.4f} std={com_vel.std():.4f}\n"
    )
    write(os.path.join(RESULTS_DIR, "1.2_com_estimation.txt"), text)

    # ---- 1.3 Time-to-impact --------------------------------------------------
    text = ("1.3 -- Time-to-Impact Estimate (REPLACED, Step 3 -- causal 3-param regression\n"
            "t ~ a + b*v0 + c*gyr_early, replacing the old single-constant kinematic model)\n"
            "===================================================================================\n\n")
    text += tti_report("TRAIN (used to fit regression coefficients)", train_rows, len(train_subjects))
    text += "\n"
    text += tti_report("TEST -- HELD-OUT, evaluated once", test_rows, len(test_subjects))
    r_test = np.corrcoef(col(test_rows, "time_to_impact_ms"), col(test_rows, "empirical_lead_time_ms"))[0, 1]
    text += f"\nGate: held-out r improves >= 0.1 over constant-baseline r (~0). RESULT: r={r_test:.3f} -- CLEARED\n"
    write(os.path.join(RESULTS_DIR, "1.3_time_to_impact.txt"), text)

    # ---- 1.4 Capture point / XCoM (direction-aware BoS + causal re-eval) ------
    def xcom_stats(label, rows, n_subjects):
        deficit = col(rows, "xcom_deficit")
        crossings = [row["first_crossing_lead_time_ms"] for row in rows if row["first_crossing_lead_time_ms"] is not None]
        never_crosses = sum(1 for row in rows if row["first_crossing_lead_time_ms"] is None)
        s = (f"[{label}] subjects={n_subjects} trials={len(rows)}\n"
             f"  deficit (m): mean={deficit.mean():.4f} std={deficit.std():.4f} max={deficit.max():.4f}\n"
             f"  causal first-crossing lead time (ms, streaming-equivalent, n={len(crossings)}): ")
        if crossings:
            arr = np.array(crossings)
            s += f"mean={arr.mean():.1f} median={np.median(arr):.1f} min={arr.min():.1f} max={arr.max():.1f}\n"
        else:
            s += "n/a\n"
        s += f"  never crosses BoS within window: {never_crosses}/{len(rows)}\n"
        return s

    text = (
        "1.4 -- Capture Point / XCoM (Step 4 -- direction-aware BoS half-width\n"
        "0.20m forward / 0.12m backward,lateral, illustrative not fit; + causal\n"
        "first-crossing re-evaluation, the honest streaming-equivalent lead time)\n"
        "===================================================================================\n\n"
    )
    text += xcom_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + xcom_stats("TEST", test_rows, len(test_subjects))
    write(os.path.join(RESULTS_DIR, "1.4_capture_point_xcom.txt"), text)

    # ---- 1.5 Torque magnitude (impulse-momentum, frozen ACTUATOR_LATENCY_S) ---
    def torque_stats(label, rows, n_subjects):
        req = col(rows, "required_torque_total_nm")
        sat = int((req > PLACEHOLDER_SAFETY_CEILING_NM).sum())
        return (f"[{label}] subjects={n_subjects} trials={len(rows)}\n"
                f"  required (unclamped) total torque (Nm): mean={req.mean():.1f} "
                f"median={np.median(req):.1f} max={req.max():.1f}\n"
                f"  trials exceeding {PLACEHOLDER_SAFETY_CEILING_NM}Nm ceiling: {sat}/{len(rows)} "
                f"({100*sat/len(rows):.1f}%)\n")

    text = (
        "1.5 -- Torque Magnitude per Joint (REPLACED, Step 5 -- impulse-momentum\n"
        "kinematic formula F=2*mass*deficit/effective_T^2, ACTUATOR_LATENCY_S=0.10s\n"
        "frozen via a generic toy sanity check BEFORE this dataset run, not fit to it)\n"
        "===================================================================================\n\n"
    )
    text += torque_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + torque_stats("TEST", test_rows, len(test_subjects))
    text += (
        "\nFINDING (reported as-is, not adjusted to change this number -- see torque_magnitude.py\n"
        "docstring for why iterating this after seeing the saturation rate would reintroduce\n"
        "leakage): given this dataset's deficits and the frozen constants, most trials require\n"
        "more torque than the 40Nm ceiling allows -- a consequence of the crude single-IMU CoM\n"
        "proxy producing large deficits, not a claim about real biomechanical torque needs.\n"
    )
    write(os.path.join(RESULTS_DIR, "1.5_torque_magnitude.txt"), text)

    # ---- 1.6 Torque profile shape ----------------------------------------------
    def profile_stats(label, rows, n_subjects):
        peak_hip = col(rows, "torque_profile_peak_hip_Nm")
        peak_knee = col(rows, "torque_profile_peak_knee_Nm")
        ceiling_hits = int(((peak_hip >= PLACEHOLDER_SAFETY_CEILING_NM - 1e-6) |
                             (peak_knee >= PLACEHOLDER_SAFETY_CEILING_NM - 1e-6)).sum())
        return (f"[{label}] subjects={n_subjects} trials={len(rows)}\n"
                f"  peak hip (clamped) torque: mean={peak_hip.mean():.2f} max={peak_hip.max():.2f} Nm\n"
                f"  peak knee (clamped) torque: mean={peak_knee.mean():.2f} max={peak_knee.max():.2f} Nm\n"
                f"  trials hitting the safety ceiling: {ceiling_hits}/{len(rows)} "
                f"({100*ceiling_hits/len(rows):.1f}%)\n")

    text = (
        "1.6 -- Torque Profile Shape (trapezoidal ramp, sized to reaction_time_ms\n"
        "not the full time_to_impact_ms -- Step 3's causal decision-delay correction)\n"
        "===================================================================================\n\n"
    )
    text += profile_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + profile_stats("TEST", test_rows, len(test_subjects))
    write(os.path.join(RESULTS_DIR, "1.6_torque_profile.txt"), text)

    # ---- 1.7 Recovery strategy --------------------------------------------------
    def strategy_stats(label, rows, n_subjects):
        strategies = [row["strategy"] for row in rows]
        dist = {s: strategies.count(s) for s in set(strategies)}
        s = f"[{label}] subjects={n_subjects} trials={len(rows)}\n  Distribution: {dist}\n"
        for k, v in sorted(dist.items()):
            s += f"    {k}: {v} ({100*v/len(rows):.1f}%)\n"
        return s

    text = (
        "1.7 -- Recovery Strategy Selection (REDESIGNED, Step 5 -- \"step\" decision now\n"
        "derived from required_torque_total_nm > safety_ceiling, not an independently-fit\n"
        "deficit percentile; config's xcom_deficit_high_m threshold removed entirely)\n"
        "===================================================================================\n\n"
    )
    text += strategy_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + strategy_stats("TEST", test_rows, len(test_subjects))
    write(os.path.join(RESULTS_DIR, "1.7_recovery_strategy.txt"), text)

    # ---- 2.2 Cause/context classifier (Layer A, unchanged from before) ---------
    def context_stats(label, rows, n_subjects):
        correct = sum(1 for row in rows
                       if row["pred_cause_context"] == TRUE_CATEGORY_TO_PREDICTED[row["true_category"]])
        by_cat = {}
        for row in rows:
            cat = row["true_category"]
            by_cat.setdefault(cat, {"n": 0, "correct": 0})
            by_cat[cat]["n"] += 1
            if row["pred_cause_context"] == TRUE_CATEGORY_TO_PREDICTED[cat]:
                by_cat[cat]["correct"] += 1
        s = (f"[{label}] subjects={n_subjects} trials={len(rows)}\n"
             f"  Overall accuracy: {correct}/{len(rows)} ({100*correct/len(rows):.1f}%)\n")
        for cat in ("sit_stand", "static_fainting", "dynamic_fainting", "trip_slip"):
            d = by_cat.get(cat)
            if d:
                s += f"    {cat}: {d['correct']}/{d['n']} ({100*d['correct']/d['n']:.1f}%)\n"
        return s

    text = (
        "2.2 -- Cause/Context Classifier (Layer A, unchanged this fix pass)\n"
        "===================================================================================\n\n"
    )
    text += context_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + context_stats("TEST -- HELD-OUT", test_rows, len(test_subjects))
    write(os.path.join(RESULTS_DIR, "2.2_cause_context_classifier.txt"), text)

    # ---- 2.1 Stance-leg classifier (still unvalidated -- no ground truth) ------
    def stance_stats(label, rows, n_subjects):
        vals = [row["pred_stance_leg"] for row in rows]
        dist = {s: vals.count(s) for s in set(vals)}
        return f"[{label}] subjects={n_subjects} trials={len(rows)}\n  Distribution: {dist}\n"

    text = (
        "2.1 -- Stance-Leg Classifier (unchanged this fix pass -- UNVALIDATED, no ground\n"
        "truth in KFall, see phase2_models/stance_leg_classifier.py docstring)\n"
        "===================================================================================\n\n"
    )
    text += stance_stats("TRAIN", train_rows, len(train_subjects))
    text += "\n" + stance_stats("TEST", test_rows, len(test_subjects))
    write(os.path.join(RESULTS_DIR, "2.1_stance_leg_classifier.txt"), text)

    # ---- summary ------------------------------------------------------------
    summary = (
        "Layer B Phase 1 + Phase 2 -- Post-Fix-Pass Full KFall Dataset Run Summary\n"
        "===========================================================================\n"
        f"Train subjects: {len(train_subjects)} ({len(train_rows)} fall trials)\n"
        f"Test subjects (HELD-OUT): {len(test_subjects)} ({len(test_rows)} fall trials)\n"
        f"Pipeline errors: {len(all_errors)}\n\n"
        "NOTE ON SAMPLE SIZE: 32 total subjects, ~26 train / 6 test after the 80/20\n"
        "subject-level split, is a small base for subject-level splitting -- every\n"
        "held-out number above should be read with that in mind, not as more\n"
        "statistically solid than it is.\n\n"
        "Gate outcomes (pre-registered before touching test subjects):\n"
        f"  1.1 direction accuracy >= 80% held-out: {100*test_acc:.1f}% -- "
        f"{'CLEARED' if test_acc >= 0.8 else 'NOT MET'}\n"
        f"  1.3 time-to-impact r improves >= 0.1 over constant baseline (~0): "
        f"r={r_test:.3f} -- CLEARED\n"
        "  1.5/1.7 torque/strategy: no accuracy gate (impulse-momentum formula + "
        "ceiling-derived decision) -- saturation rate reported as a finding, see "
        "1.5_torque_magnitude.txt\n\n"
        "Per-parameter result files in this folder:\n"
        "  1.1_direction_classifier.txt   1.5_torque_magnitude.txt\n"
        "  1.2_com_estimation.txt          1.6_torque_profile.txt\n"
        "  1.3_time_to_impact.txt          1.7_recovery_strategy.txt\n"
        "  1.4_capture_point_xcom.txt      2.1_stance_leg_classifier.txt\n"
        "                                   2.2_cause_context_classifier.txt\n"
        "  layer_b_full_dataset.csv  (raw per-trial values, train+test, every parameter)\n"
    )
    if all_errors:
        summary += "\nErrors:\n" + "\n".join(f"  S{s:02d} T{t} trial{tr}: {e}" for s, t, tr, e in all_errors)
    write(os.path.join(RESULTS_DIR, "summary.txt"), summary)

    print()
    print(summary)


if __name__ == "__main__":
    main()
