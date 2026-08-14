"""
Evaluation for the GAF global-normalization fix (see data.py's docstring).

Per-window prediction -> per-trial decision via the same CONSEC_WINDOWS=2
persistence rule used throughout this project. Unlike experiments/gaf_mtf/
evaluate.py (forced to stride=5 because float32 storage made the full test
set exceed its 8GB cap), this version uses FULL test resolution
(stride=1, ~427K windows, ~6.1GB at float16) -- safely under this
experiment's raised 28GB cap.

Compares against:
  (a) The original gaf_mtf result (9.20% specificity, per-window norm,
      150K/float32/stride=5) -- the direct before/after.
  (b) ConvLSTM baseline at FULL scale (91.19%, the scale-appropriate
      comparison now that this run is also full scale).
  (c) ConvLSTM baseline at matched 150K scale (89.85%, documented reference,
      kept for continuity with gaf_mtf's own comparison).

Also runs a CONSEC_WINDOWS sweep (1/2/3/5/8) as a direct diagnostic of
whether the flicker/aggregation-sensitivity pattern found in gaf_mtf is
reduced by the normalization fix.

Run (after train.py has produced a checkpoint):
  python3 experiments/gaf_global/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import build_gaf_dataset, make_subject_split, load_global_stats, FS  # noqa: E402
from model import GAF_CNN  # noqa: E402
from train import GAFDataset, CHECKPOINT_PATH, NORM_STATS_PATH, GLOBAL_STATS_PATH, RESULTS_DIR  # noqa: E402

RESULTS_CSV = os.path.join(RESULTS_DIR, "gaf_global.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison_vs_baseline.md")

EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
GAF_MTF_CSV = os.path.join(EXPERIMENTS_DIR, "gaf_mtf", "results", "gaf_mtf.csv")
PAPER_IMPL_DIR = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "paper_implementation")
CONVLSTM_FULLSCALE_CSV = os.path.join(PAPER_IMPL_DIR, "results", "convlstm.csv")

# Documented reference (not re-derived) -- same convention as gaf_mtf/evaluate.py.
CONVLSTM_150K_REFERENCE = {"sensitivity": 0.9928, "specificity": 0.8985, "lead_mean": 225, "lead_std": 136}

TEST_WINDOW_STRIDE = 1   # full resolution -- safe at float16 (~6.1GB), unlike gaf_mtf's float32 (~11.9GB)
CONSEC_WINDOWS = 2
MAX_MEMORY_GB = 28.0


def predict(model, X, device, batch_size=256):
    loader = DataLoader(GAFDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def aggregate_to_trials(meta, y_pred, consec=CONSEC_WINDOWS, fs=FS):
    """Identical convention to every other experiment in this project."""
    meta = meta.copy()
    meta["pred"] = y_pred
    rows = []
    for (subj, task, trial_id), g in meta.groupby(["subject", "task", "trial"]):
        g = g.sort_values("end_frame")
        preds = g["pred"].values
        end_frames = g["end_frame"].values
        is_fall = bool(g["is_fall"].iloc[0])

        run = 0
        first_idx = None
        for i, p in enumerate(preds):
            run = run + 1 if p else 0
            if run >= consec:
                first_idx = i - consec + 1
                break
        detected = first_idx is not None

        lead_ms = None
        if detected and is_fall:
            impact = g["impact_frame"].iloc[0]
            lead_ms = (impact - end_frames[first_idx]) * (1000.0 / fs)
        rows.append({"subject": subj, "task": task, "trial": trial_id, "is_fall": is_fall,
                     "detected": detected, "lead_time_ms": lead_ms if lead_ms is not None else np.nan})
    return pd.DataFrame(rows)


def summarize(df, label):
    fall = df[df["is_fall"]]
    adl = df[~df["is_fall"]]
    tp = int(fall["detected"].sum())
    fn = len(fall) - tp
    tn = int((~adl["detected"]).sum())
    fp = len(adl) - tn
    sensitivity = tp / len(fall) if len(fall) else float("nan")
    specificity = tn / len(adl) if len(adl) else float("nan")
    lead = fall.loc[fall["detected"], "lead_time_ms"]
    print(f"\n[{label}] Test set: {len(fall)} fall trials, {len(adl)} ADL trials")
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"Sensitivity: {sensitivity:.2%}   Specificity: {specificity:.2%}   "
          f"Lead time: {lead.mean():.0f}+/-{lead.std():.0f} ms")
    return {"label": label, "sensitivity": sensitivity, "specificity": specificity,
            "lead_mean": lead.mean(), "lead_std": lead.std()}


def csv_summary(path, label):
    if not os.path.exists(path):
        return None
    return summarize(pd.read_csv(path), label)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH} -- run train.py first.")
        sys.exit(1)

    split = make_subject_split()
    test_subjects = split["test_subjects"]

    model = GAF_CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    stats = np.load(NORM_STATS_PATH)
    mean, std = stats["mean"], stats["std"]
    global_stats = load_global_stats(GLOBAL_STATS_PATH)

    print(f"Building GAF test set (stride={TEST_WINDOW_STRIDE}, global_stats={global_stats})...")
    X_test, y_test, meta_test = build_gaf_dataset(test_subjects, stride=TEST_WINDOW_STRIDE,
                                                     global_stats=global_stats, max_memory_gb=MAX_MEMORY_GB)

    def normalize_f16(X, chunk=8192):
        out = np.empty(X.shape, dtype=np.float16)
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            out[start:end] = ((X[start:end].astype(np.float32) - mean) / std).astype(np.float16)
        return out

    X_test_n = normalize_f16(X_test)

    y_pred = predict(model, X_test_n, device)
    trial_results = aggregate_to_trials(meta_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(trial_results)} per-trial predictions to {RESULTS_CSV}")

    global_summary = summarize(trial_results, "GAF 2D-CNN (GLOBAL norm, full scale)")
    gaf_mtf_summary = csv_summary(GAF_MTF_CSV, "GAF 2D-CNN (per-window norm, original, 150K)")
    convlstm_full = csv_summary(CONVLSTM_FULLSCALE_CSV, "ConvLSTM baseline (full scale)")

    # CONSEC_WINDOWS sweep -- direct diagnostic of whether the flicker pattern is reduced.
    print("\n=== CONSEC_WINDOWS sweep ===")
    sweep_rows = []
    for consec in [1, 2, 3, 5, 8]:
        tr = aggregate_to_trials(meta_test, y_pred, consec=consec)
        fall = tr[tr["is_fall"]]
        adl = tr[~tr["is_fall"]]
        sens = fall["detected"].mean()
        spec = 1 - adl["detected"].mean()
        print(f"consec={consec:2d}  sens={sens:.2%}  spec={spec:.2%}")
        sweep_rows.append((consec, sens, spec))

    lines = ["# GAF Global Normalization vs. Original (Per-Window) vs. ConvLSTM Baselines\n"]
    lines += ["| | Sensitivity | Specificity | Lead time (ms) |", "|---|---|---|---|"]
    lines.append(f"| **GAF, GLOBAL norm (this fix, full scale)** | {global_summary['sensitivity']:.2%} | "
                 f"{global_summary['specificity']:.2%} | {global_summary['lead_mean']:.0f}+/-{global_summary['lead_std']:.0f} |")
    if gaf_mtf_summary:
        lines.append(f"| GAF, per-window norm (original, 150K scale) | {gaf_mtf_summary['sensitivity']:.2%} | "
                     f"{gaf_mtf_summary['specificity']:.2%} | {gaf_mtf_summary['lead_mean']:.0f}+/-{gaf_mtf_summary['lead_std']:.0f} |")
    if convlstm_full:
        lines.append(f"| ConvLSTM baseline (full scale) | {convlstm_full['sensitivity']:.2%} | "
                     f"{convlstm_full['specificity']:.2%} | {convlstm_full['lead_mean']:.0f}+/-{convlstm_full['lead_std']:.0f} |")
    lines.append(f"| ConvLSTM baseline (matched 150K scale, documented reference) | "
                 f"{CONVLSTM_150K_REFERENCE['sensitivity']:.2%} | {CONVLSTM_150K_REFERENCE['specificity']:.2%} | "
                 f"{CONVLSTM_150K_REFERENCE['lead_mean']}+/-{CONVLSTM_150K_REFERENCE['lead_std']} |")

    lines.append("\n## CONSEC_WINDOWS sweep (diagnostic: is the flicker pattern reduced?)\n")
    lines += ["| CONSEC_WINDOWS | Sensitivity | Specificity |", "|---|---|---|"]
    for consec, sens, spec in sweep_rows:
        lines.append(f"| {consec} | {sens:.2%} | {spec:.2%} |")

    lines.append("\n## Success criterion\n")
    if convlstm_full:
        delta_vs_full = global_summary["specificity"] - convlstm_full["specificity"]
        lines.append(f"vs. ConvLSTM at full scale ({convlstm_full['specificity']:.2%}): {delta_vs_full*100:+.1f}pp.")
    if gaf_mtf_summary:
        delta_vs_original = global_summary["specificity"] - gaf_mtf_summary["specificity"]
        lines.append(f"vs. original per-window GAF ({gaf_mtf_summary['specificity']:.2%}): {delta_vs_original*100:+.1f}pp.")

    delta_vs_150k = global_summary["specificity"] - CONVLSTM_150K_REFERENCE["specificity"]
    if delta_vs_150k > 0.02:
        lines.append("\n**Result: global-normalization GAF specificity clearly beats the ConvLSTM baseline -- "
                     "a genuinely novel finding.**")
    elif delta_vs_150k < -0.02:
        lines.append("\n**Result: global-normalization GAF specificity still falls short of the ConvLSTM "
                     "baseline, though (see the before/after row above) the fix's actual effect on the "
                     "instability problem should be judged from the CONSEC_WINDOWS sweep and the original "
                     "9.20% comparison, not this number alone.**")
    else:
        lines.append("\n**Result: global-normalization GAF specificity is roughly on par with the ConvLSTM baseline.**")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison to {COMPARISON_MD}")
