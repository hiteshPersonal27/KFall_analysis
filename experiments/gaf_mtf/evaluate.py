"""
Evaluation for the GAF 2D-encoding experiment (see
../../docs/GAF_MTF_Implementation_Plan.md).

Per-window prediction -> per-trial decision via the same CONSEC_WINDOWS=2
persistence rule used throughout this project. Compares against:
  (a) The ConvLSTM baseline at MATCHED 150K training scale (documented
      reference from paper_implementation/README.md, not re-derived --
      that intermediate checkpoint's per-trial CSV wasn't separately saved).
  (b) cwt_lstm's result at the same matched 150K scale (read from its saved
      CSV if present) -- both are "richer 2D representation" experiments
      answering a related question, per the plan.

Run (after train.py has produced a checkpoint):
  python3 experiments/gaf_mtf/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import build_gaf_dataset, make_subject_split, FS  # noqa: E402
from model import GAF_CNN  # noqa: E402
from train import GAFDataset, CHECKPOINT_PATH, NORM_STATS_PATH, RESULTS_DIR  # noqa: E402

RESULTS_CSV = os.path.join(RESULTS_DIR, "gaf_mtf.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison_vs_baseline.md")

EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
CWT_LSTM_CSV = os.path.join(EXPERIMENTS_DIR, "cwt_lstm", "results", "cwt_lstm.csv")

# From paper_implementation/README.md's "ConvLSTM, 150K-window training" row
# -- documented reference, not re-derived (same convention used by
# experiments/conv_transformer_nopool/evaluate.py).
CONVLSTM_150K_REFERENCE = {"sensitivity": 0.9928, "specificity": 0.8985, "lead_mean": 225, "lead_std": 136}

TEST_WINDOW_STRIDE = 5  # NOT stride=1: full test set (~427K windows) needs ~11.9GB at
                          # (3,50,50) float32/window -- over data.py's 8GB safety cap
                          # (caught by the memory guard, not a silent OOM -- see README).
                          # stride=5 (~86K windows, ~2.4GB) stays safe, same precedent as
                          # experiments/cwt_lstm/evaluate.py's identical fix.
CONSEC_WINDOWS = 2


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

    print(f"Building GAF test set (stride={TEST_WINDOW_STRIDE})...")
    X_test, y_test, meta_test = build_gaf_dataset(test_subjects, stride=TEST_WINDOW_STRIDE)
    X_test_n = (X_test - mean) / std

    y_pred = predict(model, X_test_n, device)
    trial_results = aggregate_to_trials(meta_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(trial_results)} per-trial predictions to {RESULTS_CSV}")

    gaf_summary = summarize(trial_results, "GAF 2D-CNN (150K scale)")
    cwt_summary = csv_summary(CWT_LSTM_CSV, "cwt_lstm (matched 150K scale)")

    lines = ["# GAF 2D-CNN vs. ConvLSTM Baseline vs. cwt_lstm (matched 150K scale)\n"]
    lines += ["| | Sensitivity | Specificity | Lead time (ms) |", "|---|---|---|---|"]
    lines.append(f"| **GAF 2D-CNN (this run)** | {gaf_summary['sensitivity']:.2%} | "
                 f"{gaf_summary['specificity']:.2%} | {gaf_summary['lead_mean']:.0f}+/-{gaf_summary['lead_std']:.0f} |")
    lines.append(f"| ConvLSTM baseline, matched 150K scale (documented reference) | "
                 f"{CONVLSTM_150K_REFERENCE['sensitivity']:.2%} | {CONVLSTM_150K_REFERENCE['specificity']:.2%} | "
                 f"{CONVLSTM_150K_REFERENCE['lead_mean']}+/-{CONVLSTM_150K_REFERENCE['lead_std']} |")
    if cwt_summary:
        lines.append(f"| cwt_lstm (2D CWT scalogram, matched 150K scale) | "
                     f"{cwt_summary['sensitivity']:.2%} | {cwt_summary['specificity']:.2%} | "
                     f"{cwt_summary['lead_mean']:.0f}+/-{cwt_summary['lead_std']:.0f} |")

    lines.append("\n## Success criterion: does GAF specificity meaningfully improve over, "
                 "meet, or fall short of the matched-scale ConvLSTM baseline?\n")
    delta_vs_convlstm = gaf_summary["specificity"] - CONVLSTM_150K_REFERENCE["specificity"]
    lines.append(f"vs. ConvLSTM at matched 150K scale (89.85%): {delta_vs_convlstm*100:+.1f}pp.")
    if cwt_summary:
        delta_vs_cwt = gaf_summary["specificity"] - cwt_summary["specificity"]
        lines.append(f"vs. cwt_lstm at matched 150K scale ({cwt_summary['specificity']:.2%}): {delta_vs_cwt*100:+.1f}pp.")

    if delta_vs_convlstm > 0.02:
        lines.append("\n**Result: GAF specificity clearly beats the ConvLSTM baseline -- a genuinely "
                     "novel finding worth a full-scale follow-up run.**")
    elif delta_vs_convlstm < -0.02:
        lines.append("\n**Result: GAF specificity falls short of the ConvLSTM baseline -- consistent with "
                     "the CWT precedent. Evidence that the specificity plateau is not explained by "
                     "\"signal shape isn't explicit enough\"; points back toward the LSTM's role, "
                     "aggregation rule, or training-recipe details as the more likely explanations.**")
    else:
        lines.append("\n**Result: GAF specificity is roughly on par with the ConvLSTM baseline at this scale.**")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison to {COMPARISON_MD}")
