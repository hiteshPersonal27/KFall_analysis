"""
Evaluation for the CWT + 2D-CNN experiment (see
../../docs/CWT_2DCNN_Implementation_Plan.md).

Per-window prediction -> per-trial aggregation using the SAME consecutive-
window rule established for the paper-faithful models (CONSEC_WINDOWS=2,
see paper_implementation/svm_model.py's comment for the rationale), so
results are directly comparable to the baseline ConvLSTM. Produces a
side-by-side comparison table against the baseline's saved results
(paper_implementation/results/convlstm.csv).

Run (after train.py has produced a checkpoint):
  python3 experiments/cwt_lstm/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import build_cwt_dataset, make_subject_split, FS  # noqa: E402
from model import CWT_ConvLSTM  # noqa: E402
from train import (  # noqa: E402
    set_seed, ScalogramDataset, CHECKPOINT_PATH, NORM_STATS_PATH,
    TRAIN_WINDOW_STRIDE,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "cwt_lstm.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison_vs_baseline.md")
BASELINE_CSV = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                             "paper_implementation", "results", "convlstm.csv")

TEST_WINDOW_STRIDE = 5  # NOT stride=1: full test set at stride=1 (~427K windows) would need
                          # ~8.6GB in float16 -- right at data.py's default 8GB safety cap, and
                          # this experiment already OOM'd the machine once at the training stage
                          # (see train.py's comment). stride=5 (~86K windows, ~1.7GB) stays safe.
CONSEC_WINDOWS = 2       # same default as paper_implementation/svm_model.py & convlstm_model.py


def predict(model, X, device, batch_size=256):
    loader = DataLoader(ScalogramDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def aggregate_to_trials(meta, y_pred, consec=CONSEC_WINDOWS, fs=FS):
    """Identical convention to paper_implementation/svm_model.py & convlstm_model.py."""
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
    return {"label": label, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sensitivity, "specificity": specificity,
            "lead_mean": lead.mean(), "lead_std": lead.std()}


def baseline_summary():
    if not os.path.exists(BASELINE_CSV):
        return None
    df = pd.read_csv(BASELINE_CSV)
    return summarize(df, "Baseline raw-1D ConvLSTM")


if __name__ == "__main__":
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH} -- run train.py first.")
        sys.exit(1)

    split = make_subject_split()
    test_subjects = split["test_subjects"]

    model = CWT_ConvLSTM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    stats = np.load(NORM_STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    print(f"Building CWT scalogram test set (stride={TEST_WINDOW_STRIDE})...")
    X_test, y_test, meta_test = build_cwt_dataset(test_subjects, stride=TEST_WINDOW_STRIDE, cache_tag="test")

    # Chunked normalization back to float16 (same rationale as train.py: avoid
    # materializing a full float32/float64 upcast of the whole test array).
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

    cwt_summary = summarize(trial_results, "CWT + 2D-CNN + LSTM")
    base_summary = baseline_summary()

    lines = ["# CWT + 2D-CNN vs. Baseline Raw-1D ConvLSTM\n"]
    if base_summary:
        lines += [
            "| | Sensitivity | Specificity | Lead time (ms) |",
            "|---|---|---|---|",
            f"| CWT + 2D-CNN + LSTM | {cwt_summary['sensitivity']:.2%} | "
            f"{cwt_summary['specificity']:.2%} | {cwt_summary['lead_mean']:.0f}+/-{cwt_summary['lead_std']:.0f} |",
            f"| Baseline raw-1D ConvLSTM | {base_summary['sensitivity']:.2%} | "
            f"{base_summary['specificity']:.2%} | {base_summary['lead_mean']:.0f}+/-{base_summary['lead_std']:.0f} |",
            "",
            f"Baseline specificity: {base_summary['specificity']:.2%}. "
            f"CWT specificity: {cwt_summary['specificity']:.2%}. "
            f"Delta: {(cwt_summary['specificity']-base_summary['specificity'])*100:+.1f}pp.",
        ]
        delta = cwt_summary["specificity"] - base_summary["specificity"]
        sens_drop = base_summary["sensitivity"] - cwt_summary["sensitivity"]
        if delta > 0.02 and sens_drop < 0.05:
            lines.append("\n**Result: specificity improved meaningfully without a large sensitivity drop "
                          "-- supports the frequency-information hypothesis.**")
        else:
            lines.append("\n**Result: specificity did NOT improve meaningfully over the baseline -- this "
                          "rules out \"missing frequency information\" as the explanation for the plateau.**")
    else:
        lines.append(f"(No baseline results found at {BASELINE_CSV} to compare against.)")
        lines.append(f"\nCWT + 2D-CNN + LSTM: sensitivity {cwt_summary['sensitivity']:.2%}, "
                      f"specificity {cwt_summary['specificity']:.2%}, "
                      f"lead {cwt_summary['lead_mean']:.0f}+/-{cwt_summary['lead_std']:.0f} ms")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison to {COMPARISON_MD}")
