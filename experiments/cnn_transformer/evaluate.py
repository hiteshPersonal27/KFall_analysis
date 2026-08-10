"""
Evaluation for the 1D-CNN + Transformer experiment (see
../../docs/CNN_Transformer_Implementation_Plan.md).

Per-window prediction -> per-trial aggregation using the SAME consecutive-
window rule established throughout this project (CONSEC_WINDOWS=2, see
paper_implementation/svm_model.py's comment for the rationale). Produces a
three-row comparison: this reproduction, the existing ConvLSTM baseline, and
PreFallKD's own published ViT-tiny numbers (external sanity check on
whether this reproduction is in the right range, per the plan's success
criteria).

Run (after train.py has produced a checkpoint):
  python3 experiments/cnn_transformer/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import build_raw_dataset, make_subject_split, FS  # noqa: E402
from model import PreFallTransformer  # noqa: E402
from train import PatchDataset, CHECKPOINT_PATH, NORM_STATS_PATH  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "transformer.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison.md")
BASELINE_CSV = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                             "paper_implementation", "results", "convlstm.csv")

TEST_WINDOW_STRIDE = 1  # matches the baseline's test stride -- raw window storage is cheap
                          # (no CWT-style memory blowup), so full resolution is safe here.
CONSEC_WINDOWS = 2       # same default as paper_implementation/svm_model.py & convlstm_model.py

# PreFallKD's own published numbers (external correctness check -- Section 0 of the plan).
PREFALLKD_REFERENCE = {
    "CNNLSTM (PreFallKD's repro)": {"accuracy": 0.9767, "precision": 0.8835, "recall": 0.9458,
                                      "specificity": 0.9813, "f1": 0.9136, "lead_ms": 493.5},
    "ViT-tiny (PreFallKD's teacher)": {"accuracy": 0.9836, "precision": 0.9202, "recall": 0.9573,
                                         "specificity": 0.9936, "f1": 0.9384, "lead_ms": 235.4},
    "PreFallKD (KD-distilled student)": {"accuracy": 0.9805, "precision": 0.9062, "recall": 0.9479,
                                           "specificity": 0.9853, "f1": 0.9266, "lead_ms": 551.3},
}


def predict(model, X, device, batch_size=256):
    from data import patchify_batch
    loader = DataLoader(PatchDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_size)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH} -- run train.py first.")
        sys.exit(1)

    split = make_subject_split()
    test_subjects = split["test_subjects"]

    model = PreFallTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    stats = np.load(NORM_STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    print(f"Building raw-window test set (stride={TEST_WINDOW_STRIDE})...")
    X_test, y_test, meta_test = build_raw_dataset(test_subjects, stride=TEST_WINDOW_STRIDE, oversample_fall=1)
    X_test_n = (X_test - mean) / std

    y_pred = predict(model, X_test_n, device)
    trial_results = aggregate_to_trials(meta_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(trial_results)} per-trial predictions to {RESULTS_CSV}")

    xf_summary = summarize(trial_results, "1D-CNN + Transformer (this repro)")
    base_summary = baseline_summary()

    # Derive precision/accuracy/F1/specificity for the PreFallKD-style comparison row.
    tp, fn, tn, fp = xf_summary["tp"], xf_summary["fn"], xf_summary["tn"], xf_summary["fp"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = xf_summary["sensitivity"]
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / (tp + fn + tn + fp)

    lines = ["# 1D-CNN + Transformer vs. Baseline ConvLSTM vs. PreFallKD's published numbers\n"]
    lines += [
        "| Model | Accuracy | Precision | Recall (Sens.) | Specificity | F1 | Lead time (ms) |",
        "|---|---|---|---|---|---|---|",
        f"| **This repro (transformer)** | {accuracy:.2%} | {precision:.2%} | {recall:.2%} | "
        f"{xf_summary['specificity']:.2%} | {f1:.2%} | {xf_summary['lead_mean']:.0f}+/-{xf_summary['lead_std']:.0f} |",
    ]
    for name, ref in PREFALLKD_REFERENCE.items():
        lines.append(f"| {name} (paper) | {ref['accuracy']:.2%} | {ref['precision']:.2%} | "
                      f"{ref['recall']:.2%} | {ref['specificity']:.2%} | {ref['f1']:.2%} | {ref['lead_ms']:.1f} |")
    if base_summary:
        lines.append(f"| Our ConvLSTM baseline | -- | -- | {base_summary['sensitivity']:.2%} | "
                      f"{base_summary['specificity']:.2%} | -- | "
                      f"{base_summary['lead_mean']:.0f}+/-{base_summary['lead_std']:.0f} |")

    vit_ref = PREFALLKD_REFERENCE["ViT-tiny (PreFallKD's teacher)"]
    acc_dist = abs(accuracy - vit_ref["accuracy"])
    spec_dist = abs(xf_summary["specificity"] - vit_ref["specificity"])
    lines.append(f"\n## Correctness check (primary success criterion)\n")
    lines.append(f"Distance from PreFallKD's ViT-tiny: accuracy delta {acc_dist*100:+.1f}pp, "
                 f"specificity delta {(xf_summary['specificity']-vit_ref['specificity'])*100:+.1f}pp.")
    if acc_dist < 0.05 and spec_dist < 0.05:
        lines.append("**Reproduction lands close to PreFallKD's published range -- implementation "
                     "validated.** Comparison against our own ConvLSTM baseline below is trustworthy.")
    else:
        lines.append("**Reproduction is NOT close to PreFallKD's published range -- treat this as a "
                     "signal to debug the implementation before drawing conclusions about attention vs LSTM.**")

    if base_summary:
        spec_delta_baseline = xf_summary["specificity"] - base_summary["specificity"]
        lines.append(f"\n## Research question: does attention improve specificity over our ConvLSTM baseline?\n")
        lines.append(f"Baseline specificity: {base_summary['specificity']:.2%}. "
                     f"Transformer specificity: {xf_summary['specificity']:.2%}. "
                     f"Delta: {spec_delta_baseline*100:+.1f}pp.")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison to {COMPARISON_MD}")
    print(f"\nAccuracy: {accuracy:.2%}  Precision: {precision:.2%}  F1: {f1:.2%}")
