"""
Evaluation for the NO-POOLING CNN + Transformer ablation (see model.py's
docstring for the hypothesis under test: does pooling-induced information
loss, not just token count, explain the pooled conv_transformer's
specificity regression?).

Test set uses full resolution (stride=1) regardless of the reduced training
scale, per the task instructions -- CONSEC_WINDOWS=2 per-trial aggregation,
identical to every other experiment in this project.

Compares against:
  (a) The original pooled conv_transformer result (80.08% specificity) --
      read directly from experiments/conv_transformer/results/conv_transformer.csv
      if present, so the comparison number is always the actual last run,
      not a hardcoded copy that could drift out of date.
  (b) The ConvLSTM baseline at MATCHED 150K training scale (99.28%
      sensitivity / 89.85% specificity) -- this number is NOT re-derived
      here; it's documented in paper_implementation/README.md's "ConvLSTM,
      150K-window training" row (that intermediate checkpoint's per-trial
      CSV was not separately saved -- only the final full-scale run's CSV
      exists on disk). Hardcoded with a clear citation, not silently reused
      as if it were freshly computed.
  (c) The ConvLSTM baseline at full uncapped (1.41M-window) scale (99.28%/
      91.19%), read from paper_implementation/results/convlstm.csv, for
      reference (not the primary matched-scale comparison, since training
      scales differ).

Run (after train.py has produced a checkpoint):
  python3 experiments/conv_transformer_nopool/evaluate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import ConvTransformerNoPool  # noqa: E402
from train import (  # noqa: E402
    make_subject_split, build_raw_dataset, WindowDataset, CHECKPOINT_PATH,
    RESULTS_DIR, FS, TRAIN_WINDOW_STRIDE, MAX_ADL_TRAIN_WINDOWS,
)

RESULTS_CSV = os.path.join(RESULTS_DIR, "conv_transformer_nopool.csv")
COMPARISON_MD = os.path.join(RESULTS_DIR, "comparison_vs_baseline.md")

EXPERIMENTS_DIR = os.path.dirname(SCRIPT_DIR)
POOLED_CSV = os.path.join(EXPERIMENTS_DIR, "conv_transformer", "results", "conv_transformer.csv")
PAPER_IMPL_DIR = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "paper_implementation")
CONVLSTM_FULLSCALE_CSV = os.path.join(PAPER_IMPL_DIR, "results", "convlstm.csv")

# From paper_implementation/README.md's "ConvLSTM, 150K-window training" row
# -- documented reference, not re-derived (see module docstring).
CONVLSTM_150K_REFERENCE = {"sensitivity": 0.9928, "specificity": 0.8985, "lead_mean": 225, "lead_std": 136}

TEST_WINDOW_STRIDE = 1  # full resolution regardless of training scale
CONSEC_WINDOWS = 2


def predict(model, X, device, batch_size=512):
    loader = DataLoader(WindowDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_size)
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

    model = ConvTransformerNoPool().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    stats = np.load(os.path.join(RESULTS_DIR, "norm_stats.npz"))
    mean, std = stats["mean"], stats["std"]

    print(f"Building raw-window test set (stride={TEST_WINDOW_STRIDE})...")
    X_test, y_test, meta_test = build_raw_dataset(test_subjects, stride=TEST_WINDOW_STRIDE)
    X_test_n = (X_test - mean) / std

    y_pred = predict(model, X_test_n, device)
    trial_results = aggregate_to_trials(meta_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(trial_results)} per-trial predictions to {RESULTS_CSV}")

    nopool_summary = summarize(trial_results, "CNN + Transformer, NO POOLING (50-token)")
    pooled_summary = csv_summary(POOLED_CSV, "CNN + Transformer, POOLED (6-token, original)")
    fullscale_summary = csv_summary(CONVLSTM_FULLSCALE_CSV, "ConvLSTM baseline (full 1.41M-window scale)")

    lines = ["# CNN + Transformer, No Pooling vs. Pooled vs. ConvLSTM Baselines\n"]
    lines.append(f"Training scale for this run: stride={TRAIN_WINDOW_STRIDE}, "
                 f"ADL capped at {MAX_ADL_TRAIN_WINDOWS} (reduced scale -- see train.py docstring).\n")
    lines += ["| | Sensitivity | Specificity | Lead time (ms) |", "|---|---|---|---|"]
    lines.append(f"| **No-pooling transformer (this run, 150K scale)** | "
                 f"{nopool_summary['sensitivity']:.2%} | {nopool_summary['specificity']:.2%} | "
                 f"{nopool_summary['lead_mean']:.0f}+/-{nopool_summary['lead_std']:.0f} |")
    lines.append(f"| ConvLSTM baseline, MATCHED 150K scale (documented reference, not re-run) | "
                 f"{CONVLSTM_150K_REFERENCE['sensitivity']:.2%} | {CONVLSTM_150K_REFERENCE['specificity']:.2%} | "
                 f"{CONVLSTM_150K_REFERENCE['lead_mean']}+/-{CONVLSTM_150K_REFERENCE['lead_std']} |")
    if pooled_summary:
        lines.append(f"| Pooled transformer (original, 6-token, full 1.41M scale) | "
                     f"{pooled_summary['sensitivity']:.2%} | {pooled_summary['specificity']:.2%} | "
                     f"{pooled_summary['lead_mean']:.0f}+/-{pooled_summary['lead_std']:.0f} |")
    if fullscale_summary:
        lines.append(f"| ConvLSTM baseline, full 1.41M scale (reference only, scale mismatch) | "
                     f"{fullscale_summary['sensitivity']:.2%} | {fullscale_summary['specificity']:.2%} | "
                     f"{fullscale_summary['lead_mean']:.0f}+/-{fullscale_summary['lead_std']:.0f} |")

    lines.append("\n## Success criterion: did removing pooling move specificity toward/past the ConvLSTM baseline?\n")
    if pooled_summary:
        delta_vs_pooled = nopool_summary["specificity"] - pooled_summary["specificity"]
        delta_vs_convlstm_matched = nopool_summary["specificity"] - CONVLSTM_150K_REFERENCE["specificity"]
        lines.append(f"vs. original pooled transformer (80.08%): {delta_vs_pooled*100:+.1f}pp.")
        lines.append(f"vs. ConvLSTM at matched 150K scale (89.85%): {delta_vs_convlstm_matched*100:+.1f}pp.\n")
        if delta_vs_pooled > 0.03:
            lines.append("**Specificity moved meaningfully TOWARD/PAST the ConvLSTM baseline -- "
                         "supports \"pooling destroyed information attention needed.\"**")
        elif delta_vs_pooled < -0.03:
            lines.append("**Specificity got WORSE without pooling -- refutes the pooling-information-loss "
                         "hypothesis; the token-count change alone (or something else) is not the fix.**")
        else:
            lines.append("**No meaningful change from removing pooling -- pooling-induced information loss "
                         "is NOT the (or not the only) explanation for the original regression.**")
    else:
        lines.append(f"(No pooled-version results found at {POOLED_CSV} to compare against.)")

    with open(COMPARISON_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison to {COMPARISON_MD}")
