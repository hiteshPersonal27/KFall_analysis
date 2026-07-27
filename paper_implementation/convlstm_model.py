"""
Algorithm C -- ConvLSTM fall detection on raw 9-channel sensor windows (no
hand-crafted features): 3x(Conv1d -> BatchNorm -> ReLU -> MaxPool) -> 2-layer
LSTM w/ dropout -> FC + softmax.

Same windowing (width=50 frames) and subject-level split (split.json, shared
with threshold.py/svm_model.py) as the rest of the module. Like svm_model.py,
training uses a widened stride + capped ADL subsampling for tractability
(stride=1 over the full training set is ~1.7M windows; see svm_model.py's
comment for the same rationale -- this still applies on GPU since the window
*collection*/feature-extraction pass is CPU-bound Python, not the model fit).

Uses CUDA automatically if available (falls back to CPU otherwise) -- see
device selection in __main__.

Config constants are declared at the top of the file, per the plan.

Run:  python3 paper_implementation/convlstm_model.py
"""

import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import make_subject_split, iter_subject_trials, iter_windows, window_raw9, FS  # noqa: E402

RESULTS_CSV = os.path.join(SCRIPT_DIR, "results", "convlstm.csv")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "results", "convlstm_best.pt")

# ----------------------------------------------------------------------------- #
# Config (documented, not buried inline)
# ----------------------------------------------------------------------------- #
SEED = 42
CONV_FILTERS = (32, 64, 128)
KERNEL_SIZE = 3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
VAL_SUBJECT_FRACTION = 0.2  # carved from TRAINING subjects, not test subjects

# Unlike svm_model.py's SVC (whose training cost scales quadratically-to-
# cubically with sample count, forcing a hard cap), a GPU-trained ConvLSTM's
# fit cost scales ~linearly with data via mini-batch SGD -- so it can absorb
# far more training data. The window *collection* pass is still CPU-bound
# Python, so this is slower to run, but per-trial signal computation (not
# per-window) dominates that cost, so it scales far better than naive
# window-count math suggests. Set to stride=1 (finest possible) with NO cap
# (MAX_ADL_TRAIN_WINDOWS=None) to use every available window -- the paper
# doesn't document any training-set subsampling, so this is the closest
# match to "train on the full 26-subject training set" given "I don't care
# about time, I just want to match the paper's numbers."
TRAIN_WINDOW_STRIDE = 1
TEST_WINDOW_STRIDE = 1   # denser test windows give the persistence rule (below)
                         # finer resolution to tell a sustained fall apart from
                         # a brief ADL blip -- measured to help specificity.
MAX_ADL_TRAIN_WINDOWS = None  # None = uncapped, use every ADL window
RAW_CHANNELS = 9
WINDOW_WIDTH = 50


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- #
# Dataset assembly (raw 9-channel windows)
# ----------------------------------------------------------------------------- #
def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


def build_raw_dataset(subjects, stride, max_adl_windows=None, seed=SEED):
    """Returns (X: (N, width, 9) float32, y: (N,) int64, meta: DataFrame)."""
    fall_windows, adl_windows = collect_windows(subjects, stride)
    if max_adl_windows is not None and len(adl_windows) > max_adl_windows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adl_windows), size=max_adl_windows, replace=False)
        adl_windows = [adl_windows[i] for i in idx]

    windows = [(w, 1) for w in fall_windows] + [(w, 0) for w in adl_windows]
    X = np.zeros((len(windows), WINDOW_WIDTH, RAW_CHANNELS), dtype=np.float32)
    y = np.zeros(len(windows), dtype=np.int64)
    meta = []
    df_cache = {}
    for i, (w, label) in enumerate(windows):
        X[i] = window_raw9(w, df_cache=df_cache)
        y[i] = label
        t = w.trial
        meta.append({"subject": t.subject, "task": t.task, "trial": t.trial,
                     "is_fall": t.is_fall, "end_frame": w.end,
                     "impact_frame": t.impact if t.is_fall else None})
    return X, y, pd.DataFrame(meta)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
class ConvLSTM(nn.Module):
    def __init__(self, in_channels=RAW_CHANNELS, conv_filters=CONV_FILTERS,
                 kernel_size=KERNEL_SIZE, lstm_hidden=LSTM_HIDDEN,
                 lstm_layers=LSTM_LAYERS, dropout=DROPOUT, n_classes=2):
        super().__init__()
        c1, c2, c3 = conv_filters
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size, padding=pad), nn.BatchNorm1d(c1),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size, padding=pad), nn.BatchNorm1d(c2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size, padding=pad), nn.BatchNorm1d(c3),
            nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=c3, hidden_size=lstm_hidden, num_layers=lstm_layers,
                             dropout=dropout, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x: (B, width, 9) -> conv wants (B, 9, width)
        x = x.transpose(1, 2)
        x = self.conv(x)                 # (B, c3, width')
        x = x.transpose(1, 2)            # (B, width', c3)
        out, _ = self.lstm(x)
        last = out[:, -1, :]             # causal-style: last timestep summary
        return self.fc(last)             # logits; softmax applied via CrossEntropyLoss


# ----------------------------------------------------------------------------- #
# Train / evaluate
# ----------------------------------------------------------------------------- #
def train_model(X_train, y_train, X_val, y_val, device):
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ConvLSTM().to(device)
    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)),
                                  dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_bal_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        train_loss = total_loss / len(y_train)

        model.eval()
        correct_per_class = np.zeros(2)
        total_per_class = np.zeros(2)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1).cpu().numpy()
                yb_np = yb.cpu().numpy()
                for c in (0, 1):
                    mask = yb_np == c
                    total_per_class[c] += mask.sum()
                    correct_per_class[c] += (pred[mask] == c).sum()
        val_bal_acc = np.mean(correct_per_class / np.maximum(total_per_class, 1))
        print(f"  epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  val_balanced_acc={val_bal_acc:.3f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


def predict(model, X, device, batch_size=512):
    loader = DataLoader(WindowDataset(X, np.zeros(len(X), dtype=np.int64)), batch_size=batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ----------------------------------------------------------------------------- #
# Per-trial aggregation: requires CONSEC_WINDOWS consecutive positive windows,
# not just any single one -- same rationale/fix as svm_model.py's
# aggregate_to_trials (a plain "ANY window fires" rule let isolated
# false-positive windows during fall-like ADL sub-motions condemn the whole,
# much-longer ADL trial; see svm_model.py's comment for the diagnosis and the
# measured sensitivity/specificity trade-off across consec values).
# ----------------------------------------------------------------------------- #
CONSEC_WINDOWS = 2


def aggregate_to_trials(meta, y_pred, consec=CONSEC_WINDOWS):
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
            first_end_frame = end_frames[first_idx]
            impact = g["impact_frame"].iloc[0]
            lead_ms = (impact - first_end_frame) * (1000.0 / FS)
        rows.append({"subject": subj, "task": task, "trial": trial_id, "is_fall": is_fall,
                     "detected": detected, "lead_time_ms": lead_ms if lead_ms is not None else np.nan})
    return pd.DataFrame(rows)


def summarize(df):
    fall = df[df["is_fall"]]
    adl = df[~df["is_fall"]]
    tp = int(fall["detected"].sum())
    fn = len(fall) - tp
    tn = int((~adl["detected"]).sum())
    fp = len(adl) - tn
    sensitivity = tp / len(fall) if len(fall) else float("nan")
    specificity = tn / len(adl) if len(adl) else float("nan")
    lead = fall.loc[fall["detected"], "lead_time_ms"]

    print(f"\nTest set: {len(fall)} fall trials, {len(adl)} ADL trials")
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"Sensitivity: {sensitivity:.2%}  (paper: 99.32%)")
    print(f"Specificity: {specificity:.2%}  (paper: 99.01%)")
    print(f"Lead time: {lead.mean():.0f} +/- {lead.std():.0f} ms  (paper: 403 +/- 163 ms)")

    return {"algorithm": "convlstm", "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": sensitivity, "specificity": specificity,
            "lead_time_mean_ms": lead.mean(), "lead_time_std_ms": lead.std()}


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    split = make_subject_split()
    train_subjects, test_subjects = split["train_subjects"], split["test_subjects"]

    # Validation split carved out of TRAINING subjects (not test subjects).
    rng = random.Random(SEED)
    shuffled_train = train_subjects[:]
    rng.shuffle(shuffled_train)
    n_val = max(1, round(len(shuffled_train) * VAL_SUBJECT_FRACTION))
    val_subjects = sorted(shuffled_train[:n_val])
    fit_subjects = sorted(shuffled_train[n_val:])
    print(f"Fit subjects ({len(fit_subjects)}): {fit_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

    t0 = time.time()
    print(f"\nBuilding raw-window training set (stride={TRAIN_WINDOW_STRIDE}, "
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS})...")
    X_fit, y_fit, _ = build_raw_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE,
                                         max_adl_windows=MAX_ADL_TRAIN_WINDOWS)
    val_adl_cap = None if MAX_ADL_TRAIN_WINDOWS is None else MAX_ADL_TRAIN_WINDOWS // 2
    X_val, y_val, _ = build_raw_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE,
                                         max_adl_windows=val_adl_cap)
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({time.time()-t0:.0f}s)")

    # Per-channel standardization fit on training windows only.
    mean = X_fit.reshape(-1, RAW_CHANNELS).mean(axis=0)
    std = X_fit.reshape(-1, RAW_CHANNELS).std(axis=0) + 1e-6
    X_fit = (X_fit - mean) / std
    X_val = (X_val - mean) / std

    print("\nTraining ConvLSTM...")
    model = train_model(X_fit, y_fit, X_val, y_val, device)

    print(f"\nBuilding raw-window test set (stride={TEST_WINDOW_STRIDE})...")
    X_test, y_test, meta_test = build_raw_dataset(test_subjects, stride=TEST_WINDOW_STRIDE)
    X_test = (X_test - mean) / std

    y_pred = predict(model, X_test, device)
    trial_results = aggregate_to_trials(meta_test, y_pred)

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    trial_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(trial_results)} per-trial predictions to {RESULTS_CSV}")
    print(f"Best checkpoint saved to {CHECKPOINT_PATH}")

    summarize(trial_results)
