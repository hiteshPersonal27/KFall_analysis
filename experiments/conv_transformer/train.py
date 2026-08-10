"""
Training script for the CNN + Transformer ablation (see model.py's
docstring). Deliberately mirrors paper_implementation/convlstm_model.py's
training recipe EXACTLY (same seed, batch size, epochs, LR, optimizer,
window stride, ADL cap, aggregation rule) -- the ONLY difference from the
baseline is the model architecture (model.py's ConvTransformer instead of
ConvLSTM). This isolates the LSTM-vs-attention question as cleanly as
possible; changing the optimizer/recipe too would confound the comparison.

Run:  python3 experiments/conv_transformer/train.py
"""

import os
import sys
import time
import random
import importlib.util

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import ConvTransformer  # noqa: E402

# Load paper_implementation/data.py under a distinct module name (both files
# are named "data.py" in other experiment folders; this one has no data.py
# of its own, so a plain sys.path insert + import works fine here, but kept
# consistent with the other experiments' explicit-load pattern for clarity).
PAPER_IMPL_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "paper_implementation")
_spec = importlib.util.spec_from_file_location("_paper_impl_data", os.path.join(PAPER_IMPL_DIR, "data.py"))
_paper_data = importlib.util.module_from_spec(_spec)
sys.path.insert(0, PAPER_IMPL_DIR)
_spec.loader.exec_module(_paper_data)

make_subject_split = _paper_data.make_subject_split
iter_subject_trials = _paper_data.iter_subject_trials
iter_windows = _paper_data.iter_windows
window_raw9 = _paper_data.window_raw9
FS = _paper_data.FS

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "conv_transformer_best.pt")

# Identical to paper_implementation/convlstm_model.py's constants.
SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
VAL_SUBJECT_FRACTION = 0.2
TRAIN_WINDOW_STRIDE = 1
MAX_ADL_TRAIN_WINDOWS = None
WINDOW_WIDTH = 50
RAW_CHANNELS = 9


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collect_windows(subjects, stride):
    fall_windows, adl_windows = [], []
    for subj in subjects:
        for trial in iter_subject_trials(subj):
            for w in iter_windows(trial, stride=stride):
                (fall_windows if w.label == "fall" else adl_windows).append(w)
    return fall_windows, adl_windows


def build_raw_dataset(subjects, stride, max_adl_windows=None, seed=SEED):
    """Identical to paper_implementation/convlstm_model.py's build_raw_dataset."""
    import pandas as pd
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


def train_model(X_train, y_train, X_val, y_val, device):
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ConvTransformer().to(device)
    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)),
                                  dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # same optimizer as the baseline (Adam, not AdamW)

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
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


if __name__ == "__main__":
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    split = make_subject_split()
    train_subjects = split["train_subjects"]

    rng = random.Random(SEED)
    shuffled = train_subjects[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VAL_SUBJECT_FRACTION))
    val_subjects = sorted(shuffled[:n_val])
    fit_subjects = sorted(shuffled[n_val:])
    print(f"Fit subjects ({len(fit_subjects)}): {fit_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")

    t0 = time.time()
    print(f"\nBuilding raw-window training set (stride={TRAIN_WINDOW_STRIDE}, "
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS})...")
    X_fit, y_fit, _ = build_raw_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS)
    X_val, y_val, _ = build_raw_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS)
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({time.time()-t0:.0f}s)")

    mean = X_fit.reshape(-1, RAW_CHANNELS).mean(axis=0)
    std = X_fit.reshape(-1, RAW_CHANNELS).std(axis=0) + 1e-6
    X_fit = (X_fit - mean) / std
    X_val = (X_val - mean) / std
    np.savez(os.path.join(RESULTS_DIR, "norm_stats.npz"), mean=mean, std=std)

    print("\nTraining CNN + Transformer (LSTM swapped for attention)...")
    model = train_model(X_fit, y_fit, X_val, y_val, device)
    print(f"\nBest checkpoint saved to {CHECKPOINT_PATH}")
