"""
Training script for the CWT + 2D-CNN experiment (see
../../docs/CWT_2DCNN_Implementation_Plan.md).

Same conventions as the baseline ConvLSTM (paper_implementation/
convlstm_model.py): CrossEntropyLoss with inverse-frequency class weights,
Adam, fixed seed, validation split carved from the training subjects, best
checkpoint saved by validation balanced accuracy. Uses the SAME fixed
subject split (paper_implementation/split.json) as the baseline for a fair
comparison.

Staged training per the plan: start small (MAX_ADL_TRAIN_WINDOWS below) to
get a fast first read on whether the approach works at all -- CWT
preprocessing + 2D convs are more expensive than the baseline's raw
passthrough -- then scale up in later runs by raising MAX_ADL_TRAIN_WINDOWS.

Run:  python3 experiments/cwt_lstm/train.py
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data import build_cwt_dataset, make_subject_split, NUM_SCALES, WINDOW_WIDTH, RAW_CHANNELS  # noqa: E402
from model import CWT_ConvLSTM  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "cwt_lstm_best.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats.npz")

SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
VAL_SUBJECT_FRACTION = 0.2

# CWT scalograms are NUM_SCALES=24x bigger per window than the baseline's
# raw (9, 50) window -- matching the baseline's uncapped stride=1 setting
# here needs ~74GB RAM and previously OOM'd the machine (killed VS Code and
# this training process). GPU CWT (gpu_cwt.py, ~380x faster than pywt) made
# COMPUTE cheap, but MEMORY is the real constraint now, not compute. These
# values keep fit+val well under data.py's 8GB default safety cap (see
# build_cwt_dataset's max_memory_gb) -- roughly matches the scale of the
# baseline ConvLSTM's intermediate (pre-uncapped) run for a fair reference
# point. Raise cautiously and check `free -h` first if you want more data.
TRAIN_WINDOW_STRIDE = 3
MAX_ADL_TRAIN_WINDOWS = 150000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ScalogramDataset(Dataset):
    """X is stored as float16 (see data.py's memory-safety note); cast to
    float32 per-sample here rather than up front, so the dataset itself
    still only holds the compact float16 copy in memory."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx]


def train_model(X_fit, y_fit, X_val, y_val, device):
    train_loader = DataLoader(ScalogramDataset(X_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ScalogramDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = CWT_ConvLSTM().to(device)
    class_counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)),
                                  dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_bal_acc = -1.0
    history = []
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
        train_loss = total_loss / len(y_fit)

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
        history.append({"epoch": epoch, "train_loss": train_loss, "val_balanced_acc": val_bal_acc})

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model, history


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
    print(f"\nBuilding CWT scalogram training set (stride={TRAIN_WINDOW_STRIDE}, "
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS})...")
    X_fit, y_fit, _ = build_cwt_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS, cache_tag="fit")
    val_adl_cap = None if MAX_ADL_TRAIN_WINDOWS is None else MAX_ADL_TRAIN_WINDOWS // 2
    X_val, y_val, _ = build_cwt_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=val_adl_cap, cache_tag="val")
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({time.time()-t0:.0f}s)")

    # Per-channel standardization (fit on training set only) -- CWT magnitude
    # scales vary a lot across scales/channels, per the plan's requirement.
    # X is stored as float16 (memory-safety note in data.py); mean/std use a
    # float64 accumulator (dtype=... below) for numerical precision without
    # materializing a full float64 copy of the array, and normalize_f16()
    # applies the normalization in chunks so at most one small chunk's
    # float32 temporary exists at once, not a full-dataset upcast.
    mean = X_fit.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = (X_fit.std(axis=(0, 2, 3), keepdims=True, dtype=np.float64) + 1e-6).astype(np.float32)

    def normalize_f16(X, chunk=8192):
        out = np.empty(X.shape, dtype=np.float16)
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            out[start:end] = ((X[start:end].astype(np.float32) - mean) / std).astype(np.float16)
        return out

    X_fit_n = normalize_f16(X_fit)
    X_val_n = normalize_f16(X_val)
    np.savez(NORM_STATS_PATH, mean=mean, std=std)

    print("\nTraining CWT + 2D-CNN + LSTM model...")
    model, history = train_model(X_fit_n, y_fit, X_val_n, y_val, device)
    print(f"\nBest checkpoint saved to {CHECKPOINT_PATH}")
    print(f"Normalization stats saved to {NORM_STATS_PATH}")
