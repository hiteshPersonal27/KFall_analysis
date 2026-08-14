"""
Training script for the GAF 2D-encoding experiment (see
../../docs/GAF_MTF_Implementation_Plan.md).

Imbalance handling: class-weighted CrossEntropyLoss (inverse frequency) --
matches experiments/cwt_lstm/'s convention (not vit_prefallkd's 6x fall
oversampling), chosen for consistency with the other "richer 2D
representation" experiment this one is directly compared against.

Same conventions as every experiment in this project: fixed seed,
validation split from training subjects, best checkpoint by validation
balanced accuracy. Trained at REDUCED scale (stride=3, ADL capped at
150,000 -- same scale cwt_lstm used for its own first pass), not full
~1.4M-window scale, per the plan's "find a possible negative result quickly
and cheaply before committing to a longer run" instruction.

Run:  python3 experiments/gaf_mtf/train.py
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
from data import build_gaf_dataset, make_subject_split, N_CHANNELS, WINDOW_WIDTH  # noqa: E402
from model import GAF_CNN  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "gaf_mtf_best.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats.npz")

SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
VAL_SUBJECT_FRACTION = 0.2

# Reduced scale (see module docstring) -- same as experiments/cwt_lstm/'s first pass.
TRAIN_WINDOW_STRIDE = 3
MAX_ADL_TRAIN_WINDOWS = 150000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GAFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(X_fit, y_fit, X_val, y_val, device):
    train_loader = DataLoader(GAFDataset(X_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GAFDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GAF_CNN().to(device)
    class_counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)),
                                  dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_bal_acc = -1.0
    train_t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
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
        print(f"  epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_balanced_acc={val_bal_acc:.3f}  ({time.time()-epoch_t0:.1f}s/epoch)")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    total_train_time = time.time() - train_t0
    print(f"\nTotal training wall-clock time: {total_train_time:.0f}s ({total_train_time/EPOCHS:.1f}s/epoch average)")
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
    print(f"\nBuilding GAF training set (stride={TRAIN_WINDOW_STRIDE}, "
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS})...")
    X_fit, y_fit, _ = build_gaf_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS)
    val_adl_cap = MAX_ADL_TRAIN_WINDOWS // 2 if MAX_ADL_TRAIN_WINDOWS else None
    X_val, y_val, _ = build_gaf_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=val_adl_cap)
    data_build_time = time.time() - t0
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({data_build_time:.0f}s)")
    print(f"  fit class balance: fall:adl = 1:{(y_fit==0).sum()/max((y_fit==1).sum(),1):.1f}")

    # Per-channel standardization (fit on training set only).
    mean = X_fit.reshape(X_fit.shape[0], N_CHANNELS, -1).mean(axis=(0, 2)).reshape(1, N_CHANNELS, 1, 1)
    std = X_fit.reshape(X_fit.shape[0], N_CHANNELS, -1).std(axis=(0, 2)).reshape(1, N_CHANNELS, 1, 1) + 1e-6
    X_fit = (X_fit - mean) / std
    X_val = (X_val - mean) / std
    np.savez(NORM_STATS_PATH, mean=mean, std=std)

    print("\nTraining GAF 2D-CNN...")
    model = train_model(X_fit, y_fit, X_val, y_val, device)
    print(f"\nBest checkpoint saved to {CHECKPOINT_PATH}")
