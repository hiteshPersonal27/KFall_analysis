"""
Training script for the GAF global-normalization fix (see data.py's
docstring for the full diagnosis and fix rationale).

Two deliberate changes vs. experiments/gaf_mtf/train.py:
  1. Global (not per-window) GASF normalization -- the actual fix under test.
  2. FULL training scale (stride=1, uncapped ADL windows), not the 150K
     matched scale gaf_mtf used -- per explicit user direction, prioritizing
     the best achievable result. Stored as float16 (~25GB for the full
     fit+val set) rather than float32 (~50GB, too close to available RAM)
     -- same memory-safety lesson learned from experiments/cwt_lstm/'s
     earlier OOM incident. Cast back to float32 per-sample during training.

Same conventions otherwise as every experiment in this project: fixed seed,
validation split from training subjects, best checkpoint by validation
balanced accuracy, class-weighted CrossEntropyLoss (matches gaf_mtf's
convention).

Run:  python3 experiments/gaf_global/train.py
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
from data import (  # noqa: E402
    build_gaf_dataset, make_subject_split, compute_global_stats, save_global_stats,
    N_CHANNELS, WINDOW_WIDTH, DEFAULT_MAX_MEMORY_GB,
)
from model import GAF_CNN  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "gaf_global_best.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats.npz")
GLOBAL_STATS_PATH = os.path.join(RESULTS_DIR, "global_stats.npz")

SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
VAL_SUBJECT_FRACTION = 0.2

# FULL SCALE (see module docstring) -- not gaf_mtf's 150K matched scale.
TRAIN_WINDOW_STRIDE = 1
MAX_ADL_TRAIN_WINDOWS = None
MAX_MEMORY_GB = 28.0  # raised from data.py's default per the plan -- see data.py docstring


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GAFDataset(Dataset):
    """X is stored as float16 (memory-safety note in data.py); cast to
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
    return model, total_train_time


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

    print("\nComputing global normalization stats (fit subjects only)...")
    global_stats = compute_global_stats(fit_subjects)
    print(f"  {global_stats}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_global_stats(global_stats, GLOBAL_STATS_PATH)

    t0 = time.time()
    print(f"\nBuilding GAF training set (stride={TRAIN_WINDOW_STRIDE}, "
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS}, max_memory_gb={MAX_MEMORY_GB})...")
    X_fit, y_fit, _ = build_gaf_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE, global_stats=global_stats,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS, max_memory_gb=MAX_MEMORY_GB)
    X_val, y_val, _ = build_gaf_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE, global_stats=global_stats,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS, max_memory_gb=MAX_MEMORY_GB)
    data_build_time = time.time() - t0
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({data_build_time:.0f}s)")
    print(f"  fit class balance: fall:adl = 1:{(y_fit==0).sum()/max((y_fit==1).sum(),1):.1f}")

    # Per-channel standardization of the GASF images themselves (unrelated to
    # the global min/max used INSIDE gasf_global -- this is a second,
    # separate normalization step on the resulting images, same as gaf_mtf).
    #
    # TWO memory bugs found and fixed here, in sequence:
    #  1. A naive `X_fit.astype(np.float32)` on the full ~20GB float16 array
    #     created a full ~40GB float32 COPY as a temporary (x2, for mean and
    #     std) -- this OOM'd and crashed the machine on the first attempt.
    #  2. Fixed #1 by using `.mean(..., dtype=np.float64)` (numpy computes
    #     this as a streaming sum, no full-array temporary) -- but
    #     `.std(..., dtype=np.float64)` is NOT streaming the same way:
    #     numpy's variance internally computes `array - mean` as a full
    #     temporary array before squaring, which tried to allocate ~79GB and
    #     failed (cleanly this time -- a Python MemoryError, not a system
    #     crash, but still needed fixing).
    # Fixed properly with an explicit two-pass CHUNKED std computation
    # (mean first via the streaming reduction, which IS safe; then sum of
    # squared deviations accumulated chunk-by-chunk for the variance) --
    # this never materializes more than one chunk's temporary at a time.
    X_fit_view = X_fit.reshape(X_fit.shape[0], N_CHANNELS, -1)
    mean = X_fit_view.mean(axis=(0, 2), dtype=np.float64).astype(np.float32).reshape(1, N_CHANNELS, 1, 1)

    def chunked_std(X_view, mean_flat, chunk=8192):
        sq_sum = np.zeros(N_CHANNELS, dtype=np.float64)
        n_total = 0
        for start in range(0, len(X_view), chunk):
            block = X_view[start:start + chunk].astype(np.float64)  # one chunk only, not the full array
            sq_sum += ((block - mean_flat.reshape(1, N_CHANNELS, 1)) ** 2).sum(axis=(0, 2))
            n_total += block.shape[0] * block.shape[2]
        return np.sqrt(sq_sum / n_total)

    std = (chunked_std(X_fit_view, mean.flatten()).astype(np.float32) + 1e-6).reshape(1, N_CHANNELS, 1, 1)

    def normalize_f16(X, chunk=8192):
        out = np.empty(X.shape, dtype=np.float16)
        for start in range(0, len(X), chunk):
            end = min(start + chunk, len(X))
            out[start:end] = ((X[start:end].astype(np.float32) - mean) / std).astype(np.float16)
        return out

    X_fit = normalize_f16(X_fit)
    X_val = normalize_f16(X_val)
    np.savez(NORM_STATS_PATH, mean=mean, std=std)

    print("\nTraining GAF 2D-CNN (global normalization, full scale)...")
    model, total_train_time = train_model(X_fit, y_fit, X_val, y_val, device)
    print(f"\nBest checkpoint saved to {CHECKPOINT_PATH}")
    print(f"\n=== WALL-CLOCK SUMMARY ===")
    print(f"Data build: {data_build_time:.0f}s | Training: {total_train_time:.0f}s | "
          f"Total: {data_build_time + total_train_time:.0f}s")
