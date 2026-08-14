"""
Training script for the ViT-style Transformer experiment (see
../../docs/CNN_Transformer_Implementation_Plan.md).

Follows PreFallKD's own setup: AdamW optimizer (not plain Adam), cross-
entropy loss as the "acceptable simplification" the plan allows for an
initial pass (vs. PreFallKD's own focal loss -- noted as a documented open
item, not implemented here), fall-window oversampling to handle class
imbalance (PreFallKD's own 6x factor -- our training set has a similarly
skewed fall:adl window ratio, ~1:37-42, so this was judged applicable, not
skipped).

Uses the SAME fixed subject split (paper_implementation/split.json) as the
rest of the project. Patchifying (data.py's patchify_batch) happens on the
fly in the Dataset -- it's a cheap reshape, unlike CWT, so no disk caching
or memory-safety concern the way experiments/cwt_lstm/ needed (raw (50, 9)
window storage is identical in cost to the baseline ConvLSTM's, already
proven safe at full uncapped scale).

Run:  python3 experiments/vit_prefallkd/train.py
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
    build_raw_dataset, make_subject_split, patchify_batch,
    WINDOW_WIDTH, RAW_CHANNELS,
)
from model import PreFallTransformer  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "transformer_best.pt")
NORM_STATS_PATH = os.path.join(RESULTS_DIR, "norm_stats.npz")

SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4      # AdamW, per the plan (PreFallKD's own optimizer choice)
VAL_SUBJECT_FRACTION = 0.2

# Raw (50, 9) window storage costs the same as the baseline ConvLSTM's (no
# CWT-style memory blowup -- see experiments/cwt_lstm/README.md's OOM
# incident for why that mattered there), so this can safely match the
# baseline's final full-scale settings directly.
TRAIN_WINDOW_STRIDE = 1
MAX_ADL_TRAIN_WINDOWS = None   # uncapped
OVERSAMPLE_FALL = 6             # PreFallKD's own factor for their class imbalance


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PatchDataset(Dataset):
    """Wraps raw (N, 50, 9) windows; patchifies + returns float32 tensors on the fly."""

    def __init__(self, X_raw, y):
        self.patches = torch.from_numpy(patchify_batch(X_raw))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.patches[idx], self.y[idx]


def train_model(X_fit, y_fit, X_val, y_val, device):
    train_loader = DataLoader(PatchDataset(X_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PatchDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = PreFallTransformer().to(device)
    class_counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (2 * np.maximum(class_counts, 1)),
                                  dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
          f"ADL capped at {MAX_ADL_TRAIN_WINDOWS}, fall oversample={OVERSAMPLE_FALL}x)...")
    X_fit, y_fit, _ = build_raw_dataset(fit_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS,
                                          oversample_fall=OVERSAMPLE_FALL)
    X_val, y_val, _ = build_raw_dataset(val_subjects, stride=TRAIN_WINDOW_STRIDE,
                                          max_adl_windows=MAX_ADL_TRAIN_WINDOWS,
                                          oversample_fall=1)  # no oversampling on validation
    print(f"  fit: {X_fit.shape}  val: {X_val.shape}  ({time.time()-t0:.0f}s)")
    print(f"  fit class balance: fall:adl = 1:{(y_fit==0).sum()/max((y_fit==1).sum(),1):.1f}")

    # Per-channel standardization (fit on training set only).
    mean = X_fit.reshape(-1, RAW_CHANNELS).mean(axis=0)
    std = X_fit.reshape(-1, RAW_CHANNELS).std(axis=0) + 1e-6
    X_fit = (X_fit - mean) / std
    X_val = (X_val - mean) / std
    np.savez(NORM_STATS_PATH, mean=mean, std=std)

    print("\nTraining PreFallKD-style transformer...")
    model = train_model(X_fit, y_fit, X_val, y_val, device)
    print(f"\nBest checkpoint saved to {CHECKPOINT_PATH}")
    print(f"Normalization stats saved to {NORM_STATS_PATH}")
