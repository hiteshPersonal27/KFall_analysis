"""
2D CNN for the GAF global-normalization fix (see data.py's docstring).
Identical architecture to experiments/gaf_mtf/model.py -- the model was
never the hypothesized problem, only the input normalization was.

Input: (3, 50, 50) GASF image stack (channels = ACC_M/GYR_M/VV, not RGB).

Architecture: plain CNN -> global average pool -> dense -> softmax. NOT
CNN+LSTM (unlike experiments/cwt_lstm/'s model) -- deliberate choice: a GASF
image has no time-frequency axis the way a scalogram does, so there's no
principled reason to treat its output as a sequence for a recurrent stage.

Filter progression (32/64/128) mirrors the rest of this project for
comparability; kernel size 3 and 2x2 max pooling are standard defaults for
small (50x50) image classification.

Run standalone to print the shape progression and parameter count:
  python3 experiments/gaf_global/model.py
"""

import torch
import torch.nn as nn

from data import N_CHANNELS, WINDOW_WIDTH  # noqa: E402

CONV_FILTERS = (32, 64, 128)
KERNEL_SIZE = 3
DROPOUT = 0.5


class GAF_CNN(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, conv_filters=CONV_FILTERS,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT, n_classes=2):
        super().__init__()
        c1, c2, c3 = conv_filters
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size, padding=pad), nn.BatchNorm2d(c1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size, padding=pad), nn.BatchNorm2d(c2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size, padding=pad), nn.BatchNorm2d(c3),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, n_classes)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, WINDOW_WIDTH, WINDOW_WIDTH)
            conv_out_shape = self.conv(dummy).shape
        self.conv_out_shape = conv_out_shape

    def forward(self, x):
        # x: (B, 3, 50, 50)
        x = self.conv(x)                  # (B, c3, h', w')
        x = self.global_pool(x).flatten(1)  # (B, c3)
        x = self.dropout(x)
        return self.fc(x)                   # logits; softmax applied via CrossEntropyLoss


if __name__ == "__main__":
    print(f"Input: (batch, {N_CHANNELS}, {WINDOW_WIDTH}, {WINDOW_WIDTH})")
    model = GAF_CNN()
    print(f"Conv stack output shape (per sample): {tuple(model.conv_out_shape[1:])}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: conv_filters={CONV_FILTERS}, kernel_size={KERNEL_SIZE}, dropout={DROPOUT}")
    print(f"Total parameters: {n_params:,}")

    x = torch.randn(4, N_CHANNELS, WINDOW_WIDTH, WINDOW_WIDTH)
    logits = model(x)
    print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
