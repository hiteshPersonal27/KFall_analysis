"""
2D-CNN + LSTM model for the CWT scalogram experiment (see
../../docs/CWT_2DCNN_Implementation_Plan.md).

Mirrors the baseline raw-1D ConvLSTM's structure (paper_implementation/
convlstm_model.py) so the comparison is about the 1D-vs-2D/frequency change
specifically, not a different overall depth: 3x(Conv->BatchNorm->ReLU->
MaxPool) blocks, filters 32/64/128 (same progression as the original), then
an LSTM, then a dense+softmax head. Only the conv/pool dimensionality changes
(2D instead of 1D) to consume the (9, num_scales, 50) scalogram input.

Run standalone to print the shape progression through the conv stack (this
is the thing the plan asks to explicitly track, since the original 1D model
collapsed its sequence to length 1 before the LSTM -- worth checking this
2D version doesn't do the same):
  python3 experiments/cwt_lstm/model.py
"""

import torch
import torch.nn as nn

from data import NUM_SCALES, WINDOW_WIDTH, RAW_CHANNELS  # noqa: E402

CONV_FILTERS = (32, 64, 128)   # same progression as the baseline 1D ConvLSTM
KERNEL_SIZE = 3
POOL_SIZE = 2
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.5


class CWT_ConvLSTM(nn.Module):
    def __init__(self, in_channels=RAW_CHANNELS, freq_bins=NUM_SCALES, time_steps=WINDOW_WIDTH,
                 conv_filters=CONV_FILTERS, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT, n_classes=2,
                 verbose=False):
        super().__init__()
        c1, c2, c3 = conv_filters
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size, padding=pad), nn.BatchNorm2d(c1),
            nn.ReLU(), nn.MaxPool2d(pool_size),
            nn.Conv2d(c1, c2, kernel_size, padding=pad), nn.BatchNorm2d(c2),
            nn.ReLU(), nn.MaxPool2d(pool_size),
            nn.Conv2d(c2, c3, kernel_size, padding=pad), nn.BatchNorm2d(c3),
            nn.ReLU(), nn.MaxPool2d(pool_size),
        )

        # Determine the actual post-conv shape by a dry run, rather than
        # hand-computing floor-division pooling arithmetic (error-prone) --
        # this is the "track and record the output shape" step the plan asks
        # for. freq_out is folded into the LSTM's per-timestep feature dim
        # (channels x freq), time_out becomes the LSTM sequence length.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, freq_bins, time_steps)
            dummy_out = self.conv(dummy)
        _, c_out, freq_out, time_out = dummy_out.shape
        self.time_out = time_out
        self.lstm_input_size = c_out * freq_out
        if verbose:
            print(f"Conv stack output shape (per sample): channels={c_out}, "
                  f"freq_bins={freq_out}, time_steps={time_out}")
            print(f"-> LSTM sequence length={time_out}, per-step feature size={self.lstm_input_size}")

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden,
                             num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x: (B, 9, freq_bins, time_steps)
        x = self.conv(x)                                   # (B, c3, freq_out, time_out)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)      # (B, time_out, c3*freq_out) -- time is the sequence axis
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)                                # logits; softmax applied via CrossEntropyLoss


if __name__ == "__main__":
    print(f"Input shape per sample: ({RAW_CHANNELS}, {NUM_SCALES}, {WINDOW_WIDTH})  "
          f"(channels, freq_bins, time_steps)")
    model = CWT_ConvLSTM(verbose=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Sanity forward pass with a batch.
    x = torch.randn(4, RAW_CHANNELS, NUM_SCALES, WINDOW_WIDTH)
    logits = model(x)
    print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
