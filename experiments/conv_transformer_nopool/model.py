"""
CNN + Transformer, NO POOLING -- direct follow-up to experiments/conv_transformer/'s
result (attention performed worse than LSTM, 80.08% vs 91.19% specificity).

That experiment's model compressed the 50-frame window down to just 6
time-steps via 3x MaxPool1d before the sequence stage, and README.md's
"why attention underperformed" analysis flagged that as reason #1: attention
was operating on a token count too short for its long-range advantage to
matter, while still paying the cost of learning order from scratch. This
model removes ALL pooling to test that specific hypothesis directly:
same (Conv1d -> BatchNorm1d -> ReLU) x3 structure, same filters (32/64/128),
same kernel size (3), but padding=1 (kernel_size//2) with NO MaxPool1d
between blocks, so the sequence length is preserved through every conv
layer -- the transformer sees close to the full 50-step sequence instead of
a compressed 6-step summary.

Everything else is unchanged from experiments/conv_transformer/model.py:
same position-embedding approach, same "nearest divisor of 128" head-count
logic, same global-average-pooling classification head (see that
experiment's model.py for why this replaced an earlier "last timestep"
convention copied from the LSTM baseline -- it doesn't carry the same
meaning for a non-causal-masked transformer). This isolates TOKEN
COUNT/RESOLUTION as the only variable versus that experiment.

Run standalone to print the shape progression and parameter count:
  python3 experiments/conv_transformer_nopool/model.py
"""

import torch
import torch.nn as nn

RAW_CHANNELS = 9
WINDOW_WIDTH = 50

# Identical conv config to experiments/conv_transformer/model.py -- only the
# pooling is removed.
CONV_FILTERS = (32, 64, 128)
KERNEL_SIZE = 3
DROPOUT = 0.5

TRANSFORMER_LAYERS = 2
N_HEADS = 4
MLP_SIZE = 256


class ConvTransformerNoPool(nn.Module):
    def __init__(self, in_channels=RAW_CHANNELS, conv_filters=CONV_FILTERS,
                 kernel_size=KERNEL_SIZE, transformer_layers=TRANSFORMER_LAYERS,
                 n_heads=N_HEADS, mlp_size=MLP_SIZE, dropout=DROPOUT, n_classes=2):
        super().__init__()
        c1, c2, c3 = conv_filters
        pad = kernel_size // 2  # =1 for kernel_size=3 -> preserves sequence length ("same" padding)
        # Same conv stack as experiments/conv_transformer/model.py, MaxPool1d removed.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size, padding=pad), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size, padding=pad), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size, padding=pad), nn.BatchNorm1d(c3), nn.ReLU(),
        )

        # Determine the actual post-conv sequence length by a dry run (should
        # be WINDOW_WIDTH=50, since padding=1/kernel=3/stride=1 convs preserve
        # length and there's no pooling -- computed rather than assumed).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, WINDOW_WIDTH)
            seq_len = self.conv(dummy).shape[-1]
        self.seq_len = seq_len

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, c3))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c3, nhead=n_heads, dim_feedforward=mlp_size,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.fc = nn.Linear(c3, n_classes)

    def forward(self, x):
        # x: (B, width, 9) -> conv wants (B, 9, width) -- identical convention to conv_transformer.
        x = x.transpose(1, 2)
        x = self.conv(x)                 # (B, c3, width) -- width UNCHANGED (no pooling)
        x = x.transpose(1, 2)            # (B, width, c3)
        x = x + self.pos_embed
        out = self.encoder(x)             # (B, width, c3)
        pooled = out.mean(dim=1)           # global average pool -- see conv_transformer/model.py's note
        return self.fc(pooled)             # logits; softmax applied via CrossEntropyLoss


if __name__ == "__main__":
    model = ConvTransformerNoPool()
    print(f"Conv stack output / transformer sequence length: {model.seq_len} "
          f"(vs. 6 for the pooled conv_transformer -- this is the token-count/resolution change under test)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: conv_filters={CONV_FILTERS}, transformer_layers={TRANSFORMER_LAYERS}, "
          f"n_heads={N_HEADS}, mlp_size={MLP_SIZE}, dropout={DROPOUT}")
    print(f"Total parameters: {n_params:,}")

    x = torch.randn(4, WINDOW_WIDTH, RAW_CHANNELS)
    logits = model(x)
    print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
