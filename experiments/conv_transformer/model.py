"""
CNN + Transformer -- an ablation of the baseline ConvLSTM
(paper_implementation/convlstm_model.py): the exact same 1D conv feature
extractor, with the LSTM stage swapped for a Transformer encoder.
Conv filters/kernel size, dropout, and training recipe are kept identical
on purpose, so any sensitivity/specificity difference from the baseline
isolates the LSTM-vs-attention question as cleanly as possible -- unlike
experiments/vit_prefallkd/, which reproduces PreFallKD's own (very
different) ViT-style patch tokenization architecture with no conv layers
at all.

Classification head: global average pooling over the sequence axis (NOT
the LSTM's "last timestep" convention -- see the fix note below).

Run standalone to print the shape progression and parameter count:
  python3 experiments/conv_transformer/model.py
"""

import torch
import torch.nn as nn

RAW_CHANNELS = 9
WINDOW_WIDTH = 50

# Identical to paper_implementation/convlstm_model.py's CONV_FILTERS/KERNEL_SIZE/DROPOUT.
CONV_FILTERS = (32, 64, 128)
KERNEL_SIZE = 3
DROPOUT = 0.5

# Transformer stage sizing, chosen to mirror the LSTM stage's role as
# closely as possible: TRANSFORMER_LAYERS=2 matches the original's
# LSTM_LAYERS=2. d_model is set to the conv stack's own output channel count
# (128) rather than down-projecting to LSTM_HIDDEN=64, since a transformer
# (unlike an LSTM) has no separate "hidden size" concept -- it operates
# directly on its input dimension throughout. N_HEADS=4 is the natural
# divisor of 128 closest to a "small" head count.
TRANSFORMER_LAYERS = 2
N_HEADS = 4
MLP_SIZE = 256


class ConvTransformer(nn.Module):
    def __init__(self, in_channels=RAW_CHANNELS, conv_filters=CONV_FILTERS,
                 kernel_size=KERNEL_SIZE, transformer_layers=TRANSFORMER_LAYERS,
                 n_heads=N_HEADS, mlp_size=MLP_SIZE, dropout=DROPOUT, n_classes=2):
        super().__init__()
        c1, c2, c3 = conv_filters
        pad = kernel_size // 2
        # Identical conv stack to the baseline ConvLSTM.
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size, padding=pad), nn.BatchNorm1d(c1),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size, padding=pad), nn.BatchNorm1d(c2),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size, padding=pad), nn.BatchNorm1d(c3),
            nn.ReLU(), nn.MaxPool1d(2),
        )

        # Determine the actual post-conv sequence length by a dry run
        # (known to be 6 for a 50-frame window with this conv stack, per
        # paper_implementation/convlstm_model.py's measured behavior, but
        # computed here rather than hardcoded).
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
        # No causal mask is applied to self.encoder -- every position attends
        # to every other position equally (bidirectional). An earlier version
        # of this model copied the LSTM baseline's out[:, -1, :] ("last
        # timestep") convention, but that convention only means "cumulative
        # summary" for an LSTM's inherently causal, sequential processing --
        # for a non-causal-masked transformer, the last position carries no
        # such special meaning (attention gives every position equal access
        # to the whole sequence). Global average pooling (see forward()) is
        # the principled choice instead -- doesn't add new learnable
        # parameters/tokens (a CLS token would), keeping this a minimal
        # ablation of the recurrent-vs-attention question specifically.

    def forward(self, x):
        # x: (B, width, 9) -> conv wants (B, 9, width) -- identical to the baseline.
        x = x.transpose(1, 2)
        x = self.conv(x)                 # (B, c3, width')
        x = x.transpose(1, 2)            # (B, width', c3)
        x = x + self.pos_embed
        out = self.encoder(x)             # (B, width', c3)
        pooled = out.mean(dim=1)           # global average pool over the sequence axis
        return self.fc(pooled)             # logits; softmax applied via CrossEntropyLoss


if __name__ == "__main__":
    model = ConvTransformer()
    print(f"Conv stack output / transformer sequence length: {model.seq_len} "
          f"(baseline ConvLSTM's LSTM operates on this same sequence length)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: conv_filters={CONV_FILTERS}, transformer_layers={TRANSFORMER_LAYERS}, "
          f"n_heads={N_HEADS}, mlp_size={MLP_SIZE}, dropout={DROPOUT}")
    print(f"Total parameters: {n_params:,}")

    x = torch.randn(4, WINDOW_WIDTH, RAW_CHANNELS)
    logits = model(x)
    print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
