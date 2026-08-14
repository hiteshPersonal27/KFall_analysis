"""
ViT-style transformer model (see
../../docs/CNN_Transformer_Implementation_Plan.md). Follows PreFallKD's
"tiny" transformer sizing exactly, as a faithful starting point: 3 layers,
3 attention heads, hidden size 64, MLP size 256, dropout 0.2.

Structure (Section 2a of the plan -- transformer only, no distillation yet):
  patch embedding (linear projection) -> prepend CLS token -> add position
  embeddings -> transformer encoder (multi-head self-attention + LayerNorm +
  MLP, stacked) -> classify from the CLS token's final representation.

Run standalone to print the shape progression and parameter count:
  python3 experiments/vit_prefallkd/model.py
"""

import torch
import torch.nn as nn

from data import N_PATCHES, PATCH_DIM  # noqa: E402

# PreFallKD "tiny" sizing per the plan: 3 layers, 3 heads, hidden=64, mlp=256,
# dropout=0.2. NOTE: 64 is not divisible by 3, but PyTorch's multi-head
# attention (like virtually all standard implementations) requires
# hidden_size % n_heads == 0 -- this is an inconsistency in the plan's stated
# numbers (likely a rounding/summarization artifact upstream, since we don't
# have the original PreFallKD source to check). Adjusted hidden_size to the
# nearest multiple of 3 (63, a 1.6% deviation) rather than changing the head
# count, since 3 heads is explicitly stated and 64 might just be an
# approximation of 63 or 66.
HIDDEN_SIZE = 63
N_LAYERS = 3
N_HEADS = 3
MLP_SIZE = 256
DROPOUT = 0.2


class PreFallTransformer(nn.Module):
    def __init__(self, n_patches=N_PATCHES, patch_dim=PATCH_DIM, hidden_size=HIDDEN_SIZE,
                 n_layers=N_LAYERS, n_heads=N_HEADS, mlp_size=MLP_SIZE, dropout=DROPOUT,
                 n_classes=2):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=mlp_size,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches):
        # patches: (B, n_patches, patch_dim)
        B = patches.shape[0]
        x = self.patch_embed(patches)                       # (B, n_patches, hidden)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, hidden)
        x = torch.cat([cls, x], dim=1)                        # (B, n_patches+1, hidden)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)                                    # (B, n_patches+1, hidden)
        cls_out = self.norm(x[:, 0])                            # (B, hidden) -- CLS token's final rep
        return self.head(cls_out)                                # logits; softmax via CrossEntropyLoss


if __name__ == "__main__":
    print(f"Input: (batch, {N_PATCHES} patches, {PATCH_DIM} patch_dim)")
    model = PreFallTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: hidden={HIDDEN_SIZE}, layers={N_LAYERS}, heads={N_HEADS}, "
          f"mlp={MLP_SIZE}, dropout={DROPOUT}")
    print(f"Total parameters: {n_params:,}")

    x = torch.randn(4, N_PATCHES, PATCH_DIM)
    logits = model(x)
    print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
