"""
Core BEAM components.

- MRI3DFeatureExtractor: 3D-CNN with 4 conv blocks (channels 32/64/128/256),
  MaxPool after the first three blocks only.
- CFDNAFeatureEncoder:   3-layer MLP, input -> 512 -> 512 -> d_model,
  each layer followed by BN + ReLU + Dropout(0.3).
- TransformerEncoderLayer: standard pre-norm Transformer encoder layer
  (h=8, FFN dim = 4 * d_model).
"""

import torch
import torch.nn as nn


class MRI3DFeatureExtractor(nn.Module):
    """
    3D-CNN MRI feature extractor.

    f_mri: R^{C x D x H x W} -> R^{d_model}

    H_l = Pool( ReLU( BN( Conv3D_l( H_{l-1} ) ) ) ),  l in {1, 2, 3}
    H_4 = ReLU( BN( Conv3D_4( H_3 ) ) )           # no pooling after block 4
    E_mri = FC( AdaptiveAvgPool3D(H_4) )
    """

    def __init__(self, input_channels: int = 3, d_model: int = 512):
        super().__init__()

        channels = [input_channels, 32, 64, 128, 256]

        # Four convolutional blocks
        self.blocks = nn.ModuleList()
        for i in range(4):
            block = nn.Sequential(
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(channels[i + 1]),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)

        # MaxPool only after the first three blocks
        self.pool = nn.MaxPool3d(kernel_size=2)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < 3:  # pooling after the first three blocks only
                x = self.pool(x)
        x = self.adaptive_pool(x)              # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)              # (B, 256)
        x = self.fc(x)                         # (B, d_model)
        return x


class CFDNAFeatureEncoder(nn.Module):
    """
    cfDNA MLP feature extractor.

    f_cfdna: R^{d_cfdna} -> R^{d_model}

    Three FC layers: d_cfdna -> 512 -> 512 -> d_model
    Each layer is followed by: BN -> ReLU -> Dropout(0.3)
    """

    def __init__(self, input_dim: int = 275, d_model: int = 512, dropout: float = 0.3):
        super().__init__()

        dims = [input_dim, 512, 512, d_model]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TransformerEncoderLayer(nn.Module):
    """
    Standard pre-norm Transformer encoder layer.

    A_l   = MHSA( LN(S_{l-1}) )
    S_l'  = S_{l-1} + Dropout(A_l)
    O_l   = FFN( LN(S_l') )
    S_l   = S_l' + Dropout(O_l)

    FFN = Linear(d, 4d) -> ReLU -> Dropout -> Linear(4d, d)
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x
