"""
BEAM main model.

p = F(X_mri, X_cfdna)
  = sigmoid( FC_final( mean( TransformerStack( S_0 ) ) ) )

where
  S_0 = [ LN( E_mri  + P_mri  ),
          LN( E_cfdna + P_cfdna ) ]
  E_mri   = f_mri  (X_mri)        # 3D-CNN
  E_cfdna = f_cfdna(X_cfdna)      # MLP
"""

import torch
import torch.nn as nn

from .components import MRI3DFeatureExtractor, CFDNAFeatureEncoder, TransformerEncoderLayer
from .layers import ModalityEmbedding


class BEAM(nn.Module):
    """Biomarker-Enhanced Assessment Model."""

    def __init__(
        self,
        cfdna_dim: int = 275,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Two parallel feature-extraction streams
        self.mri_extractor = MRI3DFeatureExtractor(input_channels=3, d_model=d_model)
        self.cfdna_encoder = CFDNAFeatureEncoder(input_dim=cfdna_dim, d_model=d_model)

        # Modality-specific embeddings P_mri / P_cfdna
        self.modality_embedding = ModalityEmbedding(n_modalities=2, d_model=d_model)

        # S_0 = [ LN(E_mri + P_mri); LN(E_cfdna + P_cfdna) ]
        self.norm_mri = nn.LayerNorm(d_model)
        self.norm_cfdna = nn.LayerNorm(d_model)

        # Stack of Transformer encoder layers (3 layers, 8 heads)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Classification head: MLP with FC + ReLU + Dropout, sigmoid output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, mri_data: torch.Tensor, cfdna_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mri_data:   (B, 3, D, H, W) -- T1, T2, DWI sequences
            cfdna_data: (B, cfdna_dim)
        Returns:
            logits: (B,) -- apply sigmoid to obtain the cancer probability p
        """
        # 1. Two feature-extraction streams -> 512-d embeddings
        e_mri = self.mri_extractor(mri_data)        # (B, d_model)
        e_cfdna = self.cfdna_encoder(cfdna_data)    # (B, d_model)

        # 2. Add modality embeddings and apply LayerNorm
        s_mri = self.norm_mri(self.modality_embedding(e_mri, 0))
        s_cfdna = self.norm_cfdna(self.modality_embedding(e_cfdna, 1))

        # 3. Form a length-2 token sequence
        s = torch.stack([s_mri, s_cfdna], dim=1)    # (B, 2, d_model)

        # 4. Three Transformer encoder layers
        for layer in self.transformer_layers:
            s = layer(s)

        # 5. Global average pooling over the sequence dim -> fused vector F_fused
        f_fused = s.mean(dim=1)                     # (B, d_model)

        # 6. Classifier (returns logits; sigmoid is applied in BCEWithLogitsLoss / inference)
        logits = self.classifier(f_fused).squeeze(-1)
        return logits
