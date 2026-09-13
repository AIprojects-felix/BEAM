"""Modality embeddings (P_mri / P_cfdna)."""

import torch
import torch.nn as nn


class ModalityEmbedding(nn.Module):
    """Learnable modality embeddings, one d_model-dim vector per modality."""

    def __init__(self, n_modalities: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)

    def forward(self, x: torch.Tensor, modality_idx: int) -> torch.Tensor:
        return x + self.embeddings[modality_idx]
