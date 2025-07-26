import torch
import torch.nn as nn
from .components import MRI3DFeatureExtractor, CFDNAFeatureEncoder, TransformerEncoderLayer
from .layers import ModalityEmbedding


class BEAM(nn.Module):
    """
    Biomarker-Enhanced Assessment Model
    Multi-modal fusion of MRI and cfDNA features for prostate cancer detection
    """
    def __init__(
        self,
        cfdna_dim: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Feature extractors
        self.mri_extractor = MRI3DFeatureExtractor(input_channels=3, feature_dim=d_model)
        self.cfdna_encoder = CFDNAFeatureEncoder(cfdna_dim, d_model, dropout)
        
        # Modality embeddings
        self.modality_embedding = ModalityEmbedding(n_modalities=2, d_model=d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model*4, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, mri_data, cfdna_data):
        """
        Args:
            mri_data: (B, 3, D, H, W) - T1, T2, DWI sequences
            cfdna_data: (B, cfdna_dim) - cfDNA features
        
        Returns:
            output: (B,) - cancer probability
        """
        # Extract features
        mri_features = self.mri_extractor(mri_data)  # (B, d_model)
        cfdna_features = self.cfdna_encoder(cfdna_data)  # (B, d_model)
        
        # Add modality embeddings
        mri_features = self.modality_embedding(mri_features, 0)
        cfdna_features = self.modality_embedding(cfdna_features, 1)
        
        # Stack features for transformer
        features = torch.stack([mri_features, cfdna_features], dim=1)  # (B, 2, d_model)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            features = layer(features)
        
        # Global average pooling
        pooled_features = torch.mean(features, dim=1)  # (B, d_model)
        
        # Classification
        output = self.classifier(pooled_features)  # (B, 1)
        
        return output.squeeze(-1)