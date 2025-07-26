import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_, constant_


class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with advanced features"""
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1, 
                 use_bias: bool = True, temperature: float = 1.0, 
                 attention_dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections with optional bias
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            xavier_uniform_(module.weight)
            if module.bias is not None:
                constant_(module.bias, 0.)
    
    def forward(self, query, key, value, mask=None, return_attention=False):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(attention_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        return output


class TransformerEncoderLayer(nn.Module):
    """Enhanced transformer encoder layer with advanced features"""
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, activation: str = 'relu', 
                 use_pre_norm: bool = False, use_glu: bool = False):
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Enhanced feed-forward network
        if use_glu:
            # Gated Linear Units for better expressiveness
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff * 2),
                GLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
        else:
            # Standard feed-forward with configurable activation
            activation_fn = self._get_activation_fn(activation)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation_fn,
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Optional: learnable scaling parameters
        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ffn = nn.Parameter(torch.ones(1))
    
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()
    
    def forward(self, x, mask=None, return_attention=False):
        """Forward pass with pre/post normalization options"""
        if self.use_pre_norm:
            # Pre-normalization (more stable training)
            # Multi-head attention
            normed_x = self.norm1(x)
            if return_attention:
                attn_output, attention_weights = self.multi_head_attention(
                    normed_x, normed_x, normed_x, mask, return_attention=True
                )
            else:
                attn_output = self.multi_head_attention(normed_x, normed_x, normed_x, mask)
            x = x + self.alpha_attn * attn_output
            
            # Feed forward
            ff_output = self.feed_forward(self.norm2(x))
            x = x + self.alpha_ffn * ff_output
        else:
            # Post-normalization (original transformer)
            # Multi-head attention
            if return_attention:
                attn_output, attention_weights = self.multi_head_attention(
                    x, x, x, mask, return_attention=True
                )
            else:
                attn_output = self.multi_head_attention(x, x, x, mask)
            x = self.norm1(x + self.alpha_attn * attn_output)
            
            # Feed forward
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.alpha_ffn * ff_output)
        
        if return_attention:
            return x, attention_weights
        return x


class GLU(nn.Module):
    """Gated Linear Unit"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class MRI3DFeatureExtractor(nn.Module):
    """Enhanced 3D CNN for MRI feature extraction with advanced features"""
    def __init__(self, input_channels: int = 3, feature_dim: int = 512, 
                 use_attention: bool = True, use_residual: bool = True,
                 dropout: float = 0.1, use_spectral_norm: bool = False):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Helper function for applying spectral norm
        def maybe_spectral_norm(module):
            return spectral_norm(module) if use_spectral_norm else module
        
        # Enhanced convolutional blocks with residual connections
        self.conv_blocks = nn.ModuleList()
        
        channels = [input_channels, 32, 64, 128, 256]
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            
            block = nn.ModuleDict({
                'conv1': maybe_spectral_norm(nn.Conv3d(in_ch, out_ch, 3, padding=1)),
                'bn1': nn.BatchNorm3d(out_ch),
                'conv2': maybe_spectral_norm(nn.Conv3d(out_ch, out_ch, 3, padding=1)),
                'bn2': nn.BatchNorm3d(out_ch),
                'pool': nn.MaxPool3d(2),
                'dropout': nn.Dropout3d(dropout)
            })
            
            # Residual connection for same channel dimension
            if use_residual and i > 0 and in_ch == out_ch:
                block['residual'] = True
            else:
                block['residual'] = False
                if use_residual:
                    block['downsample'] = maybe_spectral_norm(nn.Conv3d(in_ch, out_ch, 1))
            
            self.conv_blocks.append(block)
        
        # Spatial attention mechanism
        if use_attention:
            self.spatial_attention = SpatialAttention3D(256)
        
        # Adaptive pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Enhanced feature projection with residual connection
        self.feature_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Enhanced forward pass with residual connections and attention"""
        # x shape: (B, 3, D, H, W)
        
        for i, block in enumerate(self.conv_blocks):
            identity = x
            
            # First convolution
            x = F.relu(block['bn1'](block['conv1'](x)))
            
            # Second convolution
            x = block['bn2'](block['conv2'](x))
            
            # Residual connection
            if self.use_residual:
                if block['residual']:
                    x = x + identity
                elif 'downsample' in block:
                    identity = block['downsample'](identity)
                    x = x + identity
            
            x = F.relu(x)
            x = block['dropout'](x)
            x = block['pool'](x)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            x = self.spatial_attention(x)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # Feature projection
        features = self.feature_projection(x)  # (B, feature_dim)
        
        return features


class SpatialAttention3D(nn.Module):
    """3D Spatial attention mechanism"""
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class CFDNAFeatureEncoder(nn.Module):
    """Enhanced fully connected encoder for cfDNA features"""
    def __init__(self, input_dim: int, feature_dim: int = 512, dropout: float = 0.2,
                 use_attention: bool = True, num_layers: int = 3, 
                 activation: str = 'relu', use_batch_norm: bool = True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Build dynamic encoder architecture
        layers = []
        dims = self._get_layer_dims(input_dim, feature_dim, num_layers)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add batch normalization (except for last layer)
            if use_batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Add activation (except for last layer)
            if i < len(dims) - 2:
                layers.append(self._get_activation_fn(activation))
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Feature attention mechanism
        if use_attention:
            self.feature_attention = FeatureAttention(feature_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _get_layer_dims(self, input_dim: int, output_dim: int, num_layers: int) -> List[int]:
        """Calculate layer dimensions for gradual reduction"""
        if num_layers == 1:
            return [input_dim, output_dim]
        
        # Logarithmic spacing for smooth dimension reduction
        log_start = math.log(input_dim)
        log_end = math.log(output_dim)
        
        dims = [input_dim]
        for i in range(1, num_layers):
            ratio = i / (num_layers - 1)
            log_dim = log_start * (1 - ratio) + log_end * ratio
            dims.append(int(math.exp(log_dim)))
        
        dims[-1] = output_dim  # Ensure exact output dimension
        return dims
    
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Enhanced forward pass with attention"""
        # Basic encoding
        features = self.encoder(x)
        
        # Apply feature attention if enabled
        if self.use_attention:
            features = self.feature_attention(features)
        
        return features


class FeatureAttention(nn.Module):
    """Channel-wise feature attention for cfDNA features"""
    def __init__(self, feature_dim: int, reduction_ratio: int = 8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction_ratio, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Global average pooling (already 1D)
        attention_weights = self.attention(x)
        return x * attention_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention between MRI and cfDNA features"""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mri_to_cfdna_attention = MultiHeadAttention(feature_dim, num_heads, dropout)
        self.cfdna_to_mri_attention = MultiHeadAttention(feature_dim, num_heads, dropout)
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mri_features, cfdna_features):
        """
        Args:
            mri_features: (B, feature_dim)
            cfdna_features: (B, feature_dim)
        
        Returns:
            fused_features: (B, feature_dim)
        """
        # Add sequence dimension for attention
        mri_seq = mri_features.unsqueeze(1)  # (B, 1, feature_dim)
        cfdna_seq = cfdna_features.unsqueeze(1)  # (B, 1, feature_dim)
        
        # Cross-modal attention
        mri_attended = self.mri_to_cfdna_attention(mri_seq, cfdna_seq, cfdna_seq)
        cfdna_attended = self.cfdna_to_mri_attention(cfdna_seq, mri_seq, mri_seq)
        
        # Remove sequence dimension
        mri_attended = mri_attended.squeeze(1)
        cfdna_attended = cfdna_attended.squeeze(1)
        
        # Fusion
        combined = torch.cat([mri_attended, cfdna_attended], dim=-1)
        fused = self.fusion_layer(combined)
        fused = self.dropout(fused)
        
        return fused