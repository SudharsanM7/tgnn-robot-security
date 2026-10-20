"""
Temporal embedding module: Advanced temporal encoding techniques
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class TemporalEmbeddingLayer(nn.Module):
    """
    Advanced temporal embedding layer supporting multiple encoding schemes.
    
    Supports:
    - Sinusoidal (Fourier) embeddings
    - Learned embeddings
    - Linear time embeddings
    - RoPE (Rotary Position Embeddings)
    """
    
    def __init__(self, embedding_dim: int, max_timesteps: int = 10000,
                 encoding_type: str = "sine"):
        """
        Initialize temporal embedding layer.
        
        Args:
            embedding_dim: Dimension of embedding
            max_timesteps: Maximum number of timesteps to support
            encoding_type: "sine", "linear", or "learnable"
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_timesteps = max_timesteps
        self.encoding_type = encoding_type
        
        if encoding_type == "sinusoidal":
            self.register_buffer(
                "pe",
                self._create_sinusoidal_embeddings(embedding_dim, max_timesteps)
            )
        elif encoding_type == "learnable":
            self.pe = nn.Embedding(max_timesteps, embedding_dim)
        elif encoding_type == "linear":
            self.pe = nn.Linear(1, embedding_dim)
            
    def _create_sinusoidal_embeddings(self, d_model: int,
                                     max_len: int) -> torch.Tensor:
        """
        Create sinusoidal positional encodings.
        
        Based on "Attention is All You Need" (Vaswani et al., 2017)
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.
        
        Args:
            timesteps: Tensor of shape (batch_size,) or (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim) or 
            (batch_size, seq_len, embedding_dim)
        """
        if self.encoding_type == "sinusoidal":
            # Handle batching
            if len(timesteps.shape) == 1:
                timesteps = timesteps.long()
                timesteps = torch.clamp(timesteps, 0, self.max_timesteps - 1)
                return self.pe[timesteps]
            else:
                # For sequences
                timesteps = timesteps.long()
                timesteps = torch.clamp(timesteps, 0, self.max_timesteps - 1)
                return self.pe[timesteps]
        
        elif self.encoding_type == "learnable":
            timesteps = timesteps.long()
            timesteps = torch.clamp(timesteps, 0, self.max_timesteps - 1)
            return self.pe(timesteps)
        
        elif self.encoding_type == "linear":
            # Linear encoding: just scale timestep
            normalized = timesteps.float().unsqueeze(-1) / self.max_timesteps
            return self.pe(normalized)


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    More recent technique that applies rotation matrices to position embeddings.
    Particularly effective for maintaining relative position information.
    """
    
    def __init__(self, embedding_dim: int, max_seq_len: int = 10000, base: float = 10000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute RoPE embeddings for positions.
        
        Args:
            positions: Tensor of shape (batch_size, seq_len)
            
        Returns:
            RoPE embeddings for rotational application
        """
        # Compute outer product of positions and frequencies
        # Shape: (batch_size, seq_len, embedding_dim // 2)
        freqs = torch.einsum("...i,j->...ij", positions.float(), self.inv_freq)
        
        # Expand to full dimension by duplicating [sin, cos, sin, cos, ...]
        return torch.cat([freqs, freqs], dim=-1)


class TemporalContextAggregator(nn.Module):
    """
    Aggregates temporal context over a window of timesteps.
    
    Combines information from multiple time points using learned aggregation.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-layer temporal aggregation
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.activation = nn.ReLU()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features.
        
        Args:
            temporal_features: Shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Aggregated temporal context: (batch_size, hidden_dim)
        """
        # Apply layers with activation
        x = temporal_features
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Compute attention weights over time
        attn_logits = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted aggregation
        aggregated = torch.sum(x * attn_weights, dim=1)  # (batch_size, hidden_dim)
        
        return aggregated
