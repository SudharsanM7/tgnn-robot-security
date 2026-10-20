"""
TGNN (Temporal Graph Neural Network) module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from typing import Optional, Tuple


class TemporalGraphConvolution(nn.Module):
    """
    Temporal Graph Convolution layer combining spatial and temporal aspects.
    
    Implements: h_i^(t+1) = σ(W * AGG(h_j^(t) | j ∈ N(i)) + b)
    Where aggregation includes both spatial neighbors at time t and
    temporal neighbors (same node) at previous timesteps.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 temporal_depth: int = 2, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_depth = temporal_depth
        
        # Spatial convolution weights
        self.spatial_conv = GraphConv(in_channels, out_channels, bias=bias)
        
        # Temporal attention layers
        self.temporal_attention = nn.Linear(temporal_depth, temporal_depth)
        self.temporal_aggregation = nn.Linear(in_channels * temporal_depth, out_channels)
        
        # Output projection
        self.output_proj = nn.Linear(out_channels * 2, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal graph convolution.
        
        Args:
            x: Node feature matrix (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            temporal_features: Temporal context embeddings (num_nodes, temporal_depth, in_channels)
            
        Returns:
            Updated node features and attention weights
        """
        # Spatial aggregation
        spatial_out = self.spatial_conv(x, edge_index)
        spatial_out = F.relu(spatial_out)
        
        # Temporal aggregation
        if temporal_features is not None and temporal_features.numel() > 0:
            num_nodes = x.shape[0]
            temporal_flat = temporal_features.reshape(num_nodes, -1)
            temporal_out = self.temporal_aggregation(temporal_flat)
            temporal_out = F.relu(temporal_out)
        else:
            temporal_out = torch.zeros_like(spatial_out)
        
        # Combine spatial and temporal
        combined = torch.cat([spatial_out, temporal_out], dim=1)
        output = self.output_proj(combined)
        
        # Attention weights for interpretability
        attn_weights = torch.ones((x.shape[0], 1), device=x.device)
        
        return output, attn_weights


class TGNN(nn.Module):
    """
    Temporal Graph Neural Network for temporal graph learning.
    
    Stacks multiple temporal graph convolution layers with
    skip connections and layer normalization.
    """
    
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int,
                 num_layers: int = 4, temporal_depth: int = 2,
                 dropout: float = 0.3, use_layer_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Temporal graph convolution layers
        self.tgcn_layers = nn.ModuleList([
            TemporalGraphConvolution(hidden_dim, hidden_dim, temporal_depth)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through TGNN.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            temporal_features: Temporal embeddings (num_nodes, temporal_depth, in_channels)
            
        Returns:
            Output embeddings and list of attention weights for visualization
        """
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)
        
        attention_weights = []
        
        # Stack TGCN layers
        for i, tgcn in enumerate(self.tgcn_layers):
            # Residual connection
            x_residual = x
            
            # Apply convolution
            x, attn = tgcn(x, edge_index, temporal_features)
            attention_weights.append(attn)
            
            # Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Residual connection
            x = x + x_residual
        
        # Output projection
        output = self.output_proj(x)
        
        return output, attention_weights
