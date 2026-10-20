"""
Graph Attention Network (GAT) module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention Network combining multiple attention heads.
    
    Each head independently attends to neighbors, capturing different
    aspects of graph structure and node interactions.
    """
    
    def __init__(self, in_channels: int, hidden_dims: list,
                 num_heads: list, dropout: float = 0.3,
                 concat: bool = True):
        """
        Initialize multi-head GAT.
        
        Args:
            in_channels: Input feature dimension
            hidden_dims: List of hidden dimensions for each layer
            num_heads: List of number of attention heads per layer
            dropout: Dropout rate
            concat: Whether to concatenate head outputs
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        self.gat_layers = nn.ModuleList()
        self.activation = nn.ReLU()
        
        # Build GAT layers
        in_dim = in_channels
        for i, (hidden_dim, heads) in enumerate(zip(hidden_dims, num_heads)):
            # Output dimension per head when concatenating
            out_dim = hidden_dim // heads if concat else hidden_dim
            
            gat_layer = GATConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_self_loops=True
            )
            self.gat_layers.append(gat_layer)
            
            # Update dimension for next layer
            in_dim = hidden_dim
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through GAT.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_features)
            
        Returns:
            Node embeddings and attention weights for visualization
        """
        attention_weights = []
        
        for i, gat_layer in enumerate(self.gat_layers):
            # Store attention from GAT (if available)
            x = gat_layer(x, edge_index, edge_attr)
            
            if i < len(self.gat_layers) - 1:  # Not last layer
                x = self.activation(x)
                x = self.dropout_layer(x)
            
            # Note: Attention weights are not directly exposed by GATConv
            # Would need custom implementation for full attention access
        
        return x, attention_weights


class GATWithAttentionVisual(nn.Module):
    """
    GAT variant with explicit attention weight access for visualization.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Attention parameters
        self.W = nn.Linear(in_channels, num_heads * hidden_channels)
        self.a = nn.Parameter(torch.randn(1, num_heads, 2 * hidden_channels))
        
        self.output_proj = nn.Linear(num_heads * hidden_channels, out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight extraction.
        
        Returns:
            Node embeddings and attention weight matrix
        """
        num_nodes = x.shape[0]
        
        # Project features
        x_proj = self.W(x)  # (num_nodes, num_heads * hidden_channels)
        x_proj = x_proj.view(num_nodes, self.num_heads, -1)  # (num_nodes, num_heads, hidden_channels)
        
        # Compute attention logits for each edge
        edge_h = []
        attention_weights = torch.zeros(num_nodes, num_nodes, device=x.device)
        
        src, dst = edge_index
        
        # Concatenate source and destination features
        edge_src = x_proj[src]  # (num_edges, num_heads, hidden_channels)
        edge_dst = x_proj[dst]  # (num_edges, num_heads, hidden_channels)
        
        edge_feat = torch.cat([edge_src, edge_dst], dim=2)  # (num_edges, num_heads, 2*hidden_channels)
        
        # Compute attention logits
        logits = torch.einsum('ehi,iho->eho', edge_feat, self.a)  # (num_edges, num_heads, 1)
        logits = self.activation(logits).squeeze(-1)  # (num_edges, num_heads)
        
        # Compute attention coefficients
        attn = F.softmax(logits, dim=-1)
        
        # Store average attention weights
        for i in range(num_nodes):
            neighbors = dst[src == i]
            if len(neighbors) > 0:
                attention_weights[i, neighbors] = attn[src == i].mean(dim=1)
        
        # Apply attention and aggregate  
        attn_applied = attn.unsqueeze(-1) * edge_src  # (num_edges, num_heads, hidden_channels)
        
        # Aggregate for each destination node
        aggregated = torch.zeros(num_nodes, self.num_heads, self.hidden_channels, device=x.device)
        aggregated.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(attn_applied), attn_applied)
        
        # Reshape and project
        aggregated = aggregated.view(num_nodes, -1)
        output = self.output_proj(aggregated)
        output = self.dropout_layer(output)
        
        return output, attention_weights
