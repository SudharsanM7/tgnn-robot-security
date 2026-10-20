"""
Attention analysis module: Interpreting model attention patterns
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import networkx as nx


class AttentionAnalyzer:
    """
    Analyzes attention mechanisms to explain model decisions.
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_cache = {}
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from model.
        
        Returns:
            Dictionary with attention matrices from different layers
        """
        return self.model.get_attention_for_visualization()
    
    def compute_node_importance(self, attention_weights: Dict) -> np.ndarray:
        """
        Compute node importance from attention weights.
        
        Args:
            attention_weights: Attention matrices from model
            
        Returns:
            Importance scores for each node
        """
        # Average attention received from neighbors
        gat_attn = attention_weights.get('gat_attention')
        
        if gat_attn is None or gat_attn.shape[0] == 0:
            return np.ones(self.model.num_nodes if hasattr(self.model, 'num_nodes') else 10)
        
        # Sum incoming attention
        importance = gat_attn.sum(dim=0).cpu().numpy()
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        return importance
    
    def compute_edge_importance(self, edge_index: torch.Tensor,
                               attention_weights: Dict) -> np.ndarray:
        """
        Compute importance of edges based on attention.
        
        Args:
            edge_index: Edge connectivity
            attention_weights: Attention matrices
            
        Returns:
            Importance scores for each edge
        """
        gat_attn = attention_weights.get('gat_attention')
        
        if gat_attn is None or gat_attn.shape[0] == 0:
            return np.ones(edge_index.shape[1])
        
        edge_importance = []
        src, dst = edge_index
        
        for i in range(len(src)):
            s, d = src[i].item(), dst[i].item()
            # Attention weight for this edge
            if d < gat_attn.shape[0] and s < gat_attn.shape[1]:
                weight = gat_attn[d, s].item() if d < gat_attn.shape[0] and s < gat_attn.shape[1] else 0
                edge_importance.append(weight)
            else:
                edge_importance.append(0)
        
        return np.array(edge_importance)
    
    def extract_attention_subgraph(self, node_id: int,
                                  attention_weights: Dict,
                                  k_top: int = 5) -> Tuple[list, list]:
        """
        Extract top-k neighbors attending to a node.
        
        Args:
            node_id: Node of interest
            attention_weights: Attention matrices
            k_top: Number of top neighbors
            
        Returns:
            Tuple of (node_ids, attention_weights)
        """
        gat_attn = attention_weights.get('gat_attention')
        
        if gat_attn is None:
            return [], []
        
        # Get attention received by node_id
        node_attn = gat_attn[node_id, :] if node_id < gat_attn.shape[0] else None
        
        if node_attn is None:
            return [], []
        
        # Get top-k
        top_k_values, top_k_indices = torch.topk(
            node_attn, min(k_top, len(node_attn))
        )
        
        return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()


class AttentionVisualizer:
    """
    Visualizes attention patterns for interpretation.
    """
    
    @staticmethod
    def create_attention_graph(edge_index: torch.Tensor,
                              edge_weights: np.ndarray,
                              threshold: float = 0.1) -> nx.DiGraph:
        """
        Create directed graph of attention weights.
        
        Args:
            edge_index: Edge connectivity
            edge_weights: Edge weights (attention)
            threshold: Minimum weight to include edge
            
        Returns:
            Directed graph
        """
        G = nx.DiGraph()
        
        src, dst = edge_index
        for i, (s, d) in enumerate(zip(src, dst)):
            if edge_weights[i] >= threshold:
                G.add_edge(s.item(), d.item(), weight=edge_weights[i])
        
        return G
    
    @staticmethod
    def get_attention_summary(attention_weights: Dict) -> Dict[str, float]:
        """
        Compute summary statistics of attention.
        
        Args:
            attention_weights: Attention matrices
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        for name, attn in attention_weights.items():
            if attn is None:
                continue
            
            attn_np = attn.cpu().numpy() if hasattr(attn, 'cpu') else attn
            
            summary[f'{name}_mean'] = float(np.mean(attn_np))
            summary[f'{name}_std'] = float(np.std(attn_np))
            summary[f'{name}_max'] = float(np.max(attn_np))
            summary[f'{name}_min'] = float(np.min(attn_np))
        
        return summary
