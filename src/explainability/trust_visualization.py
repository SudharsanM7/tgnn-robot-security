"""
Trust visualization module: Visualizing and analyzing trust propagation
"""

import numpy as np
from typing import Dict, List, Optional
import networkx as nx


class TrustFlowAnalyzer:
    """
    Analyzes trust flow through the network.
    """
    
    @staticmethod
    def extract_trust_flow(trust_scores: np.ndarray,
                          edge_index: np.ndarray,
                          threshold: float = 0.1) -> nx.DiGraph:
        """
        Create directed graph of trust flow.
        
        Args:
            trust_scores: Trust scores for each node (num_nodes,)
            edge_index: Edge connectivity (2, num_edges)
            threshold: Minimum trust to include edge
            
        Returns:
            Directed graph of significant trust flows
        """
        G = nx.DiGraph()
        
        for i, score in enumerate(trust_scores):
            G.add_node(i, trust_score=float(score))
        
        src, dst = edge_index
        for s, d in zip(src, dst):
            s_idx = int(s)
            d_idx = int(d)
            
            # Edge weight is trust of source node
            weight = trust_scores[s_idx]
            
            if weight >= threshold:
                G.add_edge(s_idx, d_idx, weight=weight)
        
        return G
    
    @staticmethod
    def compute_trust_influence(trust_scores: np.ndarray,
                               edge_index: np.ndarray) -> Dict[int, float]:
        """
        Compute how much each node influences others through trust.
        
        Args:
            trust_scores: Trust scores
            edge_index: Edge connectivity
            
        Returns:
            Influence map
        """
        influence = {}
        src, dst = edge_index
        
        for node_id in range(len(trust_scores)):
            # Outgoing edges (influence on neighbors)
            outgoing = (src == node_id).sum()
            # Incoming edges (influenced by neighbors)
            incoming = (dst == node_id).sum()
            
            influence[node_id] = float(outgoing + incoming) * trust_scores[node_id]
        
        return influence
    
    @staticmethod
    def identify_trust_bottlenecks(trust_scores: np.ndarray,
                                  edge_index: np.ndarray,
                                  percentile: float = 75) -> List[int]:
        """
        Identify nodes that are critical to trust flow.
        
        Args:
            trust_scores: Trust scores
            edge_index: Edge connectivity
            percentile: Percentile threshold for bottleneck detection
            
        Returns:
            List of bottleneck node IDs
        """
        G = TrustFlowAnalyzer.extract_trust_flow(
            trust_scores, edge_index, threshold=0.0
        )
        
        # Compute betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        # Find nodes above percentile
        threshold = np.percentile(list(betweenness.values()), percentile)
        bottlenecks = [node for node, cent in betweenness.items() if cent >= threshold]
        
        return bottlenecks
    
    @staticmethod
    def analyze_trust_propagation_layers(layer_trusts: List[np.ndarray]) -> Dict:
        """
        Analyze how trust evolves through propagation layers.
        
        Args:
            layer_trusts: List of trust scores from each propagation layer
            
        Returns:
            Analysis of trust evolution
        """
        analysis = {}
        
        for i, layer_trust in enumerate(layer_trusts):
            analysis[f'layer_{i}_mean'] = float(np.mean(layer_trust))
            analysis[f'layer_{i}_std'] = float(np.std(layer_trust))
            analysis[f'layer_{i}_min'] = float(np.min(layer_trust))
            analysis[f'layer_{i}_max'] = float(np.max(layer_trust))
        
        # Compute convergence
        if len(layer_trusts) > 1:
            L2_distances = []
            for i in range(len(layer_trusts) - 1):
                dist = np.linalg.norm(layer_trusts[i] - layer_trusts[i+1])
                L2_distances.append(dist)
            
            analysis['convergence_curve'] = L2_distances
            analysis['final_convergence_distance'] = float(L2_distances[-1]) if L2_distances else 0.0
        
        return analysis


class TrustExplainer:
    """
    Explains trust decisions for individual nodes.
    """
    
    @staticmethod
    def explain_node_trust(node_id: int, trust_scores: np.ndarray,
                          edge_index: np.ndarray,
                          node_features: Optional[np.ndarray] = None) -> Dict:
        """
        Explain trust score for a specific node.
        
        Args:
            node_id: Node of interest
            trust_scores: Trust scores for all nodes
            edge_index: Edge connectivity
            node_features: Node feature matrix
            
        Returns:
            Explanation dictionary
        """
        src, dst = edge_index
        
        # Find incoming edges (neighbors that influence this node)
        incoming_edges = np.where(dst == node_id)[0]
        incoming_neighbors = src[incoming_edges]
        
        # Find outgoing edges ( influences)
        outgoing_edges = np.where(src == node_id)[0]
        outgoing_neighbors = dst[outgoing_edges]
        
        explanation = {
            'node_id': node_id,
            'trust_score': float(trust_scores[node_id]),
            'incoming_neighbors': [int(n) for n in incoming_neighbors],
            'outgoing_neighbors': [int(n) for n in outgoing_neighbors],
            'avg_neighbor_trust': float(np.mean(trust_scores[incoming_neighbors])) if len(incoming_neighbors) > 0 else 0.0,
            'num_neighbors': len(incoming_neighbors) + len(outgoing_neighbors)
        }
        
        # Add feature-based info if available
        if node_features is not None and node_id < len(node_features):
            explanation['feature_vector_norm'] = float(np.linalg.norm(node_features[node_id]))
        
        return explanation
