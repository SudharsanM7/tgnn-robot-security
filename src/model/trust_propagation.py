"""
Trust propagation module for graph-based trust estimation.

Implements the update rule:
T_i(t+1) = alpha * T_i(t) + (1 - alpha) * sum_j w_ij * T_j(t)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class TrustPropagationModule(nn.Module):
    """
    Trust propagation using weighted neighbor aggregation.

    This module computes trust scores over graph nodes with iterative updates
    according to the formula above.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: Optional[int] = None,
        alpha: float = 0.85,
        num_steps: int = 3,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_steps = max(1, num_steps)

        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor([torch.logit(torch.tensor(alpha))], dtype=torch.float32))
            self.learnable_alpha = True
        else:
            self.register_buffer("fixed_alpha", torch.tensor([alpha], dtype=torch.float32))
            self.learnable_alpha = False

        # Initializes T_i(0) from node embeddings when external trust is not provided.
        self.trust_initializer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Edge weight estimator w_ij from edge attributes (optional).
        if edge_attr_dim is not None and edge_attr_dim > 0:
            self.edge_weight_predictor = nn.Sequential(
                nn.Linear(edge_attr_dim, max(8, edge_attr_dim)),
                nn.ReLU(),
                nn.Linear(max(8, edge_attr_dim), 1),
                nn.Sigmoid(),
            )
        else:
            self.edge_weight_predictor = None

    def _alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit)
        return self.fixed_alpha

    def _edge_weights(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_edges = edge_index.shape[1]
        device = edge_index.device

        if num_edges == 0:
            return torch.zeros((0,), device=device)

        if edge_attr is not None and self.edge_weight_predictor is not None:
            return self.edge_weight_predictor(edge_attr).squeeze(-1).clamp(min=1e-6)

        return torch.ones((num_edges,), device=device)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        initial_trust: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            node_embeddings: Node embeddings [N, H]
            edge_index: Graph connectivity [2, E]
            edge_attr: Optional edge features [E, F_e]
            initial_trust: Optional initial trust [N] or [N, 1]

        Returns:
            trust_scores: Trust score per node [N, 1]
            debug_info: Diagnostics for analysis
        """
        num_nodes = node_embeddings.shape[0]
        device = node_embeddings.device

        if initial_trust is None:
            trust = self.trust_initializer(node_embeddings)
        else:
            trust = initial_trust
            if trust.ndim == 1:
                trust = trust.unsqueeze(-1)
            trust = trust.clamp(0.0, 1.0)

        if edge_index.numel() == 0:
            return trust, {
                "alpha": self._alpha().detach(),
                "iterations": torch.tensor(self.num_steps, device=device),
                "edge_weights": torch.zeros((0,), device=device),
            }

        src, dst = edge_index[0], edge_index[1]
        edge_weights = self._edge_weights(edge_index, edge_attr)

        for _ in range(self.num_steps):
            # Weighted incoming trust from neighbors: sum_j w_ij * T_j(t)
            weighted_source = edge_weights.unsqueeze(-1) * trust[src]

            aggregated = torch.zeros((num_nodes, 1), device=device)
            aggregated.index_add_(0, dst, weighted_source)

            # Optional normalization by incoming weight mass for stability.
            weight_mass = torch.zeros((num_nodes,), device=device)
            weight_mass.index_add_(0, dst, edge_weights)
            weight_mass = weight_mass.clamp(min=1e-6).unsqueeze(-1)
            aggregated = aggregated / weight_mass

            alpha = self._alpha()
            trust = alpha * trust + (1.0 - alpha) * aggregated
            trust = trust.clamp(0.0, 1.0)

        return trust, {
            "alpha": self._alpha().detach(),
            "iterations": torch.tensor(self.num_steps, device=device),
            "edge_weights": edge_weights.detach(),
        }


class AdaptiveTrustNetwork(nn.Module):
    """
    Backward-compatible adaptive trust stack for experimentation.

    This stacks multiple trust propagation modules and optionally suppresses
    trust for nodes already flagged as malicious.
    """

    def __init__(
        self,
        num_nodes: Optional[int],
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.85,
        num_steps: int = 3,
        edge_attr_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)

        self.layers = nn.ModuleList(
            [
                TrustPropagationModule(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    alpha=alpha,
                    num_steps=num_steps,
                    learnable_alpha=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        previous_detections: Optional[torch.Tensor] = None,
        initial_trust: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        trust = initial_trust
        debug: Dict[str, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.layers):
            trust, layer_debug = layer(
                node_embeddings=node_embeddings,
                edge_index=edge_index,
                edge_attr=edge_attr,
                initial_trust=trust,
            )
            debug[f"layer_{layer_idx}_alpha"] = layer_debug["alpha"]
            node_embeddings = self.dropout(node_embeddings)

        if previous_detections is not None:
            if previous_detections.ndim == 1:
                previous_detections = previous_detections.unsqueeze(-1)
            trust = trust * (1.0 - 0.8 * previous_detections.float())
            trust = trust.clamp(0.0, 1.0)

        debug["final_trust"] = trust.detach()
        return trust, debug
