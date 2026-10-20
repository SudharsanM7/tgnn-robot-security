"""
Hybrid GNN model: message encoder + temporal modeling + GAT + trust propagation.

This module provides a modular architecture for node classification
(normal vs malicious) and trust estimation per node.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .trust_propagation import AdaptiveTrustNetwork


class MessageEmbeddingEncoder(nn.Module):
    """
    Encodes message-level features and aggregates message context onto nodes.

    Supports either MLP or Transformer encoding for message payload vectors.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        encoder_type: str = "mlp",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "transformer":
            model_dim = max(hidden_dim, embedding_dim)
            self.input_projection = nn.Linear(1, model_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=max(1, min(num_heads, model_dim)),
                dim_feedforward=model_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
            self.output_projection = nn.Linear(model_dim, embedding_dim)
        else:
            mlp_layers: List[nn.Module] = []
            in_dim = input_dim
            for _ in range(max(1, num_layers - 1)):
                mlp_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            mlp_layers.append(nn.Linear(in_dim, embedding_dim))
            self.encoder = nn.Sequential(*mlp_layers)

    def _encode_messages(self, message_features: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "transformer":
            tokens = message_features.unsqueeze(-1)
            tokens = self.input_projection(tokens)
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.output_projection(pooled)
        return self.encoder(message_features)

    def forward(
        self,
        num_nodes: int,
        message_features: Optional[torch.Tensor],
        sender_index: Optional[torch.Tensor],
        receiver_index: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_message_context: [N, embedding_dim]
            message_embeddings: [M, embedding_dim]
        """
        device = message_features.device if message_features is not None else sender_index.device if sender_index is not None else receiver_index.device if receiver_index is not None else torch.device("cpu")

        if message_features is None or message_features.numel() == 0:
            return (
                torch.zeros((num_nodes, self.embedding_dim), device=device),
                torch.zeros((0, self.embedding_dim), device=device),
            )

        if message_features.ndim != 2:
            raise ValueError("message_features must be shaped [num_messages, message_feature_dim]")

        message_embeddings = self._encode_messages(message_features)

        if sender_index is None or receiver_index is None:
            return torch.zeros((num_nodes, self.embedding_dim), device=message_embeddings.device), message_embeddings

        node_context = torch.zeros((num_nodes, self.embedding_dim), device=message_embeddings.device)
        node_mass = torch.zeros((num_nodes, 1), device=message_embeddings.device)

        valid_sender = (sender_index >= 0) & (sender_index < num_nodes)
        valid_receiver = (receiver_index >= 0) & (receiver_index < num_nodes)

        if valid_sender.any():
            sender_ids = sender_index[valid_sender]
            sender_embeddings = message_embeddings[valid_sender]
            node_context.index_add_(0, sender_ids, sender_embeddings)
            node_mass.index_add_(0, sender_ids, torch.ones((sender_ids.shape[0], 1), device=node_mass.device))

        if valid_receiver.any():
            receiver_ids = receiver_index[valid_receiver]
            receiver_embeddings = message_embeddings[valid_receiver]
            node_context.index_add_(0, receiver_ids, receiver_embeddings)
            node_mass.index_add_(0, receiver_ids, torch.ones((receiver_ids.shape[0], 1), device=node_mass.device))

        node_mass = node_mass.clamp(min=1.0)
        node_context = node_context / node_mass

        return node_context, message_embeddings


class TemporalSequenceEncoder(nn.Module):
    """Encodes node-level temporal history with a GRU."""

    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)

        self.projection = nn.Linear(in_channels, hidden_dim)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, temporal_sequence: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Current node features [N, F]
            temporal_sequence: Optional history [N, T, F]
        """
        if temporal_sequence is None or temporal_sequence.numel() == 0:
            return self.projection(x)

        if temporal_sequence.ndim != 3:
            raise ValueError("temporal_sequence must be shaped [num_nodes, timesteps, feature_dim]")

        out, _ = self.gru(temporal_sequence)
        return out[:, -1, :]


class HybridTemporalGNN(nn.Module):
    """
    Hybrid model with four modular components:
    1. Message embedding encoder
    2. Temporal sequence encoder
    3. GAT stack
    4. Trust propagation network

    Outputs:
    - Node classification logits/probabilities (malicious vs normal)
    - Trust score per node
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int = 2,
        message_in_dim: int = 32,
        message_embedding_dim: int = 32,
        message_encoder_type: str = "mlp",
        temporal_layers: int = 1,
        gat_layers: int = 2,
        num_heads: int = 4,
        edge_attr_dim: Optional[int] = None,
        trust_alpha: float = 0.85,
        trust_steps: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)

        # 1) Message embedding encoder
        self.message_encoder = MessageEmbeddingEncoder(
            input_dim=message_in_dim,
            embedding_dim=message_embedding_dim,
            encoder_type=message_encoder_type,
            dropout=dropout,
        )

        # 2) Temporal graph learning (sequence modeling)
        self.temporal_encoder = TemporalSequenceEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # Fusion of current + temporal + message contexts
        fusion_dim = in_channels + hidden_dim + message_embedding_dim
        self.input_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3) GAT layers
        self.gat_layers = nn.ModuleList()
        gat_in_dim = hidden_dim
        for _ in range(max(1, gat_layers)):
            per_head_dim = max(1, hidden_dim // max(1, num_heads))
            self.gat_layers.append(
                GATConv(
                    in_channels=gat_in_dim,
                    out_channels=per_head_dim,
                    heads=max(1, num_heads),
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    add_self_loops=True,
                )
            )
            gat_in_dim = per_head_dim * max(1, num_heads)

        self.post_gat_proj = nn.Linear(gat_in_dim, hidden_dim)

        # 4) Trust propagation mechanism
        self.trust_net = AdaptiveTrustNetwork(
            num_nodes=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
            alpha=trust_alpha,
            num_steps=trust_steps,
            edge_attr_dim=edge_attr_dim,
        )

        # Node classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _apply_gat(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        return_attention: bool,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        attentions: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for gat in self.gat_layers:
            residual = h
            if return_attention:
                if self.edge_attr_dim is not None:
                    h, attn = gat(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                else:
                    h, attn = gat(h, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                if self.edge_attr_dim is not None:
                    h = gat(h, edge_index, edge_attr=edge_attr)
                else:
                    h = gat(h, edge_index)

            h = F.relu(h)
            h = self.dropout(h)
            if h.shape == residual.shape:
                h = h + residual

        h = self.post_gat_proj(h)
        return h, attentions

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        temporal_sequence: Optional[torch.Tensor] = None,
        message_features: Optional[torch.Tensor] = None,
        message_sender_index: Optional[torch.Tensor] = None,
        message_receiver_index: Optional[torch.Tensor] = None,
        initial_trust: Optional[torch.Tensor] = None,
        previous_detections: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            node_features: [N, F_node]
            edge_index: [2, E]
            edge_attr: Optional [E, F_edge]
            temporal_sequence: Optional [N, T, F_node]
            message_features: Optional [M, F_msg]
            message_sender_index: Optional [M]
            message_receiver_index: Optional [M]
            initial_trust: Optional [N] or [N, 1]
            previous_detections: Optional [N] binary indicator
            return_attention: If True, returns attention weights from GAT layers

        Returns:
            Dictionary with logits, class probabilities, predicted labels,
            malicious probabilities, trust scores, embeddings, and diagnostics.
        """
        num_nodes = node_features.shape[0]

        # Message context
        node_message_context, message_embeddings = self.message_encoder(
            num_nodes=num_nodes,
            message_features=message_features,
            sender_index=message_sender_index,
            receiver_index=message_receiver_index,
        )

        # Temporal context
        temporal_context = self.temporal_encoder(node_features, temporal_sequence)

        # Fuse all modalities
        fused_input = torch.cat([node_features, temporal_context, node_message_context], dim=-1)
        h = self.input_fusion(fused_input)

        # Graph attention learning
        h, attention_weights = self._apply_gat(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention=return_attention,
        )

        # Trust propagation using equation:
        # T_i(t+1) = alpha * T_i(t) + (1-alpha) * sum_j w_ij * T_j(t)
        trust_scores, trust_debug = self.trust_net(
            node_embeddings=h,
            edge_index=edge_index,
            edge_attr=edge_attr,
            previous_detections=previous_detections,
            initial_trust=initial_trust,
        )

        # Node classification
        logits = self.classifier(h)
        class_probabilities = torch.softmax(logits, dim=-1)

        # Hybrid malicious probability combines classifier and inverse trust.
        malicious_probability = 0.65 * class_probabilities[:, 1] + 0.35 * (1.0 - trust_scores.squeeze(-1))
        predicted_labels = (malicious_probability >= 0.5).long()

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "predictions": class_probabilities,
            "predicted_labels": predicted_labels,
            "malicious_probability": malicious_probability,
            "trust_scores": trust_scores,
            "node_embeddings": h,
            "message_embeddings": message_embeddings,
        }

        if return_attention:
            output["attention_weights"] = attention_weights

        # Diagnostics for trust behavior.
        output["trust_debug"] = trust_debug
        return output

    def get_model_config(self) -> Dict[str, int]:
        """Return a compact config dictionary for experiment tracking."""
        return {
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "edge_attr_dim": self.edge_attr_dim if self.edge_attr_dim is not None else 0,
        }
