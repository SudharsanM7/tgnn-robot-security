"""
Explainability module for hybrid GNN malicious-node detection.

This module generates human-readable explanations using:
1) Attention weights (which neighbors influenced a decision)
2) Feature attribution (which node features contributed most)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class NodeExplanation:
    """Structured explanation for one node prediction."""

    node_id: int
    predicted_label: int
    malicious_probability: float
    trust_score: float
    influential_neighbors: List[Dict[str, float]]
    top_feature_contributions: List[Dict[str, float]]
    explanation_text: str


class HybridGNNExplainer:
    """
    Explainer for `HybridTemporalGNN`.

    Produces explanations for each detected malicious node and diagnostics
    for model behavior.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        top_k_neighbors: int = 5,
        top_k_features: int = 8,
    ):
        self.model = model
        self.device = torch.device(device)
        self.top_k_neighbors = top_k_neighbors
        self.top_k_features = top_k_features

    def explain_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        temporal_sequence: Optional[torch.Tensor] = None,
        message_features: Optional[torch.Tensor] = None,
        message_sender_index: Optional[torch.Tensor] = None,
        message_receiver_index: Optional[torch.Tensor] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, object]:
        """
        Generate explanations for detected malicious nodes.

        Returns:
            {
                "node_explanations": List[NodeExplanation],
                "debug_insights": Dict[str, float],
                "raw_output": Dict[str, torch.Tensor]
            }
        """
        self.model.eval()

        x = node_features.to(self.device)
        ei = edge_index.to(self.device)
        ea = edge_attr.to(self.device) if edge_attr is not None else None
        ts = temporal_sequence.to(self.device) if temporal_sequence is not None else None
        mf = message_features.to(self.device) if message_features is not None else None
        ms = message_sender_index.to(self.device) if message_sender_index is not None else None
        mr = message_receiver_index.to(self.device) if message_receiver_index is not None else None

        x = x.clone().detach().requires_grad_(True)

        output = self.model(
            node_features=x,
            edge_index=ei,
            edge_attr=ea,
            temporal_sequence=ts,
            message_features=mf,
            message_sender_index=ms,
            message_receiver_index=mr,
            return_attention=True,
        )

        probs = output["malicious_probability"]
        labels = output["predicted_labels"]
        trust_scores = output["trust_scores"].squeeze(-1)
        attention_weights = output.get("attention_weights", [])

        malicious_nodes = torch.where((labels == 1) & (probs >= threshold))[0].tolist()
        node_explanations: List[NodeExplanation] = []

        for node_id in malicious_nodes:
            neighbor_scores = self._neighbor_influence(
                target_node=node_id,
                edge_index=ei,
                attention_weights=attention_weights,
            )

            feature_contrib = self._feature_contributions(
                node_features=x,
                output=output,
                target_node=node_id,
                feature_names=feature_names,
            )

            explanation_text = self._build_human_readable_explanation(
                node_id=node_id,
                malicious_probability=float(probs[node_id].detach().cpu().item()),
                trust_score=float(trust_scores[node_id].detach().cpu().item()),
                neighbor_scores=neighbor_scores,
                feature_contrib=feature_contrib,
            )

            node_explanations.append(
                NodeExplanation(
                    node_id=int(node_id),
                    predicted_label=int(labels[node_id].detach().cpu().item()),
                    malicious_probability=float(probs[node_id].detach().cpu().item()),
                    trust_score=float(trust_scores[node_id].detach().cpu().item()),
                    influential_neighbors=neighbor_scores,
                    top_feature_contributions=feature_contrib,
                    explanation_text=explanation_text,
                )
            )

        debug_insights = self._debug_insights(
            malicious_probability=probs.detach().cpu().numpy(),
            trust_scores=trust_scores.detach().cpu().numpy(),
            predicted_labels=labels.detach().cpu().numpy(),
            attention_weights=attention_weights,
        )

        return {
            "node_explanations": node_explanations,
            "debug_insights": debug_insights,
            "raw_output": output,
        }

    def _neighbor_influence(
        self,
        target_node: int,
        edge_index: torch.Tensor,
        attention_weights: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Dict[str, float]]:
        """
        Identify neighbors that most influenced classification of `target_node`.

        Uses GAT attention returned as list of (edge_index, alpha) tuples.
        """
        influence: Dict[int, List[float]] = {}

        if not attention_weights:
            src = edge_index[0].detach().cpu().numpy()
            dst = edge_index[1].detach().cpu().numpy()
            neighbor_ids = [int(s) for s, d in zip(src, dst) if int(d) == target_node]
            return [{"neighbor_id": nid, "attention_score": 0.0} for nid in neighbor_ids[: self.top_k_neighbors]]

        for attn_entry in attention_weights:
            attn_edge_index, alpha = attn_entry
            src = attn_edge_index[0].detach().cpu().numpy()
            dst = attn_edge_index[1].detach().cpu().numpy()

            alpha_np = alpha.detach().cpu().numpy()
            if alpha_np.ndim == 2:
                alpha_np = alpha_np.mean(axis=1)

            for s, d, a in zip(src, dst, alpha_np):
                if int(d) != int(target_node):
                    continue
                influence.setdefault(int(s), []).append(float(a))

        merged = [
            {"neighbor_id": neighbor_id, "attention_score": float(np.mean(scores))}
            for neighbor_id, scores in influence.items()
        ]
        merged.sort(key=lambda item: item["attention_score"], reverse=True)
        return merged[: self.top_k_neighbors]

    def _feature_contributions(
        self,
        node_features: torch.Tensor,
        output: Dict[str, torch.Tensor],
        target_node: int,
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, float]]:
        """Compute feature attributions with gradient * input."""
        if node_features.grad is not None:
            node_features.grad.zero_()

        score = output["malicious_probability"][target_node]
        score.backward(retain_graph=True)

        gradients = node_features.grad[target_node].detach().cpu().numpy()
        values = node_features[target_node].detach().cpu().numpy()
        contribution = np.abs(gradients * values)

        top_indices = np.argsort(-contribution)[: self.top_k_features]
        results: List[Dict[str, float]] = []
        for idx in top_indices:
            feature_name = feature_names[idx] if feature_names is not None and idx < len(feature_names) else f"feature_{idx}"
            results.append(
                {
                    "feature_index": int(idx),
                    "feature_name": feature_name,
                    "contribution": float(contribution[idx]),
                    "feature_value": float(values[idx]),
                    "gradient": float(gradients[idx]),
                }
            )
        return results

    def _build_human_readable_explanation(
        self,
        node_id: int,
        malicious_probability: float,
        trust_score: float,
        neighbor_scores: Sequence[Dict[str, float]],
        feature_contrib: Sequence[Dict[str, float]],
    ) -> str:
        neighbor_text = ", ".join(
            [f"node {entry['neighbor_id']} (attn={entry['attention_score']:.3f})" for entry in neighbor_scores[:3]]
        )
        if not neighbor_text:
            neighbor_text = "no dominant neighbors"

        feature_text = ", ".join(
            [f"{entry['feature_name']} ({entry['contribution']:.4f})" for entry in feature_contrib[:3]]
        )
        if not feature_text:
            feature_text = "no dominant features"

        return (
            f"Node {node_id} was flagged malicious with probability {malicious_probability:.3f} "
            f"and trust score {trust_score:.3f}. Most influential neighbors: {neighbor_text}. "
            f"Top contributing features: {feature_text}."
        )

    def _debug_insights(
        self,
        malicious_probability: np.ndarray,
        trust_scores: np.ndarray,
        predicted_labels: np.ndarray,
        attention_weights: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        malicious_mask = predicted_labels == 1
        benign_mask = predicted_labels == 0

        avg_trust_malicious = float(np.mean(trust_scores[malicious_mask])) if np.any(malicious_mask) else 0.0
        avg_trust_benign = float(np.mean(trust_scores[benign_mask])) if np.any(benign_mask) else 0.0
        avg_prob_malicious = float(np.mean(malicious_probability[malicious_mask])) if np.any(malicious_mask) else 0.0

        attn_values: List[float] = []
        for _, alpha in attention_weights:
            alpha_np = alpha.detach().cpu().numpy()
            if alpha_np.ndim == 2:
                alpha_np = alpha_np.mean(axis=1)
            attn_values.extend(alpha_np.tolist())
        avg_attention = float(np.mean(attn_values)) if attn_values else 0.0

        return {
            "num_predicted_malicious": int(np.sum(malicious_mask)),
            "avg_trust_malicious": avg_trust_malicious,
            "avg_trust_benign": avg_trust_benign,
            "avg_probability_malicious_group": avg_prob_malicious,
            "avg_attention_weight": avg_attention,
            "trust_separation": float(avg_trust_benign - avg_trust_malicious),
        }