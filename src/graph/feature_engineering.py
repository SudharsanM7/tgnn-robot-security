"""
Feature engineering module: Advanced multi-modal feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class FeatureTensorBundle:
    """Container for ready-to-train tensors."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    message_embeddings: torch.Tensor
    message_sender_index: torch.Tensor
    message_receiver_index: torch.Tensor


class RunningFeatureScaler:
    """Per-feature z-score scaler with persistent statistics."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, key: str, features: np.ndarray) -> None:
        if features.size == 0:
            self.stats[key] = {
                "mean": np.zeros((1,), dtype=np.float32),
                "std": np.ones((1,), dtype=np.float32),
            }
            return

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        self.stats[key] = {
            "mean": np.mean(features, axis=0).astype(np.float32),
            "std": (np.std(features, axis=0) + self.eps).astype(np.float32),
        }

    def transform(self, key: str, features: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            raise KeyError(f"Scaler for key '{key}' has not been fitted")

        if features.size == 0:
            return features.astype(np.float32)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"]
        normalized = (features - mean) / std
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def fit_transform(self, key: str, features: np.ndarray) -> np.ndarray:
        self.fit(key, features)
        return self.transform(key, features)


class MessagePayloadEncoder(nn.Module):
    """
    Neural message payload encoder.

    Supports:
    - MLP encoder
    - Transformer encoder over payload tokens
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
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
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=max(1, min(num_heads, model_dim)),
                dim_feedforward=model_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
            self.output_projection = nn.Linear(model_dim, embedding_dim)
        else:
            layers: List[nn.Module] = []
            in_dim = input_dim
            for _ in range(max(1, num_layers - 1)):
                layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, embedding_dim))
            self.encoder = nn.Sequential(*layers)

    def forward(self, message_vectors: torch.Tensor) -> torch.Tensor:
        if message_vectors.ndim != 2:
            raise ValueError("message_vectors must have shape [num_messages, feature_dim]")

        if self.encoder_type == "transformer":
            tokens = message_vectors.unsqueeze(-1)
            tokens = self.input_projection(tokens)
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.output_projection(pooled)

        return self.encoder(message_vectors)


class MultiModalFeatureExtractor:
    """
    Advanced graph feature engineering pipeline.

    Node features include:
    - kinematic features (position, velocity)
    - communication statistics (message rates, neighbors)
    - historical embeddings
    - initial trust score

    Edge features include:
    - latency, packet loss, signal strength

    Message embeddings include:
    - payload encoding with MLP or Transformer encoder
    """

    def __init__(
        self,
        feature_normalization: bool = True,
        history_window: int = 10,
        historical_embedding_dim: int = 12,
        payload_vector_dim: int = 32,
        message_embedding_dim: int = 32,
        message_encoder_type: str = "mlp",
        device: str = "cpu",
    ):
        self.feature_normalization = feature_normalization
        self.history_window = history_window
        self.historical_embedding_dim = historical_embedding_dim
        self.payload_vector_dim = payload_vector_dim
        self.device = torch.device(device)

        self.scaler = RunningFeatureScaler()
        self.message_encoder = MessagePayloadEncoder(
            input_dim=payload_vector_dim,
            embedding_dim=message_embedding_dim,
            encoder_type=message_encoder_type,
        ).to(self.device)

    def extract_node_features(
        self,
        environment: Any,
        robot_id: int,
        timestep: Optional[int] = None,
        state_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        robot = environment.robots[robot_id]

        # Kinematics: position + velocity + speed
        position = _to_numpy(robot.position).astype(np.float32).reshape(-1)[:2]
        velocity = _to_numpy(robot.velocity).astype(np.float32).reshape(-1)[:2]
        speed = np.asarray([np.linalg.norm(velocity)], dtype=np.float32)
        kinematic = np.concatenate([position, velocity, speed], axis=0)

        # Communication statistics: incoming rate, outgoing rate, neighbor count
        received_history = list(getattr(robot, "received_messages_history", []))
        sent_history = list(getattr(robot, "sent_message_history", []))
        received_recent = received_history[-self.history_window :]
        sent_recent = sent_history[-self.history_window :]

        incoming_rate = np.asarray([len(received_recent) / max(self.history_window, 1)], dtype=np.float32)
        outgoing_rate = np.asarray([len(sent_recent) / max(self.history_window, 1)], dtype=np.float32)

        neighbor_count = 0
        if hasattr(environment, "communication_graph") and environment.communication_graph is not None:
            graph = environment.communication_graph
            if hasattr(graph, "degree"):
                neighbor_count = int(graph.degree(robot_id))
            elif hasattr(graph, "neighbors"):
                neighbor_count = len(list(graph.neighbors(robot_id)))
        neighbor_feature = np.asarray([neighbor_count], dtype=np.float32)

        # Initial trust score: estimated baseline trust to neighbors
        trust_scores = list(getattr(robot, "trust_scores", {}).values())
        initial_trust = np.asarray([
            float(np.mean(trust_scores)) if trust_scores else 1.0
        ], dtype=np.float32)

        # Historical embedding: trend over recent state history
        history_embedding = self._compute_historical_embedding(
            robot_id=robot_id,
            timestep=timestep,
            state_history=state_history,
            fallback_position=position,
            fallback_velocity=velocity,
            fallback_trust=initial_trust[0],
        )

        return np.concatenate(
            [
                kinematic,
                incoming_rate,
                outgoing_rate,
                neighbor_feature,
                initial_trust,
                history_embedding,
            ],
            axis=0,
        ).astype(np.float32)

    def extract_edge_features(
        self,
        environment: Any,
        source_id: int,
        target_id: int,
        transmission_events: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        latency = 0.0
        packet_loss = 0.0
        signal_strength = 0.0

        if transmission_events:
            relevant = [
                event
                for event in transmission_events
                if event.get("sender_id") == source_id and event.get("receiver_id") == target_id
            ]

            if relevant:
                latency = float(np.mean([_safe_float(event.get("latency", event.get("transmission_delay", 0.0))) for event in relevant]))
                success_flags = [1.0 if event.get("success", True) else 0.0 for event in relevant]
                packet_loss = float(1.0 - np.mean(success_flags))
                signal_strength = float(np.mean([_safe_float(event.get("signal_strength", 0.0)) for event in relevant]))

        if signal_strength == 0.0 and hasattr(environment, "communication_graph"):
            graph = environment.communication_graph
            if graph is not None and hasattr(graph, "has_edge") and graph.has_edge(source_id, target_id):
                edge_data = graph[source_id][target_id]
                signal_strength = _safe_float(edge_data.get("weight", edge_data.get("signal_strength", 0.0)))
                if latency == 0.0:
                    latency = _safe_float(edge_data.get("latency", edge_data.get("transmission_delay", 0.0)))

        return np.asarray([latency, packet_loss, signal_strength], dtype=np.float32)

    def vectorize_message_payload(self, message: Dict[str, Any]) -> np.ndarray:
        """Convert a message dictionary into a fixed-size numeric payload vector."""
        base_features = [
            _safe_float(message.get("sender_id", -1), -1.0),
            _safe_float(message.get("receiver_id", -1), -1.0),
            _safe_float(message.get("timestamp", 0.0), 0.0),
            1.0 if message.get("malicious", False) else 0.0,
            _safe_float(message.get("signal_strength", 0.0), 0.0),
            _safe_float(message.get("latency", message.get("transmission_delay", 0.0)), 0.0),
            _safe_float(message.get("packet_integrity", 1.0), 1.0),
            float(len(str(message.get("payload", {})))),
        ]

        payload = message.get("payload", {})
        flattened: List[float] = []
        self._flatten_payload(payload, flattened)

        feature_vector = np.asarray(base_features + flattened, dtype=np.float32)
        if feature_vector.shape[0] < self.payload_vector_dim:
            pad_size = self.payload_vector_dim - feature_vector.shape[0]
            feature_vector = np.pad(feature_vector, (0, pad_size), mode="constant")
        else:
            feature_vector = feature_vector[: self.payload_vector_dim]

        return feature_vector.astype(np.float32)

    def extract_message_features(self, message: Dict[str, Any]) -> np.ndarray:
        """Compatibility helper: returns engineered numeric message features."""
        return self.vectorize_message_payload(message)

    def compute_temporal_statistics(
        self,
        history: Sequence[Dict[str, Any]],
        key: str = "value",
        window_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute temporal summary statistics for a history sequence."""
        if not history:
            return {"mean": 0.0, "std": 0.0, "trend": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}

        effective_window = window_size or self.history_window
        recent = history[-effective_window:]
        values = np.asarray([_safe_float(item.get(key, 0.0), 0.0) for item in recent], dtype=np.float32)

        if values.size == 0:
            return {"mean": 0.0, "std": 0.0, "trend": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}

        if values.shape[0] < 2:
            trend = 0.0
        else:
            x_axis = np.arange(values.shape[0], dtype=np.float32)
            trend = float(np.polyfit(x_axis, values, deg=1)[0])

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "trend": trend,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }

    def encode_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        fit_scaler: bool = False,
    ) -> torch.Tensor:
        if not messages:
            return torch.zeros((0, self.message_encoder.embedding_dim), dtype=torch.float32, device=self.device)

        vectors = np.stack([self.vectorize_message_payload(message) for message in messages], axis=0)
        if self.feature_normalization:
            vectors = self._maybe_scale("message", vectors, fit_scaler)

        vector_tensor = torch.from_numpy(vectors).float().to(self.device)
        return self.message_encoder(vector_tensor)

    def build_node_feature_matrix(
        self,
        environment: Any,
        timestep: Optional[int] = None,
        state_history: Optional[Sequence[Dict[str, Any]]] = None,
        fit_scaler: bool = False,
    ) -> np.ndarray:
        num_robots = len(environment.robots)
        node_features = np.stack(
            [self.extract_node_features(environment, rid, timestep=timestep, state_history=state_history) for rid in range(num_robots)],
            axis=0,
        )
        if self.feature_normalization:
            node_features = self._maybe_scale("node", node_features, fit_scaler)
        return node_features.astype(np.float32)

    def build_edge_feature_matrix(
        self,
        environment: Any,
        transmission_events: Optional[Sequence[Dict[str, Any]]] = None,
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        edges: List[Tuple[int, int]] = []
        edge_features: List[np.ndarray] = []

        graph = getattr(environment, "communication_graph", None)
        if graph is not None and hasattr(graph, "edges"):
            for source_id, target_id in graph.edges():
                edges.append((int(source_id), int(target_id)))
                edge_features.append(self.extract_edge_features(environment, int(source_id), int(target_id), transmission_events=transmission_events))

        if not edges and transmission_events:
            for event in transmission_events:
                source_id = event.get("sender_id")
                target_id = event.get("receiver_id")
                if source_id is None or target_id is None:
                    continue
                edges.append((int(source_id), int(target_id)))
                edge_features.append(self.extract_edge_features(environment, int(source_id), int(target_id), transmission_events=transmission_events))

        if not edges:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_matrix = np.stack(edge_features, axis=0).astype(np.float32)

        if self.feature_normalization:
            edge_matrix = self._maybe_scale("edge", edge_matrix, fit_scaler)

        return edge_index, edge_matrix

    def create_ready_tensors(
        self,
        environment: Any,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        transmission_events: Optional[Sequence[Dict[str, Any]]] = None,
        timestep: Optional[int] = None,
        state_history: Optional[Sequence[Dict[str, Any]]] = None,
        fit_scaler: bool = False,
    ) -> FeatureTensorBundle:
        node_matrix = self.build_node_feature_matrix(
            environment,
            timestep=timestep,
            state_history=state_history,
            fit_scaler=fit_scaler,
        )
        edge_index, edge_matrix = self.build_edge_feature_matrix(
            environment,
            transmission_events=transmission_events,
            fit_scaler=fit_scaler,
        )

        message_list = list(messages or [])
        message_embeddings = self.encode_messages(message_list, fit_scaler=fit_scaler)

        sender_index = torch.tensor(
            [int(message.get("sender_id", -1)) for message in message_list],
            dtype=torch.long,
            device=self.device,
        )
        receiver_index = torch.tensor(
            [int(message.get("receiver_id", -1)) for message in message_list],
            dtype=torch.long,
            device=self.device,
        )

        return FeatureTensorBundle(
            node_features=torch.from_numpy(node_matrix).float().to(self.device),
            edge_index=torch.from_numpy(edge_index).long().to(self.device),
            edge_features=torch.from_numpy(edge_matrix).float().to(self.device),
            message_embeddings=message_embeddings,
            message_sender_index=sender_index,
            message_receiver_index=receiver_index,
        )

    def create_temporal_tensor_sequence(
        self,
        environments: Sequence[Any],
        message_sequences: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
        transmission_sequences: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
        state_history: Optional[Sequence[Dict[str, Any]]] = None,
        fit_scaler_on_first_step: bool = True,
    ) -> List[FeatureTensorBundle]:
        sequence: List[FeatureTensorBundle] = []

        for step_idx, environment in enumerate(environments):
            messages = list(message_sequences[step_idx]) if message_sequences and step_idx < len(message_sequences) else []
            transmissions = (
                list(transmission_sequences[step_idx])
                if transmission_sequences and step_idx < len(transmission_sequences)
                else []
            )

            bundle = self.create_ready_tensors(
                environment=environment,
                messages=messages,
                transmission_events=transmissions,
                timestep=step_idx,
                state_history=state_history,
                fit_scaler=(fit_scaler_on_first_step and step_idx == 0),
            )
            sequence.append(bundle)

        return sequence

    @staticmethod
    def stack_temporal_node_tensors(sequence: Sequence[FeatureTensorBundle]) -> torch.Tensor:
        """Stack node tensors as [T, N, F] for temporal models."""
        if not sequence:
            return torch.zeros((0, 0, 0), dtype=torch.float32)
        return torch.stack([bundle.node_features for bundle in sequence], dim=0)

    def _compute_historical_embedding(
        self,
        robot_id: int,
        timestep: Optional[int],
        state_history: Optional[Sequence[Dict[str, Any]]],
        fallback_position: np.ndarray,
        fallback_velocity: np.ndarray,
        fallback_trust: float,
    ) -> np.ndarray:
        if not state_history:
            base = np.concatenate(
                [
                    fallback_position,
                    fallback_velocity,
                    np.asarray([fallback_trust], dtype=np.float32),
                ],
                axis=0,
            )
            return self._compress_history_vector(base)

        recent = list(state_history[-self.history_window :])
        trajectory: List[np.ndarray] = []

        for state in recent:
            robots = state.get("robots", {})
            if robot_id not in robots:
                continue
            robot_state = robots[robot_id]
            position = _to_numpy(robot_state.get("position", fallback_position)).astype(np.float32).reshape(-1)[:2]
            velocity = _to_numpy(robot_state.get("velocity", fallback_velocity)).astype(np.float32).reshape(-1)[:2]
            trust = np.asarray([_safe_float(robot_state.get("reputation", fallback_trust), fallback_trust)], dtype=np.float32)
            trajectory.append(np.concatenate([position, velocity, trust], axis=0))

        if not trajectory:
            base = np.concatenate([fallback_position, fallback_velocity, np.asarray([fallback_trust], dtype=np.float32)], axis=0)
            return self._compress_history_vector(base)

        trajectory_matrix = np.stack(trajectory, axis=0)
        mean_vec = trajectory_matrix.mean(axis=0)
        std_vec = trajectory_matrix.std(axis=0)
        trend = trajectory_matrix[-1] - trajectory_matrix[0]

        composed = np.concatenate([mean_vec, std_vec, trend], axis=0)
        return self._compress_history_vector(composed)

    def _compress_history_vector(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] >= self.historical_embedding_dim:
            return vector[: self.historical_embedding_dim].astype(np.float32)

        pad_size = self.historical_embedding_dim - vector.shape[0]
        return np.pad(vector, (0, pad_size), mode="constant").astype(np.float32)

    def _flatten_payload(self, payload: Any, output: List[float], depth: int = 0, max_depth: int = 3) -> None:
        if depth > max_depth:
            return

        if isinstance(payload, dict):
            for key in sorted(payload.keys()):
                self._flatten_payload(payload[key], output, depth=depth + 1, max_depth=max_depth)
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                self._flatten_payload(value, output, depth=depth + 1, max_depth=max_depth)
        else:
            output.append(_safe_float(payload, default=0.0))

    def _maybe_scale(self, key: str, values: np.ndarray, fit_scaler: bool) -> np.ndarray:
        if not self.feature_normalization:
            return values.astype(np.float32)

        if fit_scaler or key not in self.scaler.stats:
            return self.scaler.fit_transform(key, values)

        return self.scaler.transform(key, values)
