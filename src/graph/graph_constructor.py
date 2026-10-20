"""
Graph constructor module: Build temporal graphs from simulation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset
import networkx as nx
from loguru import logger


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TemporalGraphConstructor:
    """
    Constructs temporal dynamic graphs from multi-robot simulation.
    
    Key aspects:
    - Sliding window temporal aggregation
    - Multi-modal feature extraction (node, edge, message)
    - Dynamic graph evolution tracking
    - Temporal edge prediction for future timesteps
    """
    
    def __init__(self, window_size: int = 20, window_stride: int = 5,
                 max_history: int = 100):
        """
        Initialize graph constructor.
        
        Args:
            window_size: Timesteps in each temporal window
            window_stride: Sliding stride between windows
            max_history: Maximum history to maintain
        """
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_history = max_history
        
        self.message_history: List[Dict] = []
        self.topology_history: List[nx.Graph] = []
        self.node_features_history: List[np.ndarray] = []

    @classmethod
    def from_saved_logs(
        cls,
        state_log: Sequence[Dict[str, Any]],
        communication_log: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        window_size: int = 20,
        window_stride: int = 5,
        max_history: int = 100,
    ) -> "TemporalGraphDataset":
        """Build a temporal dataset from saved simulation logs."""
        constructor = cls(window_size=window_size, window_stride=window_stride, max_history=max_history)
        graphs = constructor.construct_from_state_log(state_log, communication_log=communication_log)
        return TemporalGraphDataset(graphs, window_size=window_size, window_stride=window_stride)

    @classmethod
    def from_simulation_engine(
        cls,
        simulation_engine: Any,
        window_size: int = 20,
        window_stride: int = 5,
        max_history: int = 100,
    ) -> "TemporalGraphDataset":
        """Build a temporal dataset directly from a simulation engine instance."""
        state_log = getattr(simulation_engine, "state_log", [])
        communication_log = None

        communication_manager = getattr(simulation_engine, "communication_manager", None)
        if communication_manager is not None:
            communication_log = {
                "messages": [message.to_dict() for message in communication_manager.broker.comm_log.messages],
                "transmissions": list(communication_manager.broker.comm_log.transmission_events),
            }

        return cls.from_saved_logs(
            state_log=state_log,
            communication_log=communication_log,
            window_size=window_size,
            window_stride=window_stride,
            max_history=max_history,
        )

    def construct_from_state_log(
        self,
        state_log: Sequence[Dict[str, Any]],
        communication_log: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    ) -> List[Data]:
        """Construct a list of PyG graph snapshots from saved timestep logs."""
        graphs: List[Data] = []
        for state in state_log:
            graphs.append(self.construct_snapshot_from_state(state, communication_log=communication_log))
        return graphs
        
    def construct_temporal_snapshot(self, timestep: int, environment,
                                   node_features: Optional[np.ndarray] = None,
                                   edge_features: Optional[Dict] = None) -> Data:
        """
        Construct a temporal graph snapshot at current timestep.
        
        Args:
            timestep: Current timestep
            environment: Simulation environment
            node_features: Pre-computed node features (N, F_node)
            edge_features: Pre-computed edge features
            
        Returns:
            PyTorch Geometric Data object
        """
        # Build edge index from communication graph
        G = environment.communication_graph
        edge_index = torch.tensor(
            list(G.edges()), dtype=torch.long
        ).t().contiguous()
        
        if edge_index.numel() == 0:
            # No edges - create empty edge_index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Extract node features
        if node_features is None:
            node_features = self._extract_node_features(environment)
        
        x = torch.from_numpy(node_features).float()
        
        # Extract edge features
        if edge_features is None:
            edge_features = self._extract_edge_features(environment, G)
        
        edge_attr = torch.from_numpy(edge_features).float() if edge_features is not None else None
        
        # Node labels (if available)
        y = torch.tensor(
            [1 if i in getattr(environment, '_adversary_ids', []) else 0
             for i in range(environment.num_robots)],
            dtype=torch.long
        )
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=environment.num_robots,
            timestep=timestep,
            scenario_id=str(getattr(environment, "scenario_id", "unknown")),
        )
        
        return data

    def construct_snapshot_from_state(
        self,
        state: Dict[str, Any],
        communication_log: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        graph_id: Optional[str] = None,
    ) -> Data:
        """Construct a snapshot from a serialized state entry without a live environment."""
        robot_ids = sorted(state.get("robots", {}).keys())
        num_nodes = len(robot_ids)
        timestep = int(state.get("timestep", 0))

        node_features = self._extract_node_features_from_state(state, robot_ids)
        edge_index, edge_attr, edge_labels = self._extract_edges_from_logs(
            state=state,
            robot_ids=robot_ids,
            timestep=timestep,
            communication_log=communication_log,
        )

        node_labels = self._extract_node_labels_from_state(state, robot_ids)

        data = Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels,
            num_nodes=num_nodes,
            timestep=timestep,
            scenario_id=str(state.get("scenario_id", "unknown")),
        )
        data.edge_label = edge_labels
        data.graph_id = graph_id if graph_id is not None else f"timestep_{timestep}"
        data.node_ids = torch.tensor(robot_ids, dtype=torch.long)
        data.num_edges = int(edge_index.shape[1])
        return data
    
    def construct_temporal_sequence(self, start_timestep: int, environment,
                                   sequence_length: Optional[int] = None) -> List[Data]:
        """
        Construct sequence of temporal snapshots.
        
        Args:
            start_timestep: Starting timestep
            environment: Simulation environment
            sequence_length: Number of snapshots (default: window_size)
            
        Returns:
            List of PyG Data objects over time
        """
        if sequence_length is None:
            sequence_length = self.window_size
        
        # Get historical data from environment
        graphs = []
        
        # For now, create synthetic temporal sequence
        # In practice, retrieve from logged environment states
        for t in range(start_timestep, start_timestep + sequence_length):
            # This would retrieve actual data from environment history
            graph = self.construct_temporal_snapshot(t, environment)
            graphs.append(graph)
        
        return graphs

    def construct_temporal_windows(
        self,
        graphs: Sequence[Data],
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> List[Batch]:
        """Create batched temporal windows for efficient sequence learning."""
        window_size = window_size or self.window_size
        stride = stride or self.window_stride
        windows: List[Batch] = []

        if window_size <= 0:
            raise ValueError("window_size must be positive")

        for start in range(0, max(len(graphs) - window_size + 1, 0), stride):
            sequence = list(graphs[start : start + window_size])
            if len(sequence) == window_size:
                windows.append(Batch.from_data_list(sequence))

        return windows
    
    def _extract_node_features(self, environment) -> np.ndarray:
        """
        Extract multi-modal node features for all robots.
        
        Features are restricted to values observable by a robot at inference time:
        position, velocity, heading, network degree, and local message counts.
        """
        num_robots = environment.num_robots
        features = []
        
        for robot_id in range(num_robots):
            robot = environment.robots[robot_id]
            
            # Position & velocity (4D)
            pos_vel = np.concatenate([robot.position, robot.velocity])
            
            # Heading (1D)
            heading = np.array([robot.heading])
            
            # Network degree (1D)
            degree = np.array([environment.communication_graph.degree(robot_id)])
            
            # Statistics from message history (2D)
            received_count = len(robot.message_queue)
            sent_count = len(robot.outgoing_messages)
            msg_stats = np.array([received_count, sent_count])
            
            # Concatenate all features
            node_feature = np.concatenate([
                pos_vel, heading, degree, msg_stats
            ])
            
            features.append(node_feature)
        
        return np.stack(features)  # (num_robots, F_node)

    def _extract_node_features_from_state(
        self,
        state: Dict[str, Any],
        robot_ids: Sequence[int],
    ) -> np.ndarray:
        """Extract node features from a saved state dictionary."""
        features: List[np.ndarray] = []
        robots = state.get("robots", {})

        for robot_id in robot_ids:
            robot_state = robots[robot_id]
            position = _to_numpy(robot_state.get("position", np.zeros(2)))
            velocity = _to_numpy(robot_state.get("velocity", np.zeros(2)))
            heading = np.asarray([robot_state.get("heading", 0.0)], dtype=np.float32)
            energy = np.asarray([robot_state.get("energy", 0.0) / 100.0], dtype=np.float32)

            neighbor_count = np.asarray([robot_state.get("num_neighbors", 0)], dtype=np.float32)

            feature = np.concatenate([
                position.astype(np.float32),
                velocity.astype(np.float32),
                heading,
                energy,
                neighbor_count,
            ])
            features.append(feature)

        return np.stack(features, axis=0).astype(np.float32)

    def _extract_node_labels_from_state(
        self,
        state: Dict[str, Any],
        robot_ids: Sequence[int],
    ) -> torch.Tensor:
        adversaries = set(state.get("adversary_ids", state.get("_adversary_ids", [])))
        labels = [1 if robot_id in adversaries else 0 for robot_id in robot_ids]
        return torch.tensor(labels, dtype=torch.long)
    
    def _extract_edge_features(self, environment, G: nx.Graph) -> Optional[np.ndarray]:
        """
        Extract edge features for all edges.
        
        Features per edge:
        - Signal strength (1D)
        - Distance (1D)
        - Message correlation on that edge (1D)
        - Temporal consistency (1D)
        
        Total: ~4D per edge
        """
        edges = list(G.edges(data=True))
        
        if not edges:
            return np.array([]).reshape(0, 4)
        
        edge_features = []
        
        for u, v, data in edges:
            signal_strength = np.array([data.get('weight', 0.5)])
            distance = np.array([data.get('distance', 50.0)])
            
            # Message correlation (simplified - count messages)
            msg_correlation = np.array([
                len([m for m in environment.robots[u].message_queue
                     if m.get('sender_id') == v])
            ])
            
            # Temporal consistency (1.0 if edge exists in history)
            temporal_consistency = np.array([1.0])  # Default
            
            edge_feature = np.concatenate([
                signal_strength, distance, msg_correlation, temporal_consistency
            ])
            
            edge_features.append(edge_feature)
        
        return np.stack(edge_features)  # (num_edges, F_edge)

    def _extract_edges_from_logs(
        self,
        state: Dict[str, Any],
        robot_ids: Sequence[int],
        timestep: int = 0,
        communication_log: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract edge index and features from state/network logs."""
        robot_index = {robot_id: idx for idx, robot_id in enumerate(robot_ids)}
        state_network = state.get("network", {})
        edges: List[Tuple[int, int]] = []
        edge_features: List[np.ndarray] = []
        edge_labels: List[int] = []

        graph_edges = state.get("edges") or state_network.get("edges") or []
        for edge in graph_edges:
            source = edge.get("source") if isinstance(edge, dict) else edge[0]
            target = edge.get("target") if isinstance(edge, dict) else edge[1]
            if source not in robot_index or target not in robot_index:
                continue

            edges.append((robot_index[source], robot_index[target]))
            edge_features.append(self._build_edge_feature_from_entry(edge, communication_log, source, target, state))
            edge_labels.append(int(edge.get("label", 0)) if isinstance(edge, dict) else 0)

        if not edges and communication_log:
            transmissions = list(communication_log.get("transmissions", []))
            messages = list(communication_log.get("messages", []))

            for transmission in transmissions:
                event_time = int(round(float(transmission.get("timestamp", transmission.get("received_at", timestep)))))
                if event_time != timestep:
                    continue

                source = transmission.get("sender_id")
                target = transmission.get("receiver_id")
                if source not in robot_index or target not in robot_index:
                    continue

                edges.append((robot_index[source], robot_index[target]))
                edge_features.append(self._build_edge_feature_from_entry(transmission, communication_log, source, target, state))
                edge_labels.append(1 if not transmission.get("success", True) or transmission.get("malicious", False) else 0)

            if not edges:
                for message in messages:
                    event_time = int(round(float(message.get("timestamp", timestep))))
                    if event_time != timestep:
                        continue

                    source = message.get("sender_id")
                    target = message.get("receiver_id")
                    if source not in robot_index or target not in robot_index:
                        continue

                    edges.append((robot_index[source], robot_index[target]))
                    edge_features.append(self._build_edge_feature_from_entry(message, communication_log, source, target, state))
                    edge_labels.append(1 if message.get("malicious", False) else 0)

        if not edges:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 6), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.stack(edge_features).astype(np.float32))
        edge_label_tensor = torch.tensor(edge_labels, dtype=torch.long)
        return edge_index, edge_attr, edge_label_tensor

    def _build_edge_feature_from_entry(
        self,
        edge: Any,
        communication_log: Optional[Dict[str, Sequence[Dict[str, Any]]]],
        source: int,
        target: int,
        state: Dict[str, Any],
    ) -> np.ndarray:
        """Build edge features from either a topology edge or communication logs."""
        if isinstance(edge, dict):
            signal_strength = float(edge.get("weight", edge.get("signal_strength", 0.0)))
            distance = float(edge.get("distance", 0.0))
            packet_integrity = float(edge.get("packet_integrity", edge.get("integrity", 1.0)))
            latency = float(edge.get("latency", edge.get("transmission_delay", 0.0)))
        else:
            signal_strength = 0.0
            distance = 0.0
            packet_integrity = 1.0
            latency = 0.0

        message_count = 0.0
        avg_latency = 0.0
        avg_integrity = 0.0
        if communication_log:
            relevant = []
            for message in communication_log.get("transmissions", []):
                if message.get("sender_id") == source and message.get("receiver_id") == target:
                    relevant.append(message)
            if not relevant:
                for message in communication_log.get("messages", []):
                    if message.get("sender_id") == source and message.get("receiver_id") == target:
                        relevant.append(message)

            if relevant:
                message_count = float(len(relevant))
                latencies = [float(message.get("latency", 0.0)) for message in relevant]
                avg_latency = float(np.mean(latencies))
                avg_integrity = float(np.mean([float(message.get("packet_integrity", 1.0)) for message in relevant]))

        return np.asarray([
            signal_strength,
            distance,
            message_count,
            avg_latency,
            avg_integrity,
            float(packet_integrity < 1.0) + float(latency > 0.0),
        ], dtype=np.float32)
    
    def add_temporal_context(self, data: Data, timestep: int,
                           temporal_embedding: Optional[np.ndarray] = None) -> Data:
        """
        Add temporal context information to graph.
        
        Args:
            data: PyG Data object
            timestep: Current timestep
            temporal_embedding: Pre-computed temporal embeddings
            
        Returns:
            Updated Data object with temporal information
        """
        if temporal_embedding is None:
            # Create simple temporal encoding
            # Frequency-based: sin(2π k t / T) for different frequency components
            temporal_embedding = self._create_temporal_embedding(timestep)
        
        data.timestep = timestep
        data.temporal_embedding = torch.from_numpy(temporal_embedding).float()
        
        return data
    
    def _create_temporal_embedding(self, timestep: int, embedding_dim: int = 16,
                                  period: float = 100.0) -> np.ndarray:
        """
        Create sinusoidal temporal embedding.
        
        Uses Fourier features: [sin(2π t), sin(4π t), ..., cos(2π t), ...]
        """
        embeddings = []
        
        for k in range(1, embedding_dim // 2 + 1):
            freq = 2 * np.pi * k / period
            embeddings.append(np.sin(freq * timestep))
            embeddings.append(np.cos(freq * timestep))
        
        return np.array(embeddings[:embedding_dim])


@dataclass
class TemporalGraphWindow:
    """A contiguous window of temporal graphs and its metadata."""

    graphs: List[Data]
    start_timestep: int
    end_timestep: int


class TemporalGraphDataset(Dataset):
    """PyTorch Geometric dataset of temporal graph snapshots and windows."""

    def __init__(
        self,
        graphs: Sequence[Data],
        window_size: int = 20,
        window_stride: int = 5,
    ):
        super().__init__()
        self.graphs = list(graphs)
        self.window_size = window_size
        self.window_stride = window_stride
        self._windows = self._build_windows()

    def _build_windows(self) -> List[TemporalGraphWindow]:
        windows: List[TemporalGraphWindow] = []
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

        for start in range(0, max(len(self.graphs) - self.window_size + 1, 0), self.window_stride):
            window_graphs = self.graphs[start : start + self.window_size]
            if len(window_graphs) == self.window_size:
                windows.append(
                    TemporalGraphWindow(
                        graphs=window_graphs,
                        start_timestep=int(getattr(window_graphs[0], "timestep", start)),
                        end_timestep=int(getattr(window_graphs[-1], "timestep", start + self.window_size - 1)),
                    )
                )
        return windows

    def len(self) -> int:
        return len(self._windows)

    def get(self, idx: int) -> Batch:
        window = self._windows[idx]
        batch = Batch.from_data_list(window.graphs)
        batch.window_start = window.start_timestep
        batch.window_end = window.end_timestep
        batch.sequence_length = len(window.graphs)
        return batch

    def get_temporal_sequence(self, idx: int) -> List[Data]:
        """Return the raw sequence of graphs for a temporal window."""
        return list(self._windows[idx].graphs)

    def to_batches(self) -> List[Batch]:
        """Materialize the dataset as a list of batched temporal windows."""
        return [self.get(idx) for idx in range(self.len())]
