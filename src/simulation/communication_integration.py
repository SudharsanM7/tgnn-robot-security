"""
Communication Integration: Bridges communication system with SimulationEngine

This module provides integration helpers for incorporating message passing
into the multi-robot simulation.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass

from .communication import (
    MessageBroker, CommunicationLog, NetworkSimulator, MessageType
)


@dataclass
class CommunicationConfig:
    """Configuration for communication system."""
    
    enabled: bool = True
    communication_range: float = 50.0
    base_transmission_delay: float = 0.001  # seconds
    packet_loss_rate: float = 0.01  # 1% base loss
    bandwidth: float = 1000.0  # KB/s
    signal_decay_factor: float = 2.0
    enable_logging: bool = True
    log_frequency: int = 10  # Log every N timesteps


class CommunicationManager:
    """
    Manages communication for entire robot swarm.
    
    Handles:
    - Multi-robot message broadcasting
    - Asynchronous message delivery
    - Communication logging
    - Statistics and metrics
    """
    
    def __init__(self, num_robots: int, config: CommunicationConfig):
        """
        Initialize communication manager.
        
        Args:
            num_robots: Number of robots in simulation
            config: Communication configuration
        """
        self.num_robots = num_robots
        self.config = config
        
        # Message broker
        self.broker = MessageBroker(
            communication_range=config.communication_range,
            base_transmission_delay=config.base_transmission_delay,
            packet_loss_rate=config.packet_loss_rate
        )
        
        # Per-robot incoming message queues
        self.incoming_messages: Dict[int, List] = defaultdict(list)
        
        # Statistics
        self.communication_events: List[Dict] = []
        self.timestep_count = 0
    
    def send_position_update(self, robot_id: int, position: np.ndarray,
                            velocity: np.ndarray, timestamp: float,
                            broadcast: bool = True) -> int:
        """
        Send position update message.
        
        Args:
            robot_id: Sender robot ID
            position: Robot position
            velocity: Robot velocity
            timestamp: Message timestamp
            broadcast: If True, send to all; if False, send to neighbors only
            
        Returns:
            Message count sent
        """
        payload = {
            'position': position.tolist(),
            'velocity': velocity.tolist()
        }
        
        receiver_id = None if broadcast else -1  # -1 = neighbors only
        
        msg = self.broker.send_message(
            sender_id=robot_id,
            message_type=MessageType.POSITION_UPDATE,
            payload=payload,
            receiver_id=receiver_id,
            timestamp=timestamp
        )
        
        return 1
    
    def send_sensor_data(self, robot_id: int, sensor_data: Dict,
                        timestamp: float, broadcast: bool = True) -> int:
        """Send sensor data message."""
        msg = self.broker.send_message(
            sender_id=robot_id,
            message_type=MessageType.SENSOR_DATA,
            payload={'sensor_data': sensor_data},
            receiver_id=None if broadcast else -1,
            timestamp=timestamp
        )
        return 1
    
    def send_heartbeat(self, robot_id: int, timestamp: float,
                      health_status: float = 1.0) -> int:
        """Send heartbeat message."""
        msg = self.broker.send_message(
            sender_id=robot_id,
            message_type=MessageType.HEARTBEAT,
            payload={'health_status': health_status},
            receiver_id=None,  # Broadcast
            timestamp=timestamp
        )
        return 1
    
    def process_messages(self, robot_positions: Dict[int, np.ndarray],
                        current_time: float) -> Dict[int, List]:
        """
        Process all pending messages for delivery.
        
        Args:
            robot_positions: Dict of robot_id -> position
            current_time: Current simulation time
            
        Returns:
            Dict of robot_id -> received messages
        """
        if not self.config.enabled:
            return {}
        
        # Deliver pending messages
        received = self.broker.deliver_messages(robot_positions, current_time)
        
        # Store in incoming queues
        for robot_id, messages in received.items():
            self.incoming_messages[robot_id].extend(messages)
        
        # Record communication event
        if self.timestep_count % self.config.log_frequency == 0:
            event = {
                'timestep': self.timestep_count,
                'messages_pending': len(self.broker.pending_messages),
                'messages_delivered': len(received),
                'stats': self.broker.get_statistics()
            }
            self.communication_events.append(event)
        
        self.timestep_count += 1
        
        return received
    
    def get_incoming_messages(self, robot_id: int) -> List:
        """Get and clear incoming message queue for robot."""
        messages = self.incoming_messages.get(robot_id, [])
        self.incoming_messages[robot_id] = []
        return messages
    
    def get_statistics(self) -> Dict:
        """Get communication statistics."""
        return self.broker.get_statistics()
    
    def get_communication_graph(self) -> Dict[int, List[int]]:
        """Get communication topology graph."""
        return self.broker.get_communication_graph()
    
    def export_communication_log(self, filepath: str):
        """Export communication log to file."""
        import pickle
        
        log_data = {
            'scenario_id': str(getattr(self, 'scenario_id', 'unknown')),
            'messages': [m.to_dict() for m in self.broker.comm_log.messages],
            'transmission_events': self.broker.comm_log.transmission_events,
            'delivery_stats': self.broker.comm_log.delivery_stats,
            'per_robot_stats': dict(self.broker.comm_log.per_robot_stats),
            'communication_pairs': dict(self.broker.comm_log.communication_pairs),
            'communication_events': self.communication_events
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(log_data, f)
    
    def get_network_metrics(self) -> Dict:
        """Get detailed network metrics."""
        stats = self.broker.get_statistics()
        
        # Additional metrics
        total_distance = 0.0
        valid_events = self.broker.comm_log.transmission_events
        if valid_events:
            total_distance = sum(e['distance'] for e in valid_events)
        
        avg_signal = 0.0
        if valid_events:
            avg_signal = np.mean([e['signal_strength'] for e in valid_events])
        
        return {
            'total_messages': stats['total_messages'],
            'delivery_ratio': stats['delivery_ratio'],
            'average_latency': stats['average_latency'],
            'average_signal_strength': avg_signal,
            'communication_graph_size': stats['communication_pairs'],
            'total_transmission_events': stats['total_transmission_events'],
            'bandwidth_usage': self.broker.bandwidth_mgr.total_bandwidth
        }


class NetworkAnalyzer:
    """
    Analyzes communication network topology and performance.
    """
    
    def __init__(self, communication_log: CommunicationLog):
        """
        Initialize network analyzer.
        
        Args:
            communication_log: CommunicationLog instance to analyze
        """
        self.log = communication_log
    
    def get_network_diameter(self) -> int:
        """Get network diameter (max shortest path length)."""
        graph = self.log.get_communication_graph()
        
        if not graph:
            return 0
        
        try:
            import networkx as nx
            G = nx.Graph()
            
            for source, targets in graph.items():
                for target in targets:
                    G.add_edge(source, target)
            
            if nx.is_connected(G):
                return nx.diameter(G)
            else:
                # Return diameter of largest component
                cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(cc)
                return nx.diameter(subgraph)
        except ImportError:
            return -1
    
    def get_centrality_measures(self) -> Dict:
        """Get node centrality measures."""
        graph = self.log.get_communication_graph()
        
        if not graph:
            return {}
        
        try:
            import networkx as nx
            G = nx.Graph()
            
            for source, targets in graph.items():
                for target in targets:
                    G.add_edge(source, target)
            
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            return {
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality
            }
        except ImportError:
            return {}
    
    def get_connectivity_stats(self) -> Dict:
        """Get network connectivity statistics."""
        messages = self.log.messages
        
        # Count message types
        type_counts = defaultdict(int)
        for msg in messages:
            type_counts[msg.msg_type.value] += 1
        
        # Count delivery status
        status_counts = defaultdict(int)
        for msg in messages:
            status_counts[msg.delivery_status] += 1
        
        return {
            'message_types': dict(type_counts),
            'delivery_status': dict(status_counts),
            'total_unique_pairs': len(self.log.communication_pairs),
            'per_robot_stats': dict(self.log.per_robot_stats)
        }
