"""
Communication Module: Realistic network simulation with message passing

This module provides:
- Network simulation with transmission delays, packet loss, signal decay
- Asynchronous message broker for routing
- Bandwidth constraints and queuing
- Communication event logging
- Integration with multi-robot simulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict
from loguru import logger


class MessageType(Enum):
    """Types of messages in the network"""
    POSITION_UPDATE = "position_update"
    HEARTBEAT = "heartbeat"
    SENSOR_DATA = "sensor_data"
    TRUST_QUERY = "trust_query"
    ACKNOWLEDGMENT = "ack"


@dataclass
class CommunicationMessage:
    """
    Enhanced message with network characteristics.
    
    Attributes:
        msg_id: Unique message identifier
        sender_id: Originating robot
        receiver_id: Target robot (None = broadcast)
        msg_type: Type of message
        timestamp: When message was created (seconds)
        payload: Message content (Dict or numpy array)
        payload_size: Size of payload in bytes
        hop_count: Number of hops made
        is_signed: Whether message has signature
        signature: Cryptographic signature for verification
        latency: Transmission latency in seconds
        transmission_delay: Delay before delivery
        packet_integrity: Probability of successful transmission (0-1)
        signal_strength: Quality of signal (0-1)
        received_at: Reception timestamp (None if not yet delivered)
        delivery_status: 'pending', 'delivered', 'lost', 'corrupted'
    """
    msg_id: int
    sender_id: int
    msg_type: MessageType
    timestamp: float
    payload: Dict[str, Any]
    receiver_id: Optional[int] = None  # None = broadcast
    hop_count: int = 0
    is_signed: bool = False
    signature: str = ""
    payload_size: int = 0  # Bytes
    latency: float = 0.0  # Transmission latency
    transmission_delay: float = 0.0  # Delay before delivery
    packet_integrity: float = 1.0  # Success probability (0-1)
    signal_strength: float = 1.0  # Signal quality (0-1)
    received_at: Optional[float] = None  # When message arrived
    delivery_status: str = "pending"  # pending, delivered, lost, corrupted
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/transmission."""
        return {
            'msg_id': self.msg_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'msg_type': self.msg_type.value,
            'timestamp': self.timestamp,
            'payload': self.payload if isinstance(self.payload, dict) else {},
            'hop_count': self.hop_count,
            'is_signed': self.is_signed,
            'signature': self.signature,
            'payload_size': self.payload_size,
            'latency': self.latency,
            'transmission_delay': self.transmission_delay,
            'packet_integrity': self.packet_integrity,
            'signal_strength': self.signal_strength,
            'received_at': self.received_at,
            'delivery_status': self.delivery_status
        }
    
    @property
    def is_delivered(self) -> bool:
        """Check if message was successfully delivered."""
        return self.delivery_status == 'delivered'
    
    @property
    def is_lost(self) -> bool:
        """Check if message was lost."""
        return self.delivery_status == 'lost'



class NetworkSimulator:
    """
    Simulates realistic network effects on message transmission.
    
    Features:
    - Transmission delay based on distance and payload size
    - Packet loss probability
    - Signal strength decay (inverse square law)
    - Bandwidth constraints
    - Channel interference
    """
    
    def __init__(self, 
                 communication_range: float = 50.0,
                 base_transmission_delay: float = 0.001,
                 packet_loss_rate: float = 0.01,
                 bandwidth: float = 1000.0,  # KB/s
                 signal_decay_factor: float = 2.0):
        """
        Initialize network simulator.
        
        Args:
            communication_range: Maximum communication distance
            base_transmission_delay: Base delay in seconds
            packet_loss_rate: Probability of packet loss (0-1)
            bandwidth: Channel bandwidth in KB/s
            signal_decay_factor: Decay exponent for signal (typically 2-4)
        """
        self.communication_range = communication_range
        self.base_transmission_delay = base_transmission_delay
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth = bandwidth
        self.signal_decay_factor = signal_decay_factor
    
    def compute_signal_strength(self, distance: float) -> float:
        """
        Compute signal strength based on distance (inverse square law).
        
        Typical model: P_r = P_t * (lambda / 4*pi*d)^n
        where n is path loss exponent (typically 2-4).
        
        Returns:
            Signal strength (0-1), 1 = perfect signal
        """
        if distance <= 0:
            return 1.0
        if distance >= self.communication_range:
            return 0.0
        
        # Inverse power law with decay factor
        normalized_distance = distance / self.communication_range
        signal = (1.0 - normalized_distance) ** self.signal_decay_factor
        return float(np.clip(signal, 0.0, 1.0))
    
    def compute_transmission_delay(self, distance: float, 
                                  payload_size: int) -> float:
        """
        Compute transmission delay based on distance and payload.
        
        Args:
            distance: Distance between sender and receiver (meters)
            payload_size: Message size (bytes)
            
        Returns:
            Transmission delay (seconds)
        """
        # Distance component: speed of light ~ 3e8 m/s
        speed_of_light = 3e8
        propagation_delay = distance / speed_of_light
        
        # Bandwidth component
        bandwidth_delay = payload_size / (self.bandwidth * 1024)  # Convert KB/s to bytes/s
        
        # Total delay
        total_delay = self.base_transmission_delay + propagation_delay + bandwidth_delay
        return float(total_delay)
    
    def simulate_transmission(self, distance: float, payload_size: int
                             ) -> Tuple[float, float, bool]:
        """
        Simulate message transmission through network.
        
        Args:
            distance: Distance between sender and receiver
            payload_size: Message payload size (bytes)
            
        Returns:
            Tuple: (transmission_delay, signal_strength, packet_received)
        """
        # Check if within range
        if distance > self.communication_range:
            return 0.0, 0.0, False
        
        # Compute signal strength
        signal_strength = self.compute_signal_strength(distance)
        
        # Compute transmission delay
        transmission_delay = self.compute_transmission_delay(distance, payload_size)
        
        # Check packet loss (worse signal = higher loss)
        packet_loss_prob = self.packet_loss_rate * (1.0 - signal_strength)
        packet_received = np.random.random() > packet_loss_prob
        
        return transmission_delay, signal_strength, packet_received


class BandwidthManager:
    """
    Manages bandwidth constraints and message queuing.
    
    Tracks:
    - Outgoing bandwidth per robot
    - Incoming bandwidth per robot
    - Message queues and scheduling
    """
    
    def __init__(self, total_bandwidth: float = 1000.0):
        """
        Initialize bandwidth manager.
        
        Args:
            total_bandwidth: Total channel bandwidth (KB/s)
        """
        self.total_bandwidth = total_bandwidth
        self.robot_bandwidth_used: Dict[int, float] = defaultdict(float)
        self.message_queue: Dict[int, List[CommunicationMessage]] = defaultdict(list)
        self.transmission_log: List[Dict] = []
    
    def can_transmit(self, robot_id: int, message_size: int, 
                    current_time: float) -> bool:
        """
        Check if robot can transmit given bandwidth constraints.
        
        Args:
            robot_id: Robot identifier
            message_size: Size of message (bytes)
            current_time: Current simulation time
            
        Returns:
            True if bandwidth available, False otherwise
        """
        # Convert to KB
        message_size_kb = message_size / 1024.0
        
        # Check if robot has available bandwidth
        used = self.robot_bandwidth_used.get(robot_id, 0.0)
        if used + message_size_kb <= self.total_bandwidth:
            return True
        
        return False
    
    def queue_message(self, message: CommunicationMessage, robot_id: int):
        """Queue message for transmission."""
        self.message_queue[robot_id].append(message)
    
    def get_queued_messages(self, robot_id: int) -> List[CommunicationMessage]:
        """Get all queued messages for a robot."""
        messages = self.message_queue.get(robot_id, [])
        self.message_queue[robot_id] = []  # Clear queue
        return messages


class CommunicationLog:
    """
    Logs all communication events for analysis and debugging.
    
    Tracks:
    - Message transmission events
    - Message delivery status
    - Network statistics (latency, loss, throughput)
    - Communication graph (who talks to whom)
    """
    
    def __init__(self):
        """Initialize communication logger."""
        self.messages: List[CommunicationMessage] = []
        self.transmission_events: List[Dict] = []
        self.delivery_stats: Dict[str, int] = {
            'delivered': 0,
            'lost': 0,
            'corrupted': 0,
            'pending': 0
        }
        self.per_robot_stats: Dict[int, Dict] = defaultdict(lambda: {
            'sent': 0,
            'received': 0,
            'lost': 0,
            'avg_latency': 0.0
        })
        self.communication_pairs: Dict[Tuple[int, int], int] = defaultdict(int)
    
    def log_message(self, message: CommunicationMessage):
        """Log a message."""
        self.messages.append(message)
        
        # Update delivery stats
        self.delivery_stats[message.delivery_status] += 1
        
        # Update per-robot stats
        self.per_robot_stats[message.sender_id]['sent'] += 1
        
        if message.receiver_id is not None:
            self.per_robot_stats[message.receiver_id]['received'] += 1
            
            if message.is_delivered:
                self.communication_pairs[(message.sender_id, message.receiver_id)] += 1
            
            if message.is_lost:
                self.per_robot_stats[message.receiver_id]['lost'] += 1
    
    def log_transmission(self, sender_id: int, receiver_id: int, 
                        distance: float, signal_strength: float,
                        success: bool, timestamp: float):
        """Log transmission event."""
        event = {
            'timestamp': timestamp,
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'distance': distance,
            'signal_strength': signal_strength,
            'success': success
        }
        self.transmission_events.append(event)
    
    def get_statistics(self) -> Dict:
        """Get communication statistics."""
        total_messages = sum(self.delivery_stats.values())
        
        # Calculate average latency
        latencies = [m.latency for m in self.messages if m.latency > 0]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        
        # Calculate delivery ratio
        total_sent = sum(s['sent'] for s in self.per_robot_stats.values())
        total_delivered = self.delivery_stats['delivered']
        delivery_ratio = total_delivered / max(total_sent, 1)
        
        return {
            'total_messages': total_messages,
            'delivery_stats': dict(self.delivery_stats),
            'average_latency': avg_latency,
            'delivery_ratio': delivery_ratio,
            'total_transmission_events': len(self.transmission_events),
            'per_robot_stats': dict(self.per_robot_stats),
            'communication_pairs': len(self.communication_pairs)
        }
    
    def get_communication_graph(self) -> Dict[int, List[int]]:
        """Get communication graph (adjacency list)."""
        graph: Dict[int, set] = defaultdict(set)
        for (sender, receiver), count in self.communication_pairs.items():
            if count > 0:
                graph[sender].add(receiver)
        
        return {k: list(v) for k, v in graph.items()}


class MessageBroker:
    """
    Central message routing and delivery system.
    
    Responsibilities:
    - Route messages between robots
    - Handle asynchronous delivery
    - Manage message queues
    - Simulate network effects
    - Track message status
    """
    
    def __init__(self, 
                 communication_range: float = 50.0,
                 base_transmission_delay: float = 0.001,
                 packet_loss_rate: float = 0.01):
        """Initialize message broker."""
        self.network_sim = NetworkSimulator(
            communication_range=communication_range,
            base_transmission_delay=base_transmission_delay,
            packet_loss_rate=packet_loss_rate
        )
        self.bandwidth_mgr = BandwidthManager()
        self.comm_log = CommunicationLog()
        
        # Message queues
        self.pending_messages: List[CommunicationMessage] = []
        self.delivery_queue: List[Tuple[CommunicationMessage, float]] = []  # (msg, delivery_time)
        self.message_counter = 0
    
    def send_message(self, sender_id: int, message_type: MessageType,
                    payload: Dict[str, Any], receiver_id: Optional[int] = None,
                    timestamp: float = 0.0) -> CommunicationMessage:
        """
        Create and queue a message for transmission.
        
        Args:
            sender_id: ID of sending robot
            message_type: Type of message
            payload: Message payload
            receiver_id: ID of receiving robot (None = broadcast)
            timestamp: Creation timestamp
            
        Returns:
            Created message
        """
        # Calculate payload size (rough estimate)
        payload_size = len(str(payload).encode('utf-8'))
        
        msg = CommunicationMessage(
            msg_id=self.message_counter,
            sender_id=sender_id,
            receiver_id=receiver_id,
            msg_type=message_type,
            timestamp=timestamp,
            payload=payload,
            payload_size=payload_size,
            delivery_status='pending'
        )
        
        self.message_counter += 1
        self.pending_messages.append(msg)
        
        return msg
    
    def deliver_messages(self, robot_positions: Dict[int, np.ndarray],
                        current_time: float) -> Dict[int, List[CommunicationMessage]]:
        """
        Process message delivery based on network conditions.
        
        Args:
            robot_positions: Dict of robot_id -> position
            current_time: Current simulation time
            
        Returns:
            Dict of robot_id -> received messages
        """
        received_by_robot: Dict[int, List[CommunicationMessage]] = defaultdict(list)
        
        # Process each pending message
        messages_to_remove = []
        
        for msg in self.pending_messages:
            sender_pos = robot_positions.get(msg.sender_id)
            
            if sender_pos is None:
                messages_to_remove.append(msg)
                continue
            
            # Handle broadcast or unicast
            if msg.receiver_id is None:
                # Broadcast to all within range
                for recv_id, recv_pos in robot_positions.items():
                    if recv_id != msg.sender_id:
                        self._attempt_delivery(msg, recv_id, sender_pos, recv_pos,
                                              current_time, received_by_robot)
            else:
                # Unicast to specific robot
                receiver_pos = robot_positions.get(msg.receiver_id)
                if receiver_pos is not None:
                    self._attempt_delivery(msg, msg.receiver_id, sender_pos, receiver_pos,
                                          current_time, received_by_robot)
            
            messages_to_remove.append(msg)
        
        # Remove processed messages
        for msg in messages_to_remove:
            if msg in self.pending_messages:
                self.pending_messages.remove(msg)
        
        # Log messages
        for msg in messages_to_remove:
            self.comm_log.log_message(msg)
        
        return received_by_robot
    
    def _attempt_delivery(self, message: CommunicationMessage, 
                         receiver_id: int, sender_pos: np.ndarray,
                         receiver_pos: np.ndarray, current_time: float,
                         received_by_robot: Dict[int, List[CommunicationMessage]]):
        """Attempt to deliver a message to a specific receiver."""
        # Compute distance
        distance = float(np.linalg.norm(receiver_pos - sender_pos))
        
        # Simulate transmission
        transmission_delay, signal_strength, packet_received = \
            self.network_sim.simulate_transmission(distance, message.payload_size)
        
        # Log transmission event
        self.comm_log.log_transmission(message.sender_id, receiver_id,
                                       distance, signal_strength,
                                       packet_received, current_time)
        
        # Update message
        message.latency = transmission_delay
        message.signal_strength = signal_strength
        message.transmission_delay = transmission_delay
        
        if packet_received:
            message.received_at = current_time + transmission_delay
            message.delivery_status = 'delivered'
            message.packet_integrity = signal_strength
            received_by_robot[receiver_id].append(message)
        else:
            message.delivery_status = 'lost'
            message.packet_integrity = 0.0
    
    def get_statistics(self) -> Dict:
        """Get communication statistics."""
        return self.comm_log.get_statistics()
    
    def get_communication_graph(self) -> Dict[int, List[int]]:
        """Get communication graph."""
        return self.comm_log.get_communication_graph()


class CommunicationProtocol:
    """
    Legacy protocol interface (maintained for compatibility).
    
    Manages communication protocol and message validation.
    """
    
    def __init__(self):
        self.message_counter = 0
        self.verified_senders: set = set()
    
    def create_position_update(self, robot_id: int, position: np.ndarray,
                               velocity: np.ndarray, timestamp: float) -> CommunicationMessage:
        """Create a position update message."""
        msg = CommunicationMessage(
            msg_id=self.message_counter,
            sender_id=robot_id,
            msg_type=MessageType.POSITION_UPDATE,
            timestamp=timestamp,
            payload={
                'position': position.tolist(),
                'velocity': velocity.tolist()
            },
            payload_size=len(str(position.tolist()).encode('utf-8'))
        )
        self.message_counter += 1
        return msg
    
    def create_heartbeat(self, robot_id: int, timestamp: float,
                        health_status: float = 1.0) -> CommunicationMessage:
        """Create a heartbeat/keep-alive message."""
        msg = CommunicationMessage(
            msg_id=self.message_counter,
            sender_id=robot_id,
            msg_type=MessageType.HEARTBEAT,
            timestamp=timestamp,
            payload={'health_status': health_status},
            payload_size=128
        )
        self.message_counter += 1
        return msg
    
    def create_sensor_data_message(self, robot_id: int, sensor_data: Dict,
                                   timestamp: float) -> CommunicationMessage:
        """Create a sensor data broadcast message."""
        msg = CommunicationMessage(
            msg_id=self.message_counter,
            sender_id=robot_id,
            msg_type=MessageType.SENSOR_DATA,
            timestamp=timestamp,
            payload={'sensor_data': sensor_data},
            payload_size=len(str(sensor_data).encode('utf-8'))
        )
        self.message_counter += 1
        return msg
    
    def validate_message(self, msg: CommunicationMessage) -> bool:
        """
        Validate message authenticity and integrity.
        
        In a real system, this would involve:
        - Cryptographic signature verification
        - Timestamp freshness checks
        - Sender reputation checks
        """
        # Simple heuristics for now
        if msg.hop_count > 10:  # Max hops
            return False
        
        if msg.is_signed and msg.signature:
            return True
        
        return True  # Accept unsigned for simulation purposes
    
    def sign_message(self, msg: CommunicationMessage, robot_id: int) -> CommunicationMessage:
        """Add signature to message."""
        # Simplified signing - in production use asymmetric crypto
        msg.is_signed = True
        msg.signature = f"sig_{robot_id}_{msg.msg_id}"
        return msg

