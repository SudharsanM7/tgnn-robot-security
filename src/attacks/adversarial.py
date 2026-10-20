"""
Adversarial module: Base adversarial attack interface and implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class AttackType(Enum):
    """Types of adversarial attacks"""
    MESSAGE_INJECTION = "message_injection"
    MESSAGE_REPLAY = "message_replay"
    COORDINATE_SPOOFING = "coordinate_spoofing"
    BEACON_JAMMING = "beacon_jamming"
    SYBIL_ATTACK = "sybil_attack"
    COLLUDING_ATTACK = "colluding_attack"


class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks.
    
    All attacks must implement adaptive mechanisms that evolve
    based on detection likelihood feedback.
    """
    
    def __init__(self, attacker_id: int, target_ids: Optional[List[int]] = None):
        """
        Initialize attack.
        
        Args:
            attacker_id: ID of the attacking robot
            target_ids: IDs of target robots (None = all others)
        """
        self.attacker_id = attacker_id
        self.target_ids = target_ids
        self.attack_type: Optional[AttackType] = None
        self.success_rate = 0.5  # Initial attack success probability
        self.detection_feedback = 0.0  # Feedback from detection system
        self.adaptation_rate = 0.001  # How quickly attack adapts
        self.attack_history: List[Dict] = []
        
    @abstractmethod
    def execute(self, timestep: int, environment) -> Dict:
        """
        Execute the attack on the environment.
        
        Args:
            timestep: Current simulation timestep
            environment: Reference to the environment
            
        Returns:
            Dictionary containing attack details and effects
        """
        pass
    
    def adapt(self, detection_feedback: float):
        """
        Adapt attack strategy based on detection frequency.
        
        Args:
            detection_feedback: Probability of being detected (0-1)
        """
        self.detection_feedback = detection_feedback
        
        # Decrease success rate if often detected
        if detection_feedback > 0.5:
            self.success_rate *= (1 - self.adaptation_rate)
        else:
            self.success_rate *= (1 + self.adaptation_rate * 0.5)
        
        # Clamp to valid range
        self.success_rate = np.clip(self.success_rate, 0.1, 0.95)


class MessageInjectionAttack(AdversarialAttack):
    """
    Attack that injects false/malicious messages into the network.
    
    This attack creates synthetic messages that appear to come from
    legitimate robots, potentially disrupting coordination or spreading
    misinformation.
    """
    
    def __init__(self, attacker_id: int, target_ids: Optional[List[int]] = None):
        super().__init__(attacker_id, target_ids)
        self.attack_type = AttackType.MESSAGE_INJECTION
        self.injected_count = 0
        
    def execute(self, timestep: int, environment) -> Dict:
        """Execute message injection attack."""
        if np.random.random() > self.success_rate:
            return {'success': False}
        
        # Generate false message mimicking a legitimate robot
        victim_id = np.random.randint(0, environment.num_robots)
        if victim_id == self.attacker_id:
            victim_id = (victim_id + 1) % environment.num_robots
        
        # Create malicious message
        fake_position = np.random.uniform(-100, 100, 2)
        fake_velocity = np.random.uniform(-1, 1, 2)
        
        malicious_msg = {
            'type': 'position_update',
            'sender_id': victim_id,
            'position': fake_position,
            'velocity': fake_velocity,
            'malicious': True,
            'attack_type': self.attack_type.value
        }
        
        # Broadcast to neighbors
        for neighbor_id in environment.communication_graph.neighbors(self.attacker_id):
            environment.robots[neighbor_id].add_incoming_message(malicious_msg)
        
        self.injected_count += 1
        
        return {
            'success': True,
            'attack_type': self.attack_type.value,
            'victim_id': victim_id,
            'num_targets': len(list(environment.communication_graph.neighbors(self.attacker_id)))
        }


class MessageReplayAttack(AdversarialAttack):
    """
    Attack that replays previously captured legitimate messages.
    
    This can create ghost positions or cause state inconsistency in the network.
    """
    
    def __init__(self, attacker_id: int, target_ids: Optional[List[int]] = None):
        super().__init__(attacker_id, target_ids)
        self.attack_type = AttackType.MESSAGE_REPLAY
        self.captured_messages: List[Dict] = []
        
    def capture_message(self, msg: Dict):
        """Capture a message for later replay."""
        self.captured_messages.append(msg.copy())
        
    def execute(self, timestep: int, environment) -> Dict:
        """Execute message replay attack."""
        if not self.captured_messages or np.random.random() > self.success_rate:
            return {'success': False}
        
        # Select random captured message
        replayed_msg = np.random.choice(self.captured_messages).copy()
        replayed_msg['replayed_at'] = timestep
        replayed_msg['malicious'] = True
        replayed_msg['attack_type'] = self.attack_type.value
        
        # Send to all neighbors
        for neighbor_id in environment.communication_graph.neighbors(self.attacker_id):
            environment.robots[neighbor_id].add_incoming_message(replayed_msg)
        
        return {
            'success': True,
            'attack_type': self.attack_type.value,
            'original_timestamp': replayed_msg.get('timestamp'),
            'replayed_at': timestep
        }


class CoordinateSpoofingAttack(AdversarialAttack):
    """
    Attack that corrupts coordinate/position information.
    
    This causes robots to have incorrect spatial awareness of the network,
    disrupting formation control and coordination.
    """
    
    def __init__(self, attacker_id: int, target_ids: Optional[List[int]] = None):
        super().__init__(attacker_id, target_ids)
        self.attack_type = AttackType.COORDINATE_SPOOFING
        self.spoofed_positions: Dict[int, np.ndarray] = {}
        
    def execute(self, timestep: int, environment) -> Dict:
        """Execute coordinate spoofing attack."""
        if np.random.random() > self.success_rate:
            return {'success': False}
        
        # Identify targets
        neighbors = list(environment.communication_graph.neighbors(self.attacker_id))
        num_targets = max(1, int(len(neighbors) * 0.3))  # Target 30% of neighbors
        
        targets = np.random.choice(neighbors, size=min(num_targets, len(neighbors)), 
                                  replace=False) if neighbors else []
        
        # Spoof position of attacker itself
        spoofed_position = environment.robots[self.attacker_id].position + np.random.uniform(-20, 20, 2)
        
        spoof_msg = {
            'type': 'position_spoof',
            'sender_id': self.attacker_id,
            'position': spoofed_position,
            'malicious': True,
            'attack_type': self.attack_type.value
        }
        
        for target_id in targets:
            environment.robots[target_id].add_incoming_message(spoof_msg)
        
        return {
            'success': True,
            'attack_type': self.attack_type.value,
            'num_targets': len(targets),
            'magnitude': np.linalg.norm(spoofed_position - environment.robots[self.attacker_id].position)
        }
