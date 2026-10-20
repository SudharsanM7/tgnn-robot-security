"""
Adaptation module: Adaptive attack evolution mechanisms
"""

from typing import Dict, List, Optional
import numpy as np


class AdaptationMechanism:
    """
    Handles adaptive evolution of attack strategies based on:
    - Detection likelihood
    - Network topology changes
    - Defender responses
    """
    
    def __init__(self, adaptation_rate: float = 0.001, memory_window: int = 50):
        """
        Initialize adaptation mechanism.
        
        Args:
            adaptation_rate: Rate at which attacks evolve
            memory_window: History window for computing statistics
        """
        self.adaptation_rate = adaptation_rate
        self.memory_window = memory_window
        self.detection_history: List[float] = []
        self.success_history: List[float] = []
        self.evasion_score = 0.5  # 0 = easy to detect, 1 = hard to detect
        
    def compute_detection_probability(self, timestep: int) -> float:
        """
        Estimate probability of being detected in current window.
        
        Returns:
            Probability in range [0, 1]
        """
        if not self.detection_history:
            return 0.5
        
        # Use recent history
        recent = self.detection_history[-self.memory_window:]
        return np.mean(recent)
    
    def compute_success_rate(self, timestep: int) -> float:
        """
        Compute success rate of attacks in current window.
        
        Returns:
            Success rate in range [0, 1]
        """
        if not self.success_history:
            return 0.5
        
        recent = self.success_history[-self.memory_window:]
        return np.mean(recent)
    
    def suggest_attack_variation(self, current_attack_type: str) -> str:
        """
        Suggest attack type variation based on adaptation.
        
        If current attack is being detected frequently,
        suggest switching to a different attack type.
        """
        detection_prob = self.compute_detection_probability(0)
        
        attack_types = [
            'message_injection',
            'message_replay',
            'coordinate_spoofing',
            'beacon_jamming',
            'sybil'
        ]
        
        # If highly detectable, suggest different attack
        if detection_prob > 0.7:
            # Remove current and similar attacks
            alternatives = [a for a in attack_types if a != current_attack_type]
            return np.random.choice(alternatives)
        
        return current_attack_type
    
    def suggest_attack_timing(self, current_timestep: int) -> int:
        """
        Suggest when next attack should occur.
        
        Returns:
            Recommended timestep for next attack
        """
        detection_prob = self.compute_detection_probability(current_timestep)
        
        # If frequently detected, increase inter-attack time
        base_interval = 10
        interval = base_interval * (1 + 5 * detection_prob)  # Up to 6x baseline
        
        return current_timestep + int(interval)
    
    def record_detection(self, was_detected: bool):
        """Record whether attack was detected."""
        self.detection_history.append(1.0 if was_detected else 0.0)
        
        # Update evasion score
        if was_detected:
            self.evasion_score *= (1 - self.adaptation_rate)
        else:
            self.evasion_score *= (1 + self.adaptation_rate)
        
        self.evasion_score = np.clip(self.evasion_score, 0.1, 0.9)
    
    def record_success(self, was_successful: bool):
        """Record whether attack was successful."""
        self.success_history.append(1.0 if was_successful else 0.0)
