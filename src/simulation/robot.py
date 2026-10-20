"""
Robot module: Individual robot dynamics, state management, and motion models
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from loguru import logger


class RobotState(Enum):
    """Robot operational states"""
    NORMAL = "normal"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"
    BATTERY_LOW = "battery_low"


class TaskState(Enum):
    """Robot task execution states"""
    IDLE = "idle"
    MOVING_TO_TARGET = "moving_to_target"
    EXECUTING = "executing"
    RETURNING_HOME = "returning_home"
    COMMUNICATING = "communicating"


@dataclass
class RobotProperties:
    """Physical and communication properties of a robot"""
    robot_id: int
    mass: float = 1.0
    max_velocity: float = 2.0
    max_acceleration: float = 0.5
    max_angular_velocity: float = 2.0
    communication_range: float = 50.0
    sensor_noise_std: float = 0.05
    energy_capacity: float = 100.0  # Energy units
    energy_drain_rate: float = 0.1  # Per timestep
    energy_speed_cost: float = 0.05  # Cost per unit velocity


class RobotDynamics:
    """Handles robot kinematics and physics."""
    
    def __init__(self, properties: RobotProperties, dt: float = 0.1):
        self.properties = properties
        self.dt = dt
        self.friction_coeff = 0.15
        
    def integrate_state(self, position: np.ndarray, velocity: np.ndarray,
                       heading: float, control_velocity: np.ndarray,
                       control_angular_vel: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Integrate robot dynamics using kinematic model.
        
        Mathematical model:
        p(t+1) = p(t) + v(t) * dt
        v(t+1) = (1 - friction) * v(t) + u_v
        θ(t+1) = θ(t) + ω(t) * dt
        
        Args:
            position: Current 2D position
            velocity: Current 2D velocity
            heading: Current heading angle
            control_velocity: Commanded velocity
            control_angular_vel: Commanded angular velocity
            
        Returns:
            Tuple of (new_position, new_velocity, new_heading)
        """
        # Limit command inputs
        control_velocity = np.clip(control_velocity, 
                                   -self.properties.max_velocity,
                                   self.properties.max_velocity)
        control_angular_vel = np.clip(control_angular_vel,
                                      -self.properties.max_angular_velocity,
                                      self.properties.max_angular_velocity)
        
        # Position update (semi-implicit Euler)
        acceleration = (control_velocity - velocity) * self.properties.max_acceleration
        acceleration = np.clip(acceleration, -self.properties.max_acceleration,
                              self.properties.max_acceleration)
        
        new_velocity = velocity + acceleration * self.dt
        new_velocity = new_velocity * (1 - self.friction_coeff * self.dt)
        
        new_position = position + new_velocity * self.dt
        
        # Heading update
        new_heading = heading + control_angular_vel * self.dt
        new_heading = new_heading % (2 * np.pi)
        
        return new_position, new_velocity, new_heading
    
    def compute_energy_cost(self, velocity: np.ndarray) -> float:
        """
        Compute energy cost of motion.
        
        Cost = base_drain + speed_cost * ||v||
        """
        speed = np.linalg.norm(velocity)
        return self.properties.energy_drain_rate + self.properties.energy_speed_cost * speed


class Robot:
    """
    Represents a single robot in the multi-robot network.
    
    Comprehensive robot model with dynamics, energy, task states,
    communication, and trust tracking.
    """
    
    def __init__(self, properties: RobotProperties, dt: float = 0.1):
        self.properties = properties
        self.dynamics = RobotDynamics(properties, dt)
        self.dt = dt
        
        # Physical state
        self.position = np.random.uniform(-100, 100, 2).astype(np.float32)
        self.velocity = np.random.uniform(-0.5, 0.5, 2).astype(np.float32)
        self.heading = np.random.uniform(-np.pi, np.pi)
        self.acceleration = np.zeros(2, dtype=np.float32)
        
        # Energy model
        self.energy = properties.energy_capacity
        self.max_energy = properties.energy_capacity
        
        # State tracking
        self.state = RobotState.NORMAL
        self.task_state = TaskState.IDLE
        
        # Target for navigation
        self.navigation_target: Optional[np.ndarray] = None
        self.home_position = self.position.copy()
        
        # Control inputs (persistent)
        self.control_velocity = np.zeros(2, dtype=np.float32)
        self.control_angular_vel = 0.0
        
        # Message handling
        self.message_queue: List[Dict] = []
        self.outgoing_messages: List[Dict] = []
        self.sent_message_history: List[Dict] = []
        self.received_messages_history: List[Dict] = []
        
        # Communication tracking
        self.neighbors: List[int] = []  # IDs of neighbors in comm range
        self.neighbor_distances: Dict[int, float] = {}
        
        # Trust tracking
        self.trust_scores: Dict[int, float] = {}
        self.reputation_score: float = 1.0
        
        # Internal sensor readings (subject to noise)
        self.internal_state_vector = np.random.randn(32).astype(np.float32)
        
        # Statistics
        self.timesteps_active = 0
        self.total_distance_traveled = 0.0
        self.tasks_completed = 0
        
    def update(self, external_forces: np.ndarray = None):
        """
        Execute one timestep of robot evolution.
        
        Args:
            external_forces: Optional external force (e.g., wind)
        """
        if self.state == RobotState.ISOLATED:
            self.velocity *= 0.95  # Decay
            return
        
        # Apply external forces
        if external_forces is not None:
            self.control_velocity += external_forces
        
        # Integrate dynamics
        prev_pos = self.position.copy()
        self.position, self.velocity, self.heading = self.dynamics.integrate_state(
            self.position, self.velocity, self.heading,
            self.control_velocity, self.control_angular_vel
        )
        
        # Update statistics
        distance_traveled = np.linalg.norm(self.position - prev_pos)
        self.total_distance_traveled += distance_traveled
        
        # Update energy
        energy_cost = self.dynamics.compute_energy_cost(self.velocity)
        self.energy -= energy_cost
        
        if self.energy <= 0:
            self.energy = 0
            self.state = RobotState.BATTERY_LOW
            self.velocity *= 0.1  # Heavily damped
        elif self.energy < 0.2 * self.max_energy:
            self.state = RobotState.BATTERY_LOW
        elif self.state == RobotState.BATTERY_LOW and self.energy > 0.5 * self.max_energy:
            self.state = RobotState.NORMAL
        
        # Update sensor noise
        self._update_sensor_readings()
        
        self.timesteps_active += 1
        
    def _update_sensor_readings(self):
        """Update internal sensor state with noise."""
        # Sensors drift and add noise
        drift = np.random.randn(32) * 0.01
        noise = np.random.randn(32) * self.properties.sensor_noise_std
        self.internal_state_vector += drift + noise
        self.internal_state_vector = np.clip(self.internal_state_vector, -3, 3)
        
    def set_navigation_target(self, target: np.ndarray):
        """Set target position for navigation."""
        self.navigation_target = target.copy()
        self.task_state = TaskState.MOVING_TO_TARGET
        
    def set_control_velocity(self, velocity: np.ndarray, angular_velocity: float = 0.0):
        """Set commanded velocity."""
        self.control_velocity = velocity.copy()
        self.control_angular_vel = angular_velocity
        
    def add_incoming_message(self, message: Dict):
        """Queue incoming message."""
        message['received_at_timestep'] = self.timesteps_active
        self.message_queue.append(message)
        self.received_messages_history.append(message)
        
    def queue_outgoing_message(self, msg: Dict):
        """Queue message for broadcasting."""
        msg['sent_by'] = self.properties.robot_id
        msg['sent_at_timestep'] = self.timesteps_active
        self.outgoing_messages.append(msg)
        
    def clear_message_queues(self):
        """Clear message queues after processing."""
        self.message_queue.clear()
        self.sent_message_history.extend(self.outgoing_messages)
        self.outgoing_messages.clear()
        
    def get_state_vector(self) -> np.ndarray:
        """
        Get comprehensive state vector for neural network input.
        
        Returns:
            state_vector: [pos (2), vel (2), heading (1), angular_vel (1),
                          energy (1), internal_state (32d)] = 39D
        """
        state = np.concatenate([
            self.position.astype(np.float32),
            self.velocity.astype(np.float32),
            np.array([self.heading, self.control_angular_vel], dtype=np.float32),
            np.array([self.energy / self.max_energy], dtype=np.float32),  # Normalized
            self.internal_state_vector
        ])
        return state
    
    def get_observable_state(self) -> Dict:
        """Get observable state (what other robots can measure)."""
        return {
            'robot_id': self.properties.robot_id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'heading': self.heading,
            'state': self.state.value,
            'energy_level': self.energy / self.max_energy,
            'task_state': self.task_state.value
        }
    
    def compromise(self):
        """Mark robot as compromised/adversarial."""
        self.state = RobotState.COMPROMISED
        logger.warning(f"Robot {self.properties.robot_id} compromised")
        
    def isolate(self):
        """Isolate robot from network."""
        self.state = RobotState.ISOLATED
        self.message_queue.clear()
        
    def recharge(self, amount: float):
        """Recharge energy."""
        self.energy = min(self.energy + amount, self.max_energy)
        if self.energy > 0.5 * self.max_energy and self.state == RobotState.BATTERY_LOW:
            self.state = RobotState.NORMAL
