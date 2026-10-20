"""
Motion Models: Advanced motion control algorithms for robot swarms
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class MotionModel(ABC):
    """Abstract base class for motion models."""
    
    @abstractmethod
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute desired velocity for robot.
        
        Args:
            robot_state: Current robot state dictionary
            neighbors: List of neighbor state dicts
            target: Optional navigation target
            
        Returns:
            Desired velocity vector
        """
        pass


class WaypointNavigation(MotionModel):
    """
    Simple waypoint navigation to a target.
    
    Uses proportional control to reach target position.
    """
    
    def __init__(self, max_velocity: float = 2.0, proportional_gain: float = 0.5):
        self.max_velocity = max_velocity
        self.kp = proportional_gain
        
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute velocity toward target using proportional control.
        """
        if target is None:
            return np.zeros(2, dtype=np.float32)
        
        position = robot_state['position']
        direction = target - position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return np.zeros(2, dtype=np.float32)
        
        # Proportional control
        desired_velocity = (direction / distance) * min(distance * self.kp, self.max_velocity)
        
        return desired_velocity.astype(np.float32)


class BoidFlocking(MotionModel):
    """
    Boid flocking model based on Reynolds' steering behaviors.
    
    Combines:
    - Separation: Steer to avoid crowding local flockmates
    - Alignment: Steer towards average heading of local flockmates
    - Cohesion: Steer to move toward average position of local flockmates
    """
    
    def __init__(self, max_velocity: float = 2.0,
                 separation_radius: float = 2.0,
                 separation_weight: float = 1.5,
                 alignment_weight: float = 1.0,
                 cohesion_weight: float = 1.0,
                 max_steering_force: float = 0.3):
        """
        Initialize Boid flocking model.
        
        Args:
            max_velocity: Maximum robot speed
            separation_radius: Radius for separation computation
            separation_weight: Weight for separation behavior
            alignment_weight: Weight for alignment behavior
            cohesion_weight: Weight for cohesion behavior
            max_steering_force: Maximum steering force magnitude
        """
        self.max_velocity = max_velocity
        self.separation_radius = separation_radius
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.max_steering_force = max_steering_force
        
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute flocking velocity combining three behaviors.
        """
        position = robot_state['position']
        velocity = robot_state['velocity']
        
        # Separation: repel from nearby robots
        separation = self._separation(position, neighbors)
        
        # Alignment: match velocity of neighbors
        alignment = self._alignment(velocity, neighbors)
        
        # Cohesion: move toward center of mass of neighbors
        cohesion = self._cohesion(position, neighbors)
        
        # Combine steering behaviors
        steering = (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )
        
        # Limit steering force
        steering_mag = np.linalg.norm(steering)
        if steering_mag > self.max_steering_force:
            steering = (steering / steering_mag) * self.max_steering_force
        
        # Combine with current velocity
        desired_velocity = velocity + steering
        desired_mag = np.linalg.norm(desired_velocity)
        
        if desired_mag > self.max_velocity:
            desired_velocity = (desired_velocity / desired_mag) * self.max_velocity
        
        return desired_velocity.astype(np.float32)
    
    def _separation(self, position: np.ndarray, neighbors: List[Dict]) -> np.ndarray:
        """Compute separation steering (repulsion from neighbors)."""
        steering = np.zeros(2, dtype=np.float32)
        
        for neighbor in neighbors:
            neighbor_pos = neighbor['position']
            distance = np.linalg.norm(position - neighbor_pos)
            
            if distance < self.separation_radius and distance > 0.01:
                # Repel (away from neighbor)
                direction = position - neighbor_pos
                direction = direction / distance
                # Weight by inverse distance (farther = less repulsion)
                weight = 1.0 - (distance / self.separation_radius)
                steering += direction * weight
        
        return steering
    
    def _alignment(self, velocity: np.ndarray, neighbors: List[Dict]) -> np.ndarray:
        """Compute alignment steering (match velocity of neighbors)."""
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        avg_velocity = np.mean([n['velocity'] for n in neighbors], axis=0)
        steering = avg_velocity - velocity
        
        return steering.astype(np.float32)
    
    def _cohesion(self, position: np.ndarray, neighbors: List[Dict]) -> np.ndarray:
        """Compute cohesion steering (move toward center of mass)."""
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        center_of_mass = np.mean([n['position'] for n in neighbors], axis=0)
        steering = center_of_mass - position
        
        # Scale by distance (closer = stronger attraction)
        distance = np.linalg.norm(steering)
        if distance > 0.01:
            steering = (steering / distance) * min(distance * 0.1, self.max_velocity)
        
        return steering.astype(np.float32)


class TargetCover(MotionModel):
    """
    Biased flocking model that combines flocking with target following.
    
    Robots flock together while being attracted to a target/goal.
    """
    
    def __init__(self, max_velocity: float = 2.0,
                 target_bias: float = 0.3,
                 separation_radius: float = 2.0):
        self.max_velocity = max_velocity
        self.target_bias = target_bias
        self.boid_model = BoidFlocking(max_velocity, separation_radius)
        
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute velocity combining flocking and target attraction.
        """
        # Get base flocking velocity
        flocking_velocity = self.boid_model.compute_velocity(robot_state, neighbors)
        
        # If target available, bias toward it
        if target is not None:
            position = robot_state['position']
            to_target = target - position
            distance = np.linalg.norm(to_target)
            
            if distance > 0.1:
                target_attraction = (to_target / distance) * self.max_velocity
                
                # Blend flocking and target
                combined = (
                    (1 - self.target_bias) * flocking_velocity +
                    self.target_bias * target_attraction
                )
                
                # Normalize to max velocity
                combined_mag = np.linalg.norm(combined)
                if combined_mag > self.max_velocity:
                    combined = (combined / combined_mag) * self.max_velocity
                
                return combined.astype(np.float32)
        
        return flocking_velocity


class PotentialFieldNavigation(MotionModel):
    """
    Artificial potential field navigation combining:
    - Attractive potential toward goal
    - Repulsive potential from obstacles/robots
    """
    
    def __init__(self, max_velocity: float = 2.0,
                 goal_attraction: float = 1.0,
                 obstacle_repulsion: float = 1.0,
                 repulsion_radius: float = 3.0):
        self.max_velocity = max_velocity
        self.goal_attraction = goal_attraction
        self.obstacle_repulsion = obstacle_repulsion
        self.repulsion_radius = repulsion_radius
        
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute velocity using potential field method.
        """
        position = robot_state['position']
        total_force = np.zeros(2, dtype=np.float32)
        
        # Attractive force toward goal
        if target is not None:
            direction_to_goal = target - position
            distance_to_goal = np.linalg.norm(direction_to_goal)
            
            if distance_to_goal > 0.01:
                attraction = (direction_to_goal / distance_to_goal) * self.goal_attraction
                total_force += attraction
        
        # Repulsive forces from neighbors
        for neighbor in neighbors:
            neighbor_pos = neighbor['position']
            diff = position - neighbor_pos
            distance = np.linalg.norm(diff)
            
            if distance < self.repulsion_radius and distance > 0.01:
                # Repulsive force (inverse square law)
                repulsion = (diff / distance) * self.obstacle_repulsion / (distance ** 2 + 0.1)
                total_force += repulsion
        
        # Limit to max velocity
        force_mag = np.linalg.norm(total_force)
        if force_mag > self.max_velocity:
            total_force = (total_force / force_mag) * self.max_velocity
        
        return total_force.astype(np.float32)


class HybridAdaptiveControl(MotionModel):
    """
    Hybrid model that switches between different behaviors adaptively.
    
    - Formation keeping when in swarm
    - Target seeking when isolated
    - Obstacle avoidance always active
    """
    
    def __init__(self, max_velocity: float = 2.0,
                 swarm_threshold: int = 3,
                 formation_stiffness: float = 0.5):
        self.max_velocity = max_velocity
        self.swarm_threshold = swarm_threshold
        self.formation_stiffness = formation_stiffness
        
        self.boid_model = BoidFlocking(max_velocity)
        self.potential_field = PotentialFieldNavigation(max_velocity)
        
    def compute_velocity(self, robot_state: Dict, neighbors: List[Dict],
                        target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adaptively select motion behavior based on context.
        """
        # If enough neighbors, use flocking (formation keeping)
        if len(neighbors) >= self.swarm_threshold:
            return self.boid_model.compute_velocity(robot_state, neighbors, target)
        # Otherwise use potential field (target seeking + obstacle avoidance)
        else:
            return self.potential_field.compute_velocity(robot_state, neighbors, target)


class MotionModelFactory:
    """Factory for creating motion model instances."""
    
    _models = {
        'waypoint': WaypointNavigation,
        'boid': BoidFlocking,
        'target_cover': TargetCover,
        'potential_field': PotentialFieldNavigation,
        'hybrid_adaptive': HybridAdaptiveControl
    }
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> MotionModel:
        """Create motion model by name."""
        if model_name not in cls._models:
            raise ValueError(f"Unknown motion model: {model_name}")
        
        return cls._models[model_name](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available motion models."""
        return list(cls._models.keys())
