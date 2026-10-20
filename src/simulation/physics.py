"""
Physics Simulation Module: Collision detection, boundary handling, force integration
"""

import numpy as np
from typing import Tuple


class PhysicsSimulator:
    """
    Physics simulation handling collisions and boundary conditions.
    
    Features:
    - Elastic collision response
    - Periodic boundary conditions (wrapping)
    - Hard boundary conditions (reflection)
    - Potential field force computation
    - Aerodynamic drag simulation
    """
    
    def __init__(self, dt: float, world_size: Tuple[float, float],
                 boundary_mode: str = 'periodic', collision_damping: float = 0.9):
        """
        Initialize physics simulator.
        
        Args:
            dt: Timestep duration
            world_size: (width, height) of world
            boundary_mode: 'periodic' (wrapping) or 'reflect' (bouncing)
            collision_damping: Energy retention in collisions (0-1)
        """
        self.dt = dt
        self.world_size = np.array(world_size, dtype=np.float32)
        self.boundary_mode = boundary_mode
        self.collision_damping = collision_damping
    
    def apply_periodic_boundary(self, position: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions (wrapping)."""
        return position % self.world_size
    
    def apply_reflect_boundary(self, position: np.ndarray, velocity: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply reflective boundary conditions (bouncing).
        
        Returns:
            (position, velocity) after boundary handling
        """
        pos_out = position.copy()
        vel_out = velocity.copy()
        
        # Check boundaries
        for i in range(2):  # x and y
            if pos_out[i] < 0:
                pos_out[i] = -pos_out[i]
                vel_out[i] = -vel_out[i] * self.collision_damping
            elif pos_out[i] > self.world_size[i]:
                pos_out[i] = 2 * self.world_size[i] - pos_out[i]
                vel_out[i] = -vel_out[i] * self.collision_damping
        
        return pos_out, vel_out
    
    def handle_boundary(self, position: np.ndarray, velocity: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions based on mode."""
        if self.boundary_mode == 'periodic':
            return self.apply_periodic_boundary(position), velocity
        else:  # reflect
            return self.apply_reflect_boundary(position, velocity)
    
    def apply_collision_response(self, pos_a: np.ndarray, vel_a: np.ndarray,
                                pos_b: np.ndarray, vel_b: np.ndarray,
                                mass_a: float = 1.0, mass_b: float = 1.0
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute elastic collision response between two robots.
        
        Uses 1D conservation of momentum and energy along collision normal.
        
        Args:
            pos_a, pos_b: Positions of robots A and B
            vel_a, vel_b: Velocities of robots A and B
            mass_a, mass_b: Mass of robots (default: equal mass)
            
        Returns:
            (new_vel_a, new_vel_b) after collision
        """
        # Vector from A to B
        delta_pos = pos_b - pos_a
        distance = np.linalg.norm(delta_pos)
        
        if distance < 1e-6:
            # Robots at exactly same position: separate them
            delta_pos = np.array([0.1, 0.0], dtype=np.float32)
            distance = 0.1
        
        # Collision normal (unit vector from A to B)
        normal = delta_pos / distance
        
        # Relative velocity
        delta_vel = vel_a - vel_b
        
        # Relative velocity in collision normal direction
        vel_along_normal = np.dot(delta_vel, normal)
        
        # Only resolve if objects are moving toward each other
        if vel_along_normal > 0:
            return vel_a, vel_b
        
        # Elasticity factor (perfectly elastic)
        restitution = 1.0
        
        # Impulse scalar (simplified for equal masses)
        total_mass = mass_a + mass_b
        impulse = -(1 + restitution) * vel_along_normal / (2 / total_mass)
        
        # Apply impulse
        impulse_vector = impulse * normal
        new_vel_a = vel_a + impulse_vector / mass_a
        new_vel_b = vel_b - impulse_vector / mass_b
        
        return new_vel_a, new_vel_b
    
    def compute_potential_field_force(self, 
                                     position: np.ndarray,
                                     goal: np.ndarray,
                                     obstacles: np.ndarray,
                                     goal_force_scale: float = 1.0,
                                     obstacle_force_scale: float = 1.0,
                                     obstacle_radius: float = 1.0
                                     ) -> np.ndarray:
        """
        Compute artificial potential field force.
        
        F = F_goal + F_obstacles
        
        F_goal: Attractive force toward goal
        F_obstacles: Repulsive forces from obstacles (inverse-square law)
        
        Args:
            position: Current position
            goal: Goal position
            obstacles: Nx2 array of obstacle positions
            goal_force_scale: Scale factor for goal attraction
            obstacle_force_scale: Scale factor for obstacle repulsion
            obstacle_radius: Minimum distance for obstacle force
            
        Returns:
            Force vector (magnitude and direction)
        """
        force = np.zeros(2, dtype=np.float32)
        
        # Goal attraction
        goal_dir = goal - position
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist > 1e-6:
            force += goal_force_scale * (goal_dir / goal_dist)
        
        # Obstacle repulsion
        for obstacle in obstacles:
            obs_dir = obstacle - position  # Direction FROM position TO obstacle
            obs_dist = np.linalg.norm(obs_dir)
            
            if obs_dist < obstacle_radius:
                obs_dist = obstacle_radius  # Clamp to avoid singularity
            
            # Inverse-square repulsive force (away from obstacle)
            repulsion = -obstacle_force_scale / (obs_dist ** 2 + 0.1)
            force += repulsion * (obs_dir / obs_dist)
        
        return force
    
    def compute_aerodynamic_drag(self, velocity: np.ndarray, 
                                drag_coefficient: float = 0.1) -> np.ndarray:
        """
        Compute aerodynamic drag force.
        
        F_drag = -drag_coeff * ||v|| * v (quadratic drag)
        
        Args:
            velocity: Current velocity
            drag_coefficient: Drag coefficient
            
        Returns:
            Drag force vector
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(2, dtype=np.float32)
        
        return -drag_coefficient * speed * velocity
    
    def integrate_external_forces(self, velocity: np.ndarray,
                                 external_forces: np.ndarray,
                                 mass: float = 1.0
                                 ) -> np.ndarray:
        """
        Integrate external forces to compute acceleration and update velocity.
        
        v_new = v + (F_external / mass) * dt
        
        Args:
            velocity: Current velocity
            external_forces: External force vector
            mass: Robot mass
            
        Returns:
            Updated velocity
        """
        acceleration = external_forces / mass
        return velocity + acceleration * self.dt
