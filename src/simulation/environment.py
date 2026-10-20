"""
Environm module: Multi-robot simulation environment management
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from .robot import Robot, RobotProperties, RobotState
from loguru import logger


class MultiRobotEnvironment:
    """
    Manages a multi-robot communication network simulation.
    
    This environment handles:
    - Robot dynamics and interactions
    - Network topology and communication
    - Message passing with realistic delays and noise
    - Time-step evolution
    """
    
    def __init__(self, num_robots: int, communication_range: float = 50.0,
                 dt: float = 0.1, seed: int = 42):
        """
        Initialize environment.
        
        Args:
            num_robots: Number of robots in network
            communication_range: Maximum communication range in meters
            dt: Timestep duration in seconds
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.num_robots = num_robots
        self.communication_range = communication_range
        self.dt = dt
        self.timestep = 0
        
        # Initialize robots
        self.robots: Dict[int, Robot] = {}
        for i in range(num_robots):
            props = RobotProperties(
                robot_id=i,
                communication_range=communication_range
            )
            self.robots[i] = Robot(props)
        
        # Network topology
        self.communication_graph = nx.Graph()
        self.communication_graph.add_nodes_from(range(num_robots))
        
        # Statistics tracking
        self.message_log: List[Dict] = []
        self.topology_history: List[nx.Graph] = []
        
    def compute_network_topology(self) -> nx.Graph:
        """
        Compute current communication graph based on distance.
        
        Returns:
            Updated communication graph
        """
        self.communication_graph.clear()
        self.communication_graph.add_nodes_from(range(self.num_robots))
        
        # Compute pairwise distances
        positions = np.array([self.robots[i].position for i in range(self.num_robots)])
        
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                distance = np.linalg.norm(positions[i] - positions[j])
                
                # Check if in communication range
                if distance <= self.communication_range:
                    # Add small noise to signal strength
                    signal_strength = 1.0 - distance / self.communication_range
                    signal_strength += np.random.normal(0, 0.05)
                    signal_strength = np.clip(signal_strength, 0, 1)
                    
                    self.communication_graph.add_edge(i, j, 
                                                     weight=signal_strength,
                                                     distance=distance)
        
        self.topology_history.append(self.communication_graph.copy())
        return self.communication_graph
    
    def step(self):
        """Execute one simulation timestep."""
        # Update robot positions
        for robot in self.robots.values():
            robot.update_position(self.dt)
        
        # Compute network topology
        self.compute_network_topology()
        
        # Process message passing
        self._process_message_passing()
        
        self.timestep += 1
    
    def _process_message_passing(self):
        """
        Process message passing on the communication graph.
        
        Messages are sent along edges with realistic delays.
        """
        # For each robot, send queued messages to neighbors
        for robot_id, robot in self.robots.items():
            if robot.state == RobotState.ISOLATED:
                continue
                
            for neighbor_id in self.communication_graph.neighbors(robot_id):
                # Get edge weight (signal strength)
                edge_data = self.communication_graph[robot_id][neighbor_id]
                signal_strength = edge_data.get('weight', 0.5)
                
                # Each outgoing message is sent
                for msg in robot.outgoing_messages:
                    # Simulate transmission delay
                    delay = np.random.uniform(0.01, 0.1)
                    
                    # Message may be corrupted based on signal strength
                    if np.random.random() < signal_strength:
                        received_msg = msg.copy()
                        received_msg['signal_strength'] = signal_strength
                        received_msg['transmission_delay'] = delay
                        received_msg['received_at'] = self.timestep + delay
                        
                        # With small probability, message is corrupted
                        if np.random.random() > 0.95:
                            received_msg['corrupted'] = True
                        
                        self.robots[neighbor_id].add_incoming_message(received_msg)
                        
                        # Log message
                        self.message_log.append({
                            'from': robot_id,
                            'to': neighbor_id,
                            'timestep': self.timestep,
                            'signal_strength': signal_strength,
                            'corrupted': received_msg.get('corrupted', False)
                        })
            
            # Clear outgoing queue
            robot.outgoing_messages.clear()
    
    def get_robot_state(self, robot_id: int) -> np.ndarray:
        """Get current state vector of a robot."""
        return self.robots[robot_id].get_state_vector()
    
    def get_all_robot_states(self) -> np.ndarray:
        """Get stacked state vectors of all robots."""
        states = []
        for i in range(self.num_robots):
            states.append(self.robots[i].get_state_vector())
        return np.stack(states)
    
    def reset(self):
        """Reset environment to initial state."""
        self.timestep = 0
        self.message_log.clear()
        self.topology_history.clear()
        
        # Re-initialize robots
        for i in range(self.num_robots):
            props = RobotProperties(
                robot_id=i,
                communication_range=self.communication_range
            )
            self.robots[i] = Robot(props)
