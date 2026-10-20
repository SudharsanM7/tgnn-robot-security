"""
Advanced Simulation Engine: High-fidelity multi-robot simulation orchestration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import time
from loguru import logger

from .robot import Robot, RobotProperties, RobotState, TaskState
from .motion_models import MotionModel, MotionModelFactory
from .spatial_indexing import NeighborhoodManager
from .communication import CommunicationProtocol, MessageType
from .communication_integration import CommunicationManager, CommunicationConfig
from .physics import PhysicsSimulator


class SimulationConfig:
    """Configuration for simulation."""
    
    def __init__(self):
        self.num_robots: int = 50
        self.world_size: Tuple[float, float] = (200, 200)
        self.timestep: float = 0.1
        self.simulation_time: float = 100.0
        self.communication_range: float = 50.0
        self.motion_model: str = "boid"
        self.max_velocity: float = 2.0
        self.energy_drain_rate: float = 0.1
        self.use_periodic_boundary: bool = True
        self.enable_communication: bool = True  # Communication layer enabled
        self.packet_loss_rate: float = 0.01  # 1% packet loss
        self.transmission_delay: float = 0.001  # seconds
        self.seed: int = 42
        self.log_frequency: int = 10  # Log every N timesteps
        self.target_update_interval: int = 50


class SimulationEngine:
    """
    High-performance multi-robot simulation engine.
    
    Features:
    - Efficient vectorized updates
    - Spatial indexing for neighbor queries
    - Advanced motion models
    - Energy management
    - State logging for ML datasets
    - Physics simulation with collisions
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize simulation engine."""
        np.random.seed(config.seed)
        self.config = config
        self.timestep_count = 0
        self.total_timesteps = int(config.simulation_time / config.timestep)
        
        # Create robots
        self.robots: Dict[int, Robot] = {}
        self._initialize_robots()
        
        # Initialize spatial indexing
        self.neighborhood_manager = NeighborhoodManager(
            communication_range=config.communication_range,
            max_robots=config.num_robots,
            boxsize=config.world_size if config.use_periodic_boundary else None,
            use_cache=True
        )
        
        # Initialize motion model
        self.motion_model: MotionModel = MotionModelFactory.create(
            config.motion_model,
            max_velocity=config.max_velocity
        )
        
        # Physics simulator
        self.physics_sim = PhysicsSimulator(
            dt=config.timestep,
            world_size=config.world_size
        )
        
        # Communication
        self.communication_protocol = CommunicationProtocol()
        
        # Communication manager (realistic network layer)
        comm_config = CommunicationConfig(
            enabled=config.enable_communication,
            communication_range=config.communication_range,
            base_transmission_delay=config.transmission_delay,
            packet_loss_rate=config.packet_loss_rate
        )
        self.communication_manager = CommunicationManager(config.num_robots, comm_config)
        
        # Logging and data collection
        self.state_log: List[Dict] = []
        self.action_log: List[Dict] = []
        self.network_log: List[Dict] = []
        self.energy_log: Dict[int, List[float]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_messages_sent': 0,
            'total_collisions': 0,
            'robots_compromised': 0,
            'robots_isolated': 0,
            'avg_neighbor_count': 0
        }
        
        logger.info(f"Initialized simulation with {config.num_robots} robots")
        
    def _initialize_robots(self):
        """Create and initialize all robots."""
        for robot_id in range(self.config.num_robots):
            props = RobotProperties(
                robot_id=robot_id,
                max_velocity=self.config.max_velocity,
                communication_range=self.config.communication_range,
                energy_capacity=100.0,
                energy_drain_rate=self.config.energy_drain_rate
            )
            self.robots[robot_id] = Robot(props, dt=self.config.timestep)
        
        logger.debug(f"Created {len(self.robots)} robots")
    
    def step(self, apply_attacks: Optional[Callable] = None,
            update_targets: Optional[Callable] = None):
        """
        Execute one simulation timestep.
        
        Args:
            apply_attacks: Optional callback for applying adversarial attacks
            update_targets: Optional callback for updating robot targets
        """
        # ========== UPDATE SPATIAL INDEX ==========
        positions_dict = {rid: r.position for rid, r in self.robots.items()}
        velocities_dict = {rid: r.velocity for rid, r in self.robots.items()}
        self.neighborhood_manager.update_positions(positions_dict, velocities_dict)
        
        # ========== MOTION CONTROL ==========
        # Compute desired velocities (vectorized where possible)
        self._compute_motion_commands(update_targets)
        
        # ========== ROBOT DYNAMICS UPDATE ==========
        # Apply physics and update positions
        self._update_robot_dynamics()
        
        # ========== COLLISION DETECTION ==========
        self._handle_collisions()
        
        # ========== APPLY ATTACKS ==========
        if apply_attacks is not None:
            apply_attacks(self)
        
        # ========== MESSAGE PASSING ==========
        self._process_communication()
        
        # ========== STATE LOGGING ==========
        if self.timestep_count % self.config.log_frequency == 0:
            self._log_state()
        
        # ========== CLEANUP ==========
        for robot in self.robots.values():
            robot.clear_message_queues()
        
        self.timestep_count += 1
    
    def _compute_motion_commands(self, update_targets: Optional[Callable]):
        """Compute motion commands for all robots."""
        for robot_id, robot in self.robots.items():
            if robot.state == RobotState.ISOLATED:
                continue
            
            # Get neighbors
            neighbors = self.neighborhood_manager.get_neighbors(
                robot_id, robot.position
            )
            
            # Get neighbor states
            neighbor_states = [
                self.robots[nid].get_observable_state() for nid in neighbors
                if nid in self.robots
            ]
            
            # Get navigation target if provided
            target = None
            if update_targets is not None:
                target = update_targets(robot_id, robot.position, self.timestep_count)
            elif robot.navigation_target is not None:
                target = robot.navigation_target
            
            # Compute desired velocity using motion model
            robot_state = robot.get_observable_state()
            desired_velocity = self.motion_model.compute_velocity(
                robot_state, neighbor_states, target
            )
            
            # Set control input
            robot.set_control_velocity(desired_velocity)
    
    def _update_robot_dynamics(self):
        """Update all robot positions and states."""
        external_forces = np.zeros((len(self.robots), 2), dtype=np.float32)
        
        # Optional: add environmental forces (e.g., wind)
        # external_forces[:] = WIND_VECTOR
        
        for i, (robot_id, robot) in enumerate(self.robots.items()):
            force = external_forces[i] if i < len(external_forces) else np.zeros(2)
            robot.update(external_forces=force)
    
    def _handle_collisions(self):
        """Detect and handle collisions between robots."""
        collision_threshold = 1.0  # Minimum distance before collision
        
        robot_ids = list(self.robots.keys())
        positions = np.array([self.robots[rid].position for rid in robot_ids])
        
        # Vectorized pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(positions, metric='euclidean'))
        
        # Find collision pairs
        collision_pairs = np.argwhere(
            (distances < collision_threshold) & (distances > 0)
        )
        
        for i, j in collision_pairs:
            rid_i, rid_j = robot_ids[i], robot_ids[j]
            robot_i, robot_j = self.robots[rid_i], self.robots[rid_j]
            
            # Apply collision response (elastic)
            robot_i.velocity, robot_j.velocity = self.physics_sim.apply_collision_response(
                robot_i.position, robot_i.velocity,
                robot_j.position, robot_j.velocity
            )
            
            self.stats['total_collisions'] += 1
    
    def _process_communication(self):
        """Handle realistic message passing with network simulation."""
        if not self.config.enable_communication:
            return
        
        # Collect current robot positions
        robot_positions = {rid: r.position.copy() for rid, r in self.robots.items()}
        
        # Process all pending messages through network
        received_messages = self.communication_manager.process_messages(
            robot_positions, 
            self.timestep_count * self.config.timestep
        )
        
        # Deliver received messages to robots
        for robot_id, messages in received_messages.items():
            if robot_id in self.robots:
                for msg in messages:
                    self.robots[robot_id].add_incoming_message({
                        'msg_id': msg.msg_id,
                        'sender_id': msg.sender_id,
                        'type': msg.msg_type.value,
                        'payload': msg.payload,
                        'signal_strength': msg.signal_strength,
                        'latency': msg.latency,
                        'packet_integrity': msg.packet_integrity,
                        'received_at': self.timestep_count
                    })
                    self.stats['total_messages_sent'] += 1
    
    def _log_state(self):
        """Log current simulation state for dataset generation."""
        state_entry = {
            'timestep': self.timestep_count,
            'scenario_id': str(getattr(self, 'scenario_id', 'unknown')),
            'seed': int(self.config.seed),
            'adversary_ids': list(getattr(self, '_adversary_ids', [])),
            'robots': {}
        }
        
        # Collect per-robot data
        total_neighbors = 0
        for robot_id, robot in self.robots.items():
            neighbors = self.neighborhood_manager.get_neighbors(robot_id, robot.position)
            total_neighbors += len(neighbors)
            
            state_entry['robots'][robot_id] = {
                'position': robot.position.copy(),
                'velocity': robot.velocity.copy(),
                'heading': robot.heading,
                'energy': robot.energy,
                'state': robot.state.value,
                'task_state': robot.task_state.value,
                'num_neighbors': len(neighbors),
                'reputation': robot.reputation_score
            }
            
            # Log energy
            self.energy_log[robot_id].append(robot.energy)
        
        state_entry['network'] = {
            'avg_neighbors': total_neighbors / max(len(self.robots), 1),
            'cache_efficiency': self.neighborhood_manager.get_cache_efficiency()
        }
        
        self.state_log.append(state_entry)
        
        # Update statistics
        self.stats['avg_neighbor_count'] = state_entry['network']['avg_neighbors']
        self.stats['robots_compromised'] = sum(
            1 for r in self.robots.values() if r.state == RobotState.COMPROMISED
        )
        self.stats['robots_isolated'] = sum(
            1 for r in self.robots.values() if r.state == RobotState.ISOLATED
        )
    
    def run(self, apply_attacks: Optional[Callable] = None,
           update_targets: Optional[Callable] = None,
           progress_callback: Optional[Callable] = None) -> float:
        """
        Run complete simulation.
        
        Args:
            apply_attacks: Optional function(engine) for attack simulation
            update_targets: Optional function(robot_id, position, timestep) -> target
            progress_callback: Optional function(current, total) for progress reporting
            
        Returns:
            Total simulation time (seconds)
        """
        start_time = time.time()
        
        logger.info(f"Starting simulation for {self.total_timesteps} timesteps")
        
        try:
            for _ in range(self.total_timesteps):
                self.step(apply_attacks=apply_attacks, update_targets=update_targets)
                
                if progress_callback is not None:
                    progress_callback(self.timestep_count, self.total_timesteps)
        
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average: {elapsed_time / self.total_timesteps * 1000:.2f} ms/step")
        
        return elapsed_time
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        return {
            **self.stats,
            'timestep_count': self.timestep_count,
            'total_timesteps': self.total_timesteps,
            'cache_efficiency': self.neighborhood_manager.get_cache_efficiency()
        }
    
    def export_dataset(self, output_path: str):
        """
        Export simulation data as dataset for ML training.
        
        Args:
            output_path: Path to save dataset
        """
        import pickle
        
        dataset = {
            'config': {
                'num_robots': self.config.num_robots,
                'world_size': self.config.world_size,
                'timestep': self.config.timestep,
                'simulation_time': self.config.simulation_time,
                'communication_range': self.config.communication_range,
                'motion_model': self.config.motion_model,
                'seed': self.config.seed,
                'scenario_id': str(getattr(self, 'scenario_id', 'unknown')),
            },
            'state_log': self.state_log,
            'energy_log': dict(self.energy_log),
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Exported dataset to {output_path}")
    
    def get_graph_snapshot(self, timestep: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get graph snapshot of communication network at timestep.
        
        Returns:
            Tuple of (node_features, edge_list)
        """
        if timestep is None:
            timestep = min(self.timestep_count, len(self.state_log) - 1)
        
        if timestep >= len(self.state_log):
            raise ValueError(f"Timestep {timestep} not logged")
        
        state = self.state_log[timestep]
        
        # Extract node features
        num_robots = len(state['robots'])
        node_features = []
        
        for robot_id in sorted(state['robots'].keys()):
            robot_state = state['robots'][robot_id]
            feature = np.concatenate([
                robot_state['position'],
                robot_state['velocity'],
                [robot_state['heading']],
                [robot_state['energy'] / 100.0],
                [1.0 if robot_state['state'] == 'normal' else 0.0]
            ])
            node_features.append(feature)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Build communication graph
        edges = []
        for robot_id, robot_state in state['robots'].items():
            neighbors = self.neighborhood_manager.get_neighbors(
                robot_id, robot_state['position']
            )
            for neighbor_id in neighbors:
                if neighbor_id > robot_id:  # Avoid duplicates
                    edges.append([robot_id, neighbor_id])
        
        edge_list = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        
        return node_features, edge_list
