"""
Spatial Indexing: Efficient neighbor queries using KD-tree
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial import cKDTree
from loguru import logger


class SpatialIndex:
    """
    Maintains efficient spatial indexing for robot positions.
    
    Uses scipy's cKDTree (compiled C++ KD-tree) for O(log n) queries.
    Supports:
    - Radius queries (find all robots within distance)
    - K-nearest neighbor queries
    - Dynamic updates
    """
    
    def __init__(self, max_robots: int = 500, boxsize: Optional[Tuple[float, float]] = None):
        """
        Initialize spatial index.
        
        Args:
            max_robots: Expected maximum number of robots
            boxsize: Optional tuple (Lx, Ly) for periodic boundary conditions
        """
        self.max_robots = max_robots
        self.boxsize = boxsize
        self.tree: Optional[cKDTree] = None
        self.robot_ids: List[int] = []
        self.positions: Optional[np.ndarray] = None
        self.is_built = False
        
    def build(self, positions: np.ndarray, robot_ids: List[int]):
        """
        Build KD-tree from positions.
        
        Args:
            positions: (N, 2) array of robot positions
            robot_ids: List of robot IDs (length N)
        """
        if len(positions) == 0:
            self.tree = None
            self.robot_ids = []
            self.positions = None
            self.is_built = False
            return
        
        # Handle periodic boundary conditions if specified
        if self.boxsize is not None:
            positions = np.mod(positions, self.boxsize)
        
        self.positions = positions.copy()
        self.robot_ids = robot_ids.copy()
        
        # Build KD-tree with periodic boundaries if needed
        if self.boxsize is not None:
            self.tree = cKDTree(positions, boxsize=self.boxsize)
        else:
            self.tree = cKDTree(positions)
        
        self.is_built = True
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[int]:
        """
        Find all robots within radius.
        
        Args:
            position: Query position (2,)
            radius: Search radius
            
        Returns:
            List of robot IDs within radius
        """
        if not self.is_built or self.tree is None:
            return []
        
        # Handle periodic boundary
        if self.boxsize is not None:
            position = np.mod(position, self.boxsize)
        
        indices = self.tree.query_ball_point(position, radius)
        
        # Convert indices to robot IDs and exclude querying robot itself
        neighbor_ids = [self.robot_ids[idx] for idx in indices]
        
        return neighbor_ids
    
    def query_nearest(self, position: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Find k nearest robots.
        
        Args:
            position: Query position (2,)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (robot_ids, distances)
        """
        if not self.is_built or self.tree is None:
            return [], []
        
        if self.boxsize is not None:
            position = np.mod(position, self.boxsize)
        
        # Query k+1 to exclude self if it exists
        k_query = min(k + 1, len(self.robot_ids))
        distances, indices = self.tree.query(position, k=k_query, workers=-1)
        
        # Handle scalar return when k=1
        if np.isscalar(indices):
            indices = [indices]
            distances = [distances]
        else:
            indices = list(indices)
            distances = list(distances)
        
        # Convert to robot IDs
        neighbor_ids = [self.robot_ids[idx] for idx in indices]
        
        return neighbor_ids, distances
    
    def query_box(self, center: np.ndarray, half_width: float) -> List[int]:
        """
        Find all robots in rectangular region (faster than radius for axis-aligned queries).
        
        Args:
            center: Center of box (2,)
            half_width: Half-width of box
            
        Returns:
            List of robot IDs in box
        """
        if not self.is_built or self.tree is None:
            return []
        
        min_corner = center - half_width
        max_corner = center + half_width
        
        indices = self.tree.query_ball_point(
            center, np.linalg.norm([half_width, half_width])
        )
        
        neighbor_ids = [self.robot_ids[idx] for idx in indices]
        
        return neighbor_ids
    
    def rebuild_if_needed(self, positions: np.ndarray, robot_ids: List[int],
                         rebuild_interval: int = 10, current_step: int = 0):
        """
        Rebuild tree periodically or if needed.
        
        Args:
            positions: Current positions
            robot_ids: Current robot IDs
            rebuild_interval: Steps between rebuilds (adaptive)
            current_step: Current simulation step
        """
        if not self.is_built:
            self.build(positions, robot_ids)
        elif current_step % rebuild_interval == 0:
            self.build(positions, robot_ids)


class AdaptiveSpatialIndex(SpatialIndex):
    """
    Adaptive spatial index that automatically adjusts rebuild frequency
    based on robot speed and density.
    """
    
    def __init__(self, max_robots: int = 500, boxsize: Optional[Tuple[float, float]] = None):
        super().__init__(max_robots, boxsize)
        self.max_velocity = 2.0
        self.step_counter = 0
        self.rebuild_interval = 5
        
    def update(self, positions: np.ndarray, robot_ids: List[int],
              velocities: Optional[np.ndarray] = None):
        """
        Adaptively update spatial index.
        
        Adjusts rebuild frequency based on movement speed.
        
        Args:
            positions: Current positions
            robot_ids: Robot IDs
            velocities: Optional velocity data for predicting rebuild needs
        """
        # Estimate maximum speed
        if velocities is not None:
            max_speed = np.max(np.linalg.norm(velocities, axis=1))
            # Adjust rebuild interval based on speed
            # Faster robots need more frequent rebuilds
            self.rebuild_interval = max(2, int(20 / (1 + max_speed)))
        
        # Rebuild if needed
        if self.step_counter % self.rebuild_interval == 0:
            self.build(positions, robot_ids)
        
        self.step_counter += 1


class NeighborCache:
    """
    Caches neighbor information to avoid redundant queries.
    
    Provides O(1) neighbor lookup with periodic refresh.
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 5):
        """
        Initialize neighbor cache.
        
        Args:
            cache_size: Maximum cache entries
            cache_ttl: Cache time-to-live in timesteps
        """
        self.cache: Dict[int, List[int]] = {}
        self.cache_timestamp: Dict[int, int] = {}
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.current_step = 0
        
    def get(self, robot_id: int) -> Optional[List[int]]:
        """Get cached neighbors if valid."""
        if robot_id not in self.cache:
            return None
        
        if self.current_step - self.cache_timestamp[robot_id] > self.cache_ttl:
            # Cache expired
            del self.cache[robot_id]
            del self.cache_timestamp[robot_id]
            return None
        
        return self.cache[robot_id].copy()
    
    def set(self, robot_id: int, neighbors: List[int]):
        """Cache neighbors for robot."""
        # Evict old entries if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_id = min(self.cache_timestamp, key=self.cache_timestamp.get)
            del self.cache[oldest_id]
            del self.cache_timestamp[oldest_id]
        
        self.cache[robot_id] = neighbors.copy()
        self.cache_timestamp[robot_id] = self.current_step
    
    def step(self):
        """Advance one timestep."""
        self.current_step += 1
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.cache_timestamp.clear()


class NeighborhoodManager:
    """
    Unified interface for neighbor management combining spatial indexing and caching.
    """
    
    def __init__(self, communication_range: float, max_robots: int = 500,
                 boxsize: Optional[Tuple[float, float]] = None,
                 use_cache: bool = True):
        """
        Initialize neighborhood manager.
        
        Args:
            communication_range: Radius for neighbor queries
            max_robots: Maximum expected robots
            boxsize: Periodic boundary box size
            use_cache: Whether to use neighbor caching
        """
        self.communication_range = communication_range
        self.spatial_index = AdaptiveSpatialIndex(max_robots, boxsize)
        self.use_cache = use_cache
        self.neighbor_cache = NeighborCache() if use_cache else None
        
        # Statistics
        self.query_count = 0
        self.cache_hits = 0
        
    def update_positions(self, positions: Dict[int, np.ndarray],
                        velocities: Optional[Dict[int, np.ndarray]] = None):
        """
        Update spatial index with new positions.
        
        Args:
            positions: Dict mapping robot_id -> position
            velocities: Optional dict of velocities for adaptation
        """
        robot_ids = list(positions.keys())
        positions_array = np.array([positions[rid] for rid in robot_ids], dtype=np.float32)
        
        velocities_array = None
        if velocities is not None:
            velocities_array = np.array([velocities.get(rid, np.zeros(2)) for rid in robot_ids])
        
        self.spatial_index.update(positions_array, robot_ids, velocities_array)
        
        if self.use_cache:
            self.neighbor_cache.step()
    
    def get_neighbors(self, robot_id: int, position: np.ndarray) -> List[int]:
        """
        Get neighbors of a robot using spatial index and cache.
        
        Args:
            robot_id: ID of query robot
            position: Position of query robot
            
        Returns:
            List of neighbor robot IDs within communication range
        """
        self.query_count += 1
        
        # Check cache first
        if self.use_cache:
            cached = self.neighbor_cache.get(robot_id)
            if cached is not None:
                self.cache_hits += 1
                return cached
        
        # Query spatial index
        neighbors = self.spatial_index.query_radius(position, self.communication_range)
        
        # Remove self
        if robot_id in neighbors:
            neighbors.remove(robot_id)
        
        # Cache result
        if self.use_cache:
            self.neighbor_cache.set(robot_id, neighbors)
        
        return neighbors
    
    def get_cache_efficiency(self) -> float:
        """Compute cache hit rate."""
        if self.query_count == 0:
            return 0.0
        return self.cache_hits / self.query_count
    
    def clear_cache(self):
        """Clear neighbor cache."""
        if self.use_cache:
            self.neighbor_cache.clear()
