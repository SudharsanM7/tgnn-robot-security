"""
Data utilities: Data loading, preprocessing, and batching
"""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from typing import List, Dict, Tuple, Optional


class DataBatcher:
    """
    Handles batch creation and temporal sequence management.
    """
    
    @staticmethod
    def create_temporal_batches(graphs: List[Data],
                               batch_size: int = 32,
                               shuffle: bool = True) -> DataLoader:
        """
        Create batches of temporal graphs.
        
        Args:
            graphs: List of PyG Data objects
            batch_size: Batch size
            shuffle: Whether to shuffle batches
            
        Returns:
            PyG DataLoader
        """
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def collate_temporal_sequence(sequence: List[Data]) -> Data:
        """
        Collate a sequence of graphs into single batch.
        
        Args:
            sequence: List of Data objects
            
        Returns:
            Batched Data object
        """
        from torch_geometric.data import Batch
        return Batch.from_data_list(sequence)


class FeatureNormalizer:
    """
    Normalizes features for training stability.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fit = False
    
    def fit(self, features: np.ndarray):
        """
        Fit normalizer on training data.
        
        Args:
            features: Feature matrix (N, D)
        """
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + 1e-8
        self.is_fit = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Normalized features
        """
        if not self.is_fit:
            raise RuntimeError("Normalizer not fit. Call fit() first.")
        
        return (features - self.mean) / self.std
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)


class DataSplitter:
    """
    Splits data into train/val/test sets.
    """
    
    @staticmethod
    def temporal_split(graphs: List[Data],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split temporal sequence into train/val/test.
        
        Preserves temporal order.
        
        Args:
            graphs: List of graphs ordered by time
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            Tuple of (train, val, test) graph lists
        """
        n = len(graphs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = graphs[:train_end]
        val = graphs[train_end:val_end]
        test = graphs[val_end:]
        
        return train, val, test
    
    @staticmethod
    def random_split(graphs: List[Data],
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Random split (does not preserve temporal order).
        
        Args:
            graphs: List of graphs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed
            
        Returns:
            Tuple of (train, val, test) graph lists
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(graphs))
        
        n = len(graphs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train = [graphs[i] for i in train_indices]
        val = [graphs[i] for i in val_indices]
        test = [graphs[i] for i in test_indices]
        
        return train, val, test
