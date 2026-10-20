"""
Self-supervised trainer: Contrastive learning without labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, List
import numpy as np
from loguru import logger


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRLoss(nn.Module):
    """
    Contrastive loss based on SimCLR framework.
    
    Minimizes distance between augmented versions of same graph while
    maximizing distance to different graphs.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z_i: Embeddings of augmented view 1 (batch_size, projection_dim)
            z_j: Embeddings of augmented view 2 (batch_size, projection_dim)
            
        Returns:
            Scalar loss
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2*batch_size, 2*batch_size)
        
        # Create labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels, labels], dim=0)
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        
        # Extract positive and negative pairs
        pos = similarity_matrix[mask].view(2 * batch_size, 1)
        neg = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Compute NT-Xent loss
        logits = torch.cat([pos, neg], dim=1)
        logits = logits / self.temperature
        
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SelfSupervisedTrainer:
    """
    Self-supervised trainer using contrastive learning.
    
    Learns meaningful representations without labeled data by:
    1. Creating augmented views of graphs
    2. Maximizing agreement between augmented versions
    3. Minimizing agreement with other graphs
    """
    
    def __init__(self, model: nn.Module, projection_dim: int = 128,
                 temperature: float = 0.1, learning_rate: float = 0.001,
                 device: str = "cuda"):
        """
        Initialize self-supervised trainer.
        
        Args:
            model: Base model
            projection_dim: Dimension of projection head output
            temperature: Temperature parameter for contrastive loss
            learning_rate: Learning rate
            device: Device for training
        """
        self.model = model.to(device)
        self.device = device
        
        # Projection head
        self.projection_head = ContrastiveProjectionHead(
            input_dim=model.hidden_dim,
            hidden_dim=model.hidden_dim,
            output_dim=projection_dim
        ).to(device)
        
        self.contrastive_loss = SimCLRLoss(temperature=temperature)
        
        # Optimizer for both model and projection head
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(self.projection_head.parameters()),
            lr=learning_rate
        )
        
        self.training_history = {'contrastive_loss': []}
    
    def augment_graph(self, data, augmentation_strength: float = 0.2):
        """
        Create augmented version of graph.
        
        Augmentation strategies:
        - Node feature dropout
        - Edge dropout
        - Feature Gaussian noise
        """
        from copy import deepcopy
        augmented = deepcopy(data)
        
        # 1. Feature dropout
        if np.random.random() < augmentation_strength:
            mask = torch.rand(augmented.x.shape) > augmentation_strength
            augmented.x = augmented.x * mask.to(augmented.x.device)
        
        # 2. Feature noise
        if np.random.random() < augmentation_strength:
            noise = torch.randn_like(augmented.x) * augmentation_strength
            augmented.x = augmented.x + noise
        
        # 3. Edge dropout
        if np.random.random() < augmentation_strength / 2:
            keep_prob = 1 - augmentation_strength
            mask = torch.rand(augmented.edge_index.shape[1]) > augmentation_strength
            augmented.edge_index = augmented.edge_index[:, mask]
        
        return augmented
    
    def train_epoch(self, data_loader, augmentation_strength: float = 0.2) -> Dict[str, float]:
        """
        Train for one epoch with contrastive learning.
        
        Args:
            data_loader: Data loader
            augmentation_strength: Strength of graph augmentation
            
        Returns:
            Metrics dictionary
        """
        self.model.train()
        self.projection_head.train()
        
        total_loss = 0.0
        
        for batch_data in data_loader:
            batch_data = batch_data.to(self.device)
            
            # Create two augmented views
            aug1 = self.augment_graph(batch_data, augmentation_strength)
            aug2 = self.augment_graph(batch_data, augmentation_strength)
            
            # Move to device
            aug1 = aug1.to(self.device)
            aug2 = aug2.to(self.device)
            
            # Forward passes
            out1 = self.model(
                aug1.x, aug1.edge_index, aug1.edge_attr,
                aug1.temporal_embedding if hasattr(aug1, 'temporal_embedding') else None
            )
            embeddings1 = out1['embeddings']  # (num_nodes, hidden_dim)
            
            out2 = self.model(
                aug2.x, aug2.edge_index, aug2.edge_attr,
                aug2.temporal_embedding if hasattr(aug2, 'temporal_embedding') else None
            )
            embeddings2 = out2['embeddings']
            
            # Project embeddings
            z1 = self.projection_head(embeddings1)
            z2 = self.projection_head(embeddings2)
            
            # Compute contrastive loss
            loss = self.contrastive_loss(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        self.training_history['contrastive_loss'].append(avg_loss)
        
        return {'contrastive_loss': avg_loss}
