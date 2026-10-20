"""
Supervised trainer: Standard supervised learning with adversarial labels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, WarmupLR
from typing import Dict, Optional, Tuple
import numpy as np
from loguru import logger


class SupervisedTrainer:
    """
    Trains the hybrid GNN model with labeled data.
    
    Binary classification task: Benign (0) vs Adversarial (1)
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, device: str = "cuda"):
        """
        Initialize supervised trainer.
        
        Args:
            model: Neural network model
            learning_rate: Learning rate
            weight_decay: L2 regularization coefficient
            device: Device for training (cuda/cpu)
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function with class weighting
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.7, 0.3], device=device),
            reduction='mean'
        )
        
        self.scheduler = None
        self.training_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
    
    def setup_scheduler(self, num_epochs: int, warmup_epochs: int = 10):
        """Setup learning rate scheduler."""
        # Note: WarmupLR is not in standard PyTorch, using custom implementation below
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
    
    def train_epoch(self, data_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: Batch loader with PyG Data objects
            
        Returns:
            Metrics dictionary
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data in data_loader:
            batch_data = batch_data.to(self.device)
            
            # Forward pass
            output = self.model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.temporal_embedding if hasattr(batch_data, 'temporal_embedding') else None
            )
            
            logits = output['logits']
            labels = batch_data.y
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self, data_loader) -> Dict[str, float]:
        """
        Validate on held-out data.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch_data in data_loader:
            batch_data = batch_data.to(self.device)
            
            output = self.model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.temporal_embedding if hasattr(batch_data, 'temporal_embedding') else None
            )
            
            logits = output['logits']
            labels = batch_data.y
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        
        # Compute additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        if len(np.unique(all_labels)) > 1:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
