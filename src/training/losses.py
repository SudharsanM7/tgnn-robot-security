"""
Loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Down-weights easy examples and focuses on hard negatives.
    From: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Logits (N, C)
            targets: Ground truth labels (N,)
            
        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()


class TrustAwareLoss(nn.Module):
    """
    Loss function that incorporates trust scores.
    
    Nodes with high trust are weighted less heavily to focus learning
    on ambiguous or untrustworthy nodes.
    """
    
    def __init__(self, base_loss: str = "ce"):
        super().__init__()
        self.base_loss_fn = F.cross_entropy if base_loss == "ce" else F.nll_loss
    
    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                trust_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute trust-aware loss.
        
        Args:
            predictions: Logits (N, C)
            targets: Ground truth (N,)
            trust_scores: Trust scores (N,) in [0, 1]
            
        Returns:
            Scalar loss
        """
        # Compute base loss
        base_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Weight by inverse of trust (uncertain samples weighted more)
        weights = 1.0 - trust_scores.clamp(0, 1)
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for joint supervised + self-supervised learning.
    
    L_total = λ_sup * L_supervised + λ_ss * L_self_supervised
    """
    
    def __init__(self, supervised_weight: float = 1.0,
                 self_supervised_weight: float = 0.5):
        super().__init__()
        self.supervised_weight = supervised_weight
        self.self_supervised_weight = self_supervised_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = lambda z1, z2: self._nt_xent(z1, z2)
    
    def _nt_xent(self, z_i: torch.Tensor, z_j: torch.Tensor,
                 temperature: float = 0.1) -> torch.Tensor:
        """NT-Xent contrastive loss."""
        batch_size = z_i.shape[0]
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        sim = similarity / temperature
        
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels, labels], dim=0)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim[mask] = float('-inf')
        
        pos = similarity.diag()
        loss = -torch.log(torch.exp(pos / temperature).sum(dim=1) / torch.exp(sim).sum(dim=1)).mean()
        
        return loss
    
    def forward(self, sup_predictions: torch.Tensor,
                sup_targets: torch.Tensor,
                ss_z_i: torch.Tensor,
                ss_z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            sup_predictions: Supervised predictions
            sup_targets: Supervised targets
            ss_z_i: Self-supervised embeddings view 1
            ss_z_j: Self-supervised embeddings view 2
            
        Returns:
            Combined loss
        """
        sup_loss = self.ce_loss(sup_predictions, sup_targets)
        ss_loss = self.contrastive_loss(ss_z_i, ss_z_j)
        
        total_loss = (self.supervised_weight * sup_loss +
                     self.self_supervised_weight * ss_loss)
        
        return total_loss
