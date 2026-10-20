"""
Saliency module: Gradient-based and other saliency computation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class SaliencyComputer:
    """
    Computes node saliency for interpretability.
    
    Saliency measures how much the model's prediction changes
    with small perturbations to node features.
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
    
    def compute_gradient_saliency(self, x: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 target_node: int,
                                 num_classes: int = 2) -> np.ndarray:
        """
        Compute gradient-based saliency.
        
        Saliency = |∂y / ∂x|
        
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge connectivity
            target_node: Node of interest
            num_classes: Number of output classes
            
        Returns:
            Saliency scores for each feature
        """
        x_input = x.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(x_input, edge_index)
        logits = output['logits']
        
        # Compute gradient w.r.t target node's classification
        logits_target = logits[target_node]
        
        # Sum gradients for both classes
        loss = logits_target.sum()
        loss.backward()
        
        # Extract gradient
        saliency = x_input.grad[target_node].abs().cpu().detach().numpy()
        
        return saliency
    
    def compute_integrated_gradient_saliency(self, x: torch.Tensor,
                                            edge_index: torch.Tensor,
                                            target_node: int,
                                            num_steps: int = 50) -> np.ndarray:
        """
        Compute integrated gradients for saliency.
        
        More robust than simple gradients.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            target_node: Node of interest
            num_steps: Integration steps
            
        Returns:
            Integrated gradient saliency
        """
        baseline = torch.zeros_like(x)
        integrated_grads = torch.zeros_like(x[target_node])
        
        for step in range(num_steps):
            alpha = step / num_steps
            perturbed_x = baseline + alpha * (x - baseline)
            perturbed_x.requires_grad_(True)
            
            output = self.model(perturbed_x, edge_index)
            logits = output['logits'][target_node]
            
            # Backward pass
            logits.sum().backward(retain_graph=True)
            
            integrated_grads += perturbed_x.grad[target_node]
        
        # Average and scale
        integrated_grads = (integrated_grads / num_steps).abs().cpu().detach().numpy()
        integrated_grads = integrated_grads * np.mean(np.abs(x[target_node].cpu().numpy()))
        
        return integrated_grads
    
    def compute_feature_importance(self, x: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  target_node: int,
                                  num_samples: int = 100) -> Dict[int, float]:
        """
        Compute feature importance via perturbation.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            target_node: Node of interest
            num_samples: Number of perturbation samples
            
        Returns:
            Importance scores per feature
        """
        with torch.no_grad():
            baseline_output = self.model(x, edge_index)
            baseline_pred = baseline_output['predicted_labels'][target_node]
        
        importance = {}
        num_features = x.shape[1]
        
        for feat_idx in range(num_features):
            importance_scores = []
            
            for _ in range(num_samples):
                # Perturb feature
                x_perturbed = x.clone()
                x_perturbed[target_node, feat_idx] = torch.randn(1, device=self.device)
                
                with torch.no_grad():
                    output = self.model(x_perturbed, edge_index)
                    new_pred = output['predicted_labels'][target_node]
                
                # Check if prediction changed
                if new_pred != baseline_pred:
                    importance_scores.append(1.0)
                else:
                    importance_scores.append(0.0)
            
            importance[feat_idx] = np.mean(importance_scores)
        
        return importance
