"""
Evaluation module: Testing and benchmarking
"""

import torch
import numpy as np
from typing import Dict, List
from loguru import logger

from ..explainability.explainer import HybridGNNExplainer


class ModelEvaluator:
    """
    Comprehensive model evaluation on test data.
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.test_results = {}
        self.explainer = HybridGNNExplainer(model, device=device)
    
    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation results dictionary
        """
        self.model.eval()
        
        all_predictions = []
        all_scores = []
        all_labels = []
        all_trust_scores = []
        
        for batch_data in test_loader:
            batch_data = batch_data.to(self.device)
            
            output = self.model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.temporal_embedding if hasattr(batch_data, 'temporal_embedding') else None
            )
            
            preds = output['predicted_labels'].cpu().numpy()
            scores = output['adversary_probability'].cpu().numpy()
            labels = batch_data.y.cpu().numpy()
            trust = output['trust_scores'].cpu().numpy()
            
            all_predictions.extend(preds)
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_trust_scores.extend(trust.squeeze())
        
        # Compute metrics
        from .metrics import DetectionMetrics
        
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        metrics = DetectionMetrics.compute_detection_metrics(
            all_labels, all_predictions, all_scores
        )
        
        self.test_results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'scores': all_scores,
            'labels': all_labels,
            'trust_scores': np.array(all_trust_scores)
        }
        
        return self.test_results

    def explain_detections(self, batch_data, feature_names: List[str] = None,
                           threshold: float = 0.5) -> Dict:
        """
        Explain model decisions for detected malicious nodes in a batch.

        Args:
            batch_data: PyG batch object with x, edge_index and optional edge_attr
            feature_names: Optional node feature names for human-readable output
            threshold: Malicious probability threshold

        Returns:
            Dictionary with per-node explanations and debug insights
        """
        self.model.eval()

        x = batch_data.x.to(self.device)
        edge_index = batch_data.edge_index.to(self.device)
        edge_attr = batch_data.edge_attr.to(self.device) if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None else None

        temporal_sequence = None
        if hasattr(batch_data, 'temporal_embedding') and batch_data.temporal_embedding is not None:
            temporal_sequence = batch_data.temporal_embedding.to(self.device)

        result = self.explainer.explain_batch(
            node_features=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            temporal_sequence=temporal_sequence,
            feature_names=feature_names,
            threshold=threshold,
        )

        # Convert dataclass explanations to plain dictionaries for serialization.
        serializable = {
            'node_explanations': [
                {
                    'node_id': item.node_id,
                    'predicted_label': item.predicted_label,
                    'malicious_probability': item.malicious_probability,
                    'trust_score': item.trust_score,
                    'influential_neighbors': item.influential_neighbors,
                    'top_feature_contributions': item.top_feature_contributions,
                    'explanation_text': item.explanation_text,
                }
                for item in result['node_explanations']
            ],
            'debug_insights': result['debug_insights'],
        }
        return serializable
    
    @torch.no_grad()
    def evaluate_robustness(self, test_loader, num_perturbations: int = 10) -> Dict:
        """
        Evaluate model robustness under input perturbations.
        
        Args:
            test_loader: Test data loader
            num_perturbations: Number of perturbation levels
            
        Returns:
            Robustness analysis
        """
        robustness_results = {}
        
        for perturb_level in np.linspace(0, 0.5, num_perturbations):
            perturbed_preds = []
            perturbed_labels = []
            
            for batch_data in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Add perturbation to node features
                x_perturbed = batch_data.x + (
                    torch.randn_like(batch_data.x) * perturb_level
                )
                
                output = self.model(
                    x_perturbed,
                    batch_data.edge_index,
                    batch_data.edge_attr,
                    batch_data.temporal_embedding if hasattr(batch_data, 'temporal_embedding') else None
                )
                
                preds = output['predicted_labels'].cpu().numpy()
                perturbed_preds.extend(preds)
                perturbed_labels.extend(batch_data.y.cpu().numpy())
            
            # Compute accuracy under perturbation
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(perturbed_labels, perturbed_preds)
            robustness_results[f'perturbation_{perturb_level:.2f}'] = acc
        
        return robustness_results
