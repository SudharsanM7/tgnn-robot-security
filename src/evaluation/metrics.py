"""
Metrics module: Comprehensive evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, roc_curve
)
from typing import Dict, List, Tuple


class DetectionMetrics:
    """
    Computes comprehensive metrics for adversarial attack detection.
    """
    
    @staticmethod
    def compute_detection_metrics(true_labels: np.ndarray,
                                  predictions: np.ndarray,
                                  prediction_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute all detection metrics.
        
        Args:
            true_labels: Ground truth binary labels (0=benign, 1=adversarial)
            predictions: Predicted binary labels
            prediction_scores: Probability scores for positive class
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        metrics['precision'] = precision_score(true_labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(true_labels, predictions, zero_division=0)
        metrics['f1'] = f1_score(true_labels, predictions, zero_division=0)
        
        # ROC-AUC
        if len(np.unique(true_labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(true_labels, prediction_scores)
        else:
            metrics['roc_auc'] = 0.0
        
        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(
            true_labels, prediction_scores
        )
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        # Sensitivity and Specificity
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tp = np.sum((predictions == 1) & (true_labels == 1))
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews correlation coefficient
        mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if mcc_denom > 0:
            metrics['mcc'] = (tp * tn - fp * fn) / np.sqrt(mcc_denom)
        else:
            metrics['mcc'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_detection_latency(detections: List[Dict]) -> Dict[str, float]:
        """
        Compute detection latency statistics.
        
        Measures how quickly attacks are detected after they occur.
        
        Args:
            detections: List of detection events with timesteps
            
        Returns:
            Latency statistics
        """
        if not detections:
            return {
                'mean_latency': float('inf'),
                'median_latency': float('inf'),
                'max_latency': float('inf'),
                'min_latency': float('inf')
            }
        
        latencies = [d.get('latency_frames', 0) for d in detections]
        latencies = [l for l in latencies if l > 0]
        
        if not latencies:
            return {
                'mean_latency': float('inf'),
                'median_latency': float('inf'),
                'max_latency': float('inf'),
                'min_latency': float('inf')
            }
        
        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'max_latency': np.max(latencies),
            'min_latency': np.min(latencies)
        }
    
    @staticmethod
    def compute_robustness_metrics(predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   adversarial_strength: float) -> Dict[str, float]:
        """
        Compute robustness under varying attack strength.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            adversarial_strength: Attack strength parameter (0-1)
            
        Returns:
            Robustness metrics
        """
        metrics = DetectionMetrics.compute_detection_metrics(
            true_labels, predictions, predictions.astype(float)
        )
        
        # Add robustness indicator
        metrics['robustness_score'] = metrics['f1'] * (1 - adversarial_strength)
        
        return metrics


class EvaluationReport:
    """Generates comprehensive evaluation reports."""
    
    @staticmethod
    def generate_report(true_labels: np.ndarray,
                       predictions: np.ndarray,
                       prediction_scores: np.ndarray,
                       model_name: str = "Model") -> str:
        """
        Generate formatted evaluation report.
        
        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels
            prediction_scores: Prediction scores
            model_name: Name of model for reporting
            
        Returns:
            Formatted report string
        """
        metrics = DetectionMetrics.compute_detection_metrics(
            true_labels, predictions, prediction_scores
        )
        
        report = f"""
        ╔════════════════════════════════════════════╗
        ║  ADVERSARIAL DETECTION EVALUATION REPORT  ║
        ║  Model: {model_name:<27} ║
        ╠════════════════════════════════════════════╣
        ║  Classification Metrics:                   ║
        ║    • Accuracy:        {metrics['accuracy']:.4f}                ║
        ║    • Precision:       {metrics['precision']:.4f}                ║
        ║    • Recall:          {metrics['recall']:.4f}                ║
        ║    • F1-Score:        {metrics['f1']:.4f}                ║
        ║                                            ║
        ║  ROC & PR Metrics:                        ║
        ║    • ROC-AUC:         {metrics['roc_auc']:.4f}                ║
        ║    • PR-AUC:          {metrics['pr_auc']:.4f}                ║
        ║                                            ║
        ║  Additional Metrics:                      ║
        ║    • Sensitivity:     {metrics['sensitivity']:.4f}                ║
        ║    • Specificity:     {metrics['specificity']:.4f}                ║
        ║    • MCC:             {metrics['mcc']:.4f}                ║
        ╚════════════════════════════════════════════╝
        """
        
        return report
