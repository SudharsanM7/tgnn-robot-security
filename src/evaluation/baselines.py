"""
Baseline models for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNBaseline(nn.Module):
    """
    Simple GCN baseline for comparison.
    """
    
    def __init__(self, in_channels: int, hidden_dim: int, out_classes: int):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        self.gc1 = GCNConv(in_channels, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.gc3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_classes)
    
    def forward(self, x, edge_index, edge_attr=None, temporal_features=None):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.gc3(x, edge_index))
        logits = self.classifier(x)
        
        return {
            'logits': logits,
            'predictions': torch.softmax(logits, dim=1),
            'predicted_labels': torch.argmax(logits, dim=1),
            'embeddings': x
        }


class ClassicalMLBaseline:
    """
    Classical ML baselines using traditional features.
    """
    
    def __init__(self, method: str = "rf"):
        """
        Initialize classical ML baseline.
        
        Args:
            method: "rf" (RandomForest), "svm", "xgb"
        """
        if method == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        elif method == "svm":
            from sklearn.svm import SVC
            self.model = SVC(kernel='rbf', probability=True)
        elif method == "xgb":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(n_estimators=100, max_depth=6)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
    
    def fit(self, X_train, y_train):
        """Train baseline model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predict probabilities."""
        return self.model.predict_proba(X_test)
