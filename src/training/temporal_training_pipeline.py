"""
Temporal training pipeline for hybrid GNN node classification.

This pipeline provides:
- Temporal train/val/test split over graph sequences
- Multi-objective optimization
- Adam optimizer with early stopping
- Metrics logging and checkpointing
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import Adam
from torch_geometric.data import Data


@dataclass
class TemporalTrainingConfig:
    """Configuration for temporal model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 80
    patience: int = 10
    min_delta: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    window_size: int = 6
    window_stride: int = 1
    temporal_depth: int = 4
    cls_loss_weight: float = 1.0
    temporal_loss_weight: float = 0.2
    trust_loss_weight: float = 0.3
    detection_threshold: float = 0.5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cpu"


class TemporalWindowDataset:
    """Simple temporal window dataset over graph snapshots."""

    def __init__(self, graphs: Sequence[Data], window_size: int, stride: int = 1):
        self.graphs = list(graphs)
        self.window_size = max(1, window_size)
        self.stride = max(1, stride)
        self.windows = self._build_windows()

    def _build_windows(self) -> List[List[Data]]:
        windows: List[List[Data]] = []
        total = len(self.graphs)
        if total < self.window_size:
            return windows

        for start in range(0, total - self.window_size + 1, self.stride):
            windows.append(self.graphs[start : start + self.window_size])
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> List[Data]:
        return self.windows[index]


class MetricsLogger:
    """Writes training metrics to JSONL and keeps in-memory history."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.jsonl"
        self.history: List[Dict] = []

    def log(self, record: Dict) -> None:
        self.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def save_summary(self, summary: Dict) -> None:
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss for temporal malicious-node detection.

    Components:
    - Classification loss
    - Temporal consistency loss
    - Trust regularization loss
    """

    def __init__(self, cls_weight: float = 1.0, temporal_weight: float = 0.2, trust_weight: float = 0.3):
        super().__init__()
        self.cls_weight = cls_weight
        self.temporal_weight = temporal_weight
        self.trust_weight = trust_weight
        self.classification = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_seq: Sequence[torch.Tensor],
        trust_seq: Sequence[torch.Tensor],
        labels_seq: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not logits_seq:
            raise ValueError("logits_seq cannot be empty")

        cls_losses = [self.classification(logits, labels) for logits, labels in zip(logits_seq, labels_seq)]
        cls_loss = torch.stack(cls_losses).mean()

        # Temporal consistency over logits distribution.
        temporal_terms = []
        for idx in range(1, len(logits_seq)):
            prev_probs = F.softmax(logits_seq[idx - 1], dim=-1)
            cur_probs = F.softmax(logits_seq[idx], dim=-1)
            temporal_terms.append(F.mse_loss(cur_probs, prev_probs))
        temporal_loss = torch.stack(temporal_terms).mean() if temporal_terms else torch.tensor(0.0, device=cls_loss.device)

        # Trust regularization: benign nodes high trust, malicious nodes low trust.
        trust_terms = []
        for trust_scores, labels in zip(trust_seq, labels_seq):
            target = (1.0 - labels.float()).unsqueeze(-1)
            trust_terms.append(F.mse_loss(trust_scores, target))
        trust_loss = torch.stack(trust_terms).mean() if trust_terms else torch.tensor(0.0, device=cls_loss.device)

        total_loss = (
            self.cls_weight * cls_loss
            + self.temporal_weight * temporal_loss
            + self.trust_weight * trust_loss
        )

        return total_loss, {
            "classification_loss": float(cls_loss.detach().cpu().item()),
            "temporal_consistency_loss": float(temporal_loss.detach().cpu().item()),
            "trust_regularization_loss": float(trust_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }


class EarlyStopping:
    """Early stopping based on a monitored validation score."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class TemporalTrainer:
    """Full training pipeline for temporal graph node classification."""

    def __init__(self, model: nn.Module, config: TemporalTrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = torch.device(config.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = MultiObjectiveLoss(
            cls_weight=config.cls_loss_weight,
            temporal_weight=config.temporal_loss_weight,
            trust_weight=config.trust_loss_weight,
        )
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_logger = MetricsLogger(config.log_dir)

    @staticmethod
    def temporal_split(
        graphs: Sequence[Data],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        total = len(graphs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_graphs = list(graphs[:train_end])
        val_graphs = list(graphs[train_end:val_end])
        test_graphs = list(graphs[val_end:])
        return train_graphs, val_graphs, test_graphs

    @staticmethod
    def scenario_split(
        graphs: Sequence[Data],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """Split complete simulation scenarios without crossing run boundaries."""
        scenario_ids = [str(getattr(graph, "scenario_id", "unknown")) for graph in graphs]
        unique_ids = list(dict.fromkeys(scenario_ids))
        if len(unique_ids) < 3 or "unknown" in unique_ids:
            raise ValueError(
                "Scenario-level evaluation requires at least three explicitly named "
                "scenario_id values; the current data cannot support an independent split."
            )

        train_count = max(1, int(len(unique_ids) * train_ratio))
        val_count = max(1, int(len(unique_ids) * val_ratio))
        if train_count + val_count >= len(unique_ids):
            val_count = 1
            train_count = len(unique_ids) - 2

        train_ids = set(unique_ids[:train_count])
        val_ids = set(unique_ids[train_count:train_count + val_count])
        test_ids = set(unique_ids[train_count + val_count:])
        partitions = (
            [graph for graph, sid in zip(graphs, scenario_ids) if sid in train_ids],
            [graph for graph, sid in zip(graphs, scenario_ids) if sid in val_ids],
            [graph for graph, sid in zip(graphs, scenario_ids) if sid in test_ids],
        )
        if any(not partition for partition in partitions):
            raise ValueError("Scenario-level split produced an empty partition.")
        return partitions

    def _build_windows(self, graphs: Sequence[Data]) -> TemporalWindowDataset:
        grouped: List[List[Data]] = []
        current: List[Data] = []
        previous_scenario: Optional[str] = None
        for graph in graphs:
            scenario_id = str(getattr(graph, "scenario_id", "unknown"))
            if current and scenario_id != previous_scenario:
                grouped.append(current)
                current = []
            current.append(graph)
            previous_scenario = scenario_id
        if current:
            grouped.append(current)

        windows: List[List[Data]] = []
        for group in grouped:
            dataset = TemporalWindowDataset(
                graphs=group,
                window_size=self.config.window_size,
                stride=self.config.window_stride,
            )
            windows.extend(dataset.windows)
        result = TemporalWindowDataset([], self.config.window_size, self.config.window_stride)
        result.windows = windows
        return result

    def _make_temporal_sequence(self, window: Sequence[Data], index: int) -> torch.Tensor:
        start = max(0, index - self.config.temporal_depth + 1)
        sequence_graphs = window[start : index + 1]

        tensors = [graph.x.float().to(self.device) for graph in sequence_graphs]
        while len(tensors) < self.config.temporal_depth:
            tensors.insert(0, tensors[0])
        stacked = torch.stack(tensors[-self.config.temporal_depth :], dim=1)
        return stacked

    def _forward_window(self, window: Sequence[Data]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        logits_seq: List[torch.Tensor] = []
        trust_seq: List[torch.Tensor] = []
        labels_seq: List[torch.Tensor] = []

        prediction_list: List[np.ndarray] = []
        probability_list: List[np.ndarray] = []
        timestep_labels: List[np.ndarray] = []
        timestep_predictions: List[np.ndarray] = []
        timestep_values: List[int] = []

        for idx, graph in enumerate(window):
            x = graph.x.float().to(self.device)
            edge_index = graph.edge_index.long().to(self.device)
            edge_attr = graph.edge_attr.float().to(self.device) if getattr(graph, "edge_attr", None) is not None else None
            y = graph.y.long().to(self.device)

            temporal_sequence = self._make_temporal_sequence(window, idx)

            output = self.model(
                node_features=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                temporal_sequence=temporal_sequence,
                message_features=None,
                message_sender_index=None,
                message_receiver_index=None,
                return_attention=False,
            )

            logits = output["logits"]
            trust_scores = output["trust_scores"]
            malicious_probability = output["malicious_probability"]

            logits_seq.append(logits)
            trust_seq.append(trust_scores)
            labels_seq.append(y)

            predictions = (malicious_probability >= self.config.detection_threshold).long()
            pred_np = predictions.detach().cpu().numpy()
            label_np = y.detach().cpu().numpy()

            prediction_list.append(pred_np)
            probability_list.append(malicious_probability.detach().cpu().numpy())

            timestep_labels.append(label_np)
            timestep_predictions.append(pred_np)
            timestep_values.append(int(getattr(graph, "timestep", idx)))

        latency_stats = self._compute_detection_latency(
            timestep_labels=timestep_labels,
            timestep_predictions=timestep_predictions,
            timesteps=timestep_values,
        )
        return logits_seq, trust_seq, labels_seq, prediction_list, probability_list, latency_stats

    def _compute_detection_latency(
        self,
        timestep_labels: Sequence[np.ndarray],
        timestep_predictions: Sequence[np.ndarray],
        timesteps: Sequence[int],
    ) -> Dict[str, float]:
        if not timestep_labels or not timestep_predictions:
            return {"detection_latency": float("inf")}

        num_nodes = timestep_labels[0].shape[0]
        attack_start: Dict[int, int] = {}
        detection_time: Dict[int, int] = {}

        for step_idx, (labels, preds) in enumerate(zip(timestep_labels, timestep_predictions)):
            ts = int(timesteps[step_idx]) if step_idx < len(timesteps) else step_idx
            for node_id in range(num_nodes):
                if labels[node_id] == 1 and node_id not in attack_start:
                    attack_start[node_id] = ts
                if preds[node_id] == 1 and node_id not in detection_time:
                    detection_time[node_id] = ts

        latencies: List[float] = []
        for node_id, start_ts in attack_start.items():
            detect_ts = detection_time.get(node_id)
            if detect_ts is None:
                continue
            latencies.append(float(max(0, detect_ts - start_ts)))

        if not latencies:
            return {"detection_latency": float("inf")}
        return {"detection_latency": float(np.mean(latencies))}

    def _aggregate_metrics(
        self,
        all_labels: List[np.ndarray],
        all_predictions: List[np.ndarray],
        all_probabilities: List[np.ndarray],
        loss_info: Dict[str, float],
        latency_info: Dict[str, float],
    ) -> Dict[str, float]:
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        predictions = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)

        if labels.size == 0:
            return {
                **loss_info,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "detection_latency": float("inf"),
            }

        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "detection_latency": float(latency_info.get("detection_latency", float("inf"))),
        }
        metrics.update(loss_info)
        return metrics

    def _run_epoch(self, dataset: TemporalWindowDataset, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        all_labels: List[np.ndarray] = []
        all_predictions: List[np.ndarray] = []
        all_probabilities: List[np.ndarray] = []
        total_loss_value = 0.0
        num_windows = max(1, len(dataset))
        latency_values: List[float] = []

        for window in dataset.windows:
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits_seq, trust_seq, labels_seq, preds_seq, probs_seq, latency_info = self._forward_window(window)
                loss, loss_terms = self.loss_fn(logits_seq, trust_seq, labels_seq)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss_value += float(loss.detach().cpu().item())
            latency_values.append(float(latency_info.get("detection_latency", float("inf"))))

            for labels in labels_seq:
                all_labels.append(labels.detach().cpu().numpy())
            all_predictions.extend(preds_seq)
            all_probabilities.extend(probs_seq)

        mean_latency = float(np.mean([v for v in latency_values if np.isfinite(v)])) if any(np.isfinite(v) for v in latency_values) else float("inf")

        loss_summary = {
            "total_loss": total_loss_value / num_windows,
            "classification_loss": float(loss_terms["classification_loss"]),
            "temporal_consistency_loss": float(loss_terms["temporal_consistency_loss"]),
            "trust_regularization_loss": float(loss_terms["trust_regularization_loss"]),
        }

        return self._aggregate_metrics(
            all_labels=all_labels,
            all_predictions=all_predictions,
            all_probabilities=all_probabilities,
            loss_info=loss_summary,
            latency_info={"detection_latency": mean_latency},
        )

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
        }

        last_path = self.checkpoint_dir / "last.pt"
        torch.save(payload, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_path)

    def train(self, graphs: Sequence[Data]) -> Dict[str, Dict[str, float]]:
        train_graphs, val_graphs, test_graphs = self.scenario_split(
            graphs,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
        )

        train_dataset = self._build_windows(train_graphs)
        val_dataset = self._build_windows(val_graphs)
        test_dataset = self._build_windows(test_graphs)

        if len(train_dataset) == 0:
            raise ValueError("Training dataset has zero temporal windows. Increase data length or reduce window size.")

        best_val_f1 = -float("inf")
        best_epoch = -1

        for epoch in range(self.config.num_epochs):
            train_metrics = self._run_epoch(train_dataset, train=True)
            val_metrics = self._run_epoch(val_dataset, train=False) if len(val_dataset) > 0 else train_metrics

            log_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.metrics_logger.log(log_record)

            logger.info(
                "Epoch {} | train_loss {:.4f} val_f1 {:.4f} val_latency {}",
                epoch,
                train_metrics["total_loss"],
                val_metrics["f1"],
                val_metrics["detection_latency"],
            )

            is_best = val_metrics["f1"] > best_val_f1 + self.config.min_delta
            if is_best:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            self._save_checkpoint(epoch, val_metrics, is_best=False)

            if self.early_stopping.step(val_metrics["f1"]):
                logger.info("Early stopping at epoch {}", epoch)
                break

        # Evaluate best checkpoint on test split.
        best_checkpoint = self.checkpoint_dir / "best.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self._run_epoch(test_dataset, train=False) if len(test_dataset) > 0 else {}

        summary = {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
            "final_test": test_metrics,
        }
        self.metrics_logger.save_summary(summary)

        return {
            "summary": summary,
            "last_train": train_metrics,
            "last_val": val_metrics,
            "test": test_metrics,
        }
