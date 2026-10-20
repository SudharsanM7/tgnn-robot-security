"""Train and independently evaluate the corrected TGNN on scenario artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.graph_constructor import TemporalGraphConstructor
from src.model.hybrid_gnn import HybridTemporalGNN
from src.training.temporal_training_pipeline import TemporalTrainer, TemporalTrainingConfig


OUTPUT = ROOT / "outputs" / "research_validated"
SCENARIOS = OUTPUT / "scenarios_main"


def load_scenario(entry: Dict) -> List:
    with (ROOT / entry["dataset"]).open("rb") as file:
        dataset = __import__("pickle").load(file)
    with (ROOT / entry["communication"]).open("rb") as file:
        communication = __import__("pickle").load(file)
    constructor = TemporalGraphConstructor(window_size=6, window_stride=1)
    graphs = constructor.construct_from_state_log(
        dataset["state_log"],
        communication_log={
            "messages": communication.get("messages", []),
            "transmissions": communication.get("transmission_events", []),
        },
    )
    return graphs


def metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def model_predictions(model: torch.nn.Module, graphs: Sequence, device: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    model.eval()
    labels, predictions, rows = [], [], []
    history: List[torch.Tensor] = []
    with torch.no_grad():
        for graph in graphs:
            x = graph.x.float().to(device)
            edge_index = graph.edge_index.long().to(device)
            edge_attr = graph.edge_attr.float().to(device) if graph.edge_attr.numel() else None
            history.append(x)
            sequence = torch.stack(history[-4:], dim=1)
            while sequence.shape[1] < 4:
                sequence = torch.cat([sequence[:, :1], sequence], dim=1)
            output = model(
                node_features=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                temporal_sequence=sequence,
            )
            pred = (output["malicious_probability"] >= 0.5).long().cpu().numpy()
            truth = graph.y.cpu().numpy()
            labels.extend(truth.tolist())
            predictions.extend(pred.tolist())
            scenario_id = str(graph.scenario_id)
            timestep = int(graph.timestep)
            for robot_id, (target, decision) in enumerate(zip(truth, pred)):
                rows.append({
                    "scenario_id": scenario_id,
                    "timestep": timestep,
                    "robot_id": robot_id,
                    "label": int(target),
                    "prediction": int(decision),
                    "model": "HybridTemporalGNN",
                })
    return np.asarray(labels), np.asarray(predictions), rows


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    manifest = json.loads((SCENARIOS / "scenario_manifest.json").read_text(encoding="utf-8"))
    entries = {item["scenario_id"]: item for item in manifest}
    ordered_ids = sorted(entries)
    train_ids, val_ids, test_ids = ordered_ids[:4], ordered_ids[4:5], ordered_ids[5:]
    all_graphs = []
    for scenario_id in ordered_ids:
        all_graphs.extend(load_scenario(entries[scenario_id]))

    config = TemporalTrainingConfig(
        num_epochs=4,
        patience=2,
        train_ratio=4 / 11,
        val_ratio=1 / 11,
        window_size=6,
        window_stride=1,
        temporal_depth=4,
        checkpoint_dir=str(OUTPUT / "models" / "validated_tgnn"),
        log_dir=str(OUTPUT / "logs" / "validated_tgnn"),
        device="cpu",
    )
    model = HybridTemporalGNN(
        in_channels=7,
        hidden_dim=32,
        num_classes=2,
        message_in_dim=32,
        message_embedding_dim=16,
        gat_layers=2,
        num_heads=4,
        edge_attr_dim=6,
        dropout=0.1,
    )
    start = time.perf_counter()
    result = TemporalTrainer(model, config).train(all_graphs)
    duration = time.perf_counter() - start

    checkpoint = OUTPUT / "models" / "validated_tgnn" / "best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    test_graphs = [graph for graph in all_graphs if str(graph.scenario_id) in test_ids]
    labels, predictions, rows = model_predictions(model, test_graphs, "cpu")
    model_metrics = metrics(labels, predictions)

    pred_path = OUTPUT / "predictions" / "tgnn_test_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    train_rows = []
    for scenario_id in train_ids:
        train_rows.extend(load_scenario(entries[scenario_id]))
    val_rows = []
    for scenario_id in val_ids:
        val_rows.extend(load_scenario(entries[scenario_id]))
    train_x = np.concatenate([graph.x.numpy() for graph in train_rows])
    train_y = np.concatenate([graph.y.numpy() for graph in train_rows])
    val_x = np.concatenate([graph.x.numpy() for graph in val_rows])
    val_y = np.concatenate([graph.y.numpy() for graph in val_rows])
    test_x = np.concatenate([graph.x.numpy() for graph in test_graphs])
    test_y = np.concatenate([graph.y.numpy() for graph in test_graphs])
    baseline = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    baseline.fit(train_x, train_y)
    val_prob = baseline.predict_proba(val_x)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    threshold = max(thresholds, key=lambda t: f1_score(val_y, val_prob >= t, zero_division=0))
    baseline_metrics = metrics(test_y, baseline.predict_proba(test_x)[:, 1] >= threshold)

    metric_rows = [
        {"model": "HybridTemporalGNN", "split": "test", **model_metrics, "threshold": 0.5},
        {"model": "LogisticRegression", "split": "test", **baseline_metrics, "threshold": float(threshold)},
    ]
    for scenario_id in test_ids:
        scenario_graphs = [graph for graph in test_graphs if str(graph.scenario_id) == scenario_id]
        scenario_labels, scenario_predictions, _ = model_predictions(model, scenario_graphs, "cpu")
        metric_rows.append({
            "model": "HybridTemporalGNN",
            "split": f"test:{scenario_id}",
            **metrics(scenario_labels, scenario_predictions),
            "threshold": 0.5,
        })
    with (OUTPUT / "metrics" / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {
        "train_scenarios": train_ids,
        "validation_scenarios": val_ids,
        "test_scenarios": test_ids,
        "training_duration_seconds": duration,
        "checkpoint_sha256": checkpoint_hash,
        "tgnn": model_metrics,
        "logistic_regression_baseline": baseline_metrics,
        "tgnn_per_scenario": [
            row for row in metric_rows
            if row["model"] == "HybridTemporalGNN" and row["split"].startswith("test:")
        ],
        "baseline_threshold_selected_on_validation": float(threshold),
        "trainer_summary": result["summary"],
    }
    (OUTPUT / "metrics" / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
