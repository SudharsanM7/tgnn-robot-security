"""Compare validation-selected operating points for the TGNN and baseline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_validated_benchmark import load_scenario
from src.model.hybrid_gnn import HybridTemporalGNN

OUTPUT = ROOT / "outputs" / "research_validated"
SCENARIOS = OUTPUT / "scenarios_main"


def tgnn_scores(model, graphs):
    model.eval()
    labels, scores = [], []
    history = []
    with torch.no_grad():
        for graph in graphs:
            x = graph.x.float()
            edge_attr = graph.edge_attr.float() if graph.edge_attr.numel() else None
            history.append(x)
            sequence = torch.stack(history[-4:], dim=1)
            while sequence.shape[1] < 4:
                sequence = torch.cat([sequence[:, :1], sequence], dim=1)
            output = model(
                node_features=x,
                edge_index=graph.edge_index.long(),
                edge_attr=edge_attr,
                temporal_sequence=sequence,
            )
            labels.extend(graph.y.numpy().tolist())
            scores.extend(output["malicious_probability"].numpy().tolist())
    return np.asarray(labels), np.asarray(scores)


def operating_metrics(labels, scores, threshold):
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def main():
    manifest = json.loads((SCENARIOS / "scenario_manifest.json").read_text(encoding="utf-8"))
    entries = {item["scenario_id"]: item for item in manifest}
    ordered_ids = sorted(entries)
    train_ids, val_ids, test_ids = ordered_ids[:4], ordered_ids[4:5], ordered_ids[5:]

    train_graphs = [g for sid in train_ids for g in load_scenario(entries[sid])]
    val_graphs = [g for sid in val_ids for g in load_scenario(entries[sid])]
    test_graphs = [g for sid in test_ids for g in load_scenario(entries[sid])]
    train_x = np.concatenate([g.x.numpy() for g in train_graphs])
    train_y = np.concatenate([g.y.numpy() for g in train_graphs])
    val_x = np.concatenate([g.x.numpy() for g in val_graphs])
    val_y = np.concatenate([g.y.numpy() for g in val_graphs])
    test_x = np.concatenate([g.x.numpy() for g in test_graphs])
    test_y = np.concatenate([g.y.numpy() for g in test_graphs])

    model = HybridTemporalGNN(
        in_channels=7, hidden_dim=32, num_classes=2, message_in_dim=32,
        message_embedding_dim=16, gat_layers=2, num_heads=4,
        edge_attr_dim=6, dropout=0.1,
    )
    checkpoint = OUTPUT / "models" / "validated_tgnn" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    val_tgnn_y, val_tgnn_scores = tgnn_scores(model, val_graphs)
    test_tgnn_y, test_tgnn_scores = tgnn_scores(model, test_graphs)

    baseline = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    baseline.fit(train_x, train_y)
    val_baseline_scores = baseline.predict_proba(val_x)[:, 1]
    test_baseline_scores = baseline.predict_proba(test_x)[:, 1]

    thresholds = np.linspace(0.10, 0.90, 81)
    rows = []
    selected = {}
    for name, val_y_values, val_scores, test_y_values, test_scores in [
        ("HybridTemporalGNN", val_tgnn_y, val_tgnn_scores, test_tgnn_y, test_tgnn_scores),
        ("LogisticRegression", val_y, val_baseline_scores, test_y, test_baseline_scores),
    ]:
        val_rows = []
        for threshold in thresholds:
            row = operating_metrics(val_y_values, val_scores, threshold)
            row.update({"model": name, "split": "validation"})
            val_rows.append(row)
        rows.extend(val_rows)
        for threshold in thresholds:
            row = operating_metrics(test_y_values, test_scores, threshold)
            row.update({"model": name, "split": "test"})
            rows.append(row)
        best = max(val_rows, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
        selected[name] = {
            "threshold": best["threshold"],
            "validation": best,
            "test": operating_metrics(test_y_values, test_scores, best["threshold"]),
        }

    output = OUTPUT / "metrics" / "operating_point_analysis.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "split", "threshold", "accuracy", "precision", "recall", "f1", "specificity", "false_positive_rate", "mcc"])
        writer.writeheader()
        writer.writerows(rows)
    summary_path = OUTPUT / "metrics" / "operating_point_summary.json"
    summary_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
