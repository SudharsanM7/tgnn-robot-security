"""
Training script for temporal hybrid GNN pipeline.

Usage:
    python experiments/train_temporal_pipeline.py \
        --dataset_path outputs/simulation_dataset.pkl \
        --comm_log_path outputs/comm_log.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.graph.graph_constructor import TemporalGraphConstructor
from src.model.hybrid_gnn import HybridTemporalGNN
from src.training.temporal_training_pipeline import TemporalTrainingConfig, TemporalTrainer


def load_pickle(path: Path) -> Dict:
    with path.open("rb") as file:
        return pickle.load(file)


def load_graphs(dataset_path: Path, comm_log_path: Optional[Path], window_size: int, window_stride: int):
    dataset = load_pickle(dataset_path)
    state_log = dataset.get("state_log", [])

    communication_log = None
    if comm_log_path is not None and comm_log_path.exists():
        comm_data = load_pickle(comm_log_path)
        communication_log = {
            "messages": comm_data.get("messages", []),
            "transmissions": comm_data.get("transmission_events", []),
        }

    constructor = TemporalGraphConstructor(window_size=window_size, window_stride=window_stride)
    graphs = constructor.construct_from_state_log(state_log, communication_log=communication_log)
    return graphs


def infer_feature_dims(graphs):
    if not graphs:
        raise ValueError("No graphs available for training")
    sample = graphs[0]
    in_channels = int(sample.x.shape[1])
    edge_attr_dim = int(sample.edge_attr.shape[1]) if getattr(sample, "edge_attr", None) is not None and sample.edge_attr.numel() > 0 else None
    return in_channels, edge_attr_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal hybrid GNN")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to simulation dataset pickle")
    parser.add_argument("--comm_log_path", type=str, default="", help="Optional communication log pickle path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--temporal_depth", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/temporal_hybrid")
    parser.add_argument("--log_dir", type=str, default="logs/temporal_hybrid")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    comm_log_path = Path(args.comm_log_path) if args.comm_log_path else None

    graphs = load_graphs(
        dataset_path=dataset_path,
        comm_log_path=comm_log_path,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )
    logger.info("Loaded {} graph snapshots", len(graphs))

    in_channels, edge_attr_dim = infer_feature_dims(graphs)

    model = HybridTemporalGNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        message_in_dim=32,
        message_embedding_dim=32,
        message_encoder_type="mlp",
        temporal_layers=1,
        gat_layers=2,
        num_heads=4,
        edge_attr_dim=edge_attr_dim,
        trust_alpha=0.85,
        trust_steps=3,
        dropout=0.2,
    )

    config = TemporalTrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        window_size=args.window_size,
        window_stride=args.window_stride,
        temporal_depth=args.temporal_depth,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    trainer = TemporalTrainer(model=model, config=config)
    results = trainer.train(graphs)

    logger.info("Training completed")
    logger.info("Best val F1: {}", results["summary"]["best_val_f1"])
    logger.info("Test metrics: {}", results["test"])


if __name__ == "__main__":
    main()
