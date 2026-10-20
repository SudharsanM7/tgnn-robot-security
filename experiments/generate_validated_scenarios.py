"""Generate independent scenario artifacts for leakage-controlled evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks.attack_engine import AdaptiveAttackEngine, AttackEngineConfig
from src.simulation.simulation_engine import SimulationConfig, SimulationEngine


def run_scenario(output_dir: Path, scenario_id: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    config = SimulationConfig()
    config.num_robots = args.num_robots
    config.simulation_time = args.timesteps * config.timestep
    config.log_frequency = 1
    config.seed = seed
    config.packet_loss_rate = args.packet_loss_rate
    config.communication_range = args.communication_range

    engine = SimulationEngine(config)
    engine.scenario_id = scenario_id
    engine.communication_manager.scenario_id = scenario_id
    attack_config = AttackEngineConfig(
        attack_ratio=args.attack_ratio,
        base_intensity=args.attack_intensity,
        warmup_steps=5,
        min_interval=1,
        max_interval=3,
        duration_steps=max(args.timesteps, 20),
        seed=seed,
    )
    attack_engine = AdaptiveAttackEngine(engine, config=attack_config, seed=seed)

    def apply_attacks(simulation: SimulationEngine) -> None:
        attack_engine.step(simulation.timestep_count)

    elapsed = engine.run(apply_attacks=apply_attacks)
    dataset_path = output_dir / f"{scenario_id}_dataset.pkl"
    communication_path = output_dir / f"{scenario_id}_communication.pkl"
    engine.export_dataset(str(dataset_path))
    engine.communication_manager.export_communication_log(str(communication_path))

    observed_attack_types = sorted({str(event.get("attack_type")) for event in attack_engine.attack_log})
    metadata = {
        "scenario_id": scenario_id,
        "seed": seed,
        "num_robots": config.num_robots,
        "timesteps": engine.total_timesteps,
        "elapsed_seconds": elapsed,
        "attackers": attack_engine.get_attacker_ids(),
        "attack_types_observed": observed_attack_types,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "communication": str(communication_path.relative_to(ROOT)),
        "state_count": len(engine.state_log),
        "communication_stats": engine.communication_manager.get_statistics(),
    }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/research_validated/scenarios")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--num-robots", type=int, default=20)
    parser.add_argument("--attack-ratio", type=float, default=0.2)
    parser.add_argument("--attack-intensity", type=float, default=0.6)
    parser.add_argument("--packet-loss-rate", type=float, default=0.05)
    parser.add_argument("--communication-range", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, seed in enumerate(args.seeds):
        scenario_id = f"scenario_{index:03d}_seed_{seed}"
        print(f"running {scenario_id}", flush=True)
        manifest.append(run_scenario(output_dir, scenario_id, seed, args))
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

