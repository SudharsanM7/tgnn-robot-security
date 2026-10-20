"""
Adaptive adversarial attack engine for multi-robot communication networks.

This module generates time-dependent attack behavior, injects malicious
messages into the simulation, and produces labels for ML pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from ..simulation.communication import MessageType


class AdvancedAttackType(str, Enum):
    """Supported adaptive attack modes."""

    STEALTHY_BYZANTINE = "stealthy_byzantine"
    FLOODING_DOS = "flooding_dos"
    COLLUSION = "collusion"
    TOPOLOGY_MANIPULATION = "topology_manipulation"
    TRUST_EXPLOITATION = "trust_exploitation"


@dataclass
class AttackEngineConfig:
    """Configuration for the adaptive attack engine."""

    attack_ratio: float = 0.15
    base_intensity: float = 0.45
    max_intensity: float = 1.0
    min_intensity: float = 0.1
    warmup_steps: int = 6
    min_interval: int = 1
    max_interval: int = 5
    duration_steps: int = 40
    collusion_group_size: int = 3
    flooding_burst_factor: float = 3.0
    fake_edge_probability: float = 0.35
    hidden_node_probability: float = 0.2
    trust_threshold: float = 0.65
    history_window: int = 12
    detection_sensitivity: float = 0.15
    seed: int = 42


@dataclass
class AttackProfile:
    """State for a single attacker."""

    attacker_id: int
    start_timestep: int
    end_timestep: int
    warmup_steps: int
    next_attack_timestep: int
    current_intensity: float
    stealth_level: float
    attack_bias: float
    collusion_group: List[int] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_attack_timestep: int = -10**9


class AdaptiveAttackEngine:
    """
    Time-dependent adversarial attack engine.

    The engine adapts to node history, neighbor state, and trust levels.
    It can inject malicious messages, manipulate topology metadata, and
    generate node/message/edge labels for downstream training.
    """

    def __init__(
        self,
        simulation: Any,
        config: Optional[AttackEngineConfig] = None,
        attack_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.simulation = simulation
        self.config = config or AttackEngineConfig()
        if attack_ratio is not None:
            self.config.attack_ratio = attack_ratio
        if seed is not None:
            self.config.seed = seed

        self.rng = np.random.default_rng(self.config.seed)
        self.num_robots = self._get_num_robots(simulation)

        self.attacker_ids = self._select_attackers(self.num_robots, self.config.attack_ratio)
        self.victim_ids = [rid for rid in range(self.num_robots) if rid not in self.attacker_ids]

        self.profiles: Dict[int, AttackProfile] = {}
        self.attack_log: List[Dict[str, Any]] = []
        self.label_log: List[Dict[str, Any]] = []
        self.detection_feedback: Dict[int, float] = {attacker_id: 0.0 for attacker_id in self.attacker_ids}
        self.collusion_groups = self._build_collusion_groups()

        for attacker_id in self.attacker_ids:
            self.profiles[attacker_id] = self._build_profile(attacker_id)

        self._attach_compatibility_metadata(simulation)

    def _get_num_robots(self, simulation: Any) -> int:
        if hasattr(simulation, "num_robots"):
            return int(simulation.num_robots)
        if hasattr(simulation, "robots"):
            return len(simulation.robots)
        raise ValueError("Simulation object must expose num_robots or robots")

    def _select_attackers(self, num_robots: int, attack_ratio: float) -> List[int]:
        count = max(1, int(round(num_robots * attack_ratio)))
        count = min(count, num_robots)
        return sorted(self.rng.choice(num_robots, size=count, replace=False).tolist())

    def _build_collusion_groups(self) -> List[List[int]]:
        if len(self.attacker_ids) < 2:
            return []

        groups: List[List[int]] = []
        group_size = max(2, int(self.config.collusion_group_size))
        for start in range(0, len(self.attacker_ids), group_size):
            group = self.attacker_ids[start : start + group_size]
            if len(group) >= 2:
                groups.append(group)
        return groups

    def _build_profile(self, attacker_id: int) -> AttackProfile:
        start_timestep = int(self.rng.integers(0, max(1, self.config.warmup_steps + 1)))
        duration = int(self.rng.integers(max(5, self.config.duration_steps // 2), self.config.duration_steps + 1))
        warmup = int(self.rng.integers(0, max(1, self.config.warmup_steps + 1)))
        next_attack = start_timestep + warmup
        return AttackProfile(
            attacker_id=attacker_id,
            start_timestep=start_timestep,
            end_timestep=start_timestep + duration,
            warmup_steps=warmup,
            next_attack_timestep=next_attack,
            current_intensity=float(np.clip(self.config.base_intensity, self.config.min_intensity, self.config.max_intensity)),
            stealth_level=float(self.rng.uniform(0.45, 0.9)),
            attack_bias=float(self.rng.uniform(0.1, 0.9)),
        )

    def _attach_compatibility_metadata(self, simulation: Any) -> None:
        setattr(simulation, "_adversary_ids", list(self.attacker_ids))
        setattr(simulation, "attack_engine", self)

    def get_attacker_ids(self) -> List[int]:
        return list(self.attacker_ids)

    def get_victim_ids(self) -> List[int]:
        return list(self.victim_ids)

    def step(self, timestep: int, simulation: Optional[Any] = None) -> Dict[str, Any]:
        """Generate, inject, and label all attack activity for one timestep."""
        simulation = simulation or self.simulation
        context = self._build_context(simulation, timestep)

        events: List[Dict[str, Any]] = []
        for attacker_id in self.attacker_ids:
            profile = self.profiles[attacker_id]
            if not self._is_active(profile, timestep):
                continue

            if timestep < profile.next_attack_timestep:
                continue

            attack_type = self._select_attack_type(attacker_id, profile, context, timestep)
            event = self._build_attack_event(simulation, attacker_id, attack_type, context, timestep, profile)
            if event is None:
                continue

            self._apply_event(simulation, event, timestep)
            profile.last_attack_timestep = timestep
            profile.next_attack_timestep = timestep + self._compute_next_interval(profile, context[attacker_id])
            profile.current_intensity = event["intensity"]
            profile.history.append(event)
            events.append(event)

        labels = self.generate_labels(timestep, events)
        self.attack_log.extend(events)
        self.label_log.append(labels)

        logger.debug("Attack engine produced {} events at timestep {}", len(events), timestep)
        return {
            "timestep": timestep,
            "attacker_ids": list(self.attacker_ids),
            "events": events,
            "labels": labels,
        }

    def update_detection_feedback(self, detected_attackers: Iterable[int]) -> None:
        detected = set(detected_attackers)
        for attacker_id in self.attacker_ids:
            was_detected = 1.0 if attacker_id in detected else 0.0
            self.detection_feedback[attacker_id] = (
                0.85 * self.detection_feedback[attacker_id] + 0.15 * was_detected
            )

            profile = self.profiles[attacker_id]
            if was_detected:
                profile.stealth_level = float(np.clip(profile.stealth_level * 0.92, 0.1, 0.95))
                profile.current_intensity = float(np.clip(profile.current_intensity * 0.9, self.config.min_intensity, self.config.max_intensity))
            else:
                profile.stealth_level = float(np.clip(profile.stealth_level * 1.02, 0.1, 0.95))
                profile.current_intensity = float(np.clip(profile.current_intensity * 1.01, self.config.min_intensity, self.config.max_intensity))

    def generate_labels(self, timestep: int, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        node_labels = {robot_id: 0 for robot_id in range(self.num_robots)}
        message_labels: List[Dict[str, Any]] = []
        edge_labels: List[Dict[str, Any]] = []

        for event in events:
            is_malicious = 1 if event.get("is_malicious", False) else 0
            sender_id = event["sender_id"]
            node_labels[sender_id] = max(node_labels[sender_id], is_malicious)

            for receiver_id in event.get("receiver_ids", []):
                if receiver_id is None:
                    continue
                node_labels[receiver_id] = max(node_labels[receiver_id], is_malicious)

            for message in event.get("messages", []):
                message_labels.append(
                    {
                        "timestep": timestep,
                        "msg_type": message["msg_type"],
                        "sender_id": message["sender_id"],
                        "receiver_id": message.get("receiver_id"),
                        "attack_type": event.get("attack_type"),
                        "label": is_malicious,
                    }
                )

            for edge in event.get("edge_mutations", []):
                edge_labels.append(
                    {
                        "timestep": timestep,
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "attack_type": event.get("attack_type"),
                        "label": 1,
                        "mutation": edge.get("mutation"),
                    }
                )

        return {
            "timestep": timestep,
            "node_labels": node_labels,
            "message_labels": message_labels,
            "edge_labels": edge_labels,
            "malicious_nodes": [node_id for node_id, label in node_labels.items() if label == 1],
        }

    def _is_active(self, profile: AttackProfile, timestep: int) -> bool:
        return profile.start_timestep <= timestep <= profile.end_timestep

    def _build_context(self, simulation: Any, timestep: int) -> Dict[int, Dict[str, Any]]:
        context: Dict[int, Dict[str, Any]] = {}
        for robot_id in range(self.num_robots):
            robot = simulation.robots[robot_id]
            neighbors = self._get_neighbors(simulation, robot_id)
            neighbor_states = [simulation.robots[nid].state.value for nid in neighbors if nid in simulation.robots]
            trust_values = list(robot.trust_scores.values())
            trust_mean = float(np.mean(trust_values)) if trust_values else 1.0
            message_history = robot.received_messages_history[-self.config.history_window :]
            sent_history = robot.sent_message_history[-self.config.history_window :]

            context[robot_id] = {
                "timestep": timestep,
                "position": robot.position.copy(),
                "velocity": robot.velocity.copy(),
                "state": robot.state.value,
                "task_state": robot.task_state.value,
                "neighbor_ids": neighbors,
                "neighbor_states": neighbor_states,
                "neighbor_count": len(neighbors),
                "trust_mean": trust_mean,
                "trust_values": trust_values,
                "message_history": message_history,
                "sent_history": sent_history,
                "incoming_load": len(robot.message_queue),
                "outgoing_load": len(robot.outgoing_messages),
                "recent_malicious_ratio": self._malicious_ratio(message_history),
                "recent_activity": len(message_history) + len(sent_history),
                "shared_attacker_neighbors": self._shared_attacker_neighbors(robot_id, simulation),
            }
        return context

    def _get_neighbors(self, simulation: Any, robot_id: int) -> List[int]:
        if hasattr(simulation, "neighborhood_manager"):
            robot = simulation.robots[robot_id]
            try:
                return list(simulation.neighborhood_manager.get_neighbors(robot_id, robot.position))
            except Exception:
                return []

        if hasattr(simulation, "communication_graph"):
            graph = simulation.communication_graph
            if hasattr(graph, "neighbors") and robot_id in graph:
                return list(graph.neighbors(robot_id))

        return [rid for rid in range(self.num_robots) if rid != robot_id]

    def _shared_attacker_neighbors(self, robot_id: int, simulation: Any) -> int:
        neighbors = set(self._get_neighbors(simulation, robot_id))
        return sum(1 for attacker_id in self.attacker_ids if attacker_id != robot_id and attacker_id in neighbors)

    def _malicious_ratio(self, messages: Sequence[Dict[str, Any]]) -> float:
        if not messages:
            return 0.0
        malicious = sum(1 for message in messages if message.get("malicious", False) or message.get("attack_type"))
        return malicious / max(len(messages), 1)

    def _select_attack_type(
        self,
        attacker_id: int,
        profile: AttackProfile,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
    ) -> AdvancedAttackType:
        features = context[attacker_id]
        trust_mean = features["trust_mean"]
        neighbor_count = features["neighbor_count"]
        shared_attackers = features["shared_attacker_neighbors"]
        activity = features["recent_activity"]
        malicious_ratio = features["recent_malicious_ratio"]
        progress = self._progress(profile, timestep)

        scores = {
            AdvancedAttackType.TRUST_EXPLOITATION: 0.2 + 0.5 * trust_mean + 0.2 * (1.0 - progress),
            AdvancedAttackType.STEALTHY_BYZANTINE: 0.2 + 0.35 * trust_mean + 0.25 * profile.stealth_level + 0.2 * progress,
            AdvancedAttackType.FLOODING_DOS: 0.15 + 0.35 * (neighbor_count / max(self.num_robots - 1, 1)) + 0.3 * malicious_ratio + 0.1 * activity,
            AdvancedAttackType.COLLUSION: 0.1 + 0.45 * shared_attackers + 0.2 * trust_mean + 0.15 * profile.attack_bias,
            AdvancedAttackType.TOPOLOGY_MANIPULATION: 0.15 + 0.4 * (neighbor_count / max(self.num_robots - 1, 1)) + 0.2 * profile.attack_bias + 0.15 * progress,
        }

        if timestep < profile.start_timestep + profile.warmup_steps:
            return AdvancedAttackType.TRUST_EXPLOITATION

        if features["trust_mean"] >= self.config.trust_threshold and progress < 0.5:
            scores[AdvancedAttackType.STEALTHY_BYZANTINE] += 0.25

        if shared_attackers >= 1:
            scores[AdvancedAttackType.COLLUSION] += 0.35

        if features["incoming_load"] >= max(2, neighbor_count):
            scores[AdvancedAttackType.FLOODING_DOS] += 0.3

        if features["state"] != "normal":
            scores[AdvancedAttackType.STEALTHY_BYZANTINE] += 0.1

        best_type = max(scores.items(), key=lambda item: item[1])[0]

        if best_type == AdvancedAttackType.COLLUSION and self._active_collusion_group(attacker_id) is None:
            return AdvancedAttackType.STEALTHY_BYZANTINE

        return best_type

    def _active_collusion_group(self, attacker_id: int) -> Optional[List[int]]:
        for group in self.collusion_groups:
            if attacker_id in group:
                return group
        return None

    def _progress(self, profile: AttackProfile, timestep: int) -> float:
        span = max(profile.end_timestep - profile.start_timestep, 1)
        return float(np.clip((timestep - profile.start_timestep) / span, 0.0, 1.0))

    def _compute_next_interval(self, profile: AttackProfile, features: Dict[str, Any]) -> int:
        detection_pressure = self.detection_feedback.get(profile.attacker_id, 0.0)
        base_interval = self.rng.integers(self.config.min_interval, self.config.max_interval + 1)
        activity_factor = max(1.0, features["recent_activity"] / max(self.config.history_window, 1))
        interval = int(round(base_interval * (1.0 + detection_pressure * 3.0 + activity_factor * 0.5)))
        return max(self.config.min_interval, interval)

    def _build_attack_event(
        self,
        simulation: Any,
        attacker_id: int,
        attack_type: AdvancedAttackType,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Optional[Dict[str, Any]]:
        if attack_type == AdvancedAttackType.TRUST_EXPLOITATION and timestep < profile.start_timestep + profile.warmup_steps:
            return self._build_normal_cover_event(simulation, attacker_id, context, timestep, profile)

        if attack_type == AdvancedAttackType.STEALTHY_BYZANTINE:
            return self._build_stealth_byzantine_event(simulation, attacker_id, context, timestep, profile)

        if attack_type == AdvancedAttackType.FLOODING_DOS:
            return self._build_flooding_event(simulation, attacker_id, context, timestep, profile)

        if attack_type == AdvancedAttackType.COLLUSION:
            return self._build_collusion_event(simulation, attacker_id, context, timestep, profile)

        if attack_type == AdvancedAttackType.TOPOLOGY_MANIPULATION:
            return self._build_topology_event(simulation, attacker_id, context, timestep, profile)

        return self._build_stealth_byzantine_event(simulation, attacker_id, context, timestep, profile)

    def _build_normal_cover_event(
        self,
        simulation: Any,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Dict[str, Any]:
        robot = simulation.robots[attacker_id]
        target_ids = self._select_targets(attacker_id, context, prefer_trusted=True)
        messages = []
        for target_id in target_ids:
            messages.append(
                self._make_message(
                    sender_id=attacker_id,
                    receiver_id=target_id,
                    msg_type=MessageType.POSITION_UPDATE,
                    payload={
                        "position": robot.position.tolist(),
                        "velocity": robot.velocity.tolist(),
                        "attack_type": AdvancedAttackType.TRUST_EXPLOITATION.value,
                        "malicious": False,
                    },
                    timestep=timestep,
                    malicious=False,
                )
            )

        return {
            "attacker_id": attacker_id,
            "attack_type": AdvancedAttackType.TRUST_EXPLOITATION.value,
            "phase": "warmup",
            "sender_id": attacker_id,
            "receiver_ids": target_ids,
            "messages": messages,
            "edge_mutations": [],
            "is_malicious": False,
            "intensity": float(np.clip(profile.current_intensity * 0.5, self.config.min_intensity, self.config.max_intensity)),
        }

    def _build_stealth_byzantine_event(
        self,
        simulation: Any,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Dict[str, Any]:
        robot = simulation.robots[attacker_id]
        progress = self._progress(profile, timestep)
        intensity = float(np.clip(self.config.base_intensity + 0.6 * progress + 0.3 * profile.stealth_level, self.config.min_intensity, self.config.max_intensity))
        target_ids = self._select_targets(attacker_id, context, prefer_trusted=True, max_targets=max(1, len(context[attacker_id]["neighbor_ids"]) // 2 or 1))

        deviation = self.rng.normal(0.0, 0.2 + intensity * 0.8, size=2)
        fake_position = (robot.position + deviation).tolist()
        fake_velocity = (robot.velocity + self.rng.normal(0.0, 0.05 + 0.15 * intensity, size=2)).tolist()

        messages = [
            self._make_message(
                sender_id=attacker_id,
                receiver_id=target_id,
                msg_type=MessageType.POSITION_UPDATE,
                payload={
                    "position": fake_position,
                    "velocity": fake_velocity,
                    "deviation_norm": float(np.linalg.norm(deviation)),
                    "attack_type": AdvancedAttackType.STEALTHY_BYZANTINE.value,
                    "malicious": True,
                },
                timestep=timestep,
                malicious=True,
            )
            for target_id in target_ids
        ]

        return {
            "attacker_id": attacker_id,
            "attack_type": AdvancedAttackType.STEALTHY_BYZANTINE.value,
            "phase": "active",
            "sender_id": attacker_id,
            "receiver_ids": target_ids,
            "messages": messages,
            "edge_mutations": [],
            "is_malicious": True,
            "intensity": intensity,
            "target_trust": context[attacker_id]["trust_mean"],
        }

    def _build_flooding_event(
        self,
        simulation: Any,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Dict[str, Any]:
        neighbor_ids = self._select_targets(attacker_id, context, prefer_trusted=False, all_targets=True)
        burst_size = max(2, int(round(len(neighbor_ids) * self.config.flooding_burst_factor * profile.current_intensity)))
        intensity = float(np.clip(profile.current_intensity + 0.2 * burst_size / max(self.num_robots, 1), self.config.min_intensity, self.config.max_intensity))

        messages: List[Dict[str, Any]] = []
        receiver_ids: List[int] = []
        for index in range(burst_size):
            target_id = neighbor_ids[index % max(len(neighbor_ids), 1)] if neighbor_ids else None
            receiver_ids.append(target_id if target_id is not None else attacker_id)
            messages.append(
                self._make_message(
                    sender_id=attacker_id,
                    receiver_id=target_id,
                    msg_type=MessageType.HEARTBEAT if index % 2 == 0 else MessageType.SENSOR_DATA,
                    payload={
                        "sequence": index,
                        "burst_size": burst_size,
                        "attack_type": AdvancedAttackType.FLOODING_DOS.value,
                        "malicious": True,
                        "jitter": float(self.rng.uniform(0.0, 0.1)),
                    },
                    timestep=timestep,
                    malicious=True,
                )
            )

        return {
            "attacker_id": attacker_id,
            "attack_type": AdvancedAttackType.FLOODING_DOS.value,
            "phase": "active",
            "sender_id": attacker_id,
            "receiver_ids": receiver_ids,
            "messages": messages,
            "edge_mutations": [],
            "is_malicious": True,
            "intensity": intensity,
            "burst_size": burst_size,
        }

    def _build_collusion_event(
        self,
        simulation: Any,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Dict[str, Any]:
        group = self._active_collusion_group(attacker_id) or [attacker_id]
        targets = self._select_collusion_targets(group, context)
        base_payload = {
            "attack_type": AdvancedAttackType.COLLUSION.value,
            "malicious": True,
            "group_size": len(group),
            "consensus_signal": float(np.mean([context[member]["trust_mean"] for member in group])) if group else 0.0,
        }

        messages: List[Dict[str, Any]] = []
        for member_id in group:
            member_robot = simulation.robots[member_id]
            for target_id in targets:
                payload = {
                    **base_payload,
                    "claimed_position": (member_robot.position + self.rng.normal(0.0, 0.5 + profile.current_intensity, size=2)).tolist(),
                    "claimed_velocity": (member_robot.velocity + self.rng.normal(0.0, 0.1, size=2)).tolist(),
                    "collusion_group": list(group),
                }
                messages.append(
                    self._make_message(
                        sender_id=member_id,
                        receiver_id=target_id,
                        msg_type=MessageType.POSITION_UPDATE,
                        payload=payload,
                        timestep=timestep,
                        malicious=True,
                    )
                )

        receiver_ids = [message.get("receiver_id") for message in messages]
        return {
            "attacker_id": attacker_id,
            "attack_type": AdvancedAttackType.COLLUSION.value,
            "phase": "active",
            "sender_id": attacker_id,
            "receiver_ids": receiver_ids,
            "messages": messages,
            "edge_mutations": [],
            "is_malicious": True,
            "intensity": float(np.clip(profile.current_intensity + 0.1 * len(group), self.config.min_intensity, self.config.max_intensity)),
            "group_members": list(group),
            "targets": targets,
        }

    def _build_topology_event(
        self,
        simulation: Any,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        timestep: int,
        profile: AttackProfile,
    ) -> Dict[str, Any]:
        target_ids = self._select_targets(attacker_id, context, prefer_trusted=True, max_targets=max(1, len(context[attacker_id]["neighbor_ids"]) or 1))
        edge_mutations: List[Dict[str, Any]] = []
        messages: List[Dict[str, Any]] = []

        graph = getattr(simulation, "communication_graph", None)
        if graph is not None and hasattr(graph, "add_edge"):
            for target_id in target_ids:
                if self.rng.random() < self.config.fake_edge_probability:
                    edge_mutations.append(
                        {
                            "source": attacker_id,
                            "target": target_id,
                            "mutation": "fake_edge",
                            "weight": float(np.clip(profile.current_intensity, 0.1, 1.0)),
                        }
                    )
                    graph.add_edge(attacker_id, target_id, weight=float(np.clip(profile.current_intensity, 0.1, 1.0)), distance=0.0, fake=True, malicious=True)

                if self.rng.random() < self.config.hidden_node_probability:
                    edge_mutations.append(
                        {
                            "source": attacker_id,
                            "target": target_id,
                            "mutation": "hidden_node",
                        }
                    )
                    if graph.has_edge(attacker_id, target_id):
                        graph.remove_edge(attacker_id, target_id)

        for target_id in target_ids:
            messages.append(
                self._make_message(
                    sender_id=attacker_id,
                    receiver_id=target_id,
                    msg_type=MessageType.TRUST_QUERY,
                    payload={
                        "attack_type": AdvancedAttackType.TOPOLOGY_MANIPULATION.value,
                        "malicious": True,
                        "topology_hint": "edge_bias",
                        "visible_neighbors": len(context[attacker_id]["neighbor_ids"]),
                    },
                    timestep=timestep,
                    malicious=True,
                )
            )

        return {
            "attacker_id": attacker_id,
            "attack_type": AdvancedAttackType.TOPOLOGY_MANIPULATION.value,
            "phase": "active",
            "sender_id": attacker_id,
            "receiver_ids": target_ids,
            "messages": messages,
            "edge_mutations": edge_mutations,
            "is_malicious": True,
            "intensity": float(np.clip(profile.current_intensity, self.config.min_intensity, self.config.max_intensity)),
        }

    def _select_targets(
        self,
        attacker_id: int,
        context: Dict[int, Dict[str, Any]],
        prefer_trusted: bool,
        max_targets: Optional[int] = None,
        all_targets: bool = False,
    ) -> List[int]:
        candidate_ids = list(context[attacker_id]["neighbor_ids"])
        if not candidate_ids:
            candidate_ids = [rid for rid in range(self.num_robots) if rid != attacker_id]

        if all_targets:
            return candidate_ids

        max_targets = max_targets or max(1, len(candidate_ids) // 2)
        trust_lookup = self._trust_lookup(attacker_id, context)
        candidate_ids.sort(key=lambda rid: trust_lookup.get(rid, 0.5), reverse=prefer_trusted)

        selection_size = min(max_targets, len(candidate_ids))
        if selection_size <= 0:
            return []

        return candidate_ids[:selection_size]

    def _select_collusion_targets(self, group: Sequence[int], context: Dict[int, Dict[str, Any]]) -> List[int]:
        if not group:
            return []

        shared = set(range(self.num_robots))
        for member_id in group:
            member_neighbors = set(context[member_id]["neighbor_ids"])
            shared &= member_neighbors or set()

        shared -= set(group)
        if not shared:
            focus = max(group, key=lambda rid: context[rid]["trust_mean"])
            shared = set(context[focus]["neighbor_ids"]) - set(group)

        if not shared:
            shared = set(self._select_targets(group[0], context, prefer_trusted=True))

        return sorted(shared)[: max(1, min(3, len(shared)))]

    def _trust_lookup(self, attacker_id: int, context: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        robot = self.simulation.robots[attacker_id]
        return {neighbor_id: robot.trust_scores.get(neighbor_id, 0.5) for neighbor_id in context[attacker_id]["neighbor_ids"]}

    def _make_message(
        self,
        sender_id: int,
        receiver_id: Optional[int],
        msg_type: MessageType,
        payload: Dict[str, Any],
        timestep: int,
        malicious: bool,
    ) -> Dict[str, Any]:
        return {
            "msg_id": len(self.attack_log),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "type": msg_type.value,
            "msg_type": msg_type.value,
            "timestamp": timestep,
            "payload": payload,
            "malicious": malicious,
            "attack_label": 1 if malicious else 0,
            "attack_type": payload.get("attack_type"),
        }

    def _apply_event(self, simulation: Any, event: Dict[str, Any], timestep: int) -> None:
        if hasattr(simulation, "communication_manager") and simulation.communication_manager is not None:
            broker = simulation.communication_manager.broker
            for message in event.get("messages", []):
                broker.send_message(
                    sender_id=message["sender_id"],
                    message_type=MessageType(message["msg_type"]),
                    payload=message["payload"],
                    receiver_id=message.get("receiver_id"),
                    timestamp=float(timestep),
                )
            return

        if hasattr(simulation, "robots"):
            for message in event.get("messages", []):
                receiver_id = message.get("receiver_id")
                if receiver_id is None:
                    for target_id in event.get("receiver_ids", []):
                        if target_id in simulation.robots and target_id != message["sender_id"]:
                            simulation.robots[target_id].add_incoming_message(dict(message))
                elif receiver_id in simulation.robots:
                    simulation.robots[receiver_id].add_incoming_message(dict(message))

        if hasattr(simulation, "message_log") and isinstance(simulation.message_log, list):
            simulation.message_log.append(event)


class AttackScenario(AdaptiveAttackEngine):
    """Backward-compatible wrapper around the adaptive attack engine."""

    def __init__(self, environment: Any, attack_ratio: float = 0.15, seed: int = 42):
        super().__init__(environment, AttackEngineConfig(attack_ratio=attack_ratio, seed=seed), seed=seed)

    def execute_timestep_attacks(self, timestep: int) -> Dict[str, Any]:
        return self.step(timestep)
