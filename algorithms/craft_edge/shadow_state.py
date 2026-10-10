# CRAFT-Edge Shadow Edge Node State.
#
# Provides an isolated in-memory execution state model for tracking edge node health
# and recovery mitigation factors without modifying immutable historical datasets.
# Equations and mitigation factors strictly align with the physical EdgeNode simulator
# in `simulation/edge_nodes.py`.

from dataclasses import dataclass, field
import logging
from typing import Dict, Any, Optional

from .policy import RecoveryAction

logger = logging.getLogger("CRAFTEdge.ShadowState")


@dataclass
class ShadowNodeState:
    # Shadow representation of an edge computing node's operational health.
    #
    # CRITICAL METHODOLOGICAL BOUNDARY:
    # When streaming immutable packet captures (such as Edge-IIoTset), physical host CPU,
    # RAM, and hardware metrics are NOT directly recorded by network sniffers.
    # In that mode, CPU and memory values generated here are SYNTHETIC PROXY ESTIMATES
    # derived from protocol activity and traffic volumes. They represent simulated
    # edge container pressure and MUST NOT be claimed as physical hardware power/CPU measurements.
    node_id: str
    cpu_utilization: float = 25.0
    memory_utilization: float = 30.0
    network_utilization: float = 20.0
    latency: float = 18.0
    packet_loss: float = 0.5
    workload: float = 20.0
    active: bool = True
    consecutive_faults: int = 0
    last_action: str = RecoveryAction.NO_ACTION.value
    last_action_applied: bool = True
    is_synthetic_proxy: bool = False
    metric_sources: Dict[str, str] = field(default_factory=dict)

    # Persistent mitigation scaling factors (1.0 = unmitigated, < 1.0 = mitigated)
    workload_mitigation: float = 1.0
    cpu_mitigation: float = 1.0
    mem_mitigation: float = 1.0
    net_mitigation: float = 1.0
    lat_mitigation: float = 1.0

    def get_metrics(self) -> Dict[str, float]:
        # Returns current observable metric state snapshot.
        return {
            "cpu_utilization": round(self.cpu_utilization, 2),
            "memory_utilization": round(self.memory_utilization, 2),
            "network_utilization": round(self.network_utilization, 2),
            "latency": round(self.latency, 2),
            "packet_loss": round(self.packet_loss, 2),
            "workload": round(self.workload, 2),
            "active": 1.0 if self.active else 0.0,
        }

    def get_state_metadata(self) -> Dict[str, Any]:
        # Returns metrics accompanied by explicit data-origin provenance.
        return {
            "node_id": self.node_id,
            "metrics": self.get_metrics(),
            "is_synthetic_proxy": self.is_synthetic_proxy,
            "metric_sources": dict(self.metric_sources),
            "measurement_notice": (
                "Shadow-state synthetic proxy estimates; NOT physical hardware readings."
                if self.is_synthetic_proxy else
                "Direct observable telemetry readings."
            ),
        }

    def update_from_telemetry(self, features: Dict[str, float]) -> None:
        # Ingests telemetry metrics into shadow state, labeling direct vs synthetic proxy metrics.
        self.metric_sources = {}
        raw_cpu = features.get("cpu_utilization", None)
        raw_mem = features.get("memory_utilization", None)
        raw_net = features.get("network_utilization", None)
        raw_lat = features.get("latency", None)
        raw_loss = features.get("packet_loss", None)

        if raw_cpu is not None:
            self.metric_sources["cpu_utilization"] = "DIRECT_OBSERVATION"
        else:
            self.metric_sources["cpu_utilization"] = "SYNTHETIC_PROXY_ESTIMATE"
            tcp_flags = features.get("tcp_active_flags_count", 0.0)
            pkt_len = features.get("tcp.len", 0.0) + features.get("udp.time_delta", 0.0)
            raw_cpu = 20.0 + min(75.0, tcp_flags * 15.0 + (pkt_len / 100.0))

        if raw_mem is not None:
            self.metric_sources["memory_utilization"] = "DIRECT_OBSERVATION"
        else:
            self.metric_sources["memory_utilization"] = "SYNTHETIC_PROXY_ESTIMATE"
            raw_mem = 25.0 + min(70.0, raw_cpu * 0.7)

        if raw_net is not None:
            self.metric_sources["network_utilization"] = "DIRECT_OBSERVATION"
        else:
            self.metric_sources["network_utilization"] = "SYNTHETIC_PROXY_ESTIMATE"
            udp_delta = abs(features.get("udp.time_delta", 0.0))
            raw_net = 20.0 + min(75.0, udp_delta * 30.0 + raw_cpu * 0.4)

        if raw_lat is not None:
            self.metric_sources["latency"] = "DIRECT_OBSERVATION"
        else:
            self.metric_sources["latency"] = "SYNTHETIC_PROXY_ESTIMATE"
            raw_lat = 15.0 + (raw_cpu / 100.0) ** 2 * 60.0 + (raw_net / 100.0) * 30.0

        if raw_loss is not None:
            self.metric_sources["packet_loss"] = "DIRECT_OBSERVATION"
        else:
            self.metric_sources["packet_loss"] = "SYNTHETIC_PROXY_ESTIMATE"
            raw_loss = max(0.0, (raw_net - 75.0) * 0.8) if raw_net > 75.0 else 0.5

        self.is_synthetic_proxy = any(v == "SYNTHETIC_PROXY_ESTIMATE" for v in self.metric_sources.values())

        # Apply mitigation multipliers to raw demands
        self.cpu_utilization = max(5.0, min(100.0, float(raw_cpu) * self.cpu_mitigation))
        self.memory_utilization = max(10.0, min(100.0, float(raw_mem) * self.mem_mitigation))
        self.network_utilization = max(5.0, min(100.0, float(raw_net) * self.net_mitigation))
        self.latency = max(5.0, min(500.0, float(raw_lat) * self.lat_mitigation))
        self.packet_loss = max(0.0, min(100.0, float(raw_loss) * self.net_mitigation))

    def apply_recovery_action(self, action: str) -> Dict[str, float]:
        # Applies recovery action establishing persistent mitigation factors.
        # Exactly mirrors `simulation.edge_nodes.EdgeNode.apply_recovery_action`.
        is_supported = action in RecoveryAction.all_actions()
        self.last_action_applied = is_supported
        self.last_action = action

        if not is_supported:
            logger.warning("[ShadowState] Action '%s' is unsupported/unexecutable.", action)
            return self.get_metrics()

        if action == RecoveryAction.WORKLOAD_REDISTRIBUTION.value:
            self.workload_mitigation = 0.4
            self.cpu_mitigation = 0.3
            self.cpu_utilization = max(5.0, self.cpu_utilization * 0.4)
            self.latency = max(5.0, self.latency * 0.6)
            self.consecutive_faults = 0

        elif action == RecoveryAction.TRAFFIC_REROUTING.value:
            self.net_mitigation = 0.2
            self.lat_mitigation = 0.3
            self.network_utilization = max(5.0, self.network_utilization * 0.3)
            self.packet_loss = max(0.1, self.packet_loss * 0.2)
            self.latency = max(5.0, self.latency * 0.4)
            self.consecutive_faults = 0

        elif action == RecoveryAction.TASK_MIGRATION.value:
            self.mem_mitigation = 0.2
            self.cpu_mitigation = 0.4
            self.workload_mitigation = 0.5
            self.memory_utilization = max(10.0, self.memory_utilization * 0.3)
            self.cpu_utilization = max(5.0, self.cpu_utilization * 0.5)
            self.consecutive_faults = 0

        elif action == RecoveryAction.RESOURCE_REBALANCING.value:
            self.cpu_mitigation = 0.5
            self.mem_mitigation = 0.5
            self.lat_mitigation = 0.4
            self.cpu_utilization = max(5.0, self.cpu_utilization * 0.6)
            self.memory_utilization = max(10.0, self.memory_utilization * 0.6)
            self.latency = max(5.0, self.latency * 0.5)
            self.consecutive_faults = 0

        elif action == RecoveryAction.EDGE_NODE_FAILOVER.value:
            self.active = True
            self.clear_mitigations()
            self.latency = 18.0
            self.packet_loss = 0.5
            self.cpu_utilization = 25.0
            self.consecutive_faults = 0

        elif action == RecoveryAction.NO_ACTION.value:
            self.consecutive_faults += 1

        return self.get_metrics()

    def clear_mitigations(self) -> None:
        self.workload_mitigation = 1.0
        self.cpu_mitigation = 1.0
        self.mem_mitigation = 1.0
        self.net_mitigation = 1.0
        self.lat_mitigation = 1.0
