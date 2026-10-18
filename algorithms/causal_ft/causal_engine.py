# Causal Inference Engine for Structural Root-Cause Analysis (RCA).
#
# Implements Directed Acyclic Graph (DAG) structural equation modeling (SEM)
# and exogenous parent residual attribution to distinguish causal root causes
# from downstream symptoms and mere correlations.

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import networkx as nx

from algorithms.base import RCADiagnosisResult
from evaluation.context import BenchmarkInput

logger = logging.getLogger("CausalEngine")


@dataclass
class CausalEdge:
    # Represents a directed causal mechanism u -> v.
    source: str
    target: str
    weight: float = 1.0
    is_domain_prior: bool = True
    confidence: float = 0.9


class CausalGraphSpecification:
    # Constructs and manages the Directed Acyclic Graph (DAG) representation
    # of system metrics and Edge-IoT protocol causal mechanisms.

    def __init__(self):
        self.dag: nx.DiGraph = nx.DiGraph()
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.domain_edges: Set[Tuple[str, str]] = set()

    def add_causal_edge(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        is_domain: bool = True
    ) -> None:
        # Add a directed causal edge u -> v (u causes v).
        self.dag.add_edge(u, v, weight=weight)
        self.edge_weights[(u, v)] = weight
        if is_domain:
            self.domain_edges.add((u, v))

    def build_domain_graph(self, feature_names: List[str]) -> None:
        # Initializes domain-informed causal dependencies for Edge-IIoTset telemetry.
        # All edges are explicitly labeled as domain priors.
        available = set(feature_names)

        # Domain causal mechanisms: Transport & Protocol causes flow & aggregate states
        domain_mechanisms = [
            # TCP flags cause active connection states and flag aggregations
            ("tcp.flags", "tcp_active_flags_count", 0.95),
            ("tcp.flags", "tcp.connection.syn", 0.90),
            ("tcp.flags", "tcp.connection.fin", 0.90),
            ("tcp.flags", "tcp.connection.rst", 0.90),
            ("tcp.flags", "tcp.connection.synack", 0.90),
            ("tcp.connection.syn", "tcp_seq_ack_ratio", 0.85),
            ("tcp.seq", "tcp_seq_ack_ratio", 0.80),
            ("tcp.ack", "tcp_seq_ack_ratio", 0.80),
            ("tcp.len", "tcp.checksum", 0.85),
            ("tcp.dstport", "is_well_known_dstport", 0.95),

            # UDP flow dependencies
            ("udp.port", "udp.stream", 0.80),
            ("udp.stream", "udp.time_delta", 0.85),

            # ICMP packet dynamics
            ("icmp.seq_le", "icmp.checksum", 0.85),
            ("icmp.transmit_timestamp", "icmp.checksum", 0.75),

            # HTTP transaction dependencies
            ("http.content_length", "http.response", 0.85),
            ("http_method_GET", "http.response", 0.75),
            ("http_method_POST", "http.content_length", 0.80),

            # MQTT protocol dynamics
            ("mqtt.conflags", "mqtt.conflag.cleansess", 0.90),
            ("mqtt.len", "mqtt.hdrflags", 0.85),
            ("mqtt.topic_len", "has_mqtt_topic", 0.95),
            ("is_mqtt_proto", "mqtt.msgtype", 0.85),

            # Modbus TCP dynamics
            ("mbtcp.trans_id", "mbtcp.len", 0.80),
            ("mbtcp.unit_id", "mbtcp.len", 0.75),
        ]

        for u, v, w in domain_mechanisms:
            if u in available and v in available:
                self.add_causal_edge(u, v, weight=w, is_domain=True)

        # Add remaining isolated nodes
        for f in feature_names:
            if f not in self.dag:
                self.dag.add_node(f)

        logger.info(
            "Constructed Causal DAG with %d nodes and %d causal edges.",
            self.dag.number_of_nodes(),
            self.dag.number_of_edges(),
        )

    def get_parents(self, node: str) -> List[str]:
        return list(self.dag.predecessors(node)) if node in self.dag else []

    def get_descendants(self, node: str) -> List[str]:
        return list(nx.descendants(self.dag, node)) if node in self.dag else []


class CausalInferenceEngine:
    # Executes Structural Equation Residual Analysis on the Causal DAG
    # to localize root causes from streaming observations.

    def __init__(self, causal_graph: Optional[CausalGraphSpecification] = None):
        self.causal_graph = causal_graph or CausalGraphSpecification()
        self.feature_names: List[str] = []
        self.parent_weights: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit_structural_equations(
        self,
        X_normal: np.ndarray,
        feature_names: List[str],
    ) -> None:
        # Fits structural equation coefficients from normal baseline telemetry:
        # v = sum_{p in Parents(v)} w_{pv} * p + epsilon_v
        self.feature_names = list(feature_names)
        if self.causal_graph.dag.number_of_edges() == 0:
            self.causal_graph.build_domain_graph(self.feature_names)

        feat_idx = {name: i for i, name in enumerate(feature_names)}

        for node in self.causal_graph.dag.nodes():
            parents = self.causal_graph.get_parents(node)
            if not parents:
                self.parent_weights[node] = {}
                continue

            # Ridge / OLS fit of node from its parents
            try:
                valid_parents = [p for p in parents if p in feat_idx]
                if valid_parents and len(X_normal) > len(valid_parents) + 5:
                    P_mat = X_normal[:, [feat_idx[p] for p in valid_parents]]
                    y_vec = X_normal[:, feat_idx[node]]
                    # Ridge regression: w = (P^T P + lambda I)^(-1) P^T y
                    lam = 1e-2
                    weights, _, _, _ = np.linalg.lstsq(
                        P_mat.T @ P_mat + lam * np.eye(len(valid_parents)),
                        P_mat.T @ y_vec,
                        rcond=None,
                    )
                    self.parent_weights[node] = {
                        p: float(w) for p, w in zip(valid_parents, weights)
                    }
                else:
                    # Fallback to normalized graph edge weights
                    self.parent_weights[node] = {
                        p: self.causal_graph.edge_weights.get((p, node), 1.0) / len(parents)
                        for p in parents
                    }
            except Exception as e:
                logger.debug("Failed fitting SEM for %s: %s", node, e)
                self.parent_weights[node] = {
                    p: 1.0 / len(parents) for p in parents
                }

        self.is_fitted = True

    def diagnose(
        self,
        record: BenchmarkInput,
        top_k: int = 5,
    ) -> RCADiagnosisResult:
        # Diagnose the root cause of an anomaly for a given BenchmarkInput.
        # Calculates exogenous structural residuals:
        # r(v) = |x(v) - sum_{p in Parents(v)} w_{pv} x(p)|
        t0 = time.perf_counter_ns()
        feat_dict = record.features

        if not self.feature_names:
            self.feature_names = list(feat_dict.keys())
            if self.causal_graph.dag.number_of_edges() == 0:
                self.causal_graph.build_domain_graph(self.feature_names)

        causal_scores: Dict[str, float] = {}
        residuals: Dict[str, float] = {}
        expected_values: Dict[str, float] = {}

        # 1. Compute structural residuals for each variable
        for var in self.feature_names:
            observed_val = float(feat_dict.get(var, 0.0))
            parents = self.causal_graph.get_parents(var)

            if not parents:
                # Root node in DAG: anomaly cannot be explained by parents
                expected_val = 0.0
                structural_residual = abs(observed_val)
            else:
                p_weights = self.parent_weights.get(var, {})
                expected_val = sum(
                    p_weights.get(p, 1.0 / len(parents)) * float(feat_dict.get(p, 0.0))
                    for p in parents
                )
                # Unexplained deviation from parents' causal expectation
                structural_residual = abs(observed_val - expected_val)

            residuals[var] = round(structural_residual, 4)
            expected_values[var] = round(expected_val, 4)

            # 2. Downstream causal impact propagation
            descendants = self.causal_graph.get_descendants(var)
            downstream_impact = 1.0
            for d in descendants:
                d_val = abs(float(feat_dict.get(d, 0.0)))
                downstream_impact += 0.2 * d_val

            # Causal score balances exogenous innovation with downstream causal reach
            score = structural_residual * downstream_impact
            causal_scores[var] = score

        # 3. Rank variables by causal score in descending order
        sorted_vars = sorted(causal_scores.keys(), key=lambda k: causal_scores[k], reverse=True)
        ranked_causes = sorted_vars[:max(1, top_k)]

        # Softmax / normalized confidence scores for top candidates
        top_scores = np.array([causal_scores[k] for k in ranked_causes], dtype=np.float64)
        sum_scores = np.sum(top_scores) + 1e-6
        confidence_scores = {k: round(float(causal_scores[k] / sum_scores), 4) for k in ranked_causes}

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        return RCADiagnosisResult(
            ranked_root_causes=ranked_causes,
            confidence_scores=confidence_scores,
            p_values={},
            execution_time_ms=round(elapsed_ms, 4),
            metadata={
                "top_1": ranked_causes[0] if ranked_causes else "UNKNOWN",
                "top_3": ranked_causes[:3],
                "top_5": ranked_causes[:5],
                "residuals": {k: residuals[k] for k in ranked_causes},
                "expected_values": {k: expected_values[k] for k in ranked_causes},
                "total_nodes_evaluated": len(self.feature_names),
                "is_causal_dag": True,
            }
        )
