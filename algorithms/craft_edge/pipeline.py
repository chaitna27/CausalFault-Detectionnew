# CRAFT-Edge Integration Pipeline.
#
# Implements the full Causal Real-Time Adaptive Fault Tolerance workflow:
# 1. Telemetry Ingestion (Observable BenchmarkInput / TelemetryRecord)
# 2. Online Streaming Anomaly Detection (EWMA + Baseline Deviation)
# 3. Causal Root-Cause Diagnosis (Fast Domain DAG Structural Equation Model Residuals)
# 4. Adaptive Recovery Action Selection (Efficacy Q-table with Epsilon-Greedy Exploration)
# 5. Outcome Evaluation (Health Metric Benefit minus Operational Action Cost)
# 6. Online Policy Update (Q_new = Q_old + alpha * (reward - Q_old))
import os
from dataclasses import dataclass, field
import logging
import time
from typing import Dict, Any, Optional, Set, List, Tuple
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
    MitigationResult,
)
from algorithms.causal_ft.detector import StreamingCausalDetector, DetectorConfig
from algorithms.causal_ft.causal_engine import CausalInferenceEngine, CausalGraphSpecification
import networkx as nx
from evaluation.context import BenchmarkInput
from .policy import AdaptiveEfficacyPolicy, RecoveryAction, classify_root_cause
from .evaluator import RecoveryOutcomeEvaluator, RecoveryOutcome
from .shadow_state import ShadowNodeState

logger = logging.getLogger("CRAFTEdge.Pipeline")


class CraftCausalGraphSpecification(CausalGraphSpecification):
    """Craft-optimized Causal Graph Specification with lazy reachability and adjacency caching.

    Accelerates causal root-cause analysis by lazily caching node descendants and predecessors
    on static graph topologies, while automatically invalidating the cache upon graph mutation.
    """

    def __init__(self):
        super().__init__()
        self._descendants_cache: Optional[Dict[str, Tuple[str, ...]]] = None
        self._parents_cache: Optional[Dict[str, Tuple[str, ...]]] = None

    def _invalidate_cache(self) -> None:
        self._descendants_cache = None
        self._parents_cache = None

    def add_causal_edge(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        is_domain: bool = True
    ) -> None:
        super().add_causal_edge(u, v, weight=weight, is_domain=is_domain)
        self._invalidate_cache()

    def build_domain_graph(self, feature_names: List[str]) -> None:
        super().build_domain_graph(feature_names)
        self._invalidate_cache()

    def get_parents(self, node: str) -> List[str]:
        if node not in self.dag:
            return []
        if self._parents_cache is None:
            self._parents_cache = {}
        if node not in self._parents_cache:
            self._parents_cache[node] = tuple(self.dag.predecessors(node))
        return list(self._parents_cache[node])

    def get_descendants(self, node: str) -> List[str]:
        if node not in self.dag:
            return []
        if self._descendants_cache is None:
            self._descendants_cache = {}
        if node not in self._descendants_cache:
            self._descendants_cache[node] = tuple(nx.descendants(self.dag, node))
        return list(self._descendants_cache[node])


@dataclass
class CRAFTEdgeConfig:
    # Hyperparameters for CRAFT-Edge integration.
    detection_threshold: float = 0.65
    learning_rate: float = 0.20
    epsilon: float = 0.15
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    decay_interval: int = 1
    decay_mode: str = "interval"
    benefit_weight: float = 1.0
    cost_weight: float = 0.30
    seed: int = 42


class CRAFTEdgePipeline(BaseFaultToleranceAlgorithm):
    # Unified CRAFT-Edge Fault Tolerance Pipeline conforming to BaseFaultToleranceAlgorithm.
    # Reuses existing fast SEM causal diagnosis and streaming detection while implementing
    # closed-loop adaptive recovery action selection, outcome evaluation, and policy updates.

    def __init__(self, config: Optional[Dict[str, Any]] = None, simulation: Any = None):
        cfg_dict = config or {}
        self.cfg = CRAFTEdgeConfig(
            detection_threshold=float(cfg_dict.get("detection_threshold", 0.65)),
            learning_rate=float(cfg_dict.get("learning_rate", 0.20)),
            epsilon=float(cfg_dict.get("epsilon", 0.15)),
            epsilon_min=float(cfg_dict.get("epsilon_min", 0.02)),
            epsilon_decay=float(cfg_dict.get("epsilon_decay", 0.995)),
            decay_interval=int(cfg_dict.get("decay_interval", 1)),
            decay_mode=str(cfg_dict.get("decay_mode", "interval")),
            benefit_weight=float(cfg_dict.get("benefit_weight", 1.0)),
            cost_weight=float(cfg_dict.get("cost_weight", 0.30)),
            seed=int(cfg_dict.get("seed", 42)),
        )

        # 1. Anomaly Detector (reusing StreamingCausalDetector)
        self.detector = StreamingCausalDetector(
            DetectorConfig(threshold=self.cfg.detection_threshold)
        )

        # 2. Causal Engine (reusing existing fast SEM diagnosis path with Craft-optimized lazy reachability graph)
        self.causal_engine = CausalInferenceEngine(causal_graph=CraftCausalGraphSpecification())

        # 3. Adaptive Action Efficacy Policy
        self.policy = AdaptiveEfficacyPolicy(
            learning_rate=self.cfg.learning_rate,
            epsilon=self.cfg.epsilon,
            epsilon_min=self.cfg.epsilon_min,
            epsilon_decay=self.cfg.epsilon_decay,
            decay_interval=self.cfg.decay_interval,
            decay_mode=self.cfg.decay_mode,
            seed=self.cfg.seed,
        )

        # 4. Recovery Outcome Evaluator
        self.evaluator = RecoveryOutcomeEvaluator(
            benefit_weight=self.cfg.benefit_weight,
            cost_weight=self.cfg.cost_weight,
        )

        # 5. Execution State: Live SimPy simulation if injected, otherwise shadow state
        self.simulation = simulation
        self.shadow_nodes: Dict[str, ShadowNodeState] = {}

        # Pipeline lifecycle state
        self.last_diagnosis: Optional[RCADiagnosisResult] = None
        self.last_mitigation: Optional[MitigationResult] = None
        self.last_outcome: Optional[RecoveryOutcome] = None
        self.is_running: bool = False
        self.in_fault_episode: bool = False
        self.processed_records_count: int = 0
        self.anomalies_detected_count: int = 0
        self.recoveries_executed_count: int = 0

    @property
    def name(self) -> str:
        return "CRAFT-Edge: Causal Real-Time Adaptive Fault Tolerance"

    @property
    def paper_id(self) -> str:
        return "craft_edge"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.STREAMING_DETECTION,
            AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS,
            AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE,
            AlgorithmCapability.PREEMPTIVE_MIGRATION,
        }

    def initialize(self, config: Any = None) -> None:
        if config is not None and isinstance(config, dict):
            if "detection_threshold" in config:
                self.detector.config.threshold = float(config["detection_threshold"])
            elif "threshold" in config:
                self.detector.config.threshold = float(config["threshold"])
            if "learning_rate" in config:
                self.policy.alpha = float(config["learning_rate"])
            if "epsilon" in config:
                self.policy.epsilon = float(config["epsilon"])
            if "epsilon_min" in config:
                self.policy.epsilon_min = float(config["epsilon_min"])
            if "epsilon_decay" in config:
                self.policy.epsilon_decay = float(config["epsilon_decay"])
            if "decay_interval" in config:
                self.policy.decay_interval = max(1, int(config["decay_interval"]))
            if "decay_mode" in config:
                self.policy.decay_mode = str(config["decay_mode"]).lower().strip()
            if "benefit_weight" in config:
                self.evaluator.benefit_weight = float(config["benefit_weight"])
            if "cost_weight" in config:
                self.evaluator.cost_weight = float(config["cost_weight"])
        self.reset()

    def fit(self, training_data: Any = None, feature_names: Optional[List[str]] = None) -> None:
        # Fit detector baseline and SEM causal equations using unlabeled training telemetry.
        # Operates strictly on normal training data without ground-truth labels.
        if training_data is None:
            return

        if isinstance(training_data, np.ndarray):
            X_train = training_data
            if feature_names is not None and len(feature_names) == X_train.shape[1]:
                feat_names = list(feature_names)
            else:
                # Check for standard feature names artifact if dimensions match
                art_path = os.path.join("data", "processed", "artifacts", "feature_names.json")
                if X_train.shape[1] == 62 and os.path.exists(art_path):
                    import json
                    try:
                        with open(art_path, "r") as f:
                            feat_names = json.load(f)
                    except Exception:
                        feat_names = [f"f_{i}" for i in range(X_train.shape[1])]
                else:
                    feat_names = [f"f_{i}" for i in range(X_train.shape[1])]
        elif hasattr(training_data, "values") and hasattr(training_data, "columns"):
            X_train = training_data.values
            feat_names = list(training_data.columns)
        else:
            return

        # 1. Fit streaming detector baseline
        self.detector.fit(X_train)

        # 2. Fit SEM structural equations on training telemetry
        self.causal_engine.fit_structural_equations(X_train, feat_names)

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def _get_or_create_shadow_node(self, node_id: str) -> ShadowNodeState:
        if node_id not in self.shadow_nodes:
            self.shadow_nodes[node_id] = ShadowNodeState(node_id=node_id)
        return self.shadow_nodes[node_id]

    def _convert_record_to_benchmark_input(self, record: Any) -> BenchmarkInput:
        # Normalizes input into standardized BenchmarkInput.
        if isinstance(record, BenchmarkInput):
            return record

        # Handle TelemetryRecord from simulation or data adapter
        if hasattr(record, "get_available_features"):
            feats = record.get_available_features()
            if hasattr(record, "feature_vector") and getattr(record, "feature_vector", None) is not None:
                feature_vec = record.feature_vector
            else:
                feature_vec = np.array(list(feats.values()), dtype=np.float64)
            pos = getattr(record, "stream_position", self.processed_records_count)
            ts = float(getattr(record, "timestamp", time.time()))
            dev_id = str(getattr(record, "device_id", "dev_0"))
            node_id = str(getattr(record, "edge_node_id", "node_0"))
            return BenchmarkInput(
                stream_position=pos,
                timestamp=ts,
                timestamp_str=str(ts),
                device_id=dev_id,
                edge_node_id=node_id,
                features=feats,
                feature_vector=feature_vec,
            )

        raise ValueError(f"Unsupported record type: {type(record)}")

    def detect(self, record: Any) -> DetectionResult:
        obs_input = self._convert_record_to_benchmark_input(record)
        return self.detector.detect(obs_input)

    def diagnose_root_cause(
        self,
        normal_window: Any = None,
        anomalous_window: Any = None,
    ) -> RCADiagnosisResult:
        # Diagnoses root cause using the fast SEM structural residual path.
        if anomalous_window is not None:
            if isinstance(anomalous_window, BenchmarkInput):
                return self.causal_engine.diagnose(anomalous_window, top_k=5)
            elif isinstance(anomalous_window, dict):
                dummy_input = BenchmarkInput(
                    stream_position=0,
                    timestamp=0.0,
                    timestamp_str="",
                    device_id="query",
                    edge_node_id="query",
                    features=anomalous_window,
                    feature_vector=np.array(list(anomalous_window.values())),
                )
                return self.causal_engine.diagnose(dummy_input, top_k=5)

        if self.last_diagnosis is not None:
            return self.last_diagnosis

        return RCADiagnosisResult(
            ranked_root_causes=["UNKNOWN"],
            confidence_scores={"UNKNOWN": 1.0},
            p_values={},
            execution_time_ms=0.0,
            metadata={"status": "NO_ACTIVE_ANOMALY"},
        )

    def process(self, record: Any) -> DetectionResult:
        # Full closed-loop CRAFT-Edge workflow:
        # Ingestion -> Anomaly Detection -> Causal RCA -> Action Selection -> Execution -> Outcome Eval -> Policy Update
        obs_input = self._convert_record_to_benchmark_input(record)
        self.processed_records_count += 1
        node_id = obs_input.edge_node_id or "default_node"

        # Update shadow state with current observable telemetry
        shadow_node = self._get_or_create_shadow_node(node_id)
        shadow_node.update_from_telemetry(obs_input.features)

        # 1. Online Streaming Anomaly Detection
        det_result = self.detector.detect(obs_input)

        diagnosis: Optional[RCADiagnosisResult] = None
        mitigation: Optional[MitigationResult] = None
        outcome: Optional[RecoveryOutcome] = None

        if not det_result.is_anomaly:
            # Transition from active anomaly burst to normal telemetry signals episode completion
            if self.in_fault_episode:
                self.in_fault_episode = False
                self.policy.step_episode()
        else:
            self.in_fault_episode = True
            self.anomalies_detected_count += 1

            # 2. Causal Root-Cause Diagnosis (Fast SEM path)
            diagnosis = self.causal_engine.diagnose(obs_input, top_k=5)
            self.last_diagnosis = diagnosis
            top_cause = diagnosis.ranked_root_causes[0] if diagnosis.ranked_root_causes else "GENERIC_RESOURCE_FAULT"
            cause_category = classify_root_cause(top_cause)

            # 3. Recovery-Action Selection via Adaptive Policy
            selected_action, is_explored = self.policy.select_action(cause_category, explore=True)

            # 4. Recovery Execution & Snapshot Capture
            # Capture pre-recovery state and execute action with explicit status check
            execution_applied = True
            if self.simulation is not None and hasattr(self.simulation, "nodes"):
                if node_id in self.simulation.nodes:
                    sim_node = self.simulation.nodes[node_id]
                    pre_metrics = sim_node.get_state_dict() if hasattr(sim_node, "get_state_dict") else shadow_node.get_metrics()
                    is_failed_before = getattr(sim_node, "node_failed", False)

                    # Execute action in physical simulator and capture execution result
                    applied = self.simulation.apply_recovery(node_id, selected_action)
                    execution_applied = bool(applied)
                    post_metrics = sim_node.get_state_dict() if hasattr(sim_node, "get_state_dict") else shadow_node.get_metrics()
                    is_failed_after = getattr(sim_node, "node_failed", False)
                else:
                    # Target node not found in simulator
                    execution_applied = False
                    pre_metrics = shadow_node.get_metrics()
                    post_metrics = pre_metrics
                    is_failed_before = False
                    is_failed_after = False
            else:
                # Execute in shadow state model (zero CSV mutation)
                pre_metrics = shadow_node.get_metrics()
                is_failed_before = not shadow_node.active
                post_metrics = shadow_node.apply_recovery_action(selected_action)
                execution_applied = bool(shadow_node.last_action_applied)
                is_failed_after = not shadow_node.active

            self.recoveries_executed_count += 1

            # 5. Recovery Outcome Evaluation
            outcome = self.evaluator.evaluate(
                action=selected_action,
                root_cause_category=cause_category,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                is_node_failed_before=is_failed_before,
                is_node_failed_after=is_failed_after,
                action_cost_override=self.policy.get_action_cost(selected_action),
                execution_applied=execution_applied,
            )
            self.last_outcome = outcome

            # 6. Adaptive Policy Update: Q_new = Q_old + alpha * (reward - Q_old)
            q_updated = self.policy.update(
                cause=cause_category,
                action=selected_action,
                reward=outcome.reward,
                success=outcome.success,
            )

            # Format standardized MitigationResult
            mig_bytes = 0
            if "MIGRATION" in selected_action and execution_applied:
                mig_bytes = 512 * 1024  # 512 KB simulated migration state payload

            is_mitigation_success = bool(outcome.success and execution_applied)
            tasks_affected = 1 if (selected_action != RecoveryAction.NO_ACTION.value and execution_applied) else 0

            mitigation = MitigationResult(
                action_type=selected_action,
                target_node=node_id,
                source_node=node_id,
                tasks_affected=tasks_affected,
                state_bytes_transferred=mig_bytes,
                success=is_mitigation_success,
                metadata={
                    "root_cause_metric": top_cause,
                    "root_cause_category": cause_category,
                    "q_value_updated": round(q_updated, 4),
                    "reward": outcome.reward,
                    "benefit": outcome.benefit,
                    "cost": outcome.cost,
                    "execution_applied": execution_applied,
                    "is_explored": is_explored,
                    "absolute_improvement": outcome.absolute_improvement,
                    "relative_improvement": outcome.relative_improvement,
                    "target_metric": outcome.target_metric,
                }
            )
            self.last_mitigation = mitigation

        # Attach diagnosis and mitigation results into raw_output
        det_result.raw_output["diagnosis"] = diagnosis
        det_result.raw_output["mitigation"] = mitigation
        det_result.raw_output["outcome"] = outcome

        return det_result

    def execute_mitigation(self, context: Any = None) -> MitigationResult:
        if self.last_mitigation is not None:
            return self.last_mitigation
        return MitigationResult(action_type=RecoveryAction.NO_ACTION.value, success=True)

    def get_policy_stats(self) -> Dict[str, Any]:
        return self.policy.get_stats()

    def reset(self) -> None:
        self.detector.reset()
        self.policy.reset()
        self.shadow_nodes.clear()
        self.last_diagnosis = None
        self.last_mitigation = None
        self.last_outcome = None
        self.in_fault_episode = False
        self.processed_records_count = 0
        self.anomalies_detected_count = 0
        self.recoveries_executed_count = 0
