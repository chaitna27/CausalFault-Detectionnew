# Proposed Causal Adaptive Fault-Tolerance Pipeline.
#
# Integrates:
# 1. Online streaming anomaly detection (StreamingCausalDetector)
# 2. Structural Equation Model causal root-cause analysis (CausalInferenceEngine)
# 3. Closed-loop adaptive recovery policy (AdaptiveCausalRecoveryPolicy)
# 4. Bounded state management (CausalFTState)

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
    MitigationResult,
)
from evaluation.context import BenchmarkInput
from .state import CausalFTState
from .detector import StreamingCausalDetector, DetectorConfig
from .causal_engine import CausalInferenceEngine, CausalGraphSpecification
from .recovery_policy import AdaptiveCausalRecoveryPolicy


@dataclass
class CausalFTResult:
    # Standardized composite result for Causal Fault-Tolerance pipeline.
    detection: DetectionResult
    diagnosis: Optional[RCADiagnosisResult] = None
    mitigation: Optional[MitigationResult] = None
    stream_position: int = 0
    timestamp: float = 0.0


import warnings

KNOWN_CONFIG_KEYS = {
    "detection_threshold",  # Canonical threshold key
    "threshold",            # Backward-compatible alias
    "upper_threshold",      # Harness alias
    "window_size",          # State buffer window size
}


class CausalFaultTolerancePipeline(BaseFaultToleranceAlgorithm):
    # Main proposed framework algorithm implementing end-to-end causal fault tolerance.

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self.state = CausalFTState(window_size=self._config.get("window_size", 100))
        self.detector = StreamingCausalDetector(
            DetectorConfig(threshold=self._config.get("detection_threshold", 0.65))
        )
        self.causal_engine = CausalInferenceEngine()
        self.recovery_policy = AdaptiveCausalRecoveryPolicy(self.state)

        self.last_diagnosis: Optional[RCADiagnosisResult] = None
        self.last_mitigation: Optional[MitigationResult] = None
        self.is_running: bool = False

    @property
    def name(self) -> str:
        return "Proposed Causal Fault-Tolerance"

    @property
    def paper_id(self) -> str:
        return "proposed_causal_ft"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.STREAMING_DETECTION,
            AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS,
            AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE,
        }

    def initialize(self, config: Any = None) -> None:
        # Initialize models, hyperparameters, and internal structures.
        # Canonical threshold key: 'detection_threshold'.
        # Backward-compatible aliases: 'threshold', 'upper_threshold'.
        if config is not None and isinstance(config, dict):
            for k in config.keys():
                if k not in KNOWN_CONFIG_KEYS:
                    warnings.warn(
                        f"Unknown configuration key '{k}' provided to CausalFaultTolerancePipeline. "
                        f"Supported keys: {sorted(list(KNOWN_CONFIG_KEYS))}",
                        UserWarning,
                        stacklevel=2,
                    )
            self._config.update(config)

            # Resolve threshold using canonical key first, then aliases
            thresh = config.get("detection_threshold")
            if thresh is None:
                thresh = config.get("threshold")
            if thresh is None:
                thresh = config.get("upper_threshold")

            if thresh is not None:
                self.detector.config.threshold = float(thresh)

            if "window_size" in config:
                self.state = CausalFTState(window_size=int(config["window_size"]))
        self.reset()

    def fit(self, training_data: Any = None) -> None:
        # Fit detector baseline statistics and learn SEM structural causal equations
        # using unlabeled training telemetry.
        # Never accesses labels or test records.
        if training_data is None:
            return

        if isinstance(training_data, np.ndarray):
            X_train = training_data
            feat_names = [f"f_{i}" for i in range(X_train.shape[1])]
        elif hasattr(training_data, "values") and hasattr(training_data, "columns"):
            X_train = training_data.values
            feat_names = list(training_data.columns)
        else:
            return

        # 1. Fit detector baseline
        self.detector.fit(X_train)

        # 2. Fit causal SEM equations on normal training telemetry
        self.causal_engine.fit_structural_equations(X_train, feat_names)

    def start(self) -> None:
        # Invoked before the streaming evaluation starts.
        self.is_running = True

    def stop(self) -> None:
        # Invoked after the streaming evaluation completes.
        self.is_running = False

    def process(self, record: BenchmarkInput) -> DetectionResult:
        # Process a single streaming observation BenchmarkInput.
        # Coordinates detection -> causal RCA -> adaptive recovery.
        # 1. Real-time streaming detection
        det_result = self.detect(record)

        diagnosis: Optional[RCADiagnosisResult] = None
        mitigation: Optional[MitigationResult] = None

        # 2. Conditional Causal RCA if fault is detected
        if det_result.is_anomaly:
            diagnosis = self.causal_engine.diagnose(record, top_k=5)
            self.last_diagnosis = diagnosis

            # 3. Closed-loop adaptive recovery decision
            mitigation = self.recovery_policy.decide_recovery(record, diagnosis)
            self.last_mitigation = mitigation

        # Update bounded state
        self.state.record_observation(
            position=record.stream_position,
            timestamp=record.timestamp,
            feature_vec=record.feature_vector,
            anomaly_score=det_result.anomaly_score,
            is_anomaly=det_result.is_anomaly,
        )

        # Attach RCA and mitigation artifacts into raw_output for evaluation consumers
        det_result.raw_output["diagnosis"] = diagnosis
        det_result.raw_output["mitigation"] = mitigation

        return det_result

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        # Execute streaming anomaly detection on BenchmarkInput.
        return self.detector.detect(record)

    def diagnose_root_cause(
        self,
        normal_window: Any = None,
        anomalous_window: Any = None,
    ) -> RCADiagnosisResult:
        # Executes root cause localization.
        # If windows are provided, uses anomalous record; otherwise diagnoses latest state.
        if anomalous_window is not None:
            if isinstance(anomalous_window, BenchmarkInput):
                return self.causal_engine.diagnose(anomalous_window)
            elif isinstance(anomalous_window, dict):
                # Construct synthetic BenchmarkInput for query
                dummy_input = BenchmarkInput(
                    stream_position=0,
                    timestamp=0.0,
                    timestamp_str="",
                    device_id="query",
                    edge_node_id="query",
                    features=anomalous_window,
                    feature_vector=np.array(list(anomalous_window.values())),
                )
                return self.causal_engine.diagnose(dummy_input)

        if self.last_diagnosis is not None:
            return self.last_diagnosis

        return RCADiagnosisResult(
            ranked_root_causes=[],
            confidence_scores={},
            p_values={},
            execution_time_ms=0.0,
            metadata={"status": "NO_ACTIVE_ANOMALY"},
        )

    def execute_mitigation(self, context: Any = None) -> MitigationResult:
        # Executes or queries the latest mitigation action.
        if self.last_mitigation is not None:
            return self.last_mitigation
        return MitigationResult(action_type="NO_ACTION", success=True)

    def reset(self) -> None:
        # Reset internal states, ring buffers, and counters.
        self.state.reset()
        self.detector.reset()
        self.last_diagnosis = None
        self.last_mitigation = None
