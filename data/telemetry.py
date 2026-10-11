# Common Telemetry Record and Data Source Interface.
#
# This module defines the unified TelemetryRecord class and DataSource interface.
# All simulation data and real-world dataset feeds are converted into TelemetryRecord
# instances so downstream components (Detector, Causal Analyzer, Recovery Manager, Evaluator)
# remain completely decoupled from data source schemas.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Generator


@dataclass
class TelemetryRecord:
    # Standardized Telemetry Record representation.
    #
    # Fields marked Optional may be None if a specific dataset does not collect
    # or provide that telemetry dimension.
    timestamp: float
    device_id: str
    edge_node_id: str
    cpu_utilization: Optional[float] = None       # percentage (0.0 to 100.0)
    memory_utilization: Optional[float] = None    # percentage (0.0 to 100.0)
    network_utilization: Optional[float] = None   # percentage (0.0 to 100.0)
    latency: Optional[float] = None              # in milliseconds (ms)
    packet_loss: Optional[float] = None          # percentage (0.0 to 100.0)
    throughput: Optional[float] = None           # Mbps
    workload: Optional[float] = None             # requests / sec
    fault_label: int = 0                         # 0: Normal, 1: Fault / Anomaly
    fault_type: Optional[str] = "NONE"           # e.g., "CPU_OVERLOAD", "NONE", or raw attack category
    original_label: Optional[str] = "NORMAL"     # raw label string from dataset

    dataset_name: Optional[str] = "SIMULATION"   # e.g., "TON_IoT", "SIMULATION"
    source_file: Optional[str] = None           # e.g., "IoT_Fridge.csv"
    device_type: Optional[str] = None           # e.g., "IoT_Fridge", "IoT_Weather"
    raw_features: Dict[str, Any] = field(default_factory=dict) # Raw sensor payload key-values

    # Discrete SimPy Task Processing Performance Metrics (per edge node)
    task_arrivals: Optional[int] = None
    completed_tasks: Optional[int] = None
    successful_tasks: Optional[int] = None
    failed_tasks: Optional[int] = None
    queue_length: Optional[int] = None
    tasks_in_service: Optional[int] = None
    average_task_execution_time: Optional[float] = None
    task_success_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        # Convert record to dictionary representation.
        d = asdict(self)
        return d

    def get_task_metrics(self) -> Dict[str, Any]:
        # Extract per-node discrete task performance metrics.
        return {
            "task_arrivals": self.task_arrivals,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "queue_length": self.queue_length,
            "tasks_in_service": self.tasks_in_service,
            "average_task_execution_time": self.average_task_execution_time,
            "task_success_rate": self.task_success_rate,
        }

    def get_available_features(self) -> Dict[str, float]:
        # Extract numerical telemetry feature dimensions that are not None.
        # Returns a dictionary of feature_name -> float value.
        # Includes numerical raw sensor features when available.
        features = {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "network_utilization": self.network_utilization,
            "latency": self.latency,
            "packet_loss": self.packet_loss,
            "throughput": self.throughput,
            "workload": self.workload,
        }
        res = {k: float(v) for k, v in features.items() if v is not None}
        if self.raw_features:
            for k, v in self.raw_features.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    res[k] = float(v)
        return res

    def get_numeric_features(self) -> Dict[str, float]:
        # Get all numerical telemetry metrics available in record.
        return self.get_available_features()

    def to_feature_vector(self, feature_order: Optional[List[str]] = None) -> List[float]:
        # Convert available numerical features to a flat feature vector for ML models.
        if feature_order is None:
            feature_order = [
                "cpu_utilization",
                "memory_utilization",
                "network_utilization",
                "latency",
                "packet_loss",
                "throughput",
                "workload",
            ]
        vec = []
        num_feats = self.get_numeric_features()
        for feat in feature_order:
            if feat in num_feats:
                vec.append(num_feats[feat])
            else:
                val = getattr(self, feat, None)
                vec.append(float(val) if val is not None else 0.0)
        return vec

    @classmethod
    def get_feature_names(cls) -> List[str]:
        # Get standard list of numerical telemetry feature names.
        return [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
            "latency",
            "packet_loss",
            "throughput",
            "workload",
        ]


class DataSource(ABC):
    # Abstract Interface for Telemetry Data Sources.

    @abstractmethod
    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        # Yield TelemetryRecord instances sequentially (streaming).
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        # Return human-readable identifier of the data source.
        pass


