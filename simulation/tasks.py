# Discrete Task Model for SimPy Edge-Computing Simulation.
#
# Models individual discrete task requests arriving at edge nodes,
# tracking arrival time, service demands, queue wait times, execution durations,
# completion states, and failure reasons.

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Task:
    # Discrete individual compute task submitted to an Edge Node.
    task_id: str
    edge_node_id: str
    arrival_time: float
    base_service_time: float = 0.05  # seconds
    start_service_time: Optional[float] = None
    completion_time: Optional[float] = None
    success: bool = False
    failed: bool = False
    failure_reason: Optional[str] = None
    queue_waiting_time: float = 0.0
    execution_time: float = 0.0

    @property
    def turnaround_time(self) -> float:
        # Total latency experienced by the task (queue wait + execution time).
        if self.completion_time is not None:
            return max(0.0, self.completion_time - self.arrival_time)
        return self.queue_waiting_time + self.execution_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "edge_node_id": self.edge_node_id,
            "arrival_time": round(self.arrival_time, 4),
            "start_service_time": round(self.start_service_time, 4) if self.start_service_time is not None else None,
            "completion_time": round(self.completion_time, 4) if self.completion_time is not None else None,
            "success": self.success,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "queue_waiting_time": round(self.queue_waiting_time, 4),
            "execution_time": round(self.execution_time, 4),
            "turnaround_time": round(self.turnaround_time, 4),
        }
