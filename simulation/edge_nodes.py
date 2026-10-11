# Edge Node Simulation Model.
#
# Models physical resource capacity (CPU, Memory, Network), latency, packet loss,
# and correlated system state evolution under workload and injected faults.
# Supports persistent physical mitigation state following recovery action execution.

import math
import logging
import random
from typing import Dict, Any, Optional, List, Generator
# pyrefly: ignore [missing-import]
import simpy

from .tasks import Task

logger = logging.getLogger(__name__)


class EdgeNode:
    # Simulated Edge Node managing resource utilization, quality of service,
    # discrete SimPy task queue processing, and persistent recovery state modifications.

    def __init__(
        self,
        node_id: str,
        cpu_capacity: float = 100.0,
        memory_capacity: float = 100.0,
        network_capacity: float = 100.0,
        base_latency: float = 15.0,  # ms
        concurrency: int = 4,
        max_queue_size: Optional[int] = None,
        base_service_time: float = 0.05,
        env: Optional[simpy.Environment] = None,
    ):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.network_capacity = network_capacity
        self.base_latency = base_latency
        self.concurrency = concurrency
        self.max_queue_size = max_queue_size
        self.base_service_time = base_service_time

        self.env: Optional[simpy.Environment] = env
        self.resource: Optional[simpy.Resource] = None
        if self.env is not None:
            self.resource = simpy.Resource(self.env, capacity=self.concurrency)

        self.active: bool = True

        # State metrics
        self.workload: float = 20.0        # requests / sec
        self.cpu_utilization: float = 25.0    # %
        self.memory_utilization: float = 30.0 # %
        self.network_utilization: float = 20.0# %
        self.latency: float = base_latency    # ms
        self.packet_loss: float = 0.5         # %
        self.throughput: float = 50.0         # Mbps

        # Base fault offsets
        self.fault_cpu_offset: float = 0.0
        self.fault_mem_offset: float = 0.0
        self.fault_net_offset: float = 0.0
        self.fault_lat_offset: float = 0.0
        self.fault_loss_offset: float = 0.0
        self.node_failed: bool = False

        # Persistent Recovery Mitigation Factors (1.0 = unmitigated, < 1.0 = recovered)
        self.workload_mitigation: float = 1.0
        self.cpu_mitigation: float = 1.0
        self.mem_mitigation: float = 1.0
        self.net_mitigation: float = 1.0
        self.lat_mitigation: float = 1.0

        # Discrete Task Performance Metrics
        self.task_arrivals: int = 0
        self.completed_tasks: int = 0
        self.successful_tasks: int = 0
        self.failed_tasks: int = 0
        self._total_task_execution_time: float = 0.0
        self._executed_tasks_count: int = 0
        self._total_queue_waiting_time: float = 0.0
        self._active_tasks: List[Task] = []

    @property
    def queue_length(self) -> int:
        # Current number of tasks waiting in queue for resource acquisition.
        if self.resource is not None:
            return len(self.resource.queue)
        return 0

    @property
    def tasks_in_service(self) -> int:
        # Current number of tasks actively executing in resource service slots.
        if self.resource is not None:
            return self.resource.count
        return 0

    @property
    def average_task_execution_time(self) -> float:
        # Average execution duration of tasks that completed execution.
        if self._executed_tasks_count > 0:
            return self._total_task_execution_time / self._executed_tasks_count
        return 0.0

    @property
    def average_queue_waiting_time(self) -> float:
        # Average queue waiting time of tasks before acquiring resource.
        if self._executed_tasks_count > 0:
            return self._total_queue_waiting_time / self._executed_tasks_count
        return 0.0

    @property
    def task_success_rate(self) -> float:
        # Ratio of successful tasks to completed tasks.
        if self.completed_tasks > 0:
            return self.successful_tasks / self.completed_tasks
        return 1.0 if not self.node_failed else 0.0

    def update_metrics(self, current_workload: float):
        # Update physical system state using realistic causal equations:
        # Workload ↑ => CPU utilization ↑
        # CPU utilization ↑ => Latency ↑
        # Network congestion ↑ => Packet loss ↑ => Latency ↑
        if not self.active or self.node_failed:
            self.cpu_utilization = 0.0
            self.memory_utilization = 0.0
            self.network_utilization = 0.0
            self.latency = 999.0  # Offline / unreachable
            self.packet_loss = 100.0
            self.throughput = 0.0
            self.workload = 0.0
            return

        # Apply persistent workload offloading mitigation
        self.workload = current_workload * self.workload_mitigation

        # Apply active fault offsets scaled by persistent recovery mitigations
        eff_cpu_offset = self.fault_cpu_offset * self.cpu_mitigation
        eff_mem_offset = self.fault_mem_offset * self.mem_mitigation
        eff_net_offset = self.fault_net_offset * self.net_mitigation
        eff_lat_offset = self.fault_lat_offset * self.lat_mitigation
        eff_loss_offset = self.fault_loss_offset * self.net_mitigation

        # 1. CPU Utilization
        raw_cpu = 15.0 + (self.workload / self.cpu_capacity) * 50.0 + eff_cpu_offset
        self.cpu_utilization = max(0.0, min(100.0, raw_cpu))

        # 2. Memory Utilization
        raw_mem = 20.0 + (self.workload / self.memory_capacity) * 40.0 + eff_mem_offset
        self.memory_utilization = max(0.0, min(100.0, raw_mem))

        # 3. Network Utilization
        raw_net = 15.0 + (self.workload / self.network_capacity) * 45.0 + eff_net_offset
        self.network_utilization = max(0.0, min(100.0, raw_net))

        # 4. Throughput (Mbps)
        self.throughput = max(0.0, self.workload * 2.5 * (1.0 - self.network_utilization / 150.0))

        # 5. Packet Loss
        congestion_loss = max(0.0, (self.network_utilization - 75.0) * 0.8) if self.network_utilization > 75.0 else 0.0
        raw_loss = congestion_loss + eff_loss_offset
        self.packet_loss = max(0.0, min(100.0, raw_loss))

        # 6. Processing Latency
        cpu_delay = math.pow(self.cpu_utilization / 100.0, 2.0) * 120.0
        loss_delay = self.packet_loss * 5.0
        raw_lat = self.base_latency + cpu_delay + loss_delay + eff_lat_offset
        self.latency = max(1.0, raw_lat)

    def attach_env(self, env: simpy.Environment):
        # Attach SimPy environment and initialize resource.
        self.env = env
        self.resource = simpy.Resource(env, capacity=self.concurrency)

    def schedule_task(self, task: Task, delay: float = 0.0) -> Generator[simpy.Event, None, None]:
        # Generator scheduling task arrival after optional delay offset.
        if self.env is None:
            raise RuntimeError(f"EdgeNode '{self.node_id}' has no SimPy environment attached.")
        if delay > 0.0:
            yield self.env.timeout(delay)
        yield from self.handle_task(task)

    def handle_task(self, task: Task) -> Generator[simpy.Event, None, None]:
        # Process discrete task through node queue and SimPy resource.
        #
        # Task Failure Semantics:
        # 1. NODE_FAILED: Node is failed or inactive upon task arrival.
        # 2. PACKET_LOSS_DROP: Task dropped due to network packet loss on arrival.
        # 3. QUEUE_FULL: Node queue capacity exceeded when max_queue_size is set.
        # 4. NODE_CRASH_ABORTED: Node failed while task was waiting in queue.
        # 5. NODE_CRASH_IN_SERVICE: Node failed while task was being processed.
        if self.env is None or self.resource is None:
            raise RuntimeError(f"EdgeNode '{self.node_id}' has no SimPy environment/resource attached.")

        now = self.env.now
        task.arrival_time = now
        self.task_arrivals += 1
        self._active_tasks.append(task)

        # Failure Condition 1: Node offline or crashed on arrival
        if not self.active or self.node_failed:
            task.failed = True
            task.failure_reason = "NODE_FAILED"
            task.completion_time = now
            self.failed_tasks += 1
            self.completed_tasks += 1
            if task in self._active_tasks:
                self._active_tasks.remove(task)
            return

        # Failure Condition 2: Network packet loss drops incoming request
        if self.packet_loss > 0.0 and (random.random() * 100.0 < self.packet_loss):
            task.failed = True
            task.failure_reason = "PACKET_LOSS_DROP"
            task.completion_time = now
            self.failed_tasks += 1
            self.completed_tasks += 1
            if task in self._active_tasks:
                self._active_tasks.remove(task)
            return

        # Failure Condition 3: Queue capacity full
        if self.max_queue_size is not None and len(self.resource.queue) >= self.max_queue_size:
            task.failed = True
            task.failure_reason = "QUEUE_FULL"
            task.completion_time = now
            self.failed_tasks += 1
            self.completed_tasks += 1
            if task in self._active_tasks:
                self._active_tasks.remove(task)
            return

        req = self.resource.request()
        try:
            yield req
        except simpy.Interrupt as interrupt:
            if not task.failed:
                task.failed = True
                task.failure_reason = f"NODE_CRASH_{interrupt.cause}"
                task.completion_time = self.env.now
                self.failed_tasks += 1
                self.completed_tasks += 1
            if task in self._active_tasks:
                self._active_tasks.remove(task)
            return

        # Task has acquired processing resource
        start_service_time = self.env.now
        task.start_service_time = start_service_time
        task.queue_waiting_time = max(0.0, start_service_time - task.arrival_time)
        self._total_queue_waiting_time += task.queue_waiting_time

        # Failure Condition 4: Node crashed while task was queued
        if not self.active or self.node_failed:
            self.resource.release(req)
            if not task.failed:
                task.failed = True
                task.failure_reason = "NODE_CRASH_ABORTED"
                task.completion_time = self.env.now
                self.failed_tasks += 1
                self.completed_tasks += 1
            if task in self._active_tasks:
                self._active_tasks.remove(task)
            return

        # Calculate task execution duration
        exec_time = self._calculate_execution_time(task)

        try:
            yield self.env.timeout(exec_time)

            # Failure Condition 5: Node crashed while in service
            if not self.active or self.node_failed:
                if not task.failed:
                    task.failed = True
                    task.failure_reason = "NODE_CRASH_IN_SERVICE"
                    self.failed_tasks += 1
                    self.completed_tasks += 1
            else:
                task.success = True
                self.successful_tasks += 1
                self.completed_tasks += 1

            task.completion_time = self.env.now
            task.execution_time = max(0.0, task.completion_time - start_service_time)
            self._total_task_execution_time += task.execution_time
            self._executed_tasks_count += 1

        except simpy.Interrupt as interrupt:
            if not task.failed:
                task.failed = True
                task.failure_reason = f"NODE_CRASH_{interrupt.cause}"
                task.completion_time = self.env.now
                task.execution_time = max(0.0, task.completion_time - start_service_time)
                self._total_task_execution_time += task.execution_time
                self._executed_tasks_count += 1
                self.failed_tasks += 1
                self.completed_tasks += 1
        finally:
            self.resource.release(req)
            if task in self._active_tasks:
                self._active_tasks.remove(task)

    def _calculate_execution_time(self, task: Task) -> float:
        # Task execution time influenced by CPU utilization.
        # Heavy CPU utilization / CPU overload faults stretch execution duration.
        cpu_slowdown = 1.0 + (self.cpu_utilization / 100.0) * 1.5
        return max(0.001, task.base_service_time * cpu_slowdown)

    def _abort_active_tasks(self, reason: str = "NODE_FAILED"):
        # Abort all currently waiting or in-service tasks when node fails.
        now = self.env.now if self.env else 0.0
        for task in list(self._active_tasks):
            if not task.failed and not task.success:
                task.failed = True
                task.failure_reason = f"NODE_CRASH_{reason}"
                task.completion_time = now
                self.failed_tasks += 1
                self.completed_tasks += 1
        self._active_tasks.clear()

    def reset_task_metrics(self):
        # Reset discrete task tracking counters.
        self.task_arrivals = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self._total_task_execution_time = 0.0
        self._executed_tasks_count = 0
        self._total_queue_waiting_time = 0.0
        self._active_tasks.clear()

    def get_task_metrics(self) -> Dict[str, Any]:
        return {
            "task_arrivals": self.task_arrivals,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "queue_length": self.queue_length,
            "tasks_in_service": self.tasks_in_service,
            "average_task_execution_time": round(self.average_task_execution_time, 4),
            "average_queue_waiting_time": round(self.average_queue_waiting_time, 4),
            "task_success_rate": round(self.task_success_rate, 4),
        }

    def apply_fault(
        self,
        fault_type_str: str,
        severity: float = 1.0,
    ):
        # Apply base fault offsets.
        if fault_type_str == "CPU_OVERLOAD":
            self.fault_cpu_offset = 60.0 * severity
        elif fault_type_str == "MEMORY_OVERLOAD":
            self.fault_mem_offset = 65.0 * severity
        elif fault_type_str == "NETWORK_CONGESTION":
            self.fault_net_offset = 60.0 * severity
            self.fault_loss_offset = 15.0 * severity
        elif fault_type_str == "HIGH_LATENCY":
            self.fault_lat_offset = 150.0 * severity
        elif fault_type_str == "PACKET_LOSS":
            self.fault_loss_offset = 35.0 * severity
        elif fault_type_str == "EDGE_NODE_FAILURE":
            self.node_failed = True
            self._abort_active_tasks("CRASH")

    def clear_faults(self):
        # Reset fault offsets and recovery mitigation factors to baseline.
        self.fault_cpu_offset = 0.0
        self.fault_mem_offset = 0.0
        self.fault_net_offset = 0.0
        self.fault_lat_offset = 0.0
        self.fault_loss_offset = 0.0
        self.node_failed = False

        # Reset mitigations
        self.workload_mitigation = 1.0
        self.cpu_mitigation = 1.0
        self.mem_mitigation = 1.0
        self.net_mitigation = 1.0
        self.lat_mitigation = 1.0

    def apply_recovery_action(self, action: str) -> bool:
        # Execute adaptive recovery action establishing persistent mitigation factors.
        # Returns True if action was executed successfully.

        logger.info(f"[Node {self.node_id}] Applying persistent recovery action: {action}")

        if action == "WORKLOAD_REDISTRIBUTION":
            # Offload 60% workload, reduce CPU fault impact by 70%
            self.workload_mitigation = 0.4
            self.cpu_mitigation = 0.3

        elif action == "TRAFFIC_REROUTING":
            # Reroute traffic, reduce network & packet loss impact by 80%
            self.net_mitigation = 0.2
            self.lat_mitigation = 0.3

        elif action == "TASK_MIGRATION":
            # Migrate memory/CPU tasks off node
            self.mem_mitigation = 0.2
            self.cpu_mitigation = 0.4
            self.workload_mitigation = 0.5

        elif action == "EDGE_NODE_FAILOVER":
            # Restart or failover node
            self.node_failed = False
            self.clear_faults()
            self.active = True
            if self.env is not None:
                self.resource = simpy.Resource(self.env, capacity=self.concurrency)

        elif action == "RESOURCE_REBALANCING":
            # Dynamically allocate extra virtual CPU/RAM capacity
            self.cpu_mitigation = 0.5
            self.mem_mitigation = 0.5
            self.lat_mitigation = 0.4

        # Re-evaluate metrics immediately after recovery
        self.update_metrics(self.workload / max(1e-5, self.workload_mitigation))
        return True

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "active": self.active,
            "cpu_utilization": round(self.cpu_utilization, 2),
            "memory_utilization": round(self.memory_utilization, 2),
            "network_utilization": round(self.network_utilization, 2),
            "latency": round(self.latency, 2),
            "packet_loss": round(self.packet_loss, 2),
            "throughput": round(self.throughput, 2),
            "workload": round(self.workload, 2),
            "task_arrivals": self.task_arrivals,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "queue_length": self.queue_length,
            "tasks_in_service": self.tasks_in_service,
            "average_task_execution_time": round(self.average_task_execution_time, 4),
            "task_success_rate": round(self.task_success_rate, 4),
        }
