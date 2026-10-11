# Unit and Integration Tests for SimPy Discrete Task Processing and Per-Node Metrics.
#
# Validates:
# 1. SimPy task queue modeling and concurrency constraints.
# 2. Distinction between queue waiting time, task execution time, and physical latency.
# 3. Calculation of all required per-node task metrics (arrivals, completed, successful,
#    failed, queue length, in service, average execution time, success rate).
# 4. Task failure semantics under node crash, packet loss, and queue exhaustion.
# 5. Fault injection (EDGE_NODE_FAILURE, CPU_OVERLOAD, PACKET_LOSS) and recovery (EDGE_NODE_FAILOVER).
# 6. Telemetry compatibility preserving 7-feature PyTorch autoencoder vector and causal graph.

import unittest
import simpy
import numpy as np

from simulation.tasks import Task
from simulation.edge_nodes import EdgeNode
from simulation.edge_simulation import EdgeSimulation
from data.telemetry import TelemetryRecord
from detection.pytorch_detector import PyTorchFaultDetector
from causal.causal_graph import SystemCausalGraph
from causal.causal_analysis import CausalAnalyzer


class TestDiscreteTaskProcessing(unittest.TestCase):
    # Test suite verifying discrete individual task processing in SimPy edge simulation.

    def setUp(self):
        self.env = simpy.Environment()
        self.node = EdgeNode(
            node_id="Test_Node_1",
            cpu_capacity=100.0,
            concurrency=2,
            base_service_time=0.04,
            base_latency=15.0,
            env=self.env,
        )

    def test_single_task_successful_lifecycle(self):
        # Verify a single task arrives, queues, executes, and completes successfully.
        task = Task(
            task_id="Task_001",
            edge_node_id=self.node.node_id,
            arrival_time=0.0,
            base_service_time=0.05,
        )

        self.env.process(self.node.handle_task(task))
        self.env.run(until=1.0)

        self.assertTrue(task.success)
        self.assertFalse(task.failed)
        self.assertIsNone(task.failure_reason)
        self.assertEqual(task.arrival_time, 0.0)
        self.assertEqual(task.start_service_time, 0.0)
        self.assertGreater(task.execution_time, 0.0)
        self.assertGreaterEqual(task.completion_time, task.execution_time)
        self.assertEqual(task.queue_waiting_time, 0.0)

        # Node metrics
        self.assertEqual(self.node.task_arrivals, 1)
        self.assertEqual(self.node.completed_tasks, 1)
        self.assertEqual(self.node.successful_tasks, 1)
        self.assertEqual(self.node.failed_tasks, 0)
        self.assertEqual(self.node.queue_length, 0)
        self.assertEqual(self.node.tasks_in_service, 0)
        self.assertAlmostEqual(self.node.task_success_rate, 1.0)
        self.assertGreater(self.node.average_task_execution_time, 0.0)

    def test_concurrency_and_queue_waiting_time(self):
        # Concurrency is 2. When 3 concurrent tasks arrive at t=0, the 3rd must wait in queue.
        task1 = Task("T1", self.node.node_id, 0.0, base_service_time=0.10)
        task2 = Task("T2", self.node.node_id, 0.0, base_service_time=0.10)
        task3 = Task("T3", self.node.node_id, 0.0, base_service_time=0.10)

        self.env.process(self.node.handle_task(task1))
        self.env.process(self.node.handle_task(task2))
        self.env.process(self.node.handle_task(task3))

        # Advance by 0.05s: T1 and T2 in service, T3 waiting in queue
        self.env.run(until=0.05)
        self.assertEqual(self.node.tasks_in_service, 2)
        self.assertEqual(self.node.queue_length, 1)

        # Advance until all complete
        self.env.run(until=1.0)
        self.assertEqual(self.node.tasks_in_service, 0)
        self.assertEqual(self.node.queue_length, 0)
        self.assertEqual(self.node.completed_tasks, 3)
        self.assertEqual(self.node.successful_tasks, 3)

        # T1 and T2 had zero queue waiting time; T3 waited ~0.10s
        self.assertEqual(task1.queue_waiting_time, 0.0)
        self.assertEqual(task2.queue_waiting_time, 0.0)
        self.assertGreaterEqual(task3.queue_waiting_time, 0.09)
        self.assertGreater(self.node.average_queue_waiting_time, 0.0)

    def test_distinction_between_latencies(self):
        # Validates explicit separation of:
        # 1. Queue waiting time (time in queue before resource acquisition)
        # 2. Task execution time (time actively computing on the node)
        # 3. Existing physical/network latency (node.latency in ms)
        self.node.update_metrics(current_workload=30.0)
        physical_latency_ms = self.node.latency  # in ms (e.g., > 15 ms)

        t1 = Task("T1", self.node.node_id, 0.0, base_service_time=0.05)
        t2 = Task("T2", self.node.node_id, 0.0, base_service_time=0.05)
        t3 = Task("T3", self.node.node_id, 0.0, base_service_time=0.05)

        self.env.process(self.node.handle_task(t1))
        self.env.process(self.node.handle_task(t2))
        self.env.process(self.node.handle_task(t3))

        self.env.run(until=0.5)

        # Assert physical latency is distinct and in ms
        self.assertGreaterEqual(physical_latency_ms, 15.0)

        # Assert execution times are in seconds (e.g. ~0.05s)
        self.assertLess(t1.execution_time, 1.0)
        self.assertGreater(t1.execution_time, 0.0)

        # Assert waiting time for queued task is distinct from execution time
        self.assertGreater(t3.queue_waiting_time, 0.0)
        self.assertEqual(t3.turnaround_time, t3.queue_waiting_time + t3.execution_time)

    def test_queue_overflow_drop(self):
        # When max_queue_size is enforced, excess task arrivals must fail with QUEUE_FULL.
        bounded_node = EdgeNode(
            node_id="Bounded_Node",
            concurrency=1,
            max_queue_size=1,
            base_service_time=0.10,
            env=self.env,
        )

        t1 = Task("T1", bounded_node.node_id, 0.0, base_service_time=0.10)
        t2 = Task("T2", bounded_node.node_id, 0.0, base_service_time=0.10)
        t3 = Task("T3", bounded_node.node_id, 0.0, base_service_time=0.10)

        # T1 takes service, T2 enters queue (queue_length=1), T3 dropped (QUEUE_FULL)
        self.env.process(bounded_node.handle_task(t1))
        self.env.process(bounded_node.handle_task(t2))
        self.env.process(bounded_node.handle_task(t3))

        self.env.run(until=0.5)

        self.assertEqual(t3.failed, True)
        self.assertEqual(t3.failure_reason, "QUEUE_FULL")
        self.assertEqual(bounded_node.failed_tasks, 1)
        self.assertEqual(bounded_node.successful_tasks, 2)
        self.assertEqual(bounded_node.completed_tasks, 3)
        self.assertAlmostEqual(bounded_node.task_success_rate, 2.0 / 3.0)

    def test_node_failure_and_recovery_semantics(self):
        # 1. Healthy arrivals succeed
        t1 = Task("T1", self.node.node_id, 0.0, base_service_time=0.02)
        self.env.process(self.node.handle_task(t1))
        self.env.run(until=0.05)
        self.assertTrue(t1.success)
        self.assertEqual(self.node.successful_tasks, 1)

        # 2. Inject EDGE_NODE_FAILURE
        self.node.apply_fault("EDGE_NODE_FAILURE")
        self.assertTrue(self.node.node_failed)

        # Arriving task on failed node must fail with NODE_FAILED
        t2 = Task("T2", self.node.node_id, 0.05, base_service_time=0.02)
        self.env.process(self.node.handle_task(t2))
        self.env.run(until=0.10)

        self.assertTrue(t2.failed)
        self.assertEqual(t2.failure_reason, "NODE_FAILED")
        self.assertEqual(self.node.failed_tasks, 1)

        # 3. Apply recovery EDGE_NODE_FAILOVER
        recovered = self.node.apply_recovery_action("EDGE_NODE_FAILOVER")
        self.assertTrue(recovered)
        self.assertFalse(self.node.node_failed)

        # New arrival succeeds post-recovery
        t3 = Task("T3", self.node.node_id, 0.10, base_service_time=0.02)
        self.env.process(self.node.handle_task(t3))
        self.env.run(until=0.20)

        self.assertTrue(t3.success)
        self.assertEqual(self.node.successful_tasks, 2)
        self.assertEqual(self.node.failed_tasks, 1)
        self.assertEqual(self.node.completed_tasks, 3)
        self.assertAlmostEqual(self.node.task_success_rate, 2.0 / 3.0)

    def test_packet_loss_drop_semantics(self):
        # Node with 100% packet loss must drop all arriving tasks with PACKET_LOSS_DROP.
        lossy_node = EdgeNode(
            node_id="Lossy_Node",
            concurrency=2,
            base_service_time=0.02,
            env=self.env,
        )
        lossy_node.packet_loss = 100.0

        t = Task("T_loss", lossy_node.node_id, 0.0, base_service_time=0.02)
        self.env.process(lossy_node.handle_task(t))
        self.env.run(until=0.1)

        self.assertTrue(t.failed)
        self.assertEqual(t.failure_reason, "PACKET_LOSS_DROP")
        self.assertEqual(lossy_node.failed_tasks, 1)
        self.assertEqual(lossy_node.successful_tasks, 0)
        self.assertEqual(lossy_node.task_success_rate, 0.0)

    def test_cpu_overload_execution_time_inflation(self):
        # Under CPU overload, task execution duration is inflated.
        normal_task = Task("T_norm", self.node.node_id, 0.0, base_service_time=0.04)
        self.node.cpu_utilization = 15.0
        norm_exec = self.node._calculate_execution_time(normal_task)

        overload_task = Task("T_over", self.node.node_id, 0.0, base_service_time=0.04)
        self.node.cpu_utilization = 95.0
        over_exec = self.node._calculate_execution_time(overload_task)

        self.assertGreater(over_exec, norm_exec)


class TestEdgeSimulationTaskIntegration(unittest.TestCase):
    # Test suite verifying full EdgeSimulation discrete task generation and telemetry recording.

    def setUp(self):
        self.sim = EdgeSimulation(num_nodes=3, num_devices_per_node=2, step_size=1.0, seed=42)

    def test_step_generates_discrete_tasks_and_records_metrics(self):
        # Step simulation and check that discrete tasks are processed across all nodes.
        records = self.sim.step()
        self.assertEqual(len(records), 3)

        for rec in records:
            # Per-node task metrics must be present and non-negative
            self.assertIsNotNone(rec.task_arrivals)
            self.assertGreater(rec.task_arrivals, 0)
            self.assertIsNotNone(rec.completed_tasks)
            self.assertGreater(rec.completed_tasks, 0)
            self.assertIsNotNone(rec.successful_tasks)
            self.assertGreater(rec.successful_tasks, 0)
            self.assertIsNotNone(rec.queue_length)
            self.assertIsNotNone(rec.tasks_in_service)
            self.assertIsNotNone(rec.average_task_execution_time)
            self.assertGreater(rec.average_task_execution_time, 0.0)
            self.assertIsNotNone(rec.task_success_rate)
            self.assertGreaterEqual(rec.task_success_rate, 0.0)
            self.assertLessEqual(rec.task_success_rate, 1.0)

            # Task accounting relations
            self.assertEqual(rec.completed_tasks, rec.successful_tasks + rec.failed_tasks)

    def test_telemetry_seven_features_preserved(self):
        # Crucial requirement: TelemetryRecord must strictly preserve the 7 detector features
        # and their ordering for PyTorch Autoencoder and Causal Analysis compatibility.
        records = self.sim.step()
        rec = records[0]

        expected_feature_names = [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
            "latency",
            "packet_loss",
            "throughput",
            "workload",
        ]

        self.assertEqual(TelemetryRecord.get_feature_names(), expected_feature_names)
        self.assertEqual(list(rec.get_numeric_features().keys()), expected_feature_names)

        feature_vector = rec.to_feature_vector()
        self.assertEqual(len(feature_vector), 7)

        # PyTorch Autoencoder Inference Compatibility
        detector = PyTorchFaultDetector(threshold=0.15)
        detector.train_on_baseline(records, epochs=5)
        inference_result = detector.process_record(rec)

        self.assertIn("is_fault", inference_result)
        self.assertIn("anomaly_score", inference_result)
        self.assertIn("detector", inference_result)
        self.assertEqual(inference_result["detector"], "PyTorch_Autoencoder")

    def test_causal_graph_and_analyzer_compatibility(self):
        # Verify causal DAG and Analyzer run without failure on telemetry records.
        records = []
        for _ in range(12):
            records.extend(self.sim.step())

        causal_graph = SystemCausalGraph(mode="simulation")
        self.assertIn("workload", causal_graph.get_graph().nodes())
        self.assertIn("cpu_utilization", causal_graph.get_graph().nodes())
        self.assertIn("latency", causal_graph.get_graph().nodes())

        analyzer = CausalAnalyzer(causal_graph=causal_graph)
        result = analyzer.analyze_fault_cause(records, target_node_id="Edge_Node_1")

        self.assertIsNotNone(result.root_cause)
        self.assertIn(result.causal_status, ["CAUSAL_INFERENCE", "HEURISTIC", "INSUFFICIENT_EVIDENCE"])

    def test_node_failure_injection_in_simulation(self):
        # Register EDGE_NODE_FAILURE on Edge_Node_2, verify failure metrics.
        self.sim.register_active_fault("Edge_Node_2", "EDGE_NODE_FAILURE", severity=1.0)
        records = self.sim.step()

        rec2 = [r for r in records if r.edge_node_id == "Edge_Node_2"][0]
        self.assertEqual(rec2.fault_label, 1)
        self.assertEqual(rec2.fault_type, "EDGE_NODE_FAILURE")
        self.assertEqual(rec2.latency, 999.0)
        self.assertEqual(rec2.successful_tasks, 0)
        self.assertGreater(rec2.failed_tasks, 0)
        self.assertEqual(rec2.task_success_rate, 0.0)

        # Recover Edge_Node_2
        self.sim.apply_recovery("Edge_Node_2", "EDGE_NODE_FAILOVER")
        records_after = self.sim.step()
        rec2_after = [r for r in records_after if r.edge_node_id == "Edge_Node_2"][0]

        # Post-recovery successful tasks resume
        self.assertGreater(rec2_after.successful_tasks, 0)
        self.assertGreater(rec2_after.task_success_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
