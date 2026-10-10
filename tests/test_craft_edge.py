# Unit and Integration Tests for CRAFT-Edge Integration Layer.
#
# Tests:
# 1. Base interface contract and capability validation
# 2. Adaptive policy action selection (exploitation vs exploration)
# 3. Unseen action initialization
# 4. Outcome evaluation and reward calculation (benefit - cost)
# 5. Online Q-learning efficacy updates: Q_new = Q_old + alpha * (reward - Q_old)
# 6. Shadow edge node state evolution and recovery mitigations
# 7. End-to-end closed-loop pipeline execution (Telemetry -> Detect -> RCA -> Act -> Eval -> Update)
# 8. Reset functionality and state isolation

import unittest
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
    MitigationResult,
)
from algorithms.craft_edge import (
    CRAFTEdgePipeline,
    CRAFTEdgeConfig,
    AdaptiveEfficacyPolicy,
    RecoveryAction,
    ACTION_OPERATIONAL_COSTS,
    classify_root_cause,
    RecoveryOutcomeEvaluator,
    RecoveryOutcome,
    ShadowNodeState,
)
from evaluation.context import BenchmarkInput


class TestCRAFTEdgePolicy(unittest.TestCase):
    # Test suite for AdaptiveEfficacyPolicy and online learning updates.

    def setUp(self):
        self.policy = AdaptiveEfficacyPolicy(
            learning_rate=0.2,
            epsilon=0.10,
            epsilon_min=0.01,
            epsilon_decay=0.99,
            default_q=0.50,
            seed=42,
        )

    def test_unseen_action_initialization(self):
        # Verify unseen state-action pair initializes to expected prior/default Q-value.
        q_val = self.policy.get_q_value("GENERIC_RESOURCE_FAULT", RecoveryAction.NO_ACTION.value)
        self.assertEqual(q_val, 0.50)
        self.assertIn(("GENERIC_RESOURCE_FAULT", RecoveryAction.NO_ACTION.value), self.policy.q_table)

    def test_domain_prior_initialization(self):
        # Verify domain priors are applied for established logical pairings.
        q_cpu = self.policy.get_q_value("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value)
        self.assertEqual(q_cpu, 0.65)

    def test_action_selection_exploitation(self):
        # In exploitation mode (explore=False), policy must choose the argmax Q action.
        self.policy.set_q_value("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.90)
        self.policy.set_q_value("CPU_OVERLOAD", RecoveryAction.RESOURCE_REBALANCING.value, 0.40)
        self.policy.set_q_value("CPU_OVERLOAD", RecoveryAction.NO_ACTION.value, 0.10)

        action, is_explored = self.policy.select_action(
            "CPU_OVERLOAD",
            explore=False,
            candidate_actions=[
                RecoveryAction.WORKLOAD_REDISTRIBUTION.value,
                RecoveryAction.RESOURCE_REBALANCING.value,
                RecoveryAction.NO_ACTION.value,
            ]
        )
        self.assertEqual(action, RecoveryAction.WORKLOAD_REDISTRIBUTION.value)
        self.assertFalse(is_explored)

    def test_action_selection_exploration(self):
        # With epsilon=1.0 (pure exploration), policy chooses randomly among candidates.
        self.policy.epsilon = 1.0
        actions_chosen = set()
        for _ in range(50):
            act, is_explored = self.policy.select_action("NETWORK_CONGESTION", explore=True)
            self.assertTrue(is_explored)
            actions_chosen.add(act)

        # Multiple distinct actions should have been explored
        self.assertGreater(len(actions_chosen), 1)

    def test_online_q_update_equation(self):
        # Validates exact math: Q_new = Q_old + alpha * (reward - Q_old)
        cause = "CPU_OVERLOAD"
        action = RecoveryAction.WORKLOAD_REDISTRIBUTION.value
        self.policy.set_q_value(cause, action, 0.50)
        alpha = self.policy.alpha  # 0.2
        reward = 0.80

        expected_q = 0.50 + alpha * (reward - 0.50)  # 0.50 + 0.2 * 0.30 = 0.56
        q_new = self.policy.update(cause, action, reward=reward, success=True)

        self.assertAlmostEqual(q_new, expected_q, places=4)
        self.assertAlmostEqual(self.policy.get_q_value(cause, action), expected_q, places=4)
        self.assertEqual(self.policy.action_stats[(cause, action)]["attempts"], 1)
        self.assertEqual(self.policy.action_stats[(cause, action)]["successes"], 1)

    def test_q_convergence_under_repeated_rewards(self):
        # Repeated positive rewards converge Q towards reward value.
        cause = "MEMORY_OVERLOAD"
        action = RecoveryAction.TASK_MIGRATION.value
        self.policy.set_q_value(cause, action, 0.50)

        for _ in range(25):
            self.policy.update(cause, action, reward=0.90, success=True)

        self.assertGreater(self.policy.get_q_value(cause, action), 0.85)

    def test_q_decrease_under_negative_rewards(self):
        # Repeated penalties decrease Q-value.
        cause = "TRANSPORT_FLOOD"
        action = RecoveryAction.NO_ACTION.value
        self.policy.set_q_value(cause, action, 0.50)

        for _ in range(25):
            self.policy.update(cause, action, reward=-0.60, success=False)

        self.assertLess(self.policy.get_q_value(cause, action), -0.40)

    def test_configurable_epsilon_decay_interval(self):
        # Epsilon decay triggered strictly every N updates, avoiding burst collapse
        policy = AdaptiveEfficacyPolicy(
            epsilon=0.20,
            epsilon_decay=0.90,
            decay_interval=3,
            decay_mode="interval",
        )
        self.assertEqual(policy.epsilon, 0.20)

        # Updates 1 and 2: epsilon must remain completely fixed
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        self.assertEqual(policy.epsilon, 0.20)
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        self.assertEqual(policy.epsilon, 0.20)

        # Update 3: 3rd update reaches interval -> decays by 0.90 (0.20 * 0.90 = 0.18)
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        self.assertAlmostEqual(policy.epsilon, 0.18, places=4)

        # Updates 4 and 5: stays at 0.18
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        self.assertAlmostEqual(policy.epsilon, 0.18, places=4)

        # Update 6: 6th update reaches interval -> decays to 0.18 * 0.90 = 0.162
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5)
        self.assertAlmostEqual(policy.epsilon, 0.162, places=4)

    def test_configurable_epsilon_decay_episode(self):
        # Rapid consecutive anomaly updates within single episode do not decay epsilon
        policy = AdaptiveEfficacyPolicy(
            epsilon=0.20,
            epsilon_decay=0.90,
            decay_interval=1,
            decay_mode="episode",
        )
        # 10 consecutive updates during an active anomaly burst
        for _ in range(10):
            policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, 0.5, is_episode_end=False)
            self.assertEqual(policy.epsilon, 0.20)

        # Episode completion signals resolution and triggers decay
        policy.step_episode()
        self.assertAlmostEqual(policy.epsilon, 0.18, places=4)
        self.assertEqual(policy.episode_count, 1)


class TestCRAFTEdgeEvaluator(unittest.TestCase):
    # Test suite for RecoveryOutcomeEvaluator and reward calculation.

    def setUp(self):
        self.evaluator = RecoveryOutcomeEvaluator(
            benefit_weight=1.0,
            cost_weight=0.30,
            success_threshold_ratio=0.08,
        )

    def test_successful_metric_reduction_reward(self):
        # Pre-CPU: 90%, Post-CPU: 30% -> Relative drop: (90 - 30) / 90 = 0.667
        # Workload redistribution cost: 0.20
        # Expected reward: 1.0 * 0.667 - 0.30 * 0.20 = 0.667 - 0.060 = 0.607
        pre = {"cpu_utilization": 90.0, "latency": 45.0}
        post = {"cpu_utilization": 30.0, "latency": 20.0}

        outcome = self.evaluator.evaluate(
            action=RecoveryAction.WORKLOAD_REDISTRIBUTION.value,
            root_cause_category="CPU_OVERLOAD",
            pre_metrics=pre,
            post_metrics=post,
        )
        self.assertEqual(outcome.target_metric, "cpu_utilization")
        self.assertAlmostEqual(outcome.absolute_improvement, 60.0, places=1)
        self.assertGreater(outcome.benefit, 0.60)
        self.assertEqual(outcome.cost, 0.20)
        self.assertGreater(outcome.reward, 0.50)
        self.assertTrue(outcome.success)

    def test_worsening_metric_penalty(self):
        # Pre-latency: 40 ms, Post-latency: 120 ms -> Worsening gives negative benefit
        pre = {"latency": 40.0, "packet_loss": 1.0}
        post = {"latency": 120.0, "packet_loss": 8.0}

        outcome = self.evaluator.evaluate(
            action=RecoveryAction.TRAFFIC_REROUTING.value,
            root_cause_category="NETWORK_CONGESTION",
            pre_metrics=pre,
            post_metrics=post,
        )
        self.assertLess(outcome.benefit, 0.0)
        self.assertLess(outcome.reward, 0.0)
        self.assertFalse(outcome.success)

    def test_failed_node_full_recovery(self):
        # Recovering a failed node grants full benefit (1.0)
        pre = {"latency": 999.0}
        post = {"latency": 20.0}

        outcome = self.evaluator.evaluate(
            action=RecoveryAction.EDGE_NODE_FAILOVER.value,
            root_cause_category="NODE_FAILURE",
            pre_metrics=pre,
            post_metrics=post,
            is_node_failed_before=True,
            is_node_failed_after=False,
        )
        self.assertEqual(outcome.benefit, 1.0)
        self.assertTrue(outcome.success)
        # 1.0 * 1.0 - 0.30 * 0.80 = 0.76
        self.assertAlmostEqual(outcome.reward, 0.76, places=2)

    def test_explicit_execution_failure_status(self):
        # Even if post-metrics appeared improved or unchanged, unexecutable/rejected action must fail
        pre = {"cpu_utilization": 90.0}
        post = {"cpu_utilization": 20.0}
        outcome = self.evaluator.evaluate(
            action=RecoveryAction.WORKLOAD_REDISTRIBUTION.value,
            root_cause_category="CPU_OVERLOAD",
            pre_metrics=pre,
            post_metrics=post,
            execution_applied=False,
        )
        self.assertFalse(outcome.execution_applied)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.benefit, -0.40)
        self.assertLess(outcome.reward, 0.0)
        self.assertEqual(outcome.details["failure_reason"], "ACTION_EXECUTION_REJECTED_OR_FAILED")

    def test_unexecutable_action_cannot_receive_positive_reward_or_success(self):
        # Action with unchanged metrics and execution_applied=False cannot receive positive reward
        pre = {"cpu_utilization": 80.0, "latency": 50.0}
        post = {"cpu_utilization": 80.0, "latency": 50.0}
        outcome = self.evaluator.evaluate(
            action="UNSUPPORTED_ACTION",
            root_cause_category="GENERIC_RESOURCE_FAULT",
            pre_metrics=pre,
            post_metrics=post,
            execution_applied=False,
            action_cost_override=0.20,
        )
        self.assertFalse(outcome.execution_applied)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.benefit, -0.40)
        # raw_reward = 1.0 * (-0.40) - 0.30 * 0.20 = -0.46
        self.assertAlmostEqual(outcome.reward, -0.46, places=4)


class TestCRAFTEdgeShadowState(unittest.TestCase):
    # Test suite for ShadowNodeState model.

    def setUp(self):
        self.shadow = ShadowNodeState(node_id="test_node_1")

    def test_telemetry_update_and_mitigation_application(self):
        # Feed high CPU telemetry
        self.shadow.update_from_telemetry({"cpu_utilization": 95.0, "latency": 80.0})
        self.assertGreater(self.shadow.cpu_utilization, 90.0)

        # Apply Workload Redistribution
        after_metrics = self.shadow.apply_recovery_action(RecoveryAction.WORKLOAD_REDISTRIBUTION.value)
        self.assertLess(after_metrics["cpu_utilization"], 50.0)
        self.assertEqual(self.shadow.cpu_mitigation, 0.3)
        self.assertTrue(self.shadow.last_action_applied)

    def test_shadow_proxy_metric_provenance_and_labeling(self):
        # Case 1: Ingesting direct physical metrics
        direct_feats = {
            "cpu_utilization": 75.0,
            "memory_utilization": 60.0,
            "network_utilization": 40.0,
            "latency": 35.0,
            "packet_loss": 1.2,
        }
        self.shadow.update_from_telemetry(direct_feats)
        meta_direct = self.shadow.get_state_metadata()
        self.assertFalse(self.shadow.is_synthetic_proxy)
        self.assertEqual(self.shadow.metric_sources["cpu_utilization"], "DIRECT_OBSERVATION")
        self.assertEqual(self.shadow.metric_sources["memory_utilization"], "DIRECT_OBSERVATION")
        self.assertIn("Direct observable", meta_direct["measurement_notice"])

        # Case 2: Ingesting packet-only telemetry (e.g. Edge-IIoTset pcap fields)
        pcap_feats = {
            "tcp_active_flags_count": 4.0,
            "udp.time_delta": 0.8,
            "tcp.len": 1200.0,
        }
        self.shadow.update_from_telemetry(pcap_feats)
        meta_pcap = self.shadow.get_state_metadata()
        self.assertTrue(self.shadow.is_synthetic_proxy)
        self.assertEqual(self.shadow.metric_sources["cpu_utilization"], "SYNTHETIC_PROXY_ESTIMATE")
        self.assertEqual(self.shadow.metric_sources["memory_utilization"], "SYNTHETIC_PROXY_ESTIMATE")
        self.assertIn("synthetic proxy estimates", meta_pcap["measurement_notice"])

    def test_shadow_unsupported_action_applied_flag(self):
        self.shadow.apply_recovery_action("INVALID_RECOVERY")
        self.assertFalse(self.shadow.last_action_applied)
        self.shadow.apply_recovery_action(RecoveryAction.WORKLOAD_REDISTRIBUTION.value)
        self.assertTrue(self.shadow.last_action_applied)


class TestCRAFTEdgePipelineIntegration(unittest.TestCase):
    # Integration tests for CRAFTEdgePipeline.

    def setUp(self):
        np.random.seed(42)
        self.feature_names = [
            "tcp.flags", "tcp_active_flags_count", "tcp_seq_ack_ratio",
            "udp.port", "udp.stream", "udp.time_delta",
            "http.content_length", "http.response"
        ]
        self.pipeline = CRAFTEdgePipeline(config={
            "detection_threshold": 0.60,
            "learning_rate": 0.25,
            "epsilon": 0.0,  # Zero exploration for deterministic testing
        })

    def _create_input(self, position: int, feat_dict: dict, node_id: str = "node_1") -> BenchmarkInput:
        vec = np.array([feat_dict.get(k, 0.0) for k in self.feature_names], dtype=np.float64)
        return BenchmarkInput(
            stream_position=position,
            timestamp=100.0 + position,
            timestamp_str="2026-10-11 00:00:00",
            device_id="sensor_1",
            edge_node_id=node_id,
            features=feat_dict,
            feature_vector=vec,
        )

    def test_contract_and_capabilities(self):
        self.assertIsInstance(self.pipeline, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.pipeline.paper_id, "craft_edge")
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.STREAMING_DETECTION))
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))

    def test_fit_and_normal_record_processing(self):
        # 1. Fit pipeline on synthetic normal training split
        N = 150
        D = len(self.feature_names)
        X_train = np.random.normal(0.0, 1.0, size=(N, D))
        self.pipeline.fit(X_train)
        self.assertTrue(self.pipeline.detector.is_fitted)

        # 2. Process nominal record
        normal_feats = {k: 0.02 for k in self.feature_names}
        inp = self._create_input(0, normal_feats)
        res = self.pipeline.process(inp)

        self.assertIsInstance(res, DetectionResult)
        self.assertFalse(res.is_anomaly)
        self.assertIsNone(res.raw_output["diagnosis"])
        self.assertIsNone(res.raw_output["mitigation"])

    def test_closed_loop_feedback_anomaly_to_policy_update(self):
        # Fit on baseline
        N = 100
        D = len(self.feature_names)
        X_train = np.random.normal(0.0, 0.5, size=(N, D))
        self.pipeline.fit(X_train)

        # Process extreme anomalous record that triggers detection
        anom_feats = {k: 12.0 for k in self.feature_names}
        inp = self._create_input(1, anom_feats, node_id="node_eval")

        res = self.pipeline.process(inp)
        self.assertTrue(res.is_anomaly)

        # Verify Causal RCA was performed
        diag = res.raw_output.get("diagnosis")
        self.assertIsInstance(diag, RCADiagnosisResult)
        self.assertGreater(len(diag.ranked_root_causes), 0)

        # Verify Recovery was selected and executed
        mit = res.raw_output.get("mitigation")
        self.assertIsInstance(mit, MitigationResult)
        self.assertIn("root_cause_category", mit.metadata)
        self.assertIn("q_value_updated", mit.metadata)
        self.assertIn("reward", mit.metadata)

        # Verify Outcome was evaluated
        outcome = res.raw_output.get("outcome")
        self.assertIsInstance(outcome, RecoveryOutcome)
        self.assertIsNotNone(outcome.reward)

        # Verify policy statistics reflect the online update
        stats = self.pipeline.get_policy_stats()
        self.assertEqual(stats["total_updates"], 1)

    def test_reset_clears_state(self):
        self.pipeline.process(self._create_input(0, {k: 1.0 for k in self.feature_names}))
        self.pipeline.reset()
        self.assertEqual(self.pipeline.processed_records_count, 0)
        self.assertEqual(self.pipeline.anomalies_detected_count, 0)
        self.assertEqual(len(self.pipeline.shadow_nodes), 0)
        self.assertIsNone(self.pipeline.last_diagnosis)
        self.assertIsNone(self.pipeline.last_mitigation)

    def test_action_improves_system_health(self):
        # 1. Action that improves system health yields positive benefit, positive reward, and increases Q-value
        evaluator = RecoveryOutcomeEvaluator()
        pre = {"cpu_utilization": 85.0, "latency": 60.0}
        post = {"cpu_utilization": 25.0, "latency": 20.0}
        outcome = evaluator.evaluate(
            action=RecoveryAction.WORKLOAD_REDISTRIBUTION.value,
            root_cause_category="CPU_OVERLOAD",
            pre_metrics=pre,
            post_metrics=post,
        )
        self.assertGreater(outcome.benefit, 0.5)
        self.assertGreater(outcome.reward, 0.4)
        self.assertTrue(outcome.success)

    def test_action_worsens_system_health(self):
        # 2. Action that worsens system health yields negative benefit, negative reward, and fails
        evaluator = RecoveryOutcomeEvaluator()
        pre = {"latency": 35.0, "packet_loss": 0.5}
        post = {"latency": 110.0, "packet_loss": 9.0}
        outcome = evaluator.evaluate(
            action=RecoveryAction.TRAFFIC_REROUTING.value,
            root_cause_category="NETWORK_CONGESTION",
            pre_metrics=pre,
            post_metrics=post,
        )
        self.assertLess(outcome.benefit, 0.0)
        self.assertLess(outcome.reward, 0.0)
        self.assertFalse(outcome.success)

    def test_action_no_measurable_effect(self):
        # 3. Action with no measurable effect yields benefit = 0.0 and net negative reward due to operational cost
        evaluator = RecoveryOutcomeEvaluator()
        pre = {"latency": 50.0, "packet_loss": 2.0}
        post = {"latency": 50.0, "packet_loss": 2.0}
        outcome = evaluator.evaluate(
            action=RecoveryAction.TRAFFIC_REROUTING.value,
            root_cause_category="NETWORK_CONGESTION",
            pre_metrics=pre,
            post_metrics=post,
        )
        self.assertEqual(outcome.absolute_improvement, 0.0)
        self.assertEqual(outcome.benefit, 0.0)
        # Cost is 0.25, reward = 0 - 0.3 * 0.25 = -0.075
        self.assertAlmostEqual(outcome.reward, -0.075, places=3)
        self.assertFalse(outcome.success)

    def test_unexecutable_action(self):
        # 4. Action that cannot be executed (invalid/unsupported action)
        shadow = ShadowNodeState(node_id="node_unsupported")
        shadow.update_from_telemetry({"cpu_utilization": 80.0, "latency": 55.0})
        pre_metrics = shadow.get_metrics()

        # Apply unknown action
        post_metrics = shadow.apply_recovery_action("UNSUPPORTED_ACTION")
        self.assertEqual(pre_metrics["cpu_utilization"], post_metrics["cpu_utilization"])
        self.assertEqual(pre_metrics["latency"], post_metrics["latency"])
        self.assertFalse(shadow.last_action_applied)

        # Outcome evaluator on unexecutable action receives failure penalty
        evaluator = RecoveryOutcomeEvaluator()
        outcome = evaluator.evaluate(
            action="UNSUPPORTED_ACTION",
            root_cause_category="GENERIC_RESOURCE_FAULT",
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            action_cost_override=0.40,
            execution_applied=shadow.last_action_applied,
        )
        self.assertFalse(outcome.execution_applied)
        self.assertEqual(outcome.benefit, -0.40)
        self.assertLess(outcome.reward, 0.0)
        self.assertFalse(outcome.success)

    def test_rejected_simulation_action_not_successful(self):
        # Simulation where apply_recovery rejects action
        class MockFailingSimulation:
            def __init__(self):
                self.nodes = {"node_fail": self}
                self.node_failed = False
            def get_state_dict(self):
                return {"cpu_utilization": 95.0, "latency": 80.0, "node_failed": False}
            def apply_recovery(self, node_id, action):
                return False  # Simulator rejects action

        mock_sim = MockFailingSimulation()
        pipe = CRAFTEdgePipeline(
            config={"detection_threshold": 0.50, "epsilon": 0.0},
            simulation=mock_sim,
        )
        pipe.fit(np.random.normal(0, 0.5, size=(50, len(self.feature_names))))

        anom_input = self._create_input(1, {k: 10.0 for k in self.feature_names}, node_id="node_fail")
        res = pipe.process(anom_input)

        self.assertTrue(res.is_anomaly)
        outcome = res.raw_output["outcome"]
        mitigation = res.raw_output["mitigation"]

        # Assert explicit rejection and unsuccessful classification
        self.assertFalse(outcome.execution_applied)
        self.assertFalse(outcome.success)
        self.assertLess(outcome.reward, 0.0)
        self.assertFalse(mitigation.success)
        self.assertFalse(mitigation.metadata["execution_applied"])
        self.assertEqual(mitigation.tasks_affected, 0)
        self.assertEqual(mitigation.state_bytes_transferred, 0)

    def test_episode_decay_across_stream_transition(self):
        # Episode decay triggers upon stream returning to normal, not on every burst record
        pipe = CRAFTEdgePipeline(config={
            "detection_threshold": 0.65,
            "epsilon": 0.20,
            "epsilon_decay": 0.85,
            "decay_mode": "episode",
            "decay_interval": 1,
        })
        pipe.fit(np.random.normal(0, 0.5, size=(50, len(self.feature_names))))

        # Burst of 3 anomalies in the same episode
        for pos in range(1, 4):
            anom = self._create_input(pos, {k: 3.5 for k in self.feature_names})
            res = pipe.process(anom)
            self.assertTrue(res.is_anomaly)
            # During the ongoing anomaly episode, epsilon is NOT collapsed
            self.assertEqual(pipe.policy.epsilon, 0.20)

        # Stream returns to normal -> triggers episode completion and decay
        norm = self._create_input(4, {k: 0.05 for k in self.feature_names})
        res_norm = pipe.process(norm)
        self.assertFalse(res_norm.is_anomaly)
        self.assertAlmostEqual(pipe.policy.epsilon, 0.17, places=4)
        self.assertEqual(pipe.policy.episode_count, 1)

    def test_causal_root_to_action_mapping_isolation(self):
        # 5. Causal root mapping classifies correctly and isolates Q-table updates between cause categories
        self.assertEqual(classify_root_cause("tcp_active_flags_count"), "TRANSPORT_FLOOD")
        self.assertEqual(classify_root_cause("udp.time_delta"), "TRANSPORT_FLOOD")
        self.assertEqual(classify_root_cause("http.content_length"), "APPLICATION_EXPLOIT")
        self.assertEqual(classify_root_cause("cpu_utilization"), "CPU_OVERLOAD")
        self.assertEqual(classify_root_cause("memory_utilization"), "MEMORY_OVERLOAD")

        policy = AdaptiveEfficacyPolicy()
        # Update CPU_OVERLOAD
        policy.update("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value, reward=0.90)
        # Verify TRANSPORT_FLOOD remains unaffected
        q_flood = policy.get_q_value("TRANSPORT_FLOOD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value)
        self.assertEqual(q_flood, policy.default_q)
        self.assertNotEqual(q_flood, policy.get_q_value("CPU_OVERLOAD", RecoveryAction.WORKLOAD_REDISTRIBUTION.value))

    def test_deterministic_policy_updates_fixed_sequence(self):
        # 6. Verify deterministic mathematical convergence under fixed reward sequence
        policy = AdaptiveEfficacyPolicy(learning_rate=0.20, default_q=0.50)
        cause = "CPU_OVERLOAD"
        action = RecoveryAction.WORKLOAD_REDISTRIBUTION.value
        policy.set_q_value(cause, action, 0.50)

        # Sequence of rewards
        rewards = [0.80, 0.60, -0.20, 1.00]
        # Analytical tracking:
        # Step 1: 0.50 + 0.2 * (0.80 - 0.50) = 0.5600
        # Step 2: 0.56 + 0.2 * (0.60 - 0.56) = 0.5680
        # Step 3: 0.568 + 0.2 * (-0.20 - 0.568) = 0.4144
        # Step 4: 0.4144 + 0.2 * (1.00 - 0.4144) = 0.53152
        expected_steps = [0.5600, 0.5680, 0.4144, 0.53152]

        for r, expected in zip(rewards, expected_steps):
            q_actual = policy.update(cause, action, reward=r)
            self.assertAlmostEqual(q_actual, expected, places=4)

    def test_separation_recorded_dataset_and_simulated_values(self):
        # 7. Recorded dataset values must remain immutable and unmutated by simulated recovery
        raw_feats = {
            "tcp.flags": 0.85,
            "tcp_active_flags_count": 5.0,
            "tcp_seq_ack_ratio": 1.2,
            "udp.port": 53.0,
            "udp.stream": 1.0,
            "udp.time_delta": 0.05,
            "http.content_length": 1024.0,
            "http.response": 200.0,
        }
        raw_feats_copy = dict(raw_feats)

        inp = self._create_input(0, raw_feats)
        self.pipeline.fit(np.ones((50, len(self.feature_names))))
        _ = self.pipeline.process(inp)

        # Ingested input features dictionary must be 100% unchanged
        self.assertEqual(inp.features, raw_feats_copy)
        self.assertEqual(raw_feats, raw_feats_copy)

    def test_label_isolation_enforcement(self):
        # 8. Ground truth labels must be absent from BenchmarkInput
        raw_feats = {k: 0.1 for k in self.feature_names}
        inp = self._create_input(0, raw_feats)

        self.assertNotIn("Attack_label", inp.features)
        self.assertNotIn("Attack_type", inp.features)
        self.assertFalse(hasattr(inp, "Attack_label"))
        self.assertFalse(hasattr(inp, "Attack_type"))


if __name__ == "__main__":
    unittest.main()

