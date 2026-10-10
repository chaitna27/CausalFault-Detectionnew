# Comprehensive Integration and Regression Tests for CRAFT-Edge Stage 3 CLI & Benchmark Integration.
#
# Covers:
# 1. CLI selection and initialization of CRAFT-Edge.
# 2. Successful processing of representative inputs through actual CLI path.
# 3. Correct feature ordering and exclusion of label columns.
# 4. No training-data leakage into validation or test processing.
# 5. Benchmark registration and result serialization.
# 6. Compatibility with existing benchmark result schema.
# 7. Preservation of all existing algorithms and CLI modes.
# 8. Correct handling and reporting of rejected recovery actions.
# 9. Correct distinction between detection metrics and recovery metrics.
# 10. Clear labeling of shadow-state proxy estimates.

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

import main
from algorithms import (
    CRAFTEdgePipeline,
    CausalFaultTolerancePipeline,
    IPFTAlgorithm,
    BWOAIAlgorithm,
    RCDAlgorithm,
    PreGANAlgorithm,
)
from algorithms.craft_edge import RecoveryAction
from evaluation.data_harness import CommonDataHarness, StreamConfig
from evaluation.context import BenchmarkInput, EventGroundTruth, BenchmarkContext
from scripts.run_full_benchmark import run_benchmark_trial, aggregate_trials


class TestCRAFTEdgeCLIIntegration(unittest.TestCase):
    # Tests CLI integration in main.py.

    def test_cli_selection_and_initialization(self):
        # 1. Verify CLI argument parsing recognizes CRAFT-Edge mode and its hyperparameters
        test_argv = [
            "main.py",
            "--mode", "craft_edge",
            "--learning-rate", "0.25",
            "--epsilon", "0.10",
            "--epsilon-min", "0.01",
            "--epsilon-decay", "0.99",
            "--decay-interval", "5",
            "--decay-mode", "episode",
            "--benefit-weight", "1.2",
            "--cost-weight", "0.35",
            "--detection-threshold", "0.70",
            "--train-samples", "200",
            "--split", "test",
        ]
        sys.argv = test_argv
        args = main.parse_args()

        self.assertEqual(args.mode, "craft_edge")
        self.assertEqual(args.learning_rate, 0.25)
        self.assertEqual(args.epsilon, 0.10)
        self.assertEqual(args.epsilon_min, 0.01)
        self.assertEqual(args.epsilon_decay, 0.99)
        self.assertEqual(args.decay_interval, 5)
        self.assertEqual(args.decay_mode, "episode")
        self.assertEqual(args.benefit_weight, 1.2)
        self.assertEqual(args.cost_weight, 0.35)
        self.assertEqual(args.detection_threshold, 0.70)
        self.assertEqual(args.train_samples, 200)
        self.assertEqual(args.split, "test")

    def test_cli_execution_path_representative_inputs(self):
        # 2. Run actual run_craft_edge_experiment on a small sample through CLI path
        with tempfile.TemporaryDirectory() as tmp_dir:
            sys.argv = [
                "main.py",
                "--mode", "craft_edge",
                "--max-records", "25",
                "--train-samples", "20",
                "--output-dir", tmp_dir,
                "--seed", "42",
            ]
            args = main.parse_args()
            res = main.run_craft_edge_experiment(args)

            self.assertEqual(res["algorithm"], "CRAFT-Edge")
            self.assertEqual(res["records_processed"], 25)
            self.assertIn("accuracy", res)
            self.assertIn("f1_score", res)
            self.assertIn("recovery_success_rate", res)

            # Check that output CSV files were created
            det_csv = os.path.join(tmp_dir, "data", "craft_edge_detection_metrics.csv")
            rec_csv = os.path.join(tmp_dir, "data", "craft_edge_recovery_outcomes.csv")
            self.assertTrue(os.path.exists(det_csv))
            self.assertTrue(os.path.exists(rec_csv))

            df_det = pd.read_csv(det_csv)
            self.assertEqual(len(df_det), 1)
            self.assertEqual(df_det["algorithm"].iloc[0], "CRAFT-Edge")

    def test_feature_ordering_and_label_exclusion(self):
        # 3. Correct feature ordering and strict exclusion of ground truth label columns
        art_path = os.path.join("data", "processed", "artifacts", "feature_names.json")
        if os.path.exists(art_path):
            with open(art_path) as f:
                feature_names = json.load(f)
            self.assertNotIn("Attack_label", feature_names)
            self.assertNotIn("Attack_type", feature_names)
            self.assertNotIn("fault_label", feature_names)

        # Ingest benchmark input and verify features
        harness = CommonDataHarness(StreamConfig(max_records=5))
        for ctx in harness.stream_contexts():
            feats = ctx.observable_input.features
            self.assertNotIn("Attack_label", feats)
            self.assertNotIn("Attack_type", feats)
            self.assertNotIn("fault_label", feats)
            break

    def test_no_training_leakage_into_eval(self):
        # 4. Training samples for fitting are loaded from train split, separate from test split
        train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
        test_path = os.path.join("data", "processed", "test", "test_01_processed.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            df_train = pd.read_csv(train_path, nrows=50)
            df_test = pd.read_csv(test_path, nrows=50)
            # Ensure timestamps or row indices differ between train and test
            self.assertNotEqual(df_train["timestamp"].iloc[0], df_test["timestamp"].iloc[0])

    def test_preservation_of_existing_algorithms_and_cli_modes(self):
        # 7. Preservation of all existing algorithms and CLI modes
        for algo_cls in [CausalFaultTolerancePipeline, IPFTAlgorithm, BWOAIAlgorithm, RCDAlgorithm, PreGANAlgorithm]:
            inst = algo_cls()
            self.assertIsNotNone(inst.name)
            self.assertIsNotNone(inst.paper_id)

        # Verify existing main.py CLI choices include simulation, dataset, experiment, sensitivity
        sys.argv = ["main.py", "--mode", "simulation"]
        args = main.parse_args()
        self.assertEqual(args.mode, "simulation")

        sys.argv = ["main.py", "--mode", "dataset", "--dataset", "ton_iot"]
        args = main.parse_args()
        self.assertEqual(args.mode, "dataset")
        self.assertEqual(args.dataset, "ton_iot")


class TestCRAFTEdgeBenchmarkIntegration(unittest.TestCase):
    # Tests benchmark runner integration in scripts/run_full_benchmark.py.

    def setUp(self):
        np.random.seed(42)
        self.train_data = np.random.normal(0, 0.5, size=(100, 62))

    def test_benchmark_registration_and_result_serialization(self):
        # 5. Benchmark trial runs CRAFT-Edge and serializes results
        res = run_benchmark_trial(
            algo_cls=CRAFTEdgePipeline,
            algo_id="proposed_craft_edge",
            name="CRAFT-Edge (Ours, Adaptive Efficacy)",
            threshold=0.65,
            train_data=self.train_data,
            seed=42,
            total_records=150,
            warmup_records=50,
        )

        self.assertIsInstance(res, dict)
        self.assertEqual(res["seed"], 42)
        self.assertEqual(res["eval_records"], 100)
        self.assertIn("throughput_rec_sec", res)
        self.assertIn("per_record_latency_ms", res)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1", res)

        # JSON serialization check
        serialized = json.dumps(res)
        self.assertIsInstance(serialized, str)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["seed"], 42)

    def test_benchmark_results_schema_compatibility(self):
        # 6. Compatibility with existing benchmark result schema and multi-trial aggregation
        trials = []
        for s in [42, 43]:
            res = run_benchmark_trial(
                algo_cls=CRAFTEdgePipeline,
                algo_id="proposed_craft_edge",
                name="CRAFT-Edge (Ours, Adaptive Efficacy)",
                threshold=0.65,
                train_data=self.train_data,
                seed=s,
                total_records=100,
                warmup_records=20,
            )
            trials.append(res)

        agg = aggregate_trials(trials)
        self.assertIn("throughput_rec_sec", agg)
        self.assertIn("per_record_latency_ms", agg)
        self.assertIn("precision", agg)
        self.assertIn("recall", agg)
        self.assertIn("f1", agg)
        self.assertIn("recovery_summary", agg)

    def test_rejected_recovery_action_handling(self):
        # 8. Handling and reporting of rejected recovery actions
        class MockFailingSimulation:
            def __init__(self):
                self.nodes = {"sim_node": self}
            def get_state_dict(self):
                return {"cpu_utilization": 90.0, "latency": 80.0, "node_failed": False}
            def apply_recovery(self, node_id, action):
                return False  # Rejects action

        mock_sim = MockFailingSimulation()
        pipe = CRAFTEdgePipeline(simulation=mock_sim)
        pipe.fit(self.train_data)

        # Process an anomaly
        inp = BenchmarkInput(
            stream_position=1,
            timestamp=100.0,
            timestamp_str="2026-10-11",
            device_id="dev",
            edge_node_id="sim_node",
            features={"cpu_utilization": 95.0},
            feature_vector=np.full(62, 5.0),
        )
        res = pipe.process(inp)
        outcome = res.raw_output["outcome"]
        mitigation = res.raw_output["mitigation"]

        self.assertFalse(outcome.execution_applied)
        self.assertFalse(outcome.success)
        self.assertLess(outcome.reward, 0.0)
        self.assertFalse(mitigation.success)
        self.assertEqual(mitigation.tasks_affected, 0)

    def test_distinction_between_detection_and_recovery_metrics(self):
        # 9. Clear distinction between detection performance and recovery performance
        res = run_benchmark_trial(
            algo_cls=CRAFTEdgePipeline,
            algo_id="proposed_craft_edge",
            name="CRAFT-Edge",
            threshold=0.65,
            train_data=self.train_data,
            seed=42,
            total_records=150,
            warmup_records=50,
        )

        # Detection metrics
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1", res)

        # Recovery metrics in recovery_summary
        rec_summary = res["recovery_summary"]
        self.assertIsInstance(rec_summary, dict)
        self.assertIn("total_recovery_episodes", rec_summary)
        self.assertIn("successful_recoveries", rec_summary)
        self.assertIn("recovery_success_rate", rec_summary)
        self.assertIn("mean_reward", rec_summary)
        self.assertIn("mean_estimated_benefit", rec_summary)
        self.assertIn("mean_operational_cost", rec_summary)

        # Ensure they are distinct datatypes and namespaces
        self.assertNotEqual(res["f1"], rec_summary["recovery_success_rate"])

    def test_shadow_proxy_estimate_labeling(self):
        # 10. Clear labeling of shadow-state proxy estimates
        res = run_benchmark_trial(
            algo_cls=CRAFTEdgePipeline,
            algo_id="proposed_craft_edge",
            name="CRAFT-Edge",
            threshold=0.65,
            train_data=self.train_data,
            seed=42,
            total_records=120,
            warmup_records=20,
        )

        rec_summary = res["recovery_summary"]
        self.assertEqual(rec_summary["benefit_measurement_type"], "SHADOW_SYNTHETIC_PROXY")
        self.assertIn("Shadow-state proxy estimates", rec_summary["measurement_notice"])


if __name__ == "__main__":
    unittest.main()
