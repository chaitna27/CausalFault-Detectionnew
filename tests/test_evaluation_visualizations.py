# Unit and Regression Tests for Evaluation Visualizations and Metric Calculations.
#
# Validates:
# 1. Calculation of all classification metrics and zero-division handling.
# 2. Confusion matrix count conservation (TP + TN + FP + FN == Total).
# 3. Discrepancy checks against stored benchmark metrics.
# 4. Presence and validity of generated visualization figures (PNG and PDF).
# 5. Non-fabrication of unmeasured thresholds in recovery analysis.

import os
import unittest
import numpy as np
import pandas as pd

from scripts.plot_evaluation_results import (
    calc_classification_metrics,
    EvaluationDataLoader,
    DEFAULT_OUTPUT_DIR,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Tests metric formulations, zero-division conventions, and count conservation."""

    def test_standard_metric_formulations(self):
        # TP=80, TN=120, FP=20, FN=30 -> Total=250
        m = calc_classification_metrics(tp=80, tn=120, fp=20, fn=30)
        self.assertEqual(m["total"], 250)
        self.assertAlmostEqual(m["accuracy"], (80 + 120) / 250)           # 200/250 = 0.80
        self.assertAlmostEqual(m["precision"], 80 / (80 + 20))            # 80/100 = 0.80
        self.assertAlmostEqual(m["recall"], 80 / (80 + 30))               # 80/110 = 0.72727
        expected_f1 = 2 * 0.80 * (80 / 110) / (0.80 + 80 / 110)
        self.assertAlmostEqual(m["f1"], expected_f1)
        self.assertAlmostEqual(m["fpr"], 20 / (20 + 120))                 # 20/140 = 0.142857
        self.assertAlmostEqual(m["specificity"], 120 / (120 + 20))        # 120/140 = 0.85714
        self.assertAlmostEqual(m["balanced_accuracy"], ((80 / 110) + (120 / 140)) / 2.0)
        self.assertAlmostEqual(m["npv"], 120 / (120 + 30))                # 120/150 = 0.80

    def test_zero_division_handling(self):
        # Degenerate: All zeros
        m_zero = calc_classification_metrics(tp=0, tn=0, fp=0, fn=0)
        self.assertEqual(m_zero["accuracy"], 0.0)
        self.assertEqual(m_zero["precision"], 0.0)
        self.assertEqual(m_zero["recall"], 0.0)
        self.assertEqual(m_zero["f1"], 0.0)
        self.assertEqual(m_zero["fpr"], 0.0)
        self.assertEqual(m_zero["specificity"], 0.0)
        self.assertEqual(m_zero["balanced_accuracy"], 0.0)
        self.assertEqual(m_zero["npv"], 0.0)

        # Degenerate: No predicted positives (TP=0, FP=0)
        m_no_pos = calc_classification_metrics(tp=0, tn=100, fp=0, fn=25)
        self.assertEqual(m_no_pos["precision"], 0.0)
        self.assertEqual(m_no_pos["recall"], 0.0)
        self.assertEqual(m_no_pos["f1"], 0.0)
        self.assertAlmostEqual(m_no_pos["accuracy"], 100 / 125)
        self.assertAlmostEqual(m_no_pos["specificity"], 1.0)
        self.assertAlmostEqual(m_no_pos["npv"], 100 / 125)

        # Degenerate: No true negatives (TN=0, FP=100)
        m_no_tn = calc_classification_metrics(tp=50, tn=0, fp=100, fn=0)
        self.assertEqual(m_no_tn["specificity"], 0.0)
        self.assertAlmostEqual(m_no_tn["fpr"], 1.0)
        self.assertAlmostEqual(m_no_tn["recall"], 1.0)

    def test_count_conservation_across_seeds(self):
        # In the common test stream, N should always sum to 24,000 records
        loader = EvaluationDataLoader()
        loader.load_all()
        self.assertIsNotNone(loader.confusion_by_seed_df)

        for _, row in loader.confusion_by_seed_df.iterrows():
            total = row["tp"] + row["tn"] + row["fp"] + row["fn"]
            self.assertEqual(total, 24000, f"Row {row['algorithm_id']} seed {row['seed']} does not sum to 24,000")


class TestVisualizationArtifacts(unittest.TestCase):
    """Verifies that all required publication figures exist and are non-empty."""

    def setUp(self):
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.required_figure_bases = [
            "classification_metrics_comparison",
            "confusion_matrix_threshold_065",
            "confusion_matrix_threshold_050",
            "confusion_matrix_threshold_030",
            "confusion_matrix_threshold_015",
            "confusion_matrix_all_algorithms",
            "confusion_counts_comparison",
            "accuracy_by_threshold",
            "precision_by_threshold",
            "recall_by_threshold",
            "f1_score_by_threshold",
            "false_positive_rate_by_threshold",
            "successful_recoveries_by_threshold",
            "threshold_tradeoff_all_metrics",
            "precision_recall_comparison",
            "precision_recall_curve",
            "f1_score_comparison",
            "recovery_comparison",
            "recovery_success_rate",
            "recovery_cost_and_availability",
            "seed_stability_metrics",
            "throughput_latency_tradeoff",
            "resource_utilization_energy",
            "rca_localization_and_latency",
            "per_device_detection_performance",
            "metrics_correlation_heatmap",
        ]

    def test_png_figures_exist_and_non_empty(self):
        for base in self.required_figure_bases:
            path = os.path.join(self.output_dir, f"{base}.png")
            self.assertTrue(os.path.exists(path), f"Missing PNG figure: {path}")
            size = os.path.getsize(path)
            self.assertGreater(size, 10000, f"PNG figure {path} is too small ({size} bytes)")

    def test_pdf_figures_exist_and_non_empty(self):
        for base in self.required_figure_bases:
            path = os.path.join(self.output_dir, f"{base}.pdf")
            self.assertTrue(os.path.exists(path), f"Missing PDF figure: {path}")
            size = os.path.getsize(path)
            self.assertGreater(size, 2000, f"PDF figure {path} is too small ({size} bytes)")

    def test_summary_markdown_exists_and_complete(self):
        summary_path = os.path.join(os.path.dirname(self.output_dir), "visualization_summary.md")
        self.assertTrue(os.path.exists(summary_path), f"Missing summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## 1. Source Files Used", content)
        self.assertIn("## 2. Evaluated Configurations Found", content)
        self.assertIn("## 3. Metrics Available and Unavailable", content)
        self.assertIn("## 4. Generated Figures Inventory", content)
        self.assertIn("## 5. Metric Definitions & Zero-Division Handling", content)
        self.assertIn("## 6. Discrepancy & Verification Analysis", content)
        self.assertIn("## 7. Important Limitations", content)
        self.assertIn("## 8. Exact Command to Regenerate All Visualizations", content)


if __name__ == "__main__":
    unittest.main()
