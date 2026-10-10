import unittest
import warnings
from algorithms.causal_ft import CausalFaultTolerancePipeline


class TestCausalFTThresholdRegression(unittest.TestCase):
    """Regression tests verifying robust threshold configuration handling."""

    def test_canonical_detection_threshold_key(self):
        pipeline = CausalFaultTolerancePipeline()
        pipeline.initialize({"detection_threshold": 0.15})
        self.assertAlmostEqual(pipeline.detector.config.threshold, 0.15)

    def test_backward_compatible_threshold_alias(self):
        pipeline = CausalFaultTolerancePipeline()
        pipeline.initialize({"threshold": 0.15})
        self.assertAlmostEqual(pipeline.detector.config.threshold, 0.15)

    def test_upper_threshold_alias(self):
        pipeline = CausalFaultTolerancePipeline()
        pipeline.initialize({"upper_threshold": 0.15})
        self.assertAlmostEqual(pipeline.detector.config.threshold, 0.15)

    def test_canonical_precedence_over_alias(self):
        pipeline = CausalFaultTolerancePipeline()
        pipeline.initialize({"detection_threshold": 0.15, "threshold": 0.45})
        self.assertAlmostEqual(pipeline.detector.config.threshold, 0.15)

    def test_unknown_key_emits_warning(self):
        pipeline = CausalFaultTolerancePipeline()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline.initialize({"bogus_key": 123, "detection_threshold": 0.15})
            self.assertTrue(any("Unknown configuration key 'bogus_key'" in str(warn.message) for warn in w))
        self.assertAlmostEqual(pipeline.detector.config.threshold, 0.15)


if __name__ == "__main__":
    unittest.main()
