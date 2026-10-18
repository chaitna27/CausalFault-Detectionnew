# Unit tests for Craft-specific causal graph reachability caching,
# caller immutability protection, mutation invalidation, and full-stream equivalence.

import unittest
import numpy as np
import networkx as nx
from typing import List, Dict

from algorithms.craft_edge.pipeline import (
    CRAFTEdgePipeline,
    CraftCausalGraphSpecification,
)
from algorithms.causal_ft.causal_engine import (
    CausalInferenceEngine,
    CausalGraphSpecification,
)
from evaluation.context import BenchmarkInput


class TestCraftCausalGraphCaching(unittest.TestCase):
    # Tests caller isolation, immutability, and cache invalidation under graph mutations.

    def setUp(self):
        self.graph = CraftCausalGraphSpecification()
        self.feature_names = [
            "tcp.flags", "tcp_active_flags_count", "tcp.connection.syn",
            "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.synack",
            "tcp_seq_ack_ratio", "tcp.seq", "tcp.ack", "tcp.len",
            "udp.port", "udp.stream", "udp.time_delta"
        ]
        self.graph.build_domain_graph(self.feature_names)

    def test_caller_cannot_mutate_cached_parents(self):
        # Verify caller mutating the returned list does not corrupt cached parents.
        p1 = self.graph.get_parents("tcp_active_flags_count")
        self.assertIn("tcp.flags", p1)

        # Attempt to mutate returned list
        p1.append("TAMPERED_PARENT")
        p1.clear()

        # Subsequent query must return uncorrupted fresh list
        p2 = self.graph.get_parents("tcp_active_flags_count")
        self.assertIn("tcp.flags", p2)
        self.assertNotIn("TAMPERED_PARENT", p2)
        self.assertIsNot(p1, p2, "Returned list must be a distinct, fresh instance")

    def test_caller_cannot_mutate_cached_descendants(self):
        # Verify caller mutating the returned list does not corrupt cached descendants.
        d1 = self.graph.get_descendants("tcp.flags")
        self.assertIn("tcp_active_flags_count", d1)
        original_len = len(d1)

        # Attempt to mutate returned list
        d1.append("TAMPERED_DESCENDANT")
        d1.pop(0)

        # Subsequent query must return uncorrupted fresh list
        d2 = self.graph.get_descendants("tcp.flags")
        self.assertEqual(len(d2), original_len)
        self.assertIn("tcp_active_flags_count", d2)
        self.assertNotIn("TAMPERED_DESCENDANT", d2)
        self.assertIsNot(d1, d2, "Returned list must be a distinct, fresh instance")

    def test_cache_invalidation_after_add_causal_edge(self):
        # Verify dynamic edge addition invalidates cache and returns updated reachability.
        custom_graph = CraftCausalGraphSpecification()
        custom_graph.add_causal_edge("A", "B", weight=1.0)

        # Initial query caches descendants of A as ['B']
        d_before = custom_graph.get_descendants("A")
        self.assertEqual(d_before, ["B"])

        # Mutate graph: add edge B -> C
        custom_graph.add_causal_edge("B", "C", weight=1.0)

        # After mutation, cache must be invalidated and A must now have descendants B and C
        d_after = custom_graph.get_descendants("A")
        self.assertIn("B", d_after)
        self.assertIn("C", d_after)
        self.assertEqual(len(d_after), 2)

        # Verify parents cache invalidation on node C
        p_c = custom_graph.get_parents("C")
        self.assertEqual(p_c, ["B"])

        # Add edge D -> C
        custom_graph.add_causal_edge("D", "C", weight=1.0)
        p_c_updated = custom_graph.get_parents("C")
        self.assertIn("B", p_c_updated)
        self.assertIn("D", p_c_updated)
        self.assertEqual(len(p_c_updated), 2)

    def test_cache_invalidation_after_build_domain_graph(self):
        # Verify rebuild_domain_graph invalidates cache cleanly.
        custom_graph = CraftCausalGraphSpecification()
        custom_graph.add_causal_edge("X", "Y", weight=1.0)
        _ = custom_graph.get_descendants("X")
        self.assertIsNotNone(custom_graph._descendants_cache)

        # Rebuilding domain graph must invalidate
        custom_graph.build_domain_graph(["tcp.flags", "tcp_active_flags_count"])
        self.assertIsNone(custom_graph._descendants_cache)
        self.assertIsNone(custom_graph._parents_cache)


class TestCraftFullStreamEquivalence(unittest.TestCase):
    # Verifies bit-for-bit behavioral and numerical parity across the full stream.

    def test_full_stream_unchanged_results_against_uncached_baseline(self):
        rng = np.random.RandomState(42)
        n_features = 62
        feature_names = [f"f_{i}" for i in range(n_features)]
        domain_feats = [
            "tcp.flags", "tcp_active_flags_count", "tcp.connection.syn", "tcp.connection.fin",
            "tcp.connection.rst", "tcp.connection.synack", "tcp_seq_ack_ratio", "tcp.seq",
            "tcp.ack", "tcp.len", "tcp.checksum", "tcp.dstport", "is_well_known_dstport",
            "udp.port", "udp.stream", "udp.time_delta", "icmp.seq_le", "icmp.checksum"
        ]
        for idx, name in enumerate(domain_feats):
            if idx < n_features:
                feature_names[idx] = name

        X_train = rng.normal(0.0, 0.5, size=(200, n_features))

        # 1. Pipeline with Craft-optimized cache
        pipe_craft = CRAFTEdgePipeline(config={"detection_threshold": 0.65, "seed": 42})
        pipe_craft.fit(X_train, feature_names)
        pipe_craft.start()

        # 2. Reference Pipeline with standard uncached CausalGraphSpecification
        pipe_ref = CRAFTEdgePipeline(config={"detection_threshold": 0.65, "seed": 42})
        pipe_ref.causal_engine = CausalInferenceEngine(causal_graph=CausalGraphSpecification())
        pipe_ref.fit(X_train, feature_names)
        pipe_ref.start()

        # Stream 500 records with multiple anomaly bursts
        anomaly_indices = set(rng.choice(range(20, 500), size=50, replace=False))

        for i in range(500):
            if i in anomaly_indices:
                feats = {fname: float(rng.normal(2.5, 0.5)) for fname in feature_names}
                feats["tcp.flags"] = float(rng.normal(8.0, 0.5))
                feats["tcp_active_flags_count"] = float(rng.normal(12.0, 0.8))
                feats["tcp.connection.syn"] = float(rng.normal(10.0, 0.5))
            else:
                feats = {fname: float(rng.normal(0.0, 0.5)) for fname in feature_names}

            vec = np.array([feats[f] for f in feature_names], dtype=np.float64)
            inp = BenchmarkInput(
                stream_position=i,
                timestamp=float(i),
                timestamp_str=str(i),
                device_id="sensor_01",
                edge_node_id="node_01",
                features=feats,
                feature_vector=vec
            )

            res_craft = pipe_craft.process(inp)
            res_ref = pipe_ref.process(inp)

            # Assert detection parity
            self.assertEqual(res_craft.is_anomaly, res_ref.is_anomaly)
            self.assertAlmostEqual(res_craft.anomaly_score, res_ref.anomaly_score, places=4)

            # When anomaly triggered, assert diagnosis and recovery parity
            if res_craft.is_anomaly:
                diag_craft = pipe_craft.last_diagnosis
                diag_ref = pipe_ref.last_diagnosis
                self.assertEqual(diag_craft.ranked_root_causes, diag_ref.ranked_root_causes)
                for k in diag_craft.confidence_scores:
                    self.assertAlmostEqual(
                        diag_craft.confidence_scores[k],
                        diag_ref.confidence_scores[k],
                        places=4
                    )
                # Assert mitigation action and reward parity
                self.assertEqual(
                    res_craft.raw_output.get("action"),
                    res_ref.raw_output.get("action")
                )
                self.assertAlmostEqual(
                    res_craft.raw_output.get("reward", 0.0),
                    res_ref.raw_output.get("reward", 0.0),
                    places=4
                )

        # Assert final Q-table convergence bit-for-bit parity
        self.assertEqual(len(pipe_craft.policy.q_table), len(pipe_ref.policy.q_table))
        for k in pipe_craft.policy.q_table:
            self.assertIn(k, pipe_ref.policy.q_table)
            self.assertAlmostEqual(
                pipe_craft.policy.q_table[k],
                pipe_ref.policy.q_table[k],
                places=6
            )


if __name__ == "__main__":
    unittest.main()
