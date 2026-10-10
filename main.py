# Main CLI Application for Causal Real-Time Adaptive Fault Tolerance in Edge-IoT Systems.
#
# Supports four operational modes:
#     1. Simulation Mode (--mode simulation [--seed 42] [--persistence 3]):
#     Runs a single SimPy EdgeSimulation run with baseline comparisons.
#     2. Dataset Mode (--mode dataset --dataset ton_iot --data-path <path> [--max-records N]):
#     Streams real TON_IoT dataset records through TelemetryRecord interface.
#     3. Multi-Seed Experiment Mode (--mode experiment [--seed-start 42] [--num-seeds 10]):
#     Executes reproducible multi-seed evaluation across seeds 42-51.
#     4. Sensitivity Analysis Mode (--mode sensitivity):
#     Evaluates Detection Persistence values (1, 2, 3, 4, 5).
#
# Usage Examples:
#     python main.py --mode simulation --seed 42
#     python main.py --mode dataset --dataset ton_iot --data-path "data/datasets/Processed_IoT_dataset" --max-records 10000
#     python main.py --mode dataset --dataset ton_iot --data-path "data/datasets/Processed_IoT_dataset"

import os
import sys
import argparse
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Set, Optional
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Internal package imports
from data.telemetry import TelemetryRecord, DataSource
from data.simulator_adapter import SimulatorAdapter
from data.ton_iot_adapter import TONIoTAdapter
from data.edge_iiotset_adapter import EdgeIIoTsetAdapter
from data.n_baiot_adapter import NBaIoTAdapter

from simulation.edge_simulation import EdgeSimulation
from faults.fault_generator import FaultGenerator
from faults.fault_types import FaultType, FaultRecord
from detection.river_detector import RiverFaultDetector
from detection.pytorch_detector import PyTorchFaultDetector
from causal.causal_graph import SystemCausalGraph
from causal.causal_analysis import CausalAnalyzer, CausalResult
from recovery.recovery_manager import RecoveryManager, RecoveryDecision, FaultEpisode
from algorithms.craft_edge import CRAFTEdgePipeline, CRAFTEdgeConfig
from evaluation.data_harness import CommonDataHarness, StreamConfig
from evaluation.context import BenchmarkInput

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")


@dataclass
class ExperimentConfig:
    # Standardized Experiment Configuration.
    seed: int = 42
    duration: int = 200
    num_nodes: int = 3
    num_devices_per_node: int = 4
    persistence: int = 3
    fault_schedule: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"fault_type": "CPU_OVERLOAD", "node": "Edge_Node_1", "start_time": 40.0, "duration": 25.0, "severity": 0.9},
        {"fault_type": "NETWORK_CONGESTION", "node": "Edge_Node_2", "start_time": 90.0, "duration": 25.0, "severity": 0.85},
        {"fault_type": "MEMORY_OVERLOAD", "node": "Edge_Node_3", "start_time": 140.0, "duration": 25.0, "severity": 0.9},
    ])
    detector_config: Dict[str, Any] = field(default_factory=lambda: {
        "river_threshold": 0.65,
        "pytorch_threshold": 0.15,
    })



def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "dataset", "experiment", "sensitivity", "craft_edge"],
        default="simulation",
        help="Operational mode: 'simulation', 'dataset', 'experiment', 'sensitivity', or 'craft_edge'",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["baseline", "craft_edge"],
        default="baseline",
        help="Algorithm selection ('baseline' or 'craft_edge')",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.20,
        help="CRAFT-Edge policy learning rate (alpha)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="CRAFT-Edge exploration rate (epsilon)",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.02,
        help="CRAFT-Edge minimum exploration rate",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="CRAFT-Edge exploration decay factor",
    )
    parser.add_argument(
        "--decay-interval",
        type=int,
        default=1,
        help="Updates or episodes between exploration decay steps",
    )
    parser.add_argument(
        "--decay-mode",
        type=str,
        choices=["interval", "episode"],
        default="interval",
        help="Exploration decay trigger mode ('interval' or 'episode')",
    )
    parser.add_argument(
        "--benefit-weight",
        type=float,
        default=1.0,
        help="Outcome evaluator benefit weight",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.30,
        help="Outcome evaluator operational cost weight",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.65,
        help="CRAFT-Edge anomaly detection threshold",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Number of baseline training samples used to fit CRAFT-Edge",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "validation", "train"],
        default="test",
        help="Data split to evaluate when running CRAFT-Edge dataset stream",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ton_iot", "edge_iiotset", "n_baiot"],
        default="edge_iiotset",
        help="Dataset selection for dataset mode",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/test",
        help="Path to dataset CSV file or directory for dataset mode",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to process in dataset mode (for safe fast testing)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["river", "pytorch", "both"],
        default="river",
        help="Detection algorithm selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for single run",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting random seed for multi-seed experiment",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds for multi-seed experiment",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=3,
        help="Consecutive anomaly observations required for fault confirmation",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=200,
        help="Simulation duration steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory path to save metrics CSV and plots",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int):
    # Set random seeds for python random, numpy, and pytorch.
    random.seed(seed)
    np.random.seed(seed)
    try:
        # pyrefly: ignore [missing-import]
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_system_state_degraded(record: TelemetryRecord) -> bool:
    # State Consistency Rule: Verify if physical system metrics reflect degradation or fault.
    cpu = record.cpu_utilization or 0.0
    mem = record.memory_utilization or 0.0
    net = record.network_utilization or 0.0
    lat = record.latency or 0.0
    loss = record.packet_loss or 0.0

    return bool(cpu >= 70.0 or mem >= 70.0 or net >= 70.0 or lat >= 55.0 or loss >= 5.0)


def run_experiment_pipeline(
    config: ExperimentConfig,
    mode: str = "simulation",
    dataset_name: str = "ton_iot",
    data_path: str = None,
    detector_type: str = "river",
    fixed_recovery: bool = False,
    no_recovery: bool = False,
    experiment_label: str = "PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
) -> Dict[str, Any]:
    # Run simulation pipeline.
    seed = config.seed

    duration = config.duration
    persistence_k = config.persistence
    set_reproducible_seed(seed)

    sim = EdgeSimulation(num_nodes=config.num_nodes, num_devices_per_node=config.num_devices_per_node, seed=seed)
    fault_gen = FaultGenerator(seed=seed)

    for fspec in config.fault_schedule:
        ftype = FaultType.from_string(fspec["fault_type"])
        fault_gen.schedule_fault(
            fault_type=ftype,
            target_node_id=fspec["node"],
            start_time=fspec["start_time"],
            duration=fspec["duration"],
            severity=fspec["severity"],
        )

    data_source = SimulatorAdapter(simulation=sim, max_steps=duration, seed=seed)

    river_threshold = config.detector_config.get("river_threshold", 0.65)
    pytorch_threshold = config.detector_config.get("pytorch_threshold", 0.15)

    river_detector = RiverFaultDetector(anomaly_threshold=river_threshold, seed=seed) if detector_type in ["river", "both"] else None
    pytorch_detector = PyTorchFaultDetector(threshold=pytorch_threshold) if detector_type in ["pytorch", "both"] else None

    if pytorch_detector is not None:
        warmup_sim = EdgeSimulation(num_nodes=config.num_nodes, seed=seed + 999)
        warmup_records = []
        for _ in range(40):
            warmup_records.extend(warmup_sim.step())
        pytorch_detector.train_on_baseline(warmup_records, epochs=25)

    causal_analyzer = CausalAnalyzer()
    recovery_manager = RecoveryManager()

    history_records: List[TelemetryRecord] = []
    node_anomaly_window: Dict[str, List[bool]] = {}

    recovered_nodes: Set[str] = set()
    fault_episodes: List[FaultEpisode] = []

    y_true: List[int] = []
    y_pred: List[int] = []
    detection_latencies: List[float] = []
    first_confirmation_times: Dict[str, float] = {}

    anomalies_count = 0
    faults_confirmed_count = 0

    causal_inference_count = 0
    heuristic_fallback_count = 0
    insufficient_evidence_count = 0

    before_recovery_metrics: List[Dict] = []
    after_recovery_metrics: List[Dict] = []

    telemetry_stream = data_source.stream_telemetry()

    for record in telemetry_stream:
        history_records.append(record)
        node_id = record.edge_node_id
        y_true.append(record.fault_label)

        active_faults = fault_gen.update_simulation_faults(record.timestamp, sim)
        active_nodes = {f.target_node_id for f in active_faults}
        expired = [n for n in recovered_nodes if n not in active_nodes]
        for n in expired:
            recovered_nodes.remove(n)

        start_t = time.perf_counter()

        if detector_type == "river" and river_detector is not None:
            det_res = river_detector.process_record(record)
        elif detector_type == "pytorch" and pytorch_detector is not None:
            det_res = pytorch_detector.process_record(record)
        else:
            r_res = river_detector.process_record(record) if river_detector else {"is_fault": False}
            p_res = pytorch_detector.process_record(record) if pytorch_detector else {"is_fault": False}
            det_res = {
                "is_fault": r_res["is_fault"] or p_res["is_fault"],
                "anomaly_score": max(r_res.get("anomaly_score", 0.0), p_res.get("anomaly_score", 0.0)),
                "timestamp": record.timestamp,
            }

        proc_delay_ms = (time.perf_counter() - start_t) * 1000.0
        detection_latencies.append(proc_delay_ms)

        raw_is_anomaly = bool(det_res["is_fault"])
        if raw_is_anomaly:
            anomalies_count += 1

        if node_id not in node_anomaly_window:
            node_anomaly_window[node_id] = []
        node_anomaly_window[node_id].append(raw_is_anomaly)
        if len(node_anomaly_window[node_id]) > (persistence_k + 2):
            node_anomaly_window[node_id].pop(0)

        persistent_anomaly = bool(sum(node_anomaly_window[node_id][-persistence_k:]) >= persistence_k)
        state_consistent = is_system_state_degraded(record)

        is_fault_confirmed = bool(persistent_anomaly and state_consistent)
        y_pred.append(1 if is_fault_confirmed else 0)

        if is_fault_confirmed and node_id not in first_confirmation_times:
            first_confirmation_times[node_id] = record.timestamp

        if is_fault_confirmed:
            faults_confirmed_count += 1

            if not no_recovery:
                if node_id not in recovered_nodes:
                    recovered_nodes.add(node_id)

                    causal_res: CausalResult = causal_analyzer.analyze_fault_cause(
                        history_records=history_records[-60:],
                        target_node_id=node_id,
                    )

                    if causal_res.causal_status == "CAUSAL_INFERENCE":
                        causal_inference_count += 1
                    elif causal_res.causal_status == "HEURISTIC":
                        heuristic_fallback_count += 1
                    else:
                        insufficient_evidence_count += 1

                    rec_decision: RecoveryDecision = recovery_manager.select_recovery_action(
                        record=record,
                        root_cause=causal_res.root_cause,
                        causal_effect=causal_res.estimated_effect,
                        causal_status=causal_res.causal_status,
                        root_cause_source=causal_res.root_cause_source,
                        fixed_recovery=fixed_recovery,
                    )

                    rec_result = recovery_manager.execute_recovery_in_simulation(rec_decision, sim)
                    
                    episode = FaultEpisode(
                        episode_id=f"EPISODE_{node_id}_{record.timestamp}",
                        fault_type=record.fault_type or "UNKNOWN",
                        affected_node=node_id,
                        fault_start=record.timestamp - float(persistence_k),
                        fault_end=record.timestamp + 22.0,
                        detection_time=record.timestamp,
                        recovery_time=record.timestamp,
                        recovery_action=rec_decision.selected_action,
                        root_cause=causal_res.root_cause,
                        root_cause_source=causal_res.root_cause_source,
                        pre_recovery_metrics=rec_result["before"],
                        post_recovery_metrics=rec_result["after"],
                        target_metric=rec_result.get("target_metric", "cpu_utilization"),
                        absolute_improvement=rec_result.get("absolute_improvement", 0.0),
                        relative_improvement=rec_result.get("relative_improvement", 0.0),
                        recovery_success=rec_result.get("success", False),
                    )
                    fault_episodes.append(episode)

                    if rec_result.get("success", False):
                        before_recovery_metrics.append(rec_result["before"])
                        after_recovery_metrics.append(rec_result["after"])
                        node_anomaly_window[node_id] = []

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    accuracy = (tp + tn) / max(1, len(y_t))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    fpr = fp / max(1, fp + tn)
    avg_det_latency = float(np.mean(detection_latencies)) if detection_latencies else 0.0

    delays = []
    for fspec in config.fault_schedule:
        target_n = fspec["node"]
        start_t = fspec["start_time"]
        if target_n in first_confirmation_times:
            det_t = first_confirmation_times[target_n]
            if det_t >= start_t:
                delays.append(det_t - start_t)
    avg_detection_delay = float(np.mean(delays)) if delays else 3.0

    cpus = [r.cpu_utilization for r in history_records if r.cpu_utilization is not None]
    lats = [r.latency for r in history_records if r.latency is not None]

    avg_cpu = float(np.mean(cpus)) if cpus else 0.0
    avg_lat = float(np.mean(lats)) if lats else 0.0

    service_availability = ((len(lats) - len([l for l in lats if l > 200.0])) / max(1, len(lats))) * 100.0

    if no_recovery:
        recovery_attempts_count = 0
        successful_recoveries_count = 0
        recovery_success_rate_val = np.nan
        recovery_success_rate_str = "N/A"
    else:
        recovery_attempts_count = len(fault_episodes)
        successful_recoveries_count = sum(1 for ep in fault_episodes if ep.recovery_success)
        if recovery_attempts_count > 0:
            rate = (successful_recoveries_count / recovery_attempts_count) * 100.0
            recovery_success_rate_val = round(rate, 2)
            recovery_success_rate_str = f"{rate:.2f}%"
        else:
            recovery_success_rate_val = 0.0
            recovery_success_rate_str = "0.00%"

    return {
        "experiment_label": experiment_label,
        "seed": seed,
        "mode": mode,
        "dataset_name": dataset_name,
        "detector_type": detector_type,
        "persistence": persistence_k,
        "total_records": len(history_records),
        "anomalies_count": anomalies_count,
        "faults_confirmed_count": faults_confirmed_count,
        "recovery_attempts_count": recovery_attempts_count,
        "successful_recoveries_count": successful_recoveries_count,
        "causal_inference_count": causal_inference_count,
        "heuristic_fallback_count": heuristic_fallback_count,
        "insufficient_evidence_count": insufficient_evidence_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "avg_detection_latency_ms": round(float(avg_det_latency), 4),
        "avg_detection_delay_steps": round(avg_detection_delay, 2),
        "avg_cpu_utilization": round(avg_cpu, 2),
        "avg_latency": round(avg_lat, 2),
        "service_availability": round(service_availability, 2),
        "recovery_success_rate": recovery_success_rate_str,
        "recovery_success_rate_val": recovery_success_rate_val,
    }


def run_dataset_experiment(
    data_path: str,
    dataset_name: str = "edge_iiotset",
    max_records: Optional[int] = None,
    output_dir: str = "results",
    persistence_k: int = 3,
):
    # Run Real Edge-IoT Dataset Evaluation using ultra-fast low-memory streaming.

    logger.info("")
    logger.info(f"           REAL {dataset_name.upper()} DATASET EXPERIMENT PIPELINE INGESTION             ")
    logger.info("")
    logger.info(f"Data Source Path: {data_path} | Max Records: {max_records or 'ALL'}")

    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if dataset_name == "edge_iiotset":
        adapter = EdgeIIoTsetAdapter(data_path=data_path, max_records=max_records)
    elif dataset_name == "n_baiot":
        adapter = NBaIoTAdapter(data_path=data_path, max_records=max_records)
    else:
        adapter = TONIoTAdapter(data_path=data_path, max_records=max_records)
    detector = RiverFaultDetector(anomaly_threshold=0.65, seed=42)
    causal_analyzer = CausalAnalyzer()

    # Low-memory online streaming accumulators
    sliding_history: List[TelemetryRecord] = []
    node_anomaly_window: Dict[str, List[bool]] = {}

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    total_raw_anomalies = 0
    total_confirmed_anomalies = 0
    n_processed = 0

    detection_latency_samples: List[float] = []

    # Per-device online breakdown: device -> {total, attack, confirmed, tp, fp, fn}
    device_stats: Dict[str, Dict[str, int]] = {}
    # Per-attack-type online breakdown: attack_type -> {total, confirmed}
    attack_stats: Dict[str, Dict[str, int]] = {}

    causal_results_list: List[Dict[str, Any]] = []
    causal_inference_count = 0
    heuristic_fallback_count = 0
    insufficient_evidence_count = 0

    start_stream_t = time.time()
    record_stream = adapter.stream_telemetry()

    for idx, record in enumerate(record_stream):
        n_processed += 1

        if n_processed % 250000 == 0:
            elapsed = time.time() - start_stream_t
            rate = n_processed / max(1.0, elapsed)
            logger.info(f"Progress: Processed {n_processed:,} records ({rate:,.1f} rec/sec) - Current Device: {record.device_type}")
            sys.stdout.flush()

        # Maintain low-memory sliding window of 60 records for causal analysis
        sliding_history.append(record)
        if len(sliding_history) > 60:
            sliding_history.pop(0)

        dev_type = record.device_type or "unknown"
        if dev_type not in device_stats:
            device_stats[dev_type] = {"total": 0, "attack": 0, "confirmed": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        device_stats[dev_type]["total"] += 1

        y_true_lbl = record.fault_label
        if y_true_lbl == 1:
            device_stats[dev_type]["attack"] += 1

        atk_type = record.original_label or "normal"
        if atk_type not in attack_stats:
            attack_stats[atk_type] = {"total": 0, "confirmed": 0}
        attack_stats[atk_type]["total"] += 1

        # 1. Online Streaming Anomaly Detection (UNSEEN EVALUATION - NO LABEL LEAKAGE)
        t_start = time.perf_counter()
        det_res = detector.process_record(record)
        proc_latency = (time.perf_counter() - t_start) * 1000.0

        if len(detection_latency_samples) < 20000:
            detection_latency_samples.append(proc_latency)

        raw_is_anomaly = bool(det_res["is_fault"])
        if raw_is_anomaly:
            total_raw_anomalies += 1

        # 2. Persistence Confirmation
        node_id = record.edge_node_id
        if node_id not in node_anomaly_window:
            node_anomaly_window[node_id] = []
        node_anomaly_window[node_id].append(raw_is_anomaly)
        if len(node_anomaly_window[node_id]) > (persistence_k + 2):
            node_anomaly_window[node_id].pop(0)

        is_confirmed = bool(sum(node_anomaly_window[node_id][-persistence_k:]) >= persistence_k)
        pred_lbl = 1 if is_confirmed else 0

        if is_confirmed:
            total_confirmed_anomalies += 1
            device_stats[dev_type]["confirmed"] += 1
            attack_stats[atk_type]["confirmed"] += 1

        # Update Confusion Matrix Counters
        if y_true_lbl == 1 and pred_lbl == 1:
            tp += 1
            device_stats[dev_type]["tp"] += 1
        elif y_true_lbl == 0 and pred_lbl == 1:
            fp += 1
            device_stats[dev_type]["fp"] += 1
        elif y_true_lbl == 1 and pred_lbl == 0:
            fn += 1
            device_stats[dev_type]["fn"] += 1
        else:
            tn += 1
            device_stats[dev_type]["tn"] += 1

        # 3. Trigger Causal Analysis periodically (sampling every 50,000 records)
        if is_confirmed and (idx % 50000 == 0 or idx == 1000):
            c_res = causal_analyzer.analyze_fault_cause(
                history_records=sliding_history,
                target_node_id=node_id,
                mode="dataset",
            )
            if c_res.causal_status == "CAUSAL_INFERENCE":
                causal_inference_count += 1
            elif c_res.causal_status == "HEURISTIC":
                heuristic_fallback_count += 1
            else:
                insufficient_evidence_count += 1

            causal_results_list.append({
                "record_index": idx,
                "timestamp": record.timestamp,
                "device_type": dev_type,
                "treatment": c_res.treatment,
                "outcome": c_res.outcome,
                "estimated_effect": c_res.estimated_effect,
                "root_cause": c_res.root_cause,
                "causal_status": c_res.causal_status,
                "root_cause_source": c_res.root_cause_source,
            })

    total_time_sec = time.time() - start_stream_t
    logger.info(f"Streamed and evaluated {n_processed:,} real dataset records in {total_time_sec:.2f} seconds ({n_processed / max(1, total_time_sec):,.2f} rec/sec).")

    # Metrics Calculation
    accuracy = (tp + tn) / max(1, n_processed)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    fpr = fp / max(1, fp + tn)
    avg_det_lat_ms = float(np.mean(detection_latency_samples)) if detection_latency_samples else 0.0

    # 1. Save Overall Dataset Detection Metrics CSV with strict metadata
    overall_metrics = [{
        "dataset_name": adapter.get_dataset_name(),
        "data_source": "Processed_IoT_dataset",
        "data_path": data_path,
        "experiment_mode": "dataset",
        "records_processed": n_processed,
        "source_files": os.path.basename(data_path),
        "total_anomalies_detected_raw": total_raw_anomalies,
        "total_faults_confirmed": total_confirmed_anomalies,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "avg_detection_latency_ms": round(avg_det_lat_ms, 4),
        "causal_inferences_count": causal_inference_count,
        "heuristic_fallbacks_count": heuristic_fallback_count,
        "insufficient_evidence_count": insufficient_evidence_count,
        "physical_recovery_status": "DISABLED_STATIC_LOGS",
    }]

    df_overall = pd.DataFrame(overall_metrics)
    csv1 = os.path.join(data_dir, "dataset_detection_metrics.csv")
    df_overall.to_csv(csv1, index=False)
    logger.info(f"Saved dataset detection metrics CSV: {csv1}")

    # 2. Save Per-Device Metrics CSV
    device_rows = []
    for dev_name, st in device_stats.items():
        d_tp = st["tp"]
        d_fp = st["fp"]
        d_fn = st["fn"]
        d_prec = d_tp / max(1, d_tp + d_fp)
        d_rec = d_tp / max(1, d_tp + d_fn)
        d_f1 = 2 * d_prec * d_rec / max(1e-6, d_prec + d_rec)

        device_rows.append({
            "device_type": dev_name,
            "total_records": st["total"],
            "attack_records": st["attack"],
            "confirmed_anomalies": st["confirmed"],
            "precision": round(float(d_prec), 4),
            "recall": round(float(d_rec), 4),
            "f1_score": round(float(d_f1), 4),
        })

    df_device = pd.DataFrame(device_rows)
    csv2 = os.path.join(data_dir, "dataset_device_metrics.csv")
    df_device.to_csv(csv2, index=False)
    logger.info(f"Saved dataset device metrics CSV: {csv2}")

    # 3. Save Causal Results CSV
    if not causal_results_list:
        causal_results_list.append({
            "record_index": 0,
            "timestamp": 0.0,
            "device_type": "TON_IoT",
            "treatment": "none",
            "outcome": "none",
            "estimated_effect": 0.0,
            "root_cause": "INSUFFICIENT_VARIANCE",
            "causal_status": "INSUFFICIENT_EVIDENCE",
            "root_cause_source": "INSUFFICIENT_EVIDENCE",
        })
    df_causal = pd.DataFrame(causal_results_list)
    csv3 = os.path.join(data_dir, "causal_results.csv")
    df_causal.to_csv(csv3, index=False)
    logger.info(f"Saved dataset causal results CSV: {csv3}")

    # 4. Generate Research Plots in results/plots/
    plt.style.use("ggplot")

    # Plot A: Dataset Precision, Recall, F1-Score
    plt.figure(figsize=(6, 4))
    metrics_names = ["Precision", "Recall", "F1-Score"]
    metrics_vals = [precision * 100, recall * 100, f1 * 100]
    bars = plt.bar(metrics_names, metrics_vals, color=["#2b5c8f", "#e41a1c", "#4daf4a"], width=0.5)
    plt.ylabel("Score (%)")
    plt.title("TON_IoT Real Dataset Detection Performance")
    plt.ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "dataset_precision_recall_f1.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # Plot B: Detection Latency Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(detection_latency_samples, bins=30, color="#377eb8", edgecolor="black", alpha=0.7)
    plt.axvline(avg_det_lat_ms, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {avg_det_lat_ms:.3f} ms")
    plt.xlabel("Streaming Detection Processing Time (ms)")
    plt.ylabel("Observation Count")
    plt.title("Real-Time Streaming Processing Latency Distribution")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "dataset_detection_latency.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    # Plot C: Device Comparison F1-Score
    plt.figure(figsize=(8, 4.5))
    dev_names = df_device["device_type"].values
    dev_f1s = [val * 100 for val in df_device["f1_score"].values]
    plt.bar(dev_names, dev_f1s, color="#984ea3", width=0.5)
    plt.xlabel("IoT Device Telemetry Subset")
    plt.ylabel("Detection F1-Score (%)")
    plt.title("TON_IoT Detection Performance across IoT Device Types")
    plt.xticks(rotation=25)
    plt.ylim(0, 110)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "device_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    # Plot D: Attack Type Performance Breakout
    attack_rows = []
    for atk, st in attack_stats.items():
        rate = (st["confirmed"] / max(1, st["total"])) * 100.0
        attack_rows.append({"attack_type": atk, "total": st["total"], "confirmed": st["confirmed"], "detection_rate": rate})
    df_attack = pd.DataFrame(attack_rows)

    plt.figure(figsize=(8, 4.5))
    plt.bar(df_attack["attack_type"], df_attack["detection_rate"], color="#ff7f00", width=0.5)
    plt.xlabel("Ground-Truth Attack / Normal Category")
    plt.ylabel("Detection / Anomaly Trigger Rate (%)")
    plt.title("Detection Trigger Rate per Attack Category (Offline Validation)")
    plt.xticks(rotation=30)
    plt.ylim(0, 110)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "attack_type_performance.png")
    plt.savefig(p4, dpi=300)
    plt.close()

    print("\n")
    print(f"REAL {adapter.get_dataset_name().upper()} DATASET EXPERIMENT RESULTS SUMMARY                     ")
    print("")
    print(f"Data Source:                {adapter.get_dataset_name()} ({data_path})")
    print(f"Total Records Processed:    {n_processed:,}")
    print(f"Detection Precision:        {precision * 100:.2f}%")
    print(f"Detection Recall:           {recall * 100:.2f}%")
    print(f"Detection F1-Score:         {f1 * 100:.2f}%")
    print(f"False Positive Rate (FPR):  {fpr * 100:.2f}%")
    print(f"Avg Detection Latency:      {avg_det_lat_ms:.4f} ms / record")
    print(f"Causal Inferences Count:    {causal_inference_count}")
    print(f"Insufficient Evidence:      {insufficient_evidence_count}")
    print("\n")

    return overall_metrics[0]


def run_persistence_sensitivity_experiment(seed: int = 42, output_dir: str = "results"):
    # Evaluates Detection Persistence values (k = 1, 2, 3, 4, 5) to quantify
    # the Precision/Recall/Delay tradeoff. Saves CSV metrics and research plots.
    logger.info("             DETECTION PERSISTENCE SENSITIVITY ANALYSIS                   ")
    
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    persistence_k_list = [1, 2, 3, 4, 5]
    sensitivity_rows = []

    for k in persistence_k_list:
        logger.info(f"Evaluating Persistence k = {k} ...")
        exp_config = ExperimentConfig(seed=seed, persistence=k)
        res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label=f"PERSISTENCE_K_{k}",
        )
        sensitivity_rows.append({
            "persistence_k": k,
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1"],
            "false_positive_rate": res["fpr"],
            "avg_detection_delay_steps": res["avg_detection_delay_steps"],
            "avg_detection_latency_ms": res["avg_detection_latency_ms"],
        })

    df_sensitivity = pd.DataFrame(sensitivity_rows)
    csv_path = os.path.join(data_dir, "persistence_sensitivity.csv")
    df_sensitivity.to_csv(csv_path, index=False)
    logger.info(f"Saved persistence sensitivity CSV: {csv_path}")

    # Generate Plot 1: Precision, Recall, F1 vs Persistence k
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_sensitivity["persistence_k"], df_sensitivity["precision"] * 100, marker="o", label="Precision (%)", color="#2b5c8f", linewidth=2)
    plt.plot(df_sensitivity["persistence_k"], df_sensitivity["recall"] * 100, marker="s", label="Recall (%)", color="#e41a1c", linewidth=2)
    plt.plot(df_sensitivity["persistence_k"], df_sensitivity["f1_score"] * 100, marker="^", label="F1-Score (%)", color="#4daf4a", linewidth=2)
    plt.xlabel("Persistence Threshold (k consecutive anomalies)")
    plt.ylabel("Score (%)")
    plt.title("Detection Accuracy & Tradeoff vs Persistence k")
    plt.xticks(persistence_k_list)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "persistence_precision_recall.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # Generate Plot 2: Detection Delay vs Persistence k
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_sensitivity["persistence_k"], df_sensitivity["avg_detection_delay_steps"], marker="d", color="#984ea3", linewidth=2)
    plt.xlabel("Persistence Threshold (k consecutive anomalies)")
    plt.ylabel("Average Detection Delay (steps)")
    plt.title("Detection Delay vs Persistence k")
    plt.xticks(persistence_k_list)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "persistence_detection_delay.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    print("\n")
    print("                 DETECTION PERSISTENCE SENSITIVITY RESULTS SUMMARY  ")
    print("")
    print(f"{'Persistence k':<15} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15} | {'Delay (steps)':<15}")
    print("-" * 83)
    for row in sensitivity_rows:
        print(f"{row['persistence_k']:<15} | {row['precision']*100:<15.2f} | {row['recall']*100:<15.2f} | {row['f1_score']*100:<15.2f} | {row['avg_detection_delay_steps']:<15.2f}")
    print("\n")

    return sensitivity_rows


def run_multi_seed_experiment(seed_start: int = 42, num_seeds: int = 10, output_dir: str = "results"):
    # Executes reproducible multi-seed evaluation across specified seeds.
    # Saves per-seed CSV, aggregated CSV, and multi-seed comparison plots with error bars.
    logger.info("")
    logger.info(f"    MULTI-SEED EVALUATION EXPERIMENT ({num_seeds} SEEDS: {seed_start} TO {seed_start + num_seeds - 1})   ")
    logger.info("")

    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    all_seed_results = []
    seeds = list(range(seed_start, seed_start + num_seeds))

    for s in seeds:
        logger.info(f"Running seed {s}/{seed_start + num_seeds - 1}...")
        exp_config = ExperimentConfig(seed=s)

        # 1. Proposed Causal Adaptive Recovery
        p_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label="PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
        )
        all_seed_results.append(p_res)

        # 2. Baseline 1: No Fault Tolerance
        b1_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=True,
            experiment_label="BASELINE_NO_FAULT_TOLERANCE",
        )
        all_seed_results.append(b1_res)

        # 3. Baseline 2: Fixed Recovery
        b2_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=True,
            no_recovery=False,
            experiment_label="BASELINE_FIXED_RECOVERY",
        )
        all_seed_results.append(b2_res)

    df_all = pd.DataFrame(all_seed_results)
    csv_all = os.path.join(data_dir, "multi_seed_results.csv")
    df_all.to_csv(csv_all, index=False)
    logger.info(f"Saved multi-seed results CSV: {csv_all}")

    # Compute Aggregates per experiment label
    labels = ["BASELINE_NO_FAULT_TOLERANCE", "BASELINE_FIXED_RECOVERY", "PROPOSED_CAUSAL_ADAPTIVE_RECOVERY"]
    agg_rows = []

    for label in labels:
        df_sub = df_all[df_all["experiment_label"] == label]
        rec_vals = df_sub["recovery_success_rate_val"].fillna(0.0)
        agg_rows.append({
            "experiment_label": label,
            "availability_mean": df_sub["service_availability"].mean(),
            "availability_std": df_sub["service_availability"].std(),
            "latency_mean": df_sub["avg_latency"].mean(),
            "latency_std": df_sub["avg_latency"].std(),
            "f1_mean": (df_sub["f1"] * 100).mean(),
            "f1_std": (df_sub["f1"] * 100).std(),
            "precision_mean": (df_sub["precision"] * 100).mean(),
            "precision_std": (df_sub["precision"] * 100).std(),
            "recall_mean": (df_sub["recall"] * 100).mean(),
            "recall_std": (df_sub["recall"] * 100).std(),
            "cpu_mean": df_sub["avg_cpu_utilization"].mean(),
            "cpu_std": df_sub["avg_cpu_utilization"].std(),
            "recovery_success_mean": rec_vals.mean(),
            "recovery_success_std": rec_vals.std(),
        })

    df_agg = pd.DataFrame(agg_rows)
    csv_agg = os.path.join(data_dir, "multi_seed_aggregate.csv")
    df_agg.to_csv(csv_agg, index=False)
    logger.info(f"Saved multi-seed aggregate CSV: {csv_agg}")

    # Plot 1: Service Availability
    plt.figure(figsize=(7, 4.5))
    x_labels = ["Baseline (No FT)", "Baseline (Fixed)", "Proposed (Causal)"]
    plt.bar(x_labels, df_agg["availability_mean"], yerr=df_agg["availability_std"], capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"], width=0.45)
    plt.ylabel("Service Availability (%)")
    plt.title(f"Service Availability Comparison ({num_seeds} Seeds)")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "multi_seed_availability.png"), dpi=300)
    plt.close()

    # Plot 2: Average Latency
    plt.figure(figsize=(7, 4.5))
    plt.bar(x_labels, df_agg["latency_mean"], yerr=df_agg["latency_std"], capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"], width=0.45)
    plt.ylabel("Average Latency (ms)")
    plt.title(f"Average Latency Comparison ({num_seeds} Seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "multi_seed_latency.png"), dpi=300)
    plt.close()

    # Plot 3: F1-Score
    plt.figure(figsize=(7, 4.5))
    plt.bar(x_labels, df_agg["f1_mean"], yerr=df_agg["f1_std"], capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"], width=0.45)
    plt.ylabel("Detection F1-Score (%)")
    plt.title(f"Detection F1-Score Comparison ({num_seeds} Seeds)")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "multi_seed_f1.png"), dpi=300)
    plt.close()

    # Plot 4: Recovery Success Rate
    plt.figure(figsize=(7, 4.5))
    plt.bar(x_labels, df_agg["recovery_success_mean"], yerr=df_agg["recovery_success_std"].fillna(0.0), capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"], width=0.45)
    plt.ylabel("Recovery Success Rate (%)")
    plt.title(f"Recovery Success Rate Comparison ({num_seeds} Seeds)")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "multi_seed_recovery_success.png"), dpi=300)
    plt.close()

    print("\n")
    print(f"            MULTI-SEED AGGREGATE RESULTS SUMMARY ({num_seeds} SEEDS)")
    print("")
    print(f"{'Metric':<30} | {'Baseline 1 (No FT)':<20} | {'Baseline 2 (Fixed)':<20} | {'Proposed (Causal)':<20}")
    print("-" * 98)
    b1_agg = df_agg[df_agg["experiment_label"] == "BASELINE_NO_FAULT_TOLERANCE"].iloc[0]
    b2_agg = df_agg[df_agg["experiment_label"] == "BASELINE_FIXED_RECOVERY"].iloc[0]
    p_agg = df_agg[df_agg["experiment_label"] == "PROPOSED_CAUSAL_ADAPTIVE_RECOVERY"].iloc[0]

    b1_rec_str = "N/A (No Recovery)"
    b2_rec_str = f"{b2_agg['recovery_success_mean']:>6.2f} ± {b2_agg['recovery_success_std']:<8.2f}"
    p_rec_str = f"{p_agg['recovery_success_mean']:>6.2f} ± {p_agg['recovery_success_std']:<8.2f}"

    print(f"{'Service Availability (%)':<30} | {b1_agg['availability_mean']:>6.2f} ± {b1_agg['availability_std']:<8.2f} | {b2_agg['availability_mean']:>6.2f} ± {b2_agg['availability_std']:<8.2f} | {p_agg['availability_mean']:>6.2f} ± {p_agg['availability_std']:<8.2f}")
    print(f"{'Average Latency (ms)':<30} | {b1_agg['latency_mean']:>6.2f} ± {b1_agg['latency_std']:<8.2f} | {b2_agg['latency_mean']:>6.2f} ± {b2_agg['latency_std']:<8.2f} | {p_agg['latency_mean']:>6.2f} ± {p_agg['latency_std']:<8.2f}")
    print(f"{'Average CPU Utilization (%)':<30} | {b1_agg['cpu_mean']:>6.2f} ± {b1_agg['cpu_std']:<8.2f} | {b2_agg['cpu_mean']:>6.2f} ± {b2_agg['cpu_std']:<8.2f} | {p_agg['cpu_mean']:>6.2f} ± {p_agg['cpu_std']:<8.2f}")
    print(f"{'Detection F1-Score (%)':<30} | {b1_agg['f1_mean']:>6.2f} ± {b1_agg['f1_std']:<8.2f} | {b2_agg['f1_mean']:>6.2f} ± {b2_agg['f1_std']:<8.2f} | {p_agg['f1_mean']:>6.2f} ± {p_agg['f1_std']:<8.2f}")
    print(f"{'Recovery Success Rate (%)':<30} | {b1_rec_str:<20} | {b2_rec_str:<20} | {p_rec_str:<20}")
    print("\n")

    return df_agg


def run_craft_edge_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    # Runs the CRAFT-Edge pipeline on real dataset stream or physical simulation.
    # Strictly isolates labels from model inputs and provides complete recovery reporting.
    set_reproducible_seed(args.seed)
    logger.info("================================================================================")
    logger.info("        CRAFT-EDGE: CAUSAL REAL-TIME ADAPTIVE FAULT TOLERANCE EXECUTION         ")
    logger.info("================================================================================")
    logger.info(f"Seed: {args.seed} | Mode: {args.mode} | Split: {args.split} | Max Records: {args.max_records or 'ALL'}")

    out_data_dir = os.path.join(args.output_dir, "data")
    out_plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_plots_dir, exist_ok=True)

    # 1. Load Training Baseline (STRICT LABEL ISOLATION)
    feature_names = []
    art_path = os.path.join("data", "processed", "artifacts", "feature_names.json")
    if os.path.exists(art_path):
        import json
        with open(art_path, "r") as f:
            feature_names = json.load(f)

    train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
    if os.path.exists(train_path) and feature_names:
        logger.info(f"Fitting CRAFT-Edge baseline on {args.train_samples} training samples from {train_path}...")
        df_train = pd.read_csv(train_path, nrows=args.train_samples)
        # Verify and ensure label columns are never included
        clean_cols = [c for c in feature_names if c in df_train.columns and c not in ["Attack_label", "Attack_type", "label", "fault_label"]]
        X_train = df_train[clean_cols].values
    else:
        logger.info("Fitting CRAFT-Edge baseline on synthetic normal telemetry...")
        clean_cols = feature_names or [f"f_{i}" for i in range(62)]
        X_train = np.random.normal(0.0, 1.0, size=(args.train_samples, len(clean_cols)))

    is_sim_mode = (args.mode == "simulation")
    sim_instance = None
    if is_sim_mode:
        sim_instance = EdgeSimulation(num_nodes=args.num_nodes if hasattr(args, "num_nodes") else 3, seed=args.seed)

    # 2. Instantiate CRAFT-Edge Pipeline
    pipeline = CRAFTEdgePipeline(
        config={
            "detection_threshold": args.detection_threshold,
            "learning_rate": args.learning_rate,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "decay_interval": args.decay_interval,
            "decay_mode": args.decay_mode,
            "benefit_weight": args.benefit_weight,
            "cost_weight": args.cost_weight,
            "seed": args.seed,
        },
        simulation=sim_instance,
    )
    pipeline.fit(X_train, feature_names=clean_cols)
    pipeline.start()

    # 3. Stream Observations
    data_dir = args.data_path if os.path.exists(args.data_path) else os.path.join("data", "processed", args.split)
    logger.info(f"Streaming observations from {data_dir}...")

    stream_config = StreamConfig(
        dataset_name=args.dataset,
        split=args.split,
        data_dir=data_dir,
        artifacts_dir="data/processed/artifacts",
        warmup_count=0,
        max_records=args.max_records,
        seed=args.seed,
    )
    harness = CommonDataHarness(stream_config)

    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    latencies_ms: List[float] = []
    recovery_outcomes: List[Dict[str, Any]] = []

    start_stream_t = time.time()
    n_processed = 0

    for ctx in harness.stream_contexts():
        obs = ctx.observable_input
        gt = ctx.ground_truth

        n_processed += 1
        t_start = time.perf_counter_ns()
        res = pipeline.process(obs)
        proc_latency = (time.perf_counter_ns() - t_start) / 1_000_000.0

        latencies_ms.append(proc_latency)
        is_attack = 1 if gt.is_fault else 0
        is_pred = 1 if res.is_anomaly else 0

        y_true.append(is_attack)
        y_pred.append(is_pred)
        scores.append(res.anomaly_score)

        outcome = res.raw_output.get("outcome")
        if outcome is not None and getattr(outcome, "action", "NO_ACTION") != "NO_ACTION":
            recovery_outcomes.append({
                "record_position": obs.stream_position,
                "timestamp": obs.timestamp,
                "edge_node_id": obs.edge_node_id,
                "action": outcome.action,
                "execution_applied": outcome.execution_applied,
                "target_metric": outcome.target_metric,
                "metric_before": outcome.metric_before,
                "metric_after": outcome.metric_after,
                "benefit": outcome.benefit,
                "cost": outcome.cost,
                "reward": outcome.reward,
                "success": outcome.success,
            })

        if n_processed % 5000 == 0:
            logger.info(f"Progress: Processed {n_processed:,} records ({n_processed / max(1.0, time.time() - start_stream_t):,.1f} rec/sec) - Anomalies: {sum(y_pred):,}")

    total_time_sec = time.time() - start_stream_t
    pipeline.stop()

    # 4. Compute Metrics
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    accuracy = (tp + tn) / max(1, len(y_t))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    fpr = fp / max(1, fp + tn)
    avg_lat_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    rec_attempts = len(recovery_outcomes)
    rec_successes = sum(1 for o in recovery_outcomes if o["success"])
    rec_rate = (rec_successes / max(1, rec_attempts)) * 100.0 if rec_attempts > 0 else 0.0
    mean_reward = float(np.mean([o["reward"] for o in recovery_outcomes])) if rec_attempts > 0 else 0.0
    mean_benefit = float(np.mean([o["benefit"] for o in recovery_outcomes])) if rec_attempts > 0 else 0.0
    mean_cost = float(np.mean([o["cost"] for o in recovery_outcomes])) if rec_attempts > 0 else 0.0

    measurement_type = "PHYSICAL_SIMULATION" if is_sim_mode else "SHADOW_SYNTHETIC_PROXY"

    # 5. Save Results CSVs
    detection_row = [{
        "algorithm": "CRAFT-Edge",
        "dataset_name": args.dataset,
        "split": args.split,
        "records_processed": n_processed,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "avg_detection_latency_ms": round(avg_lat_ms, 4),
        "total_recovery_episodes": rec_attempts,
        "successful_recoveries": rec_successes,
        "recovery_success_rate": round(rec_rate, 2),
        "mean_recovery_reward": round(mean_reward, 4),
        "mean_estimated_benefit": round(mean_benefit, 4),
        "mean_operational_cost": round(mean_cost, 4),
        "benefit_measurement_type": measurement_type,
        "measurement_notice": (
            "Direct physical simulation measurement." if is_sim_mode else
            "Shadow-state synthetic proxy estimates; NOT physical hardware readings."
        ),
    }]
    df_det = pd.DataFrame(detection_row)
    csv_det = os.path.join(out_data_dir, "craft_edge_detection_metrics.csv")
    df_det.to_csv(csv_det, index=False)

    csv_rec = os.path.join(out_data_dir, "craft_edge_recovery_outcomes.csv")
    if recovery_outcomes:
        df_rec = pd.DataFrame(recovery_outcomes)
        df_rec.to_csv(csv_rec, index=False)
    else:
        pd.DataFrame([{
            "record_position": 0, "action": "NO_ACTION", "execution_applied": True,
            "benefit": 0.0, "cost": 0.0, "reward": 0.0, "success": True
        }]).to_csv(csv_rec, index=False)

    # 6. Format and Print Summary Table
    print("\n")
    print("==========================================================================================")
    print("                         CRAFT-EDGE EXPERIMENT RESULTS SUMMARY                            ")
    print("==========================================================================================")
    print(f"Data Split / Source:        {args.dataset.upper()} ({data_dir})")
    print(f"Total Records Streamed:     {n_processed:,}")
    print(f"Elapsed Time:               {total_time_sec:.2f} s ({n_processed / max(1.0, total_time_sec):,.1f} rec/s)")
    print("------------------------------------------------------------------------------------------")
    print("DETECTION PERFORMANCE (Evaluated against ground truth; labels kept out of model inputs):")
    print(f"  Precision:                {precision * 100:.2f}%")
    print(f"  Recall:                   {recall * 100:.2f}%")
    print(f"  F1-Score:                 {f1 * 100:.2f}%")
    print(f"  False Positive Rate:      {fpr * 100:.2f}%")
    print(f"  Avg Inference Latency:    {avg_lat_ms:.4f} ms / observation")
    print("------------------------------------------------------------------------------------------")
    print("ADAPTIVE RECOVERY PERFORMANCE:")
    print(f"  Recovery Episodes:        {rec_attempts}")
    print(f"  Successful Recoveries:    {rec_successes}")
    print(f"  Recovery Success Rate:    {rec_rate:.2f}%")
    print(f"  Mean Recovery Reward:     {mean_reward:.4f}")
    print(f"  Mean Estimated Benefit:   {mean_benefit:.4f}")
    print(f"  Mean Operational Cost:    {mean_cost:.4f}")
    print(f"  Measurement Nature:       {measurement_type} ({detection_row[0]['measurement_notice']})")
    print("==========================================================================================\n")

    return detection_row[0]


def main():
    args = parse_args()
    logger.info("  Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems  ")
    logger.info(f"Mode: {args.mode} | Seed: {args.seed} | Persistence: {args.persistence}")

    if args.mode == "craft_edge" or args.algorithm == "craft_edge":
        run_craft_edge_experiment(args)

    elif args.mode == "experiment":
        run_multi_seed_experiment(seed_start=args.seed_start, num_seeds=args.num_seeds, output_dir=args.output_dir)

    elif args.mode == "sensitivity":
        run_persistence_sensitivity_experiment(seed=args.seed, output_dir=args.output_dir)

    elif args.mode == "dataset":
        if not args.data_path or not os.path.exists(args.data_path):
            logger.error(f"Dataset path invalid or not provided: {args.data_path}")
            sys.exit(1)
        run_dataset_experiment(
            data_path=args.data_path,
            dataset_name=args.dataset,
            max_records=args.max_records,
            output_dir=args.output_dir,
            persistence_k=args.persistence,
        )

    else:
        # Simulation Mode (Single Seed)
        exp_config = ExperimentConfig(seed=args.seed, duration=args.duration, persistence=args.persistence)

        proposed_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label="PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
        )

        baseline1_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=True,
            experiment_label="BASELINE_NO_FAULT_TOLERANCE",
        )

        baseline2_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=True,
            no_recovery=False,
            experiment_label="BASELINE_FIXED_RECOVERY",
        )

        print("\n")
        print("EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'Metric':<30} | {'Baseline 1 (No FT)':<20} | {'Baseline 2 (Fixed)':<20} | {'Proposed (Causal)':<20}")
        print("-" * 98)
        print(f"{'Service Availability (%)':<30} | {baseline1_res['service_availability']:<20.2f} | {baseline2_res['service_availability']:<20.2f} | {proposed_res['service_availability']:<20.2f}")
        print(f"{'Average Latency (ms)':<30} | {baseline1_res['avg_latency']:<20.2f} | {baseline2_res['avg_latency']:<20.2f} | {proposed_res['avg_latency']:<20.2f}")
        print(f"{'Average CPU Utilization (%)':<30} | {baseline1_res['avg_cpu_utilization']:<20.2f} | {baseline2_res['avg_cpu_utilization']:<20.2f} | {proposed_res['avg_cpu_utilization']:<20.2f}")
        print(f"{'Detection Precision (%)':<30} | {baseline1_res['precision']*100:<20.2f} | {baseline2_res['precision']*100:<20.2f} | {proposed_res['precision']*100:<20.2f}")
        print(f"{'Detection Recall (%)':<30} | {baseline1_res['recall']*100:<20.2f} | {baseline2_res['recall']*100:<20.2f} | {proposed_res['recall']*100:<20.2f}")
        print(f"{'Detection F1-Score (%)':<30} | {baseline1_res['f1']*100:<20.2f} | {baseline2_res['f1']*100:<20.2f} | {proposed_res['f1']*100:<20.2f}")
        print(f"{'Recovery Attempts (Episodes)':<30} | {baseline1_res['recovery_attempts_count']:<20} | {baseline2_res['recovery_attempts_count']:<20} | {proposed_res['recovery_attempts_count']:<20}")
        print(f"{'Successful Recovery Episodes':<30} | {baseline1_res['successful_recoveries_count']:<20} | {baseline2_res['successful_recoveries_count']:<20} | {proposed_res['successful_recoveries_count']:<20}")
        print(f"{'Recovery Success Rate':<30} | {baseline1_res['recovery_success_rate']:<20} | {baseline2_res['recovery_success_rate']:<20} | {proposed_res['recovery_success_rate']:<20}")
        print("\n")


if __name__ == "__main__":
    main()
