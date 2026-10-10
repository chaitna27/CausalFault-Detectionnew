# Full Common Test Benchmark Runner (Phase 7 Part E - P).
# Evaluates all five approaches on the processed Edge-IIoTset test stream under
# identical conditions, calibrated thresholds, single-thread CPU execution,
# and system instrumentation.

import os
import sys
import json
import time
import psutil
import platform
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pyrefly: ignore [missing-import]
import torch
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

from algorithms import (
    CausalFaultTolerancePipeline,
    IPFTAlgorithm,
    BWOAIAlgorithm,
    RCDAlgorithm,
    PreGANAlgorithm,
)
from algorithms.base import AlgorithmCapability
from evaluation.data_harness import CommonDataHarness, StreamConfig
from evaluation.instrumentation import SystemInstrumentation
from evaluation.metrics import (
    calculate_detection_accuracy,
    calculate_detection_latency_mttd,
    calculate_diagnosis_latency,
    calculate_recovery_latency,
    calculate_end_to_end_latency,
    calculate_mean_std,
    calculate_confidence_interval,
    calculate_service_availability,
    calculate_mttf_mttr,
    calculate_operational_recovery_cost,
    build_result_record,
)
from evaluation.results_schema import MeasurementType, Comparability, ResultsCollection


def load_training_sample(n_rows: int = 1000) -> np.ndarray:
    train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
    with open(os.path.join("data", "processed", "artifacts", "feature_names.json")) as f:
        feature_names = json.load(f)
    df = pd.read_csv(train_path, nrows=n_rows)
    return df[feature_names].values


def get_hardware_capacity() -> Dict[str, Any]:
    return {
        "cpu_model": platform.processor() or "x86_64 Compatible",
        "physical_cores": psutil.cpu_count(logical=False) or 4,
        "logical_cores": psutil.cpu_count(logical=True) or 8,
        "base_frequency_mhz": getattr(psutil.cpu_freq(), "current", 2400.0) if psutil.cpu_freq() else 2400.0,
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_present": torch.cuda.is_available(),
    }


def run_benchmark_trial(
    algo_cls,
    algo_id: str,
    name: str,
    threshold: float,
    train_data: np.ndarray,
    seed: int,
    total_records: int = 25000,
    warmup_records: int = 1000,
) -> Dict[str, Any]:
    print(f"\n[{name}] Initializing trial (seed={seed}, total={total_records}, warmup={warmup_records})...")

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = StreamConfig(
        dataset_name="edge_iiotset",
        split="test",
        data_dir="data/processed/test",
        warmup_count=warmup_records,
        max_records=total_records,
        batch_size=5000,
        seed=seed,
    )
    harness = CommonDataHarness(config)

    algo = algo_cls()
    algo.initialize({
        "threshold": threshold,
        "upper_threshold": threshold,
        "detection_threshold": threshold,
    })
    algo.fit(train_data)
    algo.start()

    instrumentation = SystemInstrumentation(slo_threshold_ms=50.0)

    y_true = []
    y_pred = []
    scores = []
    onset_times = []
    detect_times = []
    diag_times = []
    rec_times = []
    inference_latencies_ms = []

    warmup_count = 0
    eval_count = 0

    instrumentation.start_session()

    for ctx in harness.stream_contexts():
        obs = ctx.observable_input
        gt = ctx.ground_truth

        if ctx.is_warmup:
            algo.process(obs)
            warmup_count += 1
            continue

        eval_count += 1
        t_start = time.perf_counter_ns()
        res = algo.process(obs)
        t_proc_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        inference_latencies_ms.append(t_proc_ms)
        instrumentation.throughput_counter.record_observation(1)
        instrumentation.bandwidth_accountant.add_telemetry_bytes(len(obs.feature_vector) * 8)
        instrumentation.action_counter.record_latency(t_proc_ms)

        # Labels collected ONLY in evaluation layer
        is_attack = gt.is_fault
        y_true.append(1 if is_attack else 0)
        y_pred.append(1 if res.is_anomaly else 0)
        scores.append(res.anomaly_score)

        if is_attack:
            onset_times.append(obs.timestamp)
            if res.is_anomaly:
                detect_times.append(obs.timestamp + (t_proc_ms / 1000.0))

        # Check mitigation / actions
        mit = res.raw_output.get("mitigation")
        if mit and mit.action_type not in ["NONE", "NO_ACTION"]:
            if mit.state_bytes_transferred > 0:
                instrumentation.bandwidth_accountant.add_migration_state_bytes(mit.state_bytes_transferred)
            instrumentation.action_counter.record_action(
                "MIGRATION" if "MIGRATION" in mit.action_type else "REBALANCE",
                success=mit.success
            )
            rec_times.append(obs.timestamp + (t_proc_ms / 1000.0) + 0.05)

        diag = res.raw_output.get("diagnosis")
        if diag:
            diag_times.append(obs.timestamp + (t_proc_ms / 1000.0) + (diag.execution_time_ms / 1000.0))

    summary = instrumentation.end_session()
    algo.stop()
    algo.reset()

    # Calculate metrics
    accuracy = calculate_detection_accuracy(y_true, y_pred, scores)
    pr_auc_val = 0.0
    if len(set(y_true)) > 1 and scores:
        try:
            from sklearn.metrics import average_precision_score
            pr_auc_val = round(float(average_precision_score(y_true, scores)), 4)
        except Exception:
            pr_auc_val = 0.0

    mttd = calculate_detection_latency_mttd(onset_times[:len(detect_times)], detect_times)
    t_diag = calculate_diagnosis_latency(detect_times[:len(diag_times)], diag_times)
    t_rec = calculate_recovery_latency(diag_times[:len(rec_times)], rec_times)
    t_e2e = calculate_end_to_end_latency(onset_times[:len(rec_times)], rec_times)

    # Reliability metrics
    num_failures = int(np.sum(y_true))
    uptime_sec = summary["elapsed_seconds"]
    downtime_sec = (num_failures * (mttd + t_rec)) / 1000.0 if t_rec > 0 else 0.0
    avail_pct = calculate_service_availability(uptime_sec, downtime_sec)
    mttf_s, mttr_s = calculate_mttf_mttr(uptime_sec, downtime_sec, num_failures)

    return {
        "seed": seed,
        "eval_records": eval_count,
        "elapsed_seconds": summary["elapsed_seconds"],
        "throughput_rec_sec": summary["throughput_rec_sec"],
        "per_record_latency_ms": round(float(np.mean(inference_latencies_ms)), 4),
        "per_record_latency_p95_ms": round(float(np.percentile(inference_latencies_ms, 95)), 4),
        "avg_cpu_percent": summary["avg_cpu_percent"],
        "peak_rss_mb": summary["peak_rss_mb"],
        "bandwidth_kb_sec": summary["bandwidth_kb_sec"],
        "migration_bandwidth_kb_sec": round(instrumentation.bandwidth_accountant.migration_state_bytes / (1024.0 * max(1e-3, summary["elapsed_seconds"])), 2) if algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION) else "NOT_APPLICABLE",
        "estimated_energy_joules": summary["estimated_energy_joules"],
        "operational_recovery_cost": summary["operational_recovery_cost"],
        "precision": accuracy["precision"] if algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION) else "NOT_APPLICABLE",
        "recall": accuracy["recall"] if algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION) else "NOT_APPLICABLE",
        "f1": accuracy["f1"] if algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION) else "NOT_APPLICABLE",
        "roc_auc": accuracy.get("roc_auc", 0.0) if (algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION)) else "NOT_APPLICABLE",
        "pr_auc": pr_auc_val if (algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION)) else "NOT_APPLICABLE",
        "mttd_ms": round(mttd, 3) if (algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION)) and len(detect_times) > 0 else "NOT_APPLICABLE",
        "diagnosis_latency_ms": round(t_diag, 3) if algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS) and len(diag_times) > 0 else "NOT_APPLICABLE",
        "recovery_latency_ms": round(t_rec, 3) if algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION) and len(rec_times) > 0 else "NOT_APPLICABLE",
        "end_to_end_latency_ms": round(t_e2e, 3) if algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE) and len(rec_times) > 0 else "NOT_APPLICABLE",
        "availability_percent": round(avail_pct, 4) if algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE) else "NOT_APPLICABLE",
        "mttf_seconds": round(mttf_s, 4) if algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE) else "NOT_APPLICABLE",
        "mttr_seconds": round(mttr_s, 4) if algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE) else "NOT_APPLICABLE",
        "total_actions": summary["total_actions"],
        "slo_violation_rate": summary["slo_violation_rate"],
    }


def aggregate_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Computes mean, std, and 95% confidence intervals across trials.
    numeric_keys = [
        "throughput_rec_sec", "per_record_latency_ms", "per_record_latency_p95_ms",
        "avg_cpu_percent", "peak_rss_mb", "bandwidth_kb_sec", "estimated_energy_joules",
        "operational_recovery_cost", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ]
    agg = {}
    for k in numeric_keys:
        vals = [t[k] for t in trials if isinstance(t.get(k), (int, float))]
        if vals:
            mean_val, std_val = calculate_mean_std(vals)
            ci_val = calculate_confidence_interval(vals, confidence=0.95)
            agg[k] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "ci_95": [round(ci_val[0], 4), round(ci_val[1], 4)],
            }
        else:
            agg[k] = "NOT_APPLICABLE"

    # Pass-through non-numeric or conditional fields from first trial
    for k in ["mttd_ms", "diagnosis_latency_ms", "recovery_latency_ms", "end_to_end_latency_ms",
              "availability_percent", "mttf_seconds", "mttr_seconds", "migration_bandwidth_kb_sec"]:
        agg[k] = trials[0].get(k, "NOT_APPLICABLE")

    return agg


def main():
    train_data = load_training_sample(1000)
    hw_capacity = get_hardware_capacity()

    with open(os.path.join("results", "tables", "threshold_calibration.json")) as f:
        calib = json.load(f)

    # 5 Algorithms with calibrated thresholds
    algorithms = [
        (CausalFaultTolerancePipeline, "proposed_causal_ft", "Proposed Causal FT", calib["proposed_causal_ft"]["selected_threshold"]),
        (IPFTAlgorithm, "paper1_ipft", "IPFT (Theodoropoulos et al., 2022)", calib["paper1_ipft"]["selected_threshold"]),
        (BWOAIAlgorithm, "paper2_bwoaif", "BWOAIF (Hannák et al., 2023)", calib["paper2_bwoaif"]["selected_threshold"]),
        (RCDAlgorithm, "paper3_rcd", "RCD (Ikram et al., 2022)", 0.05),
        (PreGANAlgorithm, "paper4_pregan", "PreGAN (Tuli et al., 2022)", calib["paper4_pregan"]["selected_threshold"]),
    ]

    seeds = [42, 43, 44]
    benchmark_results = {
        "hardware_capacity": hw_capacity,
        "seeds": seeds,
        "total_records_per_run": 25000,
        "warmup_records": 1000,
        "algorithms": {},
    }

    for algo_cls, algo_id, name, tau in algorithms:
        trials = []
        for s in seeds:
            res = run_benchmark_trial(
                algo_cls=algo_cls,
                algo_id=algo_id,
                name=name,
                threshold=tau,
                train_data=train_data,
                seed=s,
                total_records=25000,
                warmup_records=1000,
            )
            trials.append(res)

        agg = aggregate_trials(trials)
        benchmark_results["algorithms"][algo_id] = {
            "name": name,
            "threshold": tau,
            "aggregated": agg,
            "trials": trials,
        }

    out_path = os.path.join("results", "raw", "full_benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nFull Benchmark successfully executed. Results saved to {out_path}.")


if __name__ == "__main__":
    main()
