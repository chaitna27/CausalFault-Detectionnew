#!/usr/bin/env python3
"""
Corrected Common Benchmark Runner (Phase 3, 4, 5).

Executes the verified five-algorithm benchmark under identical single-thread CPU execution,
calibrated thresholds, real psutil instrumentation, and robust episode-based lifecycle metrics.

Key Improvements:
1. Passes canonical 'detection_threshold' explicitly.
2. Uses real psutil monitoring (CPU %, Peak RSS MB).
3. Evaluates episode-based MTTD and detection coverage for discrete fault episodes.
4. Computes genuine confusion-matrix metrics (TP, TN, FP, FN, Acc, Bal Acc, Specificity, FPR, FNR).
5. Tracks recovery attempts, confirmed successes, and recovery success rate.
6. Strictly enforces NOT_APPLICABLE for unsupported capabilities (RCD in detection, BWOAIF/RCD in recovery).
7. Preserves original baseline results in results/raw/full_benchmark_results_v1_baseline.json
   and outputs corrected results to results/raw/full_benchmark_results_v2_corrected.json.
"""

import os
import sys
import json
import time
import platform
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from evaluation.metrics import calculate_mean_std, calculate_confidence_interval


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


def find_fault_episodes(y_true: List[int]) -> List[Tuple[int, int]]:
    """Identifies discrete contiguous fault episodes (start_idx, end_idx)."""
    episodes = []
    in_episode = False
    start_idx = 0
    for i, val in enumerate(y_true):
        if val == 1 and not in_episode:
            in_episode = True
            start_idx = i
        elif val == 0 and in_episode:
            in_episode = False
            episodes.append((start_idx, i - 1))
    if in_episode:
        episodes.append((start_idx, len(y_true) - 1))
    return episodes


def compute_episode_mttd(
    episodes: List[Tuple[int, int]],
    y_pred: List[int],
    inference_latencies_ms: Optional[List[float]] = None,
    nominal_step_ms: float = 1.0
) -> Tuple[Any, Any, float]:
    """
    Computes episode-based MTTD (in ms and discrete observation steps) and detection coverage (%).
    In streaming telemetry, each record is an observation step. Detection delay is:
    delay_ms = (idx - start_idx) * nominal_step_ms + t_inference.
    This avoids distortion from non-monotonic packet capture timestamps in shuffled datasets.
    """
    if not episodes:
        return "NOT_APPLICABLE", "NOT_APPLICABLE", 100.0

    delays_ms = []
    delays_steps = []
    detected_episodes = 0

    for start_idx, end_idx in episodes:
        for idx in range(start_idx, end_idx + 1):
            if y_pred[idx] == 1:
                step_delay = idx - start_idx
                t_proc = inference_latencies_ms[idx] if inference_latencies_ms and idx < len(inference_latencies_ms) else 0.0
                delays_steps.append(step_delay)
                delays_ms.append(step_delay * nominal_step_ms + t_proc)
                detected_episodes += 1
                break

    coverage_pct = round((detected_episodes / len(episodes)) * 100.0, 2)
    mttd_ms = round(float(np.mean(delays_ms)), 3) if delays_ms else "NOT_APPLICABLE"
    mttd_steps = round(float(np.mean(delays_steps)), 3) if delays_steps else "NOT_APPLICABLE"
    return mttd_ms, mttd_steps, coverage_pct


def compute_early_warning_time(
    episodes: List[Tuple[int, int]],
    y_pred: List[int],
    max_lookback: int = 5,
    nominal_step_ms: float = 1.0
) -> Any:
    """Computes predictive lead time (ms) before episode onset for predictive algorithms."""
    lead_times_ms = []
    for start_idx, _ in episodes:
        if start_idx == 0:
            continue
        lookback_start = max(0, start_idx - max_lookback)
        for idx in range(start_idx - 1, lookback_start - 1, -1):
            if y_pred[idx] == 1:
                lead_steps = start_idx - idx
                lead_times_ms.append(float(lead_steps) * nominal_step_ms)
                break
    return round(float(np.mean(lead_times_ms)), 3) if lead_times_ms else 0.0


def compute_confusion_metrics(y_true: List[int], y_pred: List[int], scores: List[float]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) > 1 else acc
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 and scores else 0.5
    pr_auc = average_precision_score(y_true, scores) if len(set(y_true)) > 1 and scores else 0.0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": round(float(acc * 100.0), 2),
        "balanced_accuracy": round(float(bal_acc * 100.0), 2),
        "specificity": round(float(spec * 100.0), 2),
        "fpr": round(float(fpr * 100.0), 2),
        "fnr": round(float(fnr * 100.0), 2),
        "precision": round(float(prec * 100.0), 2),
        "recall": round(float(rec * 100.0), 2),
        "f1": round(float(f1 * 100.0), 2),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
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
    print(f"\n[{name}] Running corrected trial (seed={seed}, total={total_records}, warmup={warmup_records})...")

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
    # Explicitly pass canonical detection_threshold as well as aliases
    algo.initialize({
        "detection_threshold": threshold,
        "threshold": threshold,
        "upper_threshold": threshold,
    })
    algo.fit(train_data)
    algo.start()

    instrumentation = SystemInstrumentation(slo_threshold_ms=50.0)

    y_true = []
    y_pred = []
    scores = []
    timestamps = []
    inference_latencies_ms = []
    diag_execution_latencies_ms = []
    recovery_latencies_ms = []

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

        is_attack = gt.is_fault
        y_true.append(1 if is_attack else 0)
        y_pred.append(1 if res.is_anomaly else 0)
        scores.append(float(res.anomaly_score))
        timestamps.append(float(obs.timestamp))

        # Check mitigation / actions
        mit = res.raw_output.get("mitigation")
        if mit and mit.action_type not in ["NONE", "NO_ACTION"]:
            if mit.state_bytes_transferred > 0:
                instrumentation.bandwidth_accountant.add_migration_state_bytes(mit.state_bytes_transferred)
            instrumentation.action_counter.record_action(
                "MIGRATION" if "MIGRATION" in mit.action_type else "REBALANCE",
                success=mit.success,
            )
            recovery_latencies_ms.append(0.05 * 1000.0)  # 50 ms simulated action latency

        diag = res.raw_output.get("diagnosis")
        if diag:
            diag_execution_latencies_ms.append(diag.execution_time_ms)

    summary = instrumentation.end_session()
    algo.stop()
    algo.reset()

    # Identify distinct fault episodes
    episodes = find_fault_episodes(y_true)

    # Episode-based MTTD
    mttd_val, mttd_steps, coverage_pct = compute_episode_mttd(episodes, y_pred, inference_latencies_ms)

    # Early warning lead time
    early_warning_lead = "NOT_APPLICABLE"
    if algo.supports(AlgorithmCapability.RESOURCE_PREDICTION):
        early_warning_lead = compute_early_warning_time(episodes, y_pred)

    # Confusion matrix metrics
    is_detection_capable = algo.supports(AlgorithmCapability.STREAMING_DETECTION) or algo.supports(AlgorithmCapability.RESOURCE_PREDICTION)
    conf_metrics = compute_confusion_metrics(y_true, y_pred, scores) if is_detection_capable else {}

    # Recovery metrics
    has_recovery = algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION) or algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE)
    total_actions = instrumentation.action_counter.get_total_actions()
    successful_actions = instrumentation.action_counter.mitigation_successes
    recovery_success_rate = round((successful_actions / total_actions) * 100.0, 2) if total_actions > 0 else ("NOT_APPLICABLE" if not has_recovery else 0.0)
    recovery_latency_val = round(float(np.mean(recovery_latencies_ms)), 3) if recovery_latencies_ms else ("NOT_APPLICABLE" if not has_recovery else 0.0)

    # RCA direct computation latency
    rca_computation_latency = round(float(np.mean(diag_execution_latencies_ms)), 4) if diag_execution_latencies_ms else ("NOT_APPLICABLE" if not algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS) else 0.0)

    # Service Availability under injection
    # Degraded time is proportional to unresolved fault episodes
    total_time_sec = summary["elapsed_seconds"]
    if has_recovery:
        # Availability = (healthy time / total operational time)
        # Healthy time is total time minus unmitigated attack durations
        unmitigated_attacks = int(np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]))
        degraded_ratio = unmitigated_attacks / max(1, len(y_true))
        avail_pct = round(max(0.0, (1.0 - degraded_ratio)) * 100.0, 2)
    else:
        avail_pct = "NOT_APPLICABLE"

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
        "operational_recovery_cost": summary["operational_recovery_cost"] if has_recovery else "NOT_APPLICABLE",
        "total_actions": total_actions if has_recovery else "NOT_APPLICABLE",
        "successful_actions": successful_actions if has_recovery else "NOT_APPLICABLE",
        "recovery_success_rate": recovery_success_rate,
        "recovery_latency_ms": recovery_latency_val,
        "rca_computation_latency_ms": rca_computation_latency,
        "mttd_ms": mttd_val if is_detection_capable else "NOT_APPLICABLE",
        "mttd_steps": mttd_steps if is_detection_capable else "NOT_APPLICABLE",
        "detection_coverage_percent": coverage_pct if is_detection_capable else "NOT_APPLICABLE",
        "early_warning_time_ms": early_warning_lead,
        "availability_percent": avail_pct,
        # Confusion metrics
        "tp": conf_metrics.get("tp", "NOT_APPLICABLE"),
        "tn": conf_metrics.get("tn", "NOT_APPLICABLE"),
        "fp": conf_metrics.get("fp", "NOT_APPLICABLE"),
        "fn": conf_metrics.get("fn", "NOT_APPLICABLE"),
        "accuracy": conf_metrics.get("accuracy", "NOT_APPLICABLE"),
        "balanced_accuracy": conf_metrics.get("balanced_accuracy", "NOT_APPLICABLE"),
        "specificity": conf_metrics.get("specificity", "NOT_APPLICABLE"),
        "fpr": conf_metrics.get("fpr", "NOT_APPLICABLE"),
        "fnr": conf_metrics.get("fnr", "NOT_APPLICABLE"),
        "precision": conf_metrics.get("precision", "NOT_APPLICABLE"),
        "recall": conf_metrics.get("recall", "NOT_APPLICABLE"),
        "f1": conf_metrics.get("f1", "NOT_APPLICABLE"),
        "roc_auc": conf_metrics.get("roc_auc", "NOT_APPLICABLE"),
        "pr_auc": conf_metrics.get("pr_auc", "NOT_APPLICABLE"),
        "slo_violation_rate": summary["slo_violation_rate"],
        "raw_scores": scores,
        "raw_y_true": y_true,
        "raw_y_pred": y_pred,
    }


def main():
    print("=" * 75)
    print("CORRECTED COMMON TEST BENCHMARK RUNNER (Phases 3, 4, 5)")
    print("=" * 75)

    train_data = load_training_sample(1000)
    print(f"Loaded training baseline shape: {train_data.shape}")

    calib_path = os.path.join("results", "tables", "threshold_calibration.json")
    with open(calib_path) as f:
        calib = json.load(f)

    benchmark_plan = [
        (CausalFaultTolerancePipeline, "proposed_causal_ft", "Proposed Causal FT", calib["proposed_causal_ft"]["selected_threshold"]),
        (IPFTAlgorithm, "paper1_ipft", "IPFT (Theodoropoulos et al., 2022)", calib["paper1_ipft"]["selected_threshold"]),
        (BWOAIAlgorithm, "paper2_bwoaif", "BWOAIF (Hannák et al., 2023)", calib["paper2_bwoaif"]["selected_threshold"]),
        (RCDAlgorithm, "paper3_rcd", "RCD (Ikram et al., 2022)", 0.05),
        (PreGANAlgorithm, "paper4_pregan", "PreGAN (Tuli et al., 2022)", calib["paper4_pregan"]["selected_threshold"]),
    ]

    seeds = [42, 43, 44]
    results = {
        "hardware_capacity": get_hardware_capacity(),
        "seeds": seeds,
        "total_records_per_run": 25000,
        "warmup_records": 1000,
        "algorithms": {},
    }

    per_seed_rows = []
    confusion_rows = []

    for algo_cls, algo_id, name, tau in benchmark_plan:
        print(f"\n==================================================")
        print(f"Evaluating: {name} (Threshold: {tau})")
        print(f"==================================================")

        trials = []
        for seed in seeds:
            trial_res = run_benchmark_trial(
                algo_cls=algo_cls,
                algo_id=algo_id,
                name=name,
                threshold=tau,
                train_data=train_data,
                seed=seed,
            )
            # Save raw arrays separately if needed, strip from JSON trial summary
            raw_scores = trial_res.pop("raw_scores")
            raw_y_true = trial_res.pop("raw_y_true")
            raw_y_pred = trial_res.pop("raw_y_pred")

            trials.append(trial_res)

            # Per seed table record
            seed_row = {"algorithm_id": algo_id, "algorithm_name": name}
            seed_row.update(trial_res)
            per_seed_rows.append(seed_row)

            # Confusion record
            if trial_res.get("accuracy") != "NOT_APPLICABLE":
                conf_row = {
                    "algorithm_id": algo_id,
                    "algorithm_name": name,
                    "seed": seed,
                    "tp": trial_res["tp"],
                    "tn": trial_res["tn"],
                    "fp": trial_res["fp"],
                    "fn": trial_res["fn"],
                    "accuracy": trial_res["accuracy"],
                    "balanced_accuracy": trial_res["balanced_accuracy"],
                    "specificity": trial_res["specificity"],
                    "fpr": trial_res["fpr"],
                    "fnr": trial_res["fnr"],
                    "precision": trial_res["precision"],
                    "recall": trial_res["recall"],
                    "f1": trial_res["f1"],
                    "roc_auc": trial_res["roc_auc"],
                    "pr_auc": trial_res["pr_auc"],
                }
                confusion_rows.append(conf_row)

        # Aggregate across seeds
        numeric_keys = [
            "throughput_rec_sec", "per_record_latency_ms", "per_record_latency_p95_ms",
            "avg_cpu_percent", "peak_rss_mb", "bandwidth_kb_sec", "estimated_energy_joules",
            "accuracy", "balanced_accuracy", "specificity", "fpr", "fnr", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "mttd_ms", "mttd_steps", "detection_coverage_percent",
            "early_warning_time_ms", "recovery_success_rate", "recovery_latency_ms",
            "rca_computation_latency_ms", "availability_percent", "migration_bandwidth_kb_sec", "operational_recovery_cost",
        ]

        aggregated = {}
        for key in numeric_keys:
            vals = [t[key] for t in trials if isinstance(t.get(key), (int, float))]
            if vals:
                m, s = calculate_mean_std(vals)
                ci = calculate_confidence_interval(vals)
                aggregated[key] = {
                    "mean": round(float(m), 4),
                    "std": round(float(s), 4),
                    "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
                }
            else:
                aggregated[key] = "NOT_APPLICABLE"

        results["algorithms"][algo_id] = {
            "name": name,
            "threshold": tau,
            "aggregated": aggregated,
            "trials": trials,
        }

    # Save outputs
    out_json = os.path.join("results", "raw", "full_benchmark_results_v2_corrected.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved corrected benchmark JSON: {out_json}")

    # Save per-seed results CSV
    df_seed = pd.DataFrame(per_seed_rows)
    seed_csv = os.path.join("results", "tables", "benchmark_results_by_seed.csv")
    df_seed.to_csv(seed_csv, index=False)
    print(f"[OK] Saved updated per-seed CSV: {seed_csv}")

    # Save confusion metrics CSV
    df_conf = pd.DataFrame(confusion_rows)
    conf_csv = os.path.join("results", "tables", "detection_confusion_metrics_by_seed.csv")
    df_conf.to_csv(conf_csv, index=False)
    print(f"[OK] Saved detection confusion metrics by seed: {conf_csv}")

    # Compute confusion metrics mean + std
    conf_summary = []
    for algo_id in ["proposed_causal_ft", "paper1_ipft", "paper2_bwoaif", "paper4_pregan"]:
        sub = df_conf[df_conf["algorithm_id"] == algo_id]
        if len(sub) == 0:
            continue
        row = {"algorithm_id": algo_id, "algorithm_name": sub["algorithm_name"].iloc[0]}
        for m in ["tp", "tn", "fp", "fn"]:
            row[f"{m}_mean"] = round(float(sub[m].mean()), 1)
        for m in ["accuracy", "balanced_accuracy", "specificity", "fpr", "fnr", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            m_val, s_val = calculate_mean_std(sub[m].tolist())
            row[f"{m}_mean"] = round(m_val, 2 if m not in ["roc_auc", "pr_auc"] else 4)
            row[f"{m}_std"] = round(s_val, 2 if m not in ["roc_auc", "pr_auc"] else 4)
        conf_summary.append(row)

    df_conf_sum = pd.DataFrame(conf_summary)
    conf_sum_csv = os.path.join("results", "tables", "detection_metrics_mean_std.csv")
    df_conf_sum.to_csv(conf_sum_csv, index=False)
    print(f"[OK] Saved detection confusion summary (mean ± std): {conf_sum_csv}")

    print("\n" + "=" * 75)
    print("ALL CORRECTED BENCHMARK RUNS AND CONFUSION METRICS COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
