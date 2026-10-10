# Scientific Cross-Algorithm Benchmark Evaluation: Visual Artifacts & Findings

This directory contains publication-ready comparison plots generated from Vagmin's verified benchmark suite on the **Edge-IIoTset** dataset across three independent trials (**Random Seeds 42, 43, and 44**).

---

## 1. Experimental Protocol & Execution Integrity

* **Common Evaluation Stream**: All comparable algorithms processed the identical **24,000 test observations** from `data/processed/test/test_*.csv` under common single-thread CPU affinity (`torch.set_num_threads(1)`).
* **Warm-up Isolation**: The first **1,000 records** were strictly designated as warm-up and excluded from all detection and computational metrics.
* **Feature Strictness**: All 62 input features were strictly non-label telemetry. `Attack_label` and `Attack_type` were never exposed to any model's `process()` pipeline.
* **Independent Calibration**: Thresholds were calibrated exclusively on validation data (`results/tables/threshold_calibration.json`) prior to test evaluation.
* **Error Bars**: Error bars represent the sample standard deviation ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$, with $N=3$ and `ddof=1`).
* **Hardware Environment**: 12-core / 16-thread x86_64 host (Intel Core i5-13500H), 15.69 GB RAM, CPU-only execution.

---

## 2. Capability Matrix & Scientific Fairness Boundaries

Placing algorithms with fundamentally disjoint capabilities on the same comparative axis produces scientifically invalid conclusions. The table below delineates the verified capabilities of each evaluated approach:

| Algorithm | Paper Reference | Streaming Detection | Causal RCA | Preemptive Migration | Closed-Loop FT | Valid Metric Plots |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Proposed Causal FT** | *Our Pipeline* | **Yes** | **Yes** | No | **Yes** | Detection, RCA, Computational, Reliability |
| **IPFT** | Theodoropoulos et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |
| **BWOAIF** | Hannák et al., 2023 | **Yes** | No | No | No | Detection, Computational |
| **RCD** | Ikram et al., 2022 | No | **Yes** | No | No | RCA, Computational |
| **PreGAN** | Tuli et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |

### Exclusion and N/A Rationale
1. **Why RCD is Excluded from Detection Performance**:
   RCD (*Robust Causal Diagnosis*, Ikram et al., 2022) is an offline root-cause diagnosis algorithm that takes known anomalous episodes and builds a localized causal graph. It does **not** perform online binary anomaly detection or output streaming anomaly scores. Assigning 0% precision or recall to RCD would be scientifically dishonest; it is designated as `N/A`.
2. **Why BWOAIF and RCD are Excluded from Mitigation/Recovery Plots**:
   Neither BWOAIF nor RCD implements closed-loop migration or recovery actions. They are marked `N/A` rather than 0% recovery latency.
3. **Why Accuracy and False-Positive Rate (FPR) are Not Plotted**:
   In strict accordance with the scientific mandate (*"Never hard-code, invent, estimate or manually alter metric values"*), Accuracy and FPR were not recorded in `full_benchmark_results.json` by `run_full_benchmark.py`. They are reported as `N/A (not recorded in benchmark trial output)` in summary tables.

---

## 3. Generated Plots Inventory

### Category A: Detection Performance
* **`comparison_detection_f1_precision_recall.png`**:
  Grouped bar chart displaying Precision, Recall, and F1-score (Mean ± Std, $N=3$) for Causal FT, IPFT, BWOAIF, and PreGAN.
* **`comparison_detection_auc.png`**:
  Discrimination capability evaluated by ROC-AUC and PR-AUC. Causal FT achieves the highest ROC-AUC (0.7094) and PR-AUC (0.3780).

### Category B: Root-Cause Diagnosis (RCA)
* **`comparison_rca_localization_recall.png`**:
  Top-1, Top-3, and Top-5 recall for Causal FT vs RCD on both Synthetic Controlled Ground Truth (7 trials) and Edge-IIoTset Domain Proxy (25 episodes).
* **`comparison_rca_diagnosis_latency.png`**:
  Diagnosis computation latency comparing Causal FT (0.1205 ms synthetic / 0.2511 ms Edge-IIoTset) against RCD (1.8427 ms synthetic / 13.6292 ms Edge-IIoTset).

### Category C: Recovery and Mitigation
* **`comparison_mitigation_availability.png`**:
  Service availability under injection for closed-loop models.
* **`comparison_migration_bandwidth.png`**:
  Network overhead incurred by preemptive container migration (IPFT: 64,152.25 KB/s vs PreGAN: 67,343.28 KB/s).
* **`comparison_operational_recovery_cost.png`**:
  Normalized operational cost of mitigation actions.

### Category D: Computational & System Performance (All 5 Algorithms)
* **`comparison_throughput.png`**:
  Stream processing rate (records/second). RCD achieves 1,996.66 rec/s, Causal FT achieves 1,923.23 rec/s, BWOAIF achieves 1,774.11 rec/s, IPFT achieves 625.84 rec/s, and PreGAN achieves 331.17 rec/s.
* **`comparison_inference_latency.png`**:
  Per-observation inference latency comparing Mean and P95 latency (ms/record).
* **`comparison_resource_utilization.png`**:
  Dual-panel comparison of single-core CPU utilization (%) and Peak Resident Set Size memory (MB). Causal FT exhibits the lowest memory footprint (332.76 MB vs 409–415 MB).
* **`comparison_energy_consumption.png`**:
  Total estimated energy consumption (Joules) over the 24,000 evaluated records.

### Category E: Reproducibility & Stability
* **`comparison_seed_stability_f1.png`**:
  Seed-wise F1 tracking across Seeds 42, 43, and 44. Exposes the catastrophic failure / collapse of PreGAN on Seed 43 (F1 = 0.0%, Recall = 0.0%).
* **`comparison_seed_stability_throughput.png`**:
  Throughput stability across seeds demonstrating low variance for Causal FT and BWOAIF versus extreme variance for IPFT and PreGAN.

---

## 4. Summary Table Reference
* **[benchmark_summary_mean_std.csv](../tables/benchmark_summary_mean_std.csv)**: Full aggregated statistical summary with sample means, standard deviations, and 95% confidence intervals.
* **[benchmark_results_by_seed.csv](../tables/benchmark_results_by_seed.csv)**: Granular record of every measurement for every algorithm and random seed.
