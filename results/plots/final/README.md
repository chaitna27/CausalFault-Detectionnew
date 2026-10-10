# Scientific Benchmark Results: Cross-Algorithm Comparison Suite

This directory contains a complete set of **26 publication-grade, 300-DPI figures** generated from the five-algorithm benchmark on the **Edge-IIoTset** dataset across three independent trials (**Random Seeds 42, 43, and 44**).

---

## 1. Experimental Protocol & Execution Integrity

* **Common Evaluation Stream:** All comparable algorithms processed the same **24,000 test observations** from `data/processed/test/test_*.csv` under common single-thread CPU affinity using `torch.set_num_threads(1)`.

* **Warm-up Isolation:** The first **1,000 records** were designated as warm-up observations and excluded from detection, timing, and computational performance metrics.

* **Feature Strictness:** All 62 input features consisted strictly of non-label telemetry. `Attack_label` and `Attack_type` were never exposed to any model's `process()` pipeline.

* **Independent Calibration:** Detection thresholds were calibrated exclusively using validation data prior to test evaluation.

* **Error Bars:** Error bars represent the sample standard deviation:

  $$
  s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}
  $$

  with \(N = 3\) and `ddof=1`.

* **Hardware & System Instrumentation:** Experiments were executed on an Intel Core i5-13500H host with 12 cores / 16 threads and 15.69 GB RAM. CPU utilization and resident memory usage were monitored using `psutil`.

* **Local Dependency Setup:** `psutil` was installed inside the project virtual environment using:

  ```bash
  .venv\Scripts\python.exe -m pip install psutil
  ```

  The original `requirements.txt` was preserved.

---

## 2. Root-Cause Correction of Causal FT Test F1

During the benchmark audit, a parameter-key mismatch was identified in `algorithms/causal_ft/pipeline.py`.

`CausalFaultTolerancePipeline.initialize()` expected:

```python
"detection_threshold"
```

while `run_full_benchmark.py` supplied:

```python
{"threshold": threshold, "upper_threshold": threshold}
```

As a result, the validation-calibrated threshold of **0.15** was not applied during benchmark execution. The detector instead used its default threshold of **0.65**.

At a threshold of 0.65, only extreme anomalies were detected, producing approximately:

* Recall: **2.66%**
* F1-score: **4.84%**

After correcting the parameter-key mismatch, the calibrated validation threshold of **0.15** correctly reaches the detector.

The corrected test results are:

* **F1-score:** 49.92%
* **Precision:** 41.22%
* **Recall:** 63.28%
* **Accuracy:** 70.42%
* **False Positive Rate:** 27.41%
* **Balanced Accuracy:** 67.93%

The resulting test F1 closely matches the validation F1 of **49.93%**, without performing any test-set threshold tuning.

---

## 3. Algorithm Capability Matrix & Comparison Boundaries

Algorithms with fundamentally different capabilities should not be compared on metrics they do not implement.

| Algorithm              | Reference                   | Streaming Detection | Causal RCA | Preemptive Migration | Closed-Loop FT | Applicable Metrics                         |
| ---------------------- | --------------------------- | :-----------------: | :--------: | :------------------: | :------------: | ------------------------------------------ |
| **Proposed Causal FT** | Our Pipeline                |       **Yes**       |   **Yes**  |          No          |     **Yes**    | Detection, RCA, Computational, Reliability |
| **IPFT**               | Theodoropoulos et al., 2022 |       **Yes**       |     No     |        **Yes**       |     **Yes**    | Detection, Computational, Mitigation       |
| **BWOAIF**             | Hannák et al., 2023         |       **Yes**       |     No     |          No          |       No       | Detection, Computational                   |
| **RCD**                | Ikram et al., 2022          |          No         |   **Yes**  |          No          |       No       | RCA, Computational                         |
| **PreGAN**             | Tuli et al., 2022           |       **Yes**       |     No     |        **Yes**       |     **Yes**    | Detection, Computational, Mitigation       |

### Exclusions and N/A Annotations

1. **RCD excluded from detection plots:**
   RCD (Robust Causal Diagnosis, Ikram et al., 2022) performs offline root-cause diagnosis on known anomalous episodes and does not implement streaming binary anomaly detection. It is therefore marked `N/A` for detection metrics.

2. **BWOAIF and RCD excluded from recovery/mitigation plots:**
   Neither algorithm implements closed-loop recovery or migration actions. Corresponding metrics are marked `N/A`.

3. **Accuracy vs Balanced Accuracy:**
   The test stream contains approximately 23.3% attack observations. Consequently, ordinary accuracy can be misleading for degenerate classifiers.

   For example, IPFT achieves approximately 23.3% ordinary accuracy and 50.0% balanced accuracy when predicting the positive class for nearly all observations.

   Similarly, PreGAN on Seed 43 collapses toward the negative class, producing approximately 76.69% ordinary accuracy while its balanced accuracy remains near the 50% chance baseline.

---

## 4. Complete Inventory of Benchmark Plots

### A. Detection Comparison

* `01_detection_precision_recall_f1.png`
  Precision, Recall, and F1-score comparison using Mean ± Standard Deviation.

* `02_detection_roc_pr_auc.png`
  ROC-AUC and PR-AUC discrimination comparison.

* `03_detection_accuracy_balanced_accuracy.png`
  Ordinary Accuracy versus Balanced Accuracy.

* `04_detection_fpr_fnr_tradeoff.png`
  False Positive Rate versus False Negative Rate.

* `05_detection_confusion_matrices.png`
  Empirical confusion matrices for the four streaming detection approaches.

* `06_detection_roc_curves.png`
  Multi-algorithm ROC curves evaluated on the common test stream.

* `07_detection_precision_recall_curves.png`
  Multi-algorithm Precision-Recall curves.

### B. Detection Timing

* `08_timing_mttd_comparison.png`
  Episode-based Mean Time to Detect (MTTD) and Episode Detection Coverage.

* `09_timing_inference_latency_mean_p95.png`
  Mean and P95 single-observation processing latency in milliseconds per record.

### C. Causal Root-Cause Diagnosis

* `10_rca_localization_top_k_recall.png`
  Top-1, Top-3, and Top-5 Root-Cause Localization Recall comparing Causal FT and RCD.

* `11_rca_computation_latency.png`
  CPU execution latency of root-cause localization algorithms.

### D. Recovery and Resilience

* `12_recovery_success_rate.png`
  Percentage of attempted recovery actions completed successfully.

* `13_recovery_action_latency.png`
  Mitigation action dispatch-to-completion latency.

* `14_recovery_end_to_end_time.png`
  End-to-end recovery time from fault episode onset to service restoration.

* `15_recovery_early_warning_lead_time.png`
  Predictive lead time prior to fault onset for algorithms supporting preemptive mitigation.

* `16_recovery_service_availability.png`
  Service availability under continuous fault injection.

* `17_recovery_operational_cost.png`
  Normalized operational recovery cost score.

* `18_recovery_migration_bandwidth.png`
  Network bandwidth associated with preemptive container migration.

### E. Computational Overhead

* `19_overhead_throughput.png`
  Stream-processing throughput in records per second.

* `20_overhead_cpu_utilization.png`
  CPU utilization measured using `psutil`.

* `21_overhead_peak_ram_rss.png`
  Peak resident memory footprint in MB.

* `22_overhead_energy_consumption.png`
  Estimated total electrical energy consumption in Joules.

* `23_overhead_telemetry_bandwidth.png`
  Telemetry and control communication bandwidth in KB/s.

### F. Robustness & Stability

* `24_robustness_seed_f1_stability.png`
  Seed-wise F1-score stability across Seeds 42, 43, and 44.

* `25_robustness_seed_throughput_stability.png`
  Processing-throughput stability across random seeds.

* `26_robustness_causal_ft_persistence_sensitivity.png`
  Causal FT F1-score, Precision, Recall, and FPR across persistence values \(k \in [1,5]\).

---

## 5. Summary Tables

* **[algorithm_comparison_summary.csv](../../tables/algorithm_comparison_summary.csv)**
  Comprehensive multi-metric comparison across the evaluated algorithms.

* **[key_results_summary.csv](../../tables/key_results_summary.csv)**
  Concise summary of the most important experimental results.

* **[metric_definitions.csv](../../tables/metric_definitions.csv)**
  Metric definitions, units, interpretation, and optimization direction.

---

## 6. Reproducibility

The figures and summary tables are generated programmatically from benchmark outputs rather than being manually constructed.

Relevant scripts include:

* `scripts/run_full_benchmark.py`
* `scripts/run_corrected_benchmark.py`
* `scripts/diagnose_and_optimize_causal_ft.py`
* `scripts/generate_comparison_plots.py`
* `scripts/generate_result_deliverables.py`

This ensures that benchmark tables and visualizations can be regenerated consistently after future experimental updates.
