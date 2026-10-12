# CausalFault-Detection Evaluation Visualizations: Technical Summary
This document provides a comprehensive summary of all empirical evaluation visualizations, metrics validation, source data artifacts, and methodological notes for the **CausalFault-Detection** project.
---
## 1. Source Files Used
The visualizations were generated directly from the following verified evaluation data files:
- [`results/tables/detection_confusion_metrics_by_seed.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/detection_confusion_metrics_by_seed.csv)
- [`results/tables/detection_metrics_mean_std.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/detection_metrics_mean_std.csv)
- [`results/tables/causal_ft_validation_experiments.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/causal_ft_validation_experiments.csv)
- [`results/tables/threshold_calibration.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/threshold_calibration.csv)
- [`results/tables/algorithm_comparison_summary.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/algorithm_comparison_summary.csv)
- [`results/tables/benchmark_summary_mean_std.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/tables/benchmark_summary_mean_std.csv)
- [`results/raw/full_benchmark_results.json`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/raw/full_benchmark_results.json)
- [`results/raw/rca_evaluation.json`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/raw/rca_evaluation.json)
- [`results/raw/scalability_results.json`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/raw/scalability_results.json)
- [`results/data/craft_edge_detection_metrics.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/data/craft_edge_detection_metrics.csv)
- [`results/data/craft_edge_recovery_outcomes.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/data/craft_edge_recovery_outcomes.csv)
- [`results/data/dataset_device_metrics.csv`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/results/data/dataset_device_metrics.csv)
- [`data/processed/validation/validation_01_processed.csv (25,000 records)`](file:///C:/Users/Dell/Desktop/CausalFault-Detection/data/processed/validation/validation_01_processed.csv (25,000 records))

## 2. Evaluated Configurations Found
### Evaluated Detection Algorithms:
1. **Proposed Causal FT (Ours)**: Streaming composite sigmoid detector (z-score + EWMA) calibrated at threshold **τ = 0.15**, persistence $k = 1$, weights $w_{\text{norm}} = 0.7, w_{\text{ewma}} = 0.3$.
2. **CRAFT-Edge (Ours, Adaptive Efficacy)**: Evaluated at default threshold **τ = 0.65** with closed-loop adaptive efficacy recovery.
3. **IPFT (Theodoropoulos et al., 2022)**: Evaluated at calibrated threshold **τ = 0.01** with container migration.
4. **BWOAIF (Hannák et al., 2023)**: Evaluated at calibrated threshold **τ = 0.49**.
5. **PreGAN (Tuli et al., 2022)**: Evaluated at calibrated threshold **τ = 0.01** with container migration.
6. **RCD (Ikram et al., 2022)**: Offline root-cause diagnosis baseline.

### Threshold Sweep Configurations (Proposed Causal FT):
- **τ = 0.15**: Optimal calibrated operating threshold (Validation F1 = 49.94%, Test F1 = 49.92%, Prec = 41.22%, Rec = 63.28%, Acc = 70.42%, FPR = 27.41%).
- **τ = 0.30**: Intermediate candidate threshold (Validation F1 = 6.11%, Prec = 27.98%, Rec = 3.43%, Acc = 76.01%, FPR = 2.60%).
- **τ = 0.50**: Standard symmetric decision boundary (Validation F1 = 4.27%, Prec = 24.77%, Rec = 2.34%, Acc = 76.15%, FPR = 2.09%).
- **τ = 0.65**: Original default threshold before calibration (Validation F1 = 4.19%, Prec = 25.34%, Rec = 2.28%, Acc = 76.22%, FPR = 1.98%).

## 3. Metrics Available and Unavailable
### Available Metrics (Fully Supported and Plotted):
- **Classification Metrics**: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR), Specificity, Negative Predictive Value (NPV), ROC-AUC, PR-AUC.
- **Confusion Matrix Counts**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
- **Fault Recovery Metrics**: Total recovery episodes (24), Successful recoveries (21), Failed/rejected recoveries (3), Recovery success rate (87.5%), Mean recovery reward (0.2617), Mean estimated benefit (0.3570), Mean operational cost (0.3563), Preemptive migration bandwidth (64,152 KB/s for IPFT; 67,343 KB/s for PreGAN), Observed service availability (91.4% Causal FT; 100% IPFT; 92.2% PreGAN).
- **Root-Cause Analysis (RCA)**: Top-1, Top-3, and Top-5 localization recall, direct computation latency (ms).
- **System & Computational Overheads**: Processing throughput (rec/s), mean and P95 per-record inference latency (ms), peak resident memory footprint (RSS in MB), single-core CPU utilization (%), estimated energy consumption (Joules).
- **Multi-Seed Stability**: Distributions across random seeds 42, 43, and 44.

### Unavailable / Not Applicable Metrics (Explicitly Documented Without Fabrication):
- **RCD Detection Metrics**: Marked `N/A`. RCD is an offline root-cause diagnosis model that operates on known anomalous windows; it does not perform online binary anomaly detection.
- **BWOAIF & RCD Recovery Metrics**: Marked `N/A`. Neither algorithm implements closed-loop task migration or adaptive recovery.
- **Recovery Evaluation at τ = 0.30 and τ = 0.50**: Marked `Unmeasured`. Closed-loop simulation benchmarks were executed for τ = 0.15 and τ = 0.65; no simulation runs were conducted for 0.30 or 0.50.

## 4. Generated Figures Inventory
Every major figure has been generated in two publication formats: high-resolution raster (`.png`, 300 DPI) and scalable vector (`.pdf`).

| Figure Name | Format | Primary Focus & Demonstration |
| :--- | :---: | :--- |
| [accuracy_by_threshold.png](accuracy_by_threshold.png) | PNG, PDF | Classification accuracy versus threshold across τ = 0.15, 0.30, 0.50, and 0.65. |
| [classification_metrics_comparison.png](classification_metrics_comparison.png) | PNG, PDF | Bar chart comparing 8 classification metrics (Acc, Bal Acc, Prec, Rec, F1, Spec, FPR, NPV) across all detection approaches. |
| [confusion_counts_comparison.png](confusion_counts_comparison.png) | PNG, PDF | Grouped bar chart comparing absolute counts of TP, TN, FP, and FN across evaluated algorithms. |
| [confusion_matrix_all_algorithms.png](confusion_matrix_all_algorithms.png) | PNG, PDF | 2x2 grid of confusion matrices for Proposed Causal FT, IPFT, BWOAIF, and PreGAN on test stream. |
| [confusion_matrix_threshold_015.png](confusion_matrix_threshold_015.png) | PNG, PDF | Labeled confusion matrix heatmap for optimal validation-calibrated operating threshold τ = 0.15. |
| [confusion_matrix_threshold_030.png](confusion_matrix_threshold_030.png) | PNG, PDF | Labeled confusion matrix heatmap for intermediate sweep threshold τ = 0.30. |
| [confusion_matrix_threshold_050.png](confusion_matrix_threshold_050.png) | PNG, PDF | Labeled confusion matrix heatmap for standard decision threshold τ = 0.50. |
| [confusion_matrix_threshold_065.png](confusion_matrix_threshold_065.png) | PNG, PDF | Labeled confusion matrix heatmap for uncalibrated default threshold τ = 0.65 (raw counts and normalized percentages). |
| [f1_score_by_threshold.png](f1_score_by_threshold.png) | PNG, PDF | F1-score versus threshold demonstrating peak F1 at calibrated τ = 0.15. |
| [f1_score_comparison.png](f1_score_comparison.png) | PNG, PDF | Bar comparison of Detection F1-Score across algorithms with sample standard deviation error bars. |
| [false_positive_rate_by_threshold.png](false_positive_rate_by_threshold.png) | PNG, PDF | False positive rate versus threshold demonstrating conservative behavior at higher thresholds. |
| [metrics_correlation_heatmap.png](metrics_correlation_heatmap.png) | PNG, PDF | Correlation heatmap among all key numerical classification metrics. |
| [per_device_detection_performance.png](per_device_detection_performance.png) | PNG, PDF | Per-device telemetry volume and attack breakdown across monitored IoT network endpoints. |
| [precision_by_threshold.png](precision_by_threshold.png) | PNG, PDF | Precision score versus threshold across τ = 0.15, 0.30, 0.50, and 0.65. |
| [precision_recall_comparison.png](precision_recall_comparison.png) | PNG, PDF | Scatter plot of Precision vs Recall across algorithms and threshold points with dotted iso-F1 curves. |
| [precision_recall_curve.png](precision_recall_curve.png) | PNG, PDF | Empirical continuous Precision-Recall curve calculated directly from validation continuous scores. |
| [rca_localization_and_latency.png](rca_localization_and_latency.png) | PNG, PDF | Root-cause localization Top-k recall and diagnosis latency comparing Causal FT against RCD. |
| [recall_by_threshold.png](recall_by_threshold.png) | PNG, PDF | Recall / sensitivity score versus threshold across τ = 0.15, 0.30, 0.50, and 0.65. |
| [recovery_comparison.png](recovery_comparison.png) | PNG, PDF | Breakdown of total recovery episodes, confirmed successes, and rejected actions for CRAFT-Edge. |
| [recovery_cost_and_availability.png](recovery_cost_and_availability.png) | PNG, PDF | Dual panel showing Normalized Operational Recovery Cost and Observed Service Availability. |
| [recovery_success_rate.png](recovery_success_rate.png) | PNG, PDF | Recovery mitigation success rate comparison across closed-loop capable algorithms. |
| [resource_utilization_energy.png](resource_utilization_energy.png) | PNG, PDF | Dual panel showing Resident Set Size (RSS memory in MB) and Estimated Energy Consumption (Joules). |
| [seed_stability_metrics.png](seed_stability_metrics.png) | PNG, PDF | Seed-by-seed stability analysis exposing PreGAN's collapse on Seed 43 versus Causal FT consistency. |
| [successful_recoveries_by_threshold.png](successful_recoveries_by_threshold.png) | PNG, PDF | Successful recovery count across thresholds, explicitly delineating measured vs unmeasured points. |
| [threshold_tradeoff_all_metrics.png](threshold_tradeoff_all_metrics.png) | PNG, PDF | Multi-line tradeoff chart showing simultaneous variation of Precision, Recall, F1, Accuracy, Bal Acc, and FPR. |
| [throughput_latency_tradeoff.png](throughput_latency_tradeoff.png) | PNG, PDF | Dual panel evaluating stream processing throughput (rec/s) and inference latency (ms/rec). |

## 5. Metric Definitions & Zero-Division Handling
The metrics follow canonical statistical formulations:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$
$$\text{Specificity} = \frac{TN}{TN + FP}$$
$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$$
$$\text{Negative Predictive Value (NPV)} = \frac{TN}{TN + FN}$$

**Zero-Division Convention**: When any denominator is 0 (e.g. $TP+FP = 0$ or $Precision+Recall = 0$), the metric evaluates strictly to `0.0`.

## 6. Discrepancy & Verification Analysis
All stored confusion metrics in `detection_confusion_metrics_by_seed.csv` were cross-checked against independent recalculations from raw `(TP, TN, FP, FN)` counts. **100% of stored values matched within 0.05% tolerance with zero discrepancies.**

## 7. Important Limitations
1. **Closed-Loop Simulation Scope**: Full end-to-end discrete task recovery injection is implemented in SimPy edge simulation (`simulation/`), whereas large-scale streaming benchmarks (`results/tables/`) evaluate streaming detection, causal RCA, and proxy mitigation.
2. **Hardware Shadow Proxies**: Benefit and reward measurements in `craft_edge_detection_metrics.csv` are synthetic shadow-state proxies rather than physical rack readings, as documented in the metadata.
3. **Threshold Availability**: Closed-loop recovery was measured at τ = 0.65 (CRAFT-Edge) and τ = 0.15 (Causal FT); candidate thresholds 0.30 and 0.50 were evaluated for detection performance on validation data, but not subjected to separate closed-loop recovery trials.

## 8. Exact Command to Regenerate All Visualizations
To regenerate the complete set of figures and summary:
```bash
python scripts/plot_evaluation_results.py
```
