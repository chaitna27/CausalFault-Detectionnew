# Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems

This research project implements a **Causal Real-Time Adaptive Fault Tolerance** framework for Edge-IoT systems. The architecture combines real-time streaming anomaly detection (River & PyTorch Autoencoders), NetworkX system dependency DAGs, DoWhy causal effect estimation, and adaptive recovery mechanisms to autonomously detect system faults, identify their root causes, and execute optimal recovery operations.

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                         │
│                                                            │
│   SimPy Edge Simulation │ Edge-IIoTset │ TON_IoT │ N-BaIoT │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                  Data Source Adapters
                             │
                             ▼
                Unified TelemetryRecord Format
                             │
                             ▼
                    Streaming Pipeline
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
         River Detector            PyTorch Detector
      (Half-Space Trees)             (Autoencoder)
               │                           │
               └─────────────┬─────────────┘
                             ▼
                      Fault Detected
                             │
                             ▼
                    Causal Analysis Engine
                    (NetworkX + DoWhy)
                             │
                             ▼
                     Root Cause Identified
                             │
                             ▼
                     Adaptive Recovery
             (Workload Redistribution, Task Migration,
              Traffic Rerouting, Failover, Rebalancing)
                             │
                             ▼
                    System State Update
                             │
                             ▼
                  Evaluation & Plots Output
```

---

## Dataset Pipeline: Edge-IIoTset

The project includes an end-to-end memory-efficient preprocessing and streaming pipeline for the **Edge-IIoTset** dataset partitioned under `data/datasets/`:
- `data/datasets/train/` (16 CSV files, 813.93 MB, 1,553,434 raw rows)
- `data/datasets/validation/` (4 CSV files, 174.68 MB, 332,874 raw rows)
- `data/datasets/test/` (4 CSV files, 174.53 MB, 332,893 raw rows)
- **Total**: 24 CSV files, ~1.16 GB, 2,219,201 raw rows.

### Preprocessing & Feature Engineering Principles
The pipeline implemented in [`scripts/preprocess_dataset.py`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/scripts/preprocess_dataset.py) strictly adheres to the following requirements:

1. **Complete Missing/NaN Row Removal (Zero Imputation)**: Any row containing ANY missing, NaN, empty, or sentinel value in ANY original field is completely dropped.
2. **Complete Invalid & Infinite Value Removal**: Any row containing infinite values or corrupted timestamp shifts (e.g. IP addresses appearing in `frame.time`) is completely dropped.
3. **Post-Cleaning Deduplication**: Duplicate rows are removed after missing and invalid row filtering.
4. **Preservation of Valid Extreme Values**: Valid extreme values and traffic spikes are NOT clipped or removed, as they reflect genuine attacks and system faults.
5. **Untouched Raw Files**: The raw files in `data/datasets/` remain 100% read-only and unmodified.
6. **Split Preservation**: The existing 70/15/15 train, validation, and test splits are strictly preserved without reshuffling.
7. **Zero Future Lookahead**: Point-in-time timestamp features (`frame_hour`, `frame_minute`, `frame_second`, `sin_hour`, `cos_hour`, `day_of_week`) are derived per packet without lookahead. Unsound cross-row deltas across interleaved streams are avoided.
8. **Train-Only Preprocessing Fitting**: `StandardScaler` is fitted incrementally via `partial_fit` solely on training chunks. Validation and test splits are transformed without refitting.
9. **Target Isolation**: `Attack_label` and `Attack_type` are completely isolated and never used as input features.
10. **Pre-Execution 5,000-Row Sample Verification**: Automated verification ensuring identical columns, 0 NaNs, 0 infs, and compatible labels across train, val, and test before full dataset execution.

### Preprocessed Artifacts
Saved in `data/processed/artifacts/`:
- `scaler.joblib`: Trained `StandardScaler` across 62 engineered features.
- `feature_names.json`: Complete ordered feature list.
- `preprocessing_report.json`: Execution statistics, retained rows, and dropped row breakdown.

---

## Operational Commands

### 1. Preprocessing Edge-IIoTset Dataset

#### Run 5,000-Row Sample Validation Test
```bash
python scripts/preprocess_dataset.py --sample 5000
```

#### Run Full Dataset Preprocessing (~1.16 GB)
```bash
python scripts/preprocess_dataset.py --full
```

### 2. Running Dataset Streaming Evaluation

#### Stream Processed Edge-IIoTset (Default)
```bash
python main.py --mode dataset --dataset edge_iiotset --data-path data/processed/test --max-records 5000
```

#### Full Test Set Evaluation
```bash
python main.py --mode dataset --dataset edge_iiotset --data-path data/processed/test
```

### 3. Simulation & Robustness Modes

#### Single Simulation Mode
```bash
python main.py --mode simulation --seed 42 --persistence 3
```

#### Multi-Seed Robustness Experiment (10 Seeds: 42 to 51)
```bash
python main.py --mode experiment --seed-start 42 --num-seeds 10
```

#### Detection Persistence Sensitivity Analysis ($k = 1, 2, 3, 4, 5$)
```bash
python main.py --mode sensitivity
```

---

## Generated Results & Visualizations

Results are saved under `results/`:
- `results/data/dataset_detection_metrics.csv`: Dataset streaming evaluation metrics (Precision, Recall, F1, FPR, Processing Latency).
- `results/data/dataset_device_metrics.csv`: Per-device telemetry breakdown.
- `results/data/causal_results.csv`: Inferred root causes and causal treatment effects.
- `results/plots/dataset_precision_recall_f1.png`: Detection Precision, Recall, and F1-Score.
- `results/plots/dataset_detection_latency.png`: Real-time streaming processing latency distribution.
- `results/plots/device_comparison.png`: Detection performance across IoT device types.
- `results/plots/attack_type_performance.png`: Anomaly trigger rate per attack category.
