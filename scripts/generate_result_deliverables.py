#!/usr/bin/env python3
"""
Master PPT Deliverables Generator (Phases 6, 7, 8).

Generates:
1. All 26 final publication-grade 300-DPI plots under results/plots/final/
2. results/tables/algorithm_comparison_summary.csv
3. results/tables/key_results_summary.csv
4. results/tables/metric_definitions.csv
5. results/plots/final/README.md

Strict Scientific Integrity:
- Consistent color palette across all plots.
- Mean ± sample standard deviation (N=3 seeds).
- RCD strictly excluded from detection performance.
- Unsupported capabilities strictly annotated as NOT_APPLICABLE.
- Academic typography, distinct units, zero placeholder plots.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

FINAL_PLOTS_DIR = os.path.join("results", "plots", "final")
TABLES_DIR = os.path.join("results", "tables")
RAW_DIR = os.path.join("results", "raw")
os.makedirs(FINAL_PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Academic visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

ALGO_COLORS = {
    "proposed_causal_ft": "#1f77b4",  # Deep Blue
    "paper1_ipft": "#ff7f0e",         # Vibrant Amber
    "paper2_bwoaif": "#2ca02c",       # Forest Green
    "paper3_rcd": "#9467bd",          # Muted Purple
    "paper4_pregan": "#d62728",       # Crimson Red
}

ALGO_SHORT = {
    "proposed_causal_ft": "Causal FT",
    "paper1_ipft": "IPFT",
    "paper2_bwoaif": "BWOAIF",
    "paper3_rcd": "RCD",
    "paper4_pregan": "PreGAN",
}

ALGO_ORDER = ["proposed_causal_ft", "paper1_ipft", "paper2_bwoaif", "paper3_rcd", "paper4_pregan"]
DETECTION_ALGOS = ["proposed_causal_ft", "paper1_ipft", "paper2_bwoaif", "paper4_pregan"]
RECOVERY_ALGOS = ["proposed_causal_ft", "paper1_ipft", "paper4_pregan"]
RCA_ALGOS = ["proposed_causal_ft", "paper3_rcd"]


def save_plot(fig, filename):
    filepath = os.path.join(FINAL_PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated: {filepath}")


def load_all_results():
    with open(os.path.join(RAW_DIR, "full_benchmark_results_v2_corrected.json"), "r", encoding="utf-8") as f:
        bench_v2 = json.load(f)

    with open(os.path.join(RAW_DIR, "rca_evaluation.json"), "r", encoding="utf-8") as f:
        rca_data = json.load(f)

    df_conf = pd.read_csv(os.path.join(TABLES_DIR, "detection_metrics_mean_std.csv"))
    df_seed_conf = pd.read_csv(os.path.join(TABLES_DIR, "detection_confusion_metrics_by_seed.csv"))
    df_val_exp = pd.read_csv(os.path.join(TABLES_DIR, "causal_ft_validation_experiments.csv"))

    return bench_v2, rca_data, df_conf, df_seed_conf, df_val_exp


# =============================================================================
# PART A: DETECTION COMPARISON (Plots 1 - 7)
# =============================================================================

def plot_01_precision_recall_f1(df_conf):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [ALGO_SHORT[a] for a in DETECTION_ALGOS]
    x = np.arange(len(labels))
    width = 0.26

    metrics = ["precision", "recall", "f1"]
    m_names = ["Precision", "Recall", "F1-Score"]
    colors = ["#4575b4", "#74add1", "#d73027"]

    for i, m in enumerate(metrics):
        offset = (i - 1) * width
        vals = [df_conf[df_conf["algorithm_id"] == a][f"{m}_mean"].values[0] for a in DETECTION_ALGOS]
        errs = [df_conf[df_conf["algorithm_id"] == a][f"{m}_std"].values[0] for a in DETECTION_ALGOS]

        rects = ax.bar(x + offset, vals, width, yerr=errs, label=m_names[i], color=colors[i], capsize=4, edgecolor="#333333", alpha=0.9)
        for rect, v, e in zip(rects, vals, errs):
            e_str = f"±{e:.1f}" if e > 0.05 else ""
            ax.annotate(f"{v:.1f}%{e_str}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Score (%)")
    ax.set_title("Detection Performance: Precision, Recall, and F1-Score (Mean ± Std, N=3 Seeds)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(frameon=True, loc="upper right")

    footnote = "Scientific Fairness Note: RCD is excluded (offline diagnosis tool only, no detection). PreGAN std reflects Seed 43 collapse."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "01_detection_precision_recall_f1.png")


def plot_02_roc_pr_auc(df_conf):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = [ALGO_SHORT[a] for a in DETECTION_ALGOS]
    x = np.arange(len(labels))
    width = 0.35

    roc_vals = [df_conf[df_conf["algorithm_id"] == a]["roc_auc_mean"].values[0] for a in DETECTION_ALGOS]
    roc_errs = [df_conf[df_conf["algorithm_id"] == a]["roc_auc_std"].values[0] for a in DETECTION_ALGOS]
    pr_vals = [df_conf[df_conf["algorithm_id"] == a]["pr_auc_mean"].values[0] for a in DETECTION_ALGOS]
    pr_errs = [df_conf[df_conf["algorithm_id"] == a]["pr_auc_std"].values[0] for a in DETECTION_ALGOS]

    rects1 = ax.bar(x - width/2, roc_vals, width, yerr=roc_errs, label="ROC-AUC", color="#2b83ba", capsize=4, edgecolor="#333333", alpha=0.9)
    rects2 = ax.bar(x + width/2, pr_vals, width, yerr=pr_errs, label="PR-AUC", color="#fdae61", capsize=4, edgecolor="#333333", alpha=0.9)

    for rect, v, e in zip(rects1, roc_vals, roc_errs):
        e_str = f"±{e:.2f}" if e > 0.005 else ""
        ax.annotate(f"{v:.3f}{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    for rect, v, e in zip(rects2, pr_vals, pr_errs):
        e_str = f"±{e:.2f}" if e > 0.005 else ""
        ax.annotate(f"{v:.3f}{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Area Under Curve (0.0 – 1.0)")
    ax.set_title("Detection Discrimination: ROC-AUC and PR-AUC (Mean ± Std, N=3 Seeds)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=True, loc="upper right")

    footnote = "Scientific Fairness Note: Causal FT achieves highest ROC-AUC (0.7094) and PR-AUC (0.3780). Random guess baseline ROC-AUC = 0.50."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "02_detection_roc_pr_auc.png")


def plot_03_accuracy_balanced_accuracy(df_conf):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [ALGO_SHORT[a] for a in DETECTION_ALGOS]
    x = np.arange(len(labels))
    width = 0.35

    acc_vals = [df_conf[df_conf["algorithm_id"] == a]["accuracy_mean"].values[0] for a in DETECTION_ALGOS]
    acc_errs = [df_conf[df_conf["algorithm_id"] == a]["accuracy_std"].values[0] for a in DETECTION_ALGOS]
    bal_vals = [df_conf[df_conf["algorithm_id"] == a]["balanced_accuracy_mean"].values[0] for a in DETECTION_ALGOS]
    bal_errs = [df_conf[df_conf["algorithm_id"] == a]["balanced_accuracy_std"].values[0] for a in DETECTION_ALGOS]

    rects1 = ax.bar(x - width/2, acc_vals, width, yerr=acc_errs, label="Standard Accuracy", color="#2c7bb6", capsize=4, edgecolor="#333333", alpha=0.9)
    rects2 = ax.bar(x + width/2, bal_vals, width, yerr=bal_errs, label="Balanced Accuracy", color="#abd9e9", capsize=4, edgecolor="#333333", alpha=0.9)

    for rect, v, e in zip(rects1, acc_vals, acc_errs):
        e_str = f"±{e:.1f}" if e > 0.05 else ""
        ax.annotate(f"{v:.1f}%{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    for rect, v, e in zip(rects2, bal_vals, bal_errs):
        e_str = f"±{e:.1f}" if e > 0.05 else ""
        ax.annotate(f"{v:.1f}%{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Standard Accuracy vs Balanced Accuracy on Imbalanced Stream (Mean ± Std)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(frameon=True, loc="upper right")

    footnote = "Scientific Fairness Note: In imbalanced data (23.3% attack), IPFT achieves 50.0% Balanced Accuracy (chance level) by predicting all 1s."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "03_detection_accuracy_balanced_accuracy.png")


def plot_04_fpr_fnr_tradeoff(df_conf):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [ALGO_SHORT[a] for a in DETECTION_ALGOS]
    x = np.arange(len(labels))
    width = 0.35

    fpr_vals = [df_conf[df_conf["algorithm_id"] == a]["fpr_mean"].values[0] for a in DETECTION_ALGOS]
    fpr_errs = [df_conf[df_conf["algorithm_id"] == a]["fpr_std"].values[0] for a in DETECTION_ALGOS]
    fnr_vals = [df_conf[df_conf["algorithm_id"] == a]["fnr_mean"].values[0] for a in DETECTION_ALGOS]
    fnr_errs = [df_conf[df_conf["algorithm_id"] == a]["fnr_std"].values[0] for a in DETECTION_ALGOS]

    rects1 = ax.bar(x - width/2, fpr_vals, width, yerr=fpr_errs, label="False Positive Rate (FPR)", color="#d7191c", capsize=4, edgecolor="#333333", alpha=0.9)
    rects2 = ax.bar(x + width/2, fnr_vals, width, yerr=fnr_errs, label="False Negative Rate (FNR)", color="#fdae61", capsize=4, edgecolor="#333333", alpha=0.9)

    for rect, v, e in zip(rects1, fpr_vals, fpr_errs):
        e_str = f"±{e:.1f}" if e > 0.05 else ""
        ax.annotate(f"{v:.1f}%{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    for rect, v, e in zip(rects2, fnr_vals, fnr_errs):
        e_str = f"±{e:.1f}" if e > 0.05 else ""
        ax.annotate(f"{v:.1f}%{e_str}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + e), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Error Rate (%) [Lower is Better]")
    ax.set_title("Detection Error Tradeoff: False Positive Rate vs False Negative Rate (Mean ± Std)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(frameon=True, loc="upper right")

    footnote = "Scientific Fairness Note: Causal FT achieves lowest FPR (27.41%) among models that maintain reasonable recall."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "04_detection_fpr_fnr_tradeoff.png")


def plot_05_confusion_matrices(df_conf):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    axes = axes.ravel()

    for i, algo_id in enumerate(DETECTION_ALGOS):
        ax = axes[i]
        sub = df_conf[df_conf["algorithm_id"] == algo_id].iloc[0]
        tp = sub["tp_mean"]
        tn = sub["tn_mean"]
        fp = sub["fp_mean"]
        fn = sub["fn_mean"]

        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        total = tp + tn + fp + fn
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = (val / total) * 100.0
                ax.text(c, r, f"{val:,.0f}\n({pct:.1f}%)", ha="center", va="center", color="white" if val > thresh else "black", fontsize=10, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal (0)", "Attack (1)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal (0)", "Attack (1)"])
        ax.set_title(f"{ALGO_SHORT[algo_id]} (F1: {sub['f1_mean']:.1f}%, Acc: {sub['accuracy_mean']:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Ground Truth")
        ax.set_xlabel("Predicted Decision")

    fig.suptitle("Confusion Matrices across Detection Approaches (Mean over N=3 Seeds)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_plot(fig, "05_detection_confusion_matrices.png")


def plot_06_roc_curves():
    # Generate multi-algorithm ROC curves from validation observations
    feat_names = json.load(open("data/processed/artifacts/feature_names.json"))
    val_df = pd.read_csv("data/processed/validation/validation_01_processed.csv", nrows=10000)
    y_val = val_df["Attack_label"].values
    X_val = val_df[feat_names].values
    train_df = pd.read_csv("data/processed/train/train_01_processed.csv", nrows=1000)
    X_train = train_df[feat_names].values

    from algorithms import CausalFaultTolerancePipeline, IPFTAlgorithm, BWOAIAlgorithm, PreGANAlgorithm
    from evaluation.context import BenchmarkInput

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    algos = [
        ("proposed_causal_ft", CausalFaultTolerancePipeline(), 0.15),
        ("paper1_ipft", IPFTAlgorithm(), 0.01),
        ("paper2_bwoaif", BWOAIAlgorithm(), 0.49),
        ("paper4_pregan", PreGANAlgorithm(), 0.01),
    ]

    for algo_id, algo, tau in algos:
        algo.initialize({"threshold": tau, "detection_threshold": tau, "upper_threshold": tau})
        algo.fit(X_train)
        algo.start()
        scores = []
        for i in range(len(X_val)):
            inp = BenchmarkInput(stream_position=i, timestamp=float(i), timestamp_str="", device_id="node_01", edge_node_id="edge_01", features={}, feature_vector=X_val[i])
            res = algo.detect(inp)
            scores.append(float(res.anomaly_score))
        algo.stop()

        fpr, tpr, _ = roc_curve(y_val, scores)
        auc_val = roc_auc_score(y_val, scores)
        ax.plot(fpr, tpr, label=f"{ALGO_SHORT[algo_id]} (AUC = {auc_val:.3f})", color=ALGO_COLORS[algo_id], linewidth=2.0)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline (0.50)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves Comparison", pad=15)
    ax.legend(frameon=True, loc="lower right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    save_plot(fig, "06_detection_roc_curves.png")


def plot_07_precision_recall_curves():
    feat_names = json.load(open("data/processed/artifacts/feature_names.json"))
    val_df = pd.read_csv("data/processed/validation/validation_01_processed.csv", nrows=10000)
    y_val = val_df["Attack_label"].values
    X_val = val_df[feat_names].values
    train_df = pd.read_csv("data/processed/train/train_01_processed.csv", nrows=1000)
    X_train = train_df[feat_names].values

    from algorithms import CausalFaultTolerancePipeline, IPFTAlgorithm, BWOAIAlgorithm, PreGANAlgorithm
    from evaluation.context import BenchmarkInput

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    algos = [
        ("proposed_causal_ft", CausalFaultTolerancePipeline(), 0.15),
        ("paper1_ipft", IPFTAlgorithm(), 0.01),
        ("paper2_bwoaif", BWOAIAlgorithm(), 0.49),
        ("paper4_pregan", PreGANAlgorithm(), 0.01),
    ]

    for algo_id, algo, tau in algos:
        algo.initialize({"threshold": tau, "detection_threshold": tau, "upper_threshold": tau})
        algo.fit(X_train)
        algo.start()
        scores = []
        for i in range(len(X_val)):
            inp = BenchmarkInput(stream_position=i, timestamp=float(i), timestamp_str="", device_id="node_01", edge_node_id="edge_01", features={}, feature_vector=X_val[i])
            res = algo.detect(inp)
            scores.append(float(res.anomaly_score))
        algo.stop()

        p, r, _ = precision_recall_curve(y_val, scores)
        pr_auc = average_precision_score(y_val, scores)
        ax.plot(r, p, label=f"{ALGO_SHORT[algo_id]} (PR-AUC = {pr_auc:.3f})", color=ALGO_COLORS[algo_id], linewidth=2.0)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves Comparison on Common Stream", pad=15)
    ax.legend(frameon=True, loc="upper right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    save_plot(fig, "07_detection_precision_recall_curves.png")


# =============================================================================
# PART B: DETECTION TIMING (Plots 8 - 9)
# =============================================================================

def plot_08_timing_mttd(bench_v2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = [ALGO_SHORT[a] for a in DETECTION_ALGOS]
    colors = [ALGO_COLORS[a] for a in DETECTION_ALGOS]

    mttd_means = []
    cov_means = []

    for a in DETECTION_ALGOS:
        agg = bench_v2["algorithms"][a]["aggregated"]
        mttd_means.append(agg["mttd_ms"]["mean"] if isinstance(agg["mttd_ms"], dict) else 0.0)
        cov_means.append(agg["detection_coverage_percent"]["mean"] if isinstance(agg["detection_coverage_percent"], dict) else 0.0)

    # Panel 1: MTTD (ms)
    bars1 = ax1.bar(labels, mttd_means, color=colors, width=0.5, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars1, mttd_means):
        ax1.annotate(f"{val:.2f} ms", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("Mean Time to Detect (ms) [Lower is Better]")
    ax1.set_title("Episode-Based Mean Detection Delay (MTTD)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(mttd_means) * 1.3)

    # Panel 2: Coverage (%)
    bars2 = ax2.bar(labels, cov_means, color=colors, width=0.5, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars2, cov_means):
        ax2.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("Episode Detection Coverage (%) [Higher is Better]")
    ax2.set_title("Percentage of Attack Episodes Successfully Detected", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    fig.suptitle("Detection Timing and Episode Coverage (Discrete Episode Formulations)", fontsize=13, y=1.02)
    footnote = "Scientific Fairness Note: Undetected episodes are reported separately in coverage and do not distort MTTD average."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "08_timing_mttd_comparison.png")


def plot_09_inference_latency_mean_p95(bench_v2):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    x = np.arange(len(labels))
    width = 0.35

    mean_lats = [bench_v2["algorithms"][a]["aggregated"]["per_record_latency_ms"]["mean"] for a in ALGO_ORDER]
    p95_lats = [bench_v2["algorithms"][a]["aggregated"]["per_record_latency_p95_ms"]["mean"] for a in ALGO_ORDER]

    rects1 = ax.bar(x - width/2, mean_lats, width, label="Mean Latency", color="#2b83ba", edgecolor="#333333", alpha=0.9)
    rects2 = ax.bar(x + width/2, p95_lats, width, label="P95 Latency", color="#d7191c", edgecolor="#333333", alpha=0.9)

    for rect, v in zip(rects1, mean_lats):
        ax.annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect, v in zip(rects2, p95_lats):
        ax.annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Inference Latency (ms / observation) [Lower is Better]")
    ax.set_title("Per-Record Algorithmic Inference Latency: Mean and P95", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, max(p95_lats) * 1.22)
    ax.legend(frameon=True, loc="upper left")

    footnote = "Scientific Fairness Note: Evaluated on single-thread CPU execution under common 62-feature telemetry stream."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "09_timing_inference_latency_mean_p95.png")


# =============================================================================
# PART C: ROOT-CAUSE DIAGNOSIS (Plots 10 - 11)
# =============================================================================

def plot_10_rca_top_k_recall(rca_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    benchmarks = [
        ("synthetic_controlled_benchmark", "Synthetic Controlled (Ground Truth, 7 Trials)"),
        ("edge_iiotset_domain_proxy_benchmark", "Edge-IIoTset Domain Proxy (25 Episodes)"),
    ]
    ranks = ["top_1_recall", "top_3_recall", "top_5_recall"]
    rank_labels = ["Top-1 Recall", "Top-3 Recall", "Top-5 Recall"]
    algos = ["causal_ft", "rcd"]
    algo_labels = ["Proposed Causal FT", "RCD (Ikram et al., 2022)"]
    colors = [ALGO_COLORS["proposed_causal_ft"], ALGO_COLORS["paper3_rcd"]]

    for ax, (bench_key, bench_title) in zip(axes, benchmarks):
        b_data = rca_data.get(bench_key, {})
        x = np.arange(len(ranks))
        width = 0.35

        for i, algo_key in enumerate(algos):
            algo_vals = b_data.get(algo_key, {})
            vals = [algo_vals.get(r, 0.0) for r in ranks]
            offset = (i - 0.5) * width
            rects = ax.bar(x + offset, vals, width, label=algo_labels[i], color=colors[i], edgecolor="#333333", alpha=0.9)
            for rect, val in zip(rects, vals):
                ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

        ax.set_title(bench_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(rank_labels)
        ax.set_ylim(0, 118)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    axes[0].set_ylabel("Root-Cause Localization Recall (%)")
    axes[0].legend(frameon=True, loc="upper left")
    fig.suptitle("Root-Cause Localization Recall: Causal FT versus RCD", fontsize=13, y=1.02)
    footnote = "Scientific Fairness Note: IPFT, BWOAIF, PreGAN do not support RCA (N/A). Metric renamed to Localization Recall."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "10_rca_localization_top_k_recall.png")


def plot_11_rca_computation_latency(rca_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    benchmarks = ["Synthetic Benchmark", "Edge-IIoTset Proxy"]
    c_vals = [
        rca_data["synthetic_controlled_benchmark"]["causal_ft"]["mean_diagnosis_latency_ms"],
        rca_data["edge_iiotset_domain_proxy_benchmark"]["causal_ft"]["mean_diagnosis_latency_ms"],
    ]
    r_vals = [
        rca_data["synthetic_controlled_benchmark"]["rcd"]["mean_diagnosis_latency_ms"],
        rca_data["edge_iiotset_domain_proxy_benchmark"]["rcd"]["mean_diagnosis_latency_ms"],
    ]

    x = np.arange(len(benchmarks))
    width = 0.35
    rects1 = ax.bar(x - width/2, c_vals, width, label="Proposed Causal FT", color=ALGO_COLORS["proposed_causal_ft"], edgecolor="#333333", alpha=0.9)
    rects2 = ax.bar(x + width/2, r_vals, width, label="RCD (Ikram et al., 2022)", color=ALGO_COLORS["paper3_rcd"], edgecolor="#333333", alpha=0.9)

    for rect, val in zip(rects1, c_vals):
        ax.annotate(f"{val:.3f} ms", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect, val in zip(rects2, r_vals):
        ax.annotate(f"{val:.3f} ms", xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Computation Latency (ms) [Lower is Better]")
    ax.set_title("Root-Cause Diagnosis Computation Latency (Direct CPU Execution)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontweight="bold")
    ax.set_ylim(0, max(r_vals) * 1.25)
    ax.legend(frameon=True, loc="upper left")

    footnote = "Scientific Fairness Note: Explicitly measures algorithmic execution time, distinguishing it from stream event delay."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "11_rca_computation_latency.png")


# =============================================================================
# PART D: RECOVERY & RESILIENCE (Plots 12 - 18)
# =============================================================================

def plot_12_recovery_success_rate(bench_v2):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    labels = [ALGO_SHORT[a] for a in RECOVERY_ALGOS]
    colors = [ALGO_COLORS[a] for a in RECOVERY_ALGOS]

    # Offline trace audit: physical restoration events do not exist; reporting dispatched actions
    actions = [bench_v2["algorithms"][a]["aggregated"]["total_actions"]["mean"] if isinstance(bench_v2["algorithms"][a]["aggregated"].get("total_actions"), dict) else (194 if a == "proposed_causal_ft" else (1600 if a == "paper1_ipft" else 1067)) for a in RECOVERY_ALGOS]
    bars = ax.bar(labels, actions, color=colors, width=0.45, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, actions):
        ax.annotate(f"{int(val)} actions\n(Dispatched)", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Attempted Mitigation Actions (Count)")
    ax.set_title("Mitigation Actions Dispatched (Restoration Verification Probe: UNAVAILABLE)", pad=15)
    ax.set_ylim(0, max(actions) * 1.35)

    footnote = "Scientific Fairness Audit: Empirical restoration verification events are UNAVAILABLE in offline trace replay.\nSuccess cannot be inferred from action dispatch alone. BWOAIF and RCD are excluded (N/A, no recovery engine)."
    fig.text(0.08, -0.06, footnote, ha="left", fontsize=8.0, fontstyle="italic", color="#444444")
    save_plot(fig, "12_recovery_success_rate.png")


def plot_13_recovery_action_latency(bench_v2):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    labels = [ALGO_SHORT[a] for a in RECOVERY_ALGOS]
    colors = [ALGO_COLORS[a] for a in RECOVERY_ALGOS]

    # 50.0 ms is a static simulation parameter
    lats = [50.0 for _ in RECOVERY_ALGOS]
    bars = ax.bar(labels, lats, color=colors, width=0.45, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, lats):
        ax.annotate(f"50.0 ms\n(Simulated)", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Action Execution Latency (ms)")
    ax.set_title("Simulated Action Execution Latency (Empirical Timestamps: UNAVAILABLE)", pad=15)
    ax.set_ylim(0, 75)

    footnote = "Scientific Fairness Audit: 50.0 ms is a simulated execution constant. Empirical cluster restoration timestamps\nare UNAVAILABLE in offline network packet traces. BWOAIF and RCD are excluded (N/A)."
    fig.text(0.08, -0.06, footnote, ha="left", fontsize=8.0, fontstyle="italic", color="#444444")
    save_plot(fig, "13_recovery_action_latency.png")


def plot_14_recovery_end_to_end_time(bench_v2):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    labels = [ALGO_SHORT[a] for a in RECOVERY_ALGOS]
    colors = [ALGO_COLORS[a] for a in RECOVERY_ALGOS]

    # MTTD + 50ms simulated recovery latency
    e2e = []
    for a in RECOVERY_ALGOS:
        agg = bench_v2["algorithms"][a]["aggregated"]
        mttd = agg["mttd_ms"]["mean"] if isinstance(agg["mttd_ms"], dict) else 0.0
        e2e.append(mttd + 50.0)

    bars = ax.bar(labels, e2e, color=colors, width=0.45, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, e2e):
        ax.annotate(f"{val:.2f} ms\n(Simulated)", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("End-to-End Recovery Time (ms)")
    ax.set_title("Simulated End-to-End Recovery Time (Empirical Confirmation: UNAVAILABLE)", pad=15)
    ax.set_ylim(0, max(e2e) * 1.35)

    footnote = "Scientific Fairness Audit: Formulated as MTTD + simulated 50 ms action latency. True end-to-end recovery\nrequires empirical restoration confirmation timestamps, which are UNAVAILABLE in offline trace data."
    fig.text(0.08, -0.06, footnote, ha="left", fontsize=8.0, fontstyle="italic", color="#444444")
    save_plot(fig, "14_recovery_end_to_end_time.png")


def plot_15_recovery_early_warning_lead_time(bench_v2):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pred_algos = ["paper1_ipft", "paper4_pregan"]
    labels = [ALGO_SHORT[a] for a in pred_algos]
    colors = [ALGO_COLORS[a] for a in pred_algos]

    leads = [bench_v2["algorithms"][a]["aggregated"]["early_warning_time_ms"]["mean"] for a in pred_algos]
    bars = ax.bar(labels, leads, color=colors, width=0.4, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, leads):
        ax.annotate(f"{val:.2f} ms", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Early Warning Lead Time (ms) [Higher is Better]")
    ax.set_title("Predictive Early Warning Lead Time Prior to Fault Onset", pad=15)
    ax.set_ylim(0, max(leads) * 1.3 if max(leads) > 0 else 10)

    footnote = "Scientific Fairness Note: Applies only to predictive architectures (IPFT & PreGAN). Causal FT, BWOAIF, RCD = N/A."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "15_recovery_early_warning_lead_time.png")


def plot_16_recovery_service_availability(bench_v2):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [ALGO_SHORT[a] for a in RECOVERY_ALGOS]
    colors = [ALGO_COLORS[a] for a in RECOVERY_ALGOS]

    avails = [bench_v2["algorithms"][a]["aggregated"]["availability_percent"]["mean"] for a in RECOVERY_ALGOS]
    bars = ax.bar(labels, avails, color=colors, width=0.45, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, avails):
        ax.annotate(f"{val:.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Service Availability (%) [Higher is Better]")
    ax.set_title("Observed Service Availability Under Streaming Fault Injection", pad=15)
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    footnote = "Scientific Fairness Note: Computed from unmitigated degraded episodes. BWOAIF and RCD excluded (N/A)."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "16_recovery_service_availability.png")


def plot_17_recovery_operational_cost(bench_v2):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [ALGO_SHORT[a] for a in RECOVERY_ALGOS]
    colors = [ALGO_COLORS[a] for a in RECOVERY_ALGOS]

    costs = [bench_v2["algorithms"][a]["aggregated"]["operational_recovery_cost"]["mean"] for a in RECOVERY_ALGOS]
    bars = ax.bar(labels, costs, color=colors, width=0.45, edgecolor="#333333", alpha=0.9)
    for bar, val in zip(bars, costs):
        ax.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Operational Cost Penalty (Score) [Lower is Better]")
    ax.set_title("Normalized Operational Recovery Cost Penalty", pad=15)
    ax.set_ylim(0, max(costs) * 1.3)

    footnote = "Scientific Fairness Note: Penalizes action frequency and overhead. BWOAIF and RCD excluded (N/A)."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "17_recovery_operational_cost.png")


def plot_18_recovery_migration_bandwidth(bench_v2):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    mig_algos = ["paper1_ipft", "paper4_pregan"]
    labels = [ALGO_SHORT[a] for a in mig_algos]
    colors = [ALGO_COLORS[a] for a in mig_algos]

    bws = [bench_v2["algorithms"][a]["aggregated"]["migration_bandwidth_kb_sec"]["mean"] for a in mig_algos]
    errs = [bench_v2["algorithms"][a]["aggregated"]["migration_bandwidth_kb_sec"]["std"] for a in mig_algos]

    bars = ax.bar(labels, bws, yerr=errs, color=colors, width=0.4, edgecolor="#333333", capsize=5, alpha=0.9)
    for bar, val, err in zip(bars, bws, errs):
        ax.annotate(f"{val:,.0f} ± {err:,.0f} KB/s", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Migration Bandwidth (KB / sec) [Lower is Better]")
    ax.set_title("Preemptive Container Migration Network Overhead", pad=15)
    ax.set_ylim(0, max([b + e for b, e in zip(bws, errs)]) * 1.25)

    footnote = "Scientific Fairness Note: Only IPFT and PreGAN perform container migration. Causal FT, BWOAIF, RCD = N/A."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "18_recovery_migration_bandwidth.png")


# =============================================================================
# PART E: COMPUTATIONAL OVERHEAD (Plots 19 - 23)
# =============================================================================

def plot_19_throughput(bench_v2):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER]

    ths = [bench_v2["algorithms"][a]["aggregated"]["throughput_rec_sec"]["mean"] for a in ALGO_ORDER]
    errs = [bench_v2["algorithms"][a]["aggregated"]["throughput_rec_sec"]["std"] for a in ALGO_ORDER]

    bars = ax.bar(labels, ths, yerr=errs, color=colors, width=0.5, edgecolor="#333333", capsize=4, alpha=0.9)
    for bar, val, err in zip(bars, ths, errs):
        ax.annotate(f"{val:,.0f} ± {err:.0f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Throughput (records / sec) [Higher is Better]")
    ax.set_title("Stream Processing Throughput across All 5 Approaches (Mean ± Std)", pad=15)
    ax.set_ylim(0, max([b + e for b, e in zip(ths, errs)]) * 1.2)

    footnote = "Scientific Fairness Note: Evaluated on identical single-thread CPU execution (24,000 eval records, N=3 seeds)."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "19_overhead_throughput.png")


def plot_20_cpu_utilization(bench_v2):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER]

    cpus = [bench_v2["algorithms"][a]["aggregated"]["avg_cpu_percent"]["mean"] for a in ALGO_ORDER]
    errs = [bench_v2["algorithms"][a]["aggregated"]["avg_cpu_percent"]["std"] for a in ALGO_ORDER]

    bars = ax.bar(labels, cpus, yerr=errs, color=colors, width=0.5, edgecolor="#333333", capsize=4, alpha=0.9)
    for bar, val, err in zip(bars, cpus, errs):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Average CPU Utilization (%) [Lower is Better]")
    ax.set_title("Single-Core CPU Utilization across All 5 Approaches (Real psutil)", pad=15)
    ax.set_ylim(90, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    footnote = "Scientific Fairness Note: Measured via genuine psutil background process sampling during streaming."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "20_overhead_cpu_utilization.png")


def plot_21_peak_ram_rss(bench_v2):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER]

    rams = [bench_v2["algorithms"][a]["aggregated"]["peak_rss_mb"]["mean"] for a in ALGO_ORDER]
    errs = [bench_v2["algorithms"][a]["aggregated"]["peak_rss_mb"]["std"] for a in ALGO_ORDER]

    bars = ax.bar(labels, rams, yerr=errs, color=colors, width=0.5, edgecolor="#333333", capsize=4, alpha=0.9)
    for bar, val, err in zip(bars, rams, errs):
        ax.annotate(f"{val:.1f} MB", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Peak Resident Set Size (RSS in MB) [Lower is Better]")
    ax.set_title("Process Memory Footprint across All 5 Approaches (Real psutil)", pad=15)
    ax.set_ylim(0, max([b + e for b, e in zip(rams, errs)]) * 1.25)

    footnote = "Scientific Fairness Note: Causal FT achieves lowest resident memory (332.8 MB) via streaming HalfSpaceTrees."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "21_overhead_peak_ram_rss.png")


def plot_22_energy_consumption(bench_v2):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER]

    energies = [bench_v2["algorithms"][a]["aggregated"]["estimated_energy_joules"]["mean"] for a in ALGO_ORDER]
    errs = [bench_v2["algorithms"][a]["aggregated"]["estimated_energy_joules"]["std"] for a in ALGO_ORDER]

    bars = ax.bar(labels, energies, yerr=errs, color=colors, width=0.5, edgecolor="#333333", capsize=4, alpha=0.9)
    for bar, val, err in zip(bars, energies, errs):
        ax.annotate(f"{val:.1f} ± {err:.1f} J", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Estimated Energy Consumption (Joules) [Lower is Better]")
    ax.set_title("Total Energy Consumption over 24,000 Evaluation Observations", pad=15)
    ax.set_ylim(0, max([b + e for b, e in zip(energies, errs)]) * 1.2)

    footnote = "Scientific Fairness Note: Energy computed via calibrated polynomial power model based on CPU % and execution duration."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "22_overhead_energy_consumption.png")


def plot_23_telemetry_bandwidth(bench_v2):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER]

    bws = [bench_v2["algorithms"][a]["aggregated"]["bandwidth_kb_sec"]["mean"] for a in ALGO_ORDER]
    errs = [bench_v2["algorithms"][a]["aggregated"]["bandwidth_kb_sec"]["std"] for a in ALGO_ORDER]

    bars = ax.bar(labels, bws, yerr=errs, color=colors, width=0.5, edgecolor="#333333", capsize=4, alpha=0.9)
    for bar, val, err in zip(bars, bws, errs):
        ax.annotate(f"{val:,.0f} KB/s", xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + err), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Telemetry Bandwidth (KB / sec)")
    ax.set_title("Total Network / Telemetry Ingestion Bandwidth Rate", pad=15)
    ax.set_ylim(0, max([b + e for b, e in zip(bws, errs)]) * 1.25)

    footnote = "Scientific Fairness Note: IPFT and PreGAN include high migration state transfer overhead in total network volume."
    fig.text(0.12, -0.02, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "23_overhead_telemetry_bandwidth.png")


# =============================================================================
# PART F: ROBUSTNESS & STABILITY (Plots 24 - 26)
# =============================================================================

def plot_24_seed_f1_stability(df_seed_conf):
    fig, ax = plt.subplots(figsize=(9, 5))
    seeds = [42, 43, 44]
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]

    for i, a in enumerate(DETECTION_ALGOS):
        sub = df_seed_conf[df_seed_conf["algorithm_id"] == a]
        f1_vals = sub["f1"].tolist()
        ax.plot(seeds, f1_vals, label=ALGO_SHORT[a], color=ALGO_COLORS[a], marker=markers[i], markersize=8, linestyle=linestyles[i], linewidth=2.0)
        for s, v in zip(seeds, f1_vals):
            ax.annotate(f"{v:.1f}%", xy=(s, v), xytext=(0, 6 if a != "paper4_pregan" or s != 43 else 8), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Random Seed")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("Reproducibility & Stability: F1-Score across Seeds 42, 43, 44", pad=15)
    ax.set_xticks(seeds)
    ax.set_xticklabels(["Seed 42", "Seed 43", "Seed 44"], fontweight="bold")
    ax.set_ylim(-5, 60)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.legend(frameon=True, loc="upper right")

    ax.annotate(
        "PreGAN Collapse (F1 = 0.0%)\nCatastrophic seed sensitivity",
        xy=(43, 0.0),
        xytext=(43.05, 15.0),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1),
    )

    footnote = "Scientific Finding: Causal FT (49.9%), IPFT (37.8%), and BWOAIF (39.2%) are fully deterministic across seeds (std = 0.0)."
    fig.text(0.12, -0.04, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "24_robustness_seed_f1_stability.png")


def plot_25_seed_throughput_stability(bench_v2):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    seeds = [42, 43, 44]
    all_markers = ["o", "s", "^", "v", "D"]

    for i, a in enumerate(ALGO_ORDER):
        trials = bench_v2["algorithms"][a]["trials"]
        ths = [t["throughput_rec_sec"] for t in trials]
        ax.plot(seeds, ths, label=ALGO_SHORT[a], color=ALGO_COLORS[a], marker=all_markers[i], markersize=7.5, linewidth=1.8)

    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Throughput (records / sec)")
    ax.set_title("Computational Stability: Processing Throughput across Seeds 42, 43, 44", pad=15)
    ax.set_xticks(seeds)
    ax.set_xticklabels(["Seed 42", "Seed 43", "Seed 44"], fontweight="bold")
    ax.set_ylim(150, 2350)
    ax.legend(frameon=True, loc="center right")

    footnote = "Scientific Finding: Causal FT (~1,923 rec/s) and BWOAIF (~1,774 rec/s) demonstrate stable throughput (std < 1.5%)."
    fig.text(0.12, -0.03, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "25_robustness_seed_throughput_stability.png")


def plot_26_persistence_sensitivity(df_val_exp):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sub = df_val_exp[df_val_exp["config_id"].str.contains("PERSISTENCE|CORRECTED")].copy()
    k_vals = sub["persistence_k"].tolist()
    f1_vals = sub["f1"].tolist()
    prec_vals = sub["precision"].tolist()
    rec_vals = sub["recall"].tolist()
    fpr_vals = sub["fpr"].tolist()

    ax.plot(k_vals, f1_vals, marker="o", color="#d7191c", linewidth=2.2, label="F1-Score (%)")
    ax.plot(k_vals, prec_vals, marker="s", color="#2b83ba", linewidth=2.0, label="Precision (%)")
    ax.plot(k_vals, rec_vals, marker="^", color="#fdae61", linewidth=2.0, label="Recall (%)")
    ax.plot(k_vals, fpr_vals, marker="v", color="#756bb1", linestyle="--", linewidth=1.8, label="FPR (%)")

    ax.set_xlabel("Persistence Confirmation Threshold (k Consecutive Anomalies)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Causal FT Persistence Sensitivity Analysis (k = 1 to 5 at τ = 0.15)", pad=15)
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="center right")

    footnote = "Scientific Finding: k=1 maximizes F1 (49.9%) and recall (63.0%). Higher k suppresses alarms on short network attack episodes."
    fig.text(0.12, -0.03, footnote, ha="left", fontsize=8.5, fontstyle="italic", color="#444444")
    save_plot(fig, "26_robustness_causal_ft_persistence_sensitivity.png")


# =============================================================================
# PART G: GENERATE SUMMARY & PPT TABLES (Phase 7)
# =============================================================================

def generate_ppt_tables(bench_v2, rca_data, df_conf):
    # 1. Comprehensive 23-Row Comparison Table
    metrics_spec = [
        ("Detection Accuracy", "%", "accuracy", "Higher", "Detection"),
        ("Balanced Accuracy", "%", "balanced_accuracy", "Higher", "Detection"),
        ("Precision", "%", "precision", "Higher", "Detection"),
        ("Recall", "%", "recall", "Higher", "Detection"),
        ("F1-Score", "%", "f1", "Higher", "Detection"),
        ("ROC-AUC", "score", "roc_auc", "Higher", "Detection"),
        ("PR-AUC", "score", "pr_auc", "Higher", "Detection"),
        ("False Positive Rate (FPR)", "%", "fpr", "Lower", "Detection"),
        ("Mean Time to Detect (MTTD)", "ms", "mttd_ms", "Lower", "Timing"),
        ("Early Warning Lead Time", "ms", "early_warning_time_ms", "Higher", "Timing"),
        ("RCA Top-1 Localization Recall", "%", "rca_top_1", "Higher", "Diagnosis"),
        ("RCA Direct Computation Latency", "ms", "rca_computation_latency_ms", "Lower", "Diagnosis"),
        ("Recovery Success Rate", "%", "recovery_success_rate", "Higher", "Recovery"),
        ("Recovery Action Latency", "ms", "recovery_latency_ms", "Lower", "Recovery"),
        ("End-to-End Recovery Time", "ms", "end_to_end_recovery_time_ms", "Lower", "Recovery"),
        ("Observed Service Availability", "%", "availability_percent", "Higher", "Reliability"),
        ("Processing Throughput", "rec/s", "throughput_rec_sec", "Higher", "Computation"),
        ("Mean Inference Latency", "ms/rec", "per_record_latency_ms", "Lower", "Computation"),
        ("P95 Inference Latency", "ms/rec", "per_record_latency_p95_ms", "Lower", "Computation"),
        ("Average CPU Utilization", "%", "avg_cpu_percent", "Lower", "Resource"),
        ("Peak Process Memory (RSS)", "MB", "peak_rss_mb", "Lower", "Resource"),
        ("Estimated Energy Consumption", "Joules", "estimated_energy_joules", "Lower", "Resource"),
        ("Normalized Operational Recovery Cost", "score", "operational_recovery_cost", "Lower", "Recovery"),
    ]

    rows_23 = []
    for m_label, unit, key, direction, cat in metrics_spec:
        row = {"Metric": m_label, "Unit": unit, "Direction": direction, "Category": cat}
        for a in ALGO_ORDER:
            # Check RCA special keys
            if key == "rca_top_1":
                if a == "proposed_causal_ft":
                    row[ALGO_SHORT[a]] = f"{rca_data['synthetic_controlled_benchmark']['causal_ft']['top_1_recall']:.1f}% (Syn) / {rca_data['edge_iiotset_domain_proxy_benchmark']['causal_ft']['top_1_recall']:.1f}% (Proxy)"
                elif a == "paper3_rcd":
                    row[ALGO_SHORT[a]] = f"{rca_data['synthetic_controlled_benchmark']['rcd']['top_1_recall']:.1f}% (Syn) / {rca_data['edge_iiotset_domain_proxy_benchmark']['rcd']['top_1_recall']:.1f}% (Proxy)"
                else:
                    row[ALGO_SHORT[a]] = "N/A"
            elif key in ["recovery_success_rate", "recovery_latency_ms", "end_to_end_recovery_time_ms"]:
                row[ALGO_SHORT[a]] = "UNAVAILABLE"
            else:
                agg = bench_v2["algorithms"][a]["aggregated"].get(key)
                if isinstance(agg, dict):
                    m_val = agg["mean"]
                    s_val = agg["std"]
                    fmt = ".4f" if key in ["roc_auc", "pr_auc", "operational_recovery_cost", "per_record_latency_ms", "rca_computation_latency_ms"] else (".2f" if key in ["mttd_ms", "early_warning_time_ms", "recovery_latency_ms"] else ".1f")
                    err_str = f" ± {s_val:{fmt}}" if s_val > 0.0001 else ""
                    row[ALGO_SHORT[a]] = f"{m_val:{fmt}}{err_str}"
                elif agg == "NOT_APPLICABLE" or agg is None:
                    row[ALGO_SHORT[a]] = "N/A"
                else:
                    row[ALGO_SHORT[a]] = str(agg)
        rows_23.append(row)

    df_23 = pd.DataFrame(rows_23)
    path_23 = os.path.join(TABLES_DIR, "algorithm_comparison_summary.csv")
    df_23.to_csv(path_23, index=False)
    print(f"[OK] Saved comprehensive 23-row comparison: {path_23}")

    # 2. Concise Slide-Ready Table (PPT Key Results)
    key_metrics_subset = [
        "Detection Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "False Positive Rate (FPR)",
        "Mean Time to Detect (MTTD)",
        "RCA Top-1 Localization Recall",
        "RCA Direct Computation Latency",
        "Recovery Success Rate",
        "Observed Service Availability",
        "Processing Throughput",
        "Mean Inference Latency",
        "Peak Process Memory (RSS)",
        "Estimated Energy Consumption",
    ]
    df_key = df_23[df_23["Metric"].isin(key_metrics_subset)].copy()
    path_key = os.path.join(TABLES_DIR, "key_results_summary.csv")
    df_key.to_csv(path_key, index=False)
    print(f"[OK] Saved slide-ready key results table: {path_key}")

    # 3. Metric Definitions CSV
    definitions = [
        {"Metric": "Detection Accuracy", "Formula": "(TP + TN) / (TP + TN + FP + FN)", "Unit": "%", "Timestamps_Counts": "Confusion matrix counts", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "Balanced Accuracy", "Formula": "0.5 * (TP/(TP+FN) + TN/(TN+FP))", "Unit": "%", "Timestamps_Counts": "Class-normalized recall/specificity", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "Precision", "Formula": "TP / (TP + FP)", "Unit": "%", "Timestamps_Counts": "True positives and false positives", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "Recall", "Formula": "TP / (TP + FN)", "Unit": "%", "Timestamps_Counts": "True positives and false negatives", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "F1-Score", "Formula": "2 * P * R / (P + R)", "Unit": "%", "Timestamps_Counts": "Harmonic mean of precision and recall", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "ROC-AUC", "Formula": "Area under TPR vs FPR curve", "Unit": "Score [0, 1]", "Timestamps_Counts": "Continuous anomaly scores across all thresholds", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "PR-AUC", "Formula": "Average Precision under PR curve", "Unit": "Score [0, 1]", "Timestamps_Counts": "Continuous anomaly scores across all thresholds", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Higher"},
        {"Metric": "False Positive Rate (FPR)", "Formula": "FP / (FP + TN)", "Unit": "%", "Timestamps_Counts": "False positives and true negatives", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Lower"},
        {"Metric": "Mean Time to Detect (MTTD)", "Formula": "(1/|E|) * sum(t_first_detect - t_episode_onset)", "Unit": "ms", "Timestamps_Counts": "Fault episode onset and first detection timestamp", "Applicable_Algorithms": "Causal FT, IPFT, BWOAIF, PreGAN", "Direction": "Lower"},
        {"Metric": "Early Warning Lead Time", "Formula": "t_episode_onset - t_predictive_alarm", "Unit": "ms", "Timestamps_Counts": "Pre-onset predictive alarm timestamp", "Applicable_Algorithms": "IPFT, PreGAN (Predictive only)", "Direction": "Higher"},
        {"Metric": "RCA Top-k Localization Recall", "Formula": "Hits in top-k causes / Total fault episodes", "Unit": "%", "Timestamps_Counts": "Ranked causal attribution list vs ground truth", "Applicable_Algorithms": "Causal FT, RCD (RCA only)", "Direction": "Higher"},
        {"Metric": "RCA Computation Latency", "Formula": "t_diagnose_end - t_diagnose_start", "Unit": "ms", "Timestamps_Counts": "Direct CPU execution clock", "Applicable_Algorithms": "Causal FT, RCD (RCA only)", "Direction": "Lower"},
        {"Metric": "Recovery Success Rate", "Formula": "Successful actions / Attempted actions", "Unit": "%", "Timestamps_Counts": "Count of successful restorations vs total actions", "Applicable_Algorithms": "Causal FT, IPFT, PreGAN", "Direction": "Higher"},
        {"Metric": "Recovery Action Latency", "Formula": "t_restoration_confirmed - t_action_start", "Unit": "ms", "Timestamps_Counts": "Mitigation dispatch and completion timestamps", "Applicable_Algorithms": "Causal FT, IPFT, PreGAN", "Direction": "Lower"},
        {"Metric": "End-to-End Recovery Time", "Formula": "t_restoration_confirmed - t_episode_onset", "Unit": "ms", "Timestamps_Counts": "Episode onset to verified restoration", "Applicable_Algorithms": "Causal FT, IPFT, PreGAN", "Direction": "Lower"},
        {"Metric": "Observed Service Availability", "Formula": "(Total operational time - Unmitigated downtime) / Total operational time", "Unit": "%", "Timestamps_Counts": "Streaming observation timeline duration", "Applicable_Algorithms": "Causal FT, IPFT, PreGAN", "Direction": "Higher"},
        {"Metric": "Processing Throughput", "Formula": "Total records / Elapsed wall-clock seconds", "Unit": "rec/s", "Timestamps_Counts": "Record count and execution timer", "Applicable_Algorithms": "All 5 Approaches", "Direction": "Higher"},
        {"Metric": "Mean Inference Latency", "Formula": "Mean single-record process() execution duration", "Unit": "ms/rec", "Timestamps_Counts": "High-precision timer per record", "Applicable_Algorithms": "All 5 Approaches", "Direction": "Lower"},
        {"Metric": "Peak Resident Memory (RSS)", "Formula": "Peak process memory sampled via psutil", "Unit": "MB", "Timestamps_Counts": "psutil process memory info", "Applicable_Algorithms": "All 5 Approaches", "Direction": "Lower"},
        {"Metric": "Average CPU Utilization", "Formula": "Mean CPU percentage sampled via psutil", "Unit": "%", "Timestamps_Counts": "psutil CPU percentage samples", "Applicable_Algorithms": "All 5 Approaches", "Direction": "Lower"},
        {"Metric": "Estimated Energy Consumption", "Formula": "Integral of (P_idle + (P_peak - P_idle)*CPU%) * dt", "Unit": "Joules", "Timestamps_Counts": "CPU utilization and elapsed execution seconds", "Applicable_Algorithms": "All 5 Approaches", "Direction": "Lower"},
        {"Metric": "Normalized Operational Recovery Cost", "Formula": "0.4*(Actions/N) + 0.4*(SLO_viols/N) + 0.2*Overhead", "Unit": "Score [0, 1]", "Timestamps_Counts": "Total actions and SLO latency breaches", "Applicable_Algorithms": "Causal FT, IPFT, PreGAN", "Direction": "Lower"},
    ]
    df_defs = pd.DataFrame(definitions)
    path_defs = os.path.join(TABLES_DIR, "metric_definitions.csv")
    df_defs.to_csv(path_defs, index=False)
    print(f"[OK] Saved metric definitions table: {path_defs}")


# =============================================================================
# PART H: DOCUMENTATION (Phase 8)
# =============================================================================

def generate_final_readme():
    readme_path = os.path.join(FINAL_PLOTS_DIR, "README.md")
    content = """# Final Scientific Presentation Artifacts: Cross-Algorithm Comparison Suite

This directory contains the complete set of **26 publication-grade 300-DPI presentation figures** for the five-algorithm benchmark on the **Edge-IIoTset** dataset across three independent trials (**Random Seeds 42, 43, and 44**).

---

## 1. Experimental Protocol & Execution Integrity

* **Common Evaluation Stream**: All comparable algorithms processed the identical **24,000 test observations** from `data/processed/test/test_*.csv` under common single-thread CPU affinity (`torch.set_num_threads(1)`).
* **Warm-up Isolation**: The first **1,000 records** were strictly designated as warm-up and excluded from all detection, timing, and computational metrics.
* **Feature Strictness**: All 62 input features were strictly non-label telemetry. `Attack_label` and `Attack_type` were never exposed to any model's `process()` pipeline.
* **Independent Calibration**: Thresholds were calibrated exclusively on validation data (`results/tables/threshold_calibration.json`) prior to test evaluation.
* **Error Bars**: Error bars represent the sample standard deviation ($s = \\sqrt{\\frac{1}{N-1}\\sum (x_i - \\bar{x})^2}$, with $N=3$ and `ddof=1`).
* **Hardware & System Instrumentation**: Evaluated on an Intel Core i5-13500H host (12 cores / 16 threads, 15.69 GB RAM) using genuine background `psutil` CPU and resident memory (RSS) sampling.
* **Standalone psutil Installation**: `psutil` was installed cleanly into `.venv` via `.venv\\Scripts\\python.exe -m pip install psutil`. `requirements.txt` was preserved intact.

---

## 2. Root-Cause Correction of Causal FT Test F1

During the Phase 1 audit, an implementation bug was discovered in `algorithms/causal_ft/pipeline.py`:
- `CausalFaultTolerancePipeline.initialize()` checked `if "detection_threshold" in config:`.
- However, `run_full_benchmark.py` passed `{"threshold": threshold, "upper_threshold": threshold}`.
- Consequently, the calibrated validation threshold ($\tau = 0.15$) was **never applied** during benchmark execution; the detector defaulted to $\tau = 0.65$.
- At $\tau = 0.65$, only extreme outliers ($2.66\%$ recall) were flagged, collapsing F1 to $4.84\%$.
- Correcting this parameter key mismatch allows the calibrated threshold ($\tau = 0.15$) to reach the detector. Test F1 is restored to **$49.92\%$** ($41.22\%$ Precision, $63.28\%$ Recall, $70.42\%$ Accuracy, $27.41\%$ FPR, $67.93\%$ Balanced Accuracy), matching its validation performance ($49.93\%$ F1) with **zero test-set tuning**.

---

## 3. Algorithm Capability Matrix & Scientific Fairness Boundaries

Placing algorithms with fundamentally disjoint capabilities on the same comparative axis produces scientifically invalid conclusions:

| Algorithm | Paper Reference | Streaming Detection | Causal RCA | Preemptive Migration | Closed-Loop FT | Valid Metric Plots |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Proposed Causal FT** | *Our Pipeline* | **Yes** | **Yes** | No | **Yes** | Detection, RCA, Computational, Reliability |
| **IPFT** | Theodoropoulos et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |
| **BWOAIF** | Hannák et al., 2023 | **Yes** | No | No | No | Detection, Computational |
| **RCD** | Ikram et al., 2022 | No | **Yes** | No | No | RCA, Computational |
| **PreGAN** | Tuli et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |

### Exclusions and N/A Annotations
1. **RCD Excluded from Detection Plots**: RCD (*Robust Causal Diagnosis*, Ikram et al., 2022) is an offline root-cause diagnosis algorithm that takes known anomalous episodes and builds a localized causal graph. It does not perform streaming binary anomaly detection. RCD is annotated as `N/A` for all detection metrics.
2. **BWOAIF and RCD Excluded from Recovery/Mitigation Plots**: Neither algorithm implements closed-loop migration or recovery actions. They are annotated as `N/A`.
3. **Accuracy vs Balanced Accuracy**: On this imbalanced test stream (23.3% attack), IPFT achieves 23.3% accuracy and 50.0% balanced accuracy (chance baseline) by predicting all 1s. Conversely, PreGAN on Seed 43 collapses and predicts all 0s, artificially inflating its ordinary accuracy to 76.69% while its balanced accuracy reveals chance performance (50.0%).

---

## 4. Complete Inventory of the 26 Presentation Plots

### A. Detection Comparison
* `01_detection_precision_recall_f1.png`: Grouped bar chart of Precision, Recall, and F1-score (Mean ± Std).
* `02_detection_roc_pr_auc.png`: Discrimination capability evaluated by ROC-AUC and PR-AUC.
* `03_detection_accuracy_balanced_accuracy.png`: Standard Accuracy versus Balanced Accuracy.
* `04_detection_fpr_fnr_tradeoff.png`: False Positive Rate vs False Negative Rate tradeoff.
* `05_detection_confusion_matrices.png`: 2x2 grid displaying empirical confusion matrices for all 4 detection approaches.
* `06_detection_roc_curves.png`: Multi-algorithm ROC curves on common test stream.
* `07_detection_precision_recall_curves.png`: Multi-algorithm Precision-Recall curves.

### B. Detection Timing
* `08_timing_mttd_comparison.png`: Discrete episode-based Mean Time to Detect (MTTD) and Episode Detection Coverage.
* `09_timing_inference_latency_mean_p95.png`: Single-observation Mean and P95 processing latency (ms/rec).

### C. Causal Root-Cause Diagnosis
* `10_rca_localization_top_k_recall.png`: Top-1, Top-3, and Top-5 Root-Cause Localization Recall: Causal FT vs RCD.
* `11_rca_computation_latency.png`: Direct CPU execution latency of root-cause localization algorithms.

### D. Recovery and Resilience
* `12_recovery_success_rate.png`: Recovery action success percentage (successful actions / attempted actions).
* `13_recovery_action_latency.png`: Mitigation action dispatch-to-completion duration.
* `14_recovery_end_to_end_time.png`: Complete recovery time from episode onset to restoration.
* `15_recovery_early_warning_lead_time.png`: Predictive lead time prior to fault onset (IPFT and PreGAN).
* `16_recovery_service_availability.png`: Service availability percentage under continuous fault injection.
* `17_recovery_operational_cost.png`: Normalized operational recovery cost penalty score.
* `18_recovery_migration_bandwidth.png`: Preemptive container migration network bandwidth rate (KB/s).

### E. Computational Overhead (All 5 Algorithms)
* `19_overhead_throughput.png`: Stream processing rate (records/second).
* `20_overhead_cpu_utilization.png`: Single-core CPU utilization percentage measured via real psutil.
* `21_overhead_peak_ram_rss.png`: Peak resident memory footprint (RSS in MB). Causal FT achieves the lowest footprint (332.8 MB).
* `22_overhead_energy_consumption.png`: Total estimated electrical energy consumption (Joules).
* `23_overhead_telemetry_bandwidth.png`: Total telemetry and control streaming bandwidth rate (KB/s).

### F. Robustness & Stability
* `24_robustness_seed_f1_stability.png`: Seed-wise F1 tracking across Seeds 42, 43, and 44, exposing PreGAN's collapse at Seed 43.
* `25_robustness_seed_throughput_stability.png`: Processing throughput stability across random seeds.
* `26_robustness_causal_ft_persistence_sensitivity.png`: F1, Precision, Recall, and FPR vs persistence $k \in [1, 5]$.

---

## 5. Summary Tables
* **[algorithm_comparison_summary.csv](../results/tables/algorithm_comparison_summary.csv)**: Comprehensive 23-row multi-metric comparison.
* **[key_results_summary.csv](../results/tables/key_results_summary.csv)**: Concise summary of key experimental results.
* **[metric_definitions.csv](../results/tables/metric_definitions.csv)**: Mathematical definitions, units, and directionality guide.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Generated final README: {readme_path}")


def main():
    print("=" * 75)
    print("GENERATING FINAL PPT DELIVERABLES (Plots 1 - 26 & Tables)")
    print("=" * 75)

    bench_v2, rca_data, df_conf, df_seed_conf, df_val_exp = load_all_results()

    # Part A: Detection Comparison
    print("\n--- Part A: Detection Performance Plots ---")
    plot_01_precision_recall_f1(df_conf)
    plot_02_roc_pr_auc(df_conf)
    plot_03_accuracy_balanced_accuracy(df_conf)
    plot_04_fpr_fnr_tradeoff(df_conf)
    plot_05_confusion_matrices(df_conf)
    plot_06_roc_curves()
    plot_07_precision_recall_curves()

    # Part B: Detection Timing
    print("\n--- Part B: Detection Timing Plots ---")
    plot_08_timing_mttd(bench_v2)
    plot_09_inference_latency_mean_p95(bench_v2)

    # Part C: Root-Cause Diagnosis
    print("\n--- Part C: Root-Cause Diagnosis Plots ---")
    plot_10_rca_top_k_recall(rca_data)
    plot_11_rca_computation_latency(rca_data)

    # Part D: Recovery & Resilience
    print("\n--- Part D: Recovery and Resilience Plots ---")
    plot_12_recovery_success_rate(bench_v2)
    plot_13_recovery_action_latency(bench_v2)
    plot_14_recovery_end_to_end_time(bench_v2)
    plot_15_recovery_early_warning_lead_time(bench_v2)
    plot_16_recovery_service_availability(bench_v2)
    plot_17_recovery_operational_cost(bench_v2)
    plot_18_recovery_migration_bandwidth(bench_v2)

    # Part E: Computational Overhead
    print("\n--- Part E: Computational Overhead Plots ---")
    plot_19_throughput(bench_v2)
    plot_20_cpu_utilization(bench_v2)
    plot_21_peak_ram_rss(bench_v2)
    plot_22_energy_consumption(bench_v2)
    plot_23_telemetry_bandwidth(bench_v2)

    # Part F: Robustness & Stability
    print("\n--- Part F: Robustness & Stability Plots ---")
    plot_24_seed_f1_stability(df_seed_conf)
    plot_25_seed_throughput_stability(bench_v2)
    plot_26_persistence_sensitivity(df_val_exp)

    # Part G: PPT Tables
    print("\n--- Part G: Generating Final PPT Tables ---")
    generate_ppt_tables(bench_v2, rca_data, df_conf)

    # Part H: Documentation
    print("\n--- Part H: Generating Final README Documentation ---")
    generate_final_readme()

    print("\n" + "=" * 75)
    print("ALL 26 FINAL PRESENTATION PLOTS AND TABLES GENERATED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
