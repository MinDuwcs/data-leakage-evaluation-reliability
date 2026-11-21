"""
==============================================================================
Experiment A – Preprocessing Leakage: StandardScaler
==============================================================================
So sánh hai quy trình:
  • A-Control (Correct): Split TRƯỚC → fit StandardScaler CHỈ trên Train
                          → transform Train/Val/Test bằng scaler đã fit.
  • A-Leakage (Leaked):  Fit StandardScaler trên TOÀN BỘ dữ liệu
                          → split SAU → train/evaluate.

Dataset : fraud_detection.csv  (cột target: Fraudulent)
Models  : Logistic Regression, Decision Tree, Random Forest
Metrics : Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Runs    : Lặp lại N_RUNS lần với các random seed khác nhau để đo độ biến thiên.

Output  : Bảng metric trung bình ± std, biểu đồ so sánh, confusion matrix.
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fraud_detection.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "standardscaler_leakage")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 10                    # Số lần lặp thí nghiệm
TEST_SIZE = 0.2                # Tỷ lệ tập Test
VAL_SIZE = 0.15                # Tỷ lệ tập Validation (tính trên phần còn lại sau khi tách Test)
HOLDOUT_SIZE = 0.1             # Tỷ lệ clean holdout (tách ra trước khi thí nghiệm)
BASE_SEED = 42
TARGET_COL = "Fraudulent"

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=BASE_SEED, solver="lbfgs"
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        class_weight="balanced", random_state=BASE_SEED
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=BASE_SEED, n_jobs=-1
    ),
}


# ============================================================================
# 2. LOAD & PREPARE DATA
# ============================================================================
def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset, encode categorical columns, drop ID columns, handle missing values."""
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}\n")

    # Drop ID columns (không có giá trị dự đoán)
    id_cols = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    df = df.drop(columns=id_cols, errors="ignore")

    # Tách target
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Fill missing values với median (sẽ được tính trên từng split khi cần)
    X = X.fillna(X.median())

    print(f"Features after preprocessing: {list(X.columns)}")
    print(f"Shape: X={X.shape}, y={y.shape}\n")
    return X, y


def evaluate_model(model, X_eval, y_eval) -> dict:
    """Tính toán các metrics cho một model đã train."""
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_eval, y_prob),
        "PR-AUC": average_precision_score(y_eval, y_prob),
    }


# ============================================================================
# 3. EXPERIMENT PIPELINES
# ============================================================================
def run_correct_pipeline(X, y, seed: int) -> dict:
    """
    A-Control: Quy trình ĐÚNG
    1. Split trước (Train/Val/Test)
    2. Fit StandardScaler CHỈ trên Train
    3. Transform tất cả các tập bằng scaler đã fit trên Train
    """
    # Split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    # Split: Train vs Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    # Fit scaler CHỈ trên Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train_scaled, y_train)
        results[name] = {
            "val": evaluate_model(model, X_val_scaled, y_val),
            "test": evaluate_model(model, X_test_scaled, y_test),
        }
    return results


def run_leakage_pipeline(X, y, seed: int) -> dict:
    """
    A-Leakage: Quy trình có DATA LEAKAGE
    1. Fit StandardScaler trên TOÀN BỘ dữ liệu (bao gồm Test)
    2. Transform toàn bộ
    3. SAU ĐÓ mới Split
    """
    # Fit scaler trên TOÀN BỘ dữ liệu (LEAKAGE!)
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)

    # Split SAU khi đã scale
    X_temp, X_test_scaled, y_temp, y_test = train_test_split(
        X_scaled_all, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train_scaled, y_train)
        results[name] = {
            "val": evaluate_model(model, X_val_scaled, y_val),
            "test": evaluate_model(model, X_test_scaled, y_test),
        }
    return results


# ============================================================================
# 4. RUN EXPERIMENTS
# ============================================================================
def run_all_experiments(X, y) -> pd.DataFrame:
    """Chạy N_RUNS lần cho cả 2 pipeline, thu thập metrics."""
    all_records = []

    for run_idx in range(N_RUNS):
        seed = BASE_SEED + run_idx
        print(f"  Run {run_idx + 1}/{N_RUNS} (seed={seed})")

        correct_results = run_correct_pipeline(X, y, seed)
        leakage_results = run_leakage_pipeline(X, y, seed)

        for model_name in MODELS:
            for eval_set in ["val", "test"]:
                for metric_name in correct_results[model_name][eval_set]:
                    all_records.append({
                        "Run": run_idx + 1,
                        "Seed": seed,
                        "Pipeline": "Correct (No Leakage)",
                        "Model": model_name,
                        "Eval_Set": eval_set.capitalize(),
                        "Metric": metric_name,
                        "Value": correct_results[model_name][eval_set][metric_name],
                    })
                    all_records.append({
                        "Run": run_idx + 1,
                        "Seed": seed,
                        "Pipeline": "Leakage (Fit All)",
                        "Model": model_name,
                        "Eval_Set": eval_set.capitalize(),
                        "Metric": metric_name,
                        "Value": leakage_results[model_name][eval_set][metric_name],
                    })

    return pd.DataFrame(all_records)


# ============================================================================
# 5. ANALYSIS & VISUALIZATION
# ============================================================================
def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng tóm tắt mean ± std cho mỗi (Pipeline, Model, Eval_Set, Metric)."""
    summary = (
        df.groupby(["Pipeline", "Model", "Eval_Set", "Metric"])["Value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["mean±std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )
    return summary


def compute_delta(summary: pd.DataFrame) -> pd.DataFrame:
    """Tính chênh lệch (Leakage - Correct) cho mỗi metric."""
    pivot = summary.pivot_table(
        index=["Model", "Eval_Set", "Metric"],
        columns="Pipeline",
        values="mean",
    ).reset_index()

    if "Leakage (Fit All)" in pivot.columns and "Correct (No Leakage)" in pivot.columns:
        pivot["Delta (Leakage - Correct)"] = (
            pivot["Leakage (Fit All)"] - pivot["Correct (No Leakage)"]
        )
        pivot["Delta_%"] = (
            pivot["Delta (Leakage - Correct)"] / pivot["Correct (No Leakage)"].abs() * 100
        )
    return pivot


def plot_comparison(df: pd.DataFrame, eval_set: str = "Test"):
    """Bar chart so sánh Correct vs Leakage trên từng model và metric."""
    df_filtered = df[df["Eval_Set"] == eval_set]

    metrics = df_filtered["Metric"].unique()
    models = df_filtered["Model"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"StandardScaler Leakage: Correct vs Leakage Pipeline ({eval_set} Set)",
        fontsize=16, fontweight="bold",
    )

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        metric_data = df_filtered[df_filtered["Metric"] == metric]

        x = np.arange(len(models))
        width = 0.35

        correct_vals = []
        leakage_vals = []
        correct_stds = []
        leakage_stds = []

        for model in models:
            c_row = metric_data[
                (metric_data["Model"] == model) &
                (metric_data["Pipeline"] == "Correct (No Leakage)")
            ]
            l_row = metric_data[
                (metric_data["Model"] == model) &
                (metric_data["Pipeline"] == "Leakage (Fit All)")
            ]
            correct_vals.append(c_row["mean"].values[0] if len(c_row) > 0 else 0)
            leakage_vals.append(l_row["mean"].values[0] if len(l_row) > 0 else 0)
            correct_stds.append(c_row["std"].values[0] if len(c_row) > 0 else 0)
            leakage_stds.append(l_row["std"].values[0] if len(l_row) > 0 else 0)

        bars1 = ax.bar(x - width/2, correct_vals, width, yerr=correct_stds,
                       label="Correct", color="#2ecc71", alpha=0.8, capsize=3)
        bars2 = ax.bar(x + width/2, leakage_vals, width, yerr=leakage_stds,
                       label="Leakage", color="#e74c3c", alpha=0.8, capsize=3)

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Annotate values
        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, "standardscaler_leakage_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_delta_heatmap(delta_df: pd.DataFrame, eval_set: str = "Test"):
    """Heatmap chênh lệch (%) giữa Leakage và Correct."""
    df_filtered = delta_df[delta_df["Eval_Set"] == eval_set]
    if df_filtered.empty or "Delta_%" not in df_filtered.columns:
        print("  No delta data to plot.")
        return

    pivot = df_filtered.pivot_table(index="Model", columns="Metric", values="Delta_%")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Delta (%)"}
    )
    ax.set_title(
        f"StandardScaler Leakage: Delta (%) = (Leakage - Correct) / |Correct| × 100\n({eval_set} Set)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "standardscaler_leakage_delta_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_confusion_matrices(X, y, seed: int = BASE_SEED):
    """Plot confusion matrix cho cả 2 pipeline trên Test set."""
    # Correct pipeline
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    scaler_correct = StandardScaler()
    X_train_c = scaler_correct.fit_transform(X_train)
    X_test_c = scaler_correct.transform(X_test)

    # Leakage pipeline
    scaler_leak = StandardScaler()
    X_scaled_all = scaler_leak.fit_transform(X)
    X_temp_l, X_test_l, y_temp_l, y_test_l = train_test_split(
        X_scaled_all, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train_l, _, y_train_l, _ = train_test_split(
        X_temp_l, y_temp_l, test_size=VAL_SIZE, stratify=y_temp_l, random_state=seed
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "StandardScaler Leakage: Confusion Matrices (Test Set)",
        fontsize=14, fontweight="bold",
    )

    for col_idx, (model_name, model_fn) in enumerate(MODELS.items()):
        # Correct
        model_c = model_fn()
        model_c.fit(X_train_c, y_train)
        y_pred_c = model_c.predict(X_test_c)
        cm_c = confusion_matrix(y_test, y_pred_c)

        sns.heatmap(cm_c, annot=True, fmt="d", cmap="Blues", ax=axes[0, col_idx])
        axes[0, col_idx].set_title(f"{model_name}\n(Correct)", fontsize=10)
        axes[0, col_idx].set_xlabel("Predicted")
        axes[0, col_idx].set_ylabel("Actual")

        # Leakage
        model_l = model_fn()
        model_l.fit(X_train_l, y_train_l)
        y_pred_l = model_l.predict(X_test_l)
        cm_l = confusion_matrix(y_test_l, y_pred_l)

        sns.heatmap(cm_l, annot=True, fmt="d", cmap="Reds", ax=axes[1, col_idx])
        axes[1, col_idx].set_title(f"{model_name}\n(Leakage)", fontsize=10)
        axes[1, col_idx].set_xlabel("Predicted")
        axes[1, col_idx].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, "standardscaler_confusion_matrices.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def print_scaler_stats_comparison(X, y, seed: int = BASE_SEED):
    """
    In ra sự khác biệt giữa tham số scaler fit trên Train vs fit trên All.
    Đây là bằng chứng trực tiếp cho thấy thông tin từ Test rò rỉ vào preprocessing.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    scaler_train = StandardScaler().fit(X_train)
    scaler_all = StandardScaler().fit(X)

    stats = pd.DataFrame({
        "Feature": X.columns,
        "Mean_TrainOnly": scaler_train.mean_,
        "Mean_AllData": scaler_all.mean_,
        "Std_TrainOnly": np.sqrt(scaler_train.var_),
        "Std_AllData": np.sqrt(scaler_all.var_),
    })
    stats["Mean_Diff"] = stats["Mean_AllData"] - stats["Mean_TrainOnly"]
    stats["Std_Diff"] = stats["Std_AllData"] - stats["Std_TrainOnly"]
    stats["Mean_Diff_%"] = (stats["Mean_Diff"] / stats["Mean_TrainOnly"].abs() * 100).round(4)
    stats["Std_Diff_%"] = (stats["Std_Diff"] / stats["Std_TrainOnly"].abs() * 100).round(4)

    print("\n" + "="*80)
    print("SCALER PARAMETER COMPARISON: Fit on Train Only vs Fit on All Data")
    print("="*80)
    print(stats.to_string(index=False))
    print()

    # Save to CSV
    stats_path = os.path.join(OUTPUT_DIR, "scaler_stats_comparison.csv")
    stats.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")
    return stats


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("="*80)
    print("  EXPERIMENT A: StandardScaler Preprocessing Leakage")
    print("="*80)

    # Load data
    print("\n[1/6] Loading and preparing data...")
    X, y = load_and_prepare(DATA_PATH)

    # Tách Clean Holdout trước
    print("[2/6] Splitting clean holdout...")
    X_exp, X_holdout, y_exp, y_holdout = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=BASE_SEED
    )
    print(f"  Experiment data: {X_exp.shape}")
    print(f"  Clean holdout:   {X_holdout.shape}")
    print(f"  Holdout fraud rate: {y_holdout.mean():.4f}")

    # So sánh scaler parameters
    print("\n[3/6] Comparing scaler parameters (Train vs All)...")
    print_scaler_stats_comparison(X_exp, y_exp)

    # Run experiments
    print(f"\n[4/6] Running {N_RUNS} experiment iterations...")
    results_df = run_all_experiments(X_exp, y_exp)

    # Save raw results
    raw_path = os.path.join(OUTPUT_DIR, "raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"  Saved raw results: {raw_path}")

    # Summarize
    print("\n[5/6] Summarizing results...")
    summary = summarize_results(results_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Mean ± Std across runs (Test Set)")
    print("="*80)
    test_summary = summary[summary["Eval_Set"] == "Test"][
        ["Pipeline", "Model", "Metric", "mean±std"]
    ]
    for pipeline in ["Correct (No Leakage)", "Leakage (Fit All)"]:
        print(f"\n--- {pipeline} ---")
        p_data = test_summary[test_summary["Pipeline"] == pipeline]
        pivot = p_data.pivot_table(
            index="Model", columns="Metric", values="mean±std", aggfunc="first"
        )
        print(pivot.to_string())

    # Delta analysis
    delta = compute_delta(summary)
    delta_path = os.path.join(OUTPUT_DIR, "delta_analysis.csv")
    delta.to_csv(delta_path, index=False)
    print(f"\n  Saved delta analysis: {delta_path}")

    print("\n" + "="*80)
    print("DELTA ANALYSIS (Test Set): Leakage - Correct")
    print("="*80)
    delta_test = delta[delta["Eval_Set"] == "Test"]
    if "Delta (Leakage - Correct)" in delta_test.columns:
        print(delta_test[["Model", "Metric", "Correct (No Leakage)",
                          "Leakage (Fit All)", "Delta (Leakage - Correct)",
                          "Delta_%"]].to_string(index=False))

    # Visualizations
    print("\n[6/6] Generating visualizations...")
    plot_comparison(summary, eval_set="Test")
    plot_comparison(summary, eval_set="Val")
    plot_delta_heatmap(delta, eval_set="Test")
    plot_confusion_matrices(X_exp, y_exp)

    # ---- Evaluate on Clean Holdout ----
    print("\n" + "="*80)
    print("CLEAN HOLDOUT EVALUATION (Generalization Check)")
    print("="*80)

    # Correct: fit scaler on X_exp train portion, evaluate on holdout
    X_train_h, _, y_train_h, _ = train_test_split(
        X_exp, y_exp, test_size=TEST_SIZE, stratify=y_exp, random_state=BASE_SEED
    )
    scaler_correct_h = StandardScaler().fit(X_train_h)
    X_holdout_correct = scaler_correct_h.transform(X_holdout)
    X_train_h_scaled = scaler_correct_h.transform(X_train_h)

    # Leakage: fit scaler on ALL (including holdout -- worst case)
    scaler_leak_h = StandardScaler().fit(X)
    X_holdout_leak = scaler_leak_h.transform(X_holdout)
    X_train_h_leak = scaler_leak_h.transform(X_train_h)

    holdout_records = []
    for model_name, model_fn in MODELS.items():
        # Correct
        m_c = model_fn()
        m_c.fit(X_train_h_scaled, y_train_h)
        metrics_c = evaluate_model(m_c, X_holdout_correct, y_holdout)

        # Leakage
        m_l = model_fn()
        m_l.fit(X_train_h_leak, y_train_h)
        metrics_l = evaluate_model(m_l, X_holdout_leak, y_holdout)

        for metric_name in metrics_c:
            holdout_records.append({
                "Model": model_name,
                "Pipeline": "Correct",
                "Metric": metric_name,
                "Value": metrics_c[metric_name],
            })
            holdout_records.append({
                "Model": model_name,
                "Pipeline": "Leakage",
                "Metric": metric_name,
                "Value": metrics_l[metric_name],
            })

    holdout_df = pd.DataFrame(holdout_records)
    holdout_path = os.path.join(OUTPUT_DIR, "holdout_results.csv")
    holdout_df.to_csv(holdout_path, index=False)

    holdout_pivot = holdout_df.pivot_table(
        index=["Model", "Metric"], columns="Pipeline", values="Value"
    )
    if "Correct" in holdout_pivot.columns and "Leakage" in holdout_pivot.columns:
        holdout_pivot["Delta"] = holdout_pivot["Leakage"] - holdout_pivot["Correct"]
    print(holdout_pivot.to_string())

    print("\n" + "="*80)
    print("  EXPERIMENT COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
