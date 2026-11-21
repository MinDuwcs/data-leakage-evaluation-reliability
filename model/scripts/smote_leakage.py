"""
==============================================================================
Experiment A2 – Preprocessing Leakage: SMOTE (Oversampling)
==============================================================================
So sánh hai quy trình:
  • A-Control (Correct): Split TRƯỚC → áp dụng SMOTE CHỈ trên Train
                          → train/evaluate bình thường.
  • A-Leakage (Leaked):  Áp dụng SMOTE trên TOÀN BỘ dữ liệu TRƯỚC KHI split
                          → split SAU → train/evaluate.

Leakage ở đây nghiêm trọng hơn StandardScaler vì SMOTE tạo ra các mẫu
tổng hợp (synthetic) dựa trên thông tin từ cả Train LẪN Test:
  - Các mẫu synthetic có thể nằm giữa một điểm Train và một điểm Test.
  - Test set chứa các mẫu synthetic → mô hình "nhớ" chúng → metric bị thổi phồng.
  - Phá vỡ hoàn toàn tính độc lập của tập đánh giá.

Dataset : fraud_detection.csv  (cột target: Fraudulent)
Models  : Logistic Regression, Decision Tree, Random Forest
Metrics : Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Runs    : Lặp lại N_RUNS lần với các random seed khác nhau.

Output  : Bảng metric, biểu đồ so sánh, confusion matrix, phân tích class dist.
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
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is required. Install it with: pip install imbalanced-learn"
    )

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fraud_detection.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "smote_leakage")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 10                    # Số lần lặp thí nghiệm
TEST_SIZE = 0.2                # Tỷ lệ tập Test
VAL_SIZE = 0.15                # Tỷ lệ Validation (tính trên phần còn lại)
HOLDOUT_SIZE = 0.1             # Clean holdout
BASE_SEED = 42
TARGET_COL = "Fraudulent"
SMOTE_SAMPLING_STRATEGY = 0.5  # Tỷ lệ minority/majority sau SMOTE (0.5 = 50%)

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

    # Drop ID columns
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

    # Fill missing values
    X = X.fillna(X.median())

    print(f"Features: {list(X.columns)}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Original class distribution: {dict(y.value_counts())}\n")
    return X, y


def evaluate_model(model, X_eval, y_eval) -> dict:
    """Tính toán metrics."""
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
def run_correct_pipeline(X, y, seed: int) -> tuple[dict, dict]:
    """
    A-Control: Quy trình ĐÚNG
    1. Split trước (Train/Val/Test)
    2. Scale features (fit trên Train)
    3. Áp dụng SMOTE CHỈ trên Train
    4. Train model → Evaluate trên Val/Test GỐC (không synthetic)
    """
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    # Scale (fit trên Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE CHỈ trên Train
    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    class_info = {
        "train_before_smote": dict(pd.Series(y_train).value_counts()),
        "train_after_smote": dict(pd.Series(y_train_resampled).value_counts()),
        "val": dict(pd.Series(y_val).value_counts()),
        "test": dict(pd.Series(y_test).value_counts()),
        "train_size_before": len(y_train),
        "train_size_after": len(y_train_resampled),
    }

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train_resampled, y_train_resampled)
        results[name] = {
            "val": evaluate_model(model, X_val_scaled, y_val),
            "test": evaluate_model(model, X_test_scaled, y_test),
        }
    return results, class_info


def run_leakage_pipeline(X, y, seed: int) -> tuple[dict, dict]:
    """
    A-Leakage: Quy trình có DATA LEAKAGE
    1. Scale toàn bộ (leakage nhẹ, nhưng đây không phải focus)
    2. Áp dụng SMOTE trên TOÀN BỘ dữ liệu TRƯỚC KHI split (LEAKAGE NGHIÊM TRỌNG!)
    3. Split SAU → Train/Val/Test đều chứa synthetic samples
    4. Test set chứa synthetic → metric bị thổi phồng
    """
    # Scale toàn bộ
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)

    # SMOTE trên TOÀN BỘ trước khi split (LEAKAGE!)
    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_scaled_all, y)

    class_info = {
        "before_smote": dict(pd.Series(y).value_counts()),
        "after_smote": dict(pd.Series(y_resampled).value_counts()),
        "total_before": len(y),
        "total_after": len(y_resampled),
    }

    # Split SAU khi đã SMOTE (tất cả tập đều chứa synthetic samples)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE,
        stratify=y_resampled, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    class_info.update({
        "train_leaked": dict(pd.Series(y_train).value_counts()),
        "val_leaked": dict(pd.Series(y_val).value_counts()),
        "test_leaked": dict(pd.Series(y_test).value_counts()),
    })

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train, y_train)
        results[name] = {
            "val": evaluate_model(model, X_val, y_val),
            "test": evaluate_model(model, X_test, y_test),
        }
    return results, class_info


# ============================================================================
# 4. CLEAN TEST EVALUATION (Generalization Check for Leakage Pipeline)
# ============================================================================
def run_leakage_with_clean_test(X, y, seed: int) -> dict:
    """
    Đánh giá mô hình train bằng pipeline Leakage trên dữ liệu Test GỐC (clean).
    Mục đích: cho thấy metric trên synthetic test bị thổi phồng,
              còn khi đánh giá trên dữ liệu thật thì hiệu suất thực sự thấp hơn.
    """
    # Giữ lại bản gốc chưa SMOTE để test
    X_temp_orig, X_test_clean, y_temp_orig, y_test_clean = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    # Scale toàn bộ (leakage)
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)

    # SMOTE toàn bộ
    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_scaled_all, y)

    # Split SMOTE data
    X_temp_smote, _, y_temp_smote, _ = train_test_split(
        X_resampled, y_resampled, test_size=TEST_SIZE,
        stratify=y_resampled, random_state=seed
    )
    X_train_smote, _, y_train_smote, _ = train_test_split(
        X_temp_smote, y_temp_smote, test_size=VAL_SIZE,
        stratify=y_temp_smote, random_state=seed
    )

    # Scale clean test using same leakage scaler
    X_test_clean_scaled = scaler.transform(X_test_clean)

    results = {}
    for name, model_fn in MODELS.items():
        model = model_fn()
        model.fit(X_train_smote, y_train_smote)
        results[name] = evaluate_model(model, X_test_clean_scaled, y_test_clean)
    return results


# ============================================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================================
def run_all_experiments(X, y) -> tuple[pd.DataFrame, list, list]:
    """Chạy N_RUNS lần cho cả 2 pipeline."""
    all_records = []
    correct_class_infos = []
    leakage_class_infos = []

    for run_idx in range(N_RUNS):
        seed = BASE_SEED + run_idx
        print(f"  Run {run_idx + 1}/{N_RUNS} (seed={seed})")

        correct_results, correct_ci = run_correct_pipeline(X, y, seed)
        leakage_results, leakage_ci = run_leakage_pipeline(X, y, seed)
        clean_test_results = run_leakage_with_clean_test(X, y, seed)

        correct_class_infos.append(correct_ci)
        leakage_class_infos.append(leakage_ci)

        for model_name in MODELS:
            for eval_set in ["val", "test"]:
                for metric_name in correct_results[model_name][eval_set]:
                    # Correct pipeline
                    all_records.append({
                        "Run": run_idx + 1,
                        "Seed": seed,
                        "Pipeline": "Correct (SMOTE Train Only)",
                        "Model": model_name,
                        "Eval_Set": eval_set.capitalize(),
                        "Metric": metric_name,
                        "Value": correct_results[model_name][eval_set][metric_name],
                    })
                    # Leakage pipeline (evaluated on synthetic test)
                    all_records.append({
                        "Run": run_idx + 1,
                        "Seed": seed,
                        "Pipeline": "Leakage (SMOTE Before Split)",
                        "Model": model_name,
                        "Eval_Set": eval_set.capitalize(),
                        "Metric": metric_name,
                        "Value": leakage_results[model_name][eval_set][metric_name],
                    })

            # Leakage pipeline evaluated on CLEAN test
            for metric_name in clean_test_results[model_name]:
                all_records.append({
                    "Run": run_idx + 1,
                    "Seed": seed,
                    "Pipeline": "Leakage → Clean Test",
                    "Model": model_name,
                    "Eval_Set": "Test",
                    "Metric": metric_name,
                    "Value": clean_test_results[model_name][metric_name],
                })

    return pd.DataFrame(all_records), correct_class_infos, leakage_class_infos


# ============================================================================
# 6. ANALYSIS & VISUALIZATION
# ============================================================================
def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
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
    """Tính chênh lệch giữa các pipeline."""
    records = []
    test_data = summary[summary["Eval_Set"] == "Test"]

    for model in test_data["Model"].unique():
        for metric in test_data["Metric"].unique():
            m_data = test_data[(test_data["Model"] == model) & (test_data["Metric"] == metric)]

            correct_val = m_data[m_data["Pipeline"] == "Correct (SMOTE Train Only)"]["mean"].values
            leakage_val = m_data[m_data["Pipeline"] == "Leakage (SMOTE Before Split)"]["mean"].values
            clean_test_val = m_data[m_data["Pipeline"] == "Leakage → Clean Test"]["mean"].values

            if len(correct_val) > 0 and len(leakage_val) > 0:
                records.append({
                    "Model": model,
                    "Metric": metric,
                    "Correct": correct_val[0],
                    "Leakage_SyntheticTest": leakage_val[0],
                    "Leakage_CleanTest": clean_test_val[0] if len(clean_test_val) > 0 else None,
                    "Delta_Leakage_vs_Correct": leakage_val[0] - correct_val[0],
                    "Delta_%": (leakage_val[0] - correct_val[0]) / abs(correct_val[0]) * 100 if correct_val[0] != 0 else 0,
                    "Inflation_vs_CleanTest": (leakage_val[0] - clean_test_val[0]) if len(clean_test_val) > 0 else None,
                })

    return pd.DataFrame(records)


def plot_comparison(summary: pd.DataFrame, eval_set: str = "Test"):
    """Bar chart so sánh 3 pipeline."""
    df_filtered = summary[summary["Eval_Set"] == eval_set]
    if df_filtered.empty:
        return

    metrics = sorted(df_filtered["Metric"].unique())
    models = sorted(df_filtered["Model"].unique())
    pipelines = sorted(df_filtered["Pipeline"].unique())
    colors = {"Correct (SMOTE Train Only)": "#2ecc71",
              "Leakage (SMOTE Before Split)": "#e74c3c",
              "Leakage → Clean Test": "#3498db"}

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f"SMOTE Leakage: Pipeline Comparison ({eval_set} Set)",
        fontsize=16, fontweight="bold",
    )

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        metric_data = df_filtered[df_filtered["Metric"] == metric]

        x = np.arange(len(models))
        n_bars = len(pipelines)
        width = 0.8 / n_bars

        for bar_idx, pipeline in enumerate(pipelines):
            if pipeline not in metric_data["Pipeline"].values:
                continue
            vals = []
            stds = []
            for model in models:
                row = metric_data[
                    (metric_data["Model"] == model) &
                    (metric_data["Pipeline"] == pipeline)
                ]
                vals.append(row["mean"].values[0] if len(row) > 0 else 0)
                stds.append(row["std"].values[0] if len(row) > 0 else 0)

            offset = (bar_idx - n_bars/2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width, yerr=stds,
                label=pipeline, color=colors.get(pipeline, f"C{bar_idx}"),
                alpha=0.85, capsize=2,
            )

            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, f"smote_leakage_comparison_{eval_set.lower()}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_delta_heatmap(delta_df: pd.DataFrame):
    """Heatmap chênh lệch."""
    if delta_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Delta %: Leakage vs Correct
    pivot1 = delta_df.pivot_table(index="Model", columns="Metric", values="Delta_%")
    sns.heatmap(pivot1, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0,
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "Delta (%)"})
    axes[0].set_title("Delta (%): Leakage(Synthetic Test) vs Correct", fontsize=11, fontweight="bold")

    # Inflation: Leakage Synthetic Test vs Leakage Clean Test
    if "Inflation_vs_CleanTest" in delta_df.columns:
        pivot2 = delta_df.pivot_table(index="Model", columns="Metric", values="Inflation_vs_CleanTest")
        sns.heatmap(pivot2, annot=True, fmt=".4f", cmap="RdYlGn_r", center=0,
                    linewidths=0.5, ax=axes[1], cbar_kws={"label": "Inflation"})
        axes[1].set_title("Metric Inflation: Synthetic Test vs Clean Test", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "smote_leakage_delta_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_class_distribution(correct_infos: list, leakage_infos: list):
    """Biểu đồ so sánh phân phối class giữa 2 pipeline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution: Correct vs Leakage Pipeline", fontsize=14, fontweight="bold")

    # Correct pipeline
    ci = correct_infos[0]
    labels_c = ["Train\n(Before SMOTE)", "Train\n(After SMOTE)", "Val", "Test"]
    class0_c = [
        ci["train_before_smote"].get(0, 0),
        ci["train_after_smote"].get(0, 0),
        ci["val"].get(0, 0),
        ci["test"].get(0, 0),
    ]
    class1_c = [
        ci["train_before_smote"].get(1, 0),
        ci["train_after_smote"].get(1, 0),
        ci["val"].get(1, 0),
        ci["test"].get(1, 0),
    ]

    x = np.arange(len(labels_c))
    axes[0].bar(x - 0.2, class0_c, 0.4, label="Class 0 (Legit)", color="#2ecc71")
    axes[0].bar(x + 0.2, class1_c, 0.4, label="Class 1 (Fraud)", color="#e74c3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_c)
    axes[0].set_title("Correct Pipeline\n(SMOTE on Train Only)", fontweight="bold")
    axes[0].legend()
    axes[0].set_ylabel("Count")

    for i, (c0, c1) in enumerate(zip(class0_c, class1_c)):
        axes[0].annotate(f"{c0}", (i - 0.2, c0), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=8)
        axes[0].annotate(f"{c1}", (i + 0.2, c1), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=8)

    # Leakage pipeline
    li = leakage_infos[0]
    labels_l = ["Before\nSMOTE", "After SMOTE\n(All Data)", "Train\n(Leaked)", "Val\n(Leaked)", "Test\n(Leaked)"]
    class0_l = [
        li["before_smote"].get(0, 0),
        li["after_smote"].get(0, 0),
        li.get("train_leaked", {}).get(0, 0),
        li.get("val_leaked", {}).get(0, 0),
        li.get("test_leaked", {}).get(0, 0),
    ]
    class1_l = [
        li["before_smote"].get(1, 0),
        li["after_smote"].get(1, 0),
        li.get("train_leaked", {}).get(1, 0),
        li.get("val_leaked", {}).get(1, 0),
        li.get("test_leaked", {}).get(1, 0),
    ]

    x = np.arange(len(labels_l))
    axes[1].bar(x - 0.2, class0_l, 0.4, label="Class 0 (Legit)", color="#2ecc71")
    axes[1].bar(x + 0.2, class1_l, 0.4, label="Class 1 (Fraud)", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_l, fontsize=8)
    axes[1].set_title("Leakage Pipeline\n(SMOTE on ALL Before Split)", fontweight="bold")
    axes[1].legend()
    axes[1].set_ylabel("Count")

    for i, (c0, c1) in enumerate(zip(class0_l, class1_l)):
        axes[1].annotate(f"{c0}", (i - 0.2, c0), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7)
        axes[1].annotate(f"{c1}", (i + 0.2, c1), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(OUTPUT_DIR, "smote_class_distribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_confusion_matrices(X, y, seed: int = BASE_SEED):
    """Plot confusion matrix cho cả 2 pipeline trên Test set."""
    # --- Correct pipeline ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=seed
    )

    scaler_c = StandardScaler()
    X_train_sc = scaler_c.fit_transform(X_train)
    X_test_sc = scaler_c.transform(X_test)

    smote_c = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=seed)
    X_train_res, y_train_res = smote_c.fit_resample(X_train_sc, y_train)

    # --- Leakage pipeline ---
    scaler_l = StandardScaler()
    X_scaled_all = scaler_l.fit_transform(X)
    smote_l = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=seed)
    X_res_all, y_res_all = smote_l.fit_resample(X_scaled_all, y)

    X_temp_l, X_test_l, y_temp_l, y_test_l = train_test_split(
        X_res_all, y_res_all, test_size=TEST_SIZE, stratify=y_res_all, random_state=seed
    )
    X_train_l, _, y_train_l, _ = train_test_split(
        X_temp_l, y_temp_l, test_size=VAL_SIZE, stratify=y_temp_l, random_state=seed
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SMOTE Leakage: Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")

    for col_idx, (model_name, model_fn) in enumerate(MODELS.items()):
        # Correct
        m_c = model_fn()
        m_c.fit(X_train_res, y_train_res)
        y_pred_c = m_c.predict(X_test_sc)
        cm_c = confusion_matrix(y_test, y_pred_c)

        sns.heatmap(cm_c, annot=True, fmt="d", cmap="Blues", ax=axes[0, col_idx])
        axes[0, col_idx].set_title(f"{model_name}\n(Correct - Real Test)", fontsize=10)
        axes[0, col_idx].set_xlabel("Predicted")
        axes[0, col_idx].set_ylabel("Actual")

        # Leakage (synthetic test)
        m_l = model_fn()
        m_l.fit(X_train_l, y_train_l)
        y_pred_l = m_l.predict(X_test_l)
        cm_l = confusion_matrix(y_test_l, y_pred_l)

        sns.heatmap(cm_l, annot=True, fmt="d", cmap="Reds", ax=axes[1, col_idx])
        axes[1, col_idx].set_title(f"{model_name}\n(Leakage - Synthetic Test)", fontsize=10)
        axes[1, col_idx].set_xlabel("Predicted")
        axes[1, col_idx].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(OUTPUT_DIR, "smote_confusion_matrices.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("="*80)
    print("  EXPERIMENT A2: SMOTE Preprocessing Leakage")
    print("="*80)

    # Load data
    print("\n[1/7] Loading and preparing data...")
    X, y = load_and_prepare(DATA_PATH)

    # Clean holdout
    print("[2/7] Splitting clean holdout...")
    X_exp, X_holdout, y_exp, y_holdout = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=BASE_SEED
    )
    print(f"  Experiment data: {X_exp.shape}")
    print(f"  Clean holdout:   {X_holdout.shape}")
    print(f"  Holdout fraud rate: {y_holdout.mean():.4f}")

    # Run experiments
    print(f"\n[3/7] Running {N_RUNS} experiment iterations...")
    results_df, correct_cis, leakage_cis = run_all_experiments(X_exp, y_exp)

    # Save raw results
    raw_path = os.path.join(OUTPUT_DIR, "raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"  Saved raw results: {raw_path}")

    # Summarize
    print("\n[4/7] Summarizing results...")
    summary = summarize_results(results_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary.to_csv(summary_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Mean ± Std across runs (Test Set)")
    print("="*80)
    test_summary = summary[summary["Eval_Set"] == "Test"]
    for pipeline in sorted(test_summary["Pipeline"].unique()):
        print(f"\n--- {pipeline} ---")
        p_data = test_summary[test_summary["Pipeline"] == pipeline][
            ["Model", "Metric", "mean±std"]
        ]
        pivot = p_data.pivot_table(index="Model", columns="Metric", values="mean±std", aggfunc="first")
        print(pivot.to_string())

    # Delta analysis
    print("\n[5/7] Computing delta analysis...")
    delta = compute_delta(summary)
    delta_path = os.path.join(OUTPUT_DIR, "delta_analysis.csv")
    delta.to_csv(delta_path, index=False)

    print("\n" + "="*80)
    print("DELTA ANALYSIS (Test Set)")
    print("="*80)
    print(delta.to_string(index=False))

    # Class distribution info
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION (Run 1)")
    print("="*80)
    print("\n--- Correct Pipeline ---")
    for key, val in correct_cis[0].items():
        print(f"  {key}: {val}")
    print("\n--- Leakage Pipeline ---")
    for key, val in leakage_cis[0].items():
        print(f"  {key}: {val}")

    # Visualizations
    print(f"\n[6/7] Generating visualizations...")
    plot_comparison(summary, eval_set="Test")
    plot_comparison(summary, eval_set="Val")
    plot_delta_heatmap(delta)
    plot_class_distribution(correct_cis, leakage_cis)
    plot_confusion_matrices(X_exp, y_exp)

    # ---- Clean Holdout Evaluation ----
    print("\n[7/7] Clean Holdout evaluation...")
    print("\n" + "="*80)
    print("CLEAN HOLDOUT EVALUATION (Generalization Check)")
    print("="*80)

    # Correct pipeline on holdout
    X_train_h, _, y_train_h, _ = train_test_split(
        X_exp, y_exp, test_size=TEST_SIZE, stratify=y_exp, random_state=BASE_SEED
    )
    scaler_h = StandardScaler().fit(X_train_h)
    X_train_h_sc = scaler_h.transform(X_train_h)
    X_holdout_sc = scaler_h.transform(X_holdout)

    smote_h = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=BASE_SEED)
    X_train_h_res, y_train_h_res = smote_h.fit_resample(X_train_h_sc, y_train_h)

    # Leakage pipeline on holdout
    scaler_h_l = StandardScaler().fit(X)
    X_all_sc = scaler_h_l.transform(X)
    smote_h_l = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=BASE_SEED)
    X_all_res, y_all_res = smote_h_l.fit_resample(X_all_sc, y)

    # Use all resampled data as "training" for the leakage model
    # then evaluate on holdout
    X_holdout_l_sc = scaler_h_l.transform(X_holdout)

    holdout_records = []
    for model_name, model_fn in MODELS.items():
        # Correct
        m_c = model_fn()
        m_c.fit(X_train_h_res, y_train_h_res)
        metrics_c = evaluate_model(m_c, X_holdout_sc, y_holdout)

        # Leakage (trained on all resampled data, evaluated on clean holdout)
        m_l = model_fn()
        m_l.fit(X_all_res, y_all_res)
        metrics_l = evaluate_model(m_l, X_holdout_l_sc, y_holdout)

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
