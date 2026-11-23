"""
Script trực quan hóa kết quả thực nghiệm Data Leakage cho kịch bản F3 và F4.

Tạo 3 đồ thị học thuật chất lượng cao (300 DPI) lưu tại results/figures/:
1. f3_dose_response_optimism_gap.png: Đường cong phản ứng liều lượng (Dose-Response Curve)
   thể hiện sự sụp đổ hiệu năng giữa đánh giá nội bộ và triển khai thực tế khi độ phủ kiểm toán tăng.
2. f3_metric_inflation_comparison.png: So sánh mức độ thổi phồng ROC-AUC và F1 của 3 mô hình trong F3.
3. f4_target_encoding_delta.png: Biểu đồ so sánh mức độ nhạy cảm của 3 mô hình trước
   Target Encoding Leakage (giải thích tại sao Decision Tree nhạy hơn LR và RF).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_results")


def plot_f3_dose_response(f3_df: pd.DataFrame, output_dir: Path) -> None:
    """Vẽ đường cong Dose-Response thể hiện Optimism Gap của Average Precision trong F3."""
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Tính mean và std cho từng model, coverage, branch
    agg = (
        f3_df.groupby(["model", "coverage", "branch"])["average_precision"]
        .agg(["mean", "std"])
        .reset_index()
    )

    models = ["random_forest", "logistic_regression", "decision_tree"]
    model_labels = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
    }
    colors = {
        "random_forest": "#1f77b4",
        "logistic_regression": "#2ca02c",
        "decision_tree": "#ff7f0e",
    }

    coverage_pct = np.array([10, 30, 50, 80])

    for model in models:
        # Nhánh Contaminated Retrospective (Internal)
        contam_data = agg[
            (agg["model"] == model) & (agg["branch"] == "contaminated_retrospective")
        ].sort_values("coverage")
        contam_means = contam_data["mean"].values
        contam_stds = contam_data["std"].values

        # Nhánh Deployment Mismatch (Clean Holdout)
        deploy_data = agg[
            (agg["model"] == model) & (agg["branch"] == "deployment_mismatch")
        ].sort_values("coverage")
        deploy_means = deploy_data["mean"].values
        deploy_stds = deploy_data["std"].values

        # Vẽ đường Contaminated
        plt.plot(
            coverage_pct,
            contam_means,
            marker="o",
            linewidth=2.5,
            color=colors[model],
            label=f"{model_labels[model]} (Contaminated Retrospective)",
        )
        plt.fill_between(
            coverage_pct,
            contam_means - contam_stds,
            contam_means + contam_stds,
            alpha=0.15,
            color=colors[model],
        )

        # Vẽ đường Deployment Mismatch
        plt.plot(
            coverage_pct,
            deploy_means,
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color=colors[model],
            alpha=0.75,
            label=f"{model_labels[model]} (Deployment on Clean Holdout)",
        )

    # Đường tham chiếu Baseline Control AP (~0.052)
    plt.axhline(
        y=0.0492,
        color="#d62728",
        linestyle=":",
        linewidth=2,
        label="Base Fraud Prevalence (4.92%)",
    )

    plt.title(
        "F3 Post-event Feature Leakage: Dose-Response Curve & Optimism Gap",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Audit Coverage (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Average Precision (AP / PR-AUC)", fontsize=12, fontweight="bold")
    plt.xticks(coverage_pct, ["10%", "30%", "50%", "80%"], fontsize=11)
    plt.ylim(0.0, 0.50)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = output_dir / "f3_dose_response_optimism_gap.png"
    plt.savefig(out_file)
    plt.close()
    logger.info("Đã xuất biểu đồ F3 Dose-Response: %s", out_file)


def plot_f3_metric_inflation(f3_df: pd.DataFrame, output_dir: Path) -> None:
    """Vẽ biểu đồ thanh so sánh mức độ thổi phồng ROC-AUC và F1 của 3 mô hình ở mức 80% coverage."""
    f3_80 = f3_df[f3_df["coverage"] == 0.80]

    agg = (
        f3_80.groupby(["model", "branch"])[["roc_auc", "f1"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = {
        "control": "#7f7f7f",
        "contaminated_retrospective": "#d62728",
        "deployment_mismatch": "#1f77b4",
    }
    branch_labels = {
        "control": "Valid Control",
        "contaminated_retrospective": "Contaminated (Retrospective)",
        "deployment_mismatch": "Deployment (Holdout)",
    }
    agg["branch_label"] = agg["branch"].map(branch_labels)

    # Đồ thị 1: ROC-AUC
    sns.barplot(
        data=agg,
        x="model",
        y="roc_auc",
        hue="branch_label",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("ROC-AUC Comparison (Coverage = 80%)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC Score", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_xticklabels(["Decision Tree", "Logistic Regression", "Random Forest"])
    axes[0].legend(title="Evaluation Branch", fontsize=9)

    # Đồ thị 2: F1-Score
    sns.barplot(
        data=agg,
        x="model",
        y="f1",
        hue="branch_label",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("F1-Score Comparison (Coverage = 80%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("F1-Score (Threshold = 0.5)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0.0, 0.45)
    axes[1].set_xticklabels(["Decision Tree", "Logistic Regression", "Random Forest"])
    axes[1].legend(title="Evaluation Branch", fontsize=9)

    plt.tight_layout()
    out_file = output_dir / "f3_metric_inflation_comparison.png"
    plt.savefig(out_file)
    plt.close()
    logger.info("Đã xuất biểu đồ F3 Metric Inflation: %s", out_file)


def plot_f4_target_encoding(f4_df: pd.DataFrame, output_dir: Path) -> None:
    """Vẽ biểu đồ so sánh mức độ sai lệch Target Encoding Leakage trên tập Validation."""
    f4_val = f4_df[f4_df["eval_set"] == "validation"]
    agg = (
        f4_val.groupby(["model", "branch"])[["average_precision", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    branch_labels = {
        "control_valid_oof": "Control (Valid Out-of-Fold)",
        "contaminated_val_leakage": "Contaminated (Validation Leakage)",
    }

    # Format data for plotting
    plot_data = []
    for _, row in agg.iterrows():
        plot_data.append({
            "model": row[("model", "")],
            "branch": branch_labels[row[("branch", "")]],
            "ap_mean": row[("average_precision", "mean")],
            "ap_std": row[("average_precision", "std")],
            "f1_mean": row[("f1", "mean")],
            "f1_std": row[("f1", "std")],
        })
    p_df = pd.DataFrame(plot_data)

    model_names = {
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
    p_df["model_clean"] = p_df["model"].map(model_names)

    # Đồ thị 1: Average Precision
    sns.barplot(
        data=p_df,
        x="model_clean",
        y="ap_mean",
        hue="branch",
        ax=axes[0],
        palette=["#2b83ba", "#d7191c"],
    )
    axes[0].set_title("F4: Target Encoding AP on Validation Set", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Average Precision (AP)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0.045, 0.056)
    axes[0].legend(title="Methodology", fontsize=9)

    # Đồ thị 2: F1-Score
    sns.barplot(
        data=p_df,
        x="model_clean",
        y="f1_mean",
        hue="branch",
        ax=axes[1],
        palette=["#2b83ba", "#d7191c"],
    )
    axes[1].set_title("F4: Target Encoding F1 on Validation Set", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0.0, 0.015)
    axes[1].legend(title="Methodology", fontsize=9)

    plt.tight_layout()
    out_file = output_dir / "f4_target_encoding_delta.png"
    plt.savefig(out_file)
    plt.close()
    logger.info("Đã xuất biểu đồ F4 Target Encoding: %s", out_file)


def main() -> None:
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    f3_csv = results_dir / "f3_post_event_results.csv"
    f4_csv = results_dir / "f4_target_encoding_results.csv"

    if not f3_csv.exists() or not f4_csv.exists():
        logger.error("Không tìm thấy file kết quả CSV trong results/! Hãy chạy run_experiments.py trước.")
        return

    logger.info("Đọc dữ liệu kết quả từ: %s và %s", f3_csv, f4_csv)
    f3_df = pd.read_csv(f3_csv)
    f4_df = pd.read_csv(f4_csv)

    # 1. Biểu đồ F3 Dose-Response
    plot_f3_dose_response(f3_df, figures_dir)

    # 2. Biểu đồ F3 Metric Inflation
    plot_f3_metric_inflation(f3_df, figures_dir)

    # 3. Biểu đồ F4 Target Encoding
    plot_f4_target_encoding(f4_df, figures_dir)

    logger.info("HOÀN TẤT XUẤT TOÀN BỘ BIỂU ĐỒ TRỰC QUAN HÓA TẠI: %s", figures_dir)


if __name__ == "__main__":
    main()
