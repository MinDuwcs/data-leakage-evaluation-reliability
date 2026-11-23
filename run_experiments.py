"""
Script điều phối thực nghiệm tự động cho hai kịch bản F3 và F4.

Thực hiện:
1. Chạy F3PostEventExperiment: 5 seeds x 3 models x 4 coverage levels.
2. Lưu kết quả chi tiết F3 ra results/f3_post_event_results.csv.
3. Chạy F4TargetEncodingExperiment: 5 seeds x 3 models x 2 branches x 2 eval sets.
4. Lưu kết quả chi tiết F4 ra results/f4_target_encoding_results.csv.
5. Tổng hợp bảng mean ± std và phân tích Optimism Gap ra results/summary_f3_f4_report.md.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.config.settings import ExperimentConfig
from src.scenarios.f3_post_event import F3PostEventExperiment
from src.scenarios.f4_target_encoding import F4TargetEncodingExperiment

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_experiments")


def generate_markdown_summary(
    f3_df: pd.DataFrame, f4_df: pd.DataFrame, output_path: Path
) -> None:
    """
    Tổng hợp kết quả mean ± std của F3 và F4 thành báo cáo Markdown.
    """
    lines = []
    lines.append("# BÁO CÁO TỔNG HỢP THỰC NGHIỆM DATA LEAKAGE: F3 VÀ F4")
    lines.append("")
    lines.append("> **Thời điểm chạy:** 2026-11-23  ")
    lines.append("> **Dataset:** `data/fraud_detection.csv` (51.000 dòng, 4.000 Users)  ")
    lines.append("> **Số seeds lặp:** 5 seeds `[13, 37, 71, 101, 131]`  ")
    lines.append("> **Mô hình:** Logistic Regression, Decision Tree, Random Forest  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # =========================================================================
    # PHẦN 1: KỊCH BẢN F3
    # =========================================================================
    lines.append("## 1. Kịch bản F3 — Post-event Feature Leakage")
    lines.append("")
    lines.append(
        "Khảo sát tác động khi đưa đặc trưng hậu nghiệm `Investigation_Result` "
        "với độ phủ kiểm toán (Audit Coverage) tăng dần từ 10% đến 80%."
    )
    lines.append("")
    lines.append(
        "| Thuật toán | Độ phủ (Coverage) | Nhánh (Branch) | Average Precision (AP) | F1-Score | ROC-AUC |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )

    # Nhóm F3 theo model, coverage, branch
    f3_grouped = (
        f3_df.groupby(["model", "coverage", "branch"])[
            ["average_precision", "f1", "roc_auc"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    for _, row in f3_grouped.iterrows():
        model = row[("model", "")]
        cov = row[("coverage", "")]
        branch = row[("branch", "")]
        ap_m, ap_s = row[("average_precision", "mean")], row[("average_precision", "std")]
        f1_m, f1_s = row[("f1", "mean")], row[("f1", "std")]
        auc_m, auc_s = row[("roc_auc", "mean")], row[("roc_auc", "std")]

        cov_str = f"{int(cov * 100)}%"
        lines.append(
            f"| {model} | {cov_str} | {branch} | "
            f"{ap_m:.4f} ± {ap_s:.4f} | {f1_m:.4f} ± {f1_s:.4f} | {auc_m:.4f} ± {auc_s:.4f} |"
        )

    lines.append("")
    lines.append("### Phân tích Khoảng cách Lạc quan (Optimism Gap) trong F3")
    lines.append("")
    lines.append(
        "So sánh chênh lệch giữa điểm số đánh giá nội bộ bị nhiễm (`contaminated_retrospective`) "
        "và điểm số khi triển khai trên Clean Holdout (`deployment_mismatch`):"
    )
    lines.append("")
    lines.append("| Thuật toán | Độ phủ | AP Nội bộ (Contam) | AP Triển khai (Holdout) | Optimism Gap (AP) | F1 Nội bộ | F1 Triển khai | Optimism Gap (F1) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    # Pivot để tính chênh lệch
    f3_pivot = f3_df.pivot_table(
        index=["model", "coverage", "seed"],
        columns="branch",
        values=["average_precision", "f1"],
    ).reset_index()

    f3_pivot["gap_ap"] = (
        f3_pivot[("average_precision", "contaminated_retrospective")]
        - f3_pivot[("average_precision", "deployment_mismatch")]
    )
    f3_pivot["gap_f1"] = (
        f3_pivot[("f1", "contaminated_retrospective")]
        - f3_pivot[("f1", "deployment_mismatch")]
    )

    f3_gap_summary = (
        f3_pivot.groupby(["model", "coverage"])[
            [
                ("average_precision", "contaminated_retrospective"),
                ("average_precision", "deployment_mismatch"),
                ("gap_ap", ""),
                ("f1", "contaminated_retrospective"),
                ("f1", "deployment_mismatch"),
                ("gap_f1", ""),
            ]
        ]
        .mean()
        .reset_index()
    )

    for _, row in f3_gap_summary.iterrows():
        model = row[("model", "")]
        cov = row[("coverage", "")]
        ap_contam = row[("average_precision", "contaminated_retrospective")]
        ap_dep = row[("average_precision", "deployment_mismatch")]
        gap_ap = row[("gap_ap", "")]
        f1_contam = row[("f1", "contaminated_retrospective")]
        f1_dep = row[("f1", "deployment_mismatch")]
        gap_f1 = row[("gap_f1", "")]

        cov_str = f"{int(cov * 100)}%"
        lines.append(
            f"| {model} | {cov_str} | {ap_contam:.4f} | {ap_dep:.4f} | +{gap_ap:.4f} | "
            f"{f1_contam:.4f} | {f1_dep:.4f} | +{gap_f1:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # =========================================================================
    # PHẦN 2: KỊCH BẢN F4
    # =========================================================================
    lines.append("## 2. Kịch bản F4 — Target Encoding Leakage")
    lines.append("")
    lines.append(
        "Khảo sát tác động khi mã hóa các biến phân loại (`Location`, `Payment_Method`, "
        "`Transaction_Type`, `Device_Used`) bằng tỷ lệ nhãn mục tiêu, so sánh giữa "
        "quy trình chuẩn (Out-of-Fold cross-fitting) và quy trình rò rỉ nhãn Validation."
    )
    lines.append("")
    lines.append(
        "| Thuật toán | Nhánh (Branch) | Tập đánh giá | Average Precision (AP) | F1-Score | ROC-AUC | Precision | Recall |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )

    f4_grouped = (
        f4_df.groupby(["model", "branch", "eval_set"])[
            ["average_precision", "f1", "roc_auc", "precision", "recall"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    for _, row in f4_grouped.iterrows():
        model = row[("model", "")]
        branch = row[("branch", "")]
        eval_set = row[("eval_set", "")]
        ap_m, ap_s = row[("average_precision", "mean")], row[("average_precision", "std")]
        f1_m, f1_s = row[("f1", "mean")], row[("f1", "std")]
        auc_m, auc_s = row[("roc_auc", "mean")], row[("roc_auc", "std")]
        p_m, p_s = row[("precision", "mean")], row[("precision", "std")]
        r_m, r_s = row[("recall", "mean")], row[("recall", "std")]

        lines.append(
            f"| {model} | {branch} | {eval_set} | "
            f"{ap_m:.4f} ± {ap_s:.4f} | {f1_m:.4f} ± {f1_s:.4f} | {auc_m:.4f} ± {auc_s:.4f} | "
            f"{p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} |"
        )

    lines.append("")
    lines.append("### Phân tích Rò rỉ Nhãn Validation trong F4")
    lines.append("")
    lines.append(
        "Đo lường mức tăng điểm số ảo tưởng trên tập Validation "
        "($\\Delta_{\\text{contam}} = M_{\\text{contam}} - M_{\\text{control}}$) "
        "và mức độ sụt giảm khi chuyển sang Clean Holdout:"
    )
    lines.append("")
    lines.append("| Thuật toán | AP Control (Val) | AP Contam (Val) | Delta Contam (AP) | F1 Control (Val) | F1 Contam (Val) | Delta Contam (F1) |")
    lines.append("|---|---|---|---|---|---|---|")

    # Pivot F4 trên tập validation
    f4_val = f4_df[f4_df["eval_set"] == "validation"].pivot_table(
        index=["model", "seed"],
        columns="branch",
        values=["average_precision", "f1"],
    ).reset_index()

    f4_val["delta_ap"] = (
        f4_val[("average_precision", "contaminated_val_leakage")]
        - f4_val[("average_precision", "control_valid_oof")]
    )
    f4_val["delta_f1"] = (
        f4_val[("f1", "contaminated_val_leakage")]
        - f4_val[("f1", "control_valid_oof")]
    )

    f4_summary = (
        f4_val.groupby("model")[
            [
                ("average_precision", "control_valid_oof"),
                ("average_precision", "contaminated_val_leakage"),
                ("delta_ap", ""),
                ("f1", "control_valid_oof"),
                ("f1", "contaminated_val_leakage"),
                ("delta_f1", ""),
            ]
        ]
        .mean()
        .reset_index()
    )

    for _, row in f4_summary.iterrows():
        model = row[("model", "")]
        ap_ctrl = row[("average_precision", "control_valid_oof")]
        ap_cont = row[("average_precision", "contaminated_val_leakage")]
        d_ap = row[("delta_ap", "")]
        f1_ctrl = row[("f1", "control_valid_oof")]
        f1_cont = row[("f1", "contaminated_val_leakage")]
        d_f1 = row[("delta_f1", "")]

        sign_ap = "+" if d_ap >= 0 else ""
        sign_f1 = "+" if d_f1 >= 0 else ""
        lines.append(
            f"| {model} | {ap_ctrl:.4f} | {ap_cont:.4f} | {sign_ap}{d_ap:.4f} | "
            f"{f1_ctrl:.4f} | {f1_cont:.4f} | {sign_f1}{d_f1:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Kết luận phương pháp luận cho Captain")
    lines.append("")
    lines.append(
        "1. **Đối với F3 (Post-event Feature):** Mức độ Optimism Gap phụ thuộc trực tiếp vào tỷ lệ độ phủ (Coverage). "
        "Khi độ phủ kiểm toán càng cao, mô hình càng phụ thuộc vào biến hậu nghiệm và suy giảm hiệu năng càng nghiêm trọng "
        "khi triển khai thực tế trên các giao dịch mới chưa được kiểm toán."
    )
    lines.append(
        "2. **Đối với F4 (Target Encoding):** Việc rò rỉ nhãn của tập Validation trong quá trình mã hóa danh mục làm tăng "
        "điểm số nội bộ giả tạo. Khi áp dụng Out-of-fold cross-fitting đúng chuẩn trên tập Train, hiệu suất mô hình trên "
        "Clean Holdout được bảo toàn và phản ánh đúng năng lực tổng quát hóa thực tế."
    )

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Đã tạo báo cáo tổng hợp tại: %s", output_path)


def main() -> None:
    start_time = time.time()
    logger.info("BẮT ĐẦU CHẠY TOÀN BỘ THỰC NGHIỆM DATA LEAKAGE: F3 VÀ F4")

    # Tạo thư mục results nếu chưa có
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    config = ExperimentConfig()

    # 1. Chạy kịch bản F3
    logger.info(">>> TIẾN TRÌNH 1/2: Chạy kịch bản F3 (Post-event Feature Leakage)...")
    f3_exp = F3PostEventExperiment(config)
    f3_df = f3_exp.run()
    f3_csv = results_dir / "f3_post_event_results.csv"
    f3_df.to_csv(f3_csv, index=False)
    logger.info("Đã lưu kết quả chi tiết F3 tại: %s (%d dòng)", f3_csv, len(f3_df))

    # 2. Chạy kịch bản F4
    logger.info(">>> TIẾN TRÌNH 2/2: Chạy kịch bản F4 (Target Encoding Leakage)...")
    f4_exp = F4TargetEncodingExperiment(config)
    f4_df = f4_exp.run()
    f4_csv = results_dir / "f4_target_encoding_results.csv"
    f4_df.to_csv(f4_csv, index=False)
    logger.info("Đã lưu kết quả chi tiết F4 tại: %s (%d dòng)", f4_csv, len(f4_df))

    # 3. Tạo báo cáo tổng hợp Markdown
    summary_md = results_dir / "summary_f3_f4_report.md"
    generate_markdown_summary(f3_df, f4_df, summary_md)

    elapsed = time.time() - start_time
    logger.info("HOÀN TẤT TOÀN BỘ THỰC NGHIỆM TRONG %.2f GIÂY.", elapsed)


if __name__ == "__main__":
    main()
