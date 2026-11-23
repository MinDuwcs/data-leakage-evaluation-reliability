"""
Module đánh giá hiệu suất mô hình (Model Evaluation).

Cung cấp:
- Đánh giá toàn diện các chỉ số cho bài toán mất cân bằng lớp:
  Average Precision (AP / PR-AUC), ROC-AUC, F1-Score, Precision, Recall, Accuracy.
- Xuất ma trận nhầm lẫn (Confusion Matrix) và phân phối dự đoán.
- Cấu trúc dữ liệu EvaluationResult lưu trữ kết quả có định kiểu.
"""

from src.evaluation.metrics import (
    EvaluationResult,
    evaluate_model,
    evaluate_predictions,
)
