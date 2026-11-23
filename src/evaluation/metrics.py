"""
Module tính toán các chỉ số đánh giá thực nghiệm (Evaluation Metrics).

Chịu trách nhiệm:
1. Tính toán các chỉ số phân loại nhị phân trên dữ liệu mất cân bằng (Fraud Detection).
2. Primary metrics: Average Precision (AP / PR-AUC), F1-Score, Recall, Precision.
3. Secondary metrics: ROC-AUC, Accuracy, Confusion Matrix (TN, FP, FN, TP).
4. Xử lý ngưỡng quyết định (Classification Threshold) và các trường hợp biên.

Tham chiếu: MarkDown/experimental_protocol_data_leakage_v2.md (Mục 6.2)
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Cấu trúc dữ liệu lưu trữ toàn diện kết quả đánh giá một lần chạy."""

    # Chỉ số phân biệt theo ngưỡng xác suất (Threshold-free Discrimination)
    average_precision: float
    roc_auc: float

    # Chỉ số dựa trên ngưỡng quyết định (Threshold-based)
    f1: float
    precision: float
    recall: float
    accuracy: float

    # Chi tiết ma trận nhầm lẫn (Confusion Matrix)
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    # Thống kê phân phối (Prevalences)
    threshold: float
    total_samples: int
    actual_positives: int
    predicted_positives: int
    actual_prevalence: float
    predicted_prevalence: float

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển phẳng để dễ dàng xuất DataFrame hoặc ghi log."""
        return asdict(self)


def evaluate_predictions(
    y_true: Union[np.ndarray, list],
    y_proba: Union[np.ndarray, list],
    threshold: float = 0.5,
    pos_label: int = 1,
) -> EvaluationResult:
    """
    Tính toán toàn bộ các chỉ số đánh giá từ nhãn thực tế và xác suất dự đoán.

    Args:
        y_true: Mảng nhãn thực tế (0 hoặc 1).
        y_proba: Mảng xác suất dự đoán nhãn dương (P(y=1)).
        threshold: Ngưỡng quyết định để gán nhãn dương (mặc định 0.5).
        pos_label: Giá trị nhãn dương (mặc định 1).

    Returns:
        Đối tượng EvaluationResult chứa đầy đủ các chỉ số.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if len(y_true_arr) != len(y_proba_arr):
        raise ValueError("Độ dài y_true và y_proba không khớp nhau!")

    total_samples = len(y_true_arr)
    actual_positives = int(np.sum(y_true_arr == pos_label))
    actual_prevalence = actual_positives / total_samples if total_samples > 0 else 0.0

    # Chuyển xác suất sang nhãn dự đoán nhị phân theo ngưỡng
    y_pred = (y_proba_arr >= threshold).astype(int)
    predicted_positives = int(np.sum(y_pred == pos_label))
    predicted_prevalence = (
        predicted_positives / total_samples if total_samples > 0 else 0.0
    )

    # Xử lý trường hợp biên: chỉ có 1 lớp duy nhất trong tập đánh giá
    unique_classes = np.unique(y_true_arr)
    has_both_classes = len(unique_classes) > 1

    # 1. Average Precision (PR-AUC)
    if has_both_classes:
        ap = float(average_precision_score(y_true_arr, y_proba_arr, pos_label=pos_label))
        roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    else:
        logger.warning(
            "Tập đánh giá chỉ có duy nhất 1 lớp %s. ROC-AUC và AP được gán NaN.",
            unique_classes,
        )
        ap = float("nan")
        roc_auc = float("nan")

    # 2. Threshold-based metrics
    f1 = float(f1_score(y_true_arr, y_pred, pos_label=pos_label, zero_division=0))
    prec = float(precision_score(y_true_arr, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, pos_label=pos_label, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return EvaluationResult(
        average_precision=ap,
        roc_auc=roc_auc,
        f1=f1,
        precision=prec,
        recall=rec,
        accuracy=acc,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        threshold=threshold,
        total_samples=total_samples,
        actual_positives=actual_positives,
        predicted_positives=predicted_positives,
        actual_prevalence=actual_prevalence,
        predicted_prevalence=predicted_prevalence,
    )


def evaluate_model(
    model: ClassifierMixin,
    X_eval: Any,
    y_eval: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    Tiện ích đánh giá trực tiếp một mô hình scikit-learn (hoặc Pipeline) trên tập kiểm thử.

    Tự động trích xuất xác suất dự đoán thông qua predict_proba hoặc decision_function.

    Args:
        model: Mô hình phân loại hoặc Pipeline đã được fit.
        X_eval: Ma trận đặc trưng tập đánh giá.
        y_eval: Mảng nhãn thực tế tập đánh giá.
        threshold: Ngưỡng quyết định (mặc định 0.5).

    Returns:
        EvaluationResult chứa toàn bộ chỉ số thực nghiệm.
    """
    if hasattr(model, "predict_proba"):
        # Trích xuất xác suất của positive class (cột index 1)
        proba = model.predict_proba(X_eval)
        if proba.shape[1] == 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba[:, 0]
    elif hasattr(model, "decision_function"):
        # Chuẩn hóa giá trị quyết định sang khoảng [0, 1] qua hàm Sigmoid
        scores = model.decision_function(X_eval)
        y_proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        # Nếu mô hình không có xác suất, dùng nhãn dự đoán trực tiếp
        y_proba = model.predict(X_eval).astype(float)

    return evaluate_predictions(
        y_true=y_eval,
        y_proba=y_proba,
        threshold=threshold,
    )
