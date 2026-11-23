"""
Cấu hình tập trung cho toàn bộ thực nghiệm Data Leakage.

Mọi tham số đã khóa trong Experimental Protocol v2.0 được định nghĩa tại đây.
Các module khác import ExperimentConfig thay vì hard-code giá trị cục bộ.

Tham chiếu: MarkDown/experimental_protocol_data_leakage_v2.md (Protocol v2.0)
            MarkDown/dataset_audit.md (Báo cáo kiểm toán dữ liệu)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Đường dẫn gốc của project (thư mục chứa src/, data/, MarkDown/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    """Cấu hình dữ liệu đã xác thực từ Dataset Audit."""

    # Đường dẫn tương đối đến file CSV (tính từ PROJECT_ROOT)
    csv_path: Path = Path("data/fraud_detection.csv")

    # Mã băm SHA-256 xác thực tính toàn vẹn file
    # (Nguồn: dataset_audit.md, Gate A01)
    expected_sha256: str = (
        "9a009cd6d20720b2faed8a8b4d5eb45d99008d6bc4fc77dd9b5a104166c62b9d"
    )

    # Tên cột nhãn mục tiêu
    target_column: str = "Fraudulent"

    # Các cột định danh / metadata — loại bỏ khỏi feature matrix
    id_columns: tuple = ("Transaction_ID",)

    # Cột nhóm dùng cho Group-based Split
    group_column: str = "User_ID"

    # Các cột đặc trưng số (Numerical Features)
    numeric_features: tuple = (
        "Transaction_Amount",
        "Time_of_Transaction",
        "Account_Age",
        "Number_of_Transactions_Last_24H",
        "Previous_Fraudulent_Transactions",
    )

    # Các cột đặc trưng phân loại (Categorical Features)
    categorical_features: tuple = (
        "Transaction_Type",
        "Device_Used",
        "Location",
        "Payment_Method",
    )

    @property
    def absolute_csv_path(self) -> Path:
        """Trả về đường dẫn tuyệt đối đến file CSV."""
        return PROJECT_ROOT / self.csv_path

    @property
    def all_features(self) -> tuple:
        """Trả về danh sách toàn bộ đặc trưng đầu vào (Numeric + Categorical)."""
        return self.numeric_features + self.categorical_features


@dataclass(frozen=True)
class SplitConfig:
    """Cấu hình phân chia dữ liệu theo Protocol v2.0, Section 3."""

    # Seed cố định cho việc tách Clean Holdout
    holdout_seed: int = 2026

    # Tỷ lệ Clean Holdout (theo số lượng User, không phải số dòng)
    holdout_ratio: float = 0.20

    # Danh sách seeds cho các lần lặp trên Development Pool
    dev_seeds: tuple = (13, 37, 71, 101, 131)

    # Tỷ lệ Validation trong Development Pool
    dev_validation_ratio: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    """Cấu hình siêu tham số 3 mô hình đã khóa trong Protocol v2.0, Section 6."""

    # Logistic Regression
    lr_params: Dict = field(default_factory=lambda: {
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
    })

    # Decision Tree
    dt_params: Dict = field(default_factory=lambda: {
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_split": 20,
    })

    # Random Forest
    rf_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 10,
        "n_jobs": -1,
    })


@dataclass(frozen=True)
class EvaluationConfig:
    """Cấu hình đánh giá theo Protocol v2.0, Section 6.2."""

    # Ngưỡng phân loại chuẩn
    classification_threshold: float = 0.5

    # Nhãn positive class (1 = Gian lận)
    positive_label: int = 1

    # Danh sách chỉ số đánh giá chính
    primary_metrics: tuple = (
        "average_precision",
        "f1_score",
        "recall",
        "precision",
    )

    # Danh sách chỉ số đánh giá phụ trợ
    secondary_metrics: tuple = (
        "roc_auc",
        "accuracy",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Cấu hình tổng hợp cho toàn bộ thực nghiệm.

    Sử dụng:
        from src.config import ExperimentConfig
        cfg = ExperimentConfig()
        print(cfg.dataset.absolute_csv_path)
        print(cfg.split.dev_seeds)
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

