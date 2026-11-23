"""
Module xây dựng các pipeline tiền xử lý (Preprocessing Pipelines) chuẩn.

Chịu trách nhiệm:
1. Xây dựng ColumnTransformer phân tách cho biến số và biến phân loại.
2. Numeric pipeline: SimpleImputer (median) -> StandardScaler.
3. Categorical pipeline: SimpleImputer ('Missing') -> OneHotEncoder.
4. Cung cấp hàm trích xuất tên đặc trưng sau khi biến đổi (Feature Names).

Tham chiếu: MarkDown/experimental_protocol_data_leakage_v2.md (Mục 4)
"""

import logging
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import DatasetConfig, ExperimentConfig

logger = logging.getLogger(__name__)


def build_numeric_pipeline() -> Pipeline:
    """
    Tạo pipeline tiền xử lý cho các biến số (Numerical Features).

    Quy trình:
    1. Điền giá trị khuyết thiếu bằng trung vị (median) học từ tập Train.
    2. Chuẩn hóa tỷ lệ đặc trưng về mean=0, std=1 bằng StandardScaler.

    Returns:
        Pipeline gồm SimpleImputer và StandardScaler.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """
    Tạo pipeline tiền xử lý cho các biến phân loại (Categorical Features).

    Quy trình:
    1. Điền giá trị khuyết thiếu bằng giá trị hằng số 'Missing'.
    2. Mã hóa One-Hot với tùy chọn handle_unknown='ignore' để xử lý
       các danh mục mới xuất hiện ở tập Validation/Holdout.

    Returns:
        Pipeline gồm SimpleImputer và OneHotEncoder.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_baseline_preprocessor(
    config: Optional[ExperimentConfig] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Xây dựng ColumnTransformer chuẩn mực kết hợp cả biến số và biến phân loại.

    Args:
        config: Đối tượng ExperimentConfig chứa danh sách đặc trưng mặc định.
        numeric_features: Danh sách tên cột số tùy chỉnh (ghi đè cấu hình nếu có).
        categorical_features: Danh sách tên cột phân loại tùy chỉnh.

    Returns:
        ColumnTransformer sẵn sàng để fit trên tập Train.
    """
    if config is not None:
        dataset_cfg: DatasetConfig = config.dataset
        num_cols = list(numeric_features or dataset_cfg.numeric_features)
        cat_cols = list(categorical_features or dataset_cfg.categorical_features)
    else:
        num_cols = list(numeric_features or [])
        cat_cols = list(categorical_features or [])

    if not num_cols and not cat_cols:
        raise ValueError("Cần ít nhất một đặc trưng số hoặc phân loại để xây dựng preprocessor.")

    transformers = []
    if num_cols:
        transformers.append(("num", build_numeric_pipeline(), num_cols))
    if cat_cols:
        transformers.append(("cat", build_categorical_pipeline(), cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Khởi tạo Baseline ColumnTransformer thành công: %d cột số, %d cột phân loại.",
        len(num_cols),
        len(cat_cols),
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Trích xuất danh sách tên đặc trưng sau khi đã transform qua ColumnTransformer.

    Hữu ích cho việc phân tích độ quan trọng của đặc trưng (Feature Importance)
    hoặc hệ số hồi quy (Regression Coefficients).

    Args:
        preprocessor: ColumnTransformer đã được fit trên dữ liệu.

    Returns:
        Danh sách chuỗi tên các đặc trưng sau mã hóa.
    """
    return list(preprocessor.get_feature_names_out())
