"""
Module đóng gói và khởi tạo 3 mô hình Machine Learning theo Protocol v2.0.

3 thuật toán được cố định:
1. Logistic Regression: Mốc đối chuẩn tuyến tính (Linear Baseline).
2. Decision Tree (CART): Mô hình phi tuyến dựa trên luật phân nhánh đơn lẻ.
3. Random Forest: Mô hình tập hợp (Ensemble/Bagging) chống overfitting.

Tham chiếu: MarkDown/experimental_protocol_data_leakage_v2.md (Mục 6.1)
"""

import logging
from typing import Dict, Optional

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import ExperimentConfig, ModelConfig

logger = logging.getLogger(__name__)

# Danh sách tên các mô hình được hỗ trợ chính thức
SUPPORTED_MODELS = ("logistic_regression", "decision_tree", "random_forest")


class ModelFactory:
    """
    Factory class khởi tạo các mô hình phân loại theo cấu hình thực nghiệm đã khóa.

    Sử dụng:
        factory = ModelFactory(ExperimentConfig())
        model = factory.create_model("random_forest", seed=42)
        pipeline = factory.create_pipeline(preprocessor, "logistic_regression", seed=42)
    """

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        """
        Khởi tạo ModelFactory với cấu hình thực nghiệm.

        Args:
            config: Đối tượng ExperimentConfig chứa siêu tham số mô hình.
                    Nếu None, dùng cấu hình mặc định.
        """
        self._cfg = config or ExperimentConfig()
        self._model_cfg: ModelConfig = self._cfg.models

    def create_model(self, model_name: str, seed: int) -> ClassifierMixin:
        """
        Khởi tạo một mô hình đơn lẻ với siêu tham số đã khóa và random_state xác định.

        Args:
            model_name: Tên mô hình ('logistic_regression', 'decision_tree', 'random_forest').
            seed: Giá trị random seed cho lần chạy.

        Returns:
            Đối tượng mô hình scikit-learn.

        Raises:
            ValueError: Nếu tên mô hình không nằm trong danh sách hỗ trợ.
        """
        name = model_name.lower().strip()

        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Mô hình '{model_name}' không được hỗ trợ! "
                f"Chỉ chấp nhận các mô hình: {SUPPORTED_MODELS}"
            )

        if name == "logistic_regression":
            params = dict(self._model_cfg.lr_params)
            params["random_state"] = seed
            model = LogisticRegression(**params)

        elif name == "decision_tree":
            params = dict(self._model_cfg.dt_params)
            params["random_state"] = seed
            model = DecisionTreeClassifier(**params)

        elif name == "random_forest":
            params = dict(self._model_cfg.rf_params)
            params["random_state"] = seed
            model = RandomForestClassifier(**params)

        logger.debug(
            "Khởi tạo mô hình '%s' thành công (seed=%d): %s",
            name,
            seed,
            model.get_params(),
        )
        return model

    def create_all_models(self, seed: int) -> Dict[str, ClassifierMixin]:
        """
        Khởi tạo toàn bộ 3 mô hình với cùng một random seed.

        Args:
            seed: Giá trị random seed.

        Returns:
            Dict ánh xạ từ tên mô hình sang đối tượng mô hình tương ứng.
        """
        return {
            name: self.create_model(name, seed=seed)
            for name in SUPPORTED_MODELS
        }

    def create_pipeline(
        self,
        preprocessor: ColumnTransformer,
        model_name: str,
        seed: int,
    ) -> Pipeline:
        """
        Đóng gói preprocessor và mô hình phân loại thành một scikit-learn Pipeline hoàn chỉnh.

        Việc đóng gói vào Pipeline giúp ngăn chặn rò rỉ dữ liệu ngoài ý muốn
        vì toàn bộ biến đổi chỉ được fit cùng lúc với mô hình trên tập Train.

        Args:
            preprocessor: ColumnTransformer đã cấu hình tiền xử lý.
            model_name: Tên thuật toán phân loại.
            seed: Giá trị random seed.

        Returns:
            Pipeline gồm 2 bước: ('preprocessor', preprocessor) và ('classifier', model).
        """
        model = self.create_model(model_name, seed=seed)
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )


def get_model(
    model_name: str,
    seed: int,
    config: Optional[ExperimentConfig] = None,
) -> ClassifierMixin:
    """
    Hàm tiện ích nhanh để khởi tạo một mô hình.

    Args:
        model_name: Tên mô hình.
        seed: Random seed.
        config: Cấu hình thực nghiệm (tùy chọn).

    Returns:
        Mô hình scikit-learn tương ứng.
    """
    factory = ModelFactory(config)
    return factory.create_model(model_name, seed=seed)
