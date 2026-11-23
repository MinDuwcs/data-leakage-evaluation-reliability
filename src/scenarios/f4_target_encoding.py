"""
Kịch bản thực nghiệm F4 — Target Encoding Leakage (Rò rỉ qua mã hóa mục tiêu).

Mục tiêu nghiên cứu:
Đo lường sự sai lệch đánh giá (Optimism Bias) khi mã hóa các biến phân loại
bằng tỷ lệ nhãn gian lận trung bình (Target Mean), so sánh giữa:
1. Quy trình chuẩn (Control): Out-of-fold cross-fitting chỉ trên tập Train.
2. Quy trình rò rỉ (Contaminated): Fit TargetEncoder trên toàn bộ Train ∪ Validation,
   để nhãn của tập kiểm thử trực tiếp tham gia vào việc tính toán tỷ lệ mã hóa.
3. Đánh giá khả năng tổng quát hóa trên tập Clean Holdout.

Các đặc trưng phân loại được áp dụng:
'Location' (8 nhóm), 'Payment_Method' (5 nhóm), 'Transaction_Type' (5 nhóm), 'Device_Used' (4 nhóm).

Tham chiếu:
- MarkDown/experimental_protocol_data_leakage_v2.md (Mục 5, Nhóm 2 - F4)
- MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md (Mục 5 - F4)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from src.config.settings import ExperimentConfig
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter, SplitIndices
from src.evaluation.metrics import EvaluationResult, evaluate_model
from src.models.classifiers import ModelFactory, SUPPORTED_MODELS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class F4TrialResult:
    """Kết quả chi tiết một lượt chạy thực nghiệm của kịch bản F4."""
    seed: int
    model_name: str
    branch: str  # 'control_valid_oof', 'contaminated_val_leakage', 'clean_holdout_eval'
    eval_set: str  # 'validation' hoặc 'clean_holdout'
    metrics: EvaluationResult


class F4TargetEncodingExperiment:
    """
    Điều phối toàn bộ thực nghiệm kịch bản F4 (Target Encoding Leakage).

    Sử dụng:
        experiment = F4TargetEncodingExperiment(ExperimentConfig())
        results_df = experiment.run()
    """

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        """
        Khởi tạo thực nghiệm F4 với cấu hình được phê duyệt.

        Args:
            config: Đối tượng ExperimentConfig chứa toàn bộ tham số.
        """
        self._cfg = config or ExperimentConfig()
        self._loader = DataLoader(self._cfg)
        self._splitter = DataSplitter(self._cfg)
        self._factory = ModelFactory(self._cfg)

        # Danh sách các đặc trưng phân loại áp dụng Target Encoding
        self.target_encode_features: Tuple[str, ...] = (
            "Location",
            "Payment_Method",
            "Transaction_Type",
            "Device_Used",
        )

    def _build_preprocessor(self, target_encoder: TargetEncoder) -> ColumnTransformer:
        """
        Xây dựng ColumnTransformer kết hợp biến số (StandardScaler) và biến Target Encoding.

        Args:
            target_encoder: Thể hiện của TargetEncoder (đã hoặc chưa fit).

        Returns:
            ColumnTransformer hoàn chỉnh.
        """
        num_cols = list(self._cfg.dataset.numeric_features)
        cat_cols = list(self.target_encode_features)

        # Pipeline cho biến số: Imputer (median) -> StandardScaler
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Pipeline cho biến phân loại: Imputer ('Missing') -> TargetEncoder
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("target_encoder", target_encoder),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

    def run_trial(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        seed: int,
        model_name: str,
    ) -> List[F4TrialResult]:
        """
        Chạy một lượt đối chứng hoàn chỉnh cho 1 seed và 1 model.

        Bao gồm:
        1. Nhánh Control (Valid Out-of-fold):
           - TargetEncoder sử dụng cross-fitting (cv=5) chỉ trên Train.
           - Validation được transform bằng mapping của Train.
           - Đánh giá trên Validation và trên Clean Holdout.
        2. Nhánh Contaminated (Validation-Label Leakage):
           - TargetEncoder được fit trên cả X_dev (Train ∪ Validation) và y_dev.
           - Validation được transform bằng encoder đã nhìn thấy nhãn của nó.
           - Đánh giá trên Validation (bị nhiễm) và trên Clean Holdout.

        Returns:
            Danh sách F4TrialResult cho các nhánh so sánh.
        """
        results: List[F4TrialResult] = []

        # =========================================================
        # Nhánh 1: Control (Valid Out-of-Fold Target Encoding)
        # =========================================================
        control_encoder = TargetEncoder(
            cv=5,
            smooth="auto",
            random_state=seed,
        )
        control_preprocessor = self._build_preprocessor(control_encoder)

        # Đóng gói vào pipeline: preprocessor chỉ fit trên Train
        control_pipeline = self._factory.create_pipeline(
            control_preprocessor, model_name, seed=seed
        )
        control_pipeline.fit(X_train, y_train)

        # Đánh giá trên Validation
        control_val_metrics = evaluate_model(control_pipeline, X_val, y_val)
        results.append(
            F4TrialResult(
                seed=seed,
                model_name=model_name,
                branch="control_valid_oof",
                eval_set="validation",
                metrics=control_val_metrics,
            )
        )

        # Đánh giá trên Clean Holdout
        control_holdout_metrics = evaluate_model(control_pipeline, X_holdout, y_holdout)
        results.append(
            F4TrialResult(
                seed=seed,
                model_name=model_name,
                branch="control_valid_oof",
                eval_set="clean_holdout",
                metrics=control_holdout_metrics,
            )
        )

        # =========================================================
        # Nhánh 2: Contaminated (Validation-Label Leakage)
        # =========================================================
        # Gộp Train và Validation để cố tình làm rò rỉ nhãn Validation
        X_dev_combined = pd.concat([X_train, X_val], axis=0)
        y_dev_combined = pd.concat([y_train, y_val], axis=0)

        # Encoder bị nhiễm vì học tỷ lệ mục tiêu từ toàn bộ tập DEV (kể cả nhãn Validation)
        leaked_encoder = TargetEncoder(
            smooth="auto",
            random_state=seed,
        )
        leaked_preprocessor = self._build_preprocessor(leaked_encoder)

        # Cố tình fit preprocessor trên toàn bộ tập DEV trước
        leaked_preprocessor.fit(X_dev_combined, y_dev_combined)

        # Transform độc lập từng tập bằng preprocessor đã bị nhiễm
        X_train_trans = leaked_preprocessor.transform(X_train)
        X_val_trans = leaked_preprocessor.transform(X_val)
        X_holdout_trans = leaked_preprocessor.transform(X_holdout)

        # Huấn luyện mô hình trên X_train_trans và nhãn y_train
        model = self._factory.create_model(model_name, seed=seed)
        model.fit(X_train_trans, y_train)

        # Đánh giá trên Validation bị nhiễm
        leaked_val_metrics = evaluate_model(model, X_val_trans, y_val)
        results.append(
            F4TrialResult(
                seed=seed,
                model_name=model_name,
                branch="contaminated_val_leakage",
                eval_set="validation",
                metrics=leaked_val_metrics,
            )
        )

        # Đánh giá trên Clean Holdout để đo lường mức độ suy giảm
        leaked_holdout_metrics = evaluate_model(model, X_holdout_trans, y_holdout)
        results.append(
            F4TrialResult(
                seed=seed,
                model_name=model_name,
                branch="contaminated_val_leakage",
                eval_set="clean_holdout",
                metrics=leaked_holdout_metrics,
            )
        )

        return results

    def run(self) -> pd.DataFrame:
        """
        Thực thi toàn bộ kịch bản F4 qua tất cả các mô hình và seeds lặp lại.

        Returns:
            pd.DataFrame tổng hợp kết quả của toàn bộ các lần chạy.
        """
        logger.info("Bắt đầu thực nghiệm kịch bản F4 (Target Encoding Leakage)...")

        # 1. Nạp và phân tách dữ liệu
        df = self._loader.load()
        X, y, groups = self._loader.get_features_target_groups(df)

        # Tách Clean Holdout (20% Users)
        holdout_split = self._splitter.split_holdout(df, groups)
        dev_indices = holdout_split.dev_idx
        holdout_indices = holdout_split.holdout_idx

        X_dev, y_dev, groups_dev = (
            X.iloc[dev_indices],
            y.iloc[dev_indices],
            groups.iloc[dev_indices],
        )
        X_holdout, y_holdout = X.iloc[holdout_indices], y.iloc[holdout_indices]

        all_trial_records: List[Dict] = []

        # 2. Lặp qua danh sách dev_seeds
        for seed in self._cfg.split.dev_seeds:
            logger.info("F4: Đang thực thi seed=%d", seed)
            split_res: SplitIndices = self._splitter.split_dev(
                df=X_dev,
                groups=groups_dev,
                seed=seed,
                group_aware=True,  # Giữ group-disjoint theo User_ID
            )

            X_train, y_train = (
                X_dev.iloc[split_res.train_idx],
                y_dev.iloc[split_res.train_idx],
            )
            X_val, y_val = (
                X_dev.iloc[split_res.val_idx],
                y_dev.iloc[split_res.val_idx],
            )

            # 3. Lặp qua 3 mô hình học máy
            for model_name in SUPPORTED_MODELS:
                trial_results = self.run_trial(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_holdout=X_holdout,
                    y_holdout=y_holdout,
                    seed=seed,
                    model_name=model_name,
                )

                for res in trial_results:
                    row = {
                        "scenario": "F4",
                        "seed": res.seed,
                        "model": res.model_name,
                        "branch": res.branch,
                        "eval_set": res.eval_set,
                    }
                    row.update(res.metrics.to_dict())
                    all_trial_records.append(row)

        results_df = pd.DataFrame(all_trial_records)
        logger.info(
            "Hoàn thành thực nghiệm F4. Tổng số lượt chạy: %d", len(results_df)
        )
        return results_df
