"""
Kịch bản thực nghiệm F3 — Post-event Feature Leakage (Rò rỉ đặc trưng hậu nghiệm).

Mục tiêu nghiên cứu:
Đo lường mức độ sai lệch đánh giá (Optimism Bias) khi đưa vào mô hình đặc trưng
chỉ phát sinh sau khi giao dịch đã hoàn tất và kết luận điều tra được đưa ra,
nhưng bị nhầm lẫn là có sẵn tại thời điểm dự đoán (Prediction Time).

Cơ chế phân biệt với F2:
- F2 khảo sát độ mạnh của tín hiệu (Signal Noise: 5%, 20%, 40%).
- F3 khảo sát độ phủ kiểm toán (Audit Coverage: 10%, 30%, 50%, 80%) và tính khả dụng
  theo thời gian (Time-of-Availability). Các giao dịch chưa hoặc không được điều tra
  sẽ mang trạng thái 'Unavailable'/'Pending'.

Tham chiếu:
- MarkDown/experimental_protocol_data_leakage_v2.md (Mục 5, Nhóm 2 - F3)
- MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md (Mục 5 - F3)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import ExperimentConfig
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter, SplitIndices
from src.evaluation.metrics import EvaluationResult, evaluate_model
from src.models.classifiers import ModelFactory, SUPPORTED_MODELS
from src.preprocessing.pipelines import build_baseline_preprocessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class F3TrialResult:
    """Kết quả chi tiết một lượt chạy thực nghiệm của kịch bản F3."""
    seed: int
    model_name: str
    coverage: float
    branch: str  # 'control', 'contaminated_retrospective', 'deployment_mismatch'
    metrics: EvaluationResult


class F3PostEventExperiment:
    """
    Điều phối toàn bộ thực nghiệm kịch bản F3 (Post-event Feature Leakage).

    Sử dụng:
        experiment = F3PostEventExperiment(ExperimentConfig())
        results_df = experiment.run()
    """

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        """
        Khởi tạo thực nghiệm F3 với cấu hình được phê duyệt.

        Args:
            config: Đối tượng ExperimentConfig chứa toàn bộ tham số.
        """
        self._cfg = config or ExperimentConfig()
        self._loader = DataLoader(self._cfg)
        self._splitter = DataSplitter(self._cfg)
        self._factory = ModelFactory(self._cfg)

        # Danh sách các mức độ phủ kiểm toán khảo sát (Audit Coverage Grid)
        self.coverage_levels: Tuple[float, ...] = (0.10, 0.30, 0.50, 0.80)

    @staticmethod
    def generate_post_event_feature(
        y: pd.Series,
        coverage: float,
        accuracy: float = 0.95,
        seed: int = 42,
    ) -> pd.Series:
        """
        Mô phỏng đặc trưng hậu nghiệm 'Investigation_Result' với độ phủ và độ chính xác kiểm soát.

        Logic tạo dữ liệu:
        1. Chỉ một tỷ lệ `coverage` giao dịch được đưa vào điều tra hậu kiểm.
        2. Các giao dịch không được điều tra nhận trạng thái 'Not_Investigated' (thực tế lúc quẹt thẻ).
        3. Các giao dịch được điều tra sẽ phản ánh nhãn gian lận thực tế với độ chính xác `accuracy`
           ('Fraud_Confirmed' hoặc 'Cleared_Legitimate').

        Args:
            y: Series nhãn thực tế Fraudulent (0 hoặc 1).
            coverage: Tỷ lệ giao dịch được kiểm toán hậu kỳ (0.0 đến 1.0).
            accuracy: Độ chính xác của kết luận điều tra (mặc định 95%).
            seed: Random seed đảm bảo tính tái lập.

        Returns:
            pd.Series dạng phân loại (string) chứa kết quả hậu nghiệm.
        """
        rng = np.random.default_rng(seed)
        n = len(y)

        # Mặc định tất cả giao dịch tại thời điểm quẹt thẻ là chưa điều tra
        feature_vals = np.full(n, "Not_Investigated", dtype=object)

        # Chọn ngẫu nhiên các giao dịch được kiểm toán hậu kiểm theo tỷ lệ coverage
        audit_mask = rng.random(n) < coverage

        for idx in np.where(audit_mask)[0]:
            actual_label = y.iloc[idx]
            # Mô phỏng sai số điều tra
            is_correct = rng.random() < accuracy
            if actual_label == 1:
                feature_vals[idx] = (
                    "Fraud_Confirmed" if is_correct else "Cleared_Legitimate"
                )
            else:
                feature_vals[idx] = (
                    "Cleared_Legitimate" if is_correct else "Fraud_Confirmed"
                )

        return pd.Series(feature_vals, index=y.index, name="Investigation_Result")

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
        coverage: float,
    ) -> List[F3TrialResult]:
        """
        Chạy một lượt đối chứng hoàn chỉnh cho 1 seed, 1 model và 1 mức coverage.

        Bao gồm 3 nhánh:
        1. Control: Chỉ sử dụng các đặc trưng hợp lệ có sẵn lúc dự đoán.
        2. Contaminated Retrospective: Đưa đặc trưng hậu nghiệm vào cả Train và Validation.
        3. Deployment Mismatch: Mô hình huấn luyện với đặc trưng hậu nghiệm nhưng khi triển khai
           (trên Clean Holdout) thì đặc trưng này không khả dụng ('Not_Investigated').

        Returns:
            Danh sách F3TrialResult của cả 3 nhánh.
        """
        results: List[F3TrialResult] = []

        # ---------------------------------------------------------
        # Nhánh 1: Control (Baseline hợp lệ)
        # ---------------------------------------------------------
        control_preprocessor = build_baseline_preprocessor(self._cfg)
        control_pipeline = self._factory.create_pipeline(
            control_preprocessor, model_name, seed=seed
        )
        control_pipeline.fit(X_train, y_train)

        control_metrics = evaluate_model(control_pipeline, X_val, y_val)
        results.append(
            F3TrialResult(
                seed=seed,
                model_name=model_name,
                coverage=coverage,
                branch="control",
                metrics=control_metrics,
            )
        )

        # ---------------------------------------------------------
        # Sinh đặc trưng hậu nghiệm cho Train và Validation độc lập
        # ---------------------------------------------------------
        inv_train = self.generate_post_event_feature(
            y_train, coverage=coverage, seed=seed
        )
        inv_val = self.generate_post_event_feature(
            y_val, coverage=coverage, seed=seed + 1000
        )

        X_train_leak = X_train.copy()
        X_train_leak["Investigation_Result"] = inv_train

        X_val_leak = X_val.copy()
        X_val_leak["Investigation_Result"] = inv_val

        # Cấu hình preprocessor bổ sung thêm biến phân loại 'Investigation_Result'
        dataset_cfg = self._cfg.dataset
        extended_cat_cols = list(dataset_cfg.categorical_features) + [
            "Investigation_Result"
        ]
        leaked_preprocessor = build_baseline_preprocessor(
            self._cfg, categorical_features=extended_cat_cols
        )

        leaked_pipeline = self._factory.create_pipeline(
            leaked_preprocessor, model_name, seed=seed
        )
        leaked_pipeline.fit(X_train_leak, y_train)

        # ---------------------------------------------------------
        # Nhánh 2: Contaminated Retrospective (Đánh giá nội bộ bị nhiễm)
        # ---------------------------------------------------------
        retrospective_metrics = evaluate_model(
            leaked_pipeline, X_val_leak, y_val
        )
        results.append(
            F3TrialResult(
                seed=seed,
                model_name=model_name,
                coverage=coverage,
                branch="contaminated_retrospective",
                metrics=retrospective_metrics,
            )
        )

        # ---------------------------------------------------------
        # Nhánh 3: Deployment Mismatch (Đánh giá thực tế khi triển khai)
        # Tại thời điểm triển khai, giao dịch mới chưa từng được kiểm toán hậu kỳ
        # nên toàn bộ giá trị 'Investigation_Result' là 'Not_Investigated'.
        # ---------------------------------------------------------
        X_holdout_deployed = X_holdout.copy()
        X_holdout_deployed["Investigation_Result"] = "Not_Investigated"

        deployment_metrics = evaluate_model(
            leaked_pipeline, X_holdout_deployed, y_holdout
        )
        results.append(
            F3TrialResult(
                seed=seed,
                model_name=model_name,
                coverage=coverage,
                branch="deployment_mismatch",
                metrics=deployment_metrics,
            )
        )

        return results

    def run(self) -> pd.DataFrame:
        """
        Thực thi toàn bộ kịch bản F3 qua tất cả các mô hình, seeds và mức coverage.

        Returns:
            pd.DataFrame tổng hợp kết quả của toàn bộ các lần chạy.
        """
        logger.info("Bắt đầu thực nghiệm kịch bản F3 (Post-event Feature Leakage)...")

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
            logger.info("F3: Đang thực thi seed=%d", seed)
            split_res: SplitIndices = self._splitter.split_dev(
                df=X_dev,
                groups=groups_dev,
                seed=seed,
                group_aware=True,  # Control giữ group-disjoint theo User_ID
            )

            X_train, y_train = (
                X_dev.iloc[split_res.train_idx],
                y_dev.iloc[split_res.train_idx],
            )
            X_val, y_val = (
                X_dev.iloc[split_res.val_idx],
                y_dev.iloc[split_res.val_idx],
            )

            # 3. Lặp qua các mô hình và mức độ phủ (coverage)
            for model_name in SUPPORTED_MODELS:
                for coverage in self.coverage_levels:
                    trial_results = self.run_trial(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_holdout=X_holdout,
                        y_holdout=y_holdout,
                        seed=seed,
                        model_name=model_name,
                        coverage=coverage,
                    )

                    for res in trial_results:
                        row = {
                            "scenario": "F3",
                            "seed": res.seed,
                            "model": res.model_name,
                            "coverage": res.coverage,
                            "branch": res.branch,
                        }
                        row.update(res.metrics.to_dict())
                        all_trial_records.append(row)

        results_df = pd.DataFrame(all_trial_records)
        logger.info(
            "Hoàn thành thực nghiệm F3. Tổng số lượt chạy: %d", len(results_df)
        )
        return results_df
