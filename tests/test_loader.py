"""
Unit test kiểm tra tính toàn vẹn của DataLoader, DataSplitter và Baseline Pipeline.

Đảm bảo:
1. Xác thực mã băm SHA-256 khớp tuyệt đối với số liệu kiểm toán (dataset_audit.md).
2. Schema nạp vào đúng 51.000 dòng, nhãn Fraudulent chỉ gồm {0, 1}.
3. Tách Clean Holdout và DEV đảm bảo tuyệt đối không có sự chồng lấn User_ID (Zero-overlap).
4. Phân chia ngẫu nhiên (Random Split) chứng minh được sự tồn tại của User Leakage.
5. Baseline Preprocessing Pipeline fit và transform thành công mà không để sót missing values.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import ExperimentConfig
from src.data.loader import DataIntegrityError, DataLoader
from src.data.splitter import DataSplitter
from src.preprocessing.pipelines import build_baseline_preprocessor


class TestDataLoaderAndSplitter(unittest.TestCase):
    """Bộ kiểm thử cho các thành phần nạp dữ liệu và phân tách tập thực nghiệm."""

    @classmethod
    def setUpClass(cls) -> None:
        """Khởi tạo cấu hình và đối tượng dùng chung cho toàn bộ test case."""
        cls.config = ExperimentConfig()
        cls.loader = DataLoader(cls.config)
        cls.splitter = DataSplitter(cls.config)

    def test_01_sha256_integrity(self) -> None:
        """Kiểm tra mã băm SHA-256 của file CSV thực tế có khớp với audit hay không."""
        file_path = self.config.dataset.absolute_csv_path
        self.assertTrue(file_path.exists(), f"File dữ liệu không tồn tại: {file_path}")

        actual_sha = self.loader._compute_sha256(file_path)
        expected_sha = self.config.dataset.expected_sha256

        self.assertEqual(
            actual_sha,
            expected_sha,
            f"Mã băm SHA-256 bị sai lệch!\nKỳ vọng: {expected_sha}\nThực tế: {actual_sha}",
        )

    def test_02_load_and_schema(self) -> None:
        """Kiểm tra dữ liệu nạp vào đúng kích thước và phân phối nhãn hợp lệ."""
        df = self.loader.load()

        # Kiểm tra kích thước
        self.assertEqual(len(df), 51000, "Tổng số dòng phải đúng 51.000 dòng.")
        self.assertEqual(len(df.columns), 12, "Số lượng cột phải là 12.")

        # Kiểm tra nhãn mục tiêu
        target_col = self.config.dataset.target_column
        self.assertIn(target_col, df.columns)
        self.assertFalse(df[target_col].isnull().any(), "Nhãn không được chứa giá trị null.")
        self.assertEqual(set(df[target_col].unique()), {0, 1})

    def test_03_get_features_target_groups(self) -> None:
        """Kiểm tra việc bóc tách ma trận X, nhãn y và nhóm groups (User_ID)."""
        df = self.loader.load()
        X, y, groups = self.loader.get_features_target_groups(df)

        # X phải loại bỏ Transaction_ID, User_ID, Fraudulent -> còn lại 9 cột
        self.assertEqual(X.shape[1], 9)
        self.assertEqual(len(y), 51000)
        self.assertEqual(groups.nunique(), 4000, "Phải có đúng 4.000 User_ID duy nhất.")

        # Kiểm tra các cột trong X
        expected_cols = set(
            list(self.config.dataset.numeric_features)
            + list(self.config.dataset.categorical_features)
        )
        self.assertEqual(set(X.columns), expected_cols)

    def test_04_holdout_split_zero_overlap(self) -> None:
        """Xác thực Clean Holdout không có bất kỳ User_ID nào trùng lặp với DEV."""
        df = self.loader.load()
        X, y, groups = self.loader.get_features_target_groups(df)

        holdout_split = self.splitter.split_holdout(df, groups)

        dev_users = set(groups.iloc[holdout_split.dev_idx].unique())
        holdout_users = set(groups.iloc[holdout_split.holdout_idx].unique())

        overlap = dev_users.intersection(holdout_users)
        self.assertEqual(
            len(overlap),
            0,
            f"Phát hiện rò rỉ User_ID vào Clean Holdout! Số user trùng: {len(overlap)}",
        )

        # Kiểm tra tỷ lệ user xấp xỉ 20%
        holdout_user_ratio = len(holdout_users) / (len(dev_users) + len(holdout_users))
        self.assertAlmostEqual(holdout_user_ratio, 0.20, delta=0.02)

    def test_05_dev_split_control_vs_contaminated(self) -> None:
        """Kiểm tra sự khác biệt ranh giới giữa Group-aware Split (Control) và Random Split (D1)."""
        df = self.loader.load()
        X, y, groups = self.loader.get_features_target_groups(df)

        holdout_split = self.splitter.split_holdout(df, groups)
        dev_groups = groups.iloc[holdout_split.dev_idx]
        dev_df = df.iloc[holdout_split.dev_idx]

        seed = 13

        # Nhánh 1: Control (Group-aware) -> Zero overlap
        control_split = self.splitter.split_dev(
            dev_df, dev_groups, seed=seed, group_aware=True
        )
        control_diag = self.splitter.compute_user_overlap(
            dev_groups, control_split.train_idx, control_split.val_idx
        )
        self.assertEqual(
            control_diag["n_overlap_users"],
            0,
            "Nhánh Control bắt buộc phải có zero user overlap.",
        )

        # Nhánh 2: Contaminated (Random Split D1) -> User Leakage phải xuất hiện
        contam_split = self.splitter.split_dev(
            dev_df, dev_groups, seed=seed, group_aware=False
        )
        contam_diag = self.splitter.compute_user_overlap(
            dev_groups, contam_split.train_idx, contam_split.val_idx
        )
        self.assertGreater(
            contam_diag["n_overlap_users"],
            0,
            "Nhánh Random Split phải tạo ra User Leakage (n_overlap_users > 0).",
        )
        self.assertGreater(
            contam_diag["val_leaked_tx_ratio"],
            0.50,
            "Tỷ lệ giao dịch Validation bị rò rỉ khi chia ngẫu nhiên thường rất cao (>50%).",
        )

    def test_06_baseline_pipeline_imputation_and_scaling(self) -> None:
        """Kiểm tra ColumnTransformer xử lý được missing values và chuẩn hóa thành công."""
        df = self.loader.load()
        X, y, groups = self.loader.get_features_target_groups(df)

        preprocessor = build_baseline_preprocessor(self.config)

        # Giả lập chia Train (1000 dòng) và Test (500 dòng)
        X_train_sub = X.iloc[:1000]
        X_val_sub = X.iloc[1000:1500]

        # Fit trên Train và Transform trên cả hai
        X_train_trans = preprocessor.fit_transform(X_train_sub)
        X_val_trans = preprocessor.transform(X_val_sub)

        # Kết quả sau transform không được chứa bất kỳ giá trị NaN nào
        self.assertFalse(
            np.isnan(X_train_trans).any(),
            "X_train sau khi qua ColumnTransformer vẫn còn giá trị NaN!",
        )
        self.assertFalse(
            np.isnan(X_val_trans).any(),
            "X_val sau khi qua ColumnTransformer vẫn còn giá trị NaN!",
        )


if __name__ == "__main__":
    unittest.main()
