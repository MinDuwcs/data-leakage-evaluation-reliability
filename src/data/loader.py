"""
Module nạp dữ liệu từ CSV và xác thực tính toàn vẹn.

Quy trình:
1. Tính SHA-256 của file CSV, so sánh với mã băm đã audit.
2. Đọc dữ liệu bằng pandas, ép kiểu dữ liệu cho các cột.
3. Kiểm tra schema (tên cột, missing values, nhãn mục tiêu).
4. Trả về DataFrame đã xác thực, sẵn sàng cho phân chia tập.

Tham chiếu: MarkDown/dataset_audit.md (Gate A01, A02, A03)
"""

import hashlib
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config.settings import DatasetConfig, ExperimentConfig

logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """Lỗi xác thực tính toàn vẹn dữ liệu (SHA-256 không khớp hoặc schema sai)."""
    pass


class DataLoader:
    """
    Nạp và xác thực dữ liệu thực nghiệm từ file CSV.

    Sử dụng:
        from src.data import DataLoader
        from src.config import ExperimentConfig

        loader = DataLoader(ExperimentConfig())
        df = loader.load()
        X, y, groups = loader.get_features_target_groups(df)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Khởi tạo DataLoader với cấu hình thực nghiệm.

        Args:
            config: Đối tượng ExperimentConfig chứa toàn bộ tham số.
        """
        self._dataset_cfg: DatasetConfig = config.dataset

    def _compute_sha256(self, file_path: Path) -> str:
        """
        Tính mã băm SHA-256 của file.

        Args:
            file_path: Đường dẫn tuyệt đối đến file cần tính hash.

        Returns:
            Chuỗi hex SHA-256 của file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _validate_integrity(self, file_path: Path) -> None:
        """
        Xác thực mã băm SHA-256 của file CSV.

        Raises:
            FileNotFoundError: Nếu file CSV không tồn tại.
            DataIntegrityError: Nếu mã băm SHA-256 không khớp giá trị đã audit.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại: {file_path}"
            )

        actual_hash = self._compute_sha256(file_path)
        expected_hash = self._dataset_cfg.expected_sha256

        if actual_hash != expected_hash:
            raise DataIntegrityError(
                f"Mã băm SHA-256 không khớp!\n"
                f"  Kỳ vọng (audit): {expected_hash}\n"
                f"  Thực tế (file):  {actual_hash}\n"
                f"File có thể đã bị thay đổi hoặc không đúng phiên bản."
            )

        logger.info("Xác thực SHA-256 thành công: %s", actual_hash[:16] + "...")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Kiểm tra schema của DataFrame sau khi đọc CSV.

        Kiểm tra:
        - Tồn tại tất cả các cột bắt buộc (ID, Group, Features, Target).
        - Cột nhãn mục tiêu chỉ chứa giá trị 0 và 1.
        - Không có giá trị khuyết thiếu ở cột nhãn mục tiêu.

        Raises:
            DataIntegrityError: Nếu schema không hợp lệ.
        """
        cfg = self._dataset_cfg

        # Kiểm tra cột bắt buộc
        required_columns = set(
            list(cfg.id_columns)
            + [cfg.group_column]
            + list(cfg.numeric_features)
            + list(cfg.categorical_features)
            + [cfg.target_column]
        )
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DataIntegrityError(
                f"Thiếu các cột bắt buộc trong CSV: {missing_columns}"
            )

        # Kiểm tra nhãn mục tiêu
        target = df[cfg.target_column]
        if target.isnull().any():
            raise DataIntegrityError(
                f"Cột nhãn '{cfg.target_column}' chứa giá trị khuyết thiếu."
            )

        unique_labels = set(target.unique())
        if not unique_labels.issubset({0, 1}):
            raise DataIntegrityError(
                f"Cột nhãn '{cfg.target_column}' chứa giá trị ngoài {{0, 1}}: "
                f"{unique_labels}"
            )

        logger.info(
            "Schema hợp lệ: %d dòng, %d cột. "
            "Phân phối nhãn — Lớp 0: %d (%.2f%%), Lớp 1: %d (%.2f%%)",
            len(df),
            len(df.columns),
            (target == 0).sum(),
            (target == 0).mean() * 100,
            (target == 1).sum(),
            (target == 1).mean() * 100,
        )

    def load(self) -> pd.DataFrame:
        """
        Nạp và xác thực toàn bộ dữ liệu từ file CSV.

        Quy trình:
        1. Xác thực SHA-256.
        2. Đọc CSV với kiểu dữ liệu phù hợp.
        3. Kiểm tra schema.

        Returns:
            DataFrame đã xác thực, sẵn sàng cho phân chia tập.

        Raises:
            FileNotFoundError: File CSV không tồn tại.
            DataIntegrityError: SHA-256 hoặc schema không hợp lệ.
        """
        file_path = self._dataset_cfg.absolute_csv_path

        # Bước 1: Xác thực tính toàn vẹn file
        self._validate_integrity(file_path)

        # Bước 2: Đọc CSV
        # Ép User_ID sang string để tránh bị xử lý như số khi GroupSplit
        dtype_map = {
            "Transaction_ID": str,
            "User_ID": str,
            "Fraudulent": int,
        }
        df = pd.read_csv(file_path, dtype=dtype_map)

        # Bước 3: Kiểm tra schema
        self._validate_schema(df)

        logger.info(
            "Nạp dữ liệu thành công từ: %s (%d dòng)",
            file_path.name,
            len(df),
        )
        return df

    def get_features_target_groups(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Tách DataFrame thành feature matrix, nhãn mục tiêu và cột nhóm.

        Loại bỏ các cột định danh (Transaction_ID) và cột nhãn (Fraudulent)
        khỏi feature matrix. Giữ lại User_ID làm cột nhóm riêng để phục vụ
        Group-based Split, nhưng không đưa vào feature matrix.

        Args:
            df: DataFrame đã xác thực từ phương thức load().

        Returns:
            Tuple gồm:
            - X (pd.DataFrame): Feature matrix (Numeric + Categorical).
            - y (pd.Series): Nhãn mục tiêu (0 hoặc 1).
            - groups (pd.Series): Cột nhóm User_ID cho GroupSplit.
        """
        cfg = self._dataset_cfg

        # Các cột cần loại bỏ khỏi feature matrix
        columns_to_drop = list(cfg.id_columns) + [
            cfg.group_column,
            cfg.target_column,
        ]

        X = df.drop(columns=columns_to_drop)
        y = df[cfg.target_column]
        groups = df[cfg.group_column]

        logger.info(
            "Tách dữ liệu: X=%s, y=%s, groups=%d users",
            X.shape,
            y.shape,
            groups.nunique(),
        )
        return X, y, groups

