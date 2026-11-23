"""
Module phân chia tập dữ liệu theo nhóm người dùng (Group-based Split) và phân chia ngẫu nhiên.

Chịu trách nhiệm:
1. Tách Clean Holdout (H) bất biến theo nhóm User_ID (20% Users, seed=2026).
2. Tách Development Pool (DEV) thành Train (T) và Validation (V) theo User_ID cho từng seed lặp.
3. Cung cấp phương thức phân chia ngẫu nhiên (Random Split) không theo User_ID phục vụ kịch bản D1.
4. Kiểm tra và xác thực ranh giới phân tách (zero overlap giữa H và DEV).

Tham chiếu: MarkDown/experimental_protocol_data_leakage_v2.md (Mục 3)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config.settings import ExperimentConfig, SplitConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitIndices:
    """Chỉ số dòng (row indices) cho một lượt phân chia dữ liệu."""
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass(frozen=True)
class HoldoutDevSplit:
    """Kết quả phân chia giữa Development Pool và Clean Holdout."""
    dev_idx: np.ndarray
    holdout_idx: np.ndarray


class DataSplitter:
    """
    Quản lý việc phân chia tập dữ liệu theo Protocol v2.0.

    Sử dụng:
        from src.data import DataSplitter
        from src.config import ExperimentConfig

        splitter = DataSplitter(ExperimentConfig())
        holdout_split = splitter.split_holdout(df, groups)
        
        # Lặp qua các seeds trên tập DEV
        for seed, split in splitter.iter_dev_splits(dev_df, dev_groups, group_aware=True):
            train_data = dev_df.iloc[split.train_idx]
            val_data = dev_df.iloc[split.val_idx]
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Khởi tạo DataSplitter với cấu hình thực nghiệm.

        Args:
            config: Đối tượng ExperimentConfig chứa tham số split.
        """
        self._split_cfg: SplitConfig = config.split

    def split_holdout(
        self, df: pd.DataFrame, groups: pd.Series
    ) -> HoldoutDevSplit:
        """
        Tách tập dữ liệu thành Development Pool (DEV) và Clean Holdout (H).

        Phương pháp: GroupShuffleSplit trên User_ID với tỷ lệ holdout_ratio
        và seed cố định holdout_seed (2026). Đảm bảo không có user nào
        xuất hiện ở cả DEV và H.

        Args:
            df: DataFrame toàn bộ dữ liệu.
            groups: Series chứa User_ID của từng dòng.

        Returns:
            HoldoutDevSplit chứa chỉ số dòng của DEV và Holdout.

        Raises:
            ValueError: Nếu có sự chồng lấn User_ID giữa DEV và Holdout.
        """
        cfg = self._split_cfg
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.holdout_ratio,
            random_state=cfg.holdout_seed,
        )

        train_indices, test_indices = next(gss.split(df, groups=groups))

        # Kiểm tra tính phân tách tuyệt đối (Group-disjoint)
        dev_users = set(groups.iloc[train_indices].unique())
        holdout_users = set(groups.iloc[test_indices].unique())
        overlap = dev_users.intersection(holdout_users)

        if overlap:
            raise ValueError(
                f"Phát hiện rò rỉ User_ID giữa DEV và Clean Holdout! "
                f"Số lượng user chồng lấn: {len(overlap)}"
            )

        logger.info(
            "Tách Clean Holdout thành công (seed=%d):\n"
            "  - DEV:     %d dòng (%d users)\n"
            "  - Holdout: %d dòng (%d users)\n"
            "  - Tỷ lệ User Holdout: %.2f%%",
            cfg.holdout_seed,
            len(train_indices),
            len(dev_users),
            len(test_indices),
            len(holdout_users),
            len(holdout_users) / (len(dev_users) + len(holdout_users)) * 100,
        )

        return HoldoutDevSplit(
            dev_idx=train_indices,
            holdout_idx=test_indices,
        )

    def split_dev(
        self,
        df: pd.DataFrame,
        groups: pd.Series,
        seed: int,
        group_aware: bool = True,
        stratify_col: pd.Series = None,
    ) -> SplitIndices:
        """
        Phân chia Development Pool thành Train (T) và Validation (V) cho một seed cụ thể.

        Args:
            df: DataFrame của Development Pool.
            groups: Series chứa User_ID của DEV.
            seed: Giá trị random seed cho lần chạy.
            group_aware: Nếu True, dùng GroupShuffleSplit (nhánh Control hợp lệ).
                         Nếu False, dùng train_test_split ngẫu nhiên theo dòng (nhánh Contaminated D1).
            stratify_col: Series dùng để phân tầng nếu group_aware=False (tùy chọn).

        Returns:
            SplitIndices chứa chỉ số dòng của Train và Validation.
        """
        cfg = self._split_cfg

        if group_aware:
            # Nhánh Control: Không rò rỉ User_ID giữa Train và Validation
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=cfg.dev_validation_ratio,
                random_state=seed,
            )
            train_idx, val_idx = next(gss.split(df, groups=groups))

            # Xác thực không có user overlap
            t_users = set(groups.iloc[train_idx].unique())
            v_users = set(groups.iloc[val_idx].unique())
            overlap = t_users.intersection(v_users)
            if overlap:
                raise ValueError(
                    f"Rò rỉ User_ID trong DEV split group-aware (seed={seed})! "
                    f"Số user chồng lấn: {len(overlap)}"
                )
        else:
            # Nhánh Contaminated (D1): Chia ngẫu nhiên theo dòng, cố tình tạo User Leakage
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=cfg.dev_validation_ratio,
                random_state=seed,
                stratify=stratify_col,
            )

        return SplitIndices(train_idx=train_idx, val_idx=val_idx)

    def iter_dev_splits(
        self,
        df: pd.DataFrame,
        groups: pd.Series,
        group_aware: bool = True,
        stratify_col: pd.Series = None,
    ) -> Generator[Tuple[int, SplitIndices], None, None]:
        """
        Lặp qua toàn bộ danh sách dev_seeds để sinh các cặp (seed, SplitIndices).

        Args:
            df: DataFrame của Development Pool.
            groups: Series chứa User_ID của DEV.
            group_aware: True cho Control split, False cho Contaminated split.
            stratify_col: Series phân tầng cho Random split (tùy chọn).

        Yields:
            Tuple gồm (seed, SplitIndices).
        """
        for seed in self._split_cfg.dev_seeds:
            split = self.split_dev(
                df=df,
                groups=groups,
                seed=seed,
                group_aware=group_aware,
                stratify_col=stratify_col,
            )
            yield seed, split

    def compute_user_overlap(
        self, groups: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Dict[str, float]:
        """
        Tính toán mức độ rò rỉ người dùng (User Leakage diagnostics) giữa Train và Validation.

        Đo lường:
        - Số lượng user xuất hiện ở cả hai tập.
        - Tỷ lệ user của tập Validation đã có mặt trong tập Train.
        - Tỷ lệ giao dịch trong Validation thuộc về các user đã bị lộ trong Train.

        Args:
            groups: Series User_ID.
            train_idx: Chỉ số dòng tập Train.
            val_idx: Chỉ số dòng tập Validation.

        Returns:
            Dict chứa các chỉ số đo lường rò rỉ.
        """
        train_users = set(groups.iloc[train_idx].unique())
        val_users = set(groups.iloc[val_idx].unique())
        overlap_users = train_users.intersection(val_users)

        val_total_tx = len(val_idx)
        val_leaked_tx = groups.iloc[val_idx].isin(overlap_users).sum()

        return {
            "n_train_users": len(train_users),
            "n_val_users": len(val_users),
            "n_overlap_users": len(overlap_users),
            "user_overlap_ratio": len(overlap_users) / len(val_users) if val_users else 0.0,
            "val_leaked_tx_count": int(val_leaked_tx),
            "val_leaked_tx_ratio": val_leaked_tx / val_total_tx if val_total_tx else 0.0,
        }
