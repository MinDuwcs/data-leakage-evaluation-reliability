"""
Module tiền xử lý dữ liệu thực nghiệm Data Leakage.

Cung cấp:
- Baseline preprocessor: Pipeline điền khuyết (Imputation) kết hợp chuẩn hóa (StandardScaler)
  cho biến số và mã hóa danh mục (OneHotEncoder) cho biến phân loại.
- Các hàm tiện ích hỗ trợ phân tách fit scope (chỉ fit trên Train).
"""

from src.preprocessing.pipelines import build_baseline_preprocessor
