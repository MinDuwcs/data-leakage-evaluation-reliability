"""
Module nạp dữ liệu và phân chia tập thực nghiệm.

Chịu trách nhiệm:
- Nạp file CSV, kiểm tra schema và xác thực mã băm SHA-256.
- Phân chia dữ liệu thành Clean Holdout / Development Pool / Train / Validation
  theo nhóm người dùng (User_ID) với Group-disjoint Split.
"""

from src.data.loader import DataLoader
from src.data.splitter import DataSplitter

