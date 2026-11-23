"""
Module khởi tạo và quản lý các mô hình học máy (Machine Learning Models).

Đóng gói 3 thuật toán đã chốt trong Protocol v2.0:
- Logistic Regression (Mô hình tuyến tính)
- Decision Tree (Mô hình cây quyết định đơn lẻ)
- Random Forest (Mô hình tập hợp đa cây)
"""

from src.models.classifiers import ModelFactory, get_model
