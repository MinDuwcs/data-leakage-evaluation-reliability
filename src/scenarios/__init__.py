"""
Module thực nghiệm các kịch bản Data Leakage (Leakage Scenarios).

Chứa các kịch bản thực nghiệm chuyên biệt do Captain đảm nhận:
- F3: Post-event Feature Leakage (Rò rỉ đặc trưng hậu nghiệm / thời gian khả dụng).
- F4: Target Encoding Leakage (Rò rỉ nhãn tập kiểm thử qua mã hóa danh mục).
"""

from src.scenarios.f3_post_event import F3PostEventExperiment
from src.scenarios.f4_target_encoding import F4TargetEncodingExperiment
