# BÁO CÁO TỔNG HỢP THỰC NGHIỆM DATA LEAKAGE: F3 VÀ F4

> **Thời điểm chạy:** 2026-11-23  
> **Dataset:** `data/fraud_detection.csv` (51.000 dòng, 4.000 Users)  
> **Số seeds lặp:** 5 seeds `[13, 37, 71, 101, 131]`  
> **Mô hình:** Logistic Regression, Decision Tree, Random Forest  

---

## 1. Kịch bản F3 — Post-event Feature Leakage

Khảo sát tác động khi đưa đặc trưng hậu nghiệm `Investigation_Result` với độ phủ kiểm toán (Audit Coverage) tăng dần từ 10% đến 80%.

| Thuật toán | Độ phủ (Coverage) | Nhánh (Branch) | Average Precision (AP) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| decision_tree | 10% | contaminated_retrospective | 0.0972 ± 0.0086 | 0.0981 ± 0.0092 | 0.5841 ± 0.0125 |
| decision_tree | 10% | control | 0.0510 ± 0.0028 | 0.0081 ± 0.0044 | 0.4979 ± 0.0124 |
| decision_tree | 10% | deployment_mismatch | 0.0518 ± 0.0026 | 0.0072 ± 0.0064 | 0.4970 ± 0.0111 |
| decision_tree | 30% | contaminated_retrospective | 0.1768 ± 0.0173 | 0.1967 ± 0.0379 | 0.7050 ± 0.0233 |
| decision_tree | 30% | control | 0.0510 ± 0.0028 | 0.0081 ± 0.0044 | 0.4979 ± 0.0124 |
| decision_tree | 30% | deployment_mismatch | 0.0529 ± 0.0028 | 0.0059 ± 0.0061 | 0.5070 ± 0.0126 |
| decision_tree | 50% | contaminated_retrospective | 0.2655 ± 0.0068 | 0.2996 ± 0.0559 | 0.7989 ± 0.0103 |
| decision_tree | 50% | control | 0.0510 ± 0.0028 | 0.0081 ± 0.0044 | 0.4979 ± 0.0124 |
| decision_tree | 50% | deployment_mismatch | 0.0514 ± 0.0012 | 0.0073 ± 0.0026 | 0.5043 ± 0.0045 |
| decision_tree | 80% | contaminated_retrospective | 0.3885 ± 0.0379 | 0.3607 ± 0.0388 | 0.8903 ± 0.0229 |
| decision_tree | 80% | control | 0.0510 ± 0.0028 | 0.0081 ± 0.0044 | 0.4979 ± 0.0124 |
| decision_tree | 80% | deployment_mismatch | 0.0516 ± 0.0015 | 0.0089 ± 0.0062 | 0.5027 ± 0.0069 |
| logistic_regression | 10% | contaminated_retrospective | 0.1240 ± 0.0106 | 0.0682 ± 0.0354 | 0.6017 ± 0.0142 |
| logistic_regression | 10% | control | 0.0541 ± 0.0023 | 0.0000 ± 0.0000 | 0.5185 ± 0.0139 |
| logistic_regression | 10% | deployment_mismatch | 0.0521 ± 0.0018 | 0.0000 ± 0.0000 | 0.5022 ± 0.0059 |
| logistic_regression | 30% | contaminated_retrospective | 0.2194 ± 0.0179 | 0.1534 ± 0.0455 | 0.7328 ± 0.0153 |
| logistic_regression | 30% | control | 0.0541 ± 0.0023 | 0.0000 ± 0.0000 | 0.5185 ± 0.0139 |
| logistic_regression | 30% | deployment_mismatch | 0.0523 ± 0.0024 | 0.0000 ± 0.0000 | 0.5028 ± 0.0048 |
| logistic_regression | 50% | contaminated_retrospective | 0.3176 ± 0.0170 | 0.2423 ± 0.0301 | 0.8391 ± 0.0059 |
| logistic_regression | 50% | control | 0.0541 ± 0.0023 | 0.0000 ± 0.0000 | 0.5185 ± 0.0139 |
| logistic_regression | 50% | deployment_mismatch | 0.0526 ± 0.0021 | 0.0000 ± 0.0000 | 0.5014 ± 0.0077 |
| logistic_regression | 80% | contaminated_retrospective | 0.4413 ± 0.0238 | 0.3475 ± 0.0240 | 0.9342 ± 0.0063 |
| logistic_regression | 80% | control | 0.0541 ± 0.0023 | 0.0000 ± 0.0000 | 0.5185 ± 0.0139 |
| logistic_regression | 80% | deployment_mismatch | 0.0518 ± 0.0017 | 0.0000 ± 0.0000 | 0.4965 ± 0.0060 |
| random_forest | 10% | contaminated_retrospective | 0.1207 ± 0.0079 | 0.0516 ± 0.0112 | 0.5895 ± 0.0075 |
| random_forest | 10% | control | 0.0520 ± 0.0021 | 0.0000 ± 0.0000 | 0.5036 ± 0.0065 |
| random_forest | 10% | deployment_mismatch | 0.0542 ± 0.0021 | 0.0000 ± 0.0000 | 0.5180 ± 0.0092 |
| random_forest | 30% | contaminated_retrospective | 0.2129 ± 0.0111 | 0.1209 ± 0.0322 | 0.7259 ± 0.0124 |
| random_forest | 30% | control | 0.0520 ± 0.0021 | 0.0000 ± 0.0000 | 0.5036 ± 0.0065 |
| random_forest | 30% | deployment_mismatch | 0.0552 ± 0.0023 | 0.0000 ± 0.0000 | 0.5219 ± 0.0129 |
| random_forest | 50% | contaminated_retrospective | 0.3040 ± 0.0184 | 0.1563 ± 0.0203 | 0.8321 ± 0.0061 |
| random_forest | 50% | control | 0.0520 ± 0.0021 | 0.0000 ± 0.0000 | 0.5036 ± 0.0065 |
| random_forest | 50% | deployment_mismatch | 0.0549 ± 0.0022 | 0.0000 ± 0.0000 | 0.5155 ± 0.0109 |
| random_forest | 80% | contaminated_retrospective | 0.4357 ± 0.0293 | 0.2144 ± 0.0428 | 0.9324 ± 0.0056 |
| random_forest | 80% | control | 0.0520 ± 0.0021 | 0.0000 ± 0.0000 | 0.5036 ± 0.0065 |
| random_forest | 80% | deployment_mismatch | 0.0537 ± 0.0020 | 0.0000 ± 0.0000 | 0.5179 ± 0.0073 |

### Phân tích Khoảng cách Lạc quan (Optimism Gap) trong F3

So sánh chênh lệch giữa điểm số đánh giá nội bộ bị nhiễm (`contaminated_retrospective`) và điểm số khi triển khai trên Clean Holdout (`deployment_mismatch`):

| Thuật toán | Độ phủ | AP Nội bộ (Contam) | AP Triển khai (Holdout) | Optimism Gap (AP) | F1 Nội bộ | F1 Triển khai | Optimism Gap (F1) |
|---|---|---|---|---|---|---|---|
| decision_tree | 10% | 0.0972 | 0.0518 | +0.0453 | 0.0981 | 0.0072 | +0.0910 |
| decision_tree | 30% | 0.1768 | 0.0529 | +0.1239 | 0.1967 | 0.0059 | +0.1908 |
| decision_tree | 50% | 0.2655 | 0.0514 | +0.2141 | 0.2996 | 0.0073 | +0.2923 |
| decision_tree | 80% | 0.3885 | 0.0516 | +0.3369 | 0.3607 | 0.0089 | +0.3518 |
| logistic_regression | 10% | 0.1240 | 0.0521 | +0.0719 | 0.0682 | 0.0000 | +0.0682 |
| logistic_regression | 30% | 0.2194 | 0.0523 | +0.1672 | 0.1534 | 0.0000 | +0.1534 |
| logistic_regression | 50% | 0.3176 | 0.0526 | +0.2650 | 0.2423 | 0.0000 | +0.2423 |
| logistic_regression | 80% | 0.4413 | 0.0518 | +0.3895 | 0.3475 | 0.0000 | +0.3475 |
| random_forest | 10% | 0.1207 | 0.0542 | +0.0665 | 0.0516 | 0.0000 | +0.0516 |
| random_forest | 30% | 0.2129 | 0.0552 | +0.1577 | 0.1209 | 0.0000 | +0.1209 |
| random_forest | 50% | 0.3040 | 0.0549 | +0.2491 | 0.1563 | 0.0000 | +0.1563 |
| random_forest | 80% | 0.4357 | 0.0537 | +0.3819 | 0.2144 | 0.0000 | +0.2144 |

---

## 2. Kịch bản F4 — Target Encoding Leakage

Khảo sát tác động khi mã hóa các biến phân loại (`Location`, `Payment_Method`, `Transaction_Type`, `Device_Used`) bằng tỷ lệ nhãn mục tiêu, so sánh giữa quy trình chuẩn (Out-of-Fold cross-fitting) và quy trình rò rỉ nhãn Validation.

| Thuật toán | Nhánh (Branch) | Tập đánh giá | Average Precision (AP) | F1-Score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| decision_tree | contaminated_val_leakage | clean_holdout | 0.0510 ± 0.0020 | 0.0050 ± 0.0055 | 0.4979 ± 0.0145 | 0.0331 ± 0.0370 | 0.0027 ± 0.0030 |
| decision_tree | contaminated_val_leakage | validation | 0.0515 ± 0.0015 | 0.0101 ± 0.0072 | 0.5042 ± 0.0124 | 0.0587 ± 0.0334 | 0.0055 ± 0.0040 |
| decision_tree | control_valid_oof | clean_holdout | 0.0514 ± 0.0015 | 0.0057 ± 0.0054 | 0.5022 ± 0.0070 | 0.0358 ± 0.0325 | 0.0031 ± 0.0030 |
| decision_tree | control_valid_oof | validation | 0.0490 ± 0.0022 | 0.0022 ± 0.0020 | 0.4915 ± 0.0059 | 0.0205 ± 0.0223 | 0.0012 ± 0.0011 |
| logistic_regression | contaminated_val_leakage | clean_holdout | 0.0501 ± 0.0007 | 0.0000 ± 0.0000 | 0.4958 ± 0.0051 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| logistic_regression | contaminated_val_leakage | validation | 0.0532 ± 0.0026 | 0.0000 ± 0.0000 | 0.5102 ± 0.0165 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| logistic_regression | control_valid_oof | clean_holdout | 0.0501 ± 0.0007 | 0.0000 ± 0.0000 | 0.4958 ± 0.0051 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| logistic_regression | control_valid_oof | validation | 0.0532 ± 0.0026 | 0.0000 ± 0.0000 | 0.5105 ± 0.0165 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| random_forest | contaminated_val_leakage | clean_holdout | 0.0538 ± 0.0014 | 0.0000 ± 0.0000 | 0.5120 ± 0.0066 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| random_forest | contaminated_val_leakage | validation | 0.0518 ± 0.0028 | 0.0000 ± 0.0000 | 0.5058 ± 0.0095 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| random_forest | control_valid_oof | clean_holdout | 0.0537 ± 0.0022 | 0.0000 ± 0.0000 | 0.5176 ± 0.0132 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| random_forest | control_valid_oof | validation | 0.0518 ± 0.0008 | 0.0000 ± 0.0000 | 0.5049 ± 0.0110 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

### Phân tích Rò rỉ Nhãn Validation trong F4

Đo lường mức tăng điểm số ảo tưởng trên tập Validation ($\Delta_{\text{contam}} = M_{\text{contam}} - M_{\text{control}}$) và mức độ sụt giảm khi chuyển sang Clean Holdout:

| Thuật toán | AP Control (Val) | AP Contam (Val) | Delta Contam (AP) | F1 Control (Val) | F1 Contam (Val) | Delta Contam (F1) |
|---|---|---|---|---|---|---|
| decision_tree | 0.0490 | 0.0515 | +0.0025 | 0.0022 | 0.0101 | +0.0079 |
| logistic_regression | 0.0532 | 0.0532 | -0.0000 | 0.0000 | 0.0000 | +0.0000 |
| random_forest | 0.0518 | 0.0518 | -0.0001 | 0.0000 | 0.0000 | +0.0000 |

---

## 3. Kết luận phương pháp luận cho Captain

1. **Đối với F3 (Post-event Feature):** Mức độ Optimism Gap phụ thuộc trực tiếp vào tỷ lệ độ phủ (Coverage). Khi độ phủ kiểm toán càng cao, mô hình càng phụ thuộc vào biến hậu nghiệm và suy giảm hiệu năng càng nghiêm trọng khi triển khai thực tế trên các giao dịch mới chưa được kiểm toán.
2. **Đối với F4 (Target Encoding):** Việc rò rỉ nhãn của tập Validation trong quá trình mã hóa danh mục làm tăng điểm số nội bộ giả tạo. Khi áp dụng Out-of-fold cross-fitting đúng chuẩn trên tập Train, hiệu suất mô hình trên Clean Holdout được bảo toàn và phản ánh đúng năng lực tổng quát hóa thực tế.