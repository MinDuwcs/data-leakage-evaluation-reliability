# Ảnh hưởng của Data Leakage đến độ chính xác và khả năng tổng quát hóa của mô hình Machine Learning

**Tên tiếng Anh tham khảo:** An Experimental Study of the Impact of Data Leakage on the Evaluation Performance and Generalization of Machine Learning Models for Classification

---

## 1. Tổng quan nghiên cứu

Trong quy trình phát triển Machine Learning, hiệu suất của mô hình không chỉ phụ thuộc vào thuật toán mà còn chịu ảnh hưởng trực tiếp từ chất lượng dữ liệu, quy trình tiền xử lý, phương pháp phân chia dữ liệu và thiết kế đánh giá thực nghiệm.

**Data Leakage** (rò rỉ dữ liệu) là hiện tượng thông tin không hợp lệ, thông tin ngoài luồng, hoặc thông tin không nên có sẵn tại thời điểm dự đoán trong thực tế bị đưa vào quá trình huấn luyện và đánh giá mô hình. Điều này dẫn đến sự sai lệch nghiêm trọng: mô hình có thể đạt điểm số đánh giá rất cao trên tập kiểm thử nội bộ nhưng lại suy giảm hiệu năng đột ngột khi triển khai trên dữ liệu thực tế.

Dự án này tập trung nghiên cứu thực nghiệm hiện tượng Data Leakage thông qua bài toán phân loại giao dịch gian lận tài chính (Financial Fraud Detection). Mục tiêu cốt lõi là đo lường mức độ sai lệch đánh giá giữa quy trình chuẩn (Baseline) và các kịch bản có rò rỉ dữ liệu, từ đó làm rõ mức độ ảnh hưởng của từng dạng Leakage trên các kiến trúc thuật toán khác nhau.

---

## 2. Câu hỏi nghiên cứu (Research Questions)

Nghiên cứu tập trung giải quyết câu hỏi vấn đề trung tâm:

> **Việc xuất hiện Data Leakage trong quá trình chuẩn bị dữ liệu và xây dựng mô hình làm sai lệch kết quả đánh giá Machine Learning đến mức nào, và mức độ ảnh hưởng đó có khác biệt giữa các thuật toán hay không?**

Các câu hỏi cụ thể bao gồm:

1. **Mức độ sai lệch đánh giá (RQ1):** Mỗi cơ chế leakage làm thay đổi phép đánh giá nội bộ như thế nào so với quy trình hợp lệ?
2. **Độ nhạy của thuật toán (RQ2):** Mức độ ảnh hưởng của từng dạng Leakage có khác nhau giữa các thuật toán Machine Learning thuộc các họ mô hình khác nhau (tuyến tính, cây quyết định đơn lẻ, mô hình tập hợp) hay không?
3. **Khả năng tổng quát hóa (RQ3):** Điểm số đánh giá trên dữ liệu bị ảnh hưởng bởi Leakage chênh lệch bao nhiêu so với hiệu suất thực tế trên tập kiểm thử độc lập, sạch hoàn toàn (Clean Holdout)?
4. **Biện pháp phòng ngừa (RQ4):** Những quy tắc phân tách dữ liệu, fit scope và kiểm soát đặc trưng nào có thể ngăn chặn triệt để từng loại Leakage?

---

## 3. Phạm vi và Thiết kế thực nghiệm

### 3.1. Bài toán và Dữ liệu

- **Bài toán:** Phân loại nhị phân phát hiện giao dịch gian lận tài chính (Fraud Detection).
- **Dataset:** Fraud Detection Dataset (Kaggle: `ranjitmandal/fraud-detection-dataset-csv`).
- **File lưu trữ:** `data/fraud_detection.csv`.
- **Thông số kỹ thuật thực tế (sau kiểm toán dữ liệu):**
  - Kích thước: 51.000 dòng giao dịch, 12 cột đặc trưng.
  - Cột mục tiêu (`Fraudulent`): 48.490 giao dịch hợp lệ (95,08%) và 2.510 giao dịch gian lận (4,92%).
  - Cột định danh và nhóm: `Transaction_ID` (mã giao dịch), `User_ID` (4.000 khách hàng duy nhất; 100% khách hàng có từ 2 đến 28 giao dịch).
  - Đặc trưng số: `Transaction_Amount`, `Time_of_Transaction` (giờ trong ngày), `Account_Age`, `Number_of_Transactions_Last_24H`, `Previous_Fraudulent_Transactions`.
  - Đặc trưng phân loại: `Transaction_Type`, `Device_Used`, `Location`, `Payment_Method`.
  - Dữ liệu khuyết thiếu: Xuất hiện giá trị khuyết thiếu (~5%) ở 5 cột (`Transaction_Amount`, `Time_of_Transaction`, `Device_Used`, `Location`, `Payment_Method`), đòi hỏi quy trình điền khuyết (Imputation) nghiêm ngặt trước khi chuẩn hóa và mã hóa.

### 3.2. Ba kịch bản Data Leakage khảo sát

Nghiên cứu mô phỏng có đối chứng 3 dạng rò rỉ dữ liệu phổ biến:

1. **Preprocessing Leakage (Rò rỉ tiền xử lý):**
   - Vi phạm ranh giới phân tách dữ liệu trong các bước chuẩn hóa (StandardScaler), điền khuyết thiếu (Imputation), cân bằng dữ liệu (SMOTE), hoặc chọn lọc đặc trưng có giám sát (Supervised Feature Selection).
2. **Feature Leakage (Rò rỉ đặc trưng):**
   - Đưa vào mô hình các biến proxy tạo từ nhãn mục tiêu có nhiễu (Noisy Target), biến kết quả điều tra hậu nghiệm (Post-event Feature), hoặc mã hóa mục tiêu (Target Encoding) sử dụng nhãn của tập kiểm thử.
3. **Duplicate / User-level Leakage (Rò rỉ trùng lặp và người dùng):**
   - Phân chia ngẫu nhiên (Random Split) khiến các bản ghi trùng lặp (881 dòng exact duplicate) hoặc giao dịch của cùng một khách hàng (`User_ID`) xuất hiện đồng thời ở cả tập Train và tập Test. Đối chứng với phương pháp phân chia theo nhóm khách hàng (Group-based Split).

### 3.3. Thuật toán Machine Learning đánh giá

Nghiên cứu lựa chọn 3 thuật toán đại diện cho các trường phái mô hình khác nhau:

- **Logistic Regression:** Mô hình tuyến tính đơn giản, dễ giải thích, làm mốc đối chuẩn tuyến tính.
- **Decision Tree (CART):** Mô hình phi tuyến dựa trên luật phân nhánh đơn lẻ, nhạy cảm với biến động dữ liệu.
- **Random Forest:** Mô hình tập hợp (Ensemble / Bagging) kết hợp nhiều cây quyết định, có khả năng chống overfitting tốt hơn.

### 3.4. Nguyên tắc đối chứng thực nghiệm

- **Baseline Pipeline (`CONTROL_VALID`):** Dữ liệu được chia trước theo nhóm (`User_ID`), toàn bộ bước tiền xử lý (Imputer, Scaler, Encoder) chỉ được fit trên tập Train và transform sang tập Validation/Test.
- **Kịch bản rò rỉ (`CONTAMINATED_EVAL`):** Thiết lập các pipeline vi phạm nguyên tắc phân chia dữ liệu theo từng dạng Leakage cụ thể trên tập phát triển (Development pool).
- **Tập kiểm thử độc lập (`CLEAN_HOLDOUT`):** Khóa độc lập 20% dữ liệu sạch theo nhóm người dùng (Group-disjoint Holdout), không tham gia vào bất kỳ khâu tiền xử lý, chọn biến, tuning hay leakage injection nào.
- **Chỉ số đánh giá:** Tập trung vào các chỉ số phù hợp với dữ liệu mất cân bằng: Average Precision (AP / PR-AUC), ROC-AUC, F1-Score, Precision, Recall và Confusion Matrix.

---

## 4. Cấu trúc Repository

```text
.
├── data/
│   └── fraud_detection.csv                          # Dữ liệu giao dịch gian lận (51.000 dòng)
├── MarkDown/
│   ├── dataset_audit.md                              # Báo cáo kiểm toán kỹ thuật dữ liệu (Gate A01–A08)
│   ├── experimental_protocol_data_leakage_v2.md      # Thiết kế thực nghiệm chi tiết v2.0 (Dataset mới)
│   ├── experimental_protocol_data_leakage_v1.md      # Thiết kế thực nghiệm v1.0 (Lưu trữ lịch sử)
│   ├── KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md # Kế hoạch điều chỉnh đề tài v3.0
│   └── data_leakage_research_handoff.md              # Tài liệu bàn giao nghiên cứu nền tảng
├── model/
│   └── scripts/
│       └── preprocess.ipynb                          # Notebook phục vụ tiền xử lý và thực nghiệm
├── .gitignore                                       # Cấu hình bỏ qua cache, virtualenv
└── README.md                                        # Tài liệu tổng quan dự án
```

---

## 5. Trạng thái hiện tại của đề tài

- **Giai đoạn hiện tại:** Execution P3 — Đã hoàn thành triển khai thực nghiệm đối chứng trên 2 kịch bản rò rỉ dữ liệu:
  - F3: Post-event Feature Leakage (khảo sát theo 4 mức độ phủ kiểm toán: 10%, 30%, 50%, 80%).
  - F4: Target Encoding Leakage (đối chứng Out-of-Fold vs rò rỉ nhãn Validation).
- **Quy mô thực nghiệm:** 5 seeds lặp lại `[13, 37, 71, 101, 131]`, đánh giá trên 3 thuật toán đại diện: Logistic Regression, Decision Tree và Random Forest.
- **Vị trí lưu trữ mã nguồn chi tiết:** Toàn bộ pipeline, scripts thực thi, dữ liệu kết quả raw và biểu đồ trực quan hóa được lưu trữ và phát triển tại nhánh `feature-leakage`.

---

## 6. Kết quả và Đầu ra thực nghiệm (Results & Deliverables)

### 6.1. Tóm tắt kết quả kịch bản F3 — Post-event Feature Leakage

Khảo sát tác động khi đưa đặc trưng hậu nghiệm `Investigation_Result` (kết quả thanh tra gian lận sau giao dịch) vào mô hình với độ phủ kiểm toán (Audit Coverage) tăng dần:

- **Hiện tượng lạm phát điểm số (Metric Inflation):** Khi độ phủ kiểm toán tăng từ 10% lên 80%, điểm Average Precision (AP) nội bộ bị thổi phồng giả tạo từ mức ~0.12 lên đến ~0.44 trên Logistic Regression và Random Forest (so với Baseline sạch chỉ đạt ~0.05).
- **Khoảng cách lạc quan (Optimism Gap):** Khi mang mô hình đã huấn luyện có đặc trưng hậu nghiệm triển khai trên tập kiểm thử độc lập sạch (Clean Holdout - nơi giao dịch mới chưa từng được thanh tra), điểm AP sụt giảm đột ngột về mức ngẫu nhiên (~0.05). Optimism Gap nới rộng tuyến tính theo độ phủ kiểm toán (đạt đỉnh chênh lệch +0.3895 trên Logistic Regression ở mức 80% coverage).
- **Độ nhạy thuật toán:** Cả 3 họ mô hình (tuyến tính, cây quyết định, rừng ngẫu nhiên) đều bị đánh lừa nghiêm trọng và phụ thuộc gần như tuyệt đối vào đặc trưng hậu nghiệm khi nó xuất hiện.

Bảng tổng hợp Optimism Gap tiêu biểu (AP và F1) trên nhánh `feature-leakage`:

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

### 6.2. Tóm tắt kết quả kịch bản F4 — Target Encoding Leakage

Khảo sát tác động khi mã hóa các biến phân loại danh mục (`Location`, `Payment_Method`, `Transaction_Type`, `Device_Used`) bằng xác suất mục tiêu:

- **Hiện tượng rò rỉ nhãn Validation:** Khi quy trình mã hóa sử dụng nhãn của toàn bộ tập phát triển (bao gồm cả tập Validation), điểm số nội bộ trên Validation tăng ảo (Delta Contam đạt +0.0025 AP và +0.0079 F1 trên Decision Tree).
- **Hiệu quả phòng ngừa của Out-of-Fold (OOF):** Khi áp dụng Out-of-Fold cross-fitting đúng chuẩn (chỉ fit trên các fold con của Train và transform sang Validation), hiện tượng ghi nhớ nhãn được triệt tiêu, bảo toàn điểm số đánh giá khớp hoàn toàn với năng lực tổng quát hóa thực tế trên Clean Holdout.

### 6.3. Danh mục các đầu ra nghiên cứu đã hoàn thành (Deliverables)

Toàn bộ các thành phần dưới đây đã được xây dựng hoàn chỉnh và lưu trữ tại nhánh `feature-leakage`:

1. **Hệ thống mã nguồn mô-đun hóa (`src/`):**
   - `src/config/`: Quản lý cấu hình đường dẫn, danh sách seeds, siêu tham số và ngưỡng leakage.
   - `src/data/`: `loader.py` (tải và kiểm toán kiểu dữ liệu), `splitter.py` (Group-based disjoint train/val/holdout split).
   - `src/preprocessing/`: Xây dựng pipeline tiền xử lý độc lập fit-transform, triển khai OOF Target Encoder.
   - `src/models/`: Đóng gói 3 bộ phân loại Logistic Regression, Decision Tree, Random Forest.
   - `src/evaluation/`: Tính toán đa chỉ số (AP, ROC-AUC, F1, Precision, Recall) và trích xuất độ lệch Optimism Gap.
   - `src/scenarios/`: Triển khai độc lập kịch bản F3 (`f3_post_event.py`) và F4 (`f4_target_encoding.py`).
2. **Scripts điều phối và trực quan hóa:**
   - `run_experiments.py`: Script chạy tự động toàn bộ thực nghiệm đa seed, kết xuất dữ liệu CSV.
   - `plot_results.py`: Script sinh biểu đồ xuất bản khoa học chất lượng cao (DPI 300).
3. **Báo cáo và biểu đồ kết quả (`results/`):**
   - `results/f3_post_event_results.csv`: Dữ liệu đo lường chi tiết 5 seeds cho F3.
   - `results/f4_target_encoding_results.csv`: Dữ liệu đo lường chi tiết 5 seeds cho F4.
   - `results/summary_f3_f4_report.md`: Báo cáo phân tích tổng hợp đầy đủ số liệu trung bình và độ lệch chuẩn.
   - `results/figures/`:
     - `f3_dose_response_optimism_gap.png`: Biểu đồ liều lượng - đáp ứng của Optimism Gap theo Coverage.
     - `f3_metric_inflation_comparison.png`: So sánh mức độ lạm phát điểm số AP giữa các mô hình.
     - `f4_target_encoding_delta.png`: Trực quan hóa độ lệch điểm số khi rò rỉ nhãn Validation.

---

## 7. Điều hướng nhánh mã nguồn (Branch Navigation)

Kho lưu trữ được tổ chức theo chiến lược phân nhánh chức năng:

- **Nhánh `main`:** Lưu trữ tài liệu tổng quan, phạm vi nghiên cứu và tóm tắt kết quả thực nghiệm.
- **Nhánh `feature-leakage`:** Lưu trữ toàn bộ mã nguồn thực thi, kịch bản kiểm thử, dữ liệu raw kết quả và các biểu đồ phân tích.

Để kiểm tra chi tiết mã nguồn và tái lập thực nghiệm, thực hiện chuyển nhánh:

```bash
# Chuyển sang nhánh chứa mã nguồn và kết quả thực nghiệm
git checkout feature-leakage
```

---

## 8. Hướng dẫn thiết lập môi trường

### 8.1. Yêu cầu hệ thống

- Hệ điều hành: Linux / macOS / Windows
- Ngôn ngữ: Python 3.11+
- Công cụ quản lý phiên bản: Git

### 8.2. Cài đặt môi trường ảo

Khuyến nghị sử dụng `venv` hoặc `conda` để cô lập môi trường thực nghiệm:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường trên Linux / macOS
source .venv/bin/activate

# Kích hoạt môi trường trên Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 8.3. Cài đặt các thư viện phụ thuộc

Cài đặt các gói thư viện khoa học dữ liệu và học máy:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter notebook
```

### 8.4. Tái lập thực nghiệm (trên nhánh feature-leakage)

Sau khi chuyển sang nhánh `feature-leakage`, thực hiện các lệnh sau để chạy thực nghiệm và xuất biểu đồ:

```bash
# Chạy thực nghiệm song song F3 và F4 với 5 seeds
python run_experiments.py

# Sinh biểu đồ trực quan hóa kết quả
python plot_results.py
```

---

## 9. Tài liệu tham chiếu

- [Báo cáo kiểm toán dữ liệu: MarkDown/dataset_audit.md](MarkDown/dataset_audit.md)
- [Kế hoạch điều chỉnh nghiên cứu v3.0: MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md](MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md)
- [Tài liệu bàn giao nghiên cứu ban đầu: MarkDown/data_leakage_research_handoff.md](MarkDown/data_leakage_research_handoff.md)
- [Thiết kế thực nghiệm v1.0 (Lịch sử): MarkDown/experimental_protocol_data_leakage_v1.md](MarkDown/experimental_protocol_data_leakage_v1.md)
