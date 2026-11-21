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

- **Giai đoạn hiện tại:** Planning P1/P2 — Hoàn thành Dataset Audit, chuẩn bị khóa Experimental Protocol v2.0.
- **Trạng thái mã nguồn:** Chưa bước vào giai đoạn viết code huấn luyện mô hình chính thức. Mọi thực nghiệm sẽ được bắt đầu sau khi Experimental Protocol v2.0 được phê duyệt.
- **Quy tắc học thuật:** Không đưa ra kết luận hoặc công bố số liệu khi chưa hoàn thành thực nghiệm có đối chứng và tái lập được kết quả.

---

## 6. Hướng dẫn thiết lập môi trường

### 6.1. Yêu cầu hệ thống

- Hệ điều hành: Linux / macOS / Windows
- Ngôn ngữ: Python 3.11+
- Công cụ quản lý phiên bản: Git

### 6.2. Cài đặt môi trường ảo

Khuyến nghị sử dụng `venv` hoặc `conda` để cô lập môi trường thực nghiệm:

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường trên Linux / macOS
source .venv/bin/activate

# Kích hoạt môi trường trên Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 6.3. Cài đặt các thư viện phụ thuộc

Cài đặt các gói thư viện khoa học dữ liệu và học máy:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter notebook
```

---

## 7. Tài liệu tham chiếu

- [Báo cáo kiểm toán dữ liệu: MarkDown/dataset_audit.md](MarkDown/dataset_audit.md)
- [Kế hoạch điều chỉnh nghiên cứu v3.0: MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md](MarkDown/KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md)
- [Tài liệu bàn giao nghiên cứu ban đầu: MarkDown/data_leakage_research_handoff.md](MarkDown/data_leakage_research_handoff.md)
- [Thiết kế thực nghiệm v1.0 (Lịch sử): MarkDown/experimental_protocol_data_leakage_v1.md](MarkDown/experimental_protocol_data_leakage_v1.md)
