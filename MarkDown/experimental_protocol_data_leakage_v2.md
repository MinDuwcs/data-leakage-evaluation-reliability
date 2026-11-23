# EXPERIMENTAL PROTOCOL v2.0 — Data Leakage trong Machine Learning

> **Ngày ban hành:** 2026-11-21  
> **Phiên bản:** v2.0 — Chuẩn hóa trên Dataset mới (`ranjitmandal/fraud-detection-dataset-csv`)  
> **Trạng thái:** `PROPOSED — READY FOR FREEZE` (Đã hoàn thành Dataset Audit Gate A01–A08).  
> **Tài liệu tham chiếu:** `KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md` (v3.0), `dataset_audit.md`.  
> **Tài liệu lịch sử đã thay thế:** `experimental_protocol_data_leakage_v1.md` (v1.0 ULB cũ).  
> **Nguyên tắc cốt lõi:** Chưa viết code chạy thực nghiệm khi chưa khóa Protocol. Mọi số liệu trong protocol này dựa trên số liệu thực tế đã audit từ `data/fraud_detection.csv`.

---

## 0. Tóm tắt điều hành (Executive Summary)

- **Đề tài:** Nghiên cứu thực nghiệm tác động của Data Leakage đến độ tin cậy của đánh giá mô hình Machine Learning trong bài toán phát hiện gian lận giao dịch tài chính.
- **Dữ liệu thực nghiệm chính thức:** `data/fraud_detection.csv` (SHA-256: `9a009cd6d20720b2faed8a8b4d5eb45d99008d6bc4fc77dd9b5a104166c62b9d`).
- **3 Nhóm rò rỉ:** Preprocessing Leakage, Feature Leakage, Duplicate/User-level Leakage.
- **3 Mô hình:** Logistic Regression, Decision Tree, Random Forest.
- **8 Kịch bản chi tiết:** P1, P3, P6, F2, F3, F4, D1, D3 (Toàn bộ 8 kịch bản đã được xác nhận `FEASIBLE` sau audit).

---

## 1. Câu hỏi nghiên cứu và Giả thuyết thực nghiệm

| ID | Câu hỏi nghiên cứu | Giả thuyết thực nghiệm |
|---|---|---|
| **RQ1** | Mỗi cơ chế rò rỉ dữ liệu làm sai lệch điểm số đánh giá nội bộ như thế nào so với quy trình hợp lệ? | **H1:** Các kịch bản rò rỉ dữ liệu tạo ra điểm số đánh giá nội bộ lạc quan quá mức (optimism bias) so với đánh giá hợp lệ. |
| **RQ2** | Mức độ ảnh hưởng của từng loại rò rỉ có khác nhau giữa Logistic Regression, Decision Tree và Random Forest hay không? | **H2:** Mô hình tuyến tính nhạy cảm hơn với Preprocessing Leakage (scaling); cây quyết định và rừng ngẫu nhiên nhạy cảm hơn với Duplicate/User-level Leakage và Feature Leakage. |
| **RQ3** | Hiệu suất trên tập đánh giá bị nhiễm (Contaminated Eval) sai lệch bao nhiêu so với tập kiểm thử độc lập, sạch hoàn toàn (Clean Holdout)? | **H3:** Điểm số cao trên tập bị nhiễm sụt giảm mạnh khi kiểm thử trên tập Clean Holdout (tồn tại một Optimism Gap lớn). |
| **RQ4** | Quy tắc phân chia dữ liệu, fit scope và kiểm soát đặc trưng nào có thể loại bỏ hoàn toàn các rò rỉ này? | **H4:** Khóa cứng biến đổi trong Pipeline, tách tập theo nhóm người dùng (`User_ID`) và cô lập hoàn toàn Clean Holdout sẽ ngăn chặn triệt để rò rỉ. |

---

## 2. Thông số Dữ liệu Thực chứng (Từ Dataset Audit)

- **Tổng số quan sát:** 51.000 dòng giao dịch.
- **Nhãn mục tiêu (`Fraudulent`):**
  - Lớp 0 (Hợp lệ): 48.490 dòng (95,08%).
  - Lớp 1 (Gian lận): 2.510 dòng (4,92%). Tỷ lệ mất cân bằng ~ 19,3 : 1.
- **Đặc trưng đầu vào (11 trường):**
  - *Metadata/Loại bỏ khỏi model:* `Transaction_ID`.
  - *Định danh nhóm (Group Key):* `User_ID` (4.000 users, min=2, max=28, avg=12,75 giao dịch/user).
  - *Biến số (Numerical):* `Transaction_Amount` (khuyết 4,94%), `Time_of_Transaction` (khuyết 5,00%), `Account_Age` (khuyết 0%), `Number_of_Transactions_Last_24H` (khuyết 0%), `Previous_Fraudulent_Transactions` (khuyết 0%).
  - *Biến phân loại (Categorical):* `Transaction_Type` (5 nhóm, khuyết 0%), `Device_Used` (4 nhóm, khuyết 4,85%), `Location` (8 nhóm, khuyết 4,99%), `Payment_Method` (5 nhóm, khuyết 4,84%).
- **Trùng lặp tự nhiên:** 881 dòng exact full-row duplicates; không có xung đột nhãn.

---

## 3. Chính sách Phân chia Dữ liệu và Quản trị Holdout

### 3.1. Phân tầng Dữ liệu (Data Hierarchy)

```text
Toàn bộ Dữ liệu (51.000 dòng, 4.000 Users)
│
├── Clean Holdout (H) - 20% Users (~800 Users, ~10.200 dòng)
│   └── KHÓA BẤT BIẾN (Group-disjoint, không fit, không tuning, không leak)
│
└── Development Pool (DEV) - 80% Users (~3.200 Users, ~40.800 dòng)
    ├── Training Set (T) - ~75% DEV (~2.400 Users, ~30.600 dòng)
    └── Validation Set (V) - ~25% DEV (~800 Users, ~10.200 dòng)
```

### 3.2. Quy tắc Chia tập (Split Policy)

1. **Clean Holdout (`H`):** Tách 20% users theo phương pháp `GroupShuffleSplit(groups=User_ID, test_size=0.2, random_state=2026)`. Giữ nguyên tỷ lệ nhãn xấp xỉ bằng cách kiểm tra phân phối `Fraudulent`.
2. **Development Pool (`DEV`):** Chạy thực nghiệm lặp trên 5 seeds độc lập: `[13, 37, 71, 101, 131]`. Với mỗi seed:
   - Trong nhánh **Control (Hợp lệ):** Chia `T` và `V` theo `User_ID` (Group-disjoint). Không một user nào xuất hiện ở cả `T` và `V`.
   - Trong nhánh **Contaminated (Nhiễm):** Tùy theo kịch bản (ví dụ D1 sẽ chia `T'` và `V'` ngẫu nhiên không theo user để đo rò rỉ).
3. **Quy tắc bất biến của `H`:**
   - Tuyệt đối không tính toán thống kê (imputation, mean, variance) có sự tham gia của `H`.
   - `H` chỉ được mở ra đánh giá 1 lần duy nhất sau khi toàn bộ mô hình và tham số đã cố định.

---

## 4. Tiền xử lý Chuẩn (Baseline Preprocessing Contract)

Để xử lý dữ liệu hỗn hợp và missing values trên dataset mới, một pipeline hợp lệ bắt buộc gồm 3 thành phần:

```text
Input Features
  ├── Numeric Features ────> SimpleImputer(median) ──────> StandardScaler() ───┐
  └── Categorical Features ─> SimpleImputer('missing') ───> OneHotEncoder() ────┴─> Model
```

- **Numeric Pipeline:** Điền khuyết bằng `SimpleImputer(strategy='median')`, sau đó chuẩn hóa bằng `StandardScaler()`.
- **Categorical Pipeline:** Điền khuyết bằng `SimpleImputer(strategy='constant', fill_value='Missing')`, sau đó mã hóa bằng `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Nguyên tắc fit:** Toàn bộ `fit()` chỉ được thực hiện trên tập `Train (T)`. Tập `Validation (V)` và `Holdout (H)` chỉ được `transform()`.

---

## 5. Thiết kế Chi tiết 8 Kịch bản Thực nghiệm

### Nhóm 1: Preprocessing Leakage

#### P1 — StandardScaler & Imputation Leakage
- **Cơ chế rò rỉ:** Vi phạm ranh giới fit tiền xử lý trên tập phát triển.
- **Nhánh Control (A-Control):** Fit `SimpleImputer` và `StandardScaler` trên `T`; transform sang `V`.
- **Nhánh Contaminated (A-Leakage):** Fit `SimpleImputer` và `StandardScaler` trên toàn bộ `T ∪ V`; sau đó transform `T` và `V`.
- **Kiểm soát:** Giữ nguyên các dòng `T`, `V`, mô hình và seed. Holdout `H` tuyệt đối không tham gia fit.
- **Thước đo:** $\Delta M = M_{contam}(V) - M_{control}(V)$. Đo mức thay đổi trọng số và dự đoán của Logistic Regression vs Tree models.

#### P3 — SMOTE Leakage
- **Cơ chế rò rỉ:** Thực hiện kỹ thuật cân bằng dữ liệu SMOTE trên toàn bộ pool trước khi phân chia tập.
- **Nhánh Control:** Preprocessing trên `T` -> Áp dụng `SMOTE` chỉ trên `T` -> Train model -> Đánh giá trên `V` nguyên bản (không SMOTE).
- **Nhánh Contaminated:** Áp dụng `SMOTE` trên toàn bộ `DEV (T ∪ V)` -> Chia thành `T'` và `V'` (tập `V'` chứa các mẫu nhân tạo được sinh ra từ láng giềng nằm trong `T'`).
- **Thước đo:** Báo cáo riêng biệt hiệu suất trên `V'` (bị đổi prevalence) và hiệu suất của cả 2 mô hình trên tập `H` nguyên bản.

#### P6 — Supervised Feature Selection Leakage
- **Cơ chế rò rỉ:** Sử dụng nhãn mục tiêu của tập Validation để chọn lọc đặc trưng.
- **Nhánh Control:** Dùng `SelectKBest(score_func=f_classif, k=8)` fit trên `(X_T, y_T)` sau khi encode; chọn đặc trưng cho `T` và `V`.
- **Nhánh Contaminated:** Dùng `SelectKBest` fit trên `(X_{T∪V}, y_{T∪V})`; chọn đặc trưng cho `T` và `V`.
- **Thước đo:** Mức độ trùng lặp đặc trưng được chọn (Jaccard Index) và độ chênh lệch điểm số đánh giá.

---

### Nhóm 2: Feature Leakage

#### F2 — Noisy Target Leakage
- **Cơ chế rò rỉ:** Đưa vào một đặc trưng giả lập mang thông tin nhãn mục tiêu với mức nhiễu xác định.
- **Cách sinh đặc trưng:** Tạo cột `proxy_z` nhị phân với độ chính xác kiểm soát:
  - $P(z = 1 \mid y = 1) = \text{sensitivity}$
  - $P(z = 1 \mid y = 0) = \text{false\_positive\_rate}$
  - Khảo sát 3 mức nhiễu: Thấp (Noise 5%), Trung bình (Noise 20%), Cao (Noise 40%).
- **Nhánh Control:** Huấn luyện chỉ trên các đặc trưng hợp lệ gốc.
- **Nhánh Contaminated:** Bổ sung cột `proxy_z` vào `T` và `V`.
- **Quy tắc Clean:** Không sinh `proxy_z` từ nhãn của `H`. Trên `H`, đánh giá mô hình bằng chính sách missing feature hoặc so sánh với baseline.

#### F3 — Post-event Feature Leakage
- **Cơ chế rò rỉ:** Sử dụng đặc trưng chỉ xuất hiện sau khi giao dịch đã kết thúc và được điều tra.
- **Kịch bản thực tế:** Giả lập cột `Investigation_Flag` hoặc stress-test đặc trưng `Previous_Fraudulent_Transactions` với giả định thông tin được cập nhật tức thời sau kết luận gian lận.
- **Nhánh Control:** Chỉ dùng các đặc trưng có sẵn tại thời điểm quẹt thẻ/giao dịch.
- **Nhánh Contaminated:** Đưa thêm đặc trưng hậu nghiệm vào quá trình dự đoán.

#### F4 — Target Encoding Leakage
- **Cơ chế rò rỉ:** Mã hóa biến phân loại bằng tỷ lệ gian lận của nhãn kiểm thử.
- **Đối tượng áp dụng:** Đặc trưng `Location` (8 danh mục) và `Payment_Method` (5 danh mục).
- **Nhánh Control:** Tính Target Encoding trên `T` bằng out-of-fold cross-fitting; transform `V` bằng mapping học từ `T`.
- **Nhánh Contaminated:** Tính Target Encoding mapping trực tiếp từ `(X_{T∪V}, y_{T∪V})`; transform `V` bằng mapping này.
- **Thước đo:** So sánh điểm số đánh giá giữa out-of-fold encoding hợp lệ và contaminated target encoding.

---

### Nhóm 3: Duplicate / User-level Leakage

#### D1 — User-level Group Leakage vs Random Split
- **Cơ chế rò rỉ:** Cùng một khách hàng (`User_ID`) có hành vi gian lận/hợp lệ lặp lại xuất hiện ở cả Train và Validation khi chia ngẫu nhiên.
- **Nhánh Control:** `GroupShuffleSplit` theo `User_ID` (Disjoint). User ở `T` không xuất hiện ở `V`.
- **Nhánh Contaminated:** `train_test_split` ngẫu nhiên theo dòng (Random Row-wise Split). Đo lường tỷ lệ user overlap và duplicate record overlap (từ 881 dòng duplicate tự nhiên).
- **Thước đo:** Chênh lệch hiệu suất giữa Group-based Split và Random Split; đánh giá khả năng tổng quát hóa trên tập `H` (vốn là các User hoàn toàn mới).

#### D3 — Duplicate Ratio Sensitivity
- **Cơ chế rò rỉ:** Bơm bản sao của các dòng trong tập Validation vào tập Train với tỷ lệ tăng dần.
- **Mức độ khảo sát ($q$):** $q \in \{0\%, 5\%, 10\%, 20\%\}$.
- **Nhánh Contaminated:** Chọn ngẫu nhiên $q\%$ các dòng từ `V`, sao chép nguyên vẹn vector `(X, y)` vào `T`.
- **Nhánh Control (Size-matched):** Bổ sung số lượng dòng tương đương bằng cách nhân bản nội bộ các dòng trong `T` (để loại trừ ảnh hưởng của việc tăng kích thước mẫu huấn luyện).
- **Thước đo:** Đường cong phản ứng liều lượng (Dose-response curve) của F1-Score và Average Precision theo mức $q$.

---

## 6. Mô hình và Tiêu chí Đánh giá

### 6.1. Cấu hình Mô hình (Model Configurations)
- **Logistic Regression:** `solver='lbfgs'`, `max_iter=1000`, `random_state=seed`.
- **Decision Tree:** `criterion='gini'`, `max_depth=10`, `min_samples_split=20`, `random_state=seed`.
- **Random Forest:** `n_estimators=100`, `max_depth=12`, `min_samples_split=10`, `n_jobs=-1`, `random_state=seed`.

### 6.2. Chỉ số Đánh giá (Evaluation Metrics)
- **Chỉ số ưu tiên chính:**
  - **Average Precision (AP / PR-AUC):** Đo lường diện tích dưới đường cong Precision-Recall, chỉ số quan trọng nhất cho dữ liệu mất cân bằng 4,92%.
  - **F1-Score, Recall, Precision:** Đánh giá tại ngưỡng chuẩn `threshold = 0.5`.
- **Chỉ số phụ trợ:** ROC-AUC, Accuracy, Confusion Matrix.
- **Báo cáo thống kê:** Mean ± Standard Deviation qua 5 seeds cho mọi nhánh.

---

## 7. Nhật ký Quyết định (Decision Log)

| ID | Ngày | Quyết định | Lý do & Trade-off | Trạng thái |
|---|---|---|---|---|
| D-021 | 2026-11-21 | Khóa Protocol v2.0 trên dataset `ranjitmandal/fraud-detection-dataset-csv` | Thay thế hoàn toàn protocol cũ; phù hợp dữ liệu 51.000 dòng | `PROPOSED — READY FOR FREEZE` |
| D-022 | 2026-11-21 | Bổ sung Pipeline Imputation chuẩn (median cho số, 'Missing' cho chữ) | Xử lý 5 cột bị khuyết ~5% phát hiện trong dataset audit | `PROPOSED — READY FOR FREEZE` |
| D-023 | 2026-11-21 | Chuyển trọng tâm D1 từ hash dòng sang User-level Group Split theo `User_ID` | 100% User có từ 2-28 giao dịch; User leakage là rò rỉ lớn nhất trong thực tế | `PROPOSED — READY FOR FREEZE` |
| D-024 | 2026-11-21 | Áp dụng F4 trực tiếp trên `Location` và `Payment_Method` | Tận dụng biến phân loại thật của dataset, không cần binning nhân tạo | `PROPOSED — READY FOR FREEZE` |
| D-025 | 2026-11-21 | Cố định 5 seeds lặp lại cho DEV: `[13, 37, 71, 101, 131]`, Holdout seed: `2026` | Đảm bảo tính tái lập (Reproducibility) và ước lượng độ biến thiên | `PROPOSED — READY FOR FREEZE` |

