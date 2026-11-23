# BÁO CÁO KIỂM TOÁN DỮ LIỆU (DATASET AUDIT REPORT)

> **File kiểm toán:** `data/fraud_detection.csv`  
> **Ngày thực hiện:** 2026-11-21  
> **Trạng thái:** `COMPLETED` — Đáp ứng Gate A01 đến A08 theo Kế hoạch điều chỉnh v3.0.  
> **Quy tắc:** Mọi số liệu trong báo cáo này được tính toán trực tiếp từ file CSV thực tế, không sử dụng số liệu giả định.

---

## 1. Định danh và Toàn vẹn Tệp (Gate A01)

| Thuộc tính | Giá trị kiểm toán thực tế |
|---|---|
| Đường dẫn file | `data/fraud_detection.csv` |
| Dung lượng file | 3.692.523 bytes (~3,52 MB) |
| Tổng số dòng (kể cả header) | 51.001 dòng |
| Tổng số quan sát (data rows) | 51.000 dòng |
| Tổng số cột (features) | 12 cột |
| Mã băm SHA-256 | `9a009cd6d20720b2faed8a8b4d5eb45d99008d6bc4fc77dd9b5a104166c62b9d` |
| Nguồn gốc khai báo | `ranjitmandal/fraud-detection-dataset-csv` (Kaggle) |

---

## 2. Cấu trúc Đặc trưng và Giá trị Khuyết thiếu (Gate A02, A04, A08)

| Tên cột | Kiểu dữ liệu suy diễn | Số lượng Missing | Tỷ lệ Missing (%) | Số giá trị phân biệt | Phân loại vai trò | Tính khả dụng khi dự đoán |
|---|---|---|---|---|---|---|
| `Transaction_ID` | String (Định danh) | 0 | 0,00% | 50.000 | Metadata | Loại bỏ khỏi model |
| `User_ID` | String/Int (Định danh) | 0 | 0,00% | 4.000 | Group Key | Dùng cho Group Split |
| `Transaction_Amount` | Float (Liên tục) | 2.520 | 4,94% | 44.821 | Numeric Feature | Có sẵn |
| `Transaction_Type` | String (Rời rạc) | 0 | 0,00% | 5 | Categorical Feature | Có sẵn |
| `Time_of_Transaction` | Float (Giờ trong ngày) | 2.552 | 5,00% | 24 | Numeric Feature | Có sẵn |
| `Device_Used` | String (Rời rạc) | 2.473 | 4,85% | 4 | Categorical Feature | Có sẵn |
| `Location` | String (Rời rạc) | 2.547 | 4,99% | 8 | Categorical Feature | Có sẵn |
| `Previous_Fraudulent_Transactions`| Int (Đếm) | 0 | 0,00% | 5 | Numeric Feature | Cần kiểm chứng provenance |
| `Account_Age` | Int (Đếm tháng/ngày) | 0 | 0,00% | 119 | Numeric Feature | Có sẵn |
| `Number_of_Transactions_Last_24H` | Int (Đếm) | 0 | 0,00% | 14 | Numeric Feature | Cần kiểm chứng provenance |
| `Payment_Method` | String (Rời rạc) | 2.469 | 4,84% | 5 | Categorical Feature | Có sẵn |
| `Fraudulent` | Int (0 hoặc 1) | 0 | 0,00% | 2 | Target Label | Nhãn mục tiêu |

**Nhận xét:**
- Có đúng 5 đặc trưng bị khuyết thiếu với tỷ lệ đồng đều xấp xỉ 5%: `Transaction_Amount`, `Time_of_Transaction`, `Device_Used`, `Location`, `Payment_Method`.
- Các biến phân loại gồm: `Transaction_Type` (5 loại), `Device_Used` (4 loại), `Location` (8 địa điểm), `Payment_Method` (5 phương thức).
- `Time_of_Transaction` chỉ mang 24 giá trị phân biệt (0.0 đến 23.0), phản ánh giờ trong ngày chứ không phải chuỗi thời gian timestamp tuyệt đối. Do đó không thể dùng làm mốc chia chuỗi thời gian (Temporal Split) theo ngày tháng.

---

## 3. Phân phối Nhãn Mục tiêu (Gate A03)

- **Cột nhãn:** `Fraudulent`
- **Số ca hợp lệ (`Fraudulent = 0`):** 48.490 giao dịch (chiếm 95,08%)
- **Số ca gian lận (`Fraudulent = 1`):** 2.510 giao dịch (chiếm 4,92%)
- **Tỷ lệ mất cân bằng lớp:** ~ 19,3 : 1
- **Đánh giá mức hỗ trợ (Class Support):** 2.510 mẫu dương đủ lớn để chia thành các tập Stratified Holdout / Development folds mà không lo thiếu mẫu dương (so với chỉ 492 mẫu dương ở dataset ULB cũ).

---

## 4. Kiểm toán Trùng lặp và Cấu trúc Người dùng (Gate A05, A06)

### 4.1. Bản ghi trùng lặp (Duplicates)
- **Exact Full-row Duplicates (Trùng toàn bộ 12 cột kể cả ID):** 881 dòng.
- **Exact X-only Duplicates (Trùng đặc trưng, không xét Transaction_ID):** 881 dòng.
- **Xung đột nhãn (Conflicting Labels - Cùng X nhưng khác nhãn Fraudulent):** 0 nhóm (không có mâu thuẫn nhãn).
- **Trùng Transaction_ID:** Có 1.000 dòng chia sẻ 500 `Transaction_ID` lặp lại (50.000 ID phân biệt trên 51.000 dòng).

### 4.2. Cấu trúc Người dùng (User Structure)
- **Số lượng User duy nhất:** 4.000 người dùng.
- **Số User có từ 2 giao dịch trở lên:** 4.000 / 4.000 người (100,00%).
- **Số lượng giao dịch trên mỗi User:**
  - Tối thiểu: 2 giao dịch.
  - Tối đa: 28 giao dịch (User ID `3134`).
  - Trung bình: 12,75 giao dịch / user.
- **Ý nghĩa kỹ thuật quan trọng:** Khi chia ngẫu nhiên (Random Split) không kiểm soát `User_ID`, chắc chắn xảy ra hiện tượng **User Leakage** (hành vi của cùng một khách hàng xuất hiện ở cả Train và Test). Đây là cơ sở cốt lõi để hiện thực hóa kịch bản D1 (Group-based Split vs Random Split).

---

## 5. Đánh giá Tính khả thi của 8 Kịch bản Thực nghiệm (Gate A07)

| Kịch bản | Tên kịch bản | Trạng thái khả thi | Cơ sở kỹ thuật trên Dataset mới |
|---|---|---|---|
| **P1** | StandardScaler Leakage | `FEASIBLE` | Áp dụng trên các biến số: `Transaction_Amount`, `Account_Age`, `Time_of_Transaction`, `Number_of_Transactions_Last_24H`. Đi kèm Imputation. |
| **P3** | SMOTE Leakage | `FEASIBLE` | Tỷ lệ gian lận 4,92% với 2.510 mẫu dương là điều kiện lý tưởng cho SMOTE. Cần pipeline mã hóa One-Hot/Target Encoding trước khi SMOTE. |
| **P6** | Supervised Feature Selection | `FEASIBLE` | Áp dụng sau khi encode ma trận đặc trưng. Phù hợp để đo việc chọn biến bằng cả nhãn tập Validation. |
| **F2** | Noisy Target Leakage | `FEASIBLE` | Tạo proxy hậu nghiệm từ nhãn `Fraudulent` với các mức nhiễu xác định (5%, 20%, 40%) trên Development pool. |
| **F3** | Post-event Feature Leakage | `FEASIBLE` | Mô phỏng trường hợp điều tra giao dịch bị đưa nhầm vào thời điểm dự đoán, hoặc stress-test biến `Previous_Fraudulent_Transactions`. |
| **F4** | Target Encoding Leakage | `FEASIBLE` (Rất cao) | Dataset có sẵn các biến phân loại thực tế: `Location` (8 nhóm), `Payment_Method` (5 nhóm), `Transaction_Type` (5 nhóm). Không cần tạo bin nhân tạo. |
| **D1** | Existing Duplicate / Group Leakage | `FEASIBLE` (Rất cao) | Dữ liệu có sẵn 881 exact duplicates tự nhiên VÀ 4.000 `User_ID` lặp lại 100%. So sánh trực tiếp giữa GroupKFold/GroupShuffleSplit theo `User_ID` và Random Split. |
| **D3** | Duplicate Ratio Sensitivity | `FEASIBLE` | Bơm tỷ lệ bản sao có kiểm soát (0%, 5%, 10%, 20%) từ Validation vào Train. |

