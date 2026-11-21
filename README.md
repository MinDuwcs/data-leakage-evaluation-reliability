# Ảnh hưởng của Data Leakage đến độ chính xác và khả năng tổng quát hóa của mô hình Machine Learning

**Tên tiếng Anh tham khảo:** An Experimental Study of the Impact of Data Leakage on the Evaluation Performance and Generalization of Machine Learning Models for Classification

---

## 1. Tổng quan nghiên cứu

Trong quy trình phát triển Machine Learning, hiệu suất của mô hình không chỉ phụ thuộc vào thuật toán mà còn chịu ảnh hưởng trực tiếp từ chất lượng dữ liệu, quy trình tiền xử lý, phương pháp phân chia dữ liệu và thiết kế đánh giá thực nghiệm.

**Data Leakage** (rò rỉ dữ liệu) là hiện tượng thông tin không hợp lệ, thông tin ngoài luồng, hoặc thông tin không nên có sẵn tại thời điểm dự đoán trong thực tế bị đưa vào quá trình huấn luyện và đánh giá mô hình. Điều này dẫn đến sự sai lệch nghiêm trọng: mô hình có thể đạt điểm số đánh giá rất cao trên tập kiểm thử nội bộ nhưng lại suy giảm hiệu năng đột ngột khi triển khai trên dữ liệu thực tế.

Dự án này tập trung nghiên cứu thực nghiệm hiện tượng Data Leakage thông qua bài toán phân loại giao dịch gian lận tài chính (Credit Card Fraud Detection). Mục tiêu cốt lõi là đo lường mức độ sai lệch đánh giá giữa quy trình chuẩn (Baseline) và các kịch bản có rò rỉ dữ liệu, từ đó làm rõ mức độ ảnh hưởng của từng dạng Leakage trên các kiến trúc thuật toán khác nhau.

---

## 2. Câu hỏi nghiên cứu (Research Questions)

Nghiên cứu tập trung giải quyết câu hỏi vấn đề trung tâm:

> **Việc xuất hiện Data Leakage trong quá trình chuẩn bị dữ liệu và xây dựng mô hình làm sai lệch kết quả đánh giá Machine Learning đến mức nào, và mức độ ảnh hưởng đó có khác biệt giữa các thuật toán hay không?**

Các câu hỏi cụ thể bao gồm:

1. **Mức độ sai lệch đánh giá:** Điểm số đánh giá trên dữ liệu bị ảnh hưởng bởi Leakage chênh lệch bao nhiêu so với điểm số thực tế trên tập kiểm thử độc lập, sạch?
2. **So sánh các dạng Leakage:** Trong 3 dạng rò rỉ (Preprocessing Leakage, Feature Leakage, Duplicate Leakage), dạng nào gây ra mức độ sai lệch đánh giá và ảo tưởng hiệu năng lớn nhất?
3. **Độ nhạy của thuật toán:** Mức độ ảnh hưởng của từng dạng Leakage có khác nhau giữa các thuật toán Machine Learning thuộc các họ mô hình khác nhau (tuyến tính, cây quyết định đơn lẻ, mô hình tập hợp) hay không?

---

## 3. Phạm vi và Thiết kế thực nghiệm

### 3.1. Bài toán và Dữ liệu

- **Bài toán:** Phân loại nhị phân phát hiện giao dịch gian lận thẻ tín dụng (Fraud Detection).
- **Dataset:** Credit Card Fraud Detection Dataset (ULB / Kaggle).
- **Đặc trưng dữ liệu:** 
  - Gồm 30 đặc trưng đầu vào (V1-V28 đã được biến đổi qua PCA để bảo mật thông tin, cùng đặc trưng `Time` và `Amount`).
  - Phân phối nhãn mục tiêu (`Class`) cực kỳ mất cân bằng (tỷ lệ giao dịch gian lận chiếm tỷ lệ rất nhỏ so với giao dịch thông thường).
  - File dữ liệu được lưu trữ tại `data/fraud_detection.csv`.

### 3.2. Ba kịch bản Data Leakage khảo sát

Nghiên cứu mô phỏng có đối chứng 3 dạng rò rỉ dữ liệu phổ biến:

1. **Preprocessing Leakage (Rò rỉ tiền xử lý):**
   - Thực hiện các phép biến đổi (chuẩn hóa tỷ lệ, điền giá trị khuyết, mã hóa đặc trưng) trên toàn bộ tập dữ liệu trước khi chia tập huấn luyện (Train) và tập kiểm thử (Test/Validation).
2. **Feature Leakage (Rò rỉ đặc trưng):**
   - Sử dụng các đặc trưng đầu vào mang thông tin trực tiếp từ nhãn mục tiêu hoặc các thông tin chỉ xuất hiện sau khi giao dịch gian lận đã được xác nhận trong thực tế.
3. **Duplicate Leakage (Rò rỉ trùng lặp):**
   - Sự hiện diện của các bản ghi trùng lặp hoặc gần như trùng lặp xuất hiện đồng thời trong cả tập Train và tập Test, thường do các kỹ thuật oversampling (như SMOTE) không đúng vị trí hoặc trùng lặp tự nhiên trong dữ liệu.

### 3.3. Thuật toán Machine Learning đánh giá

Nghiên cứu lựa chọn 3 thuật toán đại diện cho các trường phái mô hình khác nhau:

- **Logistic Regression:** Mô hình tuyến tính đơn giản, dễ giải thích, làm mốc đối chuẩn tuyến tính.
- **Decision Tree (CART):** Mô hình phi tuyến dựa trên luật phân nhánh đơn lẻ, nhạy cảm với biến động dữ liệu.
- **Random Forest:** Mô hình tập hợp (Ensemble / Bagging) kết hợp nhiều cây quyết định, có khả năng chống overfitting tốt hơn.

### 3.4. Nguyên tắc đối chứng thực nghiệm

- **Baseline Pipeline:** Thiết lập quy trình chuẩn mực, kiểm soát nghiêm ngặt: dữ liệu được chia trước (Train / Validation / Test độc lập), toàn bộ bước tiền xử lý chỉ được fit trên tập Train và transform sang tập Test.
- **Kịch bản rò rỉ:** Thiết lập các pipeline vi phạm nguyên tắc phân chia dữ liệu theo từng dạng Leakage đã định nghĩa.
- **Đo lường:** Đánh giá hiệu suất trên cả tập kiểm thử nội bộ (nơi xảy ra leakage) và tập kiểm thử độc lập (sạch hoàn toàn) bằng các chỉ số phù hợp với dữ liệu mất cân bằng (PR-AUC, ROC-AUC, F1-Score, Recall).

---

## 4. Cấu trúc Repository

```text
.
├── data/
│   └── fraud_detection.csv              # Dữ liệu giao dịch gian lận thẻ tín dụng
├── model/
│   └── scripts/
│       └── preprocess.ipynb             # Notebook phục vụ tiền xử lý và thực nghiệm
├── .gitignore                           # Cấu hình bỏ qua các file tạm, cache, môi trường
├── data_leakage_research_handoff.md     # Tài liệu bàn giao nghiên cứu và thiết kế chi tiết
└── README.md                            # Tài liệu tổng quan và hướng dẫn dự án
```

---

## 5. Trạng thái hiện tại của đề tài

- **Giai đoạn hiện tại:** Research Proposal & Research Design Refinement (Hoàn thiện đề cương và thiết kế thực nghiệm).
- **Trạng thái mã nguồn:** Chưa bước vào giai đoạn viết code triển khai chính thức. Mọi thực nghiệm sẽ được bắt đầu sau khi thiết kế thực nghiệm (Experimental Protocol) được thống nhất hoàn chỉnh.
- **Quy tắc học thuật:** Không đưa ra kết luận hoặc công bố số liệu khi chưa hoàn thành thực nghiệm có đối chứng.

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
pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook
```

---

## 7. Tài liệu tham chiếu

Để xem chi tiết toàn bộ các quyết định kỹ thuật, quy tắc làm việc, câu hỏi nghiên cứu phân rã, ma trận thực nghiệm và lộ trình triển khai chi tiết, vui lòng tham khảo:

- [Tài liệu bàn giao nghiên cứu: data_leakage_research_handoff.md](data_leakage_research_handoff.md)
