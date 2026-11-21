# Research Project Handoff: Data Leakage trong Machine Learning

> **Mục đích tài liệu:** Đây là tài liệu bàn giao độc lập cho các thành viên trong nhóm và AI assistant của từng thành viên. Người đọc phải có thể hiểu bối cảnh, mục tiêu, phạm vi, quyết định đã chốt, trạng thái hiện tại và công việc tiếp theo mà không cần truy cập hội thoại gốc.
>
> **Ngôn ngữ làm việc:** Tiếng Việt; giữ nguyên các thuật ngữ kỹ thuật tiếng Anh khi cần.
>
> **Trạng thái tài liệu:** Bản handoff nền tảng, được tổng hợp từ các quyết định đã xác nhận trong hội thoại. Những nội dung chưa được kiểm chứng hoặc chưa được nhóm/giảng viên phê duyệt được đánh dấu rõ là `PROPOSED`, `TODO` hoặc `OPEN`.

---

## 0. Quy tắc đọc tài liệu

Các nhãn trạng thái được sử dụng thống nhất:

- `CONFIRMED`: Đã được người dùng/nhóm xác nhận.
- `PROPOSED`: Đề xuất kỹ thuật hoặc kế hoạch, chưa được nhóm phê duyệt cuối cùng.
- `TODO`: Công việc cần thực hiện.
- `OPEN`: Câu hỏi chưa có quyết định.
- `RISK`: Rủi ro hoặc giới hạn cần theo dõi.
- `OUT_OF_SCOPE`: Không thuộc phạm vi hiện tại.

AI assistant hỗ trợ project phải:

1. Đọc toàn bộ tài liệu trước khi đề xuất thay đổi.
2. Phân biệt rõ nội dung `CONFIRMED` với `PROPOSED`.
3. Không tự ý thay đổi tên đề tài, dataset, ba loại Leakage hoặc ba mô hình đã chốt.
4. Không tự bịa kết quả thực nghiệm, số liệu, bài báo, DOI hoặc Research Gap.
5. Khi đưa ra quyết định kỹ thuật chưa có trong tài liệu, phải nêu lý do, trade-off và ghi vào mục `Decision Log`.
6. Chưa viết code cho đến khi nhóm hoàn thành và xác nhận Research Design/Experimental Protocol.
7. Khi chỉnh sửa tài liệu này, phải cập nhật `Last Updated`, `Current Phase`, `Decision Log` và `Implementation / Research Notes`.

---

## 1. Project Identity

| Trường | Giá trị | Trạng thái |
|---|---|---|
| Project name | Data Leakage in Machine Learning | `CONFIRMED` |
| Course | Phương pháp nghiên cứu khoa học cho Khoa học dữ liệu | `CONFIRMED` |
| Research type | Nghiên cứu thực nghiệm, kết hợp tổng quan tài liệu | `CONFIRMED` |
| Research domain | Machine Learning | `CONFIRMED` |
| Problem domain | Phân loại giao dịch gian lận tài chính | `CONFIRMED` |
| Dataset | Credit Card Fraud Detection (Kaggle, ULB) | `CONFIRMED` |
| Leakage categories | Preprocessing Leakage, Feature Leakage, Duplicate Leakage | `CONFIRMED` |
| ML models | Logistic Regression, Decision Tree, Random Forest | `CONFIRMED` |
| Team size | 4–5 thành viên | `CONFIRMED` |
| Expected duration | 6–10 tuần; kế hoạch tham chiếu 8 tuần | `CONFIRMED` |
| Current phase | Research Proposal / Research Design refinement | `PROPOSED` |
| Code implementation | Chưa bắt đầu | `CONFIRMED` |

### 1.1. Tên đề tài

**Tên đề tài đang sử dụng:**

“Ảnh hưởng của Data Leakage đến độ chính xác và khả năng tổng quát hóa của mô hình Machine Learning.”

Tên tiếng Anh tham khảo:

“An Experimental Study of the Impact of Data Leakage on the Evaluation Performance and Generalization of Machine Learning Models for Classification.”

`OPEN`: Cần xác nhận lại tên chính thức với giảng viên nếu trường có quy định về cấu trúc tên đề tài.

### 1.2. Tóm tắt project

Nhóm nghiên cứu hiện tượng Data Leakage trong quy trình Machine Learning thông qua bài toán phát hiện giao dịch thẻ tín dụng gian lận. Nhóm sẽ thiết lập một quy trình thực nghiệm chuẩn làm baseline, sau đó mô phỏng ba dạng Leakage: Preprocessing Leakage, Feature Leakage và Duplicate Leakage. Các kịch bản được đánh giá trên ba thuật toán Logistic Regression, Decision Tree và Random Forest.

Mục tiêu là xác định liệu Data Leakage có làm sai lệch kết quả đánh giá mô hình hay không, mức độ ảnh hưởng có khác nhau giữa các dạng Leakage và thuật toán hay không, đồng thời phân tích chênh lệch giữa kết quả đánh giá bị ảnh hưởng bởi Leakage và kết quả trên dữ liệu đánh giá độc lập, sạch.

**Lưu ý:** Đây là mục tiêu nghiên cứu, không phải kết quả đã được chứng minh. Không được viết trước kết luận rằng mọi dạng Leakage đều làm tăng điểm số hoặc một thuật toán cụ thể luôn nhạy cảm hơn.

---

## 2. Background and Motivation

### 2.1. Bối cảnh

Machine Learning được sử dụng trong nhiều lĩnh vực như tài chính, y tế, thương mại điện tử và an ninh mạng. Hiệu suất mô hình phụ thuộc không chỉ vào thuật toán mà còn vào dữ liệu, quy trình tiền xử lý, cách chia tập dữ liệu và cách đánh giá.

Data Leakage xảy ra khi thông tin không hợp lệ hoặc không nên có sẵn tại thời điểm dự đoán bị đưa vào quy trình huấn luyện/đánh giá. Leakage có thể khiến kết quả đánh giá không phản ánh đúng khả năng hoạt động của mô hình trên dữ liệu mới.

Trong bài toán phát hiện gian lận tài chính, đánh giá sai hiệu suất có thể dẫn đến bỏ sót giao dịch gian lận hoặc cảnh báo nhầm giao dịch hợp lệ.

### 2.2. Lý do chọn đề tài

- Data Leakage là rủi ro phương pháp luận quan trọng trong Machine Learning.
- Chủ đề có thể được nghiên cứu bằng thực nghiệm có đối chứng.
- Dataset công khai, không cần tự thu thập dữ liệu khách hàng.
- Các mô hình đã chọn có thể chạy bằng CPU, không cần GPU.
- Đề tài liên quan đến Data Preprocessing, Model Evaluation, Data Quality và ML Pipeline.
- Kết quả có thể giúp hình thành checklist phòng tránh Leakage trong dự án ML.

### 2.3. Research Problem

**Câu hỏi vấn đề trung tâm:**

Việc xuất hiện Data Leakage trong quá trình chuẩn bị dữ liệu và xây dựng mô hình có thể làm sai lệch kết quả đánh giá Machine Learning đến mức nào, và mức độ ảnh hưởng đó có khác nhau giữa các thuật toán hay không?

---

## 3. Research Objectives

### 3.1. Mục tiêu tổng quát

Nghiên cứu và thực nghiệm nhằm đánh giá ảnh hưởng của Data Leakage đến hiệu suất đánh giá và khả năng tổng quát hóa của các mô hình Machine Learning trong bài toán phân loại giao dịch gian lận.

### 3.2. Mục tiêu cụ thể

1. Hệ thống hóa cơ sở lý thuyết và phân loại các dạng Data Leakage.
2. Xây dựng quy trình Machine Learning chuẩn làm baseline đối chứng.
3. Thiết kế các kịch bản thực nghiệm mô phỏng ba dạng Leakage.
4. Đánh giá mức độ thay đổi hiệu suất của ba thuật toán Machine Learning.
5. Phân tích sự khác biệt giữa các thuật toán và giữa các loại Leakage.
6. Đề xuất biện pháp phát hiện, hạn chế Data Leakage trong quy trình Machine Learning.

---

## 4. Research Questions and Hypotheses

### 4.1. Research Questions

| ID | Câu hỏi nghiên cứu |
|---|---|
| RQ1 | Ba dạng Data Leakage được lựa chọn ảnh hưởng như thế nào đến hiệu suất đánh giá của các mô hình Machine Learning? |
| RQ2 | Mức độ ảnh hưởng của Data Leakage có khác nhau giữa Logistic Regression, Decision Tree và Random Forest hay không? |
| RQ3 | Data Leakage ảnh hưởng như thế nào đến khả năng tổng quát hóa của mô hình trên dữ liệu độc lập, không bị rò rỉ? |
| RQ4 | Những biện pháp nào có thể giúp phát hiện và hạn chế Data Leakage trong quy trình Machine Learning? |

### 4.2. Hypotheses

| ID | Giả thuyết | Cách kiểm chứng dự kiến |
|---|---|---|
| H1 | Data Leakage có thể làm kết quả đánh giá trên dữ liệu bị ảnh hưởng bởi Leakage khác biệt so với quy trình chuẩn. | So sánh metric giữa baseline, contaminated evaluation và clean evaluation tương ứng. |
| H2 | Mức độ ảnh hưởng của Data Leakage có thể khác nhau giữa các thuật toán Machine Learning. | So sánh mức thay đổi metric của ba mô hình trong cùng kịch bản. |
| H3 | Kết quả đánh giá trên dữ liệu có Leakage có thể không phản ánh hiệu suất trên dữ liệu độc lập, sạch. | So sánh kết quả contaminated test với clean holdout. |
| H4 | Các biện pháp kiểm soát dữ liệu và quy trình tiền xử lý phù hợp có thể hạn chế nguy cơ Data Leakage. | Đối chiếu quy trình chuẩn với quy trình cố ý vi phạm và kiểm tra biện pháp phòng tránh. |

**Nguyên tắc:** Giả thuyết là phát biểu cần kiểm chứng, không phải kết luận. Nếu kết quả không ủng hộ giả thuyết, báo cáo phải trình bày đúng kết quả.

---

## 5. Scope and Boundaries

### 5.1. Phạm vi đã xác nhận

| Hạng mục | Phạm vi |
|---|---|
| Problem type | Binary classification |
| Application | Credit card fraud detection |
| Dataset | Credit Card Fraud Detection |
| Leakage type 1 | Preprocessing Leakage |
| Leakage type 2 | Feature Leakage |
| Leakage type 3 | Duplicate Leakage |
| Model 1 | Logistic Regression |
| Model 2 | Decision Tree |
| Model 3 | Random Forest |
| Research approach | Literature Review + Experimental Research |

### 5.2. Ngoài phạm vi

- Deep Learning và Neural Networks.
- Xây dựng hệ thống phát hiện gian lận thời gian thực.
- Triển khai sản phẩm thương mại.
- Nghiên cứu chuyên sâu về Explainable AI.
- Mở rộng sang nhiều dataset hoặc nhiều thuật toán khi chưa có phê duyệt.
- Tối ưu hyperparameter quy mô lớn không phục vụ trực tiếp câu hỏi nghiên cứu.

### 5.3. Dataset

**Tên:** Credit Card Fraud Detection  
**Nguồn:** Kaggle, Machine Learning Group - Université Libre de Bruxelles (ULB)  
**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Thông tin dataset được mô tả trong trang nguồn:

- 284.807 giao dịch.
- 492 giao dịch gian lận.
- 30 cột đầu vào gồm `Time`, `V1` đến `V28`, `Amount`.
- Cột mục tiêu: `Class`.
- Tỷ lệ giao dịch gian lận xấp xỉ 0,173%.
- Dữ liệu được mô tả là các giao dịch của chủ thẻ châu Âu trong tháng 9/2013.

**TODO – xác minh khi tải dataset:**

1. Xác nhận số hàng, số cột và schema thực tế của file.
2. Kiểm tra kiểu dữ liệu, missing values và giá trị bất thường.
3. Kiểm tra số lượng bản ghi trùng lặp chính xác.
4. Xác định ý nghĩa và giới hạn của các feature đã được ẩn danh.
5. Ghi lại phiên bản, ngày tải và checksum của dataset nếu có thể.

**Cảnh báo:** Dataset có các feature `V1`–`V28` đã được biến đổi/ẩn danh. Không được tự suy diễn ý nghĩa nghiệp vụ cụ thể của các feature này.

---

## 6. Operational Definitions: Ba dạng Data Leakage

### 6.1. Preprocessing Leakage

**Định nghĩa sử dụng trong project:**

Thông tin từ tập đánh giá được sử dụng để ước lượng tham số hoặc thực hiện bước tiền xử lý trước khi mô hình được huấn luyện, thay vì chỉ fit bước tiền xử lý trên tập Train.

**Kịch bản chuẩn:**

1. Chia dữ liệu thành Train, Validation và Test.
2. Fit preprocessing transformer trên Train.
3. Transform Train, Validation và Test bằng transformer đã fit.
4. Huấn luyện mô hình trên Train đã transform.
5. Đánh giá trên Validation/Test.

**Kịch bản Leakage:**

1. Fit transformer trên toàn bộ dữ liệu trước khi chia tập.
2. Sau đó mới chia Train/Validation/Test.
3. So sánh kết quả với quy trình chuẩn.

**RISK / giới hạn diễn giải:**

- Scaling không sử dụng nhãn nên đây là dạng thông tin từ tập Test tham gia preprocessing; mức độ ảnh hưởng có thể nhỏ hoặc không đáng kể.
- Không được mặc định preprocessing leakage luôn làm tăng Accuracy hoặc các metric.
- Cần nêu rõ transformer nào được dùng và thống kê nào được học từ toàn bộ dữ liệu.
- Cần cân nhắc preprocessing phù hợp với từng mô hình. Ví dụ, scaling quan trọng hơn với Logistic Regression so với Decision Tree/Random Forest.
- Nếu kết quả không khác biệt đáng kể, đó vẫn là kết quả có giá trị cần báo cáo.

### 6.2. Feature Leakage

**Định nghĩa sử dụng trong project:**

Một feature chứa thông tin trực tiếp hoặc gián tiếp về nhãn mục tiêu, nhưng thông tin đó không thể có sẵn tại thời điểm hệ thống thực hiện dự đoán.

**Ví dụ nghiệp vụ minh họa:**

Một cột `fraud_investigation_result` chỉ được tạo sau khi giao dịch được điều tra. Nếu dùng cột này để dự đoán gian lận ngay tại thời điểm giao dịch thì mô hình đã sử dụng thông tin từ tương lai.

**Thiết kế dự kiến:**

- Baseline: chỉ sử dụng feature hợp lệ tại thời điểm dự đoán.
- Leakage scenario: bổ sung feature được tạo từ thông tin hậu nghiệm/nhãn, mô phỏng một nguồn thông tin không hợp lệ.
- Đánh giá và báo cáo rõ mô hình có sử dụng feature không hợp lệ không đại diện cho một hệ thống có thể triển khai thực tế.

**OPEN – cần thiết kế chi tiết:**

- Feature leakage sẽ được mô phỏng theo mức độ nào?
- Có thử nhiều mức nhiễm hay chỉ một kịch bản?
- Cách xây dựng tập đánh giá bị nhiễm và tập đánh giá sạch như thế nào?
- Làm sao bảo đảm so sánh không bị nhầm lẫn giữa thay đổi feature set và thay đổi cách chia dữ liệu?

**RISK:** Nếu tạo một cột sao chép chính xác `Class`, kết quả gần như hiển nhiên và ít giá trị nghiên cứu. Cần chọn cách mô phỏng có giải thích, kiểm soát và ghi rõ giới hạn. Không được gọi một feature chỉ tương quan cao với nhãn là Leakage nếu feature đó thực sự khả dụng tại thời điểm dự đoán.

### 6.3. Duplicate Leakage

**Định nghĩa sử dụng trong project:**

Các bản ghi trùng lặp hoặc gần trùng có quan hệ với nhau xuất hiện đồng thời trong Train và Test, khiến kết quả đánh giá không còn phản ánh khả năng dự đoán trên các quan sát độc lập.

**Hai kịch bản dự kiến:**

- Random Split: chia ngẫu nhiên mà không kiểm soát duplicate groups.
- Group-based Split: nhóm các bản ghi trùng lặp/gần trùng trước, sau đó chia theo nhóm để một nhóm không xuất hiện ở cả Train và Test.

**OPEN – cần xác minh:**

- Dataset gốc có exact duplicates hay không?
- Có cơ sở hợp lý để định nghĩa near-duplicate không?
- Nếu dataset không có duplicate phù hợp, nhóm có tạo dữ liệu trùng lặp có kiểm soát hay không?
- Nếu tạo dữ liệu, tỷ lệ và phương pháp tạo sẽ là gì?

**RISK:**

- Dataset không có customer ID hoặc transaction ID định danh thực tế trong các feature mô tả.
- Không được tự nhận diện hai giao dịch là cùng một giao dịch chỉ vì một số feature gần nhau.
- Nếu tạo duplicate nhân tạo, phải công khai phương pháp tạo và phân tích giới hạn về tính đại diện.
- Group-based Split chỉ có ý nghĩa nếu định nghĩa group/duplicate hợp lý.

---

## 7. Research Methodology

### 7.1. Phương pháp tổng thể

Project sử dụng hai phương pháp:

**A. Literature Review**

Mục đích:

- Tìm hiểu khái niệm và taxonomy Data Leakage.
- Tổng hợp nguyên nhân và tác động.
- Tìm hiểu các cách phát hiện/phòng tránh.
- Xác định các cách thiết kế thực nghiệm đã được sử dụng.
- Xây dựng cơ sở cho Research Gap và Experimental Protocol.

Nguồn tìm kiếm:

- Google Scholar.
- IEEE Xplore.
- ACM Digital Library.
- SpringerLink.
- ScienceDirect.
- Tài liệu chính thức của Scikit-learn.

Từ khóa gợi ý:

```text
"Data Leakage in Machine Learning"
"Data Leakage in Predictive Modeling"
"Preprocessing Leakage"
"Feature Leakage"
"Target Leakage"
"Duplicate Data Leakage"
"Data Leakage Prevention"
"Machine Learning Generalization"
"Data Leakage Evaluation Bias"
```

**B. Experimental Research**

Thiết kế đối chứng:

- Baseline: quy trình chuẩn, không cố ý tạo Leakage.
- Leakage scenarios: từng kịch bản được thực hiện độc lập.
- Clean holdout: dữ liệu độc lập, không tham gia preprocessing, model selection hoặc tuning.

Mục tiêu là đo chênh lệch giữa kết quả đánh giá bị ảnh hưởng bởi Leakage và kết quả trên dữ liệu độc lập, sạch.

### 7.2. Models

| Model | Vai trò | Trạng thái |
|---|---|---|
| Logistic Regression | Baseline tuyến tính | `CONFIRMED` |
| Decision Tree | Mô hình cây quyết định | `CONFIRMED` |
| Random Forest | Ensemble model | `CONFIRMED` |

**TODO:** Chốt preprocessing, class weighting, threshold và hyperparameters sau khi đọc tài liệu và kiểm tra dữ liệu. Các cấu hình phải được ghi lại và giữ nhất quán giữa những kịch bản cần so sánh.

### 7.3. Evaluation Metrics

| Metric | Ý nghĩa | Mức độ ưu tiên |
|---|---|---|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể | Tham khảo |
| Precision | Trong các dự đoán gian lận, tỷ lệ dự đoán đúng | Quan trọng |
| Recall | Tỷ lệ giao dịch gian lận được phát hiện | Quan trọng |
| F1-score | Trung bình điều hòa Precision và Recall | Quan trọng |
| ROC-AUC | Khả năng phân biệt hai lớp theo ngưỡng | Quan trọng |
| PR-AUC / Average Precision | Đánh giá precision-recall, hữu ích khi lớp rất mất cân bằng | Rất quan trọng |

**Lưu ý:**

- Phải định nghĩa rõ positive class là `Class = 1` (fraud).
- Không dùng Accuracy làm chỉ số duy nhất.
- Phải phân biệt ROC-AUC với PR-AUC/Average Precision.
- Nếu thay đổi decision threshold, phải ghi rõ cách chọn threshold và không chọn threshold dựa trên Test.
- Nên báo cáo confusion matrix cùng Precision, Recall và F1-score.

---

## 8. Experimental Design – Khung đề xuất

Đây là khung thiết kế đã được đề xuất trong hội thoại, chưa phải Experimental Protocol cuối cùng. Cần hoàn thiện trước khi code.

### 8.1. Baseline Experiment

Mục đích: tạo mốc tham chiếu đáng tin cậy.

Quy trình:

1. Tải và kiểm tra dataset.
2. Xác định feature và target.
3. Tách một clean holdout ngay từ đầu.
4. Dùng stratified split phù hợp với class imbalance.
5. Fit mọi preprocessing transformer chỉ trên Train.
6. Chỉ dùng Validation hoặc cross-validation trên Train để chọn cấu hình.
7. Đánh giá cuối cùng trên clean holdout.
8. Lưu metric, cấu hình, seed và thông tin lần chạy.

**Yêu cầu:** Clean holdout phải được giữ nguyên và không được dùng để fit preprocessing, chọn feature, chọn model hoặc tuning.

### 8.2. Experiment A – Preprocessing Leakage

Mục tiêu: đo ảnh hưởng của việc fit preprocessing trên toàn bộ dữ liệu thay vì chỉ Train.

So sánh:

| Nhánh | Quy trình |
|---|---|
| A-Control | Split trước; fit transformer trên Train; transform các tập còn lại |
| A-Leakage | Fit transformer trên dữ liệu bao gồm cả phần đánh giá trước khi split |

Đầu ra:

- Metric của mỗi model.
- Chênh lệch metric giữa các nhánh.
- Phân tích vì sao kết quả có hoặc không thay đổi.

**Cần làm rõ:** Cách tạo một contaminated evaluation phù hợp và cách giữ clean holdout độc lập để đánh giá generalization.

### 8.3. Experiment B – Feature Leakage

Mục tiêu: đo mức độ sai lệch khi mô hình sử dụng feature không khả dụng tại thời điểm dự đoán.

So sánh:

| Nhánh | Feature |
|---|---|
| B-Control | Chỉ feature hợp lệ |
| B-Leakage | Feature hợp lệ + feature chứa thông tin hậu nghiệm/nhãn |

Đầu ra:

- Metric của mô hình có feature leakage trên kịch bản đánh giá tương ứng.
- So sánh với baseline.
- Giải thích vì sao feature không thể được dùng khi triển khai.
- Phân tích chênh lệch giữa đánh giá không hợp lệ và đánh giá trên dữ liệu/feature hợp lệ.

**RISK:** Không được so sánh hai mô hình có feature set khác nhau rồi kết luận chênh lệch chỉ do thuật toán. Phải diễn giải đúng rằng feature leakage làm thay đổi thông tin đầu vào và khiến kết quả không còn phản ánh bài toán dự đoán thực tế.

### 8.4. Experiment C – Duplicate Leakage

Mục tiêu: đo ảnh hưởng của duplicate groups xuất hiện ở cả Train và Test.

So sánh:

| Nhánh | Cách chia |
|---|---|
| C-Control | Group-based split, giữ duplicate group ở một tập duy nhất |
| C-Leakage | Random split không kiểm soát duplicate groups |

Đầu ra:

- Số duplicate groups.
- Tỷ lệ group xuất hiện ở cả Train và Test.
- Metric của ba mô hình.
- Chênh lệch giữa random split và group-based split.

**Điều kiện tiên quyết:** Kiểm tra dữ liệu gốc trước. Nếu không có duplicate groups phù hợp, phải quyết định có tạo duplicate nhân tạo hay điều chỉnh thiết kế. Không được tự ý tạo dữ liệu mà không ghi rõ.

### 8.5. Experimental Controls

Các nguyên tắc kiểm soát:

1. Dùng cùng nguồn dataset và ghi nhận phiên bản.
2. Cố định random seed hoặc lặp lại nhiều lần với seed được xác định trước.
3. Giữ cấu hình mô hình tương ứng nhất quán.
4. Không dùng Test để tuning hoặc chọn model.
5. Không để clean holdout tham gia preprocessing hoặc bất kỳ quyết định phát triển mô hình nào.
6. Lưu kết quả của tất cả lần chạy, không chỉ lần chạy tốt nhất.
7. Ghi nhận tỷ lệ class trong mỗi split.
8. Phân tích metric cùng độ biến thiên giữa các lần chạy.
9. Giữ riêng kết quả contaminated evaluation và clean evaluation.
10. Không diễn giải metric cao hơn là bằng chứng duy nhất của generalization tốt hơn.

---

## 9. Research Variables

| Variable type | Biến | Mô tả |
|---|---|---|
| Independent | Leakage type | Preprocessing, Feature, Duplicate |
| Independent | Algorithm | Logistic Regression, Decision Tree, Random Forest |
| Independent | Leakage level | Mức độ/tỷ lệ nhiễm nếu thiết kế có nhiều mức |
| Dependent | Accuracy | Tỷ lệ dự đoán đúng |
| Dependent | Precision | Độ chính xác của dự đoán positive |
| Dependent | Recall | Khả năng phát hiện positive |
| Dependent | F1-score | Cân bằng Precision và Recall |
| Dependent | ROC-AUC | Khả năng phân biệt theo ROC |
| Dependent | PR-AUC | Khả năng phân biệt theo Precision-Recall |
| Controlled | Dataset | Nguồn dữ liệu cố định |
| Controlled | Random seed | Quy tắc seed được ghi nhận |
| Controlled | Model configuration | Cấu hình tương ứng được giữ nhất quán |
| Controlled | Split protocol | Quy trình chia tập được mô tả và kiểm soát |

**TODO:** Chuyển các biến trên thành operational definitions chính xác sau khi Experimental Protocol được duyệt.

---

## 10. Expected Outcomes

Các kết quả kỳ vọng, chưa phải kết quả thực tế:

1. Có taxonomy và phần tổng quan về ba dạng Data Leakage.
2. Có baseline ML pipeline tuân thủ nguyên tắc chống Leakage.
3. Có ba kịch bản thực nghiệm mô phỏng Leakage.
4. Có bảng metric so sánh ba mô hình giữa các kịch bản.
5. Có phân tích sự khác biệt giữa contaminated evaluation và clean evaluation.
6. Có kết luận về giả thuyết nào được ủng hộ hoặc không được ủng hộ.
7. Có checklist/khuyến nghị hạn chế Data Leakage trong ML workflow.

### 10.1. Deliverables

- Research Proposal.
- Literature Review.
- Literature Review Matrix.
- Research Design và Experimental Protocol.
- Dataset audit report.
- Python source code.
- Jupyter Notebooks.
- Experiment logs và kết quả có thể tái lập.
- Bảng/biểu đồ phân tích.
- Báo cáo nghiên cứu cuối cùng.
- Slide thuyết trình.

---

## 11. Timeline

Kế hoạch tham chiếu 8 tuần, nằm trong khung 6–10 tuần đã trao đổi.

| Tuần | Công việc | Deliverable |
|---|---|---|
| 1 | Chốt vấn đề, mục tiêu, RQ, hypotheses | Research Proposal |
| 2 | Tìm kiếm, đọc và phân tích tài liệu | Literature Review |
| 3 | Kiểm tra dataset, hoàn thiện thiết kế thí nghiệm | Dataset Audit + Research Design |
| 4 | Xây dựng baseline và xác nhận quy trình dữ liệu | Baseline Results |
| 5 | Thực hiện Preprocessing Leakage và Feature Leakage | Experiment A + B |
| 6 | Thực hiện Duplicate Leakage | Experiment C |
| 7 | Tổng hợp kết quả, phân tích và thảo luận | Results & Discussion |
| 8 | Hoàn thiện báo cáo, slide và thuyết trình | Final Paper + Slides |

**RISK:** Timeline phụ thuộc vào việc nhóm giải quyết được thiết kế Feature Leakage và Duplicate Leakage. Nếu hai thiết kế này chưa rõ ở tuần 3, cần điều chỉnh phạm vi hoặc lịch, không nên code theo giả định chưa được kiểm chứng.

---

## 12. Team Structure

Quy mô dự kiến: 4–5 thành viên.

### 12.1. Phân công nếu có 5 thành viên

| Thành viên | Trách nhiệm |
|---|---|
| Member 1 | Team lead, Research Design, điều phối và tổng hợp báo cáo |
| Member 2 | Literature Review, cơ sở lý thuyết và phân loại Data Leakage |
| Member 3 | Dataset Analysis, Data Preprocessing, Baseline |
| Member 4 | Experiment A – Preprocessing Leakage và Experiment B – Feature Leakage |
| Member 5 | Experiment C – Duplicate Leakage, Evaluation và Visualization |

### 12.2. Nếu có 4 thành viên

Gộp Literature Review vào Team Lead; các phần dataset, baseline và thực nghiệm được phân bổ lại theo năng lực thành viên.

**OPEN:** Tên thành viên, năng lực, deadline cá nhân và phân công cuối cùng chưa được cung cấp.

---

## 13. Technology Stack

| Công nghệ | Mục đích | Trạng thái |
|---|---|---|
| Python | Ngôn ngữ nghiên cứu | `PROPOSED` |
| Pandas | Data processing | `PROPOSED` |
| NumPy | Numerical operations | `PROPOSED` |
| Scikit-learn | Preprocessing và ML models | `PROPOSED` |
| Matplotlib | Visualization | `PROPOSED` |
| Seaborn | Visualization | `PROPOSED` |
| Jupyter Notebook | Thực nghiệm và trình bày phân tích | `PROPOSED` |
| Git/GitHub | Version control và cộng tác | `PROPOSED` |

Stack trên là đề xuất kỹ thuật phù hợp với phạm vi nghiên cứu, chưa có xác nhận riêng rằng nhóm đã cài đặt hoặc sử dụng.

---

## 14. Risks, Limitations and Mitigation

| Rủi ro | Mức độ | Cách xử lý |
|---|---|---|
| Dataset không có duplicate phù hợp | Cao | Dataset audit; định nghĩa duplicate có căn cứ hoặc dùng kịch bản nhân tạo có kiểm soát |
| Preprocessing Leakage tạo chênh lệch nhỏ | Trung bình | Báo cáo đúng kết quả; không ép kết luận |
| Feature Leakage quá nhân tạo | Cao | Mô phỏng theo logic thời điểm phát sinh thông tin; nêu rõ giới hạn |
| Class imbalance | Cao | Dùng PR-AUC, Recall, Precision, F1 và confusion matrix |
| Kết quả phụ thuộc random seed | Trung bình | Lặp lại thí nghiệm và báo cáo độ biến thiên |
| Phạm vi quá rộng | Trung bình | Giữ ba loại Leakage và ba model đã chốt; không mở rộng nếu chưa có lý do |
| Nhầm tương quan cao với Feature Leakage | Cao | Kiểm tra tính khả dụng của feature tại thời điểm dự đoán |
| Dùng Test để tuning | Cao | Tách Validation và clean holdout; khóa clean holdout |
| So sánh không công bằng giữa model | Trung bình | Kiểm soát split, preprocessing, metric và cấu hình |

### 14.1. Các giới hạn phải nêu trong báo cáo

- Kết quả chỉ được kiểm chứng trên dataset và thiết kế thí nghiệm đã chọn.
- Dataset giao dịch đã được ẩn danh/biến đổi, hạn chế diễn giải nghiệp vụ.
- Duplicate nhân tạo, nếu sử dụng, không đại diện đầy đủ cho mọi dạng duplicate trong dữ liệu thực tế.
- Preprocessing leakage không dùng nhãn có thể tạo ảnh hưởng khác với target/feature leakage.
- Không thể khái quát kết quả cho mọi dataset hoặc mọi thuật toán nếu chưa kiểm chứng thêm.

---

## 15. Current Project Status

### 15.1. Những gì đã hoàn thành/xác nhận

- [x] Chọn đề tài Data Leakage.
- [x] Giảng viên cho phép thực hiện thực nghiệm.
- [x] Nhóm quyết định chuyển sang hướng thực nghiệm.
- [x] Chốt ba dạng Leakage: Preprocessing, Feature, Duplicate.
- [x] Chốt dataset Credit Card Fraud Detection.
- [x] Chốt ba mô hình: Logistic Regression, Decision Tree, Random Forest.
- [x] Xây dựng bản Research Proposal sơ bộ.
- [x] Xác định RQ, hypotheses, objectives và phạm vi nghiên cứu sơ bộ.

### 15.2. Chưa hoàn thành

- [ ] Xác minh yêu cầu/format báo cáo chính thức của giảng viên.
- [ ] Tìm và đọc bài báo khoa học.
- [ ] Xây dựng Literature Review Matrix.
- [ ] Xác định Research Gap dựa trên tài liệu thật.
- [ ] Audit dataset và kiểm tra duplicates.
- [ ] Chốt cách operationalize ba dạng Leakage.
- [ ] Chốt split protocol và clean holdout.
- [ ] Chốt model configurations và evaluation protocol.
- [ ] Phê duyệt Experimental Protocol.
- [ ] Bắt đầu viết code.
- [ ] Chạy thí nghiệm.
- [ ] Phân tích kết quả và kiểm định giả thuyết.
- [ ] Hoàn thiện báo cáo và slide.

### 15.3. Current Phase

**Phase hiện tại: Research Design Refinement / Literature Review Preparation.**

Nhóm đã có Proposal sơ bộ nhưng chưa đủ cơ sở để bắt đầu implementation. Hai vấn đề kỹ thuật cần ưu tiên làm rõ là:

1. Cách mô phỏng Feature Leakage mà vẫn phân biệt được contaminated evaluation với khả năng triển khai thực tế.
2. Cách định nghĩa Duplicate Leakage phù hợp với cấu trúc dataset, sau khi kiểm tra duplicates thực tế.

---

## 16. Next Actions – Ưu tiên thực hiện

### Priority 0 – Bắt buộc trước khi code

1. Xác minh yêu cầu báo cáo của giảng viên.
2. Tải dataset và thực hiện Dataset Audit.
3. Kiểm tra exact duplicates và khả năng định nghĩa near-duplicates.
4. Hoàn thiện thiết kế ba Leakage scenarios.
5. Thiết kế immutable clean holdout.
6. Chốt Experimental Protocol bằng văn bản.

### Priority 1 – Cơ sở học thuật

1. Tìm 10–15 bài báo khoa học phù hợp.
2. Ghi lại title, authors, year, venue, DOI/URL.
3. Ghi research objective, method, dataset, findings và limitations.
4. Phân loại bài báo theo ba dạng Leakage.
5. So sánh các nghiên cứu để tìm Research Gap.
6. Cập nhật RQ và hypotheses nếu cần.

### Priority 2 – Implementation

Chỉ bắt đầu khi Priority 0 hoàn tất và được nhóm xác nhận:

1. Chuẩn bị môi trường và repository.
2. Xây dựng dataset audit.
3. Xây dựng baseline pipeline.
4. Kiểm thử quy trình chống Leakage.
5. Thực hiện từng Leakage scenario độc lập.
6. Lưu experiment metadata và metrics.
7. Tổng hợp, trực quan hóa và phân tích kết quả.

---

## 17. Decision Log

| ID | Quyết định | Trạng thái | Ghi chú |
|---|---|---|---|
| D-001 | Chọn đề tài Data Leakage trong ML | `CONFIRMED` | Người dùng đã chốt đề tài 1 |
| D-002 | Chuyển từ định hướng lý thuyết sang thực nghiệm | `CONFIRMED` | Giảng viên cho phép thực nghiệm |
| D-003 | Nghiên cứu ba loại Leakage | `CONFIRMED` | Preprocessing, Feature, Duplicate |
| D-004 | Sử dụng Credit Card Fraud Detection | `CONFIRMED` | Kaggle / ULB |
| D-005 | Sử dụng Logistic Regression, Decision Tree, Random Forest | `CONFIRMED` | Ba model cố định |
| D-006 | Dự kiến thời gian 8 tuần | `PROPOSED` | Nằm trong khung 6–10 tuần |
| D-007 | Dùng Python và Scikit-learn | `PROPOSED` | Stack kỹ thuật đề xuất |
| D-008 | Giữ clean holdout độc lập | `PROPOSED` | Nguyên tắc thiết kế cần đưa vào protocol cuối |
| D-009 | Kiểm tra duplicate trước khi quyết định tạo duplicate nhân tạo | `PROPOSED` | Chưa audit dataset |
| D-010 | Chưa viết code trước khi duyệt Experimental Protocol | `CONFIRMED` | Yêu cầu quy trình hiện tại |

### 17.1. Decision Log Template

Khi phát sinh quyết định mới, bổ sung theo mẫu:

| ID | Ngày | Quyết định | Lý do | Trade-off / Ảnh hưởng | Trạng thái |
|---|---|---|---|---|---|
| D-XXX | YYYY-MM-DD | Nội dung | Lý do | Ảnh hưởng đến scope/method/timeline | PROPOSED/CONFIRMED |

---

## 18. Research / Implementation Notes

Mục này dùng để ghi các quyết định kỹ thuật mới, thay đổi so với Proposal và lưu ý phát sinh trong quá trình nghiên cứu.

### 18.1. Notes hiện tại

- Proposal hiện là bản thiết kế sơ bộ, chưa phải Experimental Protocol cuối cùng.
- Chưa có kết quả thực nghiệm.
- Chưa xác minh duplicate trong dataset.
- Chưa có Research Gap được chứng minh bằng Literature Review.
- Preprocessing Leakage không sử dụng nhãn; tác động có thể nhỏ và cần được diễn giải đúng.
- Feature Leakage cần mô hình hóa theo tính khả dụng của feature tại thời điểm dự đoán.
- Duplicate Leakage cần định nghĩa duplicate/group có cơ sở; không được suy diễn danh tính giao dịch từ các feature ẩn danh.
- Clean holdout phải độc lập và không được dùng cho bất kỳ bước fit, tuning hoặc model selection nào.

### 18.2. Notes Template

```text
Date:
Related phase:
Related experiment:
Change / Decision:
Reason:
Alternatives considered:
Trade-offs:
Impact on Research Questions:
Impact on Hypotheses:
Impact on Timeline:
Approved by:
Status:
```

---

## 19. Source and Reference Management

### 19.1. Nguồn đã xác định

1. Kaggle – Credit Card Fraud Detection:  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Scikit-learn – Common pitfalls and recommended practices:  
   https://scikit-learn.org/stable/common_pitfalls.html

3. Google Scholar – Data Leakage in Machine Learning:  
   https://scholar.google.com/scholar?q=data+leakage+machine+learning+predictive+modeling

4. Google Scholar – Duplicate Leakage:  
   https://scholar.google.com/scholar?q=machine+learning+data+leakage+duplicate+records

### 19.2. Quy tắc quản lý tài liệu khoa học

Với mỗi bài báo, phải ghi:

| Field | Nội dung |
|---|---|
| Paper ID | Mã nội bộ |
| Title | Tên bài báo |
| Authors | Tác giả |
| Year | Năm xuất bản |
| Venue | Journal / Conference |
| DOI | DOI nếu có |
| URL | Link truy cập |
| Research objective | Mục tiêu |
| Methodology | Phương pháp |
| Dataset | Dữ liệu |
| Findings | Kết quả chính |
| Limitations | Hạn chế |
| Relevance | Liên quan đến project ở điểm nào |
| Leakage category | Nhóm Leakage liên quan |

**Quy tắc:** Không đưa một tài liệu vào References chính thức chỉ dựa trên tiêu đề hoặc snippet tìm kiếm. Phải kiểm tra thông tin thư mục và nội dung nguồn.

---

## 20. Instructions for AI Assistants

AI assistant của các thành viên phải đóng vai trò research assistant và technical collaborator, không phải người tự quyết định thay nhóm.

### 20.1. Vai trò

- Research assistant: hỗ trợ tìm kiếm, đọc, phân loại và tổng hợp tài liệu khoa học.
- Research method advisor: phản biện RQ, hypotheses, variables và experimental design.
- ML technical assistant: hỗ trợ thiết kế pipeline, kiểm soát Leakage, đánh giá mô hình và debug.
- Technical writer: hỗ trợ viết Proposal, Literature Review, Methodology, Results và Discussion.
- Reviewer: kiểm tra tính nhất quán giữa mục tiêu, câu hỏi, giả thuyết, phương pháp và kết luận.

### 20.2. Quy tắc làm việc

1. Luôn đọc tài liệu handoff trước khi hỗ trợ project.
2. Hỏi làm rõ khi thiếu thông tin ảnh hưởng đến quyết định nghiên cứu.
3. Trình bày kế hoạch trước khi thực hiện task lớn.
4. Không tự ý đổi dataset, Leakage categories hoặc model.
5. Không tạo số liệu, kết quả hoặc tài liệu tham khảo giả.
6. Phân biệt fact, assumption, hypothesis và experimental result.
7. Khi đề xuất phương pháp mới, giải thích lý do, ưu nhược điểm và trade-off.
8. Khi review phương pháp, chủ động tìm confounding factors và threats to validity.
9. Không viết code trước khi nhóm xác nhận Experimental Protocol.
10. Mọi thay đổi ngoài Proposal phải được ghi vào Decision Log và Research / Implementation Notes.

### 20.3. Nguyên tắc đánh giá chất lượng nghiên cứu

AI phải kiểm tra:

- Research problem có rõ ràng không?
- RQ có thể trả lời bằng dữ liệu và phương pháp đã chọn không?
- Hypotheses có thể kiểm chứng không?
- Leakage scenario có thực sự là Leakage theo định nghĩa không?
- Clean holdout có bị sử dụng sai mục đích không?
- Các mô hình có được so sánh công bằng không?
- Metrics có phù hợp với class imbalance không?
- Kết luận có được hỗ trợ bởi kết quả không?
- Limitations và threats to validity có được trình bày không?

---

## 21. Handoff Summary

Project đang nghiên cứu ảnh hưởng của ba dạng Data Leakage (Preprocessing Leakage, Feature Leakage, Duplicate Leakage) đến hiệu suất đánh giá và khả năng tổng quát hóa của ba mô hình Machine Learning (Logistic Regression, Decision Tree, Random Forest) trong bài toán phát hiện giao dịch thẻ tín dụng gian lận.

Nhóm đã chốt hướng nghiên cứu thực nghiệm, dataset và mô hình. Proposal sơ bộ đã có mục tiêu, Research Questions, Hypotheses, phạm vi, phương pháp, timeline và phân công dự kiến.

**Trạng thái hiện tại:** Chưa bắt đầu code và chưa có kết quả thực nghiệm. Công việc kế tiếp là Literature Review, Dataset Audit và hoàn thiện Experimental Protocol. Hai điểm kỹ thuật cần giải quyết trước tiên là thiết kế Feature Leakage và xác minh/định nghĩa Duplicate Leakage trên dataset.

**Nguyên tắc quan trọng nhất:** Không hy sinh tính hợp lệ của nghiên cứu để có kết quả metric đẹp. Mọi Leakage phải được định nghĩa, mô phỏng, đo lường và diễn giải minh bạch. Kết quả không như kỳ vọng vẫn phải được báo cáo trung thực.
