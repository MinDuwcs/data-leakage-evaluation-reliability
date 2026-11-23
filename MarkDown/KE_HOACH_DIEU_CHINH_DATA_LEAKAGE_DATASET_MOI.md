# KẾ HOẠCH ĐIỀU CHỈNH NGHIÊN CỨU DATA LEAKAGE — DATASET MỚI

> **Last Updated:** 2026-11-21  
> **Version:** 3.0 — Dataset Migration Plan / Experimental Protocol Amendment  
> **Current Phase:** Research Design Revision → Dataset Audit (chưa triển khai)  
> **Approval:** `PROPOSED` đối với mọi cấu hình kỹ thuật; `CONFIRMED` đối với dataset mới, ba nhóm leakage, ba mô hình và danh sách tám kịch bản cần xem xét.  
> **Đối tượng:** Nhóm nghiên cứu, giảng viên, Codex.  
> **Chưa viết code / chưa tạo kết quả thực nghiệm.**

## 0. Chỉ dẫn đọc cho nhóm và Codex

- Dataset **đã đổi** từ `mlg-ulb/creditcardfraud` sang **`ranjitmandal/fraud-detection-dataset-csv`**: https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv (`CONFIRMED` theo yêu cầu của nhóm).
- Tài liệu cũ `data_leakage_research_handoff_UPDATED.md` và `experimental_protocol_data_leakage_v1.md` là tài liệu lịch sử: những gì liên quan ULB (284.807 dòng; 492 fraud; `V1–V28`, `Time`, `Amount`, `Class`; 1.081 duplicate được báo cáo) **KHÔNG được áp dụng** cho dataset mới.
- Giữ ba nhóm **Preprocessing Leakage / Feature Leakage / Duplicate Leakage**, ba mô hình **Logistic Regression / Decision Tree / Random Forest**, tám scenario P1/P3/P6/F2/F3/F4/D1/D3.
- Chưa có CSV gốc trong phiên làm việc. Trang Kaggle xác nhận định danh dataset nhưng không cung cấp schema đọc trực tiếp qua công cụ hiện tại. **Tên cột, số dòng, nhãn, tỷ lệ lớp, duplicate, missing, thời gian và nguồn tạo dữ liệu vẫn cần audit trên file thật**.
- Một dự án GitHub bên thứ ba mô tả dữ liệu *cùng tên* `Fraud Detection Dataset.csv` là 51.000 dòng × 12 cột, với các trường ví dụ `User_ID`, `Transaction_Amount`, `Transaction_Type`, `Time_of_Transaction`, `Device_Used`, `Location`, `Previous_Fraudulent_Transactions`, `Account_Age`, `Number_of_Transactions_Last_24H`, `Payment_Method`, `Fraudulent`. **Chỉ là gợi ý để xây dựng checklist, không coi là schema đã kiểm chứng cho URL Kaggle mới.** Nguồn: https://github.com/Jeremiah95-Feng/fraud-detection-data-analysis
- `CONFIRMED`: quyết định nhóm; `PROPOSED`: đề xuất chờ duyệt; `OPEN`: thiếu bằng chứng; `CONDITIONAL`: phụ thuộc audit; `RISK`: nguy cơ validity; `TODO`: việc chưa làm.
- Không tự động sửa Overleaf, phát sinh code, chọn hyperparameters hoặc suy diễn thực nghiệm đã thành công.

## 1. Mục tiêu nghiên cứu và phạm vi giữ nguyên

**Research problem:** Data Leakage có thể làm sai lệch đánh giá của ba thuật toán phân loại gian lận đến mức nào, với ba nhóm cơ chế tiền xử lý, đặc trưng và bản ghi trùng? Cần phân biệt *điểm trên đánh giá bị nhiễm* với *khả năng dự đoán trên dữ liệu độc lập*.

**Giữ nguyên:** ba thuật toán; tám scenario ứng viên; yêu cầu baseline/contaminated evaluation/clean holdout; metrics PR-AUC (Average Precision), ROC-AUC, Precision, Recall, F1, confusion matrix, Accuracy tham khảo; không suy diễn rằng leakage luôn làm tăng hiệu suất.

**Điều chỉnh bắt buộc:** thực thể giao dịch, nhãn mục tiêu, feature types (numeric/categorical), thời gian, tính lệch lớp, nguồn gốc dữ liệu và duplicate groups phải được định nghĩa theo schema mới. Không sao chép feature engineering, class weights, SMOTE ratio, split rule hoặc con số ULB.

## 2. Tình trạng dữ liệu và độ tin cậy của thông tin

| Thành phần | Trạng thái | Hành động |
|---|---|---|
| URL/định danh Kaggle `ranjitmandal/fraud-detection-dataset-csv` | `CONFIRMED` | Ghi URL và ngày tải vào audit |
| Dataset CSV thực tế/phiên bản/checksum | `OPEN` | Tải file, lưu SHA-256, ghi license và provenance |
| Số hàng/cột/schema/tên nhãn | `OPEN` | Inspect CSV trực tiếp; không hard-code cột cũ |
| Numeric / categorical / ID / thời gian | `OPEN` | Lập feature dictionary, availability-at-prediction |
| Missing, invalid, class distribution | `OPEN` | Đếm và lưu báo cáo |
| Exact duplicates (toàn hàng và X-only), label conflicts | `OPEN` | Đếm nhóm theo quy tắc xác định |
| Customer/user ID, timestamps và nguồn dữ liệu | `OPEN` | Kiểm tra ý nghĩa, độ tin cậy và rủi ro split |
| Dữ liệu thực hay synthetic | `OPEN` | Xem dataset card/provenance; không tự gắn nhãn “thực tế” |

**Điểm cần phản biện:** Nếu các trường “previous fraud”, “transactions last 24h” hoặc thời gian chỉ là snapshot được tạo sau khi có outcome thì chúng có thể đã chứa leakage gốc. Không được mặc nhiên xem các trường này là feature hợp lệ hay không hợp lệ nếu không biết thời điểm quan sát.

## 3. Dataset Audit Gate (bắt buộc trước protocol freeze)

| ID | Kiểm tra | Điều kiện nghiệm thu |
|---|---|---|
| A01 | File identity | URL, tác giả, file name, version/date, license, SHA-256; giữ raw bất biến |
| A02 | Schema | Shape, dtypes, column list, label mapping, distinct/unique/null/invalid |
| A03 | Target | Xác minh nhãn 0/1 và ý nghĩa; số mẫu mỗi lớp; minority count từng split |
| A04 | Feature provenance | Mỗi feature: kiểu dữ liệu, có sẵn lúc dự đoán?, hậu nghiệm?, ID?, cần encode? |
| A05 | Duplicate | Exact full-row, exact-feature-X (loại nhãn), conflicting labels, group size, cross-split overlap |
| A06 | Time/user structure | Có timestamp theo ngày hay chỉ giờ? Có user/card ID? Một user xuất hiện nhiều lần? |
| A07 | Train/evaluation feasibility | Positive samples đủ cho stratify/CV/SMOTE? categorical strategy phù hợp? |
| A08 | Data integrity | Missing/outliers, cardinality, possible duplicated synthetic templates, strong target proxies |

Deliverables: `dataset_audit.md`, `schema_dictionary.md`, `duplicate_audit.md`, `split_design_decision.md` và manifest file/raw checksum. **Không tạo báo cáo kết quả giả**.

## 4. Khung đánh giá đối chứng chung (đề xuất)

### 4.1. Tầng dữ liệu

1. Audit `raw` chỉ dùng các phép đo mô tả và xây dựng duplicate group keys; không tính mean/variance/target rates để đưa vào model trước split.
2. Tạo `clean_holdout` ngay sau audit theo quyết định split đã được duyệt; không để bản ghi hoặc **duplicate group** xuyên sang development. Nếu có `User_ID` và mục tiêu dự đoán cho user mới, cân nhắc user-group split; nếu mục tiêu là giao dịch tương lai của user đã biết, cân nhắc temporal split. **Không khẳng định một split là luôn tối ưu**.
3. Từ development chia Train/Validation (hoặc CV có group constraints). Freeze `row_id`, group assignment, seed, class distribution; không thay đổi theo kết quả metric.
4. Chạy control và contaminated scenario trên **cùng tập dòng gốc/nhãn hợp lệ khi có thể**, giữ model configs và ngưỡng tương ứng nhất quán. Đo cả thay đổi class prevalence, group overlap, feature availability.
5. Chỉ dùng clean holdout sau khi khóa quy tắc xử lý/mô hình; nó không tham gia leakage injection, transformer fitting, SMOTE, selection hay threshold tuning. Nếu design gây thay đổi feature availability thì mô tả rõ clean deployment evaluation có thể không áp dụng trực tiếp cùng mô hình.

### 4.2. Ba mức kết quả phải tách biệt

- `CONTROL_VALID`: Pipeline hợp lệ trên validation/disjoint split, benchmark đối chứng.
- `CONTAMINATED_EVAL`: Đánh giá cố ý nhiễm (dùng nhãn/feature/duplicate bất hợp lệ theo scenario), không được gán là generalization.
- `CLEAN_HOLDOUT`: Đánh giá cuối trên dữ liệu không tham gia quyết định, **chỉ khi có quy tắc feature và transformations hợp lệ, xác định trước**.

**Không áp dụng một công thức `contaminated metric - clean holdout metric` máy móc cho mọi scenario:** Chênh lệch có thể đồng thời phản ánh distribution shift, feature availability hoặc lựa chọn feature set.

### 4.3. Common settings (PROPOSED)

- Mô hình: LR/DT/RF; biến phân loại phải được encode bằng transformer fit chỉ trên Train trong control; numeric imputation/scaling chỉ fit Train. LR nên có scaling; DT/RF không bắt buộc scaling.
- Imbalanced classification: chưa khẳng định cần SMOTE trước khi xem tỷ lệ và số lượng lớp. P3 phải có minority samples đủ theo `k_neighbors` trong mỗi training fold; nếu không thì đánh dấu không khả thi hoặc giảm k hợp lệ sau phê duyệt.
- Chia dữ liệu: đề xuất development 80%, locked holdout 20% là **ứng viên**, không phải default bắt buộc. Chọn giữa random stratified, grouped, temporal và kết hợp dựa trên audit. Với time-only (giờ trong ngày), không gọi đó là timeline để temporal split.
- Seeds: thí nghiệm lặp nhiều seed đã định trước, đề xuất 5 seed; nếu temporal split cố định thì lặp seed áp dụng cho model/sampling (không đảo timeline).
- Threshold cho F1/Precision/Recall: cố định 0.5 hoặc tune trên inner validation, không dùng test/holdout để tune; áp dụng nhất quán.
- Metrics: Average Precision (AP) ưu tiên nếu lệch lớp; ROC-AUC; Precision/Recall/F1; confusion matrix; Accuracy phụ trợ. Báo cáo prevalences vì AP/Precision phụ thuộc class ratio.
- Cần metadata cho mỗi run: dataset SHA, scenario, branch, seed, group/split fingerprint, preprocessing, missing strategy, encoder, sampling ratio, overlap rate, hyperparams, threshold, metrics, runtime.

## 5. Protocol chi tiết từng scenario — điều chỉnh với schema mới

### P1 — StandardScaler Leakage (`CONDITIONAL`)

- **Prerequisite:** Có feature numeric với biến thiên đủ; đánh dấu numeric features hợp lệ, bỏ ID và target.
- **Control:** Split trước; numeric imputer/scaler fit chỉ Train, transform Validation. Categorical pipeline xử lý y hệt cả hai nhánh.
- **Contaminated:** Scaler fit trên Train + Validation (chỉ trong development) rồi train/đánh giá cùng các dòng. **Không bao gồm clean holdout** trong fit; như vậy đo rò rỉ validation, không gây rò rỉ holdout.
- **Clean:** So sánh model control trên holdout; muốn đánh giá mô hình của nhánh nhiễm trên holdout phải ghi rõ nó đã học thống kê validation, trong khi holdout chưa từng tham gia fitting. Điều này không phải clean training pipeline.
- **Biến kiểm soát:** Numeric feature set, imputation, encoding, seed, model config, splits.
- **Thước đo:** metric delta, scaler parameter delta, biến thiên seed. **RISK:** LR có thể nhạy hơn cây; mức tác động có thể bằng 0.

### P3 — SMOTE Leakage (`CONDITIONAL`, nguy cơ confounding cao)

- **Prerequisite:** Binary target và mất cân bằng đủ để SMOTE có mục đích; ít nhất đủ minority mẫu trong từng fold; categorical data cần pipeline mã hóa đúng hoặc phương pháp có hỗ trợ categorical phù hợp. Không áp dụng vanilla SMOTE trực tiếp lên label-encoded nominal categories.
- **Control:** Split trước; oversample **chỉ Train** sau phép biến đổi fit Train; Validation giữ tỷ lệ gốc.
- **Contaminated:** Oversample development pool trước khi chia contaminated Train/Validation, để các mẫu tổng hợp liên quan dữ liệu Validation và có thể bị cross-split dependency. Giữ clean holdout bên ngoài.
- **Extra control bắt buộc:** Thử phân tích đánh giá trên Validation **gốc**, và contaminated Validation đã resample; ghi rõ case-mix/prevalence và dependency. Nếu không tách được hai yếu tố, gọi kết quả là combined resampling-and-leakage effect, không khẳng định leakage-only effect.
- **Clean:** Model mỗi nhánh chỉ infer trên holdout với hợp đồng feature hợp lệ; không fit SMOTE vào holdout.
- **RISK:** Dataset mới có thể ít/lớn mất cân bằng hoặc chưa đủ fraud samples; P3 có thể phải điều chỉnh hoặc đánh dấu infeasible.

### P6 — Supervised Feature Selection Leakage (`CONDITIONAL`)

- **Prerequisite:** Có ít nhất vài feature candidates hợp lệ sau encoding và đủ sample; không chọn target/ID làm predictor.
- **Control:** Feature selection sử dụng Train labels (CV thì fit selection trong mỗi Train fold).
- **Contaminated:** Selector học từ Train + Validation labels; train model trên Train được chọn, đánh giá trên Validation (same rows).
- **Clean:** Locked holdout chỉ được dùng để đánh giá khi toàn bộ mapping feature (và categories) đã được xây hợp lệ mà không tham gia holdout. So sánh mức optimistic bias và selected feature overlap.
- **RISK:** Nếu chỉ có ~10–12 cột, selection effects có thể nhỏ; one-hot có thể tạo chiều cao nhưng cần tránh leakage khi encode.

### F2 — Noisy Target Leakage (`CONDITIONAL`, synthetic experiment)

- **Prerequisite:** Định nghĩa rõ tại sao feature `post_event_proxy` không khả dụng ở thời điểm dự đoán.
- **Control:** Input chỉ gồm feature xác định hợp lệ tại thời điểm giao dịch.
- **Contaminated:** Tạo feature hậu nghiệm giả lập từ ground-truth `y` với xác suất nhiễu được chốt trước (ví dụ 5%, 20%, 40% — **PROPOSED**), trên **Train/Validation thuộc development**; mô hình được tiếp cận nhãn thông qua proxy này.
- **Clean:** Không được tạo proxy từ nhãn clean holdout. Chỉ có thể đánh giá triển khai bằng **một chính sách thiếu proxy đã định trước** và coi kết quả này là deployment-mismatch stress test; hoặc đánh dấu clean same-model evaluation `NOT_COMPARABLE`. Đặt baseline hợp lệ trên holdout làm đối chứng riêng.
- **Outputs:** contamination metrics, gradient theo mức nhiễu, mức độ sử dụng proxy, limitation.
- **RISK:** Khi mô phỏng F2, kết quả thể hiện cơ chế giả định chứ không là đặc điểm thực tế ngành tài chính.

### F3 — Post-event Feature Leakage (`CONDITIONAL`, tránh trùng F2)

- **Prerequisite:** Audit `Previous_Fraudulent_Transactions` và bất kỳ cột lịch sử/risk score nào: cần bằng chứng chúng có sẵn *trước* dự đoán. Không tự coi tên cột như chứng cứ hậu nghiệm.
- **Control:** Chỉ dùng feature có khả năng sẵn có tại thời điểm giao dịch.
- **Contaminated:** Một trong hai cách (chọn sau audit): (a) khai thác feature thật có nguồn gốc sau outcome nếu có chứng cứ; (b) mô phỏng `investigation_result` chỉ có sau giao dịch và ghi rõ là synthetic. Khác F2 ở **availability delay/missingness** và mốc thời gian; không chỉ đổi tên F2.
- **Clean:** Nếu feature hậu nghiệm không có lúc dự đoán, không thể đánh giá một cách tương đương mà không có missing-value rule đã phê duyệt. So với baseline trên clean holdout và báo cáo deployability gap, không diễn giải là independent generalization của cùng feature set.
- **RISK:** Feature history có thể hoàn toàn hợp lệ nếu là lịch sử quá khứ. Cần chứng minh thời gian khả dụng.

### F4 — Target Encoding Leakage (`CONDITIONAL`, thích hợp nếu có categorical)

- **Prerequisite:** Tồn tại categorical feature có đủ nhóm và support. Các trường có khả năng kiểm tra: `Transaction_Type`, `Device_Used`, `Location`, `Payment_Method` nếu có trong CSV thật.
- **Control:** Target encoding cho Train bằng out-of-fold/cross-fitting; validation mapping fit **chỉ Train** (smoothing/unknown category fallback xác định trước).
- **Contaminated:** Tính mean target cho category bằng Train + Validation labels, hoặc self-inclusive target means; đo rõ mỗi dạng vì self-inclusion và validation-label inclusion không hoàn toàn giống nhau.
- **Clean:** Holdout được transform bằng mapping học Train (đối chứng); không sử dụng nhãn holdout cho encoding dưới bất cứ hình thức nào. Đối với nhánh nhiễm, có thể transform holdout bằng mapping Train+Validation nhưng phải nêu rằng model đã sử dụng validation labels trong development.
- **RISK:** Target encoding là một dạng label-dependent feature construction có giao thoa với preprocessing. Phân loại F4 thuộc Feature Leakage theo thiết kế nhóm; trong Methodology mô tả chính xác ranh giới taxonomy.

### D1 — Existing Exact Duplicate Leakage (`CONDITIONAL`, không bảo đảm có duplicate)

- **Prerequisite:** Audit exact duplicate full-row, **X-only** và label conflicts; nhận diện `User_ID`/transaction ID và phân biệt nhóm user với bản ghi trùng. Trùng cùng feature nhưng khác label phải báo cáo riêng.
- **Control:** Group-disjoint split trên duplicate groups, stratification nếu khả thi. Giữ cùng tập quan sát và cơ chế metric để giảm confounding.
- **Contaminated:** Random row split cho phép duplicate group xuất hiện ở Train và Validation; đo thật cross-split overlap, không giả định có.
- **Clean:** Holdout tách theo duplicate group; không để bản sao cùng nhóm từ development lọt vào holdout.
- **RISK:** Nếu **không tồn tại exact duplicates đủ đáng kể**, D1 chỉ là audit finding/negative result; không bịa duplicate và gọi là D1 tự nhiên. Chuyển trọng tâm nghiên cứu sang D3 theo quyết định nhóm.

### D3 — Duplicate Ratio Sensitivity (`PROPOSED`, có thể mô phỏng)

- **Prerequisite:** Quy định rõ seed rows, split, mục tiêu duplication, lớp được chọn và metric đo độ nhiễm; không dùng clean holdout để chọn bản sao.
- **Control:** Development Train/Validation group-disjoint, không tạo cross-split clones.
- **Contaminated:** Nhân bản mẫu Train rồi đưa bản sao vào **Validation contaminated**, hoặc inject bản sao từ Validation vào Train cho nghiên cứu memorization (phải chọn **một hướng chủ đạo** trong protocol freeze). Duy trì một natural Validation cho phép đo tác động trên dữ liệu không bị duplicate.
- **Levels:** Đề xuất 0%, 5%, 10%, 20%, 40% cross-split contaminated share (`PROPOSED`; tỷ lệ đo trên các dòng validation), giữ class prevalence bằng thiết kế stratified nếu có thể; ghi actual overlap.
- **Clean:** Locked group-disjoint holdout; model không được nhìn thấy clones của bất kỳ holdout row nào.
- **RISK:** Thay đổi size/prevalence khi thêm bản sao khiến kết quả không còn là tác động leakage đơn lẻ. Cần sample-size matched comparator và báo cáo class balance.

## 6. Decision table: thay đổi so với protocol ULB

| Hạng mục | Trước đây | Kế hoạch mới |
|---|---|---|
| Dataset | ULB/PCA 28 features | Kaggle Ranjit Mandal, schema mới phải audit |
| Label | `Class` | `OPEN`; map từ nhãn file thực tế (không hard-code) |
| Encoding | Chủ yếu numeric | Nếu có categorical: OneHotEncoder/target encoding có kiểm soát |
| Split | Stratified/group duplicate theo ULB | Chọn group/time/stratification theo user/time schema mới |
| P1 | Numeric scale `Time`, `Amount`, ... | Scale numeric thực; tránh ID/categorical |
| P3 | SMOTE trên PCA numeric | Conditional; phải xử lý categorical và class ratio |
| P6 | Selection trong 30 input | Selection trong feature matrix mới, có thể chiều thấp |
| F2/F3 | Feature nhân tạo | Kiểm tra provenance của history-derived columns trước; synthetic nếu cần |
| F4 | Có thể bin Time/Amount | Ưu tiên categorical thực nếu xác nhận tồn tại |
| D1 | Duplicate gốc ULB được báo cáo | Audit lại từ đầu, **không dùng 1.081** |
| D3 | Inject duplicates | Giữ cơ chế, xác định tỷ lệ bằng split mới |
| Generalization | Clean holdout độc lập | Giữ nguyên, bổ sung group/time contract |

## 7. Kế hoạch triển khai theo phase và nghiệm thu

| Phase | Nhiệm vụ | Deliverable | Exit criteria |
|---|---|---|---|
| Planning P0 | Chốt thay dataset, chặn assumptions ULB | Decision log + plan v3 | Không còn ULB-specific assumptions trong active protocol |
| Planning P1 | Audit file thật, schema, provenance, target, missing, classes, duplicate, time/user | Audit reports + data dictionary | Mọi cột/kích thước/số lượng được xác minh bằng CSV |
| Planning P2 | Chọn split strategy, feature availability và leakage audit | Split protocol + feature availability matrix | Không overlap ngoài chủ đích; holdout locked |
| Planning P3 | Chốt tám scenario: control, contamination, clean eligibility, levels, confounders | Experimental Protocol v2 `APPROVED` | Nhóm/giảng viên duyệt, có rule cho inconclusive/infeasible |
| Execution E1 | Baseline ba model | Baseline logs + tests | Reproducible, không leakage, metrics đúng |
| Execution E2 | P1/P3/P6 | Scenario run logs | Mỗi nhánh có provenance, same-split comparator |
| Execution E3 | F2/F3/F4 | Synthetic feature log, deployment validity notes | Không dùng holdout labels tạo feature |
| Execution E4 | D1/D3 | Duplicate overlap manifests | Group disjoint control; contamination mức đo được |
| Review R1 | Phân tích effect sizes, seed variability, contradictions | Results / Discussion / limitations | Không cherry-pick hoặc đồng nhất metric nhiễm với generalization |
| Review R2 | Đồng bộ Overleaf, bảng, References, slides | Paper + presentation | Mọi số liệu có logs và nguồn; nomenclature nhất quán |
| Retro | Review nguồn dữ liệu, confounding, scope creep | Research notes | Bài học và quyết định được ghi lại |

**Ưu tiên công việc tiếp theo:** `P1 Dataset Audit` → `P2 Split & feature provenance` → `P3 Protocol Freeze`. Không code trước khi nhóm duyệt. Nếu nhóm chỉ cho phép audit code như ngoại lệ, phải ghi thành quyết định riêng.

## 8. Risks / stop rules

1. **Không xác định được target hợp lệ:** Dừng thiết kế supervised models và xác minh nhãn.
2. **Không đủ fraud samples:** Dừng hoặc điều chỉnh P3, CV/split; tránh overfit đánh giá bằng một số mẫu dương quá ít.
3. **Không có exact duplicates:** D1 chỉ báo cáo negative audit; không tráo D3 nhân tạo thành D1.
4. **Không có categorical feature:** F4 cần thiết kế derived bins hợp lệ (fit bins Train) hoặc đánh dấu `NOT_FEASIBLE`; không tự tạo nhãn test để chia bins.
5. **Không có timestamp thực:** Không thực hiện time-based leakage scenario/temporal holdout như thể có chuỗi thời gian đầy đủ.
6. **Lịch sử gian lận không rõ thời điểm:** Tạm loại feature nghi vấn khỏi clean baseline, chờ provenance.
7. **Feature post-event không khả dụng trên clean holdout:** Không báo cáo same-model clean score giả; ghi `NOT_COMPARABLE` và so baseline/deployment mismatch.
8. **Dữ liệu synthetic không rõ nguồn:** Không khái quát sang giao dịch ngân hàng thực tế.
9. **Tỷ lệ lớp thay đổi:** Báo cáo prevalences và dùng matched-control; không quy hoàn toàn thay đổi AP/Precision cho leakage.

## 9. Decision Log

| ID | Ngày | Quyết định | Status | Trade-off / hệ quả |
|---|---|---|---|---|
| D-011 | 2026-11-21 | Chuyển dataset sang `ranjitmandal/fraud-detection-dataset-csv` | `CONFIRMED` | Phải audit lại schema, duplicate, preprocessing và các giả định ULB |
| D-012 | 2026-11-21 | Giữ 3 nhóm leakage và 3 mô hình | `CONFIRMED` | Không tự mở rộng thuật toán |
| D-013 | 2026-11-21 | Tiếp tục xét P1/P3/P6/F2/F3/F4/D1/D3 | `CONFIRMED` (danh sách), `CONDITIONAL` (khả thi) | Không ép kịch bản nếu data audit không hỗ trợ |
| D-014 | 2026-11-21 | Dùng dataset audit làm gating step trước freeze | `PROPOSED` | Chậm code nhưng tránh sai nền tảng |
| D-015 | 2026-11-21 | Giữ clean holdout độc lập, kiểm soát group/time nếu có | `PROPOSED` | Cần thiết kế split theo dữ liệu thật |
| D-016 | 2026-11-21 | Không tự động sửa bản thảo Overleaf và chưa viết code | `CONFIRMED` (theo workflow hiện hành) | Chốt phương pháp trước implementation |

## 10. Research / Implementation Notes

**2026-11-21 · Dataset Migration:**
- Dataset mới **không phải ULB dataset**, xóa khỏi mọi suy luận đang áp dụng các fields `V1–V28`, `Time`, `Amount`, `Class` trừ khi CSV mới có tên tương ứng thực sự.
- Thông tin schema từ một repository bên thứ ba chỉ là *lead*, chưa phải evidence định danh cùng một file.
- P3, F4, D1 là các scenario chịu tác động trực tiếp nhất nếu schema, imbalance hoặc duplicate khác dự đoán.
- Các kịch bản F2 và F3 phải không trùng cơ chế: F2 đo strength/noise của hậu nghiệm; F3 đo availability delay/missingness.
- D1 duplicate grouping phải xem xét bỏ các ID không liên quan (nếu có) khi xác định X-only duplicates, nhưng **không được** suy luận giao dịch thực là một chỉ từ giống feature.
- Chưa có kết quả audit, hiệu suất mô hình, test, biểu đồ, thống kê rò rỉ hoặc kết luận.

## 11. Nguồn và giới hạn xác minh

1. Dataset mới (nguồn chính): https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv — xác nhận tên và định danh; **chưa đọc trực tiếp CSV**.
2. Trang thống kê danh mục Kaggle (nguồn thứ cấp): https://nichedb.dev/i/752408279 — mô tả metadata, không thay thế dataset audit.
3. Repository ví dụ **chưa xác minh trùng chính xác dataset**: https://github.com/Jeremiah95-Feng/fraud-detection-data-analysis — chỉ định hướng những cột có thể gặp.
4. Scikit-learn Common Pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
5. Scikit-learn TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
6. Imbalanced-learn Common Pitfalls: https://imbalanced-learn.org/stable/common_pitfalls.html

## 12. Hướng dẫn chuyển giao Codex

1. Đọc `KE_HOACH_DIEU_CHINH_DATA_LEAKAGE_DATASET_MOI.md` trước, rồi tham khảo các phiên bản lịch sử.
2. **Đừng viết implementation hoặc khẳng định dataset schema trước khi nhóm yêu cầu.**
3. Task ngay sau khi được nhóm cho phép: lập/check Dataset Audit (schema/target/class/missing/duplicate/time/provenance), không tự chạy huấn luyện.
4. Xuất danh sách cột thực tế và bảng khả thi `FEASIBLE / CONDITIONAL / NOT_FEASIBLE` của 8 scenario.
5. Đề xuất split strategy + holdout + contamination measurement dựa trên số liệu audit; chờ xác nhận.
6. Sau approval mới update Experimental Protocol chi tiết sang bản v2.0; ghi quyết định phát sinh vào `implementation-notes.md` và Decision Log.
7. Không sửa Overleaf khi chưa có chỉ định; nếu chỉnh, đánh dấu mọi số liệu `TODO` cho đến khi có logs hợp lệ.
