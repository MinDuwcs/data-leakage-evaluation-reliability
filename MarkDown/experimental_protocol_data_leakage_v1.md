# EXPERIMENTAL PROTOCOL v1.0 — Data Leakage trong Machine Learning

> **Ngày soạn:** 2026-11-21  
> **Trạng thái:** `PROPOSED — FOR TEAM REVIEW`; **CHƯA PHÊ DUYỆT**, **CHƯA VIẾT CODE**, **CHƯA CHẠY THỰC NGHIỆM**.  
> **Nguồn dự án:** `data_leakage_research_handoff_UPDATED.md` (v2.0, 2026-11-13).  
> **Mục tiêu sử dụng:** Tài liệu độc lập cho nhóm và Codex phản biện/thẩm định thiết kế trước khi triển khai.  
> **Quy tắc trạng thái:** `CONFIRMED` = nhóm đã chốt; `PROPOSED` = thiết kế đề xuất; `OPEN` = chờ kiểm chứng/quyết định; `RISK` = nguy cơ; `TODO` = công việc chưa làm.

## 0. Executive summary

**Đề tài:** Nghiên cứu thực nghiệm tác động của Data Leakage đến độ tin cậy của đánh giá mô hình ML trong bài toán phát hiện gian lận giao dịch thẻ tín dụng.

**CONFIRMED:** Dataset ULB/Kaggle `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`; 3 nhóm **Preprocessing Leakage, Feature Leakage, Duplicate Leakage**; 3 mô hình **Logistic Regression, Decision Tree, Random Forest**; bỏ phân chia E1–E3 cũ; nghiên cứu nhiều kịch bản; **chưa viết code cho đến khi duyệt protocol**.

**Danh sách 8 kịch bản được người dùng chọn để thiết kế chi tiết:**

| ID | Nhóm | Tên |
|---|---|---|
| P1 | Preprocessing | StandardScaler Leakage |
| P3 | Preprocessing | SMOTE Leakage |
| P6 | Preprocessing | Supervised Feature Selection Leakage |
| F2 | Feature | Noisy Target Leakage |
| F3 | Feature | Post-event Feature Leakage |
| F4 | Feature | Target Encoding Leakage |
| D1 | Duplicate | Existing Exact Duplicate Leakage |
| D3 | Duplicate | Duplicate Ratio Sensitivity |

**Ranh giới:** 8 kịch bản này được chọn để **thiết kế**, chưa mặc định tất cả đã đạt điều kiện triển khai. Không trình bày chỉ số thực nghiệm/duplicate audit chưa được xác minh. Không sửa bản thảo Overleaf khi chưa được yêu cầu.

## 1. Research questions, hypotheses và estimands

| ID | Câu hỏi |
|---|---|
| RQ1 | Mỗi cơ chế leakage làm thay đổi phép đánh giá như thế nào? |
| RQ2 | Sự thay đổi có khác nhau giữa 3 thuật toán không? |
| RQ3 | Contaminated metric sai khác thế nào so với đánh giá hợp lệ, và giới hạn suy luận về generalization là gì? |
| RQ4 | Quy tắc nào có thể ngăn mỗi loại leakage? |

**Giả thuyết định hướng (không phải kết luận):** H1 có ít nhất một cơ chế tạo chênh lệch đáng quan sát; H2 chênh lệch thay đổi theo model; H3 một số contaminated evaluation không phản ánh hiệu suất hợp lệ; H4 phân tách dữ liệu, fit/sampling đúng scope, đảm bảo feature availability và group-aware split chặn được đường rò rỉ.

**Ký hiệu thống nhất:**
- `M(model, branch, eval_set, seed, metric)`: giá trị metric; chỉ so trực tiếp nếu mẫu/nhãn/mức prevalence/test size giống nhau, hoặc đã dùng phương pháp chuẩn hóa rõ ràng.
- `Delta_contam = M(leakage, contaminated_eval) - M(control, comparable_valid_eval)` cho cùng metric, model và seed. Chỉ tính/diễn giải khi **comparable**.
- `Delta_clean = M(leakage_model, clean_holdout) - M(control_model, clean_holdout)` **chỉ** nếu đầu vào model ở clean holdout hợp lệ và cùng schema, và giao thức không dùng nhãn holdout để tạo feature.
- `Optimism gap = M(leakage, contaminated_eval) - M(leakage_model, clean_holdout)` **chỉ** nếu hai population đủ tương đương và clean inference hợp lệ; nếu không thì báo riêng, **không** gọi đó là ước lượng bias không điều kiện.
- Với Feature Leakage hậu nghiệm F2/F3, primary estimand là **độ tăng hiệu suất đánh giá phi thực tế khi dùng feature không khả dụng**; **không bắt buộc** có `Delta_clean` cho cùng model. Đánh giá thiếu-feature ở deployment là một bài toán khác và cần policy riêng.
- Với SMOTE, metric trên test bị đổi prevalence không phải ước lượng trực tiếp trên phân phối gian lận gốc.

## 2. Dataset audit (gate bắt buộc trước protocol freeze)

**Mô tả từ dataset, chưa audit file CSV:** 284.807 bản ghi; 492 gian lận (~0,173%); `Time`, `V1..V28`, `Amount`, `Class` (nhãn 1 = fraud); `V*` là thuộc tính PCA/ẩn danh. Dataset không công bố `customer_id`, `transaction_id`, thời điểm kết thúc điều tra.

**Audit checklist (`TODO`):**
1. Lưu source URL, thời gian tải, số byte, hash SHA-256, tên/phiên bản file; kiểm tra toàn vẹn.
2. Shape/schema, dtype, phân phối `Class`, missing/infinite values, các feature hằng, cardinality, phân phối `Time`, `Amount`.
3. Exact duplicates theo **(toàn bộ feature + label)** và theo **feature-only**; thống kê số group, group sizes, số group có nhãn xung đột, fraud/legitimate duplicates; phân biệt `duplicated()` count với số nhóm.
4. Lưu `source_row_id` **chỉ để audit/provenance**, tuyệt đối không làm predictive feature.
5. Xác minh các nhóm trùng exact có đủ dữ liệu để D1 có ý nghĩa. Các số duplicate truyền miệng từ nguồn thứ cấp **không** ghi là kết quả audit nhóm.
6. Kiểm tra có thể tách holdout group-disjoint và stratified mà vẫn giữ số fraud đủ đo. Nếu group-label xung đột, phải định nghĩa hướng xử lý trước split và ghi lại.
7. Không tự gán hai giao dịch là cùng sự kiện chỉ vì feature PCA gần nhau; near-duplicate không thuộc phạm vi 8 kịch bản lần này.

**Stop rule:** Nếu không lập được nhóm exact-duplicate có cơ sở thì D1 chuyển `BLOCKED` và thay bằng phần hạn chế, không dựng số liệu. Nếu không có class support đủ ở các split, phải điều chỉnh thiết kế **trước** khi xem kết quả metric.

## 3. Data governance, split hierarchy và holdout

### 3.1. Khái niệm dữ liệu

- `R`: dữ liệu raw, chỉ đọc; giữ row provenance.
- `G`: nhóm exact-match theo predictor vector (theo quy tắc thống nhất sau audit), ghi thêm nhãn để phát hiện xung đột.
- `H`: **immutable clean holdout**, group-disjoint với pool phát triển; nhãn chỉ được mở khi thực hiện **đánh giá cuối cùng sau khi freeze toàn bộ quyết định**. Không dùng `H` để fit, tạo synthetic target features, lựa chọn leakage strength, tuning, chọn threshold hoặc lựa chọn báo cáo.
- `DEV`: development pool sau khi tách H.
- `T`: train chỉ dùng fit; `V`: validation/evaluation pool trong DEV. Trong mỗi scenario có thể sử dụng `V_valid` và `V_contam`, nhưng phải ghi chính xác quan hệ và nguồn xuất xứ.
- `source_group_id`: ID kỹ thuật cho các hàng có X exact-match; không làm feature.

### 3.2. Split policy `PROPOSED`

- **Primary:** fixed group-disjoint holdout `H = 20%` raw rows (gần đúng do group constraints), còn lại `DEV = 80%`. Trong DEV, chia `T ≈ 75% DEV`, `V ≈ 25% DEV`, tạo tỷ lệ xấp xỉ **60/20/20** trên toàn bộ dữ liệu. Stratification theo `Class` khi khả thi và không phá group integrity.
- Holdout group assignment được cố định một lần, không tìm seed có điểm đẹp. Đề xuất `holdout_seed = 2026`; `development_seeds = [13, 37, 71, 101, 131]` (5 lần lặp; tùy tài nguyên, `OPEN`). Trong mỗi lần lặp, dùng cùng DEV partition cho tất cả models và các nhánh so sánh hợp lệ.
- Khi D1 cần **random split có overlap** trong DEV, giữ H hoàn toàn không bị động tới; random split được coi là evaluation bị nhiễm, so với group-disjoint V hợp lệ với cùng pool và phân phối được kiểm tra. Không thể mặc định hai split hoàn toàn tương đương nếu tỷ lệ nhãn khác.
- H được đánh giá **chỉ sau khi protocol, model configurations, threshold policy và toàn bộ bản thảo phân tích chính đã freeze**. Có thể chấm cả các mô hình hợp lệ có schema tương ứng, nhưng không dùng điểm H để thay đổi quyết định. Nếu phải thay đổi thiết kế sau khi xem H, phải báo H đã bị 'spent' và cần data ngoài để có confirmatory holdout mới.
- **Không chuẩn hóa, không SMOTE, không supervised select, không target encode trước khi tạo H.** Bước phát hiện duplicate trước split có thể đọc X và nhãn để audit/stratify, nhưng không được dùng nhãn H để tối ưu bất kỳ mô hình/treatment nào.

### 3.3. Vai trò của clean holdout trong từng kịch bản

| Scenario | H dùng như thế nào? | Lưu ý |
|---|---|---|
| P1 | Chấm model với transformer chỉ fit hợp lệ trên T; và có thể chấm leaked model nếu transformer của nó **không từng nhìn H** | H độc lập; chênh lệch V leak và H không tự động so trực tiếp nếu population đổi |
| P3 | Chấm fitted model trên H **nguyên bản**, không resample H | Metrics trên contaminated SMOTE-V có prevalence khác không so ngang AP/Precision |
| P6 | Chấm model dùng selected features trên H bằng schema y hệt và không dùng y_H chọn feature | Leaked selector có thể thấy y_V, nhưng tuyệt đối không thấy y_H |
| F2/F3 | H chỉ hỗ trợ control model dựa trên feature hợp lệ; leaked model hậu nghiệm không có đầu vào tương ứng trên H | Không sinh synthetic proxy từ y_H để gọi là clean inference |
| F4 | Leaked model có thể dùng encoder fit trên T+V nhưng không H; H transform bằng mapping learned từ DEV, không dùng y_H | Nếu encoder có test labels V, H vẫn độc lập về nhãn |
| D1/D3 | H group-disjoint / không inject copies; đánh giá tất cả mô hình có feature hợp lệ | Không đưa bất kỳ bản sao source H vào DEV |

## 4. Common model and evaluation contract

**Models `CONFIRMED`:** LogisticRegression, DecisionTreeClassifier, RandomForestClassifier. Không thêm mô hình mới nếu chưa được nhóm duyệt.

**Config `PROPOSED`:**
- Logistic Regression: fixed solver phù hợp dữ liệu, `max_iter` đủ hội tụ (ví dụ 1000), random_state theo run khi solver sử dụng; scaling `Amount` và `Time` là phần của baseline nếu có giải trình. Các `V*` vốn qua PCA nhưng có thể vẫn có scale khác; lựa chọn scope chuẩn hóa phải nhất quán.
- Decision Tree: fixed pruning/depth/leaf size sau khi chốt trên DEV, không tuned bằng V contamination / H.
- Random Forest: số cây, max depth, min samples leaf, random state, n_jobs được chốt trước.
- Không tự dùng class_weight cho một nhánh mà không dùng nhánh đối chứng. Với P3 phân biệt class_weight và resampling; lựa chọn baseline phải áp dụng nhất quán.
- Với preprocessing scenario, giữ nguyên model config giữa valid và leaked branches; feature set thay đổi chỉ khi bản thân treatment là feature selection hoặc feature leakage.

**Model training protocol:** mọi bước `fit`, `fit_resample`, supervised feature select, target encoding nằm trong training fold; với kịch bản cố tình leak, chỉ **mở rộng scope đến V trong DEV** theo đúng định nghĩa kịch bản. Các model luôn dự đoán `Class=1` là positive.

**Evaluation:**
- Primary: **Average Precision (AP)**, Recall, Precision, F1; secondary ROC-AUC, Accuracy, confusion matrix; báo cáo fraud prevalence và predicted-positive rate.
- `AP` là Average Precision; nếu tính PR-AUC trapezoidal phải nêu riêng, không gộp tên.
- Threshold: đề xuất **0.5 cố định** cho kết quả chuẩn so sánh; secondary threshold tối ưu trên **training-internal validation** (không dùng V contaminated / H), áp dụng đóng băng. `OPEN` chọn primary threshold, nhất là dữ liệu rất mất cân bằng.
- Báo cáo cả discrimination metrics (AP, ROC-AUC) và threshold-based metrics. Ở nhánh contaminated bị thay prevalence, chỉ so raw metrics khi có biện pháp matching/reweighting được tiền định.
- Báo cáo mean ± SD, median/range qua các seeds, raw per-seed, chênh lệch có dấu, và CI nếu có phương pháp hợp lệ; không tuyên bố ý nghĩa thống kê chỉ vì khác điểm. 5 seeds không tương đương 5 datasets độc lập; khi bootstrap phải tôn trọng group structure.
- Các tình huống validation chỉ có một lớp: metric không xác định → ghi `N/A` kèm lý do, không thay bằng 0 hay 1.

**Must-log per run:** `dataset_hash`, split_manifest, seed, group IDs, scenario ID, model, treatment parameters, n_samples/class_counts by split, feature list, scaler/selector/encoder fit-scope, sampler seed & provenance, duplicate overlap count/ratio, threshold, all metrics, runtime/environment/version, warnings/errors.

## 5. Scenario-specific protocols

### P1 — StandardScaler Leakage (`PROPOSED`)

**Question:** Dùng phân phối X của V để fit scaler trước khi đánh giá có làm thay đổi metric so với scaler fit từ T không?

- **Treatment variable:** statistics `mean_`, `scale_` được học từ T vs T+V. Chỉ mở rộng phạm vi transformer, không thay model, T, V hay H.
- **Control:** Train/validation split → fit StandardScaler **chỉ trên T** → transform T/V → fit model trên T → score V. H được transform bằng scaler fit T ở final.
- **Contaminated:** fit StandardScaler trên **X_T + X_V** (không lấy X_H); transform T/V → fit model trên T → score **cùng V**.
- **Fairness:** T/V indices, labels, model config, random seed, scaling columns giống nhau; duy nhất fit scope khác. Đề xuất `Time` và `Amount` là các feature được scale; **OPEN** có scale V1–V28 hay không. Nếu scale toàn bộ, làm giống nhau cả hai nhánh.
- **Evaluation:** P1 raw paired delta metric trên cùng V; sau freeze có thể đánh giá hai fitted models trên H bằng scaler riêng của mỗi branch (cả hai đều chưa thấy H).
- **Diagnostics:** compare per-feature mean/std, changed coefficients/decision scores; nếu tree predictions không đổi phải ghi nhận là kết quả hợp lệ.
- **Threat:** P1 có thể gần như không tạo hiệu ứng; không được chọn riêng seed biến thiên lớn để phóng đại.
- **Pass test:** control scaler fit rows ⊆ T; contaminated scaler fit rows = T∪V; H không thuộc fit rows.

### P3 — SMOTE Leakage (`PROPOSED`)

**Question:** SMOTE trước split làm metric bị bóp méo do synthetic neighbor/provenance crossover và class-prevalence shift đến mức nào?

- **Control:** Với T/V cố định, fit preprocessing trên T, **SMOTE chỉ trên T**, fit model trên T-resampled, score **V nguyên bản** (không SMOTE). H vẫn nguyên bản.
- **Contaminated canonical error:** Chỉ trên **DEV** (không H), áp dụng SMOTE lên toàn bộ X_DEV/y_DEV **trước** khi chia T'/V'; fit model trên T', đánh giá trên V' gồm mẫu synthetic. Đây là contaminated evaluation vì synthetic V' có thể phụ thuộc vào giao dịch nằm trong T', và class prevalence V' bị biến đổi.
- **Necessary caution:** Cặp (T,V) ban đầu và (T',V') sau resample/split không có test population đồng nhất. Vì vậy **không** dùng raw AP/Precision/F1 của V' trừ đi raw metric trên V rồi gọi toàn bộ là 'bias do leakage'.
- **Primary outputs:** (i) performance của control trên V gốc; (ii) metric contaminated trên V'; (iii) class prevalence và số synthetic/real rows mỗi tập; (iv) mức crossover theo provenance: synthetic V' được tạo từ base/neighbor thuộc T' hay T'/V' như thế nào.
- **Sensitivity/attribution plan:** thêm đánh giá *tất cả* trained variants trên H gốc, khi không động H; giữ cố định evaluation H để kiểm tra model quality. Nếu muốn tách riêng hiệu ứng prevalence, xây thêm matched real-only evaluation hoặc reweighting **định trước**, nhưng chỉ báo là secondary, và không tự gọi chúng là clean holdout.
- **Sampling levels `OPEN`:** tỷ lệ target sau SMOTE chọn trước, ví dụ minority:majority = 1:10 và 1:1; `k_neighbors` không vượt giới hạn n_minority tại từng fold; ghi seed, index provenance, fold membership; SMOTE có thể sinh synthetic trên các feature như `Time`, `Amount` nên không khẳng định mọi điểm tổng hợp là giao dịch hợp lý về nghiệp vụ.
- **Confounders:** prevalence, synthetic instances in V', duplication/near-neighbor overlap, changed training size. Cần phân tích riêng, không bảo đảm cô lập hoàn toàn chỉ một nguyên nhân.
- **Pass test:** H không xuất hiện trong fit SMOTE; no resampling V/H trong control; contaminated V' được gắn nhãn rõ, không gọi V' là test độc lập.

### P6 — Supervised Feature Selection Leakage (`PROPOSED`)

**Question:** Chọn feature bằng cả nhãn V rồi đánh giá trên chính V làm metric sai lệch như thế nào?

- **Candidate selection method:** `SelectKBest` với một supervised score thích hợp cho continuous inputs (`f_classif` là ứng viên); k chọn trước trên DEV (ví dụ k∈{5,10,20} chỉ khi được nhóm phê duyệt và chọn theo training-inner-CV, không chọn theo V/H). Chỉ 30 predictors, nên hiệu ứng có thể nhỏ.
- **Control:** fit selector trên `(X_T, y_T)`; transform T/V; fit 3 models trên selected features T; score V.
- **Contaminated:** fit selector trên `(X_T∪X_V, y_T∪y_V)`; transform T/V; train models **chỉ trên T**; score cùng V. Selector không thấy H.
- **Holdout:** sau freeze, transform H qua selector tương ứng rồi score; nếu leaked selector chỉ thấy y_V thì score H hợp lệ ở mức independent holdout (tuy quy trình phát triển model đã không hợp lệ với V).
- **Diagnostics:** feature set overlap/Jaccard, feature ranking, selected score, metrics; cùng k cả branches.
- **Threat:** select features trên nhãn của V là supervised test-label leakage; không nhầm với F2/F3 nơi predictor bản thân là hậu nghiệm. k được tuned trên V sẽ tạo thêm model selection leakage ngoài P6, cần cấm.
- **Pass test:** selectors chỉ khác nhau ở source of y; all models see same T; H không được dùng trong selection.

### F2 — Noisy Target Leakage (`PROPOSED`)

**Question:** Một proxy hậu nghiệm chứa lượng thông tin nhãn với mức nhiễu có kiểm soát làm tăng điểm số đánh giá không hợp lệ đến mức nào?

- **Synthetic definition:** `z` là nhị phân mô phỏng nguồn thông tin có **sau** kết luận điều tra, sinh từ y qua xác suất `P(z=1|y=1)=sensitivity`, `P(z=1|y=0)=false_positive_rate`. Đề xuất so sánh 2–3 cặp tỷ lệ được chốt trước; **không** gọi proxy này là feature thực của ULB hay trường ngân hàng thực tế.
- **Control:** dùng đúng 30 predictive features gốc, training on T, evaluation on V hợp lệ, cùng row IDs và nhãn.
- **Contaminated:** sinh z_T từ y_T và z_V từ y_V theo cùng rule/seed độc lập từng hàng; thêm z vào predictors T và **V**; train/score 3 models. Không sử dụng y_H hoặc z_H.
- **Primary comparison:** model với feature bất khả dụng so với model chỉ có feature hợp lệ trên **cùng V**; đây là 'oracle/post-event evaluation advantage', **không** phải generalization trong thời điểm dự đoán giao dịch.
- **Secondary deployment demonstration (optional, requires approval):** định nghĩa một policy xử lý feature không khả dụng (e.g. value constant/missing and imputer fitted on T), rồi đo degradation trên H; chỉ làm khi đã thiết kế training-time feature masking để H có schema hợp lệ. **Không** đánh giá model leaked trên H bằng z sinh từ y_H.
- **Diagnostics:** confusion matrix cho z proxy vs y trên DEV; realized sensitivity/FPR, mutual information descriptive, metric vs proxy strength.
- **Threat:** feature nhân tạo, mức signal có thể quá mạnh; phải ghi sample mechanism. Không diễn giải hệ số thành dấu hiệu gian lận thực.
- **Pass test:** z_H hoàn toàn không tồn tại; code sẽ kiểm tra mọi z được tạo từ DEV IDs mà thôi.

### F3 — Post-event Feature Leakage (`PROPOSED`)

**Question:** Việc feature chỉ hình thành sau điều tra/chargeback bị đưa vào mô hình phát hiện gian lận tại thời điểm giao dịch làm cho phép đánh giá thiếu thực tế thế nào?

- **Phân biệt với F2:** F2 khảo sát **độ mạnh/noise của proxy**; F3 khảo sát **time-of-availability/coverage**. Nếu cùng một generator chỉ đổi tên, phải **gộp** hai thí nghiệm, không báo như hai cơ chế độc lập.
- **Synthetic operationalization:** mô phỏng `investigation_flag` được phát hành sau thời điểm quyết định, với `available_at = event_time + delay` hoặc giả định kết quả hậu kiểm chỉ có ở một tỷ lệ `coverage` các giao dịch. `Time` của ULB chỉ là thời gian tương đối, không có thời điểm kết thúc investigation thực, vì thế các delay/coverage là synthetic assumptions.
- **Control / deployment-time:** chỉ dùng X gốc có sẵn lúc giao dịch; giữ hoàn toàn V/H không phụ thuộc label ngoài y để chấm.
- **Contaminated / retrospective:** tạo post-event flag ở DEV T và V từ outcome theo generator chốt trước (có thể dùng signal logic giống F2 nhưng đổi **coverage** và **availability**), điền trạng thái theo policy; fit + evaluate trên feature này như thể có sẵn tại prediction time.
- **Comparisons:** cùng original V and labels; report valid control vs retrospective contaminated, alongside fraction with flag available and signal rate. Một outcome-specific synthetic flag không được đưa sang H bằng cách suy ra từ y_H.
- **Optional realistic simulator:** mô phỏng event cutoff trên DEV, tách past/future theo `Time` và chỉ expose các flag đủ delay ở time t; **không** khẳng định mô phỏng là thời gian điều tra thực. Cần design thêm trước implementation.
- **Confounders:** missingness & coverage lệch theo class, noise signal, train/test mismatch; cố định signal characteristics giữa các mức coverage khi so.
- **Pass test:** feature hậu nghiệm được gắn source/timestamp synthetic và mọi branch kết luận rõ predict-time vs post-event-time.

### F4 — Target Encoding Leakage (`PROPOSED`)

**Question:** Dùng nhãn validation để tính tỷ lệ gian lận theo nhóm rồi dự đoán trên chính validation làm sai metric ra sao?

- **Nhóm categorical synthetic từ X hợp lệ:** `Amount_bin` (ví dụ quantile bins) hoặc `Time_bin`, boundaries học **chỉ trên T** hoặc định trước, không dùng y_V/H. Chọn số bins và smoothing từ training-only / predetermined, không theo metric V.
- **Control:** training encoding theo out-of-fold/cross-fitting trong **T**; encoder map để transform V chỉ học từ `T` labels (và fallback global prior của T cho unseen bins). Dùng feature target-encoded theo cấu hình đã chốt; đối chứng 3 models cùng feature schema.
- **Contaminated:** tính encode mapping từ `y_T ∪ y_V` rồi dùng mapping này để encode **V**; train model trên transformed T theo mapping đã nhìn y_V (mô tả chính xác nếu train T cũng encoded từ full DEV). Score chính V. Không dùng H ở mapping.
- **Secondary isolation of own-target leakage:** phân biệt (a) không dùng cross-fitting khi encode T và (b) lộ nhãn V. Hai vấn đề liên quan nhưng khác đường rò rỉ; **primary F4** chọn (b); nếu so riêng (a) phải đổi tên phân tích phụ.
- **Holdout:** transform H bằng encoder learned from DEV (không có y_H); có thể chấm leaked fitted model trên H nhưng ghi rõ training encoder đã nhìn y_V, còn H chưa bị leak.
- **Diagnostics:** bin counts/fraud counts, mapping values, smoothing parameter, encode value differences, unseen bin handling, prevalence, metrics.
- **Threat:** Fraud rất hiếm, bin rates nhiễu; số bins quá lớn sẽ tạo overfit; nếu kịch bản trở thành supervised preprocessing thì taxonomy có phần giao thoa P6. Ghi tên F4 theo *feature carries target-derived aggregate*, thừa nhận overlap taxonomy.
- **Pass test:** `y_V` không đi vào control mapping; `y_H` không xuất hiện ở cả hai; train and eval rows fixed.

### D1 — Existing Exact Duplicate Leakage (`PROPOSED`, audit-gated)

**Question:** Group exact-match chồng lấn giữa train/eval do random split có làm metric khác đi so với split group-disjoint không?

- **Prerequisite:** audit duplicate rows/groups trên original CSV, xác định `group_id = hash(X)` với xử lý chính xác string/float, so nhãn X-same. Trường hợp X-same nhưng y khác phải báo riêng, không tự gộp và không âm thầm xóa.
- **Control:** phân chia DEV T/V bằng group-aware split; không group nào xuất hiện ở cả hai. Fit model T, score V; H luôn group-disjoint.
- **Contaminated:** từ **cùng DEV** chia random row-wise T'/V' (không group constraint), đo group overlap; train T', score V'; không đưa bất kỳ hàng H vào đây.
- **Primary:** report overlap diagnostics, metric của mỗi split, class distribution, group size distribution và n rows. Difference raw metrics **không** hoàn toàn paired khi V != V'; dùng repeated split / matched criteria; gắn nhãn descriptive rather than pure causal effect.
- **Suggested stronger controlled secondary:** cố định **V group-disjoint** và đưa bản sao của một tập con V rows vào T_leak (chỉ trong DEV, không H), so với T control cùng kích thước bằng replacement rows; đây là injection-style demonstration nhưng khác D1 natural duplicates, phải ghi phương pháp riêng, không trộn kết quả.
- **Clean holdout:** chấm models trên H group-disjoint và nguyên bản; không thay H do kết quả D1.
- **Threat:** natural exact-match chưa chắc cùng giao dịch nghiệp vụ; presence of X duplicate across sets isn't always illegitimate if independent events can have identical anonymized vectors. Không được khẳng định tất cả metrics chênh lệch là do memorization nếu chưa có kiểm chứng.
- **Stop rule:** nếu overlap rất hiếm, class support quá ít hoặc group-label conflicts nghiêm trọng, chỉ làm descriptive/sensitivity chứ không hứa hiệu quả đo được.
- **Pass test:** report fraction of V groups overlapping T, group-size distribution and no overlap with H.

### D3 — Duplicate Ratio Sensitivity (`PROPOSED`)

**Question:** Tỷ lệ các mẫu đánh giá đã có bản sao/trùng hoàn toàn trong huấn luyện tác động thế nào tới metric?

- **Base:** dùng DEV T/V có group-disjoint, đã biết source row IDs; H luôn group-disjoint và không tạo clone từ H.
- **Primary contamination:** chọn `q%` các **V source rows** làm mục tiêu leak (đề xuất q={0%,5%,10%,20%}, `OPEN`), sao chép nguyên vector **X,y** của những rows đó vào training branch, giữ V nguyên bản và số dòng V/nhãn/ prevalence không đổi. Đây là *controlled crossover injection* nhằm cô lập mối liên hệ train-evaluation; mỗi cloned record phải có `original_row_id` và `clone_id`. Đây không phải luồng triển khai hợp lệ và phải gọi là deliberate data contamination.
- **Control for training size:** ở q=0 hoặc paired comparator, bổ sung số row tương đương bằng cách nhân bản **chỉ các rows gốc trong T**, không chạm V, để phân biệt ảnh hưởng training-set size/oversampling với leakage. Giữ class-specific cloning counts nếu có thể, không tạo class prevalence confound lớn.
- **Same V:** Train each model trên T_contaminated vs T_size_control, evaluate **cùng V**, với threshold/config fixed; report paired deltas. H được chấm riêng sau freeze; H không chứa clone.
- **Exposure measurement:** % evaluation rows whose source signature occurs in train, n clones, class-specific contamination rates, training size/class ratio, exact overlap checksum. Có thể set q stratified-by-Class để giữ số hàng fraud selected có kiểm soát; chọn seed trước.
- **Important:** q được định nghĩa theo **tỷ lệ V rows đã được leak**, không phải tỷ lệ bản sao trong T; nếu một V row có nhiều clones, report thêm clone multiplicity.
- **Threat:** synthetic cloned train examples có thể overweight class / làm shifted training distribution. Size-matched and class-matched T-only duplicate control giảm nhưng không loại hết confound. D3 khác D1 ở chỗ chủ động can thiệp và đo dose-response.
- **Pass test:** V/H unmodified, clones from V allowed **chỉ trong contaminated DEV branch**; q thực tế và provenance kiểm chứng được; không sao chép bất kỳ row nào từ H.

## 6. Cross-scenario contamination matrix

| Source of illegitimate information | Scenario | What may cross boundary | What NEVER crosses |
|---|---|---|---|
| Validation X statistics | P1 | X_V → scaler training state | X_H / y_H |
| Validation source/synthetic neighborhoods | P3 | X_DEV/y_DEV → SMOTE before DEV split | H samples/labels |
| Validation labels | P6 | y_V → selected feature set | y_H |
| Post-outcome label-derived proxy | F2 | y_DEV → synthetic proxy in retrospective DEV evaluation | y_H → fake clean proxy |
| Outcome/delay information | F3 | retrospective synthetic post-event flag | H outcome used as feature |
| Validation label-derived categorical rate | F4 | y_V → mapping for V encoding | y_H → mapping |
| Same-X group crossover | D1 | same-X DEV group in T' and V' | H group overlap |
| Exact copies of evaluation DEV rows | D3 | V source → clone added to T contaminated | H source / copy |

## 7. Reporting, statistical analysis & validity review

**Report structure by scenario:** scenario definition → threat model/info flow → data split & affected populations → control/contaminated → model settings → metrics with counts/prevalence → paired or descriptive delta appropriately labelled → diagnostics/provenance → interpretation → threats/limitations → prevention rule.

**Required visuals (only once data available):** metrics by branch/model/seed with SD; class prevalence table; PR curves and confusion matrices from **eligible comparable eval**; selected feature overlap P6; mapping distributions F4; duplicate overlap and dose-response D1/D3. Không dùng đồ thị như bằng chứng nhân quả nếu split khác population.

**Threats to validity registry:**
1. **Data validity:** source CSV chưa audit, PCA anonymity, exact duplicates có thể là giao dịch độc lập cùng X.
2. **Construct validity:** F2/F3 hậu nghiệm nhân tạo; F4 feature mang label-derived aggregate với taxonomy giao thoa preprocessing.
3. **Internal validity:** changed class prevalence/size P3, group distribution D1, T size D3, changed selected predictor set P6, preprocessing/model interactions P1.
4. **External validity:** một dataset, ba thuật toán truyền thống, một khoảng thời gian, fraud ratio ~0.173%.
5. **Statistical conclusion:** 492 positive tổng thể, metric volatility, các seeds phụ thuộc nhau, nhiều kịch bản/metrics, threshold sensitivity.
6. **Researcher degrees of freedom:** không chọn threshold, seed, bin count, noise strength, q dựa trên contaminated score hoặc holdout.

**Interpretation rules:**
- Không kết luận 'leakage luôn làm tăng Accuracy' hoặc 'Random Forest luôn chống leak tốt nhất'.
- Không gộp `precision`, `AP`, `PR-AUC trapezoidal` thành một chỉ số.
- Không tuyên bố H là clean nếu đã được dùng để thiết kế bất kỳ treatment/pipeline nào.
- Không nói 'same model tested on H' với F2/F3 khi nó cần feature tạo từ ground truth H.
- Không báo significance nếu chưa xác định đơn vị thống kê và phép kiểm phù hợp.
- Báo cả kết quả null/negative như kết quả thật.

## 8. Decision gates, milestones và acceptance criteria

### Gate 0 — Scope & methodology approval (`OPEN`)
- Nhóm xác nhận 8 scenario IDs là phạm vi triển khai, hoặc đánh dấu scenario nào `DEFERRED`.
- Chốt tên đề tài (handoff cũ vs Overleaf), version report, định nghĩa 3 groups.
- Xác nhận rõ F2/F3 là hai design axes khác nhau, không đếm trùng.
- Xác nhận P3 là mixed-mechanism experimental demonstration, không claim pure causal leakage effect nếu chưa controlled.

### Gate 1 — Dataset audit (`TODO`)
- Có audit report và hash thực; exact duplicates X-only/(X,y), label conflicts, group distribution.
- Xác định khả thi D1 và group-aware stratified split; clean H đủ positive count để đánh giá.
- Nhóm cho phép dataset audit bằng code hoặc thực hiện thủ công theo quy trình duyệt; chưa tự triển khai.

### Gate 2 — Parameter freeze (`OPEN`)
- Confirm train/validation/holdout scheme, H seed, DEV seeds and number of repetitions.
- Confirm scaling columns, model parameters/class weight/SMOTE ratios and `k_neighbors`, P6 score+k, F2 sensitivity/FPR grid, F3 coverage+delay, F4 bins+smoothing, D3 q levels.
- Confirm threshold and metric definitions; source provenance requirements.
- Confirm leakage scenarios never touch H; rules for stopping/invalid metrics.

### Gate 3 — Implementation permission (`OPEN`)
- Nhóm/giảng viên (nếu yêu cầu) ký xác nhận Experimental Protocol.
- Codex chỉ được chuẩn bị code sau gate này; ghi changes ngoài protocol vào `implementation-notes.md` và `Decision Log`, yêu cầu duyệt trước khi đổi experimental design.

### Gate 4 — Results/reproducibility (`TODO, AFTER CODE`)
- All 3 models × 8 scenarios (nếu khả thi), complete metadata, run logs, figures, tables, no fabricated result.
- Control paths verify no evaluation labels/feature stats leak; group overlap zero H; seeds repeatable.
- Overleaf Methods/Results cập nhật nhất quán, dùng actual outputs và limitation.

## 9. Decision log

| ID | Ngày | Quyết định / proposal | Lý do / trade-off | Trạng thái |
|---|---|---|---|---|
| D-015 | 2026-11-21 | Thiết kế chi tiết 8 IDs: P1/P3/P6/F2/F3/F4/D1/D3 | Người dùng chọn các mục; triển khai phụ thuộc approval & audit | `CONFIRMED` về danh sách cần thiết kế, `PROPOSED` về phương pháp |
| D-016 | 2026-11-21 | Tách H group-disjoint, immutable, ~20% | Bảo vệ đánh giá cuối; class support và group constraint có thể lệch tỷ lệ | `PROPOSED` |
| D-017 | 2026-11-21 | Development splits lặp 5 seeds; giữ shared comparisons khi possible | Ước lượng độ biến thiên; tăng cost CPU | `PROPOSED` |
| D-018 | 2026-11-21 | P3 báo prevalence shift và synthetic crossover riêng | Tránh quy biến động metric cho leakage thuần túy | `PROPOSED` |
| D-019 | 2026-11-21 | F2/F3 không tự tạo post-event feature từ y_H | Feature không tồn tại tại prediction time; không làm bẩn holdout | `PROPOSED` |
| D-020 | 2026-11-21 | D3 dùng fixed V, q% V rows cloned vào T và training-size matched control | Isolate exact overlap, nhưng training class distribution vẫn cần kiểm soát | `PROPOSED` |

## 10. Research / Implementation Notes

**2026-11-21 | Phase: Research Design Refinement | Change:** Tạo standalone Experimental Protocol v1.0 từ handoff v2.0 và danh sách 8 scenario đã chọn. **Reason:** cần mô tả thực nghiệm và so sánh có thể tái lập trước khi code. **Alternatives:** chạy từng script trước để xem metric (rejected: outcome-driven design, leakage risk); gộp F2/F3 (conditional nếu không có time/coverage factor); loại D1 (conditional vào audit). **Trade-offs:** thêm rigor và fit-scope auditing nhưng tốn thời gian, CPU; F2/F3 synthetic, P3 confounded nếu đánh giá không matched. **Impacts:** giữ nguyên 3 groups, 3 models và RQ/H định hướng; chưa xác minh Research Gap/overleaf current status. **Approval:** pending. **No code or experiment was run.**

### Open questions cần nhóm xác nhận (không tự coi là quyết định)
1. Chốt nhóm nào duyệt thí nghiệm và deadline thực tế.
2. H tỷ lệ 20% và seed=2026 có phù hợp class/group counts sau audit không?
3. Có thể dùng 5 DEV seeds hay cần 3/10 do nguồn lực?
4. Scaler áp dụng `Time`,`Amount` hay tất cả 30 features?
5. SMOTE target ratios, neighbors; có làm matched/prevalence-adjusted secondary study không?
6. Feature selection score (`f_classif`) và số feature k nào được cố định?
7. F2 các sensitivity/FPR nào, F3 coverage/delay và có chồng lấp F2 không?
8. F4 chọn Amount bins hay Time bins, cách đặt bounds và smoothing?
9. D1 xử lý X-same/different-y thế nào nếu có; D3 chọn q và class-stratification ra sao?
10. Threshold chính 0.5 hay fixed threshold chọn trên training-internal validation?
11. Có được chạy dataset audit bằng Python **trước** khi duyệt toàn bộ protocol không? Đây là ngoại lệ cần xác nhận rõ nếu có.
12. Ai khóa/giữ H manifests và lúc nào mở nhãn H?

## 11. Reading list — nguồn định hướng (không tự coi Literature Review đã xong)

- Dataset ULB/Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- scikit-learn, Common pitfalls and recommended practices: https://scikit-learn.org/stable/common_pitfalls.html
- imbalanced-learn, Common pitfalls and recommended practices: https://imbalanced-learn.org/stable/common_pitfalls.html
- scikit-learn, TargetEncoder & internal cross-fitting: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- scikit-learn, TargetEncoder cross-fitting example: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html

> **Không phát minh paper/DOI/research gap.** Các link trên là tài liệu kỹ thuật để kiểm tra kỹ thuật triển khai; Literature Review cần tài liệu khoa học đọc thật.

## 12. Instructions to Codex

1. Read handoff v2.0 **and this protocol v1.0**; if discrepancy, present a difference report and ask for owner decision.
2. **Do not implement or execute code** until Gate 0–3 status is explicitly approved. Do not treat this document as a command to begin coding.
3. When reviewing, check information-flow, fit/transform boundaries, synthetic provenance, class prevalence and validity of estimands before proposing libraries or code.
4. Preserve scope: ULB dataset; 3 leakage groups; 3 classifiers; 8 selected scenarios; E1–E3 removed.
5. Never infer customer identity or feature business meaning from anonymized V1–V28.
6. All deviations require amendment to `implementation-notes.md`, protocol Decision Log, owner approval, and relevant tests once coding is allowed.
7. Do not invent audit outcomes, performance numbers, citations or clean-holdout results.
