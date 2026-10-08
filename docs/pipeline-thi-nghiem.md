# Pipeline thí nghiệm — Khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**Mục đích file này:** mô tả toàn bộ pipeline từ bước đầu đến bước cuối, để cả nhóm có thể triển khai song song theo từng module mà không giẫm chân nhau. Thiết kế thí nghiệm chi tiết (bảng tham số, công thức) và cấu trúc thư mục sẽ là hai file riêng, làm sau file này.

**Lưu ý quan trọng:** File này là bản thiết kế pipeline. **Khung thư mục trống đã được tạo** tại `data-quality-project/` trên máy (xem `cau-truc-thu-muc-project.md`), nhưng **chưa có dữ liệu nào được tải và chưa có code thật trong `src/`**. Việc tải 3 dataset và bổ sung thư viện vào `requirements.txt` đã được xác nhận về chủ trương (xem mục cuối file) — nhưng vẫn chưa thực hiện, chờ lệnh triển khai cụ thể.

---

## 0. Sơ đồ tổng thể

```
                        ┌─────────────────────────────┐
                        │   Giai đoạn 1: Chuẩn bị      │
                        │   (dataset + môi trường)     │
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                  ▼
   ┌─────────────────────┐                          ┌─────────────────────────┐
   │   VAI TRÒ A          │                          │   VAI TRÒ B              │
   │   (mô hình hóa)       │                          │   (scalability/Spark)    │
   │   Beijing Air Quality,│                          │   HIGGS full (11M dòng)  │
   │   HIGGS (mẫu), Covertype│                          │                         │
   └──────────┬───────────┘                          └──────────┬──────────────┘
              │                                                  │
   ┌──────────▼───────────┐                          ┌──────────▼──────────────┐
   │ 2. Module Profiling   │                          │ 9. Spark profiling      │
   │    (pandas/sklearn)   │                          │    scale-up study       │
   └──────────┬───────────┘                          └──────────────────────────┘
              │
   ┌──────────▼───────────┐
   │ 3. Module Pollution   │
   │    (4 mức × 4-5 dim)  │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │ 4. Module ML downstream│
   │    (2 thuật toán/tác vụ)│
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │ 5. Experiment Runner  │
   │    → bộ dữ liệu        │
   │    (profile, ΔPerf)   │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │ 6. Meta-model          │
   │    (profile → ΔPerf)  │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │ 7. Baselines           │
   │    (rule/weighted/     │
   │     DQSOps-style)      │
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │ 8. Đánh giá & so sánh  │
   └──────────┬───────────┘
              │
              └──────────────────┬───────────────────────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │ 10. Báo cáo & biểu đồ │
                      └─────────────────────┘
```

Hai nhánh A và B độc lập về mặt dữ liệu và có thể làm song song bởi hai thành viên khác nhau — chỉ gặp nhau ở bước báo cáo cuối.

---

## 1. Giai đoạn chuẩn bị (Dataset + Môi trường)

**Mục tiêu:** có đủ dữ liệu thô và môi trường chạy được trước khi code bất kỳ module nào.

| Việc cần làm | Vai trò | Ghi chú |
|---|---|---|
| Tải Beijing Multi-Site Air Quality (12 trạm, ~420K dòng) | A — regression | Mở rộng từ `AirPollution.csv` hiện có (1 trạm). Có cả numeric và 1 cột categorical (hướng gió) |
| Tải mẫu HIGGS (~500K-1M dòng, lấy ngẫu nhiên từ 11M) | A — classification | Thuần numeric, class gần cân bằng — dùng cho vòng lặp phát triển nhanh |
| Tải Covertype full (581K dòng) | A — clustering | Đa lớp (7 lớp), có categorical dạng one-hot sẵn |
| Tải HIGGS full (11M dòng, ~2.8GB) | B — scalability | Dùng chung dataset với vai trò A (classification) để tiết kiệm công chuẩn bị dữ liệu. Chỉ cần cho bước 9, không cần tải sớm |
| Bổ sung `pyspark` vào `requirements.txt` | Môi trường | Hiện file chỉ có pandas, numpy, matplotlib, scikit-learn, tensorflow |
| Cân nhắc thêm `xgboost` hoặc `lightgbm` | Môi trường | Dùng cho meta-model ở bước 6 |

**Đã được xác nhận triển khai** (xem mục cuối file) — chờ lệnh cụ thể để bắt đầu tải/cài đặt.

---

## 2. Module Profiling (tính điểm chất lượng)

**Mục tiêu:** từ một dataset (đã hoặc chưa bị làm bẩn), tính ra vector điểm chất lượng theo từng dimension.

**4-5 dimension chọn (dựa trên Mohammed et al., bỏ consistent representation vì công thức có thể tăng ngược khi ô nhiễm cao):**
- Completeness
- Feature Accuracy
- Target Accuracy
- Uniqueness
- (tùy chọn) Target Class Balance — chỉ áp dụng cho classification/clustering

**Input:** một DataFrame + bản sạch gốc để so sánh (ground truth)
**Output:** 1 vector điểm (4-5 số, mỗi số trong [0,1]) + file log

**Công cụ:** pandas/numpy cho vai trò A; phiên bản PySpark song song cho vai trò B (bước 9) — cùng công thức, khác engine thực thi.

---

## 3. Module Pollution (làm bẩn dữ liệu có kiểm soát)

**Mục tiêu:** tạo các phiên bản dữ liệu bị ô nhiễm ở từng mức độ, theo từng dimension, để vừa tính profile vừa đo tác động lên hiệu năng.

**Tham số (đã chốt):**
- 4 mức độ ô nhiễm: 0.0, 0.2, 0.5, 0.8
- Scenario duy nhất: làm bẩn cả train và test (Scenario 3 trong bài gốc — tình huống thực tế nhất)
- Polluter tái sử dụng định nghĩa toán học từ Mohammed et al. (missing value MCAR cho completeness, nhiễu Gaussian/đổi giá trị ngẫu nhiên cho feature/target accuracy, nhân bản có kiểm soát cho uniqueness)

**Input:** dataset sạch + dimension + mức độ
**Output:** dataset đã làm bẩn, có thể tái lập (cố định seed)

---

## 4. Module ML downstream (huấn luyện & đo hiệu năng)

**Mục tiêu:** huấn luyện mô hình trên dữ liệu (sạch hoặc đã làm bẩn) và đo hiệu năng.

| Tác vụ | Dataset | Thuật toán (đã chốt — 2/tác vụ) | Thước đo |
|---|---|---|---|
| Classification | HIGGS (mẫu) | Logistic Regression, Random Forest | F1-score |
| Regression | Beijing Air Quality | Ridge Regression, Gradient Boosting | R² |
| Clustering | Covertype | k-Means, Gaussian Mixture | Adjusted Mutual Information |

**Baseline sạch:** chạy 1 lần trên dữ liệu gốc không ô nhiễm để có mốc so sánh cho mỗi (dataset, thuật toán).

**Output của mỗi lượt chạy:** 1 giá trị hiệu năng, cộng với **ΔPerformance = Hiệu năng sạch − Hiệu năng đã làm bẩn** — đây chính là nhãn huấn luyện cho meta-model ở bước 6.

---

## 5. Experiment Runner (bộ điều phối)

**Mục tiêu:** chạy toàn bộ tổ hợp (dataset × dimension × mức độ × thuật toán × 3 lần lặp) một cách tự động, ghi kết quả thành 1 bảng dữ liệu duy nhất.

**Khối lượng ước tính (đã chốt tham số):**
- 4 dimension cốt lõi × 4 mức × 1 scenario × 2 thuật toán × 3 lần lặp = 96 lượt/tác vụ
- Classification và Clustering có thêm dimension Target Class Balance (tùy chọn): +24 lượt mỗi tác vụ → 120 lượt/tác vụ
- Regression: 96 lượt (không áp dụng Target Class Balance)

**Tổng chính xác (đã khớp với `thiet-ke-thi-nghiem-chi-tiet.md` mục 5): 96 (regression) + 120 (classification) + 96 (clustering) = 312 lượt chạy** cho cả 3 tác vụ — giảm đáng kể so với ước tính ban đầu (450-540), khả thi chạy trên máy cá nhân trong 1-2,5 giờ.

**Output:** 1 bảng (CSV/parquet) với các cột: `dataset, task, algorithm, dimension, pollution_level, run_id, [profile_vector 4-5 cột], raw_performance, delta_performance`. Đây là "bộ dữ liệu huấn luyện" cho bước 6.

---

## 6. Meta-model (mô hình dự đoán ΔPerformance)

**Mục tiêu:** học một mô hình ánh xạ `profile vector + đặc trưng tác vụ/thuật toán → ΔPerformance dự đoán`.

**Input đặc trưng:**
- Vector chất lượng (4-5 số từ bước 2)
- One-hot: loại tác vụ (classification/regression/clustering)
- One-hot: họ thuật toán (linear/tree-based)

**Mô hình thử nghiệm:** Gradient Boosting (sklearn hoặc XGBoost/LightGBM nếu cài thêm) — chọn vì có thể trích xuất feature importance / SHAP để suy ra "trọng số" từng dimension theo từng tác vụ, đúng với phần đóng góp task-aware đã chốt.

**Output:** mô hình đã huấn luyện + bảng trọng số/importance theo từng tác vụ.

---

## 7. Baselines (để so sánh)

| Baseline | Cách tính |
|---|---|
| Rule-based | Ngưỡng cố định trên từng dimension → nhãn Good/Bad, không dự đoán giá trị ΔPerformance liên tục |
| Weighted score | Trung bình có trọng số bằng nhau giữa các dimension, tương quan tuyến tính với hiệu năng |
| DQSOps-style | Tổng hợp profile vector bằng PCA thành 1 điểm duy nhất (theo đúng cách DQSOps làm), dùng điểm này làm proxy dự đoán |

Baseline DQSOps-style là baseline quan trọng nhất cần làm đúng, vì đây là điểm so sánh trực tiếp chứng minh lợi ích của việc gắn nhãn với hiệu năng downstream thật thay vì một quality score tự định nghĩa.

---

## 8. Đánh giá & so sánh

**Thước đo so sánh:**
- MAE, R² giữa ΔPerformance dự đoán và thực tế (cho meta-model và từng baseline)
- Hệ số tương quan Spearman (xếp hạng dataset/cấu hình theo mức độ suy giảm dự đoán vs thực tế)
- Phân tích theo từng tác vụ riêng (có baseline nào thắng ở tác vụ nào không)

**Output:** 1 bảng so sánh tổng hợp — chính là Bảng so sánh baseline đã phác thảo trong đề cương ban đầu.

---

## 9. Scalability study (Vai trò B — Spark)

**Mục tiêu:** chứng minh module profiling (bước 2) chạy được ở quy mô lớn bằng Spark.

**Thiết kế:**
- Lấy các mốc kích thước tăng dần từ HIGGS full: ví dụ 1M, 3M, 6M, 11M dòng (hoặc theo GB: ~0.25GB, 0.7GB, 1.4GB, 2.8GB)
- Chạy module profiling bằng PySpark trên từng mốc, đo runtime và memory
- Đối chiếu với phiên bản pandas (chỉ chạy được ở các mốc nhỏ) để thấy rõ điểm "gãy" — nơi pandas không còn khả thi nhưng Spark vẫn chạy ổn

**Không liên quan đến việc huấn luyện lại ML hàng trăm lần — chỉ đo tốc độ/bộ nhớ của bước profiling.**

**Output:** biểu đồ runtime/memory theo kích thước dữ liệu, cho cả 2 engine.

---

## 10. Báo cáo & biểu đồ

- Đường cong suy giảm hiệu năng theo mức độ ô nhiễm, theo từng dimension (giống Hình 6-19 trong bài gốc, nhưng với bộ dataset/thuật toán đã thu gọn)
- Bảng so sánh meta-model vs 3 baseline (từ bước 8)
- Biểu đồ trọng số/importance theo tác vụ (từ bước 6, thể hiện tính task-aware)
- Biểu đồ scalability (từ bước 9)
- Tổng hợp thành báo cáo đồ án

---

## Tóm tắt trình tự triển khai đề xuất

1. Xin phép tải dữ liệu, cập nhật `requirements.txt` (đã được đồng ý về chủ trương — chờ lệnh triển khai cụ thể)
2. Viết module Profiling (bước 2) — có thể test độc lập trên dataset nhỏ có sẵn
3. Viết module Pollution (bước 3) — test độc lập
4. Viết module ML downstream (bước 4) — test độc lập
5. Ghép thành Experiment Runner (bước 5), chạy thử ở quy mô nhỏ trước khi chạy full
6. Chạy full Experiment Runner → có bộ dữ liệu cho meta-model
7. Xây Meta-model + 3 Baseline (bước 6-7), song song với bước 9 (Spark scalability) nếu có người làm riêng
8. Đánh giá, so sánh, vẽ biểu đồ, viết báo cáo (bước 8-10)

---

## Các quyết định đã chốt

- [x] Bộ 3 dataset: Beijing Multi-Site Air Quality (regression), HIGGS (classification + scalability), Covertype (clustering)
- [x] Bổ sung `pyspark` (và có thể `xgboost`/`lightgbm`) vào `requirements.txt`
- [x] 2 thuật toán/tác vụ (tổng 6 thuật toán: LogR+RF, Ridge+GB, k-Means+GMM)
- [x] 4 mức độ ô nhiễm: 0.0, 0.2, 0.5, 0.8
- [x] Khối lượng thí nghiệm chính xác: 312 lượt chạy cho cả 3 tác vụ
- [x] Khung thư mục trống đã được tạo tại `data-quality-project/` trên máy (xem `cau-truc-thu-muc-project.md`)

Bước tiếp theo: viết code cho từng module trong `src/` theo đúng cấu trúc đã tạo, bắt đầu từ `configs/` và `src/profiling/`. Việc tải dataset thật vẫn cần một lệnh xác nhận riêng trước khi thực hiện.
