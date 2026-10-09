# Kiến trúc mô hình và kỹ thuật triển khai

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**File này mô tả:** kiến trúc kỹ thuật cụ thể của từng module trong pipeline — thuật toán, công thức cài đặt, thư viện, siêu tham số mặc định. Dùng song song với `pipeline-thi-nghiem.md` (quy trình tổng thể) và `thiet-ke-thi-nghiem-chi-tiet.md` (ma trận thí nghiệm).

**Trạng thái:** khung thư mục trống cho các module dưới đây (`src/profiling/`, `src/pollution/`, ...) đã được tạo tại `data-quality-project/` trên máy (xem `cau-truc-thu-muc-project.md`), nhưng chưa có code thật — file này vẫn là tài liệu thiết kế để nhóm viết code theo.

---

## 0. Kiến trúc hệ thống tổng thể

```
┌───────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│   Beijing Air Quality | HIGGS (mẫu + full) | Covertype              │
└──────────────────────────────┬───────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                  ▼
┌─────────────────────┐                        ┌─────────────────────┐
│  PROFILING ENGINE     │                        │  POLLUTION ENGINE    │
│  (pandas / PySpark)   │◄───────────────────────│  (numpy-based)        │
│  5 công thức dimension│                        │  5 polluter tương ứng │
└──────────┬───────────┘                        └──────────────────────┘
           │ profile vector (4-5 số)
           ▼
┌─────────────────────┐        ┌─────────────────────┐
│  DOWNSTREAM ML LAYER  │───────▶│  EXPERIMENT RUNNER    │
│  scikit-learn models  │        │  (ghi log, lặp vòng)  │
└──────────────────────┘        └──────────┬───────────┘
                                            │ bảng (profile, ΔPerf)
                                            ▼
                        ┌───────────────────────────────────┐
                        │        META-MODEL LAYER              │
                        │  Gradient Boosting Regressor          │
                        │  + SHAP explainability                │
                        └──────────────┬────────────────────┘
                                       │
                        ┌───────────────┴────────────────┐
                        ▼                                  ▼
            ┌─────────────────────┐          ┌─────────────────────────┐
            │  BASELINE LAYER       │          │  EVALUATION LAYER         │
            │  Rule-based            │          │  MAE / R² / Spearman      │
            │  Weighted score         │          │  so sánh meta vs baseline │
            │  DQSOps-style (PCA)      │          └─────────────────────────┘
            └─────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│              SCALABILITY LAYER (nhánh riêng — Vai trò B)            │
│   PySpark DataFrame API chạy Profiling Engine trên HIGGS full        │
│   đo runtime/memory theo kích thước tăng dần                         │
└───────────────────────────────────────────────────────────────────┘
```

---

## 1. Profiling Engine

**Vai trò:** tính vector điểm chất lượng từ 1 dataset (công thức xem `thiet-ke-thi-nghiem-chi-tiet.md` mục 2).

**Cài đặt bằng pandas (vai trò A):**

| Dimension | Hàm pandas/numpy chính |
|---|---|
| Completeness | `df.isnull().mean()` theo từng cột, trung bình lại |
| Feature Accuracy | So sánh `df` với bản gốc lưu riêng: `np.abs(df[col] - clean[col]).mean() / clean[col].mean()` (numeric); `(df[col] != clean[col]).mean()` (categorical) |
| Target Accuracy | Tương tự Feature Accuracy, áp dụng riêng cột target |
| Uniqueness | `(df.duplicated(keep=False).sum())` để đếm bản ghi trùng, tính theo công thức chuẩn hóa |
| Class Balance | `df[target].value_counts()`, tính tổng chênh lệch cặp theo công thức |

**Cài đặt bằng PySpark (vai trò B, dùng cho Scalability study):**

| Dimension | Hàm PySpark tương ứng |
|---|---|
| Completeness | `df.select([count(when(col(c).isNull(), c)) for c in cols])` |
| Feature/Target Accuracy | Join dataframe polluted với dataframe gốc theo khóa dòng, dùng `withColumn` + hàm `abs()` để tính khoảng cách, `agg(avg())` |
| Uniqueness | `df.groupBy(df.columns).count()` rồi lọc `count > 1` |
| Class Balance | `df.groupBy(target).count()`, tính pairwise difference bằng Python sau khi `collect()` (số lớp nhỏ nên collect về driver an toàn) |

**Nguyên tắc thiết kế:** hai bản cài đặt (pandas và PySpark) phải cho cùng công thức toán học, chỉ khác engine thực thi — để kết quả profiling nhất quán giữa vai trò A và B, và để phần so sánh runtime ở bước Scalability là so sánh công bằng (cùng thuật toán, khác engine).

---

## 2. Pollution Engine

**Vai trò:** tạo phiên bản dữ liệu bị ô nhiễm có kiểm soát, theo tham số (dimension, mức độ λ, seed).

**Kỹ thuật cụ thể từng polluter:**

- **Completeness polluter:** chọn ngẫu nhiên `λ × n` dòng (có seed), gán placeholder ngoài domain cho feature được chỉ định. Dùng `np.random.RandomState(seed).choice(n, size=int(λ*n), replace=False)` để chọn index.
- **Feature/Target Accuracy polluter:**
  - Numeric: `df[col] += np.random.RandomState(seed).normal(0, λ, size=len(df)) * df[col].mean()`
  - Categorical: hoán đổi giá trị bằng cách lấy mẫu từ `df[col].unique()` loại trừ giá trị hiện tại
- **Uniqueness polluter:** nhân bản dòng theo hệ số ρ (suy ra từ λ qua công thức `Uniqueness = 1/ρ`), số lần nhân bản mỗi dòng lấy mẫu từ phân phối normal(mean=1, std=5), cắt tại ρ=5
- **Class Balance polluter:** sắp xếp lớp theo kích thước, loại bớt dòng ở các lớp lớn hơn trung vị theo công thức tuyến tính để đạt mức imbalance λ mong muốn, giữ seed cho việc chọn dòng bị loại

**Thiết kế module:** mỗi polluter là 1 hàm độc lập nhận `(df, level, seed) → df_polluted`, không phụ thuộc lẫn nhau — cho phép áp dụng tuần tự nhiều dimension nếu sau này mở rộng (hiện tại mỗi lượt chạy chỉ áp dụng 1 dimension tại 1 thời điểm, đúng thiết kế factorial đơn biến của Mohammed et al.).

---

## 3. Downstream ML Layer

Toàn bộ dùng **scikit-learn**, không cần deep learning (khác bài gốc có dùng MLP/TabNet — đồ án môn học thu gọn chỉ cần 2 thuật toán/tác vụ để so sánh họ linear vs tree-based).

| Tác vụ | Thuật toán | Thư viện | Siêu tham số mặc định đề xuất |
|---|---|---|---|
| Classification | `LogisticRegression` | `sklearn.linear_model` | `max_iter=1000`, giữ mặc định còn lại |
| Classification | `RandomForestClassifier` | `sklearn.ensemble` | `n_estimators=100`, `random_state=seed` |
| Regression | `Ridge` | `sklearn.linear_model` | `alpha=1.0` (mặc định) |
| Regression | `GradientBoostingRegressor` | `sklearn.ensemble` | `n_estimators=100`, `random_state=seed` |
| Clustering | `KMeans` | `sklearn.cluster` | `n_clusters` = số lớp thật của dataset (2 cho HIGGS nếu dùng, 7 cho Covertype), `random_state=seed` |
| Clustering | `GaussianMixture` | `sklearn.mixture` | `n_components` = số lớp thật, `random_state=seed` |

**Chi tiết cài đặt (Giai đoạn 4, xem `src/downstream/`):** siêu tham số nằm trong `configs/algorithms.yaml`; cả 6 thuật toán đi qua cùng một pipeline học trên tập train — điền ô thiếu bằng trung vị → chuẩn hóa z-score → mô hình. k-Means dùng `n_init=10`, GMM dùng `covariance_type='full'` (mặc định).

**Thước đo:** `f1_score(average='macro')` (classification), `r2_score` (regression), `adjusted_mutual_info_score` (clustering) — đều có sẵn trong `sklearn.metrics`.

---

## 4. Meta-model Layer — trọng tâm kỹ thuật của đề tài

**Mục tiêu:** học ánh xạ `(profile vector, task, algorithm) → ΔPerformance`.

**Kiến trúc:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`, có thể thay bằng `xgboost.XGBRegressor` hoặc `lightgbm.LGBMRegressor` nếu cài thêm — ba lựa chọn này tương đương về bản chất, khác nhau ở tốc độ huấn luyện).

**Vì sao chọn Gradient Boosting thay vì mô hình tuyến tính hay neural network:**
- Dữ liệu huấn luyện nhỏ (~312 dòng) — không đủ cho neural network học ổn định
- Có khả năng học quan hệ phi tuyến giữa profile và ΔPerformance (bài gốc Mohammed et al. cho thấy nhiều dimension có ảnh hưởng dạng ngưỡng/phi tuyến, không phải tuyến tính đơn giản)
- Hỗ trợ tốt SHAP (SHapley Additive exPlanations) để giải thích — đây là cách tạo ra "trọng số task-aware" tường minh, khác với PCA ẩn của DQSOps

**Đặc trưng đầu vào (feature vector):**
```
[profile_completeness, profile_feature_accuracy, profile_target_accuracy,
 profile_uniqueness, profile_class_balance,
 task_classification, task_regression, task_clustering,     # one-hot
 algo_family_linear, algo_family_tree]                        # one-hot
```

**Quy trình huấn luyện:**
1. Chia bảng kết quả (312 dòng) thành train/validation theo tỷ lệ 80/20, stratify theo `task` để mỗi tập đều có đủ 3 tác vụ
2. Huấn luyện Gradient Boosting Regressor dự đoán `delta_performance`
3. Đánh giá trên validation bằng MAE và R² (xem Bước 8 trong pipeline)
4. Dùng thư viện `shap` để tính SHAP value cho từng dimension, tách riêng theo từng `task` → ra bảng "dimension nào quan trọng nhất cho task nào" — đây chính là phần đóng góp "task-aware weighting" của đề tài

**Lưu ý về cỡ mẫu nhỏ:** với ~250 dòng train, cần giới hạn độ sâu cây (`max_depth=3`) và dùng `n_estimators` vừa phải (50-100) để tránh overfit; nên dùng k-fold cross-validation (vd 5-fold) thay vì 1 lần chia train/val duy nhất để đánh giá ổn định hơn.

---

## 5. Baseline Layer

### 5.1 Rule-based
**Kỹ thuật:** đặt ngưỡng cố định trên từng profile score (vd `completeness < 0.9` → "Bad"). Không dự đoán giá trị ΔPerformance liên tục mà chỉ phân loại Good/Bad — khi so sánh với meta-model (dự đoán số liên tục), cần rời rạc hóa `delta_performance` thực tế thành Good/Bad theo cùng ngưỡng để so sánh công bằng (dùng accuracy/F1 cho baseline này thay vì MAE/R²).

### 5.2 Weighted score
**Công thức:**
```
Q = (1/5) × Σ profile_i    (trọng số bằng nhau)
```
Sau đó tính tương quan tuyến tính (hồi quy đơn biến) giữa `Q` và `delta_performance` để ra một "công thức dự đoán" đơn giản — đây là baseline yếu nhất, dùng để chứng minh giá trị gia tăng của việc học trọng số.

### 5.3 DQSOps-style (PCA composite)
**Kỹ thuật (tái hiện đúng cách DQSOps làm):**
1. Chuẩn hóa z-score từng cột profile: `z_i = (profile_i - mean_i) / std_i`
2. Áp dụng PCA (`sklearn.decomposition.PCA`), lấy principal component đầu tiên (PC1) làm quality score tổng hợp
3. Dùng PC1 làm biến dự đoán duy nhất, hồi quy tuyến tính đơn giản để dự đoán `delta_performance`

Đây là baseline quan trọng nhất cần cài đúng, vì nó tái hiện chính xác phương pháp tổng hợp của bài báo DQSOps — điểm so sánh trực tiếp để chứng minh lợi ích của việc gắn nhãn với hiệu năng downstream thật (Gradient Boosting có nhãn thật) so với một quality score tự định nghĩa không tham chiếu downstream (PCA composite).

---

## 6. Scalability Layer (Spark)

**Kỹ thuật:** chạy lại đúng 5 công thức dimension (mục 1, cột PySpark) trên các mốc kích thước tăng dần của HIGGS full (1M, 3M, 6M, 11M dòng).

**Đo lường:**
- Runtime: dùng `time.time()` bao quanh lệnh `.collect()` hoặc `.count()` để buộc Spark thực thi (tránh lazy evaluation làm sai lệch phép đo)
- Memory: dùng Spark UI (`spark.sparkContext.statusTracker()`) hoặc theo dõi qua `/proc` nếu chạy local mode

**Cấu hình Spark đề xuất cho máy cá nhân/Colab:** local mode (`local[*]`), không cần cluster thật — đủ để minh họa xu hướng runtime tăng tuyến tính/sub-tuyến tính theo kích thước dữ liệu, đúng tinh thần "khả năng mở rộng" mà không cần hạ tầng cloud thật.

---

## 7. Bảng tổng hợp công nghệ/thư viện

| Thành phần | Thư viện | Vai trò |
|---|---|---|
| Xử lý dữ liệu (vai trò A) | pandas, numpy | Profiling, Pollution |
| Xử lý dữ liệu quy mô lớn (vai trò B) | pyspark | Profiling ở quy mô lớn |
| Downstream ML | scikit-learn | 6 thuật toán + metrics |
| Meta-model | scikit-learn (GradientBoostingRegressor), hoặc xgboost/lightgbm | Dự đoán ΔPerformance |
| Giải thích mô hình | shap | Trọng số task-aware |
| Baseline PCA | scikit-learn (PCA) | DQSOps-style baseline |
| Trực quan hóa | matplotlib | Đường cong suy giảm, biểu đồ scalability |
| Lưu trữ kết quả | pandas (to_parquet/to_csv) | Bảng kết quả thí nghiệm |

**Cần bổ sung vào `requirements.txt`:** `pyspark`, `shap`, và tùy chọn `xgboost`/`lightgbm` (việc này đã được xác nhận trong `pipeline-thi-nghiem.md`, chưa thực thi).

---

## 8. Giới hạn kỹ thuật cần lưu ý khi trình bày

- Gradient Boosting huấn luyện trên ~312 dòng là một mẫu nhỏ cho một bài toán meta-learning — kết quả cần đi kèm cross-validation và khoảng tin cậy, tránh diễn giải quá chắc chắn.
- Spark chạy ở local mode trên máy cá nhân không phản ánh đầy đủ hành vi của cluster phân tán thật — phần kết luận nên nêu rõ đây là minh họa xu hướng khả năng mở rộng, không phải benchmark production.
- PCA composite (baseline DQSOps-style) giả định quan hệ tuyến tính giữa các profile score — đây chính là điểm yếu được chỉ ra trong phần so sánh, không phải lỗi cài đặt.
