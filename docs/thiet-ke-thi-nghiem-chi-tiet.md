# Thiết kế thí nghiệm chi tiết

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**File này dựa trên:** `pipeline-thi-nghiem.md` (các quyết định đã chốt) và `tong-quan-bai-goc.md` (định nghĩa dimension/polluter từ Mohammed et al.)

**Phạm vi:** bản thiết kế — bảng tham số, công thức, ma trận thí nghiệm, quy trình từng lượt chạy. **Khung thư mục trống đã được tạo** tại `data-quality-project/` trên máy (xem `cau-truc-thu-muc-project.md`), nhưng chưa thực thi, chưa tải dữ liệu, chưa có code thật trong `src/`.

---

## 1. Ba dataset và cách chuẩn bị

### 1.1 Beijing Multi-Site Air Quality (Regression)
- **Nguồn:** UCI, 12 trạm × 35.064 giờ ≈ 420.768 dòng
- **Target:** PM2.5 (numeric, liên tục)
- **Feature:** numeric (PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, month, day, hour) + categorical (`wd` — hướng gió, `station`). *Tên cột theo dữ liệu thực tế đã tải; cột `No`, `year` bị bỏ.*
- **Chuẩn bị trước khi thí nghiệm:**
  - Gộp 12 file trạm thành 1 bảng, thêm cột `station` làm feature categorical
  - Dataset gốc đã có missing value tự nhiên (như bài gốc Mohammed et al. từng gặp với Houses/IMDB) → cần tạo **bản "baseline" sạch** bằng cách loại bỏ hoặc impute các dòng thiếu trước khi dùng làm điểm xuất phát cho polluter, để không nhầm lẫn giữa missing tự nhiên và missing do mình chủ động tiêm vào
  - One-hot encode `wd` và `station` trước khi đưa vào mô hình

### 1.2 HIGGS, bản mẫu (Classification)
- **Nguồn:** UCI, lấy mẫu ngẫu nhiên ~500.000-1.000.000 dòng từ 11 triệu dòng gốc (stratified theo target để giữ tỷ lệ lớp)
- **Target:** nhị phân (signal/background)
- **Feature:** 28 cột, thuần numeric → không cần polluter cho dimension liên quan categorical
- **Chuẩn bị:** không cần impute vì HIGGS không có missing value gốc; cột numeric đã chuẩn hóa sẵn theo thang riêng của bài toán vật lý hạt, giữ nguyên

### 1.3 Covertype, full size (Clustering)
- **Nguồn:** UCI, 581.012 dòng
- **Target (dùng để đánh giá AMI, không dùng để train vì clustering là unsupervised):** 7 lớp loại rừng
- **Feature:** numeric (elevation, slope...) + categorical dạng one-hot sẵn (wilderness area, soil type)
- **Chuẩn bị:** không cần gộp thêm, dùng trực tiếp

### 1.4 HIGGS, full size (chỉ dùng cho Scalability study — Bước 9, không nằm trong ma trận thí nghiệm chính ở mục 3)
- 11.000.000 dòng, ~2,8GB — chỉ dùng để đo runtime/memory của module Profiling bằng Spark, không train lại ML.

---

## 2. Công thức 4-5 dimension và cách polluter áp dụng cho từng dataset

Tái sử dụng định nghĩa toán học từ Mohammed et al. (2025). Dưới đây là công thức rút gọn và lưu ý khi áp dụng lên từng dataset.

### 2.1 Completeness
**Công thức:**
```
Completeness(d) = 1 - (1/f) × Σ missing(c_i)
```
trong đó `missing(c_i)` là tỷ lệ giá trị thiếu trên feature `c_i`, `f` là số feature (không tính target).

**Polluter:** chèn missing value kiểu MCAR (Missing Completely At Random) theo tỷ lệ λ, dùng placeholder nằm ngoài domain của feature (ví dụ -1 cho cột tuổi/elevation dương, "unknown" cho categorical).

**Lưu ý theo dataset:**
- Beijing: áp dụng trên cả numeric (TEMP, PRES...) và categorical (wd)
- HIGGS: chỉ numeric, placeholder cần chọn giá trị nằm ngoài khoảng chuẩn hóa gốc
- Covertype: numeric + categorical one-hot (missing trên one-hot nghĩa là set toàn bộ nhóm one-hot của 1 dòng về 0, biểu diễn "không rõ loại đất")

### 2.2 Feature Accuracy
**Công thức (numeric):**
```
nFAcc(c) = 1 - avg_dist(c) / mean_gt(c)
```
**Công thức (categorical):**
```
cFAcc(c) = 1 - mismatches(c) / n
```

**Polluter:**
- Numeric: cộng nhiễu Gaussian `noise(c) = X × mean_gt(c)`, X ~ N(0, λ²)
- Categorical: đổi ngẫu nhiên sang giá trị khác trong domain, tỷ lệ λ số dòng bị đổi

**Lưu ý theo dataset:**
- HIGGS: 100% numeric nên toàn bộ polluter chỉ cần nhánh Gaussian noise, đơn giản hóa cài đặt
- Covertype: cột one-hot (soil type, wilderness area) xử lý như categorical — đổi sang 1 nhóm one-hot khác hoàn toàn (set đúng 1 bit = 1)

### 2.3 Target Accuracy
Giống công thức Feature Accuracy nhưng áp dụng riêng cho cột target — mô phỏng label noise.

**Lưu ý:**
- Beijing (regression): nhiễu Gaussian lên giá trị PM2.5
- HIGGS (classification nhị phân): flip nhãn ngẫu nhiên theo tỷ lệ λ
- Covertype: **không áp dụng dimension này cho clustering** vì clustering không dùng target để train — chỉ dùng target để tính AMI sau khi cluster xong. Giữ target sạch khi đánh giá.

### 2.4 Uniqueness
**Công thức:**
```
Uniqueness(d) = (unique_samples(d) - 1) / (n - 1)
```
**Polluter:** nhân bản ngẫu nhiên các dòng theo hệ số ρ, phân phối số lần nhân bản theo normal (mean=1, std=5, cắt ở ρ=5 để giữ thời gian chạy hợp lý, đúng như bài gốc).

**Lưu ý theo dataset:** áp dụng giống nhau cho cả 3 dataset, không có khác biệt đặc thù.

### 2.5 Target Class Balance (tùy chọn — chỉ Classification và Clustering)
**Công thức:**
```
Balance(d) = 1 - ImBalance(d) / ε
```
**Polluter:** loại bớt dòng ở lớp thiểu số theo tỷ lệ λ để tạo mất cân bằng có kiểm soát, giữ thứ tự lớp cố định để tái lập được.

**Lưu ý:** HIGGS là nhị phân nên dễ áp dụng. Covertype 7 lớp — cần chọn trước 1-2 lớp nhỏ nhất để giảm, tránh làm sập toàn bộ kích thước dataset khi λ cao.

---

## 3. Ma trận thí nghiệm đầy đủ (đã chốt tham số)

**Tham số chung:**
- Mức độ ô nhiễm: λ ∈ {0.0, 0.2, 0.5, 0.8} (4 mức)
- Scenario: chỉ Scenario 3 — làm bẩn cả train và test cùng mức λ
- Số lần lặp: 3 (seed cố định 42, 43, 44)
- Split: stratified 80/20 (classification/regression theo target đã rời rạc hóa nếu cần; clustering không cần split vì không có train/test riêng — làm bẩn toàn bộ dataset)

| Tác vụ | Dataset | Dimension áp dụng | Thuật toán | Số tổ hợp (dim × mức × thuật toán × run) |
|---|---|---|---|---|
| Classification | HIGGS (mẫu) | Completeness, Feature Accuracy, Target Accuracy, Uniqueness, Class Balance (5 dim) | LogR, RF | 5 × 4 × 2 × 3 = **120** |
| Regression | Beijing Air Quality | Completeness, Feature Accuracy, Target Accuracy, Uniqueness (4 dim, không có Class Balance) | Ridge, GB | 4 × 4 × 2 × 3 = **96** |
| Clustering | Covertype | Completeness, Feature Accuracy, Uniqueness, Class Balance (4 dim, không có Target Accuracy) | k-Means, GMM | 4 × 4 × 2 × 3 = **96** |

**Tổng: 120 + 96 + 96 = 312 lượt chạy ô nhiễm**, cộng thêm **18 lượt baseline sạch** (3 tác vụ × 2 thuật toán × 3 lần lặp, chạy ở λ=0 riêng để làm mốc so sánh, không lặp lại trong bảng trên vì λ=0.0 đã nằm trong 4 mức).

→ **Tổng cộng khoảng 312 lượt chạy** (λ=0.0 đã được tính trong ma trận, không cộng thêm riêng) — khớp với ước tính ở file pipeline (~336, chênh lệch do Target Accuracy không áp dụng cho Clustering và Class Balance phân bổ khác giữa 3 tác vụ).

---

## 4. Quy trình 1 lượt chạy (pseudocode)

```
for task in [classification, regression, clustering]:
    for dataset in task.datasets:
        clean_train, clean_test = stratified_split(dataset, seed=base_seed)
        baseline_perf[task][algo] = train_and_evaluate(algo, clean_train, clean_test)  # chạy 1 lần/thuật toán

        for dimension in task.dimensions:
            for level in [0.0, 0.2, 0.5, 0.8]:
                for run in [1, 2, 3]:
                    seed = seed_table[run]
                    polluted_train = polluter[dimension](clean_train, level, seed)
                    polluted_test  = polluter[dimension](clean_test, level, seed)

                    profile_vector = compute_profile(polluted_train)  # 4-5 điểm chất lượng

                    for algo in task.algorithms:
                        perf = train_and_evaluate(algo, polluted_train, polluted_test)
                        delta_perf = baseline_perf[task][algo] - perf

                        log_row(dataset, task, algo, dimension, level, run,
                                profile_vector, perf, delta_perf)
```

**Lưu ý triển khai:**
- `compute_profile` tính trên **tập train đã làm bẩn** (vì đây là thứ một hệ thống thực tế quan sát được trước khi biết kết quả mô hình)
- Với Clustering, không có `train/test` tách biệt — polluted toàn bộ dataset, chạy thuật toán trực tiếp trên đó, tính AMI so với nhãn thật (nhãn thật không bị polluted)
- `seed_table` cố định trước (không random mỗi lần chạy) để đảm bảo tái lập được toàn bộ kết quả

---

## 5. Schema bảng kết quả (output của Experiment Runner)

| Cột | Kiểu | Ví dụ |
|---|---|---|
| `dataset` | string | "higgs_sample" |
| `task` | string | "classification" |
| `algorithm` | string | "random_forest" |
| `dimension` | string | "completeness" |
| `pollution_level` | float | 0.5 |
| `run_id` | int | 2 |
| `profile_completeness` | float | 0.52 |
| `profile_feature_accuracy` | float | 1.0 |
| `profile_target_accuracy` | float | 1.0 |
| `profile_uniqueness` | float | 1.0 |
| `profile_class_balance` | float | 0.91 |
| `raw_performance` | float | 0.71 (F1-score) |
| `baseline_performance` | float | 0.84 |
| `delta_performance` | float | 0.13 |

File lưu dạng `.parquet` (khuyến nghị, nhẹ hơn CSV khi số dòng lớn) hoặc `.csv` nếu nhóm quen dùng hơn — cả hai đều phù hợp với ~312 dòng kết quả.

---

## 6. Ước tính thời gian chạy

| Tác vụ | Thuật toán nặng nhất | Thời gian ước tính/lượt | Tổng thời gian (120/96/96 lượt) |
|---|---|---|---|
| Classification (HIGGS mẫu 500K-1M dòng) | Random Forest | ~10-30 giây | ~20-60 phút |
| Regression (Beijing, 420K dòng) | Gradient Boosting | ~5-15 giây | ~10-25 phút |
| Clustering (Covertype, 581K dòng) | Gaussian Mixture | ~15-40 giây | ~25-65 phút |

**Tổng ước tính: khoảng 1-2,5 giờ chạy tuần tự trên máy cá nhân** (CPU thông thường, không cần GPU). Có thể chạy song song theo tác vụ nếu máy đủ nhân CPU để rút ngắn còn ~1 giờ.

---

## 7. Checklist trước khi chạy thật

- [ ] Dataset đã tải và đặt đúng cấu trúc thư mục (file riêng, làm sau)
- [ ] Module Profiling (mục 2) đã test đơn vị trên dữ liệu mẫu nhỏ, đối chiếu tay 1-2 ví dụ
- [ ] Module Pollution đã test: polluted ở λ=0 phải cho profile ≈ dataset gốc (sai số do làm tròn)
- [ ] Baseline sạch đã chạy và lưu lại trước khi vào vòng lặp ô nhiễm
- [ ] `seed_table` đã cố định và ghi vào config, không hardcode rải rác trong code

---

## Nguồn công thức

- Mohammed, S. et al. (2025). *The Effects of Data Quality on Machine Learning Performance on Tabular Data*. Information Systems, 132, 102549 — mục 3 (định nghĩa dimension và polluter).
