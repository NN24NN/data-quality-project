# Kết quả thực nghiệm (tổng hợp cho báo cáo)

**Mục đích file này:** gom **số liệu đã thực sự chạy được** của từng giai đoạn vào một chỗ, để viết báo cáo và làm slide. Mỗi kết quả ghi rõ chạy ở đâu (cell notebook), commit nào, ngày nào. Cách làm và lý do của từng quyết định nằm ở `cach-trien-khai-chi-tiet-tung-giai-doan.md`; điểm lệch so với bài gốc nằm ở `khac-biet-so-voi-bai-goc.md`.

**Quy ước:** chỉ ghi số đã chạy và kiểm chứng. Phần chưa chạy để trống, có ghi chú *Chưa có*. Số thập phân dùng dấu phẩy.

| Giai đoạn | Trạng thái kết quả |
|---|---|
| 1 — Dữ liệu | Có |
| 2 — Profiling | Có |
| 3 — Pollution | Có |
| 4 — Downstream (baseline sạch) | Có |
| 5 — Experiment Runner (312 lượt) | Có đủ: test và chạy thử (5.1), regression (5.2), clustering (5.3), classification (5.4), tổng kết (5.5) |
| 6 — Meta-model + 3 baseline | Có: test, MAE xem nhanh, trọng số PC1, SHAP theo tác vụ (6.1–6.5) |
| 7 — Đánh giá & so sánh | Có: bảng chính (7.1), bootstrap (7.2), bảng phụ 288 (7.3), Good/Bad (7.4), chẩn đoán SHAP (7.5), tổng kết (7.6) |
| 8 — Scalability (Spark) | Có: chuẩn bị dữ liệu (8.1), lần 1 (8.2, bộ nhớ không dùng), **lần 2 — bảng chính (8.3)**, tổng kết (8.4) |

---

## 1. Dữ liệu (Giai đoạn 1)

*Nguồn: notebook cell 7.1–7.3, 8.1–8.2, chạy trên Colab.*

| Dataset | Tác vụ | Bản thô | Bản sạch (`data/processed/`) | Ghi chú |
|---|---|---|---|---|
| Beijing Multi-Site Air Quality (UCI) | Regression, target `PM2.5` | 12 trạm × 35.064 dòng × 18 cột, gộp được 420.768 dòng | **382.168 × 42** | Bỏ 38.600 dòng có missing (9,2%); one-hot `wd`, `station`; NaN = 0, trùng = 0 |
| HIGGS (mẫu) | Classification nhị phân | 11.000.000 × 29 (UCI) | **1.000.000 × 29** | Lấy mẫu stratified, seed 42; nhãn 52,99% / 47,01%; 2.281 dòng trùng (0,23%), giữ nguyên |
| Covertype (UCI, qua scikit-learn) | Clustering, 7 lớp | 581.012 × 55 | **581.012 × 55** | Lớp nhỏ nhất 2.747 dòng, lớn nhất 283.301 dòng; trùng = 0 |

Số ô thiếu gốc của Beijing nhiều nhất ở `CO` (20.701), `O3` (13.277), `NO2` (12.116), `SO2` (9.021), `PM2.5` (8.739).

---

## 2. Profiling (Giai đoạn 2)

*Nguồn: cell 9.2, 9.3; commit `43c6ba3`.*

- Test: `12 passed`.
- Bản pandas và bản PySpark khớp nhau trên mẫu 50.000 dòng của cả 3 dataset, sai số tương đối ≤ 1e-9.

**Profile của bản sạch** (toàn bộ dữ liệu, pandas). Đây là điểm xuất phát của mọi lượt làm bẩn:

| Dataset | n | Completeness | Feature Acc. | Target Acc. | Uniqueness | Class Balance |
|---|---|---|---|---|---|---|
| Beijing | 382.168 | 1 | 1 | 1 | 1 | NaN (regression) |
| HIGGS mẫu | 1.000.000 | 1 | 1 | 1 | 0,997719 | 0,887077 |
| Covertype | 581.012 | 1 | 1 | 1 | 1 | 0,255949 |

Ba số khác 1 đã đối chiếu bằng tính tay: Uniqueness HIGGS = (997.719 − 1) / 999.999; Class Balance HIGGS = 0,47008 / 0,52992; Class Balance Covertype = 1 − 2.529.486 / 3.399.612.

---

## 3. Pollution (Giai đoạn 3)

*Nguồn: cell 10.1, 10.2; commit `bbd1721`; mẫu 100.000 dòng mỗi dataset, seed của run 1.*

- Test: `43 passed` (31 pollution + 12 profiling).

**Điểm của dimension bị làm bẩn, đo lại bằng profiler** (λ = 0 / 0,2 / 0,5 / 0,8):

| Dimension | HIGGS mẫu | Beijing | Covertype | Lý thuyết |
|---|---|---|---|---|
| Completeness | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | 1 − λ |
| Feature Accuracy | 1 / 0,6433 / 0,4360 / 0,2866 | 1 / 0,7216 / 0,4680 / 0,2433 | 1 / 0,7216 / 0,4679 / 0,2431 | Cột số: 1 − √λ·√(2/π); có one-hot thì lấy trung bình với 1 − λ |
| Target Accuracy | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,6425 / 0,4347 / 0,2850 | không áp dụng | Phân loại: 1 − λ; target số: như cột số |
| Uniqueness | 0,9998 / 0,7998 / 0,4999 / 0,2000 | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | ≈ 1 − λ |
| Class Balance | 0,8885 / 0,7108 / 0,4443 / 0,1777 | không áp dụng | **0,2541 / 0,3033 / 0,3771 / 0,4508** (đi ngược) | 2 lớp: baseline × (1 − λ) |

**Số dòng sau khi làm bẩn:**
- Uniqueness: 100.000 → 125.000 / 200.000 / 500.000.
- Class Balance: HIGGS → 90.590 / 76.476 / 62.361; Covertype → 89.732 / 74.330 / 58.929.

**Tác dụng phụ chéo** (λ = 0 / 0,2 / 0,5 / 0,8):

| Polluter | Cột bị kéo theo | Số đo |
|---|---|---|
| Completeness | Uniqueness | HIGGS 0,9998 / 1 / 1 / 0,9959; Beijing 1 / 1 / 1 / 0,9476; Covertype 1 / 1 / 0,9993 / **0,7800** |
| Target Accuracy (HIGGS, đổi nhãn) | Class Balance | 0,8885 / 0,9321 / 0,9975 / 0,9309 |

**Phát hiện để đưa vào báo cáo:** với công thức Class Balance của bài gốc, polluter làm điểm Balance của Covertype (7 lớp) **tăng** thay vì giảm. Đã quyết định giữ 24 lượt này, đến Giai đoạn 7 mới quyết định có đưa vào bảng so sánh không.

---

## 4. Downstream — baseline sạch (Giai đoạn 4)

*Nguồn: cell 11.1–11.5. Bản cuối ở commit `87d3f20`, ngày 2026-10-09; lưu tại `results/baseline_performance.csv` trên Drive.*

- Test: `93 passed`.

**Baseline cuối cùng** (toàn bộ dữ liệu; train/test 80/20 stratified; 3 seed):

| Dataset | Thuật toán | Thước đo | Run 1 / 2 / 3 | Trung bình ± độ lệch chuẩn | Thời gian/lượt |
|---|---|---|---|---|---|
| HIGGS mẫu | Logistic Regression | F1-macro | 0,6347 / 0,6347 / 0,6347 | 0,6347 ± 0 | 7,7 giây |
| HIGGS mẫu | Random Forest (`max_samples` 0,1) | F1-macro | 0,7250 / 0,7246 / 0,7249 | 0,7249 ± 0,0002 | 122 giây |
| Beijing | Ridge | R² | 0,8580 / 0,8580 / 0,8580 | 0,8580 ± 0 | 2,5 giây |
| Beijing | Gradient Boosting | R² | 0,9157 / 0,9157 / 0,9157 | 0,9157 ± 0 | 112 giây |
| Covertype | k-Means | AMI | 0,1846 / 0,1529 / 0,1794 | 0,1723 ± 0,0170 | 36 giây |
| Covertype | GMM (`diag`, `n_init` 10) | AMI | 0,2016 / 0,1894 / 0,1974 | 0,1961 ± 0,0062 | 121 giây |

**Thí nghiệm phụ dùng để chốt cấu hình:**

| Thí nghiệm | Kết quả | Quyết định |
|---|---|---|
| Random Forest không lấy mẫu con (cell 11.4, HIGGS sạch, 1 seed) | F1 0,7328 so với 0,7250; 851 giây so với 129 giây | Dùng `max_samples: 0.1`: F1 chỉ thấp hơn 0,008 mà nhanh gấp 6,6 lần |
| Độ ổn định GMM theo seed (cell 11.5, 5 seed) | Độ lệch chuẩn AMI: `full`/1 lần khởi tạo 0,045; `diag`/1 lần 0,042; `diag`/5 lần 0,020; `diag`/10 lần 0,010 | Dùng `diag`, `n_init: 10` |

Baseline trên mẫu 100.000 dòng (cell 11.2, run 1) dùng để đo thời gian: F1 0,6297 / 0,7197; R² 0,8475 / 0,9090; AMI 0,1604 / 0,1735.

---

## 5. Experiment Runner — 312 lượt (Giai đoạn 5)

### 5.1 Test và chạy thử (đã có)

*Nguồn: cell 12.1 và 12.2 trên Colab, commit `8f264e7`, ngày 2026-10-23.*

- Cell 12.1: `100 passed, 2 warnings in 55,61s` (7 test runner + 93 test cũ). Hai warning là `PerformanceWarning` của pandas trong code test, đã sửa bằng `sort_index()`; không ảnh hưởng kết quả.
- Cell 12.2: regression × completeness × λ ∈ {0; 0,5} × run 1, 3,0 phút cho 4 dòng:

| λ | Thuật toán | profile_completeness | raw (R²) | baseline | Δ |
|---|---|---|---|---|---|
| 0 | Ridge | 1,0 | 0,8580 | 0,8580 | +0,0000 |
| 0 | Gradient Boosting | 1,0 | 0,9157 | 0,9157 | −0,0000 |
| 0,5 | Ridge | 0,5 | 0,5797 | 0,8580 | +0,2783 |
| 0,5 | Gradient Boosting | 0,5 | 0,6677 | 0,9157 | +0,2480 |

- Δ = 0 ở λ = 0 khớp kỳ vọng (cùng dữ liệu, seed, cấu hình với baseline). Ở λ = 0,5 chỉ điểm Completeness đổi (0,5), bốn điểm profile còn lại giữ nguyên 1,0.
- Thời gian: lượt λ = 0 mất 102 giây (gồm đọc dữ liệu và huấn luyện hai thuật toán), lượt λ = 0,5 mất 77 giây (gồm làm bẩn và profile).

### 5.2 Regression — 96 dòng (đã có)

*Nguồn: cell 12.3 (`TASK = 'regression'`) và 12.4 trên Colab, commit `7760252`, ngày 2026-10-23. Tổng thời gian **140,3 phút**.*

- Kiểm tra file (12.4): 96 / 96 dòng regression; trùng khóa 0; NaN chỉ ở `profile_class_balance` (96 dòng, đúng vì regression không áp dụng Class Balance); max |Δ| ở λ = 0 là 1,1e-16.

**ΔR² trung bình ± độ lệch chuẩn qua 3 run** (Δ = baseline − raw, dương nghĩa là hiệu năng giảm; baseline R²: Ridge 0,8580, Gradient Boosting 0,9157):

| Dimension | Thuật toán | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | Ridge | 0,1089 ± 0,0013 | 0,2844 ± 0,0058 | 0,5461 ± 0,0022 |
| Completeness | Gradient Boosting | 0,0870 ± 0,0011 | 0,2522 ± 0,0037 | 0,5433 ± 0,0021 |
| Feature Accuracy | Ridge | 0,1030 ± 0,0008 | 0,2021 ± 0,0011 | 0,2732 ± 0,0011 |
| Feature Accuracy | Gradient Boosting | 0,1129 ± 0,0016 | 0,2044 ± 0,0028 | 0,2674 ± 0,0023 |
| Target Accuracy | Ridge | 0,1405 ± 0,0026 | 0,2816 ± 0,0034 | 0,3761 ± 0,0035 |
| Target Accuracy | Gradient Boosting | 0,1508 ± 0,0021 | 0,3014 ± 0,0028 | 0,4023 ± 0,0028 |
| Uniqueness | Ridge | 0,0040 ± 0,0026 | 0,0013 ± 0,0040 | −0,0010 ± 0,0009 |
| Uniqueness | Gradient Boosting | 0,0010 ± 0,0011 | 0,0016 ± 0,0017 | 0,0002 ± 0,0006 |

**Nhận xét để đưa vào báo cáo:**
- Ba dimension Completeness, Feature Accuracy, Target Accuracy đều làm Δ tăng đơn điệu theo λ, độ lệch chuẩn giữa 3 run nhỏ (≤ 0,006), nên nhãn Δ cho meta-model ổn định.
- **Uniqueness gần như không ảnh hưởng hiệu năng regression** (|Δ| ≤ 0,007, cùng cỡ độ dao động giữa các run, có lúc âm) dù điểm Uniqueness đo được giảm xuống 0,2. Nhân bản dòng ở cả train lẫn test không làm mô hình tệ đi đáng kể. Meta-model sẽ thấy Uniqueness gần như không có sức dự đoán ở tác vụ này; kỳ vọng SHAP của Uniqueness thấp ở regression.
- Completeness ở λ = 0,8 làm hiệu năng giảm mạnh nhất (Δ ≈ 0,54, R² còn 0,31–0,37). Feature Accuracy bão hòa dần (Δ 0,20 → 0,27 từ λ = 0,5 lên 0,8). Target Accuracy làm Gradient Boosting giảm nhiều hơn Ridge, ngược với Completeness.

**Thời gian chạy thật mỗi đơn vị (3 run, cả hai thuật toán):**

| Dimension | λ = 0 | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | 101–103 giây | 92–94 | 73–74 | 47–50 |
| Feature Accuracy | chép (0 giây) | 421–431 | 423–425 | 420–426 |
| Target Accuracy | chép | 100–102 | 99–103 | 98–101 |
| Uniqueness | chép | 127–129 | 212–217 | 569–576 |

- Completeness nhanh dần khi λ tăng (nhiều ô điền trung vị nên cột gần hằng số, Gradient Boosting có ít ngưỡng chia hơn).
- Feature Accuracy chậm gấp 4 so với baseline (~425 giây so với ~101 giây) và không đổi theo λ. *Giải thích (suy đoán, chưa kiểm chứng):* nhiễu Gaussian làm mọi giá trị trong cột số trở nên duy nhất, nên Gradient Boosting phải thử nhiều ngưỡng chia hơn.
- Uniqueness tăng theo kích thước dữ liệu (gấp 1,25 / 2 / 5,6 lần thời gian baseline).

### 5.3 Clustering — 96 dòng (đã có)

*Nguồn: cell 12.3 (`TASK = 'clustering'`) và 12.4 trên Colab, ngày 2026-10-23. Tổng thời gian **126,4 phút**. Hiện file có 192/312 dòng.*

- Kiểm tra file (12.4): 96 / 96 dòng clustering; trùng khóa 0; NaN chỉ ở `profile_class_balance` (96 dòng, thuộc regression); max |Δ| ở λ = 0 là 1,1e-16.

**ΔAMI trung bình ± độ lệch chuẩn qua 3 run** (Δ = baseline − raw; baseline AMI theo từng seed: k-Means 0,1846 / 0,1529 / 0,1794, GMM 0,2016 / 0,1894 / 0,1974):

| Dimension | Thuật toán | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | k-Means | 0,0245 ± 0,0165 | 0,0716 ± 0,0407 | 0,1304 ± 0,0311 |
| Completeness | GMM | 0,0423 ± 0,0090 | 0,1077 ± 0,0178 | 0,1425 ± 0,0143 |
| Feature Accuracy | k-Means | 0,0953 ± 0,0069 | 0,1574 ± 0,0130 | 0,1712 ± 0,0165 |
| Feature Accuracy | GMM | 0,1111 ± 0,0107 | 0,1657 ± 0,0096 | 0,1934 ± 0,0055 |
| Uniqueness | k-Means | −0,0021 ± 0,0231 | 0,0089 ± 0,0214 | 0,0128 ± 0,0086 |
| Uniqueness | GMM | 0,0072 ± 0,0032 | 0,0220 ± 0,0056 | 0,0214 ± 0,0089 |
| **Class Balance (24 lượt đang xét)** | k-Means | 0,0233 ± 0,0132 | 0,0510 ± 0,0142 | 0,0962 ± 0,0140 |
| **Class Balance (24 lượt đang xét)** | GMM | 0,0492 ± 0,0005 | 0,0581 ± 0,0106 | 0,1038 ± 0,0172 |

**Nhận xét để đưa vào báo cáo:**
- **24 lượt Class Balance × clustering cho ΔAMI rõ rệt và tăng đều theo λ** (GMM 0,049 → 0,058 → 0,104; k-Means 0,023 → 0,051 → 0,096), không gần 0. Trong khi đó điểm Class Balance đo được ở Giai đoạn 3 lại **tăng** khi λ tăng (0,254 → 0,451). Tức là điểm Balance tăng nhưng hiệu năng giảm. Đây là căn cứ cho quyết định ở Giai đoạn 7: ba baseline coi điểm cao là dữ liệu tốt nên sẽ dự đoán ngược chiều ở 24 lượt này. *Chưa kiểm tra cột `profile_class_balance` trong file kết quả; sẽ đối chiếu khi vẽ biểu đồ.*
- Feature Accuracy làm AMI sụp gần về 0 ở λ = 0,5–0,8 (raw 0,001–0,038), nên ΔAMI bị chặn trên bởi chính baseline (~0,17–0,20): đường cong bão hòa vì không thể giảm dưới 0.
- **Uniqueness gần như không ảnh hưởng** (|Δ| ≤ 0,022), giống regression. Với k-Means, giá trị trung bình thậm chí âm ở λ = 0,2 (−0,002 ± 0,023).
- **k-Means nhiễu hơn GMM** do seed: độ lệch chuẩn lên tới 0,041 (Completeness λ = 0,5) so với 0,018 của GMM, cùng cỡ độ dao động baseline (± 0,017). Một số lượt k-Means có Δ âm khi Uniqueness (ví dụ run 2: −0,029). ΔAMI của k-Means vì thế nhiễu hơn ΔF1/ΔR²; ghép baseline theo seed đã giảm nhưng không loại hết phần này.

**Thời gian chạy thật mỗi đơn vị (3 run, cả hai thuật toán):**

| Dimension | λ = 0 | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | 152–177 giây | 151–180 | 109–172 | 119–155 |
| Feature Accuracy | chép (0 giây) | 129–156 | 91–103 | 80–84 |
| Uniqueness | chép | 176–202 | 280–314 | 742–870 |
| Class Balance | chép | 132–148 | 90–113 | 81–87 |

- Tổng 126,4 phút, sát ước tính (2,2–3 giờ). Uniqueness λ = 0,8 chậm nhất (khoảng 5 lần baseline) và chiếm gần 40% thời gian cả tác vụ.
- Không có hiện tượng chậm gấp 4 như Feature Accuracy của regression: k-Means và GMM không dựa vào sắp xếp giá trị. Ngược lại, Feature Accuracy ở λ cao còn nhanh hơn baseline.

### 5.4 Classification — 120 dòng (đã có)

*Nguồn: cell 12.3 (`TASK = 'classification'`) và 12.4 trên Colab, ngày 2026-10-23. Tổng thời gian **147,8 phút**.*

- Kiểm tra file (12.4): 120 / 120 dòng classification; **tổng 312 / 312**; trùng khóa 0; NaN chỉ ở `profile_class_balance` (96 dòng, thuộc regression); max |Δ| ở λ = 0 là 1,1e-16.

**ΔF1-macro trung bình ± độ lệch chuẩn qua 3 run** (Δ = baseline − raw; baseline F1: Logistic Regression 0,6347, Random Forest 0,7250 / 0,7246 / 0,7249 theo seed):

| Dimension | Thuật toán | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | Logistic Regression | 0,0377 ± 0,0002 | 0,0783 ± 0,0007 | 0,1444 ± 0,0017 |
| Completeness | Random Forest | 0,0372 ± 0,0012 | 0,0874 ± 0,0008 | 0,1550 ± 0,0004 |
| Feature Accuracy | Logistic Regression | 0,0657 ± 0,0016 | 0,0870 ± 0,0020 | 0,0995 ± 0,0022 |
| Feature Accuracy | Random Forest | 0,1294 ± 0,0009 | 0,1617 ± 0,0004 | 0,1781 ± 0,0006 |
| Target Accuracy | Logistic Regression | 0,0588 ± 0,0010 | **0,1385 ± 0,0019** | 0,0581 ± 0,0010 |
| Target Accuracy | Random Forest | 0,0980 ± 0,0003 | **0,2263 ± 0,0011** | 0,0984 ± 0,0007 |
| Uniqueness | Logistic Regression | −0,0001 ± 0,0010 | 0,0007 ± 0,0016 | 0,0004 ± 0,0008 |
| Uniqueness | Random Forest | −0,0004 ± 0,0024 | −0,0006 ± 0,0010 | −0,0056 ± 0,0009 |
| Class Balance | Logistic Regression | 0,0178 ± 0,0006 | 0,0733 ± 0,0006 | 0,1328 ± 0,0010 |
| Class Balance | Random Forest | 0,0065 ± 0,0005 | 0,0472 ± 0,0007 | 0,1640 ± 0,0009 |

**Nhận xét để đưa vào báo cáo:**
- **Target Accuracy không đơn điệu theo λ:** Δ đạt đỉnh ở λ = 0,5 (Random Forest 0,226) rồi **giảm** ở λ = 0,8 về gần bằng λ = 0,2 (0,058 so với 0,059 ở Logistic Regression; 0,098 so với 0,098 ở Random Forest). Giải thích: HIGGS là nhị phân, đổi nhãn ở 80% dòng (cả train lẫn test, Scenario 3) gần như đảo ngược nhãn một cách nhất quán, nên mô hình học lại được quy luật ngược. Ở λ = 0,5 nhãn là nhiễu thuần túy nên tệ nhất. Hệ quả: điểm Target Accuracy 0,8 (λ = 0,2) và 0,2 (λ = 0,8) cho cùng mức giảm hiệu năng, tức quan hệ giữa điểm này và ΔF1 là hình chữ U ngược. *Giải thích này nhất quán với dữ liệu nhưng chưa kiểm chứng riêng.* Mô hình tuyến tính (baseline PCA, weighted score) khó nắm quan hệ này; meta-model phi tuyến thì có thể học được.
- **Random Forest nhạy với Feature Accuracy hơn Logistic Regression** (ΔF1 0,13–0,18 so với 0,07–0,10), và nhạy hơn với Class Balance ở λ cao (0,164 so với 0,133).
- **Uniqueness lại gần như không ảnh hưởng** (|Δ| ≤ 0,006; Random Forest thậm chí F1 tăng nhẹ ở λ = 0,8). Cả ba tác vụ cho cùng kết luận: nhân bản dòng ở cả train lẫn test hầu như không làm giảm hiệu năng.
- **Class Balance (HIGGS, 2 lớp):** điểm Balance giảm theo λ (Giai đoạn 3) và ΔF1 tăng theo λ, cùng chiều như kỳ vọng; trái với clustering (Covertype) nơi điểm Balance tăng nhưng hiệu năng vẫn giảm.
- Độ lệch chuẩn giữa 3 seed rất nhỏ (≤ 0,0024), nên nhãn ΔF1 ổn định hơn nhiều so với ΔAMI của k-Means.

**Thời gian chạy thật mỗi đơn vị (3 run, cả hai thuật toán):**

| Dimension | λ = 0 | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|---|
| Completeness | 126–130 giây | 123–127 | 112–114 | 112–114 |
| Feature Accuracy | chép (0 giây) | 182–185 | 191–193 | 191–196 |
| Target Accuracy | chép | 132–138 | **208–210** | 135–142 |
| Uniqueness | chép | 160–164 | 261–266 | 688–726 |
| Class Balance | chép | 115–117 | 96–102 | 79 |

- Target Accuracy ở λ = 0,5 chậm hơn hẳn (khoảng 209 giây so với 135 giây ở hai mức còn lại): nhãn nhiễu thuần túy làm Random Forest mọc cây sâu hơn. Khớp với dự đoán trước khi chạy.

### 5.5 Tổng kết Giai đoạn 5

- **312 / 312 dòng** trong `results/experiment_results.parquet` (Drive), 14 cột đúng schema. Tổng thời gian chạy **414,5 phút ≈ 6,9 giờ** (regression 140,3 + clustering 126,4 + classification 147,8 phút), nằm trong khoảng ước tính (6,5–7,5 giờ).
- **Năm phát hiện đáng đưa vào báo cáo:**
  1. **Đánh giá chất lượng dữ liệu phải theo ngữ cảnh tác vụ** (luận điểm chính của đề tài). Cùng một dimension, cùng mức λ, tác động rất khác nhau giữa các tác vụ và thuật toán. Ví dụ Completeness ở λ = 0,8: ΔR² ≈ 0,54 (regression), ΔF1 ≈ 0,15 (classification), ΔAMI ≈ 0,13 (clustering). Cùng tác vụ cũng khác nhau theo thuật toán: Feature Accuracy làm Random Forest giảm 0,13–0,18 F1, Logistic Regression chỉ 0,07–0,10.
  2. **Uniqueness gần như không làm giảm hiệu năng ở cả 3 tác vụ** (|Δ| ≤ 0,022, thường ≤ 0,007), dù điểm Uniqueness đo được giảm xuống 0,2 (nhân bản dòng ở cả train lẫn test). Hệ quả: một điểm chất lượng tổng hợp có trọng số bằng nhau cho mọi dimension (weighted score) sẽ cảnh báo oan dimension này; meta-model nhiều khả năng cho Uniqueness trọng số SHAP rất thấp.
  3. **Quan hệ giữa điểm profile và Δ không phải lúc nào cũng đơn điệu hay cùng chiều.** (a) Target Accuracy trên HIGGS có dạng chữ U ngược: điểm 0,8 và điểm 0,2 cho cùng mức giảm F1 (nhãn bị đảo nhất quán ở cả train và test, mô hình học lại được). (b) Class Balance trên Covertype: điểm Balance tăng (0,254 → 0,451) nhưng AMI giảm (ΔAMI 0,02–0,10). Cả hai trường hợp đều thách thức baseline giả định "điểm cao = dữ liệu tốt" và quan hệ tuyến tính (PCA composite của DQSOps), và là chỗ meta-model phi tuyến có thể thể hiện ưu thế.
  4. **Hiệu ứng bão hòa (floor).** Feature Accuracy làm AMI của clustering sụp gần về 0 ở λ ≥ 0,5 nên ΔAMI bị chặn bởi baseline (~0,17–0,20); Feature Accuracy của regression cũng tăng chậm dần (ΔR² 0,20 → 0,27 từ λ = 0,5 lên 0,8). Δ tuyệt đối vì thế phụ thuộc mạnh vào mức baseline của từng tác vụ, nên khi gộp ba tác vụ cần cân nhắc dùng Δ tương đối (Δ / baseline) bên cạnh Δ tuyệt đối.
  5. **Độ ổn định của nhãn ΔPerformance khác nhau theo tác vụ:** classification rất ổn định giữa 3 seed (độ lệch chuẩn ≤ 0,0024), regression ổn định (≤ 0,006), nhưng ΔAMI của k-Means nhiễu (độ lệch chuẩn tới 0,041, có lượt Δ âm). Ghép baseline theo seed (mục 4.2 của `khac-biet-so-voi-bai-goc.md`) đã giảm nhưng không loại hết phần nhiễu này; tác vụ clustering sẽ là tác vụ khó dự đoán nhất cho meta-model. Cần nêu trong phần giới hạn.
- **Tổng thời gian và quy mô:** 312 dòng, 414,5 phút trên Colab CPU. Thuật toán chậm nhất là Gradient Boosting với Feature Accuracy (nhiễu làm mọi giá trị trong cột số trở nên duy nhất, chậm gấp 4 baseline) và Uniqueness λ = 0,8 (dữ liệu lớn gấp 5, chậm gấp 5–7 lần).- **Còn mở cho Giai đoạn 7:** có đưa 24 lượt Class Balance × clustering vào bảng so sánh chính không (ΔAMI 0,02–0,10, rõ rệt, nhưng điểm Balance tăng thay vì giảm).
## 6. Meta-model và 3 baseline (Giai đoạn 6)

*Nguồn: notebook cell 13.1–13.3, Colab CPU, commit `e47cda9`, ngày 2026-10-26. Thiết kế: `khac-biet-so-voi-bai-goc.md` mục 4.3–4.5; tham số: `configs/metamodel.yaml`.*

### 6.1 Cấu hình đã chạy

| Thành phần | Giá trị |
|---|---|
| Meta-model | `GradientBoostingRegressor`, 100 cây, sâu 3, learning rate 0,1, không tinh chỉnh; 14 đặc trưng (5 profile + 3 task + 6 thuật toán) |
| Nhãn | Δ tuyệt đối (chính), Δ tương đối = Δ / baseline (phụ) |
| Chia chính | `StratifiedGroupKFold` 5 fold, 42 nhóm (task, dimension, λ), lặp 3 seed (42, 43, 44) |
| Chia phụ | Nội suy: train λ ∈ {0; 0,2; 0,8} (234 dòng), test λ = 0,5 (78 dòng), 1 lần chia |
| Rule-based | Bad nếu bất kỳ dimension < 0,9 |

### 6.2 Kiểm chứng

- Cell 13.1: **`116 passed`** trong 60,8 giây (100 test cũ + 16 test mới). Có 2 `PerformanceWarning` của pandas trong `test_shap_importance_task_aware` (tra chỉ mục nhiều tầng chưa sắp xếp), đã sửa bằng `sort_index()` sau lần chạy này.
- Cell 13.2: 312 dòng đầu vào, **8.112 dòng dự đoán** (đúng bằng 312 × 3 seed × 2 nhãn × 4 phương pháp + 78 × 2 × 4), chạy trong 5 giây.

### 6.3 MAE out-of-fold (xem nhanh — số liệu chính, kèm R² và Spearman, tính ở Giai đoạn 7)

Trung bình qua các lần lặp; rule-based chỉ cho nhãn Good/Bad nên không có MAE. **In đậm** = thấp nhất trong cột.

**Group K-Fold (đánh giá chính):**

| Nhãn | Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| Tuyệt đối | Meta-model | **0,0387** | **0,0374** | **0,0931** | **0,0550** |
| | Weighted score | 0,0452 | 0,0717 | 0,1210 | 0,0767 |
| | DQSOps-PCA | 0,0734 | 0,0550 | 0,1272 | 0,0843 |
| Tương đối | Meta-model | 0,0822 | **0,1869** | **0,0981** | **0,1193** |
| | Weighted score | **0,0639** | 0,2656 | 0,1129 | 0,1410 |
| | DQSOps-PCA | 0,1222 | 0,2878 | 0,1736 | 0,1890 |

**Nội suy (test λ = 0,5, 78 dòng, 1 lần chia):**

| Nhãn | Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| Tuyệt đối | Meta-model | 0,0500 | 0,0541 | **0,0970** | **0,0657** |
| | Weighted score | 0,0516 | 0,0538 | 0,1492 | 0,0823 |
| | DQSOps-PCA | **0,0371** | **0,0456** | 0,1337 | 0,0694 |
| Tương đối | Meta-model | 0,1247 | **0,1640** | **0,0576** | **0,1162** |
| | Weighted score | **0,0860** | 0,2717 | 0,1240 | 0,1548 |
| | DQSOps-PCA | 0,1019 | 0,3093 | 0,1601 | 0,1836 |

**Nhận xét sơ bộ (chờ R² và Spearman ở Giai đoạn 7 mới kết luận):**
- **Đánh giá chính, Δ tuyệt đối:** meta-model có MAE thấp nhất ở **cả 3 tác vụ** và tổng thể. MAE tổng thể là 0,055, thấp hơn weighted score 28% và DQSOps-PCA 35%.
- **Không thắng tuyệt đối:** với Δ tương đối, ở classification weighted score tốt hơn (0,064 so với 0,082). Ở thí nghiệm nội suy Δ tuyệt đối, DQSOps-PCA tốt hơn ở classification và clustering. Đây là kết quả trung thực, nên báo cáo đúng như vậy, không chỉ báo số gộp. Riêng tập nội suy chỉ có 78 dòng và 1 lần chia, không có độ lệch chuẩn, nên chênh lệch nhỏ (vd 0,050 so với 0,037) chưa chắc có ý nghĩa.
- **Lợi thế của meta-model lớn nhất ở clustering và regression**, đúng nơi ảnh hưởng của dimension phụ thuộc mạnh vào tác vụ: clustering Δ tuyệt đối 0,037 so với 0,055–0,072.
- Regression có MAE lớn nhất ở mọi phương pháp vì Δ của regression có biên độ lớn nhất (tới ~0,54).

### 6.4 Trọng số PC1 của DQSOps-PCA (fit trên cả 312 dòng — dữ liệu cho H9)

| Dimension | Trọng số PC1 |
|---|---|
| Completeness | 0,227 |
| Feature Accuracy | −0,001 |
| Target Accuracy | **−0,704** |
| Uniqueness | 0,314 |
| Class Balance | **0,595** |

PC1 chỉ giải thích **26,8%** phương sai, gần mức 20% của 5 cột độc lập.

**Nhận xét (điểm đáng đưa vào báo cáo):** điểm tổng hợp kiểu DQSOps **không phản ánh chất lượng dữ liệu** trên bộ thí nghiệm này, vì ba lẽ:
- Feature Accuracy là dimension gây hại mạnh ở cả 3 tác vụ nhưng có trọng số ≈ 0.
- Target Accuracy mang trọng số âm, tức điểm Target Accuracy giảm (dữ liệu xấu đi) lại làm "điểm chất lượng" tăng.
- Hai trọng số lớn nhất rơi vào Class Balance và Target Accuracy, hai cột có mức nền khác nhau theo tác vụ: Class Balance là 1,0 (điền cho regression), 0,89 (HIGGS), 0,25 (Covertype); Target Accuracy luôn là 1 ở clustering. PC1 vì thế chủ yếu nắm "dòng này thuộc tác vụ nào" chứ không nắm "dữ liệu bẩn đến đâu".

Nguyên nhân gốc: mỗi lượt chỉ làm bẩn một dimension, nên các dimension gần như không tương quan và không có một "trục chất lượng chung" cho PCA bắt được. Đây là bằng chứng cụ thể cho giới hạn đã nêu ở `khac-biet-so-voi-bai-goc.md` mục 4.5, và cho luận điểm "trọng số cố định, không gắn với downstream" của đề tài.

### 6.5 Tỷ trọng SHAP theo tác vụ (meta-model cuối, fit trên cả 312 dòng — dữ liệu cho H8)

Tỷ trọng = trung bình |SHAP| của dimension / tổng của các dimension áp dụng được cho tác vụ đó; "—" = không áp dụng.

| Dimension | Classification (tuyệt đối / tương đối) | Clustering (tuyệt đối / tương đối) | Regression (tuyệt đối / tương đối) |
|---|---|---|---|
| Completeness | 0,242 / 0,250 | 0,323 / 0,276 | **0,359 / 0,364** |
| Feature Accuracy | **0,274 / 0,321** | **0,435 / 0,467** | 0,260 / 0,331 |
| Target Accuracy | 0,256 / 0,162 | — | 0,327 / 0,265 |
| Uniqueness | 0,107 / 0,064 | 0,040 / 0,039 | 0,055 / 0,040 |
| Class Balance | 0,121 / 0,203 | 0,203 / 0,218 | — |

**Nhận xét:**
- **Thứ tự quan trọng khác nhau theo tác vụ** (luận điểm task-aware). Đứng đầu là Feature Accuracy với classification và clustering, nhưng Completeness với regression. Feature Accuracy chiếm gần một nửa (0,44–0,47) ở clustering, khớp hiện tượng AMI sụp về ≈ 0 ở mục 5.5.
- **Uniqueness có tỷ trọng thấp nhất ở cả 3 tác vụ** (0,04–0,11), khớp phát hiện 2 ở mục 5.5 (Uniqueness gần như không làm giảm hiệu năng). Ngược hẳn với DQSOps-PCA, nơi Uniqueness có trọng số dương lớn thứ ba (0,314).
- Hai thang nhãn cho cùng thứ tự đứng đầu ở mỗi tác vụ, nên kết luận không phụ thuộc cách chuẩn hóa nhãn.
- **Cần xem lại ở Giai đoạn 7:** Uniqueness ở classification có tỷ trọng 0,107, cao hơn hai tác vụ kia dù ΔF1 của nó gần 0. Có thể vì điểm Uniqueness cũng dao động ở các lượt dimension khác (vd Completeness λ = 0,8 kéo Uniqueness xuống, mục 4 của `khac-biet`), nên mô hình gán một phần ảnh hưởng của Completeness cho Uniqueness. Chưa kiểm chứng.

## 7. Đánh giá và so sánh (Giai đoạn 7)

*Nguồn: notebook cell 14.1–14.3, Colab CPU, commit `755bf48`, ngày 2026-10-26. Thiết kế: `khac-biet-so-voi-bai-goc.md` mục 4.6. File trên Drive `results/`: `method_comparison.csv`, `method_bootstrap.csv`, `goodbad_comparison.csv`, `shap_by_row_dimension.csv`.*

- Cell 14.1: **`124 passed`** trong 65,1 giây (116 cũ + 8 mới).
- **Bảng chính tính trên 312 lượt**; bảng phụ 288 lượt (bỏ Class Balance × clustering, chỉ lọc phần đánh giá) ở mục 7.3.
- Số liệu dạng "trung bình ± độ lệch chuẩn qua 3 seed" (Group K-Fold). **In đậm** = tốt nhất trong cột.

### 7.1 Bảng so sánh chính — Group K-Fold, Δ tuyệt đối (dữ liệu cho H5)

| Chỉ số | Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| MAE ↓ | Meta-model | **0,039 ± 0,008** | **0,037 ± 0,009** | **0,093 ± 0,013** | **0,055 ± 0,006** |
| | Weighted score | 0,045 ± 0,003 | 0,072 ± 0,001 | 0,121 ± 0,003 | 0,077 ± 0,002 |
| | DQSOps-PCA | 0,073 ± 0,003 | 0,055 ± 0,004 | 0,127 ± 0,005 | 0,084 ± 0,003 |
| Spearman ↑ | Meta-model | **0,720 ± 0,081** | **0,616 ± 0,046** | **0,635 ± 0,085** | **0,660 ± 0,035** |
| | Weighted score | 0,508 ± 0,092 | 0,319 ± 0,059 | 0,607 ± 0,020 | 0,323 ± 0,042 |
| | DQSOps-PCA | −0,014 ± 0,068 | −0,053 ± 0,177 | 0,013 ± 0,095 | 0,058 ± 0,018 |
| R² ↑ | Meta-model | −0,235 ± 0,381 | −0,098 ± 1,068 | **0,453 ± 0,150** | **0,383 ± 0,122** |
| | Weighted score | **0,228 ± 0,052** | −0,900 ± 0,105 | 0,000 ± 0,002 | 0,071 ± 0,015 |
| | DQSOps-PCA | −0,698 ± 0,168 | **−0,028 ± 0,171** | −0,097 ± 0,040 | −0,029 ± 0,049 |

**Δ tương đối (bảng phụ về thang nhãn):**

| Chỉ số | Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| MAE ↓ | Meta-model | 0,082 ± 0,008 | **0,187 ± 0,016** | **0,098 ± 0,012** | **0,119 ± 0,003** |
| | Weighted score | **0,064 ± 0,006** | 0,266 ± 0,006 | 0,113 ± 0,004 | 0,141 ± 0,004 |
| | DQSOps-PCA | 0,122 ± 0,002 | 0,288 ± 0,006 | 0,174 ± 0,013 | 0,189 ± 0,006 |
| Spearman ↑ | Meta-model | **0,661 ± 0,116** | **0,466 ± 0,114** | **0,775 ± 0,031** | **0,586 ± 0,029** |
| | Weighted score | 0,563 ± 0,034 | 0,373 ± 0,029 | 0,609 ± 0,022 | 0,469 ± 0,021 |
| | DQSOps-PCA | −0,129 ± 0,190 | −0,522 ± 0,070 | −0,200 ± 0,296 | −0,158 ± 0,132 |
| R² ↑ | Meta-model | −0,900 ± 0,060 | **0,563 ± 0,079** | **0,513 ± 0,144** | **0,538 ± 0,052** |
| | Weighted score | **−0,053 ± 0,094** | 0,180 ± 0,024 | 0,429 ± 0,033 | 0,341 ± 0,024 |
| | DQSOps-PCA | −1,425 ± 0,046 | −0,168 ± 0,036 | −0,260 ± 0,179 | −0,073 ± 0,054 |

**Thí nghiệm nội suy (test λ = 0,5, 78 dòng, 1 lần chia)**, MAE tuyệt đối / Spearman tuyệt đối:

| Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|
| Meta-model | 0,050 / 0,726 | 0,054 / 0,715 | **0,097 / 0,830** | **0,066 / 0,839** |
| Weighted score | 0,052 / −0,219 | 0,054 / 0,674 | 0,149 / 0,435 | 0,082 / 0,179 |
| DQSOps-PCA | **0,037 / 0,927** | **0,046 / 0,803** | 0,134 / 0,684 | 0,069 / 0,715 |

Với Δ tương đối, meta-model tốt nhất ở regression (MAE 0,058, Spearman 0,927) và clustering (0,164 / 0,925), kém ở classification (0,125 / 0,143).

### 7.2 Bootstrap theo nhóm — hiệu MAE meta-model − baseline, CI 95% (âm = meta-model tốt hơn)

Group K-Fold, 312 lượt. **In đậm** = CI không chứa 0 (khác biệt có ý nghĩa ở mức 95%).

| Nhãn | So với | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| Tuyệt đối | DQSOps-PCA | **−0,035 [−0,059; −0,001]** | −0,018 [−0,035; +0,007] | −0,034 [−0,084; +0,002] | **−0,029 [−0,048; −0,011]** |
| | Weighted score | −0,007 [−0,032; +0,031] | −0,034 [−0,063; +0,005] | −0,028 [−0,095; +0,030] | −0,022 [−0,047; +0,005] |
| Tương đối | DQSOps-PCA | −0,040 [−0,103; +0,050] | **−0,101 [−0,226; −0,016]** | **−0,076 [−0,127; −0,030]** | **−0,070 [−0,119; −0,028]** |
| | Weighted score | +0,018 [−0,039; +0,068] | **−0,079 [−0,179; −0,006]** | −0,015 [−0,083; +0,029] | −0,022 [−0,063; +0,012] |

Thí nghiệm nội suy: khác biệt có ý nghĩa chỉ ở regression (tuyệt đối, so với DQSOps-PCA: −0,037 [−0,066; −0,000]) và ở Δ tương đối với clustering, regression (so với cả hai baseline ở clustering; so với DQSOps-PCA ở regression). Ở classification và clustering với Δ tuyệt đối, DQSOps-PCA nhỉnh hơn nhưng không có ý nghĩa (CI chứa 0).

### 7.3 Bảng phụ 288 lượt (bỏ 24 lượt Class Balance × clustering)

| Nhãn | Phương pháp | Classification | Clustering | Regression | Tất cả |
|---|---|---|---|---|---|
| Tuyệt đối | Meta-model | **0,039** | **0,038** | **0,093** | **0,057** |
| | Weighted score | 0,045 | 0,078 | 0,121 | 0,079 |
| | DQSOps-PCA | 0,073 | 0,061 | 0,127 | 0,088 |
| Tương đối | Meta-model | 0,082 | **0,178** | **0,098** | **0,111** |
| | Weighted score | **0,064** | 0,285 | 0,113 | 0,136 |
| | DQSOps-PCA | 0,122 | 0,323 | 0,174 | 0,190 |

(MAE, Group K-Fold.) Bootstrap trên 288 lượt cho cùng kết luận: meta-model hơn DQSOps-PCA có ý nghĩa ở cả hai nhãn (tuyệt đối −0,032 [−0,053; −0,011]); hơn weighted score không có ý nghĩa ở tổng thể. **Kết luận không phụ thuộc 24 lượt gây tranh cãi**, thậm chí khoảng cách meta-model − baseline ở clustering còn lớn hơn khi bỏ chúng.

### 7.4 Good/Bad — ngưỡng Δ tương đối 10%, Group K-Fold, 312 lượt (dữ liệu cho H7)

| Phương pháp | F1 lớp Bad | Balanced accuracy | Accuracy | Tỷ lệ đoán Bad | tp / fp / fn / tn (tất cả tác vụ) |
|---|---|---|---|---|---|
| Weighted score | **0,790** | **0,741** | **0,747** | 0,692 | 149 / 67 / 12 / 84 |
| Meta-model | 0,746 | 0,709 | 0,713 | 0,618 | 132 / 61 / 29 / 90 |
| Rule-based | 0,717 | 0,579 | 0,593 | 0,923 | 161 / 127 / 0 / 24 |
| DQSOps-PCA | 0,645 | 0,463 | 0,478 | 0,955 | 148 / 150 / 13 / 1 |

Tỷ lệ Bad thật: **51,6%** (regression 53,1%, clustering 61,5%, classification 42,5%).

Balanced accuracy theo tác vụ (classification / clustering / regression): weighted 0,840 / 0,500 / 0,778; meta-model 0,826 / 0,512 / 0,703; rule-based 0,500 / 0,500 / 0,767; DQSOps-PCA 0,478 / 0,500 / 0,402.

Độ nhạy theo ngưỡng (F1 lớp Bad, tất cả tác vụ):

| Phương pháp | 5% | 10% | 20% |
|---|---|---|---|
| Weighted score | **0,831** | **0,790** | 0,560 |
| Meta-model | 0,792 | 0,746 | **0,687** |
| Rule-based | 0,782 | 0,717 | 0,542 |
| DQSOps-PCA | 0,724 | 0,645 | 0,338 |

### 7.5 Chẩn đoán SHAP của Uniqueness ở classification

Trung bình |SHAP| (nhãn tuyệt đối), tách theo dimension đang bị làm bẩn của dòng:

| Dòng thuộc lượt | Completeness | Feature Acc. | Target Acc. | Uniqueness | Class Balance | Số dòng |
|---|---|---|---|---|---|---|
| λ = 0 | 0,0186 | 0,0215 | 0,0166 | 0,0132 | 0,0063 | 30 |
| Completeness | **0,0770** | 0,0156 | 0,0146 | 0,0043 | 0,0146 | 18 |
| Feature Accuracy | 0,0117 | **0,0841** | 0,0156 | 0,0050 | 0,0073 | 18 |
| Target Accuracy | 0,0132 | 0,0144 | **0,0862** | 0,0169 | 0,0056 | 18 |
| Uniqueness | 0,0194 | 0,0215 | 0,0166 | 0,0118 | 0,0065 | 18 |
| Class Balance | 0,0168 | 0,0203 | 0,0184 | 0,0146 | **0,0402** | 18 |

### 7.6 Tổng kết Giai đoạn 7 — kết luận cho báo cáo

1. **Meta-model ước lượng mức suy giảm tốt nhất.** Với Δ tuyệt đối (bảng chính), MAE thấp nhất và Spearman cao nhất ở **cả 3 tác vụ** (Spearman 0,62–0,72 so với 0,32–0,61 của weighted score). Với Δ tương đối, meta-model vẫn có Spearman cao nhất ở cả 3 tác vụ và MAE tốt nhất ở clustering, regression.
2. **Meta-model hơn DQSOps-PCA có ý nghĩa thống kê** ở tổng thể với cả hai nhãn (tuyệt đối −0,029 [−0,048; −0,011]). **Hơn weighted score nhưng chưa có ý nghĩa ở tổng thể** (−0,022 [−0,047; +0,005], p = 0,057); có ý nghĩa ở clustering (Δ tương đối). Với 42 nhóm, sức mạnh thống kê có hạn — cần nói đúng như vậy, không nói "vượt trội mọi baseline".
3. **DQSOps-PCA gần như không xếp hạng được mức suy giảm:** Spearman ≈ 0 với Δ tuyệt đối và âm với Δ tương đối (−0,52 ở clustering). Khớp phát hiện ở mục 6.4 (PC1 nắm "tác vụ nào" chứ không nắm "bẩn đến đâu"). Đây là bằng chứng mạnh nhất cho luận điểm "điểm chất lượng không gắn downstream thì không dự đoán được hiệu năng".
4. **Weighted score là baseline mạnh ở bài toán Good/Bad** (F1 0,790 so với 0,746 của meta-model ở ngưỡng 10%), nhưng meta-model bền hơn khi ngưỡng cao (20%: 0,687 so với 0,560). Diễn giải: điểm trung bình đủ để nhận ra "có dimension nào đó bị bẩn", nhưng không đủ để ước lượng **bẩn đến mức hại bao nhiêu** — phần đó cần biết tác vụ.
5. **Rule-based gần như luôn báo Bad** (92,3% số lượt; chỉ 24 lượt Good), vì Class Balance sạch của **cả HIGGS (0,887) lẫn Covertype (0,254)** đã dưới ngưỡng 0,9: balanced accuracy 0,500 ở classification và clustering. Luật cố định không thích nghi được với mức nền tự nhiên của từng dataset.
6. **R² trong từng tác vụ không ổn định** (meta-model clustering −0,098 ± 1,068): R² phạt nặng vài lỗi lớn khi cả một nhóm (vd λ = 0,8) bị giấu, trong khi phương sai Δ trong một tác vụ nhỏ. MAE và Spearman là thước đo chính đáng tin hơn; báo R² kèm giải thích này.
7. **Thí nghiệm nội suy cho tín hiệu lẫn lộn** (78 dòng, phần lớn CI chứa 0): meta-model tốt nhất ở regression và tổng thể; DQSOps-PCA nhỉnh hơn ở classification và clustering với Δ tuyệt đối nhưng không có ý nghĩa.
8. **Kết luận đứng vững trên 288 lượt** (mục 7.3).
9. **Chẩn đoán SHAP (mục 7.5) bác bỏ giả thuyết** "Uniqueness mượn ảnh hưởng của Completeness": |SHAP| của Uniqueness ở các dòng Completeness chỉ 0,0043, thấp nhất. Nó trải đều ở dòng λ = 0, Target Accuracy, Class Balance, Uniqueness (0,012–0,017). Diễn giải: điểm Uniqueness hầu như chỉ đổi ở lượt làm bẩn Uniqueness (nơi Δ ≈ 0), nên mô hình dùng nó như tín hiệu **loại trừ** — "điểm Uniqueness bình thường → đây không phải lượt vô hại → dự đoán Δ cao hơn trung bình". Tỷ trọng 0,107 vì thế không có nghĩa Uniqueness gây hại cho classification; báo cáo cần nêu rõ cách đọc này khi trình bày H8.
10. **Dự đoán sai của nhóm khi thiết kế:** đã đoán lớp Bad là thiểu số, thực tế 51,6% — accuracy vì thế không bị lệch mạnh, nhưng F1 lớp Bad và balanced accuracy vẫn là thước đo đúng (thấy rõ ở DQSOps-PCA: accuracy 0,478 nhưng gần như đoán Bad hết).

## 8. Scalability (Giai đoạn 8)

*Nguồn: notebook cell 15.2–15.3, Colab CPU tiêu chuẩn, commit `1054af0`, ngày 2026-10-26. Thiết kế: `khac-biet-so-voi-bai-goc.md` mục 4.7.*

### 8.1 Chuẩn bị dữ liệu (cell 15.2)

HIGGS full tải từ UCI, đổi sang parquet trên Drive: **11.000.000 dòng, 0,69 GB** (nhỏ hơn ước tính 1,2GB nhờ nén parquet). Ổ `/content` của Colab: 108GB, còn trống 88GB.

### 8.2 Lần chạy 1 (cell 15.3) — thời gian dùng được, bộ nhớ **không dùng được**

Trung vị 3 lần, mọi lần đều `ok`:

| Số dòng | Pandas (giây) | PySpark (giây) | PySpark chậm hơn | Đối chiếu điểm profile |
|---|---|---|---|---|
| 0,1M | 0,4 | 11,4 | 28,5 lần | khớp (sai số 0,0) |
| 1M | 3,8 | 40,2 | 10,6 lần | khớp (0,0) |
| 3M | 14,0 | 83,9 | 6,0 lần | khớp (0,0) |
| 6M | 35,7 | 155,2 | 4,3 lần | khớp (0,0) |
| 11M | 65,7 | 303,3 | 4,6 lần | khớp (0,0) |

**Bộ nhớ đỉnh của pandas đo được ở lần này bị sai** (2.021,6 / 2.021,6 / 2.247,0 / 4.141,9 / 7.314,8 MB): giá trị ở 0,1M và 1M giống hệt nhau. Nguyên nhân là lỗi trong code đo — `resource.getrusage(...).ru_maxrss` trên Linux giữ lại mức đỉnh của tiến trình cha (đã nạp ~2GB HIGGS để cắt mốc) khi tạo tiến trình con, nên mọi số đo có "sàn" ~2GB. Đã sửa (đọc `VmHWM`, cắt mốc trong tiến trình con riêng) và đo lại ở mục 8.3. Giá trị ở 6M và 11M cao hơn sàn nên có thể gần đúng, nhưng không dùng.

**Nhận xét lần 1:** (1) Pandas và PySpark cho điểm profile **giống hệt** ở mọi mốc đến 11M dòng — cùng công thức, khác engine, kiểm chứng ở quy mô lớn. (2) PySpark chậm hơn ở mọi mốc, nhưng khoảng cách thu hẹp từ 28 lần (0,1M) xuống ~4,5 lần (≥ 6M): đúng dự đoán về thời gian khởi động truy vấn của Spark. (3) **Pandas không hết RAM ở 11M dòng** trên máy Colab tiêu chuẩn, nên chưa đo được điểm gãy — người dùng chốt thêm mốc 22M (phương án B, `khac-biet-so-voi-bai-goc.md` mục 4.7 điểm 8).

### 8.3 Lần chạy 2 — đo lại pandas + mốc 22M (cell 15.4) — **bảng chính cho H10**

*Colab CPU tiêu chuẩn, commit `54a7837`, ngày 2026-10-26. Máy: **2 nhân CPU, 13,6 GB RAM**. Cell 15.1: **`134 passed`** (197,9 giây). Cắt 6 mốc mất 44 giây; đo lại pandas 7,8 phút, Spark mốc 22M 28,6 phút.*

Trung vị 3 lần. Pandas đo lại ở lần 2 với cách đo RAM đã sửa; Spark mốc 0,1M–11M giữ từ lần 1 (phiên Colab khác, cùng cấu hình máy), mốc 22M đo ở lần 2. Spark chạy với giới hạn `spark.driver.memory = 4g`, không đo bộ nhớ.

| Số dòng | Pandas (giây) | PySpark (giây) | PySpark / pandas | Đỉnh RAM pandas | Trạng thái |
|---|---|---|---|---|---|
| 0,1M | 0,3 | 11,4 | 38 lần | 0,20 GB | cả hai ok |
| 1M | 3,4 | 40,2 | 12 lần | 0,85 GB | cả hai ok |
| 3M | 14,2 | 83,9 | 5,9 lần | 2,24 GB | cả hai ok |
| 6M | 31,9 | 155,2 | 4,9 lần | 4,11 GB | cả hai ok |
| 11M | 75,0 | 303,3 | 4,0 lần | 7,32 GB | cả hai ok |
| **22M** (HIGGS ×2) | **— (`oom`)** | **526,6** | — | **vượt 13,6 GB** | **pandas hết RAM, Spark ok 3/3** |

Spark ở mốc 22M: 526,6 / 532,7 / 508,1 giây (ổn định). Đối chiếu điểm profile pandas và PySpark ở 5 mốc cả hai cùng chạy: khớp, sai số tương đối lớn nhất 1,1e-16 (bằng sai số làm tròn của số thực).

Ghi chú: thời gian pandas ở 11M lần này là 75,0 giây (lần 1: 65,7 giây) — dao động giữa các phiên Colab khoảng 10–15%; các lần trong cùng một phiên dao động ít hơn (63,9–75,9 giây).

### 8.4 Tổng kết Giai đoạn 8 — kết luận cho báo cáo

1. **Điểm gãy của pandas là số đo thật:** pandas profile được 11M dòng HIGGS (7,3 GB RAM, 54% RAM máy) nhưng **hết RAM ở 22M dòng** trên máy 13,6 GB; **PySpark chạy hết 22M dòng trong giới hạn 4 GB**, ổn định qua 3 lần (508–533 giây). Đây là bằng chứng chính cho đóng góp "profiling ở quy mô lớn".
2. **Bộ nhớ pandas tăng tuyến tính theo số dòng:** khoảng 0,65 GB mỗi triệu dòng (0,85 → 7,32 GB từ 1M đến 11M). Theo đường này, 22M dòng cần khoảng 14,4 GB — vượt 13,6 GB RAM, khớp với kết quả `oom`. Hệ số này phản ánh việc pandas giữ 2 bản dữ liệu (bản cần đo và ground truth) cùng các bảng trung gian khi tìm dòng trùng.
3. **PySpark chậm hơn pandas ở mọi mốc pandas còn chạy được**, nhưng khoảng cách thu hẹp đều: 38 lần (0,1M) → 12 → 5,9 → 4,9 → **4,0 lần (11M)**. Ở dữ liệu nhỏ, thời gian khởi động truy vấn của Spark chiếm gần hết thời gian chạy. Thời gian pandas tăng nhanh hơn số dòng (1M → 11M: số dòng × 11, thời gian × 22), còn Spark tăng chậm hơn hoặc bằng số dòng (1M → 11M: × 7,5; 11M → 22M: × 1,74).
4. **Kết luận thực tế:** với dữ liệu vừa RAM, pandas là lựa chọn đúng (nhanh hơn 4–38 lần trên máy 2 nhân); PySpark đáng dùng khi dữ liệu vượt RAM — đổi tốc độ lấy khả năng chạy được trong giới hạn bộ nhớ cố định.
5. **Cùng công thức, khác engine — kiểm chứng đến 11M dòng:** điểm profile của hai engine khớp tới 1e-16 ở mọi mốc, mở rộng kết quả đối chiếu 50.000 dòng của Giai đoạn 2.
6. **Giới hạn cần nêu:** (a) Spark chạy local trên 2 nhân, không phải cluster thật — minh họa xu hướng; (b) mốc 22M là HIGGS ghép 2 lần (mỗi dòng có bản trùng), không phải dữ liệu thật 22M dòng; tỷ lệ 11M → 22M của Spark (× 1,74) có thể thấp hơn dữ liệu thật vì bước tìm dòng trùng gặp nhiều trùng; (c) Spark 0,1M–11M và 22M đo ở hai phiên Colab khác nhau; (d) không đo bộ nhớ thật của Spark, chỉ biết nó chạy trong giới hạn heap 4 GB.

---

## 9. Danh mục biểu đồ và bảng cho báo cáo (người dùng chốt ngày 2026-10-23)

Biểu đồ vẽ bằng matplotlib và lưu `.png` vào `results/figures/`. Notebook vẽ là `notebooks/03_visualize_results.ipynb` (đã có trong cấu trúc thư mục). Code đọc dữ liệu từ file kết quả trên Drive, không chép số bằng tay.

| Mã | Biểu đồ / bảng | Dữ liệu nguồn | Mục đích trong báo cáo | Vẽ được từ |
|---|---|---|---|---|
| H1 | Điểm profile đo được theo λ, mỗi dimension một ô, kèm đường lý thuyết | Kết quả cell 10.2 (hiện chỉ in ra, cần lưu thành CSV) | Chứng minh polluter làm bẩn đúng liều lượng | Giai đoạn 3 |
| H2 | Baseline sạch: cột hiệu năng của 6 thuật toán, có thanh sai số theo 3 seed | `baseline_performance.csv` | Mốc so sánh và mức nhiễu nền | Giai đoạn 4 |
| H3 | **Đường cong suy giảm:** Δ theo λ, mỗi tác vụ một hàng, mỗi dimension một đường, dải sai số theo 3 seed | `experiment_results.parquet` | Kết quả chính, đặt cạnh Hình 6–19 của bài gốc | Giai đoạn 5 |
| H4 | Heatmap Δ tương đối (Δ / baseline) ở λ = 0,8: hàng là dimension, cột là thuật toán | `experiment_results.parquet` | Nhìn một lần thấy dimension nào hại tác vụ nào, cho thấy cần đánh giá theo tác vụ | Giai đoạn 5 |
| H5 | Bảng và biểu đồ cột nhóm: MAE / R² / Spearman của meta-model so với 3 baseline, chia theo tác vụ | Kết quả Giai đoạn 7 | So sánh chính của đề tài | Giai đoạn 7 |
| H6 | Scatter Δ dự đoán so với Δ thật, 4 ô (meta-model, rule-based, weighted, DQSOps-PCA), tô màu theo tác vụ | Kết quả Giai đoạn 7 | Thấy trực quan phương pháp nào bám đường chéo | Giai đoạn 7 |
| H7 | Ma trận nhầm lẫn Good/Bad của rule-based so với meta-model (đã rời rạc hóa theo cùng ngưỡng) | Kết quả Giai đoạn 7 | Rule-based chỉ cho nhãn Good/Bad nên cần so sánh riêng | Giai đoạn 7 |
| H8 | SHAP importance theo tác vụ: cột nhóm, hàng là dimension, mỗi màu một tác vụ | `shap_importance.csv` | Chứng minh "task-aware": cùng dimension nhưng trọng số khác theo tác vụ | Giai đoạn 6 |
| H9 | Trọng số PCA (PC1) của DQSOps, một bộ cho mọi tác vụ, đặt cạnh SHAP theo tác vụ | Giai đoạn 6 | Đối chiếu trực tiếp với DQSOps: trọng số cố định so với trọng số theo ngữ cảnh | Giai đoạn 6 |
| H10 | Runtime (và memory) theo số dòng, trục log, pandas so với PySpark | `scalability_benchmark.csv` | Chứng minh phần "quy mô lớn", chỉ ra điểm pandas không còn chạy được | Giai đoạn 8 |
| Phụ | Độ ổn định GMM theo seed (cell 11.5); thời gian chạy theo tác vụ | Cell 11.5, log Giai đoạn 5 | Phụ lục, giải thích lựa chọn cấu hình | Giai đoạn 4–5 |

**Lưu ý khi vẽ:** H3, H4 và H5 cần hai phiên bản, có và không có 24 lượt Class Balance × clustering, cho tới khi Giai đoạn 7 chốt. Ngoài ra, ΔAMI nhỏ hơn nhiều so với ΔF1 và ΔR², nên H3 dùng trục y riêng cho từng tác vụ.

### 9.1 Trạng thái vẽ (Colab 2026-10-28, commit `db03b4a`)

Đã vẽ và xem cả 10 hình (H1 và H6 sau khi sửa bố cục, commit `6265b20`): H1, H2, H3 (312 và 288 lượt), H4 (312 và 288 lượt), H5 (312 và 288 lượt), H7, H8, H9, H10; test `tests/test_visualization.py` `4 passed`. H1 và H6 ban đầu có hai lỗi bố cục (thiếu tiêu đề cột class_balance, thang trục bị kéo bởi rule-based), đã sửa và chạy lại đạt. Hình lưu ở `results/figures/` trên Drive.

Nhận xét khi xem hình (đối chiếu số liệu mục 6–8, chưa có số mới):
- H1: đường đo được bám đường lý thuyết; Feature Accuracy của Beijing và Covertype thấp hơn công thức ở λ = 0,2 (đã nêu ở Giai đoạn 3).
- H3: Target Accuracy × classification đạt đỉnh ở λ = 0,5 rồi giảm lại (rõ ở random forest); giả thuyết (chưa kiểm chứng): nhãn bị đảo quá nửa thì phần nhãn đảo lại mang quy luật ngược.
- H4: ở λ = 0,8 Feature Accuracy làm hại clustering nhiều nhất (Δ/baseline 0,99); Uniqueness gần như không ảnh hưởng regression và classification.
- H6, H7: rule-based gần như luôn dự đoán Bad (recall 1,00, precision 0,56); meta-model recall 0,82, precision 0,69.
- H10: pandas hết RAM ở 22M dòng; RAM chỉ đo cho pandas (thiết kế Giai đoạn 8), nên ô RAM không có đường Spark.
