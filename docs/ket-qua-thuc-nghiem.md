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
| 6 — Meta-model + 3 baseline | *Chưa có* |
| 7 — Đánh giá & so sánh | *Chưa có* |
| 8 — Scalability (Spark) | *Chưa có* |

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

*Chưa có.* Sẽ ghi: siêu tham số meta-model, cách chia train/test cho meta-model, bảng SHAP importance theo tác vụ, trọng số PCA (PC1) của baseline DQSOps, và ngưỡng của baseline rule-based.

## 7. Đánh giá và so sánh (Giai đoạn 7)

*Chưa có.* Sẽ ghi: bảng MAE / R² / Spearman cho meta-model và 3 baseline (toàn bộ và từng tác vụ); accuracy/F1 Good/Bad cho baseline rule-based; bảng chính tính trên 312 hay 288 lượt.

## 8. Scalability (Giai đoạn 8)

*Chưa có.* Sẽ ghi: runtime và memory của profiling pandas so với PySpark ở các mốc 1M / 3M / 6M / 11M dòng HIGGS.

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
