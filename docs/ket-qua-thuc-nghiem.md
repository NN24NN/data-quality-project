# Kết quả thực nghiệm (tổng hợp cho báo cáo)

**Mục đích file này:** gom **số liệu đã thực sự chạy được** của từng giai đoạn vào một chỗ, để viết báo cáo và làm slide. Mỗi kết quả ghi rõ chạy ở đâu (cell notebook), commit nào, ngày nào. Cách làm và lý do của từng quyết định nằm ở `cach-trien-khai-chi-tiet-tung-giai-doan.md`; điểm lệch so với bài gốc nằm ở `khac-biet-so-voi-bai-goc.md`.

**Quy ước:** chỉ ghi số đã chạy và kiểm chứng. Phần chưa chạy để trống, có ghi chú *Chưa có*. Số thập phân dùng dấu phẩy.

| Giai đoạn | Trạng thái kết quả |
|---|---|
| 1 — Dữ liệu | Có |
| 2 — Profiling | Có |
| 3 — Pollution | Có |
| 4 — Downstream (baseline sạch) | Có |
| 5 — Experiment Runner (312 lượt) | *Chưa có* (đã viết code, chưa chạy) |
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

*Chưa có.* Sẽ ghi các mục sau:
- Kết quả test 12.1 và lượt chạy thử 12.2, kèm thời gian của một lượt.
- Kiểm tra file kết quả (12.4): số dòng từng tác vụ, NaN, max |Δ| ở λ = 0.
- Bảng Δ trung bình ± độ lệch chuẩn theo (tác vụ, thuật toán, dimension, λ).
- Thời gian chạy thật của từng tác vụ.
- ΔAMI của 24 lượt Class Balance × clustering, làm căn cứ cho quyết định ở Giai đoạn 7.

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
