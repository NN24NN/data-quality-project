# Checklist triển khai đề tài

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**Mục đích file này:** danh sách các việc cần làm theo thứ tự, dùng để theo dõi tiến độ — tick dần khi hoàn thành từng việc. Tham chiếu chi tiết kỹ thuật cho mỗi việc nằm trong 4 file thiết kế: `pipeline-thi-nghiem.md`, `thiet-ke-thi-nghiem-chi-tiet.md`, `kien-truc-mo-hinh-va-ky-thuat.md`, `cau-truc-thu-muc-project.md`.

> ## ▶ BẮT ĐẦU PHIÊN SAU TỪ ĐÂY (ghi ngày 2026-10-26)
>
> **Đang dừng ở:** Giai đoạn 6 **xong** (chạy trên Colab ngày 2026-10-26, commit `e47cda9`; `116 passed`). Kết quả trên Drive `results/`: `metamodel_cv_predictions.parquet`, `metamodel.pkl`, `dqsops_pca_weights.csv`, `shap_importance.csv`. Số liệu và nhận xét: `ket-qua-thuc-nghiem.md` mục 6.
>
> **Giai đoạn 7 xong** (Colab 2026-10-26, commit `755bf48`, `124 passed`). Kết quả và 10 kết luận: `ket-qua-thuc-nghiem.md` mục 7.
>
> **Giai đoạn 8 xong** (Colab 2026-10-26, commit `54a7837`, `134 passed`): pandas hết RAM ở 22M dòng, PySpark chạy được trong giới hạn 4 GB; số liệu và 6 kết luận ở `ket-qua-thuc-nghiem.md` mục 8.3–8.4. Giai đoạn 0–8 đều xong (trừ cài `requirements.txt` local — bỏ qua vì chạy trên Colab).
>
> **Kết thúc phiên ngày 2026-10-26** (commit cuối của phiên được ghi trong `git log`). Người dùng quyết định **giữ** `higgs_full_raw.parquet` (0,69 GB) trên Drive. Mọi file kết quả nằm trên Drive `results/`, không có trong git: `experiment_results.parquet`, `baseline_performance.csv`, `metamodel_cv_predictions.parquet`, `metamodel.pkl`, `dqsops_pca_weights.csv`, `shap_importance.csv`, `shap_by_row_dimension.csv`, `method_comparison.csv`, `method_bootstrap.csv`, `goodbad_comparison.csv`, `scalability_benchmark.csv`.
>
> **Cập nhật 2026-10-28 (Giai đoạn 9 đã bắt đầu):** đã viết `src/visualization/plots.py`, `notebooks/03_visualize_results.ipynb`, `tests/test_visualization.py`, và sửa cell 10.2 để ghi `pollution_check.csv`. **Chưa chạy trên Colab nên chưa tick mục nào.** Việc kế tiếp: commit/push (khi người dùng đồng ý) → chạy cell 0b (test), các cell hình, và cell 10.2 trên Colab → xem hình, chỉnh bố cục → tick H1–H10. Cấu trúc báo cáo còn để thảo luận.

**Việc tiếp theo: Giai đoạn 9 — biểu đồ và báo cáo.**
> 1. Gợi ý mức model: vẽ biểu đồ ít thiết kế → có thể dùng Sonnet; viết báo cáo/diễn giải kết quả → Opus. Nhắc người dùng trước khi bắt đầu, theo `CLAUDE.md`.
> 2. Trước khi viết code: chốt với người dùng cách tổ chức `notebooks/03_visualize_results.ipynb` (đọc file kết quả từ Drive, lưu `.png` vào `results/figures/`), thứ tự vẽ, và cấu trúc báo cáo. Danh mục H1–H10 ở `ket-qua-thuc-nghiem.md` mục 9; dữ liệu nguồn của từng hình đã có đủ trừ H1.
> 3. Giai đoạn 9: biểu đồ H1–H10 (người dùng chốt để dồn về Giai đoạn 9). Còn treo cho H1: sửa cell 10.2 để lưu CSV. H8 cần chú thích cách đọc tỷ trọng SHAP của Uniqueness ở classification (`ket-qua-thuc-nghiem.md` mục 7.6 ý 9).
>
> **Cách làm việc (máy local không có Python):** viết code → rà bằng tay → commit/push khi người dùng đồng ý → người dùng chạy cell trên Colab (mục 1, 2, 9.1 trước mỗi phiên mới; chọn runtime CPU) → gửi kết quả **dán dạng văn bản** → ghi số liệu vào `ket-qua-thuc-nghiem.md`, tick checklist, cập nhật các `.md` liên quan.

**Trạng thái tổng quan:** Giai đoạn 0, 1 (trừ HIGGS full, làm sau), 2 (module Profiling), 3 (module Pollution) và 4 (module ML downstream) đã xong. Giai đoạn 5 (Experiment Runner): **đã xong: 312/312 dòng (regression 140 + clustering 126 + classification 148 phút, tổng ~6,9 giờ), kiểm tra đạt.** λ = 0 chỉ huấn luyện thật một lần cho mỗi (thuật toán, run), chép sang các dimension khác, nên chỉ 252 trong 312 dòng được huấn luyện thật. Giai đoạn 6 (meta-model + 3 baseline): **đã xong ngày 2026-10-26** (`116 passed`; Δ tuyệt đối, Group K-Fold: MAE meta-model 0,055 so với 0,077 weighted score và 0,084 DQSOps-PCA). Giai đoạn 7 (đánh giá & so sánh): **đã xong ngày 2026-10-26** (`124 passed`; meta-model hơn DQSOps-PCA có ý nghĩa thống kê, hơn weighted score chưa có ý nghĩa ở tổng thể). Giai đoạn 8 (Spark scalability): **đã xong ngày 2026-10-26** (pandas nhanh hơn 4–38 lần nhưng hết RAM ở 22M dòng; PySpark chạy được 22M trong giới hạn 4 GB). Tiếp theo: Giai đoạn 9 (biểu đồ + báo cáo). **Đã chốt (2026-10-09):** giữ Class Balance cho clustering (Covertype) khi chạy thí nghiệm — vẫn 312 lượt — dù điểm Balance đo được tăng khi λ tăng (0,254 → 0,451); đến Giai đoạn 7 mới quyết định có đưa 24 lượt này vào bảng so sánh baseline không.

**Ghi chú Giai đoạn 1:** nhóm A (3 file config) đã xong; 3 dataset (Beijing, HIGGS mẫu, Covertype) đã tải xong trên Google Drive qua notebook Colab (đã kiểm tra shape). Nhóm D (3 bản sạch) đã chạy và kiểm tra xong. Còn lại của Giai đoạn 1: HIGGS full (làm sau, trước Giai đoạn 8) và mục cài `requirements.txt` ở local (có thể bỏ qua vì chạy trên Colab).

**Lưu ý lệch tài liệu thiết kế (phát hiện khi tải Beijing):** tên cột thực tế là `wd` (không phải `cbwd`), `RAIN`/`WSPM` (không phải `Ir`/`Is`/`Iws`), có thêm `PM10, SO2, NO2, CO, O3` và các cột thời gian `year, month, day, hour`, `No`. Đã sửa `thiet-ke-thi-nghiem-chi-tiet.md` mục 1.1. Chi tiết cách làm: `cach-trien-khai-chi-tiet-tung-giai-doan.md`.

---

## Giai đoạn 0 — Thiết kế (đã hoàn tất)

- [x] Review ý tưởng gốc, xác định base paper + research gap
- [x] Đặt tên đề tài
- [x] Tổng quan 2 bài báo gốc (`tong-quan-bai-goc.md`)
- [x] Thu gọn phạm vi thí nghiệm (4-5 dimension, 4 mức ô nhiễm, 2 thuật toán/tác vụ, Scenario 3)
- [x] Viết pipeline tổng thể (`pipeline-thi-nghiem.md`)
- [x] Viết thiết kế thí nghiệm chi tiết — ma trận 312 lượt chạy (`thiet-ke-thi-nghiem-chi-tiet.md`)
- [x] Viết kiến trúc mô hình & kỹ thuật (`kien-truc-mo-hinh-va-ky-thuat.md`)
- [x] Thiết kế cấu trúc thư mục + tạo khung thư mục trống thật trên máy (`cau-truc-thu-muc-project.md`)

---

## Giai đoạn 1 — Chuẩn bị dữ liệu & môi trường

- [x] Xác nhận và tải Beijing Multi-Site Air Quality (12 trạm, UCI) vào `data/raw/beijing_air_quality/` — *12 file CSV `PRSA_Data_*.csv`, mỗi file 35.064 dòng × 18 cột (đã có sẵn cột `station`)*
- [x] Xác nhận và tải mẫu HIGGS (~500K-1M dòng) vào `data/raw/higgs_sample/` — *`higgs_sample_raw.parquet`, 1.000.000 × 29, target 1: 52,99% / 0: 47,01% (seed 42)*
- [x] Xác nhận và tải Covertype full (581K dòng) vào `data/raw/covertype/` — *`covertype_raw.parquet`, 581.012 × 55, 7 lớp (lớp 4 chỉ 2.747 dòng)*
- [x] (Có thể làm sau) Tải HIGGS full (11M dòng, ~2.8GB) vào `data/raw/higgs_full/` — chỉ cần cho Giai đoạn 8 — *cell 15.2 ngày 2026-10-26: `higgs_full_raw.parquet` trên Drive, 11.000.000 dòng, 0,69GB*
- [x] Gộp 12 file trạm Beijing thành 1 bảng, thêm cột `station` — *420.768 × 18 (cột `station` đã có sẵn trong từng file)*
- [x] Tạo bản "baseline" sạch cho Beijing (xử lý missing tự nhiên, one-hot `wd`/`station`) → `data/processed/beijing_clean.parquet` — *382.168 × 42 (bỏ 38.600 dòng missing = 9,2%), NaN = 0, trùng = 0*
- [x] Tạo bản sạch cho HIGGS sample (lấy mẫu stratified theo target) → `data/processed/higgs_sample_clean.parquet` — *1.000.000 × 29; có 2.281 dòng trùng (0,23%), giữ nguyên*
- [x] Tạo bản sạch cho Covertype (dùng trực tiếp, không cần gộp) → `data/processed/covertype_clean.parquet` — *581.012 × 55, trùng = 0*
- [ ] Cập nhật `requirements.txt`: bỏ comment `pyspark`, `shap`, `xgboost`; cài đặt (`pip install -r requirements.txt`) — *đã sửa file (thêm cả `pyyaml`, `pyarrow`); **chưa cài đặt** (cần xác nhận, nên cài trên Colab)*
- [x] Tạo `configs/seeds.yaml` (seed cố định 42, 43, 44)
- [x] Tạo `configs/pollution_levels.yaml` ([0.0, 0.2, 0.5, 0.8])
- [x] Tạo `configs/experiment_matrix.yaml` (ma trận dataset × dimension × thuật toán, theo `thiet-ke-thi-nghiem-chi-tiet.md` mục 3)

---

## Giai đoạn 2 — Module Profiling (`src/profiling/`)

*Trạng thái: xong. Đã chạy trên Colab (pytest 12/12 passed; pandas và PySpark khớp trên mẫu 50.000 dòng của cả 3 dataset).*

- [x] Viết `pandas_profiler.py`: công thức Completeness, Feature Accuracy, Target Accuracy, Uniqueness, (Target Class Balance) — *công thức nằm ở `common.py`, dùng chung hai engine*
- [x] Viết test tay (`tests/test_profiling.py`): kiểm tra công thức đúng trên ví dụ nhỏ có đáp án tay — *12 passed*
- [x] Viết `spark_profiler.py`: cùng công thức, bản PySpark — chạy thử trên mẫu nhỏ trước khi dùng cho Giai đoạn 6 — *chạy trên mẫu 50.000 dòng × 3 dataset*
- [x] Đối chiếu kết quả `pandas_profiler.py` và `spark_profiler.py` trên cùng 1 dataset nhỏ để đảm bảo khớp — *khớp tới sai số tương đối 1e-9*

## Giai đoạn 3 — Module Pollution (`src/pollution/`)

*Trạng thái: xong. Đã chạy trên Colab ngày 2026-10-09 ở commit `bbd1721` (pytest 43/43 passed; bảng 10.2 đủ 52 dòng trên mẫu 100.000 dòng × 3 dataset, điểm đo được khớp giá trị lý thuyết). Quyết định về Class Balance cho clustering: xem dòng "Trạng thái tổng quan".*

- [x] Viết `completeness.py` (MCAR missing value theo mức độ) — *Completeness đo được đúng 1 − λ trên cả 3 dataset*
- [x] Viết `feature_accuracy.py` (nhiễu Gaussian/đổi giá trị ngẫu nhiên) — *HIGGS 0,643 / 0,436 / 0,287; Beijing và Covertype 0,722 / 0,468 / 0,243*
- [x] Viết `target_accuracy.py` — *HIGGS (đổi nhãn) đúng 1 − λ; Beijing (nhiễu số) 0,643 / 0,435 / 0,285*
- [x] Viết `uniqueness.py` (nhân bản có kiểm soát) — *Uniqueness ≈ 1 − λ; số dòng 100.000 → 125.000 / 200.000 / 500.000*
- [x] Viết `class_balance.py` — *HIGGS giảm đúng hướng 0,889 → 0,711 / 0,444 / 0,178; **Covertype đi ngược** 0,254 → 0,303 / 0,377 / 0,451*
- [x] Viết test (`tests/test_pollution.py`): dữ liệu polluted ở mức λ=0 phải ≈ dataset gốc — *31 test pollution, passed*
- [x] Kiểm tra mọi polluter đều dùng seed từ `configs/seeds.yaml` → tái lập được chính xác — *seed là tham số của `pollute`, nơi gọi đọc từ `seeds.yaml`; test cùng seed cho cùng kết quả passed cho cả 5 polluter*

## Giai đoạn 4 — Module ML downstream (`src/downstream/`)

*Trạng thái: xong. Đã chạy trên Colab ngày 2026-10-09 ở commit `87d3f20` (pytest 93/93 passed; baseline toàn bộ dữ liệu, 3 seed). Random Forest dùng `max_samples: 0.1` (mỗi cây học 10% số dòng): lượt đối chứng cho thấy F1 chỉ thấp hơn bản mặc định 0,008 (0,725 so với 0,733) mà nhanh gấp 6,6 lần (129 giây so với 851 giây). AMI của GMM dao động mạnh theo seed (0,10 / 0,20 / 0,21) nên đã đổi GMM sang `diag`, `n_init: 10` (đo ở cell 11.5: độ lệch chuẩn 0,045 → 0,010); baseline đã chạy lại với cấu hình này.*

- [x] Viết `classification.py` (Logistic Regression + Random Forest, đo F1-macro) cho HIGGS — *mẫu 100.000 dòng: F1 0,630 (LogR) / 0,720 (RF)*
- [x] Viết `regression.py` (Ridge + Gradient Boosting, đo R²) cho Beijing — *mẫu 100.000 dòng: R² 0,848 (Ridge) / 0,909 (GB)*
- [x] Viết `clustering.py` (k-Means + Gaussian Mixture, đo AMI) cho Covertype — *mẫu 100.000 dòng: AMI 0,160 (k-Means) / 0,174 (GMM) — thấp, ΔAMI có ít dư địa*
- [x] Chạy baseline sạch (không ô nhiễm) cho mỗi (dataset, thuật toán) để có mốc so sánh — *toàn bộ dữ liệu, 3 seed, lưu ở `results/baseline_performance.csv` trên Drive: F1 0,635 (LogR) / 0,725 (RF); R² 0,858 (Ridge) / 0,916 (GB); AMI 0,172 ± 0,017 (k-Means) / 0,196 ± 0,006 (GMM, cấu hình `diag`, `n_init: 10`)*

## Giai đoạn 5 — Experiment Runner (`src/experiment_runner/`)

- [x] Viết `run_experiment.py`: vòng lặp task → dataset → dimension → level → run (ghép module 2+3+4) — *kèm 7 test (`tests/test_experiment_runner.py`); Colab 2026-10-23: 100 passed*
- [x] Chạy thử ở quy mô nhỏ (1 dataset, 1 dimension, vài mức) để kiểm tra pipeline không lỗi — *cell 12.2: regression × completeness × λ ∈ {0; 0,5} × run 1; Δ = 0 ở λ = 0, Δ +0,278 (Ridge) / +0,248 (GB) ở λ = 0,5; 3,0 phút*
- [x] Chạy full 312 lượt, ghi ra `results/experiment_results.parquet` theo đúng schema đã định — *Colab 2026-10-23: regression 140 phút, clustering 126 phút, classification 148 phút; tổng 414,5 phút*
- [x] Kiểm tra sơ bộ dữ liệu output: không có dòng lỗi/NaN bất thường, số dòng đúng 312 — *cell 12.4: 312/312, trùng khóa 0, NaN chỉ ở `profile_class_balance` của regression (96 dòng, đúng thiết kế), max \|Δ\| ở λ = 0 là 1,1e-16*

## Giai đoạn 6 — Meta-model & Baselines

*Trạng thái: xong. Đã chạy trên Colab ngày 2026-10-26 ở commit `e47cda9` (cell 13.1: `116 passed`; 13.2: 8.112 dòng dự đoán out-of-fold; 13.3: SHAP theo tác vụ). Số liệu: `ket-qua-thuc-nghiem.md` mục 6.*

- [x] Viết `metamodel/train_metamodel.py` (Gradient Boosting Regressor dự đoán ΔPerformance) — *kèm đánh giá chéo Group K-Fold + nội suy; Δ tuyệt đối, MAE tổng thể 0,055 so với 0,077 (weighted) / 0,084 (DQSOps-PCA)*
- [x] Viết `metamodel/explain_shap.py` (tính SHAP, xuất bảng trọng số theo task) → `results/shap_importance.csv` — *Uniqueness thấp nhất ở cả 3 tác vụ; dimension đứng đầu khác nhau theo tác vụ*
- [x] Viết `baselines/rule_based.py`
- [x] Viết `baselines/weighted_score.py`
- [x] Viết `baselines/dqsops_pca.py` (baseline quan trọng nhất — PCA composite theo đúng cách DQSOps) — *PC1 chỉ giải thích 26,8% phương sai; Feature Accuracy trọng số ≈ 0, Target Accuracy âm*
- [x] Lưu mô hình đã huấn luyện → `results/metamodel.pkl`

## Giai đoạn 7 — Đánh giá & So sánh

*Trạng thái: xong. Đã chạy trên Colab ngày 2026-10-26 ở commit `755bf48` (cell 14.1: `124 passed`; 14.2: 3 bảng kết quả; 14.3: chẩn đoán SHAP). Số liệu và 10 kết luận: `ket-qua-thuc-nghiem.md` mục 7.*

- [x] Tính MAE, R² cho meta-model và từng baseline — *Δ tuyệt đối, Group K-Fold: MAE meta-model 0,055 ± 0,006 so với 0,077 (weighted) / 0,084 (DQSOps-PCA); R² trong từng tác vụ không ổn định*
- [x] Tính hệ số tương quan Spearman — *meta-model 0,62–0,72 theo tác vụ; DQSOps-PCA ≈ 0*
- [x] Phân tích theo từng tác vụ riêng (baseline nào thắng ở tác vụ nào) — *bootstrap theo nhóm: meta-model hơn DQSOps-PCA có ý nghĩa; hơn weighted score chưa có ý nghĩa ở tổng thể; weighted score tốt nhất ở Good/Bad*
- [x] Quyết định có đưa 24 lượt Class Balance × clustering (Covertype) vào bảng so sánh baseline không — xem ΔAMI của 24 lượt này trước; báo cáo phải nêu rõ bảng chính tính trên tập nào (312 hay 288 lượt) — *chốt 2026-10-26: bảng chính 312 lượt, bảng phụ 288 lượt (chỉ lọc phần đánh giá); lý do ở `khac-biet-so-voi-bai-goc.md` mục 4.6*
- [x] Tổng hợp bảng so sánh cuối cùng — *`ket-qua-thuc-nghiem.md` mục 7.1 (bảng chính) và 7.6 (kết luận)*

## Giai đoạn 8 — Scalability study (Spark, có thể làm song song từ Giai đoạn 2)

*Trạng thái: đã chốt thiết kế (2026-10-26: thêm mốc 0,1M; profile bản sạch so với chính nó; mỗi lần đo một tiến trình con, 3 lần, trung vị; chỉ đo bộ nhớ pandas, Spark giới hạn 4g; lưu HIGGS full parquet lên Drive) và viết code. **Xong ngày 2026-10-26** (lần 2, commit `54a7837`, `134 passed`; máy Colab 2 nhân, 13,6 GB RAM). Lần 1 đo sai bộ nhớ pandas và pandas không hết RAM ở 11M → sửa cách đo + thêm mốc 22M (HIGGS ghép 2 lần). Số liệu: `ket-qua-thuc-nghiem.md` mục 8.*

- [x] Viết `scalability/spark_scale_study.py`
- [x] Chạy `spark_profiler.py` trên các mốc kích thước tăng dần của HIGGS full (1M, 3M, 6M, 11M dòng) — *thêm 0,1M và 22M; Spark chạy được mọi mốc, 22M trong 527 giây với giới hạn 4 GB*
- [x] Đo runtime/memory, đối chiếu với bản pandas ở các mốc nhỏ — *pandas nhanh hơn 4–38 lần nhưng **hết RAM ở 22M** (7,3 GB ở 11M, máy 13,6 GB); điểm profile hai engine khớp tới 1e-16 đến 11M*
- [x] Lưu kết quả → `results/scalability_benchmark.csv` — *36 dòng trên Drive*

## Giai đoạn 9 — Báo cáo & Biểu đồ

Danh mục biểu đồ đã chốt ngày 2026-10-23, mô tả chi tiết ở `ket-qua-thuc-nghiem.md` mục 9. Vẽ trong `notebooks/03_visualize_results.ipynb`, lưu `.png` vào `results/figures/`.

- [ ] Sửa cell 10.2 để lưu kết quả kiểm chứng polluter ra CSV (dữ liệu cho H1), chạy lại
- [ ] H1 — điểm profile đo được theo λ, kèm đường lý thuyết
- [ ] H2 — baseline sạch 6 thuật toán, thanh sai số theo 3 seed
- [ ] H3 — đường cong suy giảm Δ theo λ, theo dimension, mỗi tác vụ một hàng (có và không có 24 lượt Class Balance × clustering)
- [ ] H4 — heatmap Δ tương đối ở λ = 0,8 (dimension × thuật toán)
- [ ] H5 — bảng + biểu đồ cột MAE / R² / Spearman: meta-model vs 3 baseline, theo tác vụ
- [ ] H6 — scatter Δ dự đoán vs Δ thật, 4 phương pháp
- [ ] H7 — ma trận nhầm lẫn Good/Bad: rule-based vs meta-model
- [ ] H8 — SHAP importance theo tác vụ
- [ ] H9 — trọng số PCA (DQSOps) đặt cạnh SHAP theo tác vụ
- [ ] H10 — runtime/memory pandas vs PySpark theo số dòng
- [ ] Viết báo cáo đồ án hoàn chỉnh, tổng hợp toàn bộ kết quả

---

## Lưu ý khi dùng file này

- Đánh dấu `[x]` khi hoàn thành từng việc, giữ nguyên các mục chưa làm là `[ ]`
- Giai đoạn 2-4 có thể làm song song bởi các thành viên khác nhau (độc lập về module)
- Giai đoạn 8 (Spark) độc lập hoàn toàn, có thể bắt đầu ngay sau Giai đoạn 2
- Mọi việc tải dữ liệu, cài thư viện, hoặc thay đổi thật trên máy vẫn cần xác nhận trước khi Claude thực hiện, theo đúng quy tắc đã thống nhất từ đầu
