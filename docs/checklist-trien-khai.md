# Checklist triển khai đề tài

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**Mục đích file này:** danh sách các việc cần làm theo thứ tự, dùng để theo dõi tiến độ — tick dần khi hoàn thành từng việc. Tham chiếu chi tiết kỹ thuật cho mỗi việc nằm trong 4 file thiết kế: `pipeline-thi-nghiem.md`, `thiet-ke-thi-nghiem-chi-tiet.md`, `kien-truc-mo-hinh-va-ky-thuat.md`, `cau-truc-thu-muc-project.md`.

**Trạng thái tổng quan:** Giai đoạn 0, 1 (trừ HIGGS full, làm sau), 2 (module Profiling) và 3 (module Pollution) đã xong. Tiếp theo: Giai đoạn 4 (module ML downstream). **Đã chốt (2026-10-09):** giữ Class Balance cho clustering (Covertype) khi chạy thí nghiệm — vẫn 312 lượt — dù điểm Balance đo được tăng khi λ tăng (0,254 → 0,451); đến Giai đoạn 7 mới quyết định có đưa 24 lượt này vào bảng so sánh baseline không.

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
- [ ] (Có thể làm sau) Tải HIGGS full (11M dòng, ~2.8GB) vào `data/raw/higgs_full/` — chỉ cần trước khi làm Giai đoạn 6
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

- [ ] Viết `classification.py` (Logistic Regression + Random Forest, đo F1-macro) cho HIGGS
- [ ] Viết `regression.py` (Ridge + Gradient Boosting, đo R²) cho Beijing
- [ ] Viết `clustering.py` (k-Means + Gaussian Mixture, đo AMI) cho Covertype
- [ ] Chạy baseline sạch (không ô nhiễm) cho mỗi (dataset, thuật toán) để có mốc so sánh

## Giai đoạn 5 — Experiment Runner (`src/experiment_runner/`)

- [ ] Viết `run_experiment.py`: vòng lặp task → dataset → dimension → level → run (ghép module 2+3+4)
- [ ] Chạy thử ở quy mô nhỏ (1 dataset, 1 dimension, vài mức) để kiểm tra pipeline không lỗi
- [ ] Chạy full 312 lượt, ghi ra `results/experiment_results.parquet` theo đúng schema đã định
- [ ] Kiểm tra sơ bộ dữ liệu output: không có dòng lỗi/NaN bất thường, số dòng đúng 312

## Giai đoạn 6 — Meta-model & Baselines

- [ ] Viết `metamodel/train_metamodel.py` (Gradient Boosting Regressor dự đoán ΔPerformance)
- [ ] Viết `metamodel/explain_shap.py` (tính SHAP, xuất bảng trọng số theo task) → `results/shap_importance.csv`
- [ ] Viết `baselines/rule_based.py`
- [ ] Viết `baselines/weighted_score.py`
- [ ] Viết `baselines/dqsops_pca.py` (baseline quan trọng nhất — PCA composite theo đúng cách DQSOps)
- [ ] Lưu mô hình đã huấn luyện → `results/metamodel.pkl`

## Giai đoạn 7 — Đánh giá & So sánh

- [ ] Tính MAE, R² cho meta-model và từng baseline
- [ ] Tính hệ số tương quan Spearman
- [ ] Phân tích theo từng tác vụ riêng (baseline nào thắng ở tác vụ nào)
- [ ] Quyết định có đưa 24 lượt Class Balance × clustering (Covertype) vào bảng so sánh baseline không — xem ΔAMI của 24 lượt này trước; báo cáo phải nêu rõ bảng chính tính trên tập nào (312 hay 288 lượt)
- [ ] Tổng hợp bảng so sánh cuối cùng

## Giai đoạn 8 — Scalability study (Spark, có thể làm song song từ Giai đoạn 2)

- [ ] Viết `scalability/spark_scale_study.py`
- [ ] Chạy `spark_profiler.py` trên các mốc kích thước tăng dần của HIGGS full (1M, 3M, 6M, 11M dòng)
- [ ] Đo runtime/memory, đối chiếu với bản pandas ở các mốc nhỏ
- [ ] Lưu kết quả → `results/scalability_benchmark.csv`

## Giai đoạn 9 — Báo cáo & Biểu đồ

- [ ] Vẽ đường cong suy giảm hiệu năng theo mức độ ô nhiễm (theo dimension) → `results/figures/`
- [ ] Vẽ bảng so sánh meta-model vs 3 baseline
- [ ] Vẽ biểu đồ trọng số/importance theo tác vụ (SHAP)
- [ ] Vẽ biểu đồ scalability
- [ ] Viết báo cáo đồ án hoàn chỉnh, tổng hợp toàn bộ kết quả

---

## Lưu ý khi dùng file này

- Đánh dấu `[x]` khi hoàn thành từng việc, giữ nguyên các mục chưa làm là `[ ]`
- Giai đoạn 2-4 có thể làm song song bởi các thành viên khác nhau (độc lập về module)
- Giai đoạn 8 (Spark) độc lập hoàn toàn, có thể bắt đầu ngay sau Giai đoạn 2
- Mọi việc tải dữ liệu, cài thư viện, hoặc thay đổi thật trên máy vẫn cần xác nhận trước khi Claude thực hiện, theo đúng quy tắc đã thống nhất từ đầu
