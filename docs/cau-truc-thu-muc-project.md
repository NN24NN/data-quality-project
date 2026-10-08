# Cấu trúc thư mục project

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**File này dựa trên:** `pipeline-thi-nghiem.md` (10 module), `thiet-ke-thi-nghiem-chi-tiet.md` (ma trận thí nghiệm), `kien-truc-mo-hinh-va-ky-thuat.md` (kỹ thuật từng module)

**Trạng thái:** Khung thư mục trống **đã được tạo thật** tại `/Users/namnguyen/Documents/data-quality-project/` trên máy, đúng theo cây thư mục dưới đây — gồm đầy đủ thư mục con, mỗi thư mục có `README.md` ngắn giải thích mục đích, và `docs/` đã chứa bản copy của 5 file thiết kế (bao gồm chính file này). **Chưa có dữ liệu nào được tải và chưa có code thật trong `src/`.**

---

## 1. Cây thư mục đề xuất

```
data-quality-project/
│
├── requirements.txt                 # đã có sẵn — sẽ bổ sung pyspark, shap, xgboost/lightgbm
├── README.md                        # tổng quan project, hướng dẫn chạy
│
├── configs/
│   ├── seeds.yaml                   # seed_table cố định (42, 43, 44) — dùng xuyên suốt mọi module
│   ├── pollution_levels.yaml        # [0.0, 0.2, 0.5, 0.8]
│   └── experiment_matrix.yaml       # ma trận dataset × dimension × thuật toán (từ file thiết kế chi tiết)
│
├── data/
│   ├── raw/                         # dữ liệu tải về, giữ nguyên bản gốc, KHÔNG sửa trực tiếp
│   │   ├── beijing_air_quality/     # 12 file trạm gốc từ UCI
│   │   ├── higgs_sample/            # mẫu 500K-1M dòng
│   │   ├── higgs_full/              # 11M dòng — chỉ dùng cho bước 9 (Scalability)
│   │   └── covertype/
│   │
│   └── processed/                   # bản đã làm sạch/gộp, dùng làm "baseline" cho polluter
│       ├── beijing_clean.parquet
│       ├── higgs_sample_clean.parquet
│       └── covertype_clean.parquet
│
├── src/
│   ├── profiling/
│   │   ├── pandas_profiler.py       # 5 công thức dimension, bản pandas
│   │   └── spark_profiler.py        # 5 công thức dimension, bản PySpark
│   │
│   ├── pollution/
│   │   ├── completeness.py
│   │   ├── feature_accuracy.py
│   │   ├── target_accuracy.py
│   │   ├── uniqueness.py
│   │   └── class_balance.py
│   │
│   ├── downstream/
│   │   ├── classification.py        # LogR, RF + evaluate (F1)
│   │   ├── regression.py            # Ridge, GB + evaluate (R²)
│   │   └── clustering.py            # k-Means, GMM + evaluate (AMI)
│   │
│   ├── experiment_runner/
│   │   └── run_experiment.py        # vòng lặp chính, ghi log theo schema đã định
│   │
│   ├── metamodel/
│   │   ├── train_metamodel.py       # Gradient Boosting Regressor
│   │   └── explain_shap.py          # tính SHAP, xuất bảng trọng số theo task
│   │
│   ├── baselines/
│   │   ├── rule_based.py
│   │   ├── weighted_score.py
│   │   └── dqsops_pca.py
│   │
│   └── scalability/
│       └── spark_scale_study.py     # chạy spark_profiler.py trên các mốc kích thước HIGGS full
│
├── notebooks/
│   ├── 01_explore_datasets.ipynb    # khám phá dữ liệu, kiểm tra phân phối
│   ├── 02_validate_profiling.ipynb  # test công thức bằng tay trên ví dụ nhỏ
│   └── 03_visualize_results.ipynb   # vẽ đường cong suy giảm, biểu đồ scalability
│
├── results/
│   ├── experiment_results.parquet   # output chính của Experiment Runner (schema đã định)
│   ├── metamodel.pkl                # mô hình đã huấn luyện
│   ├── shap_importance.csv          # bảng trọng số theo task
│   ├── scalability_benchmark.csv    # runtime/memory theo kích thước
│   └── figures/                     # toàn bộ biểu đồ xuất ra (.png)
│
├── tests/
│   ├── test_profiling.py            # kiểm tra công thức đúng trên ví dụ có đáp án tay
│   └── test_pollution.py            # kiểm tra polluted ở λ=0 ≈ dataset gốc
│
└── docs/
    ├── tong-quan-bai-goc.md         # đã có trong project
    ├── pipeline-thi-nghiem.md       # đã có trong project
    ├── thiet-ke-thi-nghiem-chi-tiet.md
    ├── kien-truc-mo-hinh-va-ky-thuat.md
    └── cau-truc-thu-muc-project.md  # chính file này
```

---

## 2. Giải thích các quyết định cấu trúc

### 2.1 Vì sao tách `data/raw/` và `data/processed/`
`raw/` giữ nguyên dữ liệu gốc tải về, không bao giờ sửa trực tiếp — nếu polluter hay profiling có lỗi, luôn có bản gốc để đối chiếu/chạy lại từ đầu. `processed/` chứa bản đã gộp (Beijing 12 trạm) và đã xử lý missing tự nhiên để làm điểm xuất phát "sạch" cho polluter.

### 2.2 Vì sao không có `data/polluted/`
Với ~312 lượt chạy, nếu lưu mỗi bản dữ liệu đã làm bẩn ra đĩa sẽ tốn dung lượng rất lớn và không cần thiết — vì `polluter(df, level, seed)` luôn tái lập được y hệt nhờ seed cố định. **Dữ liệu polluted nên được tạo trong bộ nhớ (in-memory) tại thời điểm chạy, không persist ra đĩa.** Đây là lý do `configs/seeds.yaml` quan trọng: nó là "nguồn sự thật" để tái lập lại bất kỳ lượt chạy nào mà không cần lưu file.

### 2.3 Vì sao `src/` chia theo module thay vì theo dataset
Mỗi module (profiling, pollution, downstream, ...) là một trách nhiệm kỹ thuật độc lập, có thể test riêng (khớp với `tests/`) và giao cho từng thành viên trong nhóm phụ trách — đúng tinh thần "triển khai thuận tiện, làm song song" mà bạn yêu cầu từ đầu. Việc map module nào dùng cho dataset nào nằm trong `configs/experiment_matrix.yaml`, không hard-code trong code.

### 2.4 Vì sao có `configs/` riêng
Tách tham số (seed, mức độ ô nhiễm, ma trận thí nghiệm) ra khỏi code giúp:
- Đổi tham số (vd thêm 1 mức ô nhiễm) không cần sửa code
- Nhóm review nhanh "đang chạy với tham số gì" mà không cần đọc code
- Dễ ghi vào báo cáo đồ án phần "Thiết lập thí nghiệm" bằng cách trích trực tiếp từ file config

### 2.5 Vì sao `docs/` chứa cả các file .md đã viết
Để toàn bộ tài liệu thiết kế nằm cùng chỗ với code, thuận tiện khi nộp bài hoặc khi thành viên mới vào nhóm đọc để bắt kịp tiến độ — không cần tìm lại trên project claude.ai.

---

## 3. Quy ước đặt tên (naming convention)

- File Python: `snake_case.py`
- File kết quả: có tiền tố mô tả rõ nội dung (`experiment_results.parquet`, không đặt `results.parquet` chung chung)
- Cột trong bảng kết quả: đúng theo schema đã định trong `thiet-ke-thi-nghiem-chi-tiet.md` mục 5, không đổi tên tùy tiện giữa các module
- Seed: luôn đọc từ `configs/seeds.yaml`, không hardcode số seed rải rác trong code

---

## 4. Thứ tự tạo thư mục khi bắt đầu code (gợi ý)

1. `configs/` + `requirements.txt` (cập nhật) — làm trước tiên vì mọi module khác phụ thuộc vào đây
2. `data/raw/` + `data/processed/` — sau khi tải dataset
3. `src/profiling/` + `tests/test_profiling.py` — làm trước để có thể test ngay công thức
4. `src/pollution/` + `tests/test_pollution.py`
5. `src/downstream/`
6. `src/experiment_runner/` — ghép 3 module trên lại
7. `results/` — tự sinh ra khi Experiment Runner chạy lần đầu
8. `src/metamodel/` + `src/baselines/` — sau khi có `results/experiment_results.parquet`
9. `src/scalability/` — độc lập, có thể làm song song từ bước 3
10. `notebooks/` — làm xuyên suốt, không có thứ tự cố định

---

## Đã hoàn tất

Khung thư mục trống đã được tạo thật tại `/Users/namnguyen/Documents/data-quality-project/` (dùng folder cha do bạn tạo sẵn), gồm:
- Toàn bộ cây thư mục ở mục 1 (configs/, data/raw/ + processed/, src/ với 7 module, notebooks/, results/ + figures/, tests/, docs/)
- Mỗi thư mục có `README.md` ngắn nêu mục đích (thay cho `.gitkeep` đơn thuần)
- `docs/` chứa bản copy của 5 file thiết kế: `tong-quan-bai-goc.md`, `pipeline-thi-nghiem.md`, `thiet-ke-thi-nghiem-chi-tiet.md`, `kien-truc-mo-hinh-va-ky-thuat.md`, `cau-truc-thu-muc-project.md`
- `README.md` và `requirements.txt` ở gốc project

**Việc còn lại (cần xác nhận riêng trước khi thực hiện):**
1. Tải 3 dataset thật vào `data/raw/`
2. Cài `pyspark`, `shap`, `xgboost` (đang để comment trong `requirements.txt`)
3. Viết code thật cho từng module trong `src/`
