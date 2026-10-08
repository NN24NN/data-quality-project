# Cách triển khai chi tiết từng giai đoạn

**Mục đích file này:** ghi lại *đã làm gì* và *làm như thế nào* để hoàn thành từng mục trong `checklist-trien-khai.md`. Cập nhật mỗi khi xong một mục hoặc một giai đoạn (quy tắc trong `CLAUDE.md`). Mục nào chưa chạy/kiểm chứng thì ghi rõ là chưa.

---

## Giai đoạn 0 — Thiết kế

Đã hoàn tất trước khi có file này: các tài liệu trong `docs/` và khung thư mục trống. Không ghi lại chi tiết ở đây.

---

## Giai đoạn 1 — Chuẩn bị dữ liệu & môi trường

### Nhóm A — File cấu hình (xong, tick trong checklist)

**Đã làm:**
- Tạo `configs/seeds.yaml`: `base_seed: 42` và `runs: {1: 42, 2: 43, 3: 44}`.
- Tạo `configs/pollution_levels.yaml`: `levels: [0.0, 0.2, 0.5, 0.8]`, `scenario: 3`.
- Tạo `configs/experiment_matrix.yaml`: 3 tác vụ, mỗi tác vụ có dataset, metric, danh sách dimension, danh sách thuật toán, kiểu split.
- Sửa `requirements.txt`: bỏ comment `pyspark`, `shap`, `xgboost`, thêm `pyyaml`, `pyarrow`.

**Làm như thế nào:**
- Đọc `docs/thiet-ke-thi-nghiem-chi-tiet.md` mục 3 để lấy đúng tham số (không tự nghĩ ra số).
- Đối chiếu số lượt: classification 5 dim × 4 mức × 2 thuật toán × 3 run = 120; regression 4 dim = 96; clustering 4 dim = 96; tổng 312.
- Clustering để `split: null` vì không tách train/test; Target Accuracy bị loại khỏi clustering theo mục 2.3 của file thiết kế.
- Thêm `pyyaml` (đọc config) và `pyarrow` (đọc/ghi parquet), vì hai việc này cần chúng.
- Phát hiện checklist ghi sai tham chiếu "mục 5" (ma trận thực ra ở mục 3) nên sửa lại.

**Chưa làm:** cài `requirements.txt` (cần xác nhận; không cần ở local vì mọi thứ chạy trên Colab) → mục này chưa tick.

### Nhóm B + C — Tải dữ liệu thô trên Colab (xong, tick trong checklist)

**Đã làm:** thêm mục 7 vào `notebooks/data_quality_project_colab_setup.ipynb` (3 cell), bạn chạy trên Colab, kết quả lưu trên Google Drive (`MyDrive/data-quality-project/data/raw/`).

**Làm như thế nào:**
- **Beijing (7.1):** tải zip từ UCI (id 501), giải nén, xử lý trường hợp zip lồng zip, lấy 12 file `PRSA_Data_*.csv`, có `assert` đúng 12 file rồi copy vào `data/raw/beijing_air_quality/`.
- **Covertype (7.2):** `sklearn.datasets.fetch_covtype(as_frame=True)`, lưu thành `covertype_raw.parquet`.
- **HIGGS mẫu (7.3):** tải `HIGGS.csv.gz` (~2,8GB) về ổ tạm `/content` (không lưu lên Drive), đọc bằng `float32` để tiết kiệm RAM, lấy mẫu stratified theo target bằng `groupby('target').sample(frac, random_state=seed)` với `seed = seeds['base_seed']` đọc từ `seeds.yaml`, lưu 1.000.000 dòng.
- Mọi cell bỏ qua nếu file đã tồn tại → chạy lại không tải lặp.

**Kiểm chứng (từ PDF kết quả Colab):** Beijing 12 file × (35.064 × 18); Covertype 581.012 × 55, 7 lớp; HIGGS mẫu 1.000.000 × 29, tỷ lệ nhãn 52,99% / 47,01%.

**Sự cố trong lúc làm:** một lần dùng `NotebookEdit` thiếu `edit_mode: insert` nên ghi đè nhầm cell Beijing; đã khôi phục ngay và kiểm tra lại các cell trước khi bàn giao. Lần chèn cell 8.2 cũng bị sai thứ tự và đã xóa, chèn lại đúng chỗ.

**Phát hiện:** tên cột Beijing thực tế khác tài liệu thiết kế (`wd` thay cho `cbwd`; `WSPM`, `RAIN` thay cho `Iws`, `Is`, `Ir`; có thêm `PM10, SO2, NO2, CO, O3`, `year/month/day/hour`, `No`; cột `station` đã có sẵn). Đã sửa `thiet-ke-thi-nghiem-chi-tiet.md` mục 1.1.

### Nhóm D — Gộp Beijing và tạo bản sạch (code đã viết, **chưa chạy** → chưa tick)

**Đã làm:** thêm mục 8 vào notebook Colab (cell 8.1 và 8.2), khi bạn chạy sẽ ghi các file `data/processed/*.parquet` trên Drive.

**Quyết định thiết kế (ảnh hưởng toàn bộ 312 lượt, nên ghi rõ lý do):**

| Quyết định | Lý do |
|---|---|
| Beijing: **bỏ mọi dòng có missing** thay vì impute | Baseline có Completeness đúng bằng 1, không lẫn missing tự nhiên với missing do mình tiêm; impute sẽ tạo giá trị giả làm méo Feature Accuracy |
| Bỏ cột `No` và `year` | `No` chỉ là số thứ tự; `year` chỉ có 2013–2017, không phải đặc trưng đo chất lượng và dễ gây rò rỉ theo thời gian. Giữ `month, day, hour` |
| One-hot `wd` và `station` (`int8`) | Theo thiết kế mục 1.1; dùng `int8` cho nhẹ |
| Đặt target `PM2.5` ở cột đầu | Dễ tách target thống nhất giữa các module |
| HIGGS mẫu, Covertype: chỉ kiểm tra rồi lưu | Không có missing gốc (cell có `assert`); mẫu HIGGS đã stratified từ bước 7.3 |

**Làm như thế nào:** `pd.concat` 12 file → in số ô thiếu theo cột và số dòng bị bỏ → `dropna` → `drop(['No','year'])` → `get_dummies` → lưu parquet → đọc lại in shape, số NaN, số dòng trùng.

**Việc cần làm tiếp:** chạy cell 8.1 và 8.2 trên Colab, gửi kết quả in ra để kiểm tra, sau đó tick 4 mục (gộp Beijing + 3 bản sạch).
