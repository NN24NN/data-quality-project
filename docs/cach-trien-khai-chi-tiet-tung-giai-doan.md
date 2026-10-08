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

### Nhóm D — Gộp Beijing và tạo bản sạch (xong, đã chạy trên Colab và kiểm tra, tick trong checklist)

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

**Kết quả kiểm chứng (từ PDF Colab):**

| File | Shape | Ghi chú |
|---|---|---|
| `beijing_clean.parquet` | 382.168 × 42 | Gộp ra 420.768 × 18; bỏ 38.600 dòng (9,2%); NaN = 0; dòng trùng = 0 |
| `higgs_sample_clean.parquet` | 1.000.000 × 29 | Không missing; **2.281 dòng trùng (0,23%)** |
| `covertype_clean.parquet` | 581.012 × 55 | Không missing; dòng trùng = 0 |

Số ô thiếu gốc của Beijing nhiều nhất ở `CO` (20.701), `O3` (13.277), `NO2` (12.116), `SO2` (9.021), `PM2.5` (8.739).

**Lưu ý cho các giai đoạn sau:**
- Beijing sau làm sạch còn 382.168 dòng, thấp hơn "~420K" trong `thiet-ke-thi-nghiem-chi-tiet.md`; số dòng thật phải theo `beijing_clean.parquet`.
- HIGGS mẫu có 0,23% dòng trùng nên Uniqueness của baseline hơi nhỏ hơn 1 (gần 1). Giữ nguyên, không xóa, để bản sạch phản ánh đúng dữ liệu gốc; khi viết profiler và đo ΔPerformance cần tính theo baseline thật chứ không giả định bằng 1.
- Beijing bỏ dòng missing có thể lệch phân phối nhẹ (ví dụ mùa nhiều missing); chấp nhận được vì mọi mức ô nhiễm đều xuất phát từ cùng bản sạch này.

**Kết luận Giai đoạn 1:** các mục dữ liệu và config đã xong. Còn lại HIGGS full (làm sau, trước Giai đoạn 8) và cài `requirements.txt` ở local (bỏ qua được).

### Đồng bộ code qua GitHub (hạ tầng, không nằm trong checklist)

- Tạo `.gitignore` (chặn dữ liệu trong `data/`, file kết quả nặng), `git init`, commit, gắn remote `https://github.com/NN24NN/data-quality-project.git` (repo Private), push sau khi được xác nhận.
- Colab mở notebook thẳng từ tab GitHub; code trong `src/` được kéo về `/content` bằng cell 9.1 (cần token `GITHUB_TOKEN` trong Colab Secrets).
- Sự cố: một lần push bị từ chối vì Colab đã tự lưu notebook lên repo (commit "Created using Colab"); xử lý bằng `git pull --rebase` rồi push lại, không mất gì.

---

## Giai đoạn 2 — Module Profiling (xong, đã chạy trên Colab và kiểm tra, tick trong checklist)

Máy Windows không có Python, nên toàn bộ test chạy trên Colab (cell 9.2, 9.3).

**Đã làm:**

| File | Nội dung |
|---|---|
| `configs/datasets.yaml` | Cột target, loại target, tiền tố nhóm one-hot của 3 bản sạch (không hardcode trong code) |
| `src/profiling/common.py` | 5 công thức quy đổi thống kê thô → điểm, dùng chung cho hai engine |
| `src/profiling/pandas_profiler.py` | `compute_profile(...)` bản pandas |
| `src/profiling/spark_profiler.py` | `compute_profile(...)` bản PySpark |
| `tests/test_profiling.py`, `tests/conftest.py` | 12 test: công thức thuần, ví dụ có đáp án tính tay, đối chiếu pandas/PySpark |
| Notebook mục 9 | 9.1 kéo code từ GitHub, 9.2 chạy pytest, 9.3 profile dữ liệu thật |

**Làm như thế nào:**
- **Tra lại công thức gốc** trong bài Mohammed et al. (arXiv 2207.14529v6) trước khi viết, vì tài liệu thiết kế chỉ ghi rút gọn. Nhờ đó xác định được: `ImBalance` là tổng chênh lệch mọi cặp lớp, `ε = ⌈m/2⌉⌊m/2⌋·n_cmax`; điểm Feature Accuracy của dataset là trung bình của hai trung bình (numeric, categorical).
- **Tách "thống kê thô" khỏi "công thức":** mỗi engine chỉ đếm (số ô thiếu, tổng khoảng cách, số dòng phân biệt, số dòng mỗi lớp), còn phép tính ra điểm nằm duy nhất trong `common.py`. Đây là cách bảo đảm quy tắc "hai bản cùng công thức toán học" bằng cấu trúc code chứ không phải bằng lời hứa.
- **Ghép dòng bẩn với dòng sạch** qua index (pandas) hoặc cột id (PySpark), để Feature/Target Accuracy vẫn tính đúng khi polluter nhân bản hoặc xóa dòng.
- Bản Spark gộp các phép đếm vào ít lần quét (1 `agg` cho completeness, 1 `agg` sau join cho accuracy), bọc backtick cho tên cột có dấu chấm (`PM2.5`), và coi cả null lẫn NaN là ô thiếu (NaN của pandas sang Spark không thành null).

**Quyết định thiết kế:**

| Quyết định | Lý do |
|---|---|
| Mẫu số `nFAcc` là `mean(\|gt\|)`, cắt dưới tại 0 (khác bài gốc dùng `mean_gt`) | Cột có trung bình ≈ 0 hoặc âm (eta/phi của HIGGS, `DEWP` của Beijing) làm công thức gốc chia cho ≈ 0, điểm vọt ra ngoài [0, 1]. Với cột dương hai cách trùng nhau |
| Nhóm cột one-hot = 1 feature categorical | Đúng ngữ nghĩa (hướng gió là 1 feature, không phải 16); thiếu = cả nhóm bằng 0, sai = khác bản sạch ở bất kỳ bit nào |
| Ô đang thiếu không tính vào accuracy | Một lỗi không bị đếm ở cả Completeness lẫn Accuracy, các dimension độc lập với nhau |
| `n_cmax` = kích thước lớp lớn nhất quan sát được | Bài gốc không nói rõ; với 2 lớp cho `Balance = n_min / n_max`, dễ diễn giải |
| Không áp dụng được thì trả NaN (Class Balance với regression; accuracy khi không có bản sạch) | Không bịa điểm 1.0; cách điền giá trị để module meta-model quyết định |

**Hệ quả cho Giai đoạn 3 (polluter):** nhiễu numeric phải nhân với `mean(|gt|)` (không phải `mean_gt`); completeness polluter chèn NaN / đặt cả nhóm one-hot về 0; polluter nhân bản phải giữ index của dòng gốc.

**Kết quả kiểm chứng (từ PDF Colab, code tại commit `43c6ba3`):**

- Cell 9.2: `12 passed in 45.12s`.
- Cell 9.3: pandas và PySpark khớp trên mẫu 50.000 dòng của cả 3 dataset (sai số tương đối ≤ 1e-9).

Profile của bản sạch, tính trên toàn bộ dữ liệu bằng pandas (đây là điểm baseline):

| Dataset | n | Completeness | Feature Acc. | Target Acc. | Uniqueness | Class Balance |
|---|---|---|---|---|---|---|
| Beijing | 382.168 | 1 | 1 | 1 | 1 | NaN (regression) |
| HIGGS mẫu | 1.000.000 | 1 | 1 | 1 | 0,997719 | 0,887077 |
| Covertype | 581.012 | 1 | 1 | 1 | 1 | 0,255949 |

Ba con số khác 1 đã được đối chiếu tay với số liệu Giai đoạn 1:
- Uniqueness HIGGS: 2.281 dòng trùng → 997.719 dòng phân biệt → (997.719 − 1) / 999.999 = 0,997719.
- Class Balance HIGGS: 2 lớp nên bằng n_min / n_max = 0,47008 / 0,52992 = 0,8871.
- Class Balance Covertype: tổng chênh lệch cặp của 7 lớp = 2.529.486; ε = 4 × 3 × 283.301 = 3.399.612 → 1 − 0,74405 = 0,25595.

**Sự cố:** cell 9.3 lần đầu báo `UNABLE_TO_INFER_SCHEMA` vì file tạm đặt tên bắt đầu bằng `_` (Spark coi là file ẩn). Lỗi nằm ở cell notebook, không ở module; đã đổi tên file.

**Giới hạn của phép kiểm chứng:** trên dữ liệu thật mới so bản sạch với chính nó, nên Feature/Target Accuracy hiển nhiên bằng 1. Hai công thức accuracy khi dữ liệu thật sự bị bẩn mới chỉ được kiểm bằng ví dụ tính tay và dữ liệu ngẫu nhiên trong test; sẽ kiểm lại trên dữ liệu thật khi có polluter ở Giai đoạn 3.

**Lưu ý cho giai đoạn sau:** Class Balance baseline của Covertype chỉ 0,256 (mất cân bằng tự nhiên), nên polluter Class Balance trên Covertype có ít dư địa để làm xấu thêm.
