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

---

## Giai đoạn 3 — Module Pollution (xong, đã kiểm chứng trên Colab ngày 2026-10-09)

**Đã làm:**

| File | Nội dung |
|---|---|
| `src/pollution/completeness.py` | MCAR: mỗi feature chọn độc lập tỷ lệ λ số dòng để làm thiếu |
| `src/pollution/feature_accuracy.py` | Nhiễu Gaussian cho feature số; đổi giá trị cho feature one-hot |
| `src/pollution/target_accuracy.py` | Nhiễu Gaussian (target số) hoặc đổi nhãn (target phân loại) |
| `src/pollution/uniqueness.py` | Nhân bản dòng theo hệ số `ρ = 1 / (1 − λ)` |
| `src/pollution/class_balance.py` | Giữ lớp lớn nhất, xóa tỷ lệ λ số dòng của các lớp còn lại |
| `src/pollution/_helpers.py`, `__init__.py` | Hàm dùng chung; bảng `POLLUTERS` tra theo tên dimension |
| `tests/test_pollution.py` | 31 test (cùng 12 test profiling là 43) |
| Notebook mục 10 | 10.1 chạy toàn bộ test, 10.2 làm bẩn dữ liệu thật rồi đo lại bằng profiler |

**Làm như thế nào:**
- Mọi polluter cùng chữ ký `pollute(df, level, seed, target, target_type, onehot_prefixes)`, nên Experiment Runner chỉ cần tra `POLLUTERS[tên]`. Seed là tham số, nơi gọi đọc từ `configs/seeds.yaml`.
- Ba ràng buộc do Giai đoạn 2 đặt ra đều được giữ: nhiễu số nhân với `mean(|gt|)`; ô thiếu là NaN hoặc cả nhóm one-hot về 0; bản sao giữ index của dòng gốc.
- Đổi nhãn / đổi giá trị one-hot dùng phép cộng vòng `(vị trí cũ + số ngẫu nhiên từ 1 đến m−1) mod m`, bảo đảm giá trị mới luôn khác giá trị cũ, nên tỷ lệ sai đúng bằng λ.
- Uniqueness: với target phân loại, nhân bản riêng trong từng lớp với cùng hệ số, để dimension này không vô tình đổi Class Balance.
- **Dùng profiler của Giai đoạn 2 làm thước đo trong test**: làm bẩn ở mức λ rồi kiểm tra điểm đo được khớp giá trị lý thuyết (Completeness = 1 − λ; Target Accuracy phân loại = 1 − λ; accuracy số ≈ 1 − √λ · √(2/π); Uniqueness ≈ 1 − λ). Cách này kiểm tra luôn hai công thức accuracy trên dữ liệu bị bẩn thật — phần còn thiếu của Giai đoạn 2.

**Quyết định thiết kế:**

| Quyết định | Lý do |
|---|---|
| Phương sai nhiễu `σ² = λ` | Bạn chọn phương án A, giữ đúng bài gốc |
| `ρ = 1 / (1 − λ)` | Để Uniqueness ≈ 1 − λ, cùng thang λ với các dimension khác; khớp ghi chú "cắt ở ρ = 5" của thiết kế (λ = 0,8) |
| Class Balance: giữ lớp lớn nhất, xóa ở các lớp còn lại | Theo thiết kế ("loại bớt dòng ở lớp thiểu số"); λ = 0 giữ nguyên dữ liệu gốc. Khác cách "bậc thang" của bài gốc — đã ghi vào `khac-biet-so-voi-bai-goc.md` |
| Tham số phân phối nhân bản (mean 1, std 5) là đối số có giá trị mặc định | Là hằng số định nghĩa polluter, không phải tham số thí nghiệm |

**Kết quả kiểm chứng (Colab, commit `bbd1721`):**
- Cell 10.1: `43 passed in 44.85s` (31 test pollution + 12 test profiling).
- Cell 10.2: đủ 52 dòng (13 cặp dataset × dimension, 4 mức λ), mẫu 100.000 dòng mỗi dataset, seed của run 1. Điểm của dimension bị làm bẩn ở λ = 0 / 0,2 / 0,5 / 0,8:

| Dimension | HIGGS mẫu | Beijing | Covertype | Lý thuyết |
|---|---|---|---|---|
| Completeness | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | 1 − λ |
| Feature Accuracy | 1 / 0,6433 / 0,4360 / 0,2866 | 1 / 0,7216 / 0,4680 / 0,2433 | 1 / 0,7216 / 0,4679 / 0,2431 | Số: 1 − √λ·√(2/π) = 0,643 / 0,436 / 0,286. Có one-hot: trung bình với 1 − λ = 0,722 / 0,468 / 0,243 |
| Target Accuracy | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,6425 / 0,4347 / 0,2850 | không áp dụng | Phân loại: 1 − λ. Số: như trên |
| Uniqueness | 0,9998 / 0,7998 / 0,4999 / 0,2000 | 1 / 0,8 / 0,5 / 0,2 | 1 / 0,8 / 0,5 / 0,2 | ≈ 1 − λ |
| Class Balance | 0,8885 / 0,7108 / 0,4443 / 0,1777 | không áp dụng | 0,2541 / 0,3033 / 0,3771 / 0,4508 | 2 lớp: baseline × (1 − λ) |

- Số dòng sau khi làm bẩn: Uniqueness 100.000 → 125.000 / 200.000 / 500.000 (đúng `ρ = 1/(1 − λ)`); Class Balance HIGGS → 90.590 / 76.476 / 62.361, Covertype → 89.732 / 74.330 / 58.929; ba dimension còn lại giữ 100.000 dòng.
- Hai công thức accuracy trên dữ liệu thật bị bẩn (phần còn thiếu của Giai đoạn 2) khớp lý thuyết tới 3 chữ số thập phân.

**Tác dụng phụ chéo (đọc từ bảng đầy đủ 6 cột điểm; bản PDF đầu tiên bị cắt hai cột `uniqueness` và `class_balance`, người dùng đã gửi lại ảnh chụp output đủ cột):**

- Các cột Completeness, Feature Accuracy, Target Accuracy chỉ đổi khi chính dimension đó bị làm bẩn; ở mọi dòng khác đều bằng 1,0000.
- Polluter Uniqueness **giữ nguyên Class Balance** trên dữ liệu thật (HIGGS 0,8885 và Covertype 0,2541 ở cả 4 mức) — xác nhận thiết kế nhân bản riêng trong từng lớp.
- Polluter Class Balance không làm đổi Uniqueness (HIGGS 0,9998 → 1,0000; Covertype giữ 1,0000).
- Có **hai tác dụng phụ thật**, đều là hệ quả tất yếu của cách làm bẩn, không phải lỗi:

| Polluter | Cột bị kéo theo | Số đo (λ = 0 / 0,2 / 0,5 / 0,8) | Giải thích |
|---|---|---|---|
| Completeness | Uniqueness | HIGGS 0,9998 / 1,0000 / 1,0000 / 0,9959; Beijing 1 / 1 / 1 / 0,9476; Covertype 1 / 1 / 0,9993 / **0,7800** | Ở λ cao, nhiều dòng mất gần hết feature nên trùng nhau. Covertype nặng nhất vì chỉ có 12 feature (10 số + 2 nhóm one-hot): xác suất một dòng mất cả 12 là 0,8¹² ≈ 6,9%, cộng thêm các dòng chỉ còn một nhóm one-hot ít giá trị |
| Target Accuracy (đổi nhãn, HIGGS) | Class Balance | 0,8885 / 0,9321 / 0,9975 / 0,9309 | Đổi nhãn ngẫu nhiên trộn hai lớp về phía 50/50; λ = 0,5 cho cân bằng gần hoàn hảo, λ = 0,8 đối xứng với λ = 0,2. Tính tay từ tỷ lệ 53/47 ra 0,932 / 1,000 / 0,932 — khớp |

Hệ quả cho các giai đoạn sau: meta-model nhận cả vector profile làm đầu vào nên vẫn thấy các thay đổi kéo theo này; nhưng khi diễn giải SHAP theo từng dimension (Giai đoạn 6) cần nhớ ở λ = 0,8 lượt chạy "Completeness" trên Covertype cũng mang tín hiệu Uniqueness.

**Dự đoán đã được xác nhận — Class Balance trên Covertype đi ngược:** điểm Balance **tăng** 0,254 → 0,303 / 0,377 / 0,451 (tính tay trước đó: 0,31 / 0,38 / 0,45), vì công thức `ε` coi "một nửa số lớp đầy, một nửa rỗng" là xấu nhất, còn polluter đẩy dữ liệu về "một lớp lớn, các lớp còn lại nhỏ đều". Trên HIGGS (2 lớp) điểm giảm đúng hướng. **Quyết định (người dùng chốt ngày 2026-10-09):** giữ 24 lượt Class Balance × clustering khi chạy thí nghiệm (vẫn 312 lượt); đến Giai đoạn 7 mới quyết định có đưa vào bảng so sánh baseline không. Lý do: chưa biết ΔAMI của 24 lượt này — nếu gần 0 thì có bằng chứng đo được để loại, nếu rõ rệt thì là một phát hiện; lọc 24 dòng về sau rất dễ, còn chạy bổ sung phải mở lại Colab. Các phương án đã cân nhắc và không chọn: bỏ hẳn (288 lượt); sửa polluter chỉ xóa ở lớp nhỏ (điểm chỉ giảm khoảng 0,03 — ước lượng, chưa đo); làm bậc thang đúng bài gốc (phải bỏ khoảng 97% Covertype); đổi công thức sang entropy (lệch toán so với bài gốc). Lưu ý khi so sánh: ba baseline coi điểm cao là dữ liệu tốt, nên 24 lượt này bất lợi cho chúng.

**Điều cần theo dõi ở giai đoạn sau:**
- Uniqueness ở λ = 0,8 làm dữ liệu lớn gấp 5 (HIGGS train 800K → 4 triệu dòng): thời gian huấn luyện ở Giai đoạn 5 sẽ dài hơn ước tính trong tài liệu thiết kế.
- Đổi 80% nhãn nhị phân của HIGGS là đảo nhãn, F1 có thể tăng lại so với mức 0,5.

---

## Giai đoạn 4 — Module ML downstream (3 module xong và đã kiểm chứng; baseline toàn bộ dữ liệu chưa chạy)

**Đã làm:**

| File | Nội dung |
|---|---|
| `src/downstream/classification.py` | Logistic Regression + Random Forest, trả về F1-macro trên test |
| `src/downstream/regression.py` | Ridge + Gradient Boosting, trả về R² trên test |
| `src/downstream/clustering.py` | k-Means + Gaussian Mixture, trả về AMI so với target |
| `src/downstream/common.py` | Tiền xử lý dùng chung, `fit_and_score`, `split_train_test` |
| `src/downstream/__init__.py` | Bảng `EVALUATORS` tra theo tên task |
| `configs/algorithms.yaml` | Siêu tham số của 6 thuật toán |
| `tests/test_downstream.py` | 50 test (cùng 43 test cũ là 93) |
| Notebook mục 11 | 11.1 chạy toàn bộ test; 11.2 baseline trên mẫu 100.000 dòng, đo thời gian; 11.3 baseline toàn bộ dữ liệu, 3 seed, lưu `results/baseline_performance.csv` |

**Làm như thế nào:**
- Ba tác vụ cùng chữ ký `evaluate(algorithm, train, test, target, seed, params)`, nên Experiment Runner chỉ cần tra `EVALUATORS[tên task]` — cùng kiểu với `POLLUTERS` của Giai đoạn 3. Clustering không tách train/test nên bỏ qua tham số `test`.
- Seed và siêu tham số đều là tham số của hàm; nơi gọi đọc từ `configs/seeds.yaml` và `configs/algorithms.yaml`. `random_state` của mô hình = seed của lượt chạy.
- X, y lấy ra bằng `to_numpy()`, không dựa vào index — vì polluter Uniqueness giữ index của dòng gốc nên index bị trùng.
- Mọi thuật toán đi qua cùng một pipeline, học trên tập train: điền ô thiếu bằng trung vị → chuẩn hóa z-score → mô hình. Nhóm one-hot bị làm thiếu (cả nhóm = 0) giữ nguyên.
- Clustering: số cụm = số lớp thật của target (7 với Covertype); target không vào mô hình, chỉ dùng để tính AMI.
- Chia train/test: stratified 80/20 bằng `base_seed`; target số (PM2.5) chia thành 10 khoảng phân vị để stratify.

**Quyết định thiết kế (tự chốt, bài gốc và tài liệu thiết kế không nêu — đã ghi vào `khac-biet-so-voi-bai-goc.md` mục 4.1):**

| Quyết định | Lý do |
|---|---|
| Điền ô thiếu bằng trung vị của tập train | Polluter Completeness tạo NaN (đã chốt ở Giai đoạn 3) mà scikit-learn không nhận NaN với 5/6 thuật toán; trung vị là cách điền đơn giản, không nhạy với ngoại lệ |
| Chuẩn hóa z-score cho cả 6 thuật toán | Logistic Regression, Ridge, k-Means, GMM nhạy với thang đo; mô hình cây không bị ảnh hưởng nên dùng chung một pipeline cho gọn |
| Siêu tham số đặt trong `configs/algorithms.yaml` | Nhiều khả năng phải chỉnh theo thời gian chạy thực tế; theo quy tắc không hardcode tham số thí nghiệm |
| Baseline chạy với 3 seed của `runs` (18 lượt) | Theo `thiet-ke-thi-nghiem-chi-tiet.md` mục 3; cho biết độ dao động do seed (mức nhiễu nền của ΔPerformance). `pipeline-thi-nghiem.md` ghi "chạy 1 lần" — hai tài liệu thiết kế lệch nhau ở điểm này |

**Test kiểm tra gì:** học được dữ liệu dễ (điểm > 0,9 cho cả 6 thuật toán); target không lọt vào feature (target độc lập với feature thì điểm ở mức ngẫu nhiên); cùng seed cho cùng điểm; chạy được trên dữ liệu bị làm bẩn ở mức cao nhất của mọi dimension trong ma trận (26 tổ hợp); Completeness cao làm điểm giảm; chia train/test đúng tỷ lệ, không trùng dòng, tái lập được; mọi thuật toán trong ma trận đều có code và có siêu tham số.

**Điều cần theo dõi khi có kết quả 11.2:**
- **Thời gian chạy** là rủi ro chính. Random Forest 100 cây không giới hạn độ sâu trên 800.000 dòng HIGGS, Gradient Boosting của scikit-learn (chạy 1 nhân) trên 305.000 dòng Beijing, và GMM `full` trên 581.000 dòng × 54 cột đều có thể mất nhiều phút mỗi lượt — ước tính 10–40 giây/lượt trong tài liệu thiết kế nhiều khả năng quá lạc quan. Giai đoạn 5 có 312 lượt, riêng Uniqueness ở λ = 0,8 làm dữ liệu lớn gấp 5. Từ thời gian đo trên mẫu 100.000 dòng sẽ ngoại suy và quyết định có cần giới hạn mô hình (vd `max_depth`, `max_samples`, GMM `diag`) hoặc giảm cỡ mẫu không.
- **AMI baseline của Covertype** có thể thấp (k-Means/GMM thường không tách tốt 7 loại rừng); nếu thấp thì ΔAMI có ít dư địa.

**Kết quả kiểm chứng (Colab, commit `eca41d4`, ngày 2026-10-09):**
- Cell 11.1: `93 passed in 52.67s`.
- Cell 11.2: baseline trên mẫu 100.000 dòng mỗi dataset, 1 seed (run 1):

| Dataset | Thuật toán | Số dòng train | Thước đo | Điểm | Thời gian (giây) |
|---|---|---|---|---|---|
| HIGGS mẫu | Logistic Regression | 80.000 | F1-macro | 0,6297 | 0,7 |
| HIGGS mẫu | Random Forest | 80.000 | F1-macro | 0,7197 | 66,2 |
| Beijing | Ridge | 80.000 | R² | 0,8475 | 1,0 |
| Beijing | Gradient Boosting | 80.000 | R² | 0,9090 | 24,4 |
| Covertype | k-Means | 100.000 | AMI | 0,1604 | 7,6 |
| Covertype | GMM | 100.000 | AMI | 0,1735 | 18,1 |

- Điểm hợp lý: mô hình cây hơn mô hình tuyến tính ở cả classification và regression; HIGGS vốn là bài toán khó (F1 khoảng 0,63–0,72).
- **AMI của Covertype thấp (0,16–0,17)** như đã lường: k-Means/GMM không tách tốt 7 loại rừng. ΔAMI tối đa chỉ khoảng 0,17, nhỏ hơn nhiều so với ΔF1 và ΔR² — cần lưu ý khi meta-model học chung ba tác vụ (Giai đoạn 6).

**Ngoại suy thời gian cho Giai đoạn 5 (ước lượng, chưa đo):** giả định thời gian tăng tuyến tính theo số dòng (Random Forest tăng nhanh hơn một chút, ~n·log n), và tính cả việc Uniqueness làm dữ liệu lớn gấp 1,25 / 2 / 5 lần.

| Thuật toán | Một lượt trên toàn bộ dữ liệu | Tổng Giai đoạn 5 |
|---|---|---|
| Random Forest (800.000 dòng) | ~13 phút | ~16 giờ |
| Gradient Boosting (305.000 dòng) | ~1,5 phút | ~1,7 giờ |
| GMM (581.000 dòng) | ~1,8 phút | ~1,8 giờ |
| k-Means (581.000 dòng) | ~45 giây | ~45 phút |
| Logistic Regression, Ridge | vài giây | không đáng kể |

Tổng khoảng 20 giờ, trong đó Random Forest chiếm khoảng 16 giờ — không khả thi trên Colab. Dữ liệu bị nhiễu nhãn còn làm cây sâu hơn, nên con số thật có thể cao hơn. 
**Quyết định (người dùng chốt ngày 2026-10-09):** thêm `max_samples: 0.1` cho Random Forest trong `configs/algorithms.yaml` — mỗi cây học trên 10% số dòng. Ước tính Random Forest giảm từ ~16 giờ xuống ~1,5 giờ, cả Giai đoạn 5 còn khoảng 6–7 giờ (ước lượng, chưa đo; các lượt Uniqueness λ = 0,8 vẫn chậm vì `max_samples` là tỷ lệ). Phương án không chọn: giảm mẫu HIGGS xuống 200.000 dòng (Random Forest vẫn ~3 giờ, mất yếu tố quy mô lớn). Bất lợi đã biết ghi ở `khac-biet-so-voi-bai-goc.md` mục 4.1. Kèm theo: cell 11.4 chạy một lượt Random Forest không lấy mẫu con trên HIGGS sạch để đo chênh lệch F1 của baseline.

**Chưa làm:** cell 11.3 (baseline toàn bộ dữ liệu, 3 seed, lưu `results/baseline_performance.csv`) và cell 11.4 (đối chứng). Test chưa chạy lại sau khi đổi config (test đọc siêu tham số từ `algorithms.yaml`).
