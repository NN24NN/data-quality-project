# Thay đổi và khác biệt so với bài báo gốc

**Mục đích file này:** ghi lại mọi điểm đề tài làm khác hai bài báo gốc, kèm lý do — làm cơ sở cho slide báo cáo giữa kỳ và cuối kỳ. Cập nhật mỗi khi phát sinh thêm điểm khác biệt (quy tắc trong `CLAUDE.md`).

**Hai bài gốc:** Mohammed et al. (2025) — bài nền tảng chính; DQSOps (Bayram et al., 2023) — baseline. Tóm tắt từng bài: `tong-quan-bai-goc.md`.

**Cách đọc cột "Trạng thái":** *Đã cài* = có code và đã kiểm chứng; *Đã viết* = có code, chưa chạy test; *Thiết kế* = mới chốt trên giấy, chưa có code.

---

## 1. Khác biệt về mục tiêu (đóng góp của đề tài)

| | Mohammed et al. | DQSOps | Đề tài |
|---|---|---|---|
| Loại nghiên cứu | Thực nghiệm giải thích: vẽ đường cong hiệu năng theo mức ô nhiễm | Xây mô hình dự đoán quality score | Xây mô hình dự đoán **mức suy giảm hiệu năng downstream** |
| Nhãn để học | Không có (không huấn luyện mô hình dự đoán) | Quality score tự định nghĩa (PCA composite) | ΔF1, ΔR², ΔAMI so với baseline sạch |
| Phân biệt theo tác vụ | Quan sát được, không mô hình hóa | Không (1 điểm cho mọi mục đích) | Có — meta-model nhận loại tác vụ và họ thuật toán làm đầu vào |
| Trọng số từng dimension | Không có | Ẩn trong PCA | Tường minh qua SHAP, tách theo tác vụ |
| Quy mô dữ liệu | 1.000–25.000 dòng | 1 use case công nghiệp, time-series | 382K–1M dòng, và 11M dòng cho phần Spark |

*Trạng thái: Thiết kế (meta-model, SHAP, Spark scalability thuộc Giai đoạn 6–8).*

---

## 2. Khác biệt về phạm vi thí nghiệm (so với Mohammed et al.)

| Thành phần | Bài gốc | Đề tài | Lý do |
|---|---|---|---|
| Dataset | 10 bộ: Credit, Contraceptive, Telco, COVID, Houses, IMDB, Cars, Bank, Covertype, Letter | 3 bộ: HIGGS (classification), Beijing Air Quality (regression), Covertype (clustering) | Cần dữ liệu đủ lớn cho phần scalability; chỉ **Covertype** trùng với bài gốc |
| Dimension | 6 | 4–5: bỏ **Consistent Representation** | Thu gọn cho đồ án môn học; HIGGS toàn cột số nên dimension này không áp dụng được |
| Thuật toán | 19 (có MLP, TabNet, KNN, SVM...) | 6: LogR + RF, Ridge + GB, k-Means + GMM | Chỉ cần so sánh họ tuyến tính với họ cây; không dùng deep learning |
| Kịch bản | 3 (bẩn train / bẩn test / bẩn cả hai) | Chỉ Scenario 3 (bẩn cả hai) | Tình huống thực tế nhất |
| Số lượt chạy | 4.905 | 312 | Hệ quả của các thu gọn trên |
| Target Accuracy cho clustering | — | Không áp dụng | Clustering không dùng target để huấn luyện |

*Trạng thái: Thiết kế (đã chốt ở Giai đoạn 0, cấu hình trong `configs/experiment_matrix.yaml`).*

---

## 3. Điều chỉnh công thức đo chất lượng

Đây là các điểm lệch về **toán học** — phần cần giải thích kỹ nhất khi bị hỏi.

### 3.1 Feature / Target Accuracy cho cột số — đổi mẫu số

| | Bài gốc | Đề tài |
|---|---|---|
| Công thức | `1 − avg_dist(c) / mean_gt(c)` | `1 − avg_dist(c) / mean(\|gt\|)` |
| Cận dưới | Feature: không có cận dưới. Target: cắt tại 0 | Cắt tại 0 cho cả feature và target |

**Lý do:** công thức gốc chia cho trung bình của cột, tức ngầm giả định cột dương và trung bình xa 0. HIGGS có các cột góc (eta, phi) đối xứng quanh 0; Beijing có `TEMP`, `DEWP` nhận cả giá trị âm lẫn dương. Ví dụ với sai lệch trung bình 0,5 đơn vị:

| Cột | `mean(gt)` | Điểm theo công thức gốc |
|---|---|---|
| Dương, trung bình 50 | 50 | 0,99 — hợp lý |
| Đối xứng quanh 0, trung bình 0,001 | 0,001 | −499 — vô nghĩa |
| Trung bình âm, −2 | −2 | 1,25 — càng bẩn điểm càng cao hơn 1 |

**Điểm cần nói rõ khi trình bày:** với cột toàn giá trị dương, `mean(|gt|) = mean(gt)`, nên trên những cột đó kết quả **trùng với bài gốc**. Đây là mở rộng công thức cho loại cột bài gốc không gặp, không phải thay công thức.

**Chưa kiểm chứng:** nhận định "cột nào trung bình gần 0" dựa trên hiểu biết về hai bộ dữ liệu, chưa đo trên bản sạch. Cần in `mean` và `mean(|·|)` từng cột để có số liệu thật cho slide.

*Trạng thái: Đã cài (`src/profiling/common.py`, hàm `numeric_accuracy`; test 12/12 passed).*

### 3.2 Class Balance — diễn giải `n_cmax`

Công thức giữ nguyên bài gốc: `Balance = 1 − ImBalance / ε`, `ImBalance = Σ_{i<j} |n_i − n_j|`, `ε = ⌈m/2⌉ × ⌊m/2⌋ × n_cmax`.

Bài gốc mô tả `n_cmax` là "số dòng tối đa một lớp có thể có" mà không nêu cách tính. Đề tài lấy **kích thước lớp lớn nhất quan sát được** trong dataset đang đo. Với 2 lớp: `Balance = n_min / n_max`.

Điểm baseline đo được: HIGGS mẫu 0,887; Covertype 0,256 (mất cân bằng tự nhiên).

*Trạng thái: Đã cài (`common.class_balance`).*

### 3.3 Các quy ước bổ sung (bài gốc không nêu, đề tài phải tự chốt)

| Quy ước | Lý do |
|---|---|
| Mỗi **nhóm cột one-hot** tính là 1 feature categorical (thiếu = cả nhóm bằng 0; sai = khác bản sạch ở bất kỳ bit nào) | Dữ liệu sạch đã one-hot sẵn (`wd`, `station`, `Wilderness_Area`, `Soil_Type`); nếu tính từng cột thì hướng gió bị đếm thành 16 feature |
| Ô đang **thiếu** không tính vào accuracy | Một lỗi không bị đếm ở cả Completeness lẫn Accuracy; các dimension độc lập với nhau |
| Dimension **không áp dụng** trả `NaN`, không gán 1.0 | Class Balance không có nghĩa với regression; không bịa điểm |

*Trạng thái: Đã cài.*

### 3.4 Những gì giữ nguyên bài gốc

- Completeness: `1 − (1/f) Σ missing(c_i)`.
- Feature Accuracy cho cột categorical: `1 − mismatches / n`.
- Uniqueness: `(unique_samples − 1) / (n − 1)`.
- Cách gộp Feature Accuracy của dataset: trung bình của (trung bình các feature numeric) và (trung bình các feature categorical).

---

## 4. Khác biệt về cách làm bẩn dữ liệu (polluter)

*Trạng thái cả mục: **Đã cài** (`src/pollution/`; test 43/43 passed, đã đo lại bằng profiler trên mẫu 100.000 dòng của cả 3 dataset — số liệu ở `cach-trien-khai-chi-tiet-tung-giai-doan.md`, Giai đoạn 3).*

| Thành phần | Bài gốc | Đề tài | Lý do |
|---|---|---|---|
| Thang nhiễu cho cột số | `noise = X × mean_gt(c)` | `noise = X × mean(\|gt\|)` | Cùng lý do mục 3.1: cột trung bình ≈ 0 thì nhiễu ≈ 0, "làm bẩn" mà dữ liệu không đổi |
| Độ lớn nhiễu | `X ~ N(0, σ²)` với `σ² = λ` | **Giữ đúng bài gốc** (`σ² = λ`) | Đã chốt; tài liệu thiết kế ban đầu ghi nhầm `N(0, λ²)`. Độ lệch chuẩn ở λ = 0,2 / 0,5 / 0,8 là 0,45 / 0,71 / 0,89 |
| Biểu diễn ô thiếu | Placeholder ngoài miền giá trị (vd −1, "empty") | NaN / đặt cả nhóm one-hot về 0; module downstream tự điền trước khi huấn luyện | Profiler nhận diện ô thiếu thống nhất trên cả pandas và Spark |
| Bản ghi trùng có sẵn | Xóa hết trước khi làm bẩn Uniqueness | Giữ nguyên | Bản sạch phản ánh đúng dữ liệu gốc; xem mục 5 |
| Hệ số nhân bản | Tham số `ρ`, Uniqueness = 1/ρ | `ρ = 1 / (1 − λ)`, tức Uniqueness ≈ 1 − λ; λ = 0,8 cho ρ = 5 | Để cả 5 dimension dùng chung thang λ; dữ liệu lớn gấp 5 lần ở mức cao nhất |
| Phân phối số lần nhân bản | Uniform / normal / Zipf | Chỉ normal (mean 1, std 5), tối thiểu 1 bản sao | Thu gọn |
| Class Balance | Xếp kích thước lớp thành bậc thang đều (chênh nhau hằng số Δ), bớt ở lớp nhỏ và thêm vào lớp lớn; λ = 0 là cân bằng hoàn toàn | Giữ nguyên lớp lớn nhất, xóa tỷ lệ λ số dòng của mọi lớp còn lại; λ = 0 là dữ liệu gốc | Cách của bài gốc đòi cân bằng lại dữ liệu ngay ở λ = 0, làm mất baseline thật; với Covertype sẽ phải bỏ phần lớn dữ liệu |

**Hiện tượng cần biết khi trình bày kết quả (không phải lỗi):**
- **Nhãn nhị phân đổi 80%** (Target Accuracy, HIGGS, λ = 0,8): nhãn gần như bị đảo ngược chứ không phải nhiễu hơn. Vì train và test bẩn cùng mức (Scenario 3), mô hình học được quan hệ đảo và F1 có thể **tăng trở lại** so với λ = 0,5 (mức nhiễu tối đa với 2 lớp).
- **Class Balance trên dữ liệu nhiều lớp**: với công thức `ε` của bài gốc, điểm Balance không nhất thiết giảm khi xóa bớt các lớp nhỏ. Đo thật trên Covertype (7 lớp): điểm **tăng** 0,254 → 0,303 / 0,377 / 0,451 khi λ = 0,2 / 0,5 / 0,8; trên HIGGS (2 lớp) điểm giảm đúng hướng 0,889 → 0,711 / 0,444 / 0,178. Nguyên nhân: `ε` coi "một nửa số lớp đầy, một nửa rỗng" là xấu nhất, còn polluter đẩy dữ liệu về "một lớp lớn, các lớp còn lại nhỏ đều". *Đã chốt: vẫn chạy Class Balance cho clustering (312 lượt); đến Giai đoạn 7 mới quyết định có đưa 24 lượt này vào bảng so sánh baseline không.*
- **Làm bẩn một dimension có thể kéo theo điểm của dimension khác** (đo ở cell 10.2): Completeness ở λ = 0,8 làm Uniqueness giảm (Covertype 0,78; Beijing 0,95; HIGGS 0,996) vì nhiều dòng mất gần hết feature nên trùng nhau; đổi nhãn HIGGS làm Class Balance tăng về phía 1 (0,889 → 0,932 / 0,998 / 0,931). Các cặp còn lại không ảnh hưởng nhau.

### 4.1 Xử lý trước khi huấn luyện mô hình downstream

*Trạng thái: **Đã cài** (`src/downstream/common.py`; test 93/93 passed, baseline toàn bộ dữ liệu với 3 seed đã chạy — số liệu ở `cach-trien-khai-chi-tiet-tung-giai-doan.md`, Giai đoạn 4).*

| Thành phần | Bài gốc | Đề tài | Lý do |
|---|---|---|---|
| Ô thiếu khi vào mô hình | Mô hình nhận thẳng giá trị placeholder | Điền bằng **trung vị của tập train**; nhóm one-hot thiếu giữ nguyên là "cả nhóm = 0" | Hệ quả của việc biểu diễn ô thiếu bằng NaN (bảng trên); scikit-learn không nhận NaN |
| Chuẩn hóa feature | Chưa đối chiếu với bài gốc | z-score (học trên tập train) cho cả 6 thuật toán | Mô hình tuyến tính, k-Means, GMM nhạy với thang đo; mô hình cây không bị ảnh hưởng |
| Siêu tham số | Chưa đối chiếu với bài gốc | Mặc định của scikit-learn, trừ `max_iter=1000` (LogR), `n_init=10` (k-Means); liệt kê trong `configs/algorithms.yaml` | Có thể phải chỉnh theo thời gian chạy — sẽ cập nhật nếu đổi |
| Random Forest | Chưa đối chiếu với bài gốc | `max_samples = 0.1`: mỗi cây học trên 10% số dòng (bootstrap), 100 cây | Bản mặc định mất ~13 phút/lượt trên 800.000 dòng HIGGS (ngoại suy từ 66 giây trên 80.000 dòng), tức ~16 giờ cho Giai đoạn 5. Người dùng chốt ngày 2026-10-09 |
| Gaussian Mixture | Chưa đối chiếu với bài gốc | `covariance_type = diag`, `n_init = 10` (mặc định của scikit-learn là `full`, 1 lần khởi tạo) | Với 1 lần khởi tạo, AMI trên Covertype sạch dao động 0,10–0,21 theo seed, gần bằng ΔAMI tối đa (~0,19) nên nhãn ΔAMI rất nhiễu. 10 lần khởi tạo giảm độ lệch chuẩn từ 0,045 xuống 0,010. `diag` cho AMI trên dữ liệu sạch gần như trùng `full` (đo ở cell 11.5) mà nhanh gấp 4. Người dùng chốt ngày 2026-10-09 |
| Số cụm (clustering) | — | Bằng số lớp thật của target | Theo tài liệu thiết kế |

**Điểm cần nói rõ khi trình bày:** vì ô thiếu được điền bằng trung vị thay vì placeholder, ảnh hưởng của Completeness lên mô hình có thể khác bài gốc (placeholder ngoài miền giá trị cho mô hình cây một tín hiệu "ô này thiếu", trung vị thì không).

**Giới hạn của việc lấy mẫu con cho Random Forest (cần nêu trong báo cáo):**
- Cây học trên ít dòng hơn thì nông hơn và ít học thuộc nhiễu hơn, nên ΔF1 đo được (nhất là với Target Accuracy) có thể **nhỏ hơn** so với Random Forest mặc định. Kết quả là của "Random Forest có lấy mẫu con". *Chưa đo — đo được thì phải chạy bản mặc định trên dữ liệu bẩn, quay lại bài toán thời gian.*
- Baseline thấp hơn bản mặc định một chút. **Đã đo (cell 11.4, HIGGS sạch, 1 seed): F1 = 0,7250 so với 0,7328 của bản mặc định — chênh 0,0078; thời gian 129 giây so với 851 giây (nhanh gấp 6,6 lần).**
- Năm thuật toán còn lại học trên toàn bộ dữ liệu; riêng Random Forest mỗi cây thấy 10% (cả rừng 100 cây vẫn thấy gần như mọi dòng).
- Giá trị 0,1 chọn theo thời gian chạy, không phải kết quả tinh chỉnh.

**Giới hạn của phần clustering (cần nêu trong báo cáo):**
- AMI baseline trên Covertype thấp (k-Means 0,17; GMM 0,20), nên ΔAMI nhỏ hơn nhiều so với ΔF1 và ΔR².
- `diag` giả định các feature độc lập trong mỗi cụm. Mới so với `full` trên dữ liệu sạch; trên dữ liệu bẩn hai cấu hình có thể phản ứng khác nhau. *Chưa đo.*
- `full` và `diag` (1 lần khởi tạo) cho AMI gần như trùng nhau, gợi ý rằng trên dữ liệu này GMM gần như giữ nguyên cách chia cụm của bước khởi tạo bằng k-Means, tức hai thuật toán clustering có thể hành xử khá giống nhau. *Suy đoán, chưa kiểm chứng.*

---

## 5. Khác biệt về chuẩn bị dữ liệu

| Điểm | Chi tiết | Hệ quả |
|---|---|---|
| Beijing: bỏ dòng thiếu thay vì impute | Bỏ 38.600 / 420.768 dòng (9,2%), còn 382.168 dòng | Baseline có Completeness đúng bằng 1; phân phối có thể lệch nhẹ nếu missing tập trung theo mùa |
| Beijing: bỏ cột `No`, `year` | `No` là số thứ tự; `year` dễ gây rò rỉ theo thời gian | Giữ `month`, `day`, `hour` |
| HIGGS: lấy mẫu 1.000.000 / 11.000.000 dòng | Stratified theo target, seed 42 | Tỷ lệ nhãn giữ nguyên 52,99% / 47,01% |
| HIGGS mẫu có 2.281 dòng trùng (0,23%) | Giữ nguyên, không xóa | Uniqueness của baseline hơi nhỏ hơn 1; ΔPerformance luôn tính so với baseline thật |
| Covertype mất cân bằng tự nhiên | Lớp nhỏ nhất 2.747 dòng, lớp lớn nhất 283.301 dòng | Class Balance của baseline đã thấp sẵn |

*Trạng thái: Đã cài (đã chạy và kiểm tra trên Colab ở Giai đoạn 1).*

---

## 6. Khác biệt về cài đặt

| Điểm | Bài gốc | Đề tài |
|---|---|---|
| Engine profiling | Một bản (Python) | Hai bản pandas và PySpark, dùng chung một file công thức (`common.py`) |
| Kiểm chứng | — | Test có đáp án tính tay + đối chiếu kết quả pandas với PySpark |

Kết quả: 12/12 test passed; pandas và PySpark khớp trên mẫu 50.000 dòng của cả 3 dataset.

*Trạng thái: Đã cài.*

---

## 7. Gợi ý dùng cho slide

- **Giữa kỳ:** mục 1 (định vị đề tài), mục 2 (phạm vi), mục 5 (dữ liệu đã chuẩn bị), và mục 3.1 như một phát hiện khi mở rộng sang dữ liệu lớn.
- **Cuối kỳ:** thêm mục 4 (sau khi polluter đã chốt) và kết quả thực nghiệm; mục 3 và 4 đưa vào phần "Giới hạn và điều chỉnh so với bài gốc".
- Câu hỏi dễ bị hỏi nhất: *"Vì sao không dùng đúng công thức của bài gốc?"* — trả lời bằng bảng ví dụ ở mục 3.1 và ý "trùng với bài gốc trên cột dương".
