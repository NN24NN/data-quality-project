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

*Trạng thái: Đã viết (`src/profiling/common.py`, hàm `numeric_accuracy`).*

### 3.2 Class Balance — diễn giải `n_cmax`

Công thức giữ nguyên bài gốc: `Balance = 1 − ImBalance / ε`, `ImBalance = Σ_{i<j} |n_i − n_j|`, `ε = ⌈m/2⌉ × ⌊m/2⌋ × n_cmax`.

Bài gốc mô tả `n_cmax` là "số dòng tối đa một lớp có thể có" mà không nêu cách tính. Đề tài lấy **kích thước lớp lớn nhất quan sát được** trong dataset đang đo. Với 2 lớp: `Balance = n_min / n_max`.

*Trạng thái: Đã viết (`common.class_balance`).*

### 3.3 Các quy ước bổ sung (bài gốc không nêu, đề tài phải tự chốt)

| Quy ước | Lý do |
|---|---|
| Mỗi **nhóm cột one-hot** tính là 1 feature categorical (thiếu = cả nhóm bằng 0; sai = khác bản sạch ở bất kỳ bit nào) | Dữ liệu sạch đã one-hot sẵn (`wd`, `station`, `Wilderness_Area`, `Soil_Type`); nếu tính từng cột thì hướng gió bị đếm thành 16 feature |
| Ô đang **thiếu** không tính vào accuracy | Một lỗi không bị đếm ở cả Completeness lẫn Accuracy; các dimension độc lập với nhau |
| Dimension **không áp dụng** trả `NaN`, không gán 1.0 | Class Balance không có nghĩa với regression; không bịa điểm |

*Trạng thái: Đã viết.*

### 3.4 Những gì giữ nguyên bài gốc

- Completeness: `1 − (1/f) Σ missing(c_i)`.
- Feature Accuracy cho cột categorical: `1 − mismatches / n`.
- Uniqueness: `(unique_samples − 1) / (n − 1)`.
- Cách gộp Feature Accuracy của dataset: trung bình của (trung bình các feature numeric) và (trung bình các feature categorical).

---

## 4. Khác biệt về cách làm bẩn dữ liệu (polluter)

*Toàn bộ mục này ở trạng thái **Thiết kế** — module pollution thuộc Giai đoạn 3, chưa có code. Cần rà lại mục này khi viết xong.*

| Thành phần | Bài gốc | Đề tài (dự kiến) | Lý do |
|---|---|---|---|
| Thang nhiễu cho cột số | `noise = X × mean_gt(c)` | `noise = X × mean(\|gt\|)` | Cùng lý do mục 3.1: cột trung bình ≈ 0 thì nhiễu ≈ 0, "làm bẩn" mà dữ liệu không đổi |
| Độ lớn nhiễu | `X ~ N(0, σ²)` với `σ² = λ` | Tài liệu thiết kế ghi `X ~ N(0, λ²)` | **Chưa chốt** — hai cách khác nhau, cần quyết định ở Giai đoạn 3 |
| Biểu diễn ô thiếu | Placeholder ngoài miền giá trị (vd −1, "empty") | NaN / đặt cả nhóm one-hot về 0; module downstream tự điền trước khi huấn luyện | Profiler nhận diện ô thiếu thống nhất trên cả pandas và Spark |
| Bản ghi trùng có sẵn | Xóa hết trước khi làm bẩn Uniqueness | Giữ nguyên | Bản sạch phản ánh đúng dữ liệu gốc; xem mục 5 |
| Phân phối số lần nhân bản | Uniform / normal / Zipf | Chỉ normal | Thu gọn |

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

*Trạng thái: Đã viết, chưa chạy test.*

---

## 7. Gợi ý dùng cho slide

- **Giữa kỳ:** mục 1 (định vị đề tài), mục 2 (phạm vi), mục 5 (dữ liệu đã chuẩn bị), và mục 3.1 như một phát hiện khi mở rộng sang dữ liệu lớn.
- **Cuối kỳ:** thêm mục 4 (sau khi polluter đã chốt) và kết quả thực nghiệm; mục 3 và 4 đưa vào phần "Giới hạn và điều chỉnh so với bài gốc".
- Câu hỏi dễ bị hỏi nhất: *"Vì sao không dùng đúng công thức của bài gốc?"* — trả lời bằng bảng ví dụ ở mục 3.1 và ý "trùng với bài gốc trên cột dương".
