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

**Lợi ích cho báo cáo (điểm cộng):** cho thấy nhóm phát hiện được một điểm yếu của công thức gốc khi đưa sang loại dữ liệu bài gốc chưa gặp (cột đối xứng quanh 0, cột âm), và sửa theo cách **tương thích ngược** (trùng bài gốc trên cột dương). Nếu không sửa, điểm Feature Accuracy của HIGGS sẽ vô nghĩa và toàn bộ meta-model học trên số rác.

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

**Lợi ích cho báo cáo (điểm cộng, chung cho 3.2 và 3.3):** bài gốc để ngỏ các chi tiết này; đề tài chốt tường minh, có lý do và có test đáp án tính tay. Nhờ vậy kết quả **tái lập được**, và các dimension độc lập với nhau (một lỗi không bị đếm hai lần), điều cần thiết khi đưa profile vào meta-model và đọc SHAP theo từng dimension.

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

**Lợi ích cho báo cáo (điểm cộng):**
- **Cả 5 dimension dùng chung một thang λ** (điểm đo được ≈ 1 − λ, đã kiểm chứng bằng profiler), nên so sánh được tác động giữa các dimension ở cùng mức bẩn và meta-model học trên các feature cùng thang.
- **Giữ baseline thật** (không xóa trùng, không cân bằng lại ở λ = 0): Δ đo đúng mức suy giảm so với dữ liệu thực tế, không so với một bản dữ liệu đã bị chỉnh.
- **Đã đo lại liều lượng** bằng profiler (cell 10.2) thay vì tin vào lý thuyết; phát hiện và ghi nhận được hai hiện tượng không hiển nhiên dưới đây, là nội dung thảo luận tốt.

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

**Lợi ích cho báo cáo (điểm cộng):** mọi lựa chọn cấu hình đều **có số đo đi kèm** (Random Forest: F1 chỉ giảm 0,008 mà nhanh gấp 6,6; GMM: độ lệch chuẩn theo seed giảm 0,045 → 0,010). Đây là các đánh đổi có chủ đích và có bằng chứng, nhờ vậy 312 lượt chạy vừa trong ~7 giờ Colab và nhãn ΔAMI đủ ổn định để học.

**Giới hạn của việc lấy mẫu con cho Random Forest (cần nêu trong báo cáo):**
- Cây học trên ít dòng hơn thì nông hơn và ít học thuộc nhiễu hơn, nên ΔF1 đo được (nhất là với Target Accuracy) có thể **nhỏ hơn** so với Random Forest mặc định. Kết quả là của "Random Forest có lấy mẫu con". *Chưa đo — đo được thì phải chạy bản mặc định trên dữ liệu bẩn, quay lại bài toán thời gian.*
- Baseline thấp hơn bản mặc định một chút. **Đã đo (cell 11.4, HIGGS sạch, 1 seed): F1 = 0,7250 so với 0,7328 của bản mặc định — chênh 0,0078; thời gian 129 giây so với 851 giây (nhanh gấp 6,6 lần).**
- Năm thuật toán còn lại học trên toàn bộ dữ liệu; riêng Random Forest mỗi cây thấy 10% (cả rừng 100 cây vẫn thấy gần như mọi dòng).
- Giá trị 0,1 chọn theo thời gian chạy, không phải kết quả tinh chỉnh.

**Giới hạn của phần clustering (cần nêu trong báo cáo):**
- AMI baseline trên Covertype thấp (k-Means 0,17; GMM 0,20), nên ΔAMI nhỏ hơn nhiều so với ΔF1 và ΔR².
- `diag` giả định các feature độc lập trong mỗi cụm. Mới so với `full` trên dữ liệu sạch; trên dữ liệu bẩn hai cấu hình có thể phản ứng khác nhau. *Chưa đo.*
- `full` và `diag` (1 lần khởi tạo) cho AMI gần như trùng nhau, gợi ý rằng trên dữ liệu này GMM gần như giữ nguyên cách chia cụm của bước khởi tạo bằng k-Means, tức hai thuật toán clustering có thể hành xử khá giống nhau. *Suy đoán, chưa kiểm chứng.*

### 4.2 Cách tính ΔPerformance (Experiment Runner)

Bài gốc (Mohammed et al.) không mô hình hóa ΔPerformance, chỉ vẽ đường cong hiệu năng theo mức ô nhiễm, nên đề tài phải tự chốt quy ước. Pseudocode trong `thiet-ke-thi-nghiem-chi-tiet.md` mục 4 ban đầu ghi baseline "chạy 1 lần/thuật toán", nhưng không nói rõ seed nào.

| Thành phần | Đề tài | Lý do |
|---|---|---|
| Baseline để trừ | **Cùng seed** với lượt chạy: `Δ = baseline(thuật toán, seed s) − raw(thuật toán, seed s)`, lấy từ `results/baseline_performance.csv` | Meta-model chỉ thấy vector profile, không thấy seed, nên phần Δ do seed gây ra là nhiễu nó không học được. Ghép theo seed khử phần nhiễu này: ở λ = 0, Δ = 0 tuyệt đối. Nếu lấy trung bình 3 seed, Δ của k-Means ở λ = 0 dao động khoảng ±0,017, đáng kể so với AMI sạch chỉ 0,17. Người dùng chốt ngày 2026-10-23 |
| Phương án không chọn | Trung bình 3 seed | Đơn giản hơn, nhưng nhiễu seed dồn vào nhãn của meta-model. Vẫn tính lại được từ `raw_performance` và `baseline_performance.csv` mà không phải chạy lại |
| Dấu của Δ | `baseline − raw`, nên Δ > 0 nghĩa là hiệu năng giảm | Theo `thiet-ke-thi-nghiem-chi-tiet.md` mục 4 |

**Lợi ích cho báo cáo (điểm cộng):** nhãn sạch hơn cho meta-model (Δ = 0 tuyệt đối ở λ = 0, kiểm chứng max |Δ| = 1,1e-16). Phần nhiễu do seed (mô hình không thể học được) được tách khỏi phần do ô nhiễm, nên sai số còn lại của meta-model phản ánh đúng khả năng dự đoán.

**Giới hạn cần nêu khi trình bày:** seed còn điều khiển cả cách làm bẩn, nên hai lượt cùng seed không khớp hoàn toàn. Ở λ cao, dữ liệu đổi nhiều và cùng seed có thể không còn giữ tương quan, nên lợi thế của cách ghép theo seed giảm dần. Chỉ ảnh hưởng đáng kể tới k-Means, GMM (và rất ít tới Random Forest); bốn thuật toán còn lại có độ lệch theo seed ≈ 0 nên hai cách cho gần như cùng kết quả.

*Trạng thái: **Đã cài** (`src/experiment_runner/run_experiment.py`; 100 passed; đã chạy đủ 312 lượt trên Colab ngày 2026-10-23).*

### 4.3 Cách chia train/test cho meta-model và 3 baseline

Mohammed et al. không huấn luyện mô hình dự đoán nên không có quy ước này. Thiết kế ban đầu của đề tài (`kien-truc-mo-hinh-va-ky-thuat.md` mục 4) ghi "chia ngẫu nhiên 80/20, stratify theo task". Sau khi có bảng 312 dòng thì thấy cách đó bị **rò rỉ dữ liệu**:
- 3 run của cùng (task, thuật toán, dimension, λ) có profile gần như giống nhau, và Δ lệch chuẩn ≤ 0,006 (trừ k-Means). Nếu một run nằm ở train và run khác nằm ở test, mô hình chỉ cần "nhớ".
- Các dòng λ = 0 là bản chép giống hệt nhau giữa các dimension (mục 4.2, Giai đoạn 5). Ngoài ra, 2 thuật toán của cùng một mức bẩn có chung vector profile.

| Thành phần | Đề tài (người dùng chốt ngày 2026-10-26) | Lý do |
|---|---|---|
| **Đánh giá chính (A)** | `StratifiedGroupKFold` 5 fold, nhóm = (task, dimension, λ); riêng λ = 0 gộp thành **1 nhóm mỗi task**. Có 42 nhóm (regression 13, classification 16, clustering 13), stratify theo task, lặp lại 3 lần với seed lấy từ `configs/seeds.yaml` | Kiểm tra đúng mục đích sử dụng: "gặp một mức bẩn chưa thấy, đoán Δ có đúng không". Không còn rò rỉ qua run lặp, qua bản chép λ = 0, hay qua thuật toán còn lại. Mọi dòng đều có dự đoán out-of-fold, nên H5–H7 vẽ được trên đủ 312 dòng; báo cáo dạng trung bình ± độ lệch chuẩn qua các lần lặp |
| **Thí nghiệm phụ (C)** | Leave-one-λ-out: train trên λ ∈ {0; 0,2; 0,8}, test trên λ = 0,5 | Kiểm tra khả năng **nội suy** theo mức bẩn; rất rẻ. Không giấu λ = 0,8 vì đó là ngoại suy, mô hình cây chắc chắn không làm được |
| Baseline có tham số | Weighted score (hệ số hồi quy) và DQSOps-PCA (z-score, PCA, hồi quy) được **fit lại trên đúng fold train** rồi dự đoán fold test. Rule-based có ngưỡng cố định nên không cần fit | So sánh công bằng: fit PCA trên cả 312 dòng thì baseline cũng bị rò rỉ |
| SHAP (H8) | Tính trên meta-model huấn luyện lại bằng **toàn bộ** 312 dòng | Mục đích là giải thích chứ không phải đánh giá |
| Phương án không chọn | Chia ngẫu nhiên theo dòng; leave-one-dimension-out | Chia ngẫu nhiên thì rò rỉ (như trên). Leave-one-dimension-out thì feature của dimension bị giấu luôn ≈ 1 trong train, nên cả 4 phương pháp cùng thất bại, không phân biệt được phương pháp nào tốt hơn |

**Giới hạn cần nêu khi trình bày:** mỗi fold test chỉ có khoảng 8 nhóm nên số liệu dao động giữa các fold, vì vậy phải báo kèm độ lệch chuẩn. Meta-model dạng cây không ngoại suy ra ngoài khoảng λ đã thấy.

**Lợi ích cho báo cáo (điểm cộng):**
- **Kết quả đáng tin hơn:** con số MAE/R² không bị thổi phồng bởi rò rỉ. Đây là lỗi phổ biến trong các bài meta-learning trên bảng kết quả thí nghiệm có lượt lặp.
- **Trả lời trước câu hỏi của hội đồng** ("3 run giống nhau thì chia thế nào?"), cho thấy nhóm hiểu bản chất dữ liệu chứ không chỉ gọi `train_test_split`.
- **Hai góc đánh giá bổ trợ nhau:** A đo khả năng tổng quát sang mức bẩn mới, C đo khả năng nội suy theo λ. C còn cho thấy điểm yếu của baseline tuyến tính (PCA, weighted) khi quan hệ giữa profile và Δ phi tuyến.
- **So sánh công bằng giữa 4 phương pháp:** cùng fold, cùng dữ liệu train, nên chênh lệch phản ánh phương pháp chứ không phải cách chia.

*Trạng thái: **Đã cài** (`src/metamodel/train_metamodel.py`; `116 passed`; chạy trên Colab ngày 2026-10-26, kết quả ở `ket-qua-thuc-nghiem.md` mục 6).*

### 4.4 Nhãn của meta-model: Δ tuyệt đối (chính) và Δ tương đối (phụ)

Mohammed et al. chỉ vẽ hiệu năng thô theo mức ô nhiễm, không định nghĩa nhãn Δ. DQSOps dùng quality score tự định nghĩa, cũng không có Δ. Thiết kế ban đầu của đề tài chỉ dùng Δ tuyệt đối (`baseline − raw`, mục 4.2).

**Vấn đề phát hiện sau Giai đoạn 5:** ba tác vụ có thang đo rất khác nhau, nên cùng một con số Δ mang nghĩa khác nhau.

| Tác vụ | Baseline sạch | Δ lớn nhất đo được | Δ = 0,13 nghĩa là |
|---|---|---|---|
| Regression (R²) | 0,86–0,92 | ~0,54 | mất ~15% hiệu năng |
| Classification (F1) | 0,63–0,73 | ~0,18 | mất ~19% hiệu năng |
| Clustering (AMI) | 0,17–0,20 | ~0,19 (chạm trần: AMI về ≈ 0) | mất ~70% hiệu năng |

Thêm vào đó, Feature Accuracy làm AMI sụp về ≈ 0 từ λ = 0,5, nên ΔAMI bị chặn bởi baseline (hiệu ứng trần, `ket-qua-thuc-nghiem.md` mục 5.5 phát hiện 4).

| Thành phần | Đề tài (người dùng chốt ngày 2026-10-26) | Lý do |
|---|---|---|
| **Nhãn chính** | Δ tuyệt đối = `delta_performance` (đúng schema) | Dễ diễn giải ("F1 giảm 0,1"), khớp schema, khớp cách trình bày đường cong của bài gốc |
| **Nhãn phụ** | Δ tương đối = `delta_performance / baseline_performance`, tính lúc chạy, **không thêm cột vào parquet** | Đưa 3 tác vụ về cùng thang "tỷ lệ hiệu năng bị mất" (≈ 0–1), giảm ảnh hưởng của hiệu ứng trần |
| Cách chạy | Meta-model và 3 baseline chạy **hai lần**, mỗi lần một nhãn; cùng code, chỉ đổi tham số chọn target; cùng cách chia ở mục 4.3 | Chi phí thêm gần như bằng 0 |
| Cách báo cáo | Bảng chính (H5) luôn báo MAE / R² / Spearman **theo từng tác vụ**, kèm số gộp. Bản Δ tương đối đặt ở bảng phụ | Báo theo từng tác vụ thì MAE gộp bị regression chi phối cũng không làm sai kết luận |
| Phương án không chọn | Chỉ Δ tuyệt đối; chỉ Δ tương đối | Chỉ tuyệt đối: MAE gộp bị regression chi phối, clustering bị đánh giá thấp. Chỉ tương đối: khó diễn giải, và baseline AMI nhỏ làm nhiễu bị phóng đại |

**Lợi ích cho báo cáo (điểm cộng):**
- **Kiểm tra độ bền của kết luận (robustness check):** nếu meta-model thắng 3 baseline ở cả hai thang đo thì kết luận không phụ thuộc vào cách chuẩn hóa. Đây là bằng chứng mạnh hơn một con số đơn lẻ.
- **Trả lời trước câu hỏi "sao không chuẩn hóa theo baseline?"** và cho thấy nhóm nhận ra vấn đề thang đo khi gộp nhiều tác vụ, điều bài gốc không phải đối mặt vì không gộp.
- **Liên kết với SHAP:** SHAP trên Δ tương đối cho phép so sánh trực tiếp "dimension này làm mất bao nhiêu phần trăm hiệu năng" giữa các tác vụ, củng cố luận điểm task-aware.
- **Nhất quán với H4** (heatmap), vốn đã dùng Δ tương đối.

**Giới hạn cần nêu khi trình bày:** với clustering, baseline AMI nhỏ nên Δ tương đối khuếch đại nhiễu. Ví dụ độ lệch chuẩn ΔAMI 0,041 của k-Means tương ứng ~24% trên thang tương đối. Kết quả clustering trên thang tương đối cần đọc thận trọng.

*Trạng thái: **Đã cài** (`src/metamodel/train_metamodel.py`; `116 passed`; chạy trên Colab ngày 2026-10-26, kết quả ở `ket-qua-thuc-nghiem.md` mục 6).*

### 4.5 Đầu vào của meta-model và cấu hình 3 baseline

Bài gốc không có các chi tiết này: Mohammed et al. không có meta-model; DQSOps chỉ có PCA composite trên một use case, không có rule-based hay weighted score. Người dùng chốt cả 5 điểm ngày 2026-10-26. Tham số nằm trong `configs/metamodel.yaml`.

| # | Thành phần | Đề tài | Lý do | Phương án không chọn |
|---|---|---|---|---|
| 1 | Dimension không áp dụng (Class Balance của regression, NaN ở 96 dòng) | **Meta-model và DQSOps-PCA:** điền 1,0 chỉ trong ma trận đầu vào; parquet giữ NaN. **Weighted score:** trung bình trên các dimension áp dụng được (regression: 4). **Rule-based:** bỏ qua NaN. **SHAP:** dimension không thuộc ma trận thí nghiệm của tác vụ (Class Balance của regression, Target Accuracy của clustering) đánh dấu "không áp dụng" và không tính vào tỷ trọng | `GradientBoostingRegressor` và PCA không nhận NaN. Trong regression giá trị này là hằng số nên không mang thông tin; one-hot task đã tách regression riêng. 1,0 nghĩa là "không có vấn đề mất cân bằng" | Đổi sang `HistGradientBoostingRegressor` (nhận NaN): lệch thiết kế mô hình; điền trung bình cột: bịa ra một mức mất cân bằng không có thật |
| 2 | Đặc trưng thuật toán | **One-hot 6 thuật toán** (thay `algo_family` linear/tree), giữ one-hot 3 task. Tổng 14 đặc trưng: 5 profile + 3 task + 6 thuật toán | k-Means và GMM không thuộc họ "tuyến tính" hay "cây"; gán họ cho chúng là tùy tiện. One-hot thuật toán chính xác hơn, SHAP vẫn gộp được theo task | Giữ linear/tree và gán clustering vào một họ: mất thông tin, khó bảo vệ |
| 3 | Class Balance của Covertype đi ngược (điểm tăng khi dữ liệu xấu đi, mục 4) | **Không sửa ở baseline nào.** Weighted score và DQSOps-PCA dùng một công thức chung cho mọi tác vụ (không biết task), đúng tinh thần DQSOps | Đây chính là điểm yếu của điểm chất lượng cố định mà đề tài muốn chỉ ra; meta-model biết task nên học được chiều ngược này. Giai đoạn 7 báo cáo cả 312 và 288 lượt | Đảo dấu điểm Balance riêng cho Covertype: can thiệp thủ công, che mất đúng hiện tượng cần chứng minh |
| 4 | Rule-based | Ngưỡng **mọi dimension < 0,9 → Bad**, cố định theo ví dụ của thiết kế, không tinh chỉnh. "Bad" thật: **Δ tương đối > 10%** (mất trên 10% hiệu năng); phân tích độ nhạy với 5% và 20%. Meta-model được rời rạc hóa bằng dự đoán Δ tương đối với cùng ngưỡng | Δ tương đối cho phép một ngưỡng chung cho cả 3 tác vụ. Tinh chỉnh ngưỡng thì rule-based thành một mô hình học, không còn là baseline "luật cố định" | Ngưỡng trên Δ tuyệt đối: một con số không phù hợp cả R² ~0,9 lẫn AMI ~0,2 |
| 5 | Siêu tham số meta-model | Cố định theo thiết kế: `n_estimators = 100`, `max_depth = 3`, `learning_rate = 0.1`; seed từ `seeds.yaml`. **Không tinh chỉnh** | 312 dòng mà tinh chỉnh thì phải dùng nested CV, dễ overfit lên chính tập đánh giá; giữ cố định cho số liệu trung thực | Grid search trong CV thường: số liệu bị lạc quan giả tạo |

**Hệ quả cần biết khi đọc kết quả rule-based:** Class Balance của Covertype ở baseline sạch đã là 0,254 < 0,9, nên rule-based báo **Bad cho mọi lượt Covertype, kể cả λ = 0**. *Đo thật (Giai đoạn 7) cho thấy HIGGS cũng vậy:* Class Balance sạch là 0,887, chỉ dưới ngưỡng một chút, nên rule-based báo Bad cho mọi lượt classification; tổng cộng 92,3% số lượt bị báo Bad. Đây là điểm yếu thật của luật cố định khi dữ liệu mất cân bằng tự nhiên, nên giữ nguyên và nêu trong báo cáo.

**Lợi ích cho báo cáo (điểm cộng):**
- **Mỗi lựa chọn đều có lý do và phương án thay thế đã cân nhắc**, nên trả lời được câu hỏi "sao không làm cách khác?" cho từng điểm.
- **Baseline được cài trung thực, không bị làm yếu đi có chủ đích:** công thức của từng baseline đúng như thiết kế, chỉ khác ở những gì bài gốc không quy định. Nếu meta-model thắng thì là thắng công bằng.
- **Điểm 3 biến một "lỗi" thành một minh họa:** cùng một điểm Balance tăng, công thức cố định đoán sai hướng, còn mô hình theo ngữ cảnh đoán đúng. Đây là ví dụ cụ thể, dễ hiểu cho luận điểm task-aware.
- **Không tinh chỉnh siêu tham số** cho thấy kết quả không đến từ việc "vặn" mô hình trên tập đánh giá.

**Giới hạn cần nêu khi trình bày:** điền 1,0 cho Class Balance của regression làm cột này có phương sai gắn với tác vụ khi PCA gộp cả 3 tác vụ, nên PC1 của DQSOps phần nào phản ánh "đây là tác vụ nào". Đây là hệ quả của việc dùng một điểm chung cho mọi tác vụ, đúng như DQSOps làm.

**Đã xác nhận bằng số liệu (Colab 2026-10-26, `ket-qua-thuc-nghiem.md` mục 6.4):** PC1 chỉ giải thích 26,8% phương sai. Trọng số lớn nhất rơi vào Class Balance (0,595) và Target Accuracy (−0,704, âm), còn Feature Accuracy, dimension gây hại mạnh nhất, có trọng số ≈ 0. Đây là **điểm cộng cho báo cáo**: một bằng chứng cụ thể rằng điểm chất lượng tổng hợp không gắn downstream có thể "đo" sai thứ (tác vụ thay vì mức bẩn). Trong khi đó SHAP của meta-model xếp Feature Accuracy đứng đầu ở 2/3 tác vụ và Uniqueness thấp nhất ở cả 3.

*Trạng thái: **Đã cài** (`src/baselines/`, `src/metamodel/`; `116 passed`; chạy trên Colab ngày 2026-10-26).*

### 4.6 Cách đánh giá và so sánh 4 phương pháp (Giai đoạn 7)

Hai bài gốc không có bước này: Mohammed et al. không có mô hình dự đoán; DQSOps chỉ báo sai số dự đoán điểm chất lượng trên một use case, không so với baseline nào. Thiết kế ban đầu của đề tài (`pipeline-thi-nghiem.md` mục 8) chỉ ghi "MAE, R², Spearman, phân tích theo tác vụ". Người dùng chốt 6 điểm dưới đây ngày 2026-10-26. Tham số nằm trong `configs/metamodel.yaml`, mục `evaluation`.

| # | Thành phần | Đề tài | Lý do | Phương án không chọn |
|---|---|---|---|---|
| 1 | Nơi đặt code | `src/metamodel/evaluate_methods.py`, đọc `metamodel_cv_predictions.parquet`, **không huấn luyện lại** | Không đổi cấu trúc thư mục đã chốt; tách tính toán (Giai đoạn 7) khỏi huấn luyện (Giai đoạn 6) nên chạy lại rất rẻ | Thư mục mới `src/evaluation/`: rõ hơn nhưng đổi cấu trúc đã chốt |
| 2 | Gộp chỉ số | MAE, R², Spearman tính riêng từng lần lặp (seed 42/43/44), báo **trung bình ± độ lệch chuẩn**. **Bảng chính tách theo tác vụ**; cột "tất cả" là phụ | Với Δ tuyệt đối, R²/Spearman gộp 3 tác vụ chủ yếu đo "phân biệt regression với clustering" (thang Δ khác nhau), không đo khả năng dự đoán | Chỉ báo số gộp: đẹp giả tạo, che mất tác vụ mà phương pháp thua |
| 3 | **Ý nghĩa thống kê (mới)** | **Bootstrap theo nhóm**: lấy mẫu lại có hoàn lại 42 nhóm (task, dimension, λ), giữ nguyên số nhóm của mỗi tác vụ, 1.000 lần, seed từ `seeds.yaml`. Báo khoảng tin cậy 95% của hiệu MAE (meta-model − từng baseline), theo tác vụ | Độ lệch chuẩn qua 3 seed chỉ phản ánh dao động do cách chia fold, không trả lời "hơn có thật không". Lấy mẫu theo nhóm (không theo dòng) vì các dòng trong một nhóm phụ thuộc nhau; lấy theo dòng sẽ cho khoảng tin cậy hẹp giả tạo | Chỉ báo trung bình ± độ lệch chuẩn: không kết luận được về ý nghĩa; kiểm định t: giả định độc lập và phân phối chuẩn không hợp với 42 nhóm |
| 4 | Good/Bad (H7) | Nhãn thật: Bad nếu Δ tương đối > 10%. Rule-based dùng nhãn 0/1 của nó; **cả 3 phương pháp hồi quy** rời rạc hóa từ dự đoán của bản chạy nhãn Δ tương đối, cùng ngưỡng. Chỉ số: accuracy, **F1 lớp Bad**, balanced accuracy, precision, recall, kèm tỷ lệ Bad thật và ma trận nhầm lẫn. Độ nhạy: ngưỡng 5% và 20% | Lớp Bad dự kiến là thiểu số (λ = 0 và Uniqueness gần như luôn Good), nên accuracy đơn thuần đẹp giả tạo; F1 lớp Bad và balanced accuracy không bị ảnh hưởng. So cả 4 phương pháp trên cùng một thước đo phân loại cho so sánh công bằng với rule-based | Chỉ accuracy: thiên vị phương pháp luôn đoán Good |
| 5 | 312 hay 288 lượt | **Bảng chính 312 lượt**; bảng phụ 288 lượt (bỏ 24 lượt Class Balance × clustering) chỉ **lọc phần đánh giá**, không huấn luyện lại | Bỏ đúng 24 lượt mà baseline đoán sai hướng thì giống chọn dữ liệu có lợi; ΔAMI ở đó có thật (0,02–0,10) và meta-model học được nhờ biết tác vụ — chính là luận điểm cần chứng minh. Bảng 288 cho thấy kết luận có đứng vững khi bỏ phần gây tranh cãi | Bảng chính 288: an toàn hơn về mặt số đo nhưng mất một minh họa tốt và có thể bị xem là né tránh |
| 6 | Chẩn đoán SHAP | Trung bình \|SHAP\| của từng dimension, **tách theo dimension đang bị làm bẩn của dòng** → `shap_by_row_dimension.csv` | Giải thích vì sao Uniqueness ở classification có tỷ trọng 0,107 dù ΔF1 ≈ 0 (`ket-qua-thuc-nghiem.md` mục 6.5) | Bỏ qua: để lại một con số khó giải thích trong báo cáo |

**Lợi ích cho báo cáo (điểm cộng):**
- **Kết luận có khoảng tin cậy**, không chỉ là "số này nhỏ hơn số kia": nói được meta-model hơn baseline ở tác vụ nào là có ý nghĩa, ở đâu thì chưa đủ bằng chứng. Ít đồ án môn học làm bước này.
- **Bootstrap theo nhóm nhất quán với cách chia train/test** (mục 4.3): cùng đơn vị độc lập là nhóm (task, dimension, λ), nên toàn bộ phần đánh giá dùng một quan điểm thống nhất về tính độc lập của dữ liệu.
- **So sánh công bằng với rule-based:** cả 4 phương pháp được chấm trên cùng thang Good/Bad, với thước đo không bị lệch do lớp thiểu số.
- **Bảng phụ 288 lượt** chặn trước câu hỏi "có phải meta-model thắng nhờ 24 lượt Class Balance kỳ lạ?".
- **Báo cáo theo tác vụ** giữ đúng tinh thần task-aware của đề tài và không che chỗ meta-model thua (mục 6.3 của `ket-qua-thuc-nghiem.md` đã thấy weighted score thắng ở classification với Δ tương đối).

**Giới hạn cần nêu khi trình bày:** 42 nhóm là ít cho bootstrap, khoảng tin cậy chỉ mang tính tham khảo. Thí nghiệm nội suy chỉ có 13 nhóm ở λ = 0,5 nên khoảng tin cậy ở đó rất rộng. Lỗi tuyệt đối của mỗi dòng được lấy trung bình qua 3 lần lặp trước khi bootstrap.

**Đã xác nhận bằng số liệu (Colab 2026-10-26, `ket-qua-thuc-nghiem.md` mục 7):**
- **Bootstrap đổi cách diễn đạt kết luận:** nếu chỉ nhìn MAE (0,055 so với 0,077), dễ nói "meta-model vượt trội cả hai baseline". Bootstrap cho thấy chỉ hơn DQSOps-PCA là có ý nghĩa; hơn weighted score thì CI [−0,047; +0,005] còn chứa 0. Báo cáo nói đúng mức này — một kết luận được kiểm chứng, khó bị bắt bẻ.
- **Báo theo tác vụ** làm lộ ra chỗ weighted score thắng (classification với Δ tương đối, và bài toán Good/Bad), thay vì bị con số gộp che đi.
- **Bảng 288 lượt** cho cùng kết luận, khoảng cách ở clustering còn lớn hơn: meta-model không thắng nhờ 24 lượt Class Balance.
- **Dự đoán sai khi thiết kế:** lớp Bad không phải thiểu số (51,6%). F1 lớp Bad và balanced accuracy vẫn cần thiết: DQSOps-PCA có accuracy 0,478 nhưng balanced accuracy 0,463 vì gần như đoán Bad hết.
- **Chẩn đoán SHAP bác bỏ giả thuyết ban đầu** và cho một cách đọc đúng hơn (Uniqueness là tín hiệu loại trừ), giúp trình bày H8 không bị hiểu sai.

*Trạng thái: **Đã cài** (`src/metamodel/evaluate_methods.py`, chẩn đoán trong `explain_shap.py`; `124 passed`; chạy trên Colab ngày 2026-10-26).*

### 4.7 Cách đo khả năng mở rộng (Giai đoạn 8)

Cả hai bài gốc đều không có phần này: Mohammed et al. dùng dữ liệu 1.000–25.000 dòng; DQSOps chạy trên một use case công nghiệp, không đo theo quy mô. Đây là đóng góp "quy mô lớn" của đề tài (mục 1). Thiết kế ban đầu (`kien-truc-mo-hinh-va-ky-thuat.md` mục 6) chỉ ghi các mốc 1M–11M và gợi ý đo bộ nhớ qua Spark UI hoặc `/proc`. Người dùng chốt các điểm dưới đây ngày 2026-10-26; tham số trong `configs/scalability.yaml`.

| # | Thành phần | Đề tài | Lý do | Phương án không chọn |
|---|---|---|---|---|
| 1 | Dữ liệu đem profile | **Bản sạch so với chính nó** (đọc parquet 2 lần: một bản làm dữ liệu cần đo, một bản làm ground truth) | Thời gian profiling phụ thuộc khối lượng phép tính (quét ô thiếu, ghép theo `row_id` cho Accuracy, băm dòng tìm trùng, đếm lớp), không phụ thuộc dữ liệu có bẩn hay không. Làm bẩn 11M dòng phải dùng polluter pandas — tự nó có thể hết RAM — và phải ghi dữ liệu bẩn ra đĩa, trái quy tắc "không persist dữ liệu bẩn" của project | Profile dữ liệu bẩn: cần viết polluter bằng Spark (ngoài phạm vi), lẫn chi phí làm bẩn vào số đo |
| 2 | Các mốc | **0,1M**, 1M, 3M, 6M, 11M dòng (mốc nhỏ là phần đầu của mốc lớn; HIGGS đã xáo sẵn) | Spark có thời gian khởi động cố định cho mỗi truy vấn (lập kế hoạch, chia việc); mốc 0,1M cho thấy vùng pandas nhanh hơn, và thêm một điểm so sánh được nếu pandas hết RAM sớm | Chỉ 4 mốc thiết kế: có thể chỉ còn 2 điểm cả hai engine cùng chạy được |
| 3 | Thời gian | Tổng **đọc parquet + profile**; 3 lần mỗi mốc, báo **trung vị**; Spark chạy khởi động một lần trên mốc nhỏ nhất (bỏ kết quả) trước khi bấm giờ | Spark đọc dữ liệu kiểu lười nên không tách được phần đọc một cách công bằng; trung vị ít nhạy với một lần chạy bất thường; bỏ thời gian khởi JVM vì đó không phải chi phí của profiling | Đo một lần; tính cả khởi JVM |
| 4 | Bộ nhớ | **Chỉ đo đỉnh RSS của pandas** (`resource.getrusage`, không tốn thêm gì); **Spark chạy với giới hạn cố định** `spark.driver.memory = 4g` | Thông điệp cần chứng minh là "pandas cần bộ nhớ tăng theo dữ liệu đến khi chạm trần RAM; Spark chạy hết 11M dòng trong một giới hạn cố định". Bộ nhớ đo được của JVM phản ánh vùng heap đã giữ trước, khó diễn giải | Đo cả hai bằng `psutil`: phức tạp, số của Spark dễ gây hiểu nhầm; bỏ hẳn: mất phần giải thích *vì sao* pandas thất bại |
| 5 | An toàn khi đo | **Mỗi lần đo là một tiến trình con riêng** (cả nhóm tiến trình, gồm JVM). Hết RAM → `oom`; quá 20 phút → `timeout`; notebook vẫn chạy. Engine đã thất bại ở một mốc thì các mốc lớn hơn ghi `skipped`. Ghi kết quả sau mỗi lần đo, chạy lại thì bỏ qua lần đã xong | Pandas hết RAM trong chính notebook sẽ làm chết cả phiên Colab; mốc lớn hơn chắc chắn cũng thất bại nên không tốn thời gian thử | Đo trong notebook: mất phiên khi pandas hết RAM |
| 6 | Kiểm tra đúng đắn | Ở mọi mốc cả hai engine cùng chạy được, đối chiếu kết quả profile, sai số tương đối ≤ 1e-9 | Mở rộng phép đối chiếu Giai đoạn 2 (50.000 dòng) lên quy mô triệu dòng | — |
| 7 | Lưu dữ liệu | HIGGS full đổi sang parquet một lần, lưu Drive `data/raw/higgs_full/higgs_full_raw.parquet` (~1,2GB, giữ nguyên nội dung gốc); các mốc kèm cột `row_id` cắt ra ổ tạm `/content` đầu mỗi phiên; lúc đo luôn đọc từ `/content` | Phiên sau không phải tải lại 2,8GB từ UCI; đọc từ Drive chậm và dao động theo mạng nên làm sai số đo | Không lưu: mỗi phiên tải lại 10–15 phút |

**Lợi ích cho báo cáo (điểm cộng):**
- **Kết luận trung thực về Spark:** thiết kế chấp nhận trước khả năng Spark chậm hơn pandas trên máy 2 nhân, và đo đúng thứ Spark mang lại (chạy được khi dữ liệu vượt RAM, trong giới hạn bộ nhớ cố định). Thông điệp "pandas cho dữ liệu nhỏ, Spark cho dữ liệu lớn" thuyết phục hơn "Spark luôn nhanh hơn".
- **Điểm gãy của pandas là số đo thật** (`oom` cùng bộ nhớ đỉnh tăng dần), không phải ước tính trên giấy.
- **Cùng công thức, khác engine được kiểm chứng ở quy mô lớn** (điểm 6), nên so sánh thời gian là công bằng.
- **Tái lập được:** cùng một bản dữ liệu trên Drive, các mốc cắt xác định (phần đầu), tham số trong config.

**Giới hạn cần nêu khi trình bày:** Spark ở chế độ local trên một máy ảo Colab (khoảng 2 nhân) không phản ánh cluster phân tán thật — đây là minh họa xu hướng (thiết kế mục 8 của `kien-truc`). Kết quả profile ở Giai đoạn 8 gần như bằng 1 (dữ liệu sạch); khả năng phát hiện bẩn đã kiểm chứng ở Giai đoạn 3. Pandas đọc dữ liệu 2 lần (bản cần đo và ground truth) như cách dùng thật, nên bộ nhớ của nó gấp đôi so với chỉ profile một bảng không cần Accuracy.

*Trạng thái: Đã viết (`src/scalability/spark_scale_study.py`; chưa chạy trên Colab).*

---

## 5. Khác biệt về chuẩn bị dữ liệu

| Điểm | Chi tiết | Hệ quả |
|---|---|---|
| Beijing: bỏ dòng thiếu thay vì impute | Bỏ 38.600 / 420.768 dòng (9,2%), còn 382.168 dòng | Baseline có Completeness đúng bằng 1; phân phối có thể lệch nhẹ nếu missing tập trung theo mùa |
| Beijing: bỏ cột `No`, `year` | `No` là số thứ tự; `year` dễ gây rò rỉ theo thời gian | Giữ `month`, `day`, `hour` |
| HIGGS: lấy mẫu 1.000.000 / 11.000.000 dòng | Stratified theo target, seed 42 | Tỷ lệ nhãn giữ nguyên 52,99% / 47,01% |
| HIGGS mẫu có 2.281 dòng trùng (0,23%) | Giữ nguyên, không xóa | Uniqueness của baseline hơi nhỏ hơn 1; ΔPerformance luôn tính so với baseline thật |
| Covertype mất cân bằng tự nhiên | Lớp nhỏ nhất 2.747 dòng, lớp lớn nhất 283.301 dòng | Class Balance của baseline đã thấp sẵn |

**Lợi ích cho báo cáo (điểm cộng):** dữ liệu lớn hơn bài gốc 15–40 lần (382K–1M so với 1–25K dòng) mà vẫn có bản sạch rõ ràng, tách được missing tự nhiên khỏi missing do làm bẩn, và tránh rò rỉ theo thời gian ở Beijing.

*Trạng thái: Đã cài (đã chạy và kiểm tra trên Colab ở Giai đoạn 1).*

---

## 6. Khác biệt về cài đặt

| Điểm | Bài gốc | Đề tài |
|---|---|---|
| Engine profiling | Một bản (Python) | Hai bản pandas và PySpark, dùng chung một file công thức (`common.py`) |
| Kiểm chứng | — | Test có đáp án tính tay + đối chiếu kết quả pandas với PySpark |

Kết quả: 12/12 test passed; pandas và PySpark khớp trên mẫu 50.000 dòng của cả 3 dataset.

**Lợi ích cho báo cáo (điểm cộng):** một file công thức dùng chung đảm bảo hai engine **cho cùng kết quả toán học** (khớp tới sai số tương đối 1e-9), nên phần scalability so sánh tốc độ một cách công bằng. Đây là nền cho đóng góp "quy mô lớn" mà cả hai bài gốc đều không có.

*Trạng thái: Đã cài.*

---

## 7. Gợi ý dùng cho slide

- **Giữa kỳ:** mục 1 (định vị đề tài), mục 2 (phạm vi), mục 5 (dữ liệu đã chuẩn bị), và mục 3.1 như một phát hiện khi mở rộng sang dữ liệu lớn.
- **Cuối kỳ:** thêm mục 4 (sau khi polluter đã chốt) và kết quả thực nghiệm; mục 3 và 4 đưa vào phần "Giới hạn và điều chỉnh so với bài gốc".
- Câu hỏi dễ bị hỏi nhất: *"Vì sao không dùng đúng công thức của bài gốc?"* — trả lời bằng bảng ví dụ ở mục 3.1 và ý "trùng với bài gốc trên cột dương".
- **Cuối kỳ, phần phương pháp đánh giá meta-model:** mục 4.3 (chia theo nhóm, chống rò rỉ), mục 4.4 (hai thang Δ, kiểm tra độ bền) và mục 4.6 (bootstrap theo nhóm, bảng chính 312 / phụ 288). Mỗi mục đã có sẵn phần "Lợi ích cho báo cáo" để chuyển thành slide.
