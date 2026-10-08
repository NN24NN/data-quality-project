# Tổng quan hai bài báo gốc và khoảng trống nghiên cứu

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**Tài liệu này dùng để:** giúp các thành viên trong nhóm hiểu đề tài đang dựa trên nền tảng nào, điểm mới nằm ở đâu, và vì sao thiết kế thí nghiệm sẽ đi theo hướng đã chọn.

---

## 1. Hai bài báo gốc

| | Bài 1 (nền tảng chính) | Bài 2 (baseline ML-based) |
|---|---|---|
| Tên | *The Effects of Data Quality on Machine Learning Performance on Tabular Data* | *DQSOps: Data Quality Scoring Operations Framework for Data-Driven Applications* |
| Tác giả | Mohammed, Budach, Feuerpfeil, Ihde, Nathansen, Noack, Patzlaff, Naumann, Harmouch | Bayram, Ahmed, Hallin, Engman |
| Nơi đăng | Information Systems, vol. 132 (2025) | EASE 2023 |
| Link | arXiv:2207.14529 | arXiv:2303.15068 |
| Code | github.com/HPI-Information-Systems/DQ4AI | không công khai kiến trúc đầy đủ |

---

## 2. Bài 1 — Mohammed et al. (2025): bài gốc chính

### Mục tiêu
Nghiên cứu thực nghiệm (không xây mô hình dự đoán) về mối quan hệ giữa 6 chiều chất lượng dữ liệu và hiệu năng của 19 thuật toán ML trên 3 tác vụ: classification, regression, clustering.

### 6 chiều chất lượng dữ liệu (có định nghĩa toán học + polluter tương ứng)

| Dimension | Ý nghĩa | Cách polluter tạo lỗi |
|---|---|---|
| Consistent Representation | Một thực thể có nhiều cách biểu diễn khác nhau (vd: "New York" vs "NYC") | Thêm k biến thể biểu diễn mới cho mỗi giá trị categorical |
| Completeness | Tỷ lệ giá trị không bị thiếu | Chèn missing value (MCAR) theo tỷ lệ λ |
| Feature Accuracy | Mức sai lệch của giá trị feature so với ground truth | Categorical: đổi ngẫu nhiên sang giá trị khác trong domain. Numerical: cộng nhiễu Gaussian tỷ lệ với λ |
| Target Accuracy | Mức sai lệch của nhãn/target so với ground truth | Tương tự Feature Accuracy nhưng áp dụng lên cột target (label noise) |
| Uniqueness | Tỷ lệ bản ghi không trùng lặp | Thêm duplicate theo hệ số ρ, phân phối duplicate theo uniform/normal/Zipf |
| Target Class Balance | Mức cân bằng giữa các lớp | Chủ động tạo imbalance có kiểm soát giữa các lớp |

Mỗi dimension có công thức định lượng quality score (0–1) riêng và polluter có tham số điều chỉnh mức độ ô nhiễm liên tục — đây là phần có thể tái sử dụng trực tiếp làm "profile vector" cho pipeline của mình.

### Thiết kế thí nghiệm
- **3 kịch bản** (chỉ áp dụng cho classification/regression, clustering chỉ có 1 kịch bản vì không có train/test riêng):
  - Scenario 1: chỉ làm bẩn tập train
  - Scenario 2: chỉ làm bẩn tập test
  - Scenario 3: làm bẩn cả hai
- **10 dataset**: Credit, Contraceptive, Telco, COVID (classification); Houses, IMDB, Cars, COVID (regression); Bank, Covertype, Letter, COVID (clustering)
- **19 thuật toán**: linear (LogR, SVM, LR, RR), tree-based (DT, RF, GB), KNN, MLP (1/5/10 lớp ẩn), TabNet (transformer), và 5 thuật toán clustering (Gaussian mixture, k-means/k-prototypes, agglomerative, OPTICS, autoencoder)
- **Thước đo hiệu năng**: F1-score (classification), R² (regression), Adjusted Mutual Information (clustering)
- Tổng cộng **4.905 lượt thí nghiệm**

### Phát hiện chính (liên quan trực tiếp đến đề tài của mình)
- Cùng một mức độ "chất lượng thấp" ảnh hưởng **khác nhau rõ rệt tùy theo thuật toán và tác vụ** — đây là bằng chứng thực nghiệm mạnh nhất cho luận điểm "context-aware/task-aware" của đề tài.
- Completeness, Feature Accuracy, Target Accuracy có ảnh hưởng mạnh nhất; Uniqueness và Target Class Balance có ảnh hưởng yếu hơn trong nhiều trường hợp.
- Mô hình cây/ensemble (RF, GB) thường robust hơn mô hình tuyến tính và MLP trước nhiều loại lỗi.
- Huấn luyện trên dữ liệu bẩn tương tự dữ liệu test khi suy luận (Scenario 3) thường cho kết quả tốt hơn so với lệch pha giữa train/test (Scenario 1, 2).

### Giới hạn quan trọng — chính là chỗ đề tài của mình khai thác
- Đây là nghiên cứu **thuần giải thích (explanatory)**: vẽ đường cong hiệu năng theo mức độ ô nhiễm, **không xây mô hình học để dự đoán** hiệu năng từ một profile chất lượng cho trước.
- Không có cơ chế tổng hợp 6 dimension thành một "quality score" duy nhất, càng không có trọng số học được theo tác vụ.
- Không đánh giá khả năng mở rộng (Spark/quy mô lớn) — toàn bộ thí nghiệm chạy trên dataset nhỏ-vừa (1.000–25.000 dòng).
- Tác giả tự nhận đây là bước đầu, dự kiến mở rộng sang nhiều dimension đồng thời và lỗi thực tế (hiện mới dùng lỗi tổng hợp từng dimension riêng lẻ).

---

## 3. Bài 2 — DQSOps (Bayram et al., 2023): baseline ML-based

### Kiến trúc tổng thể
```
Data source → Data window → [Mutant simulator] → Method activator
                                                        │
                        ┌───────────────┬───────────────┤
                        ▼               ▼               ▼
                Standard-based     ML model          Retrain
                scoring (ground    (regression,       signal
                truth)             dự đoán score)
                        │               │
                        └───────┬───────┘
                                ▼
                           Test Oracle
                  (so sánh dự đoán vs ground truth,
                   kích hoạt retrain nếu lệch quá ngưỡng τ)
```

### 5 chiều chất lượng (khác bộ 6 dimension của bài 1)
Accuracy (tỷ lệ giá trị bất thường), Completeness (tỷ lệ missing), Consistency (vi phạm ràng buộc toàn vẹn), Timeliness (kiểm định Kolmogorov–Smirnov so với phân phối tham chiếu), Skewness (phân kỳ Jensen-Shannon so với phân phối lịch sử).

### Cách tạo nhãn huấn luyện (quan trọng — khác đề tài của mình)
- Mỗi dimension có công thức tính điểm 0–1.
- Tổng hợp 5 điểm thành **1 quality score duy nhất bằng PCA** (sau khi chuẩn hóa z-score) — tác giả chủ động từ chối dùng trung bình cộng/trung bình có trọng số vì "không hợp lý về mặt thống kê".
- **Nhãn (ground truth) để huấn luyện mô hình ML chính là quality score tổng hợp này** — không phải hiệu năng của một mô hình ML downstream nào cả.
- Dùng "data mutation" (lấy cảm hứng từ mutation testing trong kiểm thử phần mềm) để tạo lỗi tổng hợp ở giai đoạn khởi tạo, giúp mô hình có đủ dữ liệu đa dạng để học.

### Thực nghiệm
- Use case công nghiệp: cảm biến áp suất của quy trình luyện thép (Electroslag Remelting) tại nhà máy Uddeholm, Thụy Điển — dữ liệu time-series đơn biến, truyền qua Kafka.
- Mô hình: Random Forest và XGBoost.
- Kết quả: tốc độ tăng **74–819 lần** so với cách tính thủ công (standard-based), trong khi R² đạt tới 0,89 khi dùng đủ 5 dimension.
- Tỷ lệ mutation tối ưu khi khởi tạo là khoảng 20% (quá thấp hoặc quá cao đều làm giảm hiệu năng — dạng đường cong chữ U).

### Giới hạn quan trọng — chính là chỗ đề tài của mình khai thác
- **Nhãn dự đoán là một chỉ số chất lượng tự định nghĩa (self-referential), hoàn toàn không gắn với hiệu năng của bất kỳ mô hình ML downstream nào.** Đây là khác biệt cốt lõi so với định vị "dự đoán hiệu năng downstream" của đề tài mình.
- Chỉ thử nghiệm trên **một** use case, dữ liệu time-series đơn biến — chưa kiểm chứng trên dữ liệu dạng bảng đa dạng (tabular) hay nhiều loại tác vụ ML (classification/regression/clustering) như bài 1.
- Không có khái niệm "theo tác vụ" (task-aware) — chỉ có một quality score duy nhất cho mọi mục đích sử dụng dữ liệu.
- Trọng số giữa các dimension không tường minh (ẩn trong PCA), không giải thích được dimension nào quan trọng với tác vụ nào.

---

## 4. Khoảng trống cụ thể mà đề tài khai thác

```
                    Bài 1 (Mohammed)              Bài 2 (DQSOps)             Đề tài của mình
Dữ liệu            10 dataset tabular đa dạng     1 dataset industrial        kế thừa dataset
                   3 tác vụ ML                    time-series                 đa dạng của bài 1
                                                   
Nhãn dự đoán       KHÔNG CÓ (chỉ vẽ biểu đồ)       Quality score tự định      Hiệu năng downstream
                                                   nghĩa (PCA composite)      (F1/R²/AMI suy giảm)

Tính task-aware    CÓ (quan sát được qua           KHÔNG (1 score cho mọi    CÓ (trọng số/mô hình
                   thực nghiệm, nhưng không         mục đích)                 theo từng loại tác vụ)
                   mô hình hóa tường minh)

Khả năng mở rộng   Không đánh giá                  Có (tối ưu tốc độ) nhưng   Cần đánh giá (Spark,
(quy mô lớn)                                       chưa thử dữ liệu lớn       nhiều kích thước dữ liệu)
```

**Khoảng trống = chưa có công trình nào học một mô hình dự đoán mức suy giảm hiệu năng ML downstream (không phải một quality score tự định nghĩa) từ một profile chất lượng đa chiều, có phân biệt theo loại tác vụ (classification/regression/clustering), và có đánh giá khả năng mở rộng ở quy mô lớn.**

### Đóng góp dự kiến của đề tài
1. **Nhãn huấn luyện lấy từ downstream thật** (ΔF1, ΔR², ΔAMI so với baseline sạch) — khác DQSOps ở điểm này.
2. **Mô hình task-conditioned**: input gồm profile chất lượng (có thể tái dùng 6 dimension + polluter của bài 1) cộng đặc trưng loại tác vụ/mô hình, output là mức suy giảm hiệu năng dự kiến. Dùng gradient boosting + SHAP để rút ra "trọng số" từng dimension theo từng tác vụ — khác DQSOps ở việc trọng số tường minh, giải thích được, thay vì ẩn trong PCA.
3. **Profiling có khả năng mở rộng bằng Spark** — phần cả hai bài gốc chưa làm tốt (bài 1 không có, bài 2 chỉ test tốc độ scoring chứ không test trên dữ liệu lớn).
4. **So sánh với baseline**: rule-based, weighted score, và chính DQSOps (dùng PCA composite score làm proxy) để chứng minh việc gắn trực tiếp với downstream performance cho kết quả dự đoán tốt hơn.

---

## 5. Việc cần làm tiếp theo

- Thiết kế chi tiết thí nghiệm: chọn tập con dimension/dataset từ bài 1 để tái sử dụng, định nghĩa chính xác nhãn ΔPerformance, và kiến trúc mô hình dự đoán.
- Quyết định có cần thêm dataset quy mô lớn hơn `AirPollution.csv` hiện có (~43.824 dòng) để test phần Spark/scalability hay không.
- Chốt baseline cụ thể sẽ cài đặt: rule-based (ngưỡng cố định), weighted score (trung bình có trọng số thủ công), và DQSOps-style (PCA composite).

---

## Nguồn

- Mohammed, S., Budach, L., Feuerpfeil, M., Ihde, N., Nathansen, A., Noack, N., Patzlaff, H., Naumann, F., Harmouch, H. (2025). *The Effects of Data Quality on Machine Learning Performance on Tabular Data*. Information Systems, 132, 102549. https://arxiv.org/abs/2207.14529
- Bayram, F., Ahmed, B. S., Hallin, E., Engman, A. (2023). *DQSOps: Data Quality Scoring Operations Framework for Data-Driven Applications*. EASE 2023. https://arxiv.org/abs/2303.15068
