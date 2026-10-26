# configs/

Tham số tách khỏi code. Xem `docs/cau-truc-thu-muc-project.md` mục 2.4.

| File | Nội dung | Trạng thái |
|---|---|---|
| `seeds.yaml` | Seed cố định 42, 43, 44 (run 1-3) | Đã tạo |
| `pollution_levels.yaml` | Mức ô nhiễm [0.0, 0.2, 0.5, 0.8] | Đã tạo |
| `experiment_matrix.yaml` | Ma trận dataset × dimension × thuật toán (312 lượt) | Đã tạo |
| `datasets.yaml` | Cột target, loại target, tiền tố nhóm one-hot của từng bản sạch | Đã tạo |
| `algorithms.yaml` | Siêu tham số của 6 thuật toán downstream | Đã tạo (có thể chỉnh sau khi đo thời gian chạy) |
| `scalability.yaml` | Giai đoạn 8: nguồn HIGGS full, các mốc, số lần đo, giới hạn thời gian, cấu hình Spark (giới hạn bộ nhớ 4g), ngưỡng đối chiếu | Đã tạo (2026-10-26) |
| `metamodel.yaml` | Giai đoạn 6: siêu tham số meta-model (cố định), giá trị điền NaN, 2 nhãn, số fold, mức λ nội suy, ngưỡng rule-based | Đã tạo (2026-10-26) |
