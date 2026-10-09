# src/pollution/

5 polluter độc lập, cùng chữ ký `pollute(df, level, seed, target, target_type, onehot_prefixes) → df_polluted`. Gọi qua `pollution.POLLUTERS[<tên dimension>]`; tên trùng với `configs/experiment_matrix.yaml`.

| File | Cách làm bẩn |
|---|---|
| `completeness.py` | MCAR: mỗi feature thiếu tỷ lệ λ số dòng (số → NaN, one-hot → cả nhóm về 0) |
| `feature_accuracy.py` | Số: nhiễu Gaussian phương sai λ, nhân với `mean(\|gt\|)`. One-hot: tỷ lệ λ số dòng đổi sang giá trị khác |
| `target_accuracy.py` | Như trên, áp dụng cho cột target |
| `uniqueness.py` | Nhân bản dòng theo hệ số `1 / (1 − λ)` |
| `class_balance.py` | Giữ lớp lớn nhất, xóa tỷ lệ λ số dòng của mọi lớp còn lại |

Mọi polluter giữ index của dòng gốc, không sửa df đầu vào, và ở λ = 0 trả về dữ liệu y nguyên. Seed do nơi gọi truyền vào (đọc từ `configs/seeds.yaml`).

Test: `python -m pytest tests/test_pollution.py -q` — đã chạy trên Colab, passed (cùng test profiling là 43/43).

Lưu ý: trên dữ liệu nhiều lớp (Covertype), điểm Class Balance đo được **tăng** khi λ tăng — xem `docs/khac-biet-so-voi-bai-goc.md` mục 4.
