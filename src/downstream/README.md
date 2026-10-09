# src/downstream/

Module ML downstream (scikit-learn). Ba tác vụ cùng chữ ký `evaluate(algorithm, train, test, target, seed, params) → điểm hiệu năng`. Gọi qua `downstream.EVALUATORS[<tên task>]`; tên task và tên thuật toán trùng với `configs/experiment_matrix.yaml`.

| File | Thuật toán | Thước đo |
|---|---|---|
| `classification.py` | `logistic_regression`, `random_forest` | F1-macro trên test |
| `regression.py` | `ridge`, `gradient_boosting` | R² trên test |
| `clustering.py` | `kmeans`, `gmm` | AMI so với target (không tách train/test, `test` không dùng; số cụm = số lớp thật) |
| `common.py` | Tiền xử lý dùng chung và `split_train_test` (stratified; target số chia 10 khoảng phân vị) | |

Tiền xử lý giống nhau cho cả 6 thuật toán, học trên tập train: điền ô thiếu bằng trung vị → chuẩn hóa z-score → mô hình. Nhóm one-hot bị làm thiếu (cả nhóm = 0) giữ nguyên.

Siêu tham số do nơi gọi truyền vào (đọc từ `configs/algorithms.yaml`); `random_state` lấy từ seed của lượt chạy (`configs/seeds.yaml`).

Test: `python -m pytest tests/test_downstream.py -q` — đã chạy trên Colab, passed (cùng test cũ là 93/93).
