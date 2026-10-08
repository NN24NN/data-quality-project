# src/profiling/

Module tính điểm chất lượng (Completeness, Feature Accuracy, Target Accuracy, Uniqueness, Class Balance).

| File | Vai trò |
|---|---|
| `common.py` | Công thức quy đổi thống kê thô → điểm, dùng chung cho cả hai engine |
| `pandas_profiler.py` | Tính thống kê thô bằng pandas |
| `spark_profiler.py` | Tính thống kê thô bằng PySpark (dùng cho Giai đoạn 8) |

Cả hai bản có cùng hàm `compute_profile(df, target, target_type, clean=None, onehot_prefixes=(), n_classes=None)` (bản Spark thêm `id_col`), trả về dict 5 khóa `profile_*` đúng tên cột trong schema kết quả. Thông tin cột của từng dataset đọc từ `configs/datasets.yaml`.

Test: `python -m pytest tests/test_profiling.py -q`.
