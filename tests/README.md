# tests/

- `test_profiling.py` — công thức đúng trên ví dụ nhỏ có đáp án tính tay; pandas và PySpark cho cùng kết quả.
- `test_pollution.py` — λ = 0 giữ nguyên dữ liệu; cùng seed cho cùng kết quả; mức bẩn đo bằng profiler khớp λ.

Chạy tất cả: `python -m pytest tests/ -q` (trên Colab: cell 10.1).
