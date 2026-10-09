# tests/

- `test_profiling.py` — công thức đúng trên ví dụ nhỏ có đáp án tính tay; pandas và PySpark cho cùng kết quả.
- `test_pollution.py` — λ = 0 giữ nguyên dữ liệu; cùng seed cho cùng kết quả; mức bẩn đo bằng profiler khớp λ.
- `test_downstream.py` — 6 thuật toán học được dữ liệu dễ; target không lọt vào feature; cùng seed cho cùng điểm; chạy được trên dữ liệu đã làm bẩn; chia train/test stratified.

Chạy tất cả: `python -m pytest tests/ -q` (trên Colab: cell 11.1, kỳ vọng `93 passed`).
