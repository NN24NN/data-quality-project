# tests/

- `test_profiling.py` — công thức đúng trên ví dụ nhỏ có đáp án tính tay; pandas và PySpark cho cùng kết quả.
- `test_pollution.py` — λ = 0 giữ nguyên dữ liệu; cùng seed cho cùng kết quả; mức bẩn đo bằng profiler khớp λ.
- `test_downstream.py` — 6 thuật toán học được dữ liệu dễ; target không lọt vào feature; cùng seed cho cùng điểm; chạy được trên dữ liệu đã làm bẩn; chia train/test stratified.

- `test_experiment_runner.py` — đủ số dòng, đúng schema, λ = 0 cho Δ = 0, chạy tiếp được từ chỗ dừng.
- `test_metamodel.py` — đặc trưng và nhóm chống rò rỉ; Group K-Fold không tách nhóm, dự đoán mỗi dòng đúng 1 lần; 3 baseline xử lý NaN đúng và chỉ fit trên train; meta-model thắng baseline trên dữ liệu phụ thuộc tác vụ; SHAP tỷ trọng cộng bằng 1.

- `test_evaluate_methods.py` — chỉ số hồi quy/phân loại khớp tính tay; bootstrap lấy lại theo nhóm, tái lập được; bảng phụ bỏ đúng 24 lượt; đủ dòng ở 3 bảng kết quả; chẩn đoán SHAP. Dùng bảng giả lập của `test_metamodel.py`.

- `test_scalability.py` — chuẩn bị dữ liệu và cắt mốc đúng (kể cả mốc vượt dữ liệu gốc); đo pandas trong tiến trình con, RAM không kế thừa từ tiến trình cha; `--redo` chỉ đo lại engine được chọn; phân loại thất bại (`oom`/`error`); engine thất bại thì mốc lớn hơn `skipped`; chạy tiếp được; pandas và PySpark khớp đầu-cuối; quá giờ thì giết tiến trình. Chỉ chạy trên Linux; phần PySpark cần Java.

Chạy tất cả: `python -m pytest tests/ -q` (trên Colab: cell 15.1; Giai đoạn 7 đạt `124 passed` ngày 2026-10-26, Giai đoạn 8 thêm 10 test, đạt `134 passed` ngày 2026-10-26).
