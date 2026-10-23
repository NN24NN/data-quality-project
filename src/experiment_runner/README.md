# src/experiment_runner/

`run_experiment.py` là bộ điều phối chính. Nó ghép module pollution, profiling và downstream, chạy vòng lặp task → dimension → level → run, và ghi kết quả ra `results/experiment_results.parquet` (312 dòng, schema theo `docs/thiet-ke-thi-nghiem-chi-tiet.md` mục 5).

- ΔPerformance = baseline − raw. Baseline lấy **cùng seed** với lượt chạy, đọc từ `results/baseline_performance.csv` (notebook cell 11.3).
- Kết quả được ghi lại sau mỗi (dimension, level, run). Chạy lại thì runner bỏ qua các lượt đã có, nên tiếp tục được từ chỗ dừng.
- Ở λ = 0, mỗi (thuật toán, run) chỉ huấn luyện một lần; kết quả được chép sang các dimension còn lại.
- Có thể chạy một phần bằng `--tasks`, `--dimensions`, `--levels`, `--runs`. Muốn ghi ra file khác thì dùng `--output`.

Chạy trên Colab: xem notebook mục 12. Test: `tests/test_experiment_runner.py`.
