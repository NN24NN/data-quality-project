# src/scalability/

`spark_scale_study.py` — đo thời gian profiling bằng pandas và PySpark (`local[*]`) trên HIGGS full ở các mốc 0,1M / 1M / 3M / 6M / 11M dòng (Giai đoạn 8). Chạy trên Colab: notebook mục 15.

- `prepare`: tải HIGGS full về `/content`, đổi sang parquet và lưu Drive `data/raw/higgs_full/higgs_full_raw.parquet` (bỏ qua nếu đã có).
- `run`: cắt các mốc (kèm `row_id`) ra `/content/higgs_scale`, đo từng (mốc, engine, lần) trong tiến trình con riêng, ghi dần `results/scalability_benchmark.csv`; chạy lại thì bỏ qua các lần đã có.
- Profile bản sạch so với chính nó; thời gian = đọc parquet + profile, 3 lần mỗi mốc, báo trung vị; Spark chạy khởi động một lần trước khi bấm giờ.
- Bộ nhớ: chỉ đo đỉnh RSS của pandas; Spark chạy với giới hạn cố định `spark.driver.memory`. Pandas hết RAM → `oom`, quá giờ → `timeout`, các mốc lớn hơn → `skipped`.
- Đối chiếu điểm profile pandas và PySpark ở mọi mốc cả hai cùng chạy được (sai số tương đối ≤ 1e-9).

Tham số: `configs/scalability.yaml`. Lý do thiết kế: `docs/khac-biet-so-voi-bai-goc.md` mục 4.7.

*Trạng thái: đã viết, chưa chạy trên Colab.*
