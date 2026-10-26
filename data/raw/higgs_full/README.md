# data/raw/higgs_full/

HIGGS full 11M dòng — chỉ cần cho Giai đoạn 8 (Spark scalability study). Không tải về máy Windows.

Trên Drive: `higgs_full_raw.parquet` (~1,2GB), tạo bằng notebook cell 15.2 (`src/scalability/spark_scale_study.py prepare`): tải `HIGGS.csv.gz` (~2,8GB) từ UCI về ổ tạm `/content`, đổi sang parquet giữ nguyên nội dung (cột `target` int8, `f01`–`f28` float32, cùng tên cột với HIGGS mẫu). File `.gz` không lưu lên Drive. Các mốc dùng để đo được cắt ra `/content` đầu mỗi phiên, không lưu ở đây.

Giai đoạn 8 đã xong (2026-10-26); người dùng quyết định **giữ** file này trên Drive để đo lại được nếu cần. Kết quả đo nằm trong `results/scalability_benchmark.csv`.

*Trạng thái: đã tải ngày 2026-10-26 (cell 15.2): 11.000.000 dòng, 0,69GB trên Drive.*

Mốc 22M dùng ở cell 15.4 là HIGGS full ghép nối tiếp 2 lần, chỉ tạo ở `/content`, không lưu ở đây.
