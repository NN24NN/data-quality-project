# data/raw/higgs_full/

HIGGS full 11M dòng — chỉ cần cho Giai đoạn 8 (Spark scalability study). Không tải về máy Windows.

Trên Drive: `higgs_full_raw.parquet` (~1,2GB), tạo bằng notebook cell 15.2 (`src/scalability/spark_scale_study.py prepare`): tải `HIGGS.csv.gz` (~2,8GB) từ UCI về ổ tạm `/content`, đổi sang parquet giữ nguyên nội dung (cột `target` int8, `f01`–`f28` float32, cùng tên cột với HIGGS mẫu). File `.gz` không lưu lên Drive. Các mốc dùng để đo được cắt ra `/content` đầu mỗi phiên, không lưu ở đây.

Sau khi xong Giai đoạn 8 có thể xóa để giải phóng dung lượng Drive (hỏi người dùng trước); kết quả đo đã nằm trong `results/scalability_benchmark.csv`.

*Trạng thái: chưa tải (chờ chạy cell 15.2).*
