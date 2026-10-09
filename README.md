# Data Quality Project

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

Đồ án môn Xử Lý Data. Toàn bộ tài liệu thiết kế nằm trong `docs/`, bắt đầu từ `docs/checklist-trien-khai.md` để theo dõi tiến độ từng bước.

## Trạng thái

Giai đoạn 0 (thiết kế), 1 (dữ liệu, trừ HIGGS full), 2 (module Profiling) và 3 (module Pollution) đã xong và đã kiểm chứng trên Google Colab. Giai đoạn 4 (module ML downstream) đã viết code và test, chưa chạy kiểm chứng — xem `docs/checklist-trien-khai.md`.

## Cấu trúc thư mục

Xem chi tiết tại `docs/cau-truc-thu-muc-project.md`.

## Cài đặt môi trường

```
pip install -r requirements.txt
```

Bỏ comment các dòng `pyspark`, `shap`, `xgboost` trong `requirements.txt` khi cần (xem ghi chú trong file).
