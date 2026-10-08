# Data Quality Project

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

Đồ án môn Xử Lý Data. Toàn bộ tài liệu thiết kế nằm trong `docs/`, bắt đầu từ `docs/checklist-trien-khai.md` để theo dõi tiến độ từng bước.

## Trạng thái

Khung thư mục đã được tạo (Giai đoạn 0 — thiết kế — đã hoàn tất). Thí nghiệm chính (Giai đoạn 1 trở đi) dự kiến chạy trên Google Colab — xem `docs/checklist-trien-khai.md`.

## Cấu trúc thư mục

Xem chi tiết tại `docs/cau-truc-thu-muc-project.md`.

## Cài đặt môi trường

```
pip install -r requirements.txt
```

Bỏ comment các dòng `pyspark`, `shap`, `xgboost` trong `requirements.txt` khi cần (xem ghi chú trong file).
