# Data Quality Project

**Đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết tác vụ, dự đoán mức suy giảm hiệu năng mô hình downstream

Đồ án môn Xử Lý Data. Toàn bộ tài liệu thiết kế nằm trong `docs/`, bắt đầu từ `docs/checklist-trien-khai.md` để theo dõi tiến độ từng bước.

## Trạng thái

Giai đoạn 0–8 đã xong và đã kiểm chứng trên Google Colab (cập nhật 2026-10-26, `134 passed`):
- 0–4: thiết kế, dữ liệu (kể cả HIGGS full), module Profiling, Pollution, ML downstream.
- 5: Experiment Runner — đủ 312 dòng kết quả.
- 6: meta-model (Gradient Boosting + SHAP) và 3 baseline (rule-based, weighted score, DQSOps-PCA).
- 7: đánh giá & so sánh — meta-model ước lượng và xếp hạng mức suy giảm tốt nhất ở cả 3 tác vụ; hơn DQSOps-PCA có ý nghĩa thống kê.
- 8: scalability — pandas hết RAM ở 22M dòng, PySpark chạy được trong giới hạn 4 GB.

Tiếp theo: Giai đoạn 9 (biểu đồ H1–H10 và báo cáo) — xem `docs/checklist-trien-khai.md`. Số liệu: `docs/ket-qua-thuc-nghiem.md`; khác biệt so với bài gốc: `docs/khac-biet-so-voi-bai-goc.md`.

## Cấu trúc thư mục

Xem chi tiết tại `docs/cau-truc-thu-muc-project.md`.

## Cài đặt môi trường

```
pip install -r requirements.txt
```

Bỏ comment các dòng `pyspark`, `shap`, `xgboost` trong `requirements.txt` khi cần (xem ghi chú trong file).
