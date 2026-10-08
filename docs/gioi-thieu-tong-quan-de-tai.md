# Giới thiệu tổng quan đề tài

**Tên đề tài:** Xây dựng khung đánh giá chất lượng dữ liệu quy mô lớn có nhận biết ngữ cảnh, gắn với hiệu năng mô hình downstream

**Môn học:** Xử Lý Data

---

## 1. Vấn đề đặt ra

Chất lượng dữ liệu ảnh hưởng trực tiếp đến hiệu năng mô hình học máy, nhưng mức độ ảnh hưởng đó **khác nhau tùy theo loại tác vụ** (classification/regression/clustering) và **loại thuật toán** (tuyến tính/cây). Phần lớn công cụ đo chất lượng dữ liệu hiện nay (vd DQSOps) lại tổng hợp nhiều chiều chất lượng thành **một điểm số duy nhất, không phân biệt theo ngữ cảnh sử dụng**, và điểm số này không gắn với hiệu năng thật của mô hình downstream.

Đề tài dựa trên 2 bài báo nền tảng — xem chi tiết tại [`tong-quan-bai-goc.md`](./tong-quan-bai-goc.md):
- **Mohammed et al. (2025)** — chứng minh bằng thực nghiệm rằng ảnh hưởng của chất lượng dữ liệu là task-dependent, nhưng chỉ dừng ở mức quan sát, không xây mô hình dự đoán.
- **DQSOps (Bayram et al., 2023)** — xây được mô hình dự đoán nhanh, nhưng nhãn huấn luyện là một quality score tự định nghĩa (PCA composite), không gắn với hiệu năng ML downstream thật, và không phân biệt theo tác vụ.

## 2. Khoảng trống & đóng góp của đề tài

Chưa có công trình nào học một **mô hình dự đoán mức suy giảm hiệu năng ML downstream thật** (không phải quality score tự định nghĩa) từ một profile chất lượng đa chiều, **có phân biệt theo loại tác vụ**, và có đánh giá khả năng mở rộng ở quy mô lớn.

Đề tài đóng góp:
1. **Nhãn huấn luyện lấy từ downstream thật** — ΔF1 (classification), ΔR² (regression), ΔAMI (clustering) so với baseline sạch.
2. **Mô hình task-conditioned** — Gradient Boosting + SHAP để rút ra "trọng số" từng dimension chất lượng theo từng tác vụ, tường minh và giải thích được (khác PCA ẩn của DQSOps).
3. **Profiling có khả năng mở rộng bằng Spark** — minh họa xu hướng runtime/memory khi dữ liệu tăng quy mô.
4. **So sánh với 3 baseline**: rule-based, weighted score, và DQSOps-style (PCA composite) — để chứng minh lợi ích của việc gắn nhãn với hiệu năng downstream thật.

## 3. Phạm vi thí nghiệm (đã chốt)

| Thành phần | Lựa chọn |
|---|---|
| Dataset | Beijing Multi-Site Air Quality (regression), HIGGS mẫu + full (classification + scalability), Covertype (clustering) |
| Dimension chất lượng | Completeness, Feature Accuracy, Target Accuracy, Uniqueness, (tùy chọn) Target Class Balance |
| Mức độ ô nhiễm | 0.0, 0.2, 0.5, 0.8 |
| Scenario | Scenario 3 — làm bẩn cả train và test (tình huống thực tế nhất) |
| Thuật toán | 2 thuật toán/tác vụ: LogR + RF (classification), Ridge + GB (regression), k-Means + GMM (clustering) |
| Khối lượng thí nghiệm | 312 lượt chạy (96 regression + 120 classification + 96 clustering) |
| Meta-model | Gradient Boosting Regressor + SHAP |
| Baseline so sánh | Rule-based, Weighted score, DQSOps-style (PCA) |
| Scalability | PySpark, profiling trên HIGGS full ở các mốc 1M–11M dòng |

Chi tiết đầy đủ: [`pipeline-thi-nghiem.md`](./pipeline-thi-nghiem.md), [`thiet-ke-thi-nghiem-chi-tiet.md`](./thiet-ke-thi-nghiem-chi-tiet.md), [`kien-truc-mo-hinh-va-ky-thuat.md`](./kien-truc-mo-hinh-va-ky-thuat.md).

## 4. Hạ tầng triển khai

- **Máy thực thi chính:** laptop Windows cá nhân (không dùng máy công ty).
- **Google Colab:** dùng cho các bước nặng CPU/dung lượng — tải dataset lớn, chạy Experiment Runner (312 lượt), Spark scalability study — để tránh chiếm dụng ổ cứng máy local.
- **VS Code (local):** dùng để viết code từng module (`src/`), file config nhẹ, chỉnh sửa báo cáo.

## 5. Trạng thái hiện tại

Giai đoạn 0 (thiết kế) đã hoàn tất. Giai đoạn 1 (dữ liệu & môi trường) gần xong: 3 dataset đã tải, 3 bản sạch đã tạo trên Google Drive, config đã có; còn HIGGS full (làm sau). Xem tiến độ chi tiết và các bước tiếp theo tại [`checklist-trien-khai.md`](./checklist-trien-khai.md).

## 6. Cấu trúc thư mục & quy tắc làm việc

Xem [`cau-truc-thu-muc-project.md`](./cau-truc-thu-muc-project.md) cho cấu trúc thư mục, và `CLAUDE.md` ở thư mục gốc project cho các quy tắc Claude cần tuân theo khi làm việc trên project này.
