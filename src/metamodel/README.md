# src/metamodel/

`train_metamodel.py` (Gradient Boosting Regressor dự đoán ΔPerformance) và `explain_shap.py` (tính SHAP, xuất bảng trọng số task-aware theo từng tác vụ).

- **Đặc trưng (14):** 5 điểm profile (NaN → 1,0) + one-hot 3 tác vụ + one-hot 6 thuật toán.
- **Nhãn:** Δ tuyệt đối (chính) và Δ tương đối = Δ / baseline (phụ).
- **Đánh giá chéo:** `StratifiedGroupKFold` 5 fold, nhóm = (task, dimension, λ), λ = 0 một nhóm mỗi task; lặp với 3 seed; thêm thí nghiệm nội suy (test λ = 0,5). 3 baseline fit lại trên cùng fold.
- **Output (`results/`):** `metamodel_cv_predictions.parquet` (dự đoán out-of-fold, dạng dài), `metamodel.pkl` (mô hình cuối cho mỗi nhãn), `dqsops_pca_weights.csv`, `shap_importance.csv`.
- Tham số: `configs/metamodel.yaml`. Lý do thiết kế: `docs/khac-biet-so-voi-bai-goc.md` mục 4.3–4.5. Chạy: notebook mục 13.

*Trạng thái: xong, đã chạy trên Colab ngày 2026-10-26 (`116 passed`). Kết quả: `docs/ket-qua-thuc-nghiem.md` mục 6.*

**Giai đoạn 7 — `evaluate_methods.py`:** đọc `metamodel_cv_predictions.parquet` (không huấn luyện lại), ghi `method_comparison.csv` (MAE / R² / Spearman, trung bình ± độ lệch chuẩn qua 3 seed), `method_bootstrap.csv` (hiệu MAE meta-model − baseline, CI bootstrap theo nhóm), `goodbad_comparison.csv` (Good/Bad của 4 phương pháp, 3 ngưỡng). Mỗi bảng có cột `subset`: `all` (312 lượt, bảng chính) và `excl_clustering_class_balance` (288 lượt). `explain_shap.py` ghi thêm `shap_by_row_dimension.csv` (chẩn đoán). Chạy: notebook mục 14. *Xong, đã chạy trên Colab ngày 2026-10-26 (`124 passed`). Kết quả: `docs/ket-qua-thuc-nghiem.md` mục 7.*
