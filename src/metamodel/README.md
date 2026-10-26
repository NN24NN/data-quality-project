# src/metamodel/

`train_metamodel.py` (Gradient Boosting Regressor dự đoán ΔPerformance) và `explain_shap.py` (tính SHAP, xuất bảng trọng số task-aware theo từng tác vụ).

- **Đặc trưng (14):** 5 điểm profile (NaN → 1,0) + one-hot 3 tác vụ + one-hot 6 thuật toán.
- **Nhãn:** Δ tuyệt đối (chính) và Δ tương đối = Δ / baseline (phụ).
- **Đánh giá chéo:** `StratifiedGroupKFold` 5 fold, nhóm = (task, dimension, λ), λ = 0 một nhóm mỗi task; lặp với 3 seed; thêm thí nghiệm nội suy (test λ = 0,5). 3 baseline fit lại trên cùng fold.
- **Output (`results/`):** `metamodel_cv_predictions.parquet` (dự đoán out-of-fold, dạng dài), `metamodel.pkl` (mô hình cuối cho mỗi nhãn), `dqsops_pca_weights.csv`, `shap_importance.csv`.
- Tham số: `configs/metamodel.yaml`. Lý do thiết kế: `docs/khac-biet-so-voi-bai-goc.md` mục 4.3–4.5. Chạy: notebook mục 13.

*Trạng thái: xong, đã chạy trên Colab ngày 2026-10-26 (`116 passed`). Kết quả: `docs/ket-qua-thuc-nghiem.md` mục 6.*
