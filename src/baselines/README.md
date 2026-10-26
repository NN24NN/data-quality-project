# src/baselines/

3 baseline so sánh với meta-model: `rule_based.py`, `weighted_score.py`, `dqsops_pca.py` (quan trọng nhất — tái hiện đúng cách DQSOps tổng hợp bằng PCA).

Cùng giao diện `fit(results, y)` / `predict(results)` với meta-model, nhận thẳng bảng kết quả thí nghiệm. Weighted score và DQSOps-PCA dùng **một công thức chung cho mọi tác vụ** (không biết task), đúng tinh thần DQSOps.

| Baseline | Cách làm | Dimension không áp dụng (NaN) |
|---|---|---|
| `RuleBased` | Bad (1) nếu bất kỳ điểm < 0,9, ngược lại Good (0); không học | Bỏ qua |
| `WeightedScore` | Q = trung bình điểm profile, hồi quy tuyến tính Q → Δ | Bỏ khỏi trung bình |
| `DQSOpsPCA` | z-score → PCA → PC1 (đảo dấu để điểm cao = tốt) → hồi quy tuyến tính | Điền 1,0 |

Ngưỡng và giá trị điền: `configs/metamodel.yaml`. Lý do: `docs/khac-biet-so-voi-bai-goc.md` mục 4.5.

*Trạng thái: đã viết, chưa chạy test trên Colab.*
