"""Baseline 1 — Rule-based: Bad (1) nếu bất kỳ dimension áp dụng được có điểm < ngưỡng, ngược lại Good (0).

Theo docs/kien-truc-mo-hinh-va-ky-thuat.md mục 5.1. Ngưỡng cố định (configs/metamodel.yaml), không học từ dữ liệu.
Dimension không áp dụng (NaN) bị bỏ qua — docs/khac-biet-so-voi-bai-goc.md mục 4.5.
Chỉ cho nhãn Good/Bad, không dự đoán Δ liên tục: Giai đoạn 7 so sánh bằng accuracy/F1 thay vì MAE/R².
"""
from profiling.common import PROFILE_KEYS


class RuleBased:
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, results, y=None):
        return self                                   # ngưỡng cố định, không có gì để học

    def predict(self, results):
        # NaN < ngưỡng là False -> dimension không áp dụng không bao giờ làm dòng thành Bad
        return (results[PROFILE_KEYS] < self.threshold).any(axis=1).astype(float).to_numpy()
