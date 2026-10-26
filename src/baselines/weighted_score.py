"""Baseline 2 — Weighted score: Q = trung bình các điểm profile (trọng số bằng nhau), hồi quy tuyến tính Q -> Δ.

Theo docs/kien-truc-mo-hinh-va-ky-thuat.md mục 5.2. Một công thức chung cho mọi tác vụ (không biết task).
Dimension không áp dụng (NaN) bị bỏ khỏi trung bình — regression lấy trung bình 4 điểm
(docs/khac-biet-so-voi-bai-goc.md mục 4.5).
"""
from sklearn.linear_model import LinearRegression

from profiling.common import PROFILE_KEYS


def quality_score(results):
    return results[PROFILE_KEYS].mean(axis=1, skipna=True).to_numpy().reshape(-1, 1)


class WeightedScore:
    def fit(self, results, y):
        self.reg_ = LinearRegression().fit(quality_score(results), y)
        return self

    def predict(self, results):
        return self.reg_.predict(quality_score(results))
