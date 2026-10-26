"""Baseline 3 — DQSOps-style: z-score từng điểm profile -> PCA, lấy PC1 làm quality score -> hồi quy tuyến tính PC1 -> Δ.

Tái hiện cách DQSOps (Bayram et al., 2023) tổng hợp nhiều dimension thành một điểm — docs/kien-truc-mo-hinh-va-ky-thuat.md mục 5.3.
Một điểm chung cho mọi tác vụ (không biết task), đúng như DQSOps.

- Dimension không áp dụng (NaN) -> điền `fill_value` (1.0, configs/metamodel.yaml), như meta-model.
- Scaler và PCA chỉ fit trên dữ liệu train của fold (docs/khac-biet-so-voi-bai-goc.md mục 4.3).
- Dấu của PC1 là tùy ý; đảo dấu để tổng trọng số dương, tức điểm cao = chất lượng cao, dễ đọc ở biểu đồ H9.
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from profiling.common import PROFILE_KEYS


class DQSOpsPCA:
    def __init__(self, fill_value):
        self.fill_value = fill_value

    def _matrix(self, results):
        return results[PROFILE_KEYS].fillna(self.fill_value).to_numpy()

    def fit(self, results, y):
        X = self._matrix(results)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=1, svd_solver='full').fit(self.scaler_.transform(X))
        self.sign_ = 1.0 if self.pca_.components_[0].sum() >= 0 else -1.0
        self.reg_ = LinearRegression().fit(self.score(results).reshape(-1, 1), y)
        return self

    def score(self, results):
        """Quality score tổng hợp (PC1) — tương đương điểm chất lượng của DQSOps."""
        return self.sign_ * self.pca_.transform(self.scaler_.transform(self._matrix(results)))[:, 0]

    def predict(self, results):
        return self.reg_.predict(self.score(results).reshape(-1, 1))

    def weights(self):
        """Trọng số PC1 của từng dimension (dùng cho H9) và tỷ lệ phương sai PC1 giải thích được."""
        return (pd.Series(self.sign_ * self.pca_.components_[0], index=PROFILE_KEYS, name='pc1_weight'),
                float(self.pca_.explained_variance_ratio_[0]))
