"""Clustering downstream: k-Means + Gaussian Mixture, đo AMI so với nhãn thật."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from .common import with_preprocessing, xy

# (lớp mô hình, tên tham số số cụm)
MODELS = {'kmeans': (KMeans, 'n_clusters'), 'gmm': (GaussianMixture, 'n_components')}


def evaluate(algorithm, train, test, target, seed, params):
    """Phân cụm toàn bộ `train`, trả về AMI giữa cụm tìm được và target.

    Clustering không tách train/test nên `test` không dùng (giữ tham số để cùng chữ ký với hai tác vụ kia).
    Target không đưa vào mô hình, chỉ dùng để chấm điểm; số cụm = số lớp thật của target.
    """
    X, y = xy(train, target)
    cls, n_arg = MODELS[algorithm]
    model = with_preprocessing(cls(**{n_arg: len(np.unique(y))}, random_state=seed, **params))
    return float(adjusted_mutual_info_score(y, model.fit_predict(X)))
