"""Thao tác làm bẩn dùng chung cho feature_accuracy và target_accuracy."""
import numpy as np


def add_gaussian_noise(values, level, rng):
    """noise = X * mean(|gt|), X ~ N(0, sigma^2 = level).

    Phương sai bằng level như bài gốc; thang nhân là mean(|gt|) thay cho mean(gt),
    khớp với mẫu số của profiling.common.numeric_accuracy.
    """
    scale = np.abs(values).mean()
    return values + rng.normal(0.0, np.sqrt(level), len(values)) * scale


def flip_labels(values, rows, rng):
    """Đổi nhãn ở các dòng `rows` sang một nhãn KHÁC, chọn đều trong các nhãn còn lại."""
    classes = np.unique(values)
    current = np.searchsorted(classes, values[rows])
    out = values.copy()
    out[rows] = classes[(current + rng.integers(1, len(classes), len(rows))) % len(classes)]
    return out


def n_polluted(level, n):
    return int(round(level * n))
