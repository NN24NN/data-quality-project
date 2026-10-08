"""Class Balance polluter: giữ nguyên lớp lớn nhất, xóa tỷ lệ `level` số dòng của mọi lớp còn lại."""
import numpy as np

from ._helpers import n_polluted


def pollute(df, level, seed, target, target_type='categorical', onehot_prefixes=()):
    if target_type != 'categorical':
        raise ValueError("Class Balance chỉ áp dụng cho target phân loại")
    y = df[target].to_numpy()
    classes, counts = np.unique(y, return_counts=True)       # thứ tự lớp cố định -> tái lập được
    largest = classes[counts.argmax()]
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), dtype=bool)
    for cls in classes:
        if cls == largest:
            continue
        rows = np.flatnonzero(y == cls)
        keep[rng.choice(rows, n_polluted(level, len(rows)), replace=False)] = False
    return df[keep]
