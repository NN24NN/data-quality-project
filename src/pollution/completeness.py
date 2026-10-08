"""Completeness polluter: chèn ô thiếu kiểu MCAR, mỗi feature thiếu đúng tỷ lệ `level`."""
import numpy as np

from profiling.common import split_features

from ._helpers import n_polluted


def pollute(df, level, seed, target, target_type=None, onehot_prefixes=()):
    """Feature số -> NaN; feature one-hot -> cả nhóm về 0. Target không bị đụng tới.

    Mỗi feature chọn dòng thiếu độc lập (MCAR theo từng ô), nên Completeness = 1 - level.
    """
    out = df.copy()
    n = len(out)
    k = n_polluted(level, n)
    if k == 0:
        return out
    numeric, groups = split_features(out.columns, target, onehot_prefixes)
    rng = np.random.default_rng(seed)
    for c in numeric:
        values = out[c].to_numpy(dtype='float64', copy=True)   # cột số nguyên phải sang float mới chứa được NaN
        values[rng.choice(n, k, replace=False)] = np.nan
        out[c] = values
    for cols in groups.values():
        bits = out[cols].to_numpy(copy=True)
        bits[rng.choice(n, k, replace=False)] = 0
        out[cols] = bits
    return out
