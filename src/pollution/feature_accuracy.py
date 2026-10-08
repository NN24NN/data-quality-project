"""Feature Accuracy polluter: nhiễu Gaussian cho feature số, đổi giá trị cho feature one-hot."""
import numpy as np

from profiling.common import split_features

from ._helpers import add_gaussian_noise, n_polluted


def pollute(df, level, seed, target, target_type=None, onehot_prefixes=()):
    """Feature số: cộng nhiễu lên mọi dòng. Feature one-hot: tỷ lệ `level` số dòng đổi sang giá trị khác."""
    out = df.copy()
    if level == 0:
        return out
    n = len(out)
    numeric, groups = split_features(out.columns, target, onehot_prefixes)
    rng = np.random.default_rng(seed)
    for c in numeric:
        out[c] = add_gaussian_noise(out[c].to_numpy(dtype='float64'), level, rng)
    for cols in groups.values():
        m = len(cols)
        rows = rng.choice(n, n_polluted(level, n), replace=False)
        bits = out[cols].to_numpy(copy=True)
        new = (bits[rows].argmax(axis=1) + rng.integers(1, m, len(rows))) % m     # luôn khác giá trị cũ
        bits[rows] = np.eye(m, dtype=bits.dtype)[new]
        out[cols] = bits
    return out
