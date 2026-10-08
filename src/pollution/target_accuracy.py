"""Target Accuracy polluter: nhiễu nhãn — nhiễu Gaussian (target số) hoặc đổi nhãn (target phân loại)."""
import numpy as np

from ._helpers import add_gaussian_noise, flip_labels, n_polluted


def pollute(df, level, seed, target, target_type, onehot_prefixes=()):
    out = df.copy()
    if level == 0:
        return out
    rng = np.random.default_rng(seed)
    if target_type == 'numeric':
        out[target] = add_gaussian_noise(out[target].to_numpy(dtype='float64'), level, rng)
    else:
        rows = rng.choice(len(out), n_polluted(level, len(out)), replace=False)
        out[target] = flip_labels(out[target].to_numpy(), rows, rng)
    return out
