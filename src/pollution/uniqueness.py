"""Uniqueness polluter: nhân bản dòng theo hệ số rho = 1 / (1 - level), nên Uniqueness ~ 1 - level."""
import numpy as np

from ._helpers import n_polluted


def _pick_duplicates(positions, n_extra, rng, dup_mean, dup_std):
    """Chọn n_extra bản sao từ `positions`; số bản sao của mỗi dòng được chọn ~ |N(dup_mean, dup_std)|, tối thiểu 1."""
    if n_extra == 0:
        return positions[:0]
    order = rng.permutation(positions)
    counts = np.maximum(1, np.rint(np.abs(rng.normal(dup_mean, dup_std, len(order))))).astype(int)
    return np.resize(np.repeat(order, counts), n_extra)      # cắt bớt, hoặc quay vòng nếu chưa đủ


def pollute(df, level, seed, target, target_type, onehot_prefixes=(), dup_mean=1.0, dup_std=5.0):
    """Bản sao giữ index của dòng gốc (profiler dựa vào đó để ghép với bản sạch).

    Target phân loại: mỗi lớp được nhân bản cùng hệ số, tỷ lệ lớp không đổi.
    """
    if not 0 <= level < 1:
        raise ValueError("level phải nằm trong [0, 1)")
    rho = 1.0 / (1.0 - level)
    rng = np.random.default_rng(seed)
    all_rows = np.arange(len(df))
    if target_type == 'categorical':
        y = df[target].to_numpy()
        blocks = [all_rows[y == cls] for cls in np.unique(y)]
    else:
        blocks = [all_rows]
    extra = [_pick_duplicates(b, n_polluted(rho - 1.0, len(b)), rng, dup_mean, dup_std) for b in blocks]
    return df.iloc[np.concatenate([all_rows, *extra])]
