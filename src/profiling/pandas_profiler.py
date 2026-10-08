"""Profiling bằng pandas — tính thống kê thô, công thức điểm nằm ở common.py."""
import numpy as np

from . import common


def _group_missing(df, cols):
    """Feature one-hot bị thiếu ở 1 dòng: có bit null, hoặc cả nhóm bằng 0."""
    g = df[cols]
    return (g.isna().any(axis=1) | (g.sum(axis=1) == 0)).to_numpy()


def _numeric_accuracy(v, gt, scale_source):
    v = v.to_numpy(dtype='float64')
    gt = gt.to_numpy(dtype='float64')
    present = ~np.isnan(v)
    sum_abs_dist = float(np.abs(gt[present] - v[present]).sum())
    scale = float(np.abs(scale_source.to_numpy(dtype='float64')).mean())
    return common.numeric_accuracy(sum_abs_dist, int(present.sum()), scale)


def compute_profile(df, target, target_type, clean=None, onehot_prefixes=(), n_classes=None):
    """Trả về dict 5 điểm chất lượng (khóa theo common.PROFILE_KEYS).

    clean: bản sạch làm ground truth cho Feature/Target Accuracy; dòng của df được ghép với
    clean theo index (dòng nhân bản phải giữ index của dòng gốc). Không có clean -> NaN.
    Class Balance chỉ tính khi target_type == 'categorical', ngược lại NaN.
    """
    numeric, groups = common.split_features(df.columns, target, onehot_prefixes)
    n = len(df)

    missing = [int(df[c].isna().sum()) for c in numeric]
    missing += [int(_group_missing(df, cols).sum()) for cols in groups.values()]

    profile = dict.fromkeys(common.PROFILE_KEYS, common.NAN)
    profile['profile_completeness'] = common.completeness(missing, n)
    profile['profile_uniqueness'] = common.uniqueness(len(df.drop_duplicates()), n)

    if target_type == 'categorical':
        counts = df[target].value_counts().tolist()
        profile['profile_class_balance'] = common.class_balance(counts, n_classes)

    if clean is not None:
        gt = clean.loc[df.index]

        numeric_accs = [_numeric_accuracy(df[c], gt[c], clean[c]) for c in numeric]
        categorical_accs = []
        for cols in groups.values():
            present = ~_group_missing(df, cols)
            mismatch = (df[cols].to_numpy() != gt[cols].to_numpy()).any(axis=1)
            categorical_accs.append(
                common.categorical_accuracy(int((present & mismatch).sum()), int(present.sum())))
        profile['profile_feature_accuracy'] = common.feature_accuracy(numeric_accs, categorical_accs)

        if target_type == 'numeric':
            profile['profile_target_accuracy'] = _numeric_accuracy(df[target], gt[target], clean[target])
        else:
            present = df[target].notna().to_numpy()
            mismatch = df[target].to_numpy() != gt[target].to_numpy()
            profile['profile_target_accuracy'] = common.categorical_accuracy(
                int((present & mismatch).sum()), int(present.sum()))

    return profile
