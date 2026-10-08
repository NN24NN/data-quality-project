"""Công thức điểm chất lượng dùng chung cho bản pandas và bản PySpark.

Hai engine chỉ tính thống kê thô (số ô thiếu, tổng khoảng cách, số dòng phân biệt, số dòng mỗi lớp);
mọi phép quy đổi thống kê thô -> điểm nằm ở đây, nên hai bản luôn cùng công thức toán học.

Định nghĩa theo Mohammed et al. (2025), mục 3 — xem docs/thiet-ke-thi-nghiem-chi-tiet.md mục 2.
"""

PROFILE_KEYS = [
    'profile_completeness',
    'profile_feature_accuracy',
    'profile_target_accuracy',
    'profile_uniqueness',
    'profile_class_balance',
]

NAN = float('nan')


def split_features(columns, target, onehot_prefixes=(), id_col=None):
    """Tách cột feature thành (numeric_cols, groups).

    groups: {prefix: [cột one-hot]} — mỗi nhóm là 1 feature categorical.
    """
    groups = {p: [] for p in onehot_prefixes}
    numeric = []
    for c in columns:
        if c == target or c == id_col:
            continue
        for p in onehot_prefixes:
            if c.startswith(p):
                groups[p].append(c)
                break
        else:
            numeric.append(c)
    empty = [p for p, cols in groups.items() if not cols]
    if empty:
        raise ValueError(f"Không có cột nào khớp tiền tố one-hot: {empty}")
    return numeric, groups


def completeness(missing_counts, n):
    """Completeness(d) = 1 - (1/f) * sum(missing(c_i)), missing(c_i) = số ô thiếu / n."""
    if n == 0 or not missing_counts:
        return NAN
    return 1.0 - sum(m / n for m in missing_counts) / len(missing_counts)


def numeric_accuracy(sum_abs_dist, n_present, scale):
    """nFAcc(c) = 1 - avg_dist(c) / scale(c), cắt dưới tại 0.

    Khác bài gốc ở mẫu số: scale = mean(|gt|) thay cho mean(gt), vì mean(gt) ~ 0 hoặc âm
    (vd các cột eta/phi của HIGGS) làm công thức gốc vô nghĩa. Với cột dương thì hai cách trùng nhau.
    """
    if n_present == 0:
        return 1.0
    avg_dist = sum_abs_dist / n_present
    if scale == 0:
        return 1.0 if avg_dist == 0 else 0.0
    return max(0.0, 1.0 - avg_dist / scale)


def categorical_accuracy(mismatches, n_present):
    """cFAcc(c) = 1 - mismatches(c) / n."""
    if n_present == 0:
        return 1.0
    return 1.0 - mismatches / n_present


def feature_accuracy(numeric_accs, categorical_accs):
    """Trung bình của nFAccuracy(d) và cFAccuracy(d) (mỗi cái là trung bình theo feature cùng loại)."""
    means = [sum(a) / len(a) for a in (numeric_accs, categorical_accs) if a]
    return sum(means) / len(means) if means else NAN


def uniqueness(n_distinct, n):
    """Uniqueness(d) = (unique_samples(d) - 1) / (n - 1)."""
    if n <= 1:
        return 1.0
    return (n_distinct - 1) / (n - 1)


def class_balance(counts, n_classes=None):
    """Balance(d) = 1 - ImBalance(d) / eps.

    ImBalance = tổng |n_i - n_j| trên mọi cặp lớp; eps = ceil(m/2) * floor(m/2) * n_cmax,
    n_cmax là kích thước lớp lớn nhất quan sát được. n_classes: số lớp thật, để tính cả lớp đã bị xóa hết.
    """
    counts = list(counts)
    if n_classes is not None:
        counts += [0] * (n_classes - len(counts))
    m = len(counts)
    eps = ((m + 1) // 2) * (m // 2) * max(counts, default=0)
    if eps == 0:
        return 1.0
    imbalance = sum(abs(counts[i] - counts[j]) for i in range(m) for j in range(i + 1, m))
    return 1.0 - imbalance / eps
