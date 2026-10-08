"""Profiling bằng PySpark — cùng công thức với pandas_profiler (đều gọi common.py), khác engine."""
from functools import reduce
from operator import add, or_

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

from . import common


def _col(name):
    """Tên cột có dấu chấm (vd 'PM2.5') phải bọc backtick."""
    return F.col(f'`{name}`')


def _missing(df, c):
    """Ô thiếu: null, hoặc NaN với cột số thực (NaN của pandas sang Spark không thành null)."""
    cond = _col(c).isNull()
    if isinstance(df.schema[c].dataType, (FloatType, DoubleType)):
        cond = cond | F.isnan(_col(c))
    return cond


def _group_missing(df, cols):
    """Feature one-hot bị thiếu ở 1 dòng: có bit null, hoặc cả nhóm bằng 0."""
    any_null = reduce(or_, [_missing(df, c) for c in cols])
    return any_null | (reduce(add, [_col(c) for c in cols]) == 0)


def _count(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def compute_profile(df, target, target_type, clean=None, onehot_prefixes=(), n_classes=None,
                    id_col=None):
    """Trả về dict 5 điểm chất lượng (khóa theo common.PROFILE_KEYS).

    clean: DataFrame sạch làm ground truth cho Feature/Target Accuracy; ghép với df qua id_col
    (dòng nhân bản phải giữ id của dòng gốc; mọi id của df phải có trong clean). Không có clean -> NaN.
    Class Balance chỉ tính khi target_type == 'categorical', ngược lại NaN.
    """
    numeric, groups = common.split_features(df.columns, target, onehot_prefixes, id_col)
    group_cols = list(groups.values())
    data_cols = [c for c in df.columns if c != id_col]

    row = df.agg(
        F.count(F.lit(1)).alias('n'),
        *[_count(_missing(df, c)).alias(f'm{i}') for i, c in enumerate(numeric)],
        *[_count(_group_missing(df, cols)).alias(f'g{i}') for i, cols in enumerate(group_cols)],
    ).first()
    n = row['n']
    missing = [row[f'm{i}'] for i in range(len(numeric))] + [row[f'g{i}'] for i in range(len(group_cols))]

    profile = dict.fromkeys(common.PROFILE_KEYS, common.NAN)
    profile['profile_completeness'] = common.completeness(missing, n)
    n_distinct = df.select([_col(c) for c in data_cols]).distinct().count()
    profile['profile_uniqueness'] = common.uniqueness(n_distinct, n)

    if target_type == 'categorical':
        counts = [r['count'] for r in df.groupBy(_col(target)).count().collect() if r[0] is not None]
        profile['profile_class_balance'] = common.class_balance(counts, n_classes)

    if clean is not None:
        if id_col is None:
            raise ValueError("Cần id_col để ghép df với clean")
        gt_name = {c: f'__gt{i}' for i, c in enumerate(data_cols)}
        gt = clean.select(_col(id_col), *[_col(c).alias(gt_name[c]) for c in data_cols])
        joined = df.join(gt, on=id_col, how='inner')

        def dist(c):
            return F.abs(_col(c).cast('double') - F.col(gt_name[c]).cast('double'))

        numeric_target = [target] if target_type == 'numeric' else []
        num_cols = numeric + numeric_target
        aggs = []
        for i, c in enumerate(num_cols):
            present = ~_missing(df, c)
            aggs += [F.sum(F.when(present, dist(c))).alias(f's{i}'), _count(present).alias(f'p{i}')]
        for i, cols in enumerate(group_cols):
            present = ~_group_missing(df, cols)
            mismatch = reduce(or_, [_col(c) != F.col(gt_name[c]) for c in cols])
            aggs += [_count(present & mismatch).alias(f'gx{i}'), _count(present).alias(f'gp{i}')]
        if not numeric_target:
            present = ~_missing(df, target)
            aggs += [_count(present & (_col(target) != F.col(gt_name[target]))).alias('tx'),
                     _count(present).alias('tp')]
        stats = joined.agg(*aggs).first()
        scales = clean.agg(
            *[F.avg(F.abs(_col(c).cast('double'))).alias(f'a{i}') for i, c in enumerate(num_cols)]
        ).first() if num_cols else {}

        num_accs = [common.numeric_accuracy(stats[f's{i}'] or 0.0, stats[f'p{i}'], scales[f'a{i}'])
                    for i in range(len(num_cols))]
        categorical_accs = [common.categorical_accuracy(stats[f'gx{i}'], stats[f'gp{i}'])
                            for i in range(len(group_cols))]
        profile['profile_feature_accuracy'] = common.feature_accuracy(
            num_accs[:len(numeric)], categorical_accs)
        if numeric_target:
            profile['profile_target_accuracy'] = num_accs[-1]
        else:
            profile['profile_target_accuracy'] = common.categorical_accuracy(stats['tx'], stats['tp'])

    return profile
