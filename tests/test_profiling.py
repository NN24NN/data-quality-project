"""Test module profiling: ví dụ nhỏ có đáp án tính tay + đối chiếu pandas với PySpark.

Chạy: python -m pytest tests/test_profiling.py -q
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from profiling import common, pandas_profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIXES = ['g_']

# Đáp án tính tay cho cặp (CLEAN, POLLUTED) bên dưới:
#   completeness     : x thiếu 1/4, nhóm g_ thiếu 1/4 (dòng 3 toàn 0)      -> 1 - (0.25 + 0.25)/2 = 0.75
#   feature accuracy : x  — 3 ô có mặt, khoảng cách 0, 6, 0 -> avg 2; mean|gt| = 25 -> 1 - 2/25 = 0.92
#                      g_ — 3 dòng có mặt, dòng 1 sai                      -> 1 - 1/3
#                      trung bình 2 loại                                   -> (0.92 + 2/3)/2
#   target accuracy  : 1 nhãn sai / 4                                      -> 0.75
#   uniqueness       : 4 dòng đều khác nhau                                -> 1
#   class balance    : lớp 0 có 1 dòng, lớp 1 có 3 dòng -> ImBalance 2, eps = 1*1*3 -> 1 - 2/3
CLEAN = pd.DataFrame({
    'x':   [10.0, 20.0, 30.0, 40.0],
    'g_a': [1, 0, 1, 0],
    'g_b': [0, 1, 0, 1],
    'y':   [0, 0, 1, 1],
})
POLLUTED = pd.DataFrame({
    'x':   [10.0, np.nan, 36.0, 40.0],
    'g_a': [1, 1, 1, 0],
    'g_b': [0, 0, 0, 0],
    'y':   [0, 1, 1, 1],
})
EXPECTED = {
    'profile_completeness': 0.75,
    'profile_feature_accuracy': (0.92 + 2 / 3) / 2,
    'profile_target_accuracy': 0.75,
    'profile_uniqueness': 1.0,
    'profile_class_balance': 1 / 3,
}

# Target số, tên cột có dấu chấm như Beijing: |12-10| + 0 -> avg 1; mean|gt| = 15 -> 1 - 1/15
CLEAN_NUM = pd.DataFrame({'PM2.5': [10.0, 20.0], 'x': [1.0, 2.0]})
POLLUTED_NUM = pd.DataFrame({'PM2.5': [12.0, 20.0], 'x': [1.0, 2.0]})


def approx(expected):
    return pytest.approx(expected, rel=1e-9, abs=1e-12, nan_ok=True)


# ---------- công thức thuần (common) ----------

def test_class_balance_formula():
    assert common.class_balance([5, 5, 5]) == 1.0
    assert common.class_balance([4, 2], n_classes=3) == approx(0.0)   # |4-2|+|4-0|+|2-0| = 8 = eps
    assert common.class_balance([3, 1]) == approx(1 / 3)


def test_uniqueness_formula():
    assert common.uniqueness(3, 4) == approx(2 / 3)
    assert common.uniqueness(1, 1) == 1.0


def test_numeric_accuracy_clipped_and_zero_scale():
    assert common.numeric_accuracy(100.0, 2, 10.0) == 0.0     # 1 - 50/10 < 0 -> cắt về 0
    assert common.numeric_accuracy(0.0, 2, 0.0) == 1.0


# ---------- bản pandas ----------

def test_pandas_hand_example():
    got = pandas_profiler.compute_profile(POLLUTED, 'y', 'categorical', clean=CLEAN,
                                          onehot_prefixes=PREFIXES)
    assert got == approx(EXPECTED)


def test_pandas_clean_against_itself():
    got = pandas_profiler.compute_profile(CLEAN, 'y', 'categorical', clean=CLEAN,
                                          onehot_prefixes=PREFIXES)
    assert got == approx(dict.fromkeys(common.PROFILE_KEYS, 1.0))


def test_pandas_numeric_target():
    got = pandas_profiler.compute_profile(POLLUTED_NUM, 'PM2.5', 'numeric', clean=CLEAN_NUM)
    assert got['profile_target_accuracy'] == approx(1 - 1 / 15)
    assert got['profile_feature_accuracy'] == 1.0
    assert np.isnan(got['profile_class_balance'])


def test_pandas_duplicates_keep_origin_index():
    dup = CLEAN.loc[[0, 0, 1, 2]]                 # dòng 0 nhân đôi, giữ index gốc
    got = pandas_profiler.compute_profile(dup, 'y', 'categorical', clean=CLEAN,
                                          onehot_prefixes=PREFIXES)
    assert got['profile_uniqueness'] == approx(2 / 3)       # (3 - 1) / (4 - 1)
    assert got['profile_feature_accuracy'] == 1.0
    assert got['profile_target_accuracy'] == 1.0


def test_pandas_without_clean_gives_nan_accuracy():
    got = pandas_profiler.compute_profile(POLLUTED, 'y', 'categorical', onehot_prefixes=PREFIXES)
    assert np.isnan(got['profile_feature_accuracy']) and np.isnan(got['profile_target_accuracy'])
    assert got['profile_completeness'] == 0.75


# ---------- bản PySpark + đối chiếu với pandas ----------

@pytest.fixture(scope='module')
def spark():
    pytest.importorskip('pyspark')
    from pyspark.sql import SparkSession
    session = SparkSession.builder.master('local[2]').appName('test_profiling').getOrCreate()
    yield session
    session.stop()


def to_spark(spark, pdf):
    return spark.createDataFrame(pdf.rename_axis('row_id').reset_index())


def spark_profile(spark, df, clean, target, target_type, prefixes=()):
    from profiling import spark_profiler
    return spark_profiler.compute_profile(
        to_spark(spark, df), target, target_type,
        clean=None if clean is None else to_spark(spark, clean),
        onehot_prefixes=prefixes, id_col='row_id')


def test_spark_hand_example(spark):
    assert spark_profile(spark, POLLUTED, CLEAN, 'y', 'categorical', PREFIXES) == approx(EXPECTED)


def test_spark_numeric_target(spark):
    got = spark_profile(spark, POLLUTED_NUM, CLEAN_NUM, 'PM2.5', 'numeric')
    assert got['profile_target_accuracy'] == approx(1 - 1 / 15)
    assert got['profile_feature_accuracy'] == 1.0


@pytest.mark.parametrize('target_type', ['categorical', 'numeric'])
def test_pandas_spark_parity_random(spark, target_type):
    with open(os.path.join(ROOT, 'configs', 'seeds.yaml')) as f:
        rng = np.random.default_rng(yaml.safe_load(f)['base_seed'])
    n = 300
    onehot = np.eye(3, dtype='int8')[rng.integers(0, 3, n)]
    clean = pd.DataFrame({
        'a': rng.normal(50, 10, n),
        'b': rng.normal(0, 1, n).astype('float32'),          # trung bình ~ 0 như eta/phi của HIGGS
        'c': rng.integers(0, 24, n).astype('float64'),
        'g_0': onehot[:, 0], 'g_1': onehot[:, 1], 'g_2': onehot[:, 2],
    })
    clean['t'] = rng.normal(80, 30, n) if target_type == 'numeric' else rng.choice([1, 2, 3], n, p=[.6, .3, .1])

    df = clean.copy()
    df['a'] += rng.normal(0, 5, n)
    df.loc[rng.choice(n, 40, replace=False), 'a'] = np.nan
    df.loc[rng.choice(n, 30, replace=False), 'b'] = np.nan
    df.loc[rng.choice(n, 25, replace=False), ['g_0', 'g_1', 'g_2']] = 0          # thiếu feature one-hot
    flip = rng.choice(n, 35, replace=False)
    df.loc[flip, ['g_0', 'g_1', 'g_2']] = df.loc[flip, ['g_1', 'g_2', 'g_0']].to_numpy()
    noisy = rng.choice(n, 50, replace=False)
    df.loc[noisy, 't'] = df.loc[noisy, 't'] + 1 if target_type == 'numeric' else rng.choice([1, 2, 3], 50)
    df = pd.concat([df, df.loc[rng.choice(n, 60)]])                              # nhân bản, giữ index gốc

    got_pandas = pandas_profiler.compute_profile(df, 't', target_type, clean=clean, onehot_prefixes=PREFIXES)
    got_spark = spark_profile(spark, df, clean, 't', target_type, PREFIXES)
    assert got_spark == approx(got_pandas)
    # dữ liệu test phải thực sự làm bẩn mọi dimension, nếu không phép đối chiếu là vô nghĩa
    assert all(0 < v < 1 for v in got_pandas.values() if not np.isnan(v))
