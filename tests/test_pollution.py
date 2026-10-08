"""Test module pollution: λ=0 giữ nguyên dữ liệu, tái lập theo seed, và mức bẩn đo được khớp λ.

Chạy: python -m pytest tests/test_pollution.py -q
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from pollution import POLLUTERS
from profiling import pandas_profiler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(name):
    with open(os.path.join(ROOT, 'configs', name)) as f:
        return yaml.safe_load(f)


SEEDS = _config('seeds.yaml')
LEVELS = _config('pollution_levels.yaml')['levels']
POSITIVE = [lv for lv in LEVELS if lv > 0]
SEED = SEEDS['runs'][1]
PREFIXES = ['g_']
N = 4000
E_ABS = np.sqrt(2 / np.pi)        # E|X| với X ~ N(0, 1): nhiễu trung bình = sqrt(λ) * E_ABS * mean(|gt|)


def make_clean(target_type):
    rng = np.random.default_rng(SEEDS['base_seed'])
    onehot = np.eye(3, dtype='int8')[rng.integers(0, 3, N)]
    df = pd.DataFrame({
        'a': rng.normal(50, 10, N),
        'b': rng.normal(0, 1, N).astype('float32'),       # trung bình ~ 0
        'c': rng.integers(0, 24, N),                      # cột số nguyên
        'g_0': onehot[:, 0], 'g_1': onehot[:, 1], 'g_2': onehot[:, 2],
    })
    df['t'] = rng.normal(80, 30, N) if target_type == 'numeric' else rng.choice([1, 2, 3], N, p=[.5, .3, .2])
    return df


CLEAN = {tt: make_clean(tt) for tt in ('categorical', 'numeric')}


def pollute(dim, level, target_type='categorical', seed=SEED, df=None):
    df = CLEAN[target_type] if df is None else df
    return POLLUTERS[dim](df, level, seed, target='t', target_type=target_type, onehot_prefixes=PREFIXES)


def profile(df, target_type='categorical', clean=None):
    clean = CLEAN[target_type] if clean is None else clean
    return pandas_profiler.compute_profile(df, 't', target_type, clean=clean, onehot_prefixes=PREFIXES)


def exact_rate(level, n=N):
    return 1 - round(level * n) / n


# ---------- tính chất chung của mọi polluter ----------

@pytest.mark.parametrize('dim', list(POLLUTERS))
def test_level_zero_keeps_data_and_profile(dim):
    clean = CLEAN['categorical']
    before = clean.copy()
    out = pollute(dim, 0.0)
    assert_frame_equal(out, clean)
    assert profile(out) == pytest.approx(profile(clean))
    assert_frame_equal(clean, before)                    # không sửa df đầu vào


@pytest.mark.parametrize('dim', list(POLLUTERS))
def test_reproducible_with_same_seed(dim):
    level = POSITIVE[0]
    first = pollute(dim, level)
    assert_frame_equal(first, pollute(dim, level))
    assert not first.equals(pollute(dim, level, seed=SEEDS['runs'][2]))
    assert set(first.index) <= set(CLEAN['categorical'].index)      # giữ index của dòng gốc
    assert_frame_equal(CLEAN['categorical'], make_clean('categorical'))   # không sửa df đầu vào


# ---------- từng polluter: mức bẩn đo bằng profiler khớp λ ----------

@pytest.mark.parametrize('level', POSITIVE)
def test_completeness(level):
    p = profile(pollute('completeness', level))
    assert p['profile_completeness'] == pytest.approx(exact_rate(level))
    assert p['profile_feature_accuracy'] == 1.0          # ô còn lại không bị đổi
    assert p['profile_target_accuracy'] == 1.0


@pytest.mark.parametrize('level', POSITIVE)
def test_feature_accuracy(level):
    p = profile(pollute('feature_accuracy', level))
    numeric = 1 - np.sqrt(level) * E_ABS                 # kỳ vọng của nFAcc, phương sai nhiễu = λ
    assert p['profile_feature_accuracy'] == pytest.approx((numeric + exact_rate(level)) / 2, abs=0.02)
    assert p['profile_completeness'] == 1.0
    assert p['profile_target_accuracy'] == 1.0


@pytest.mark.parametrize('level', POSITIVE)
def test_target_accuracy_categorical(level):
    out = pollute('target_accuracy', level)
    p = profile(out)
    assert p['profile_target_accuracy'] == pytest.approx(exact_rate(level))
    assert p['profile_feature_accuracy'] == 1.0
    assert set(out['t']) <= {1, 2, 3}


@pytest.mark.parametrize('level', POSITIVE)
def test_target_accuracy_numeric(level):
    p = profile(pollute('target_accuracy', level, 'numeric'), 'numeric')
    assert p['profile_target_accuracy'] == pytest.approx(1 - np.sqrt(level) * E_ABS, abs=0.03)
    assert p['profile_feature_accuracy'] == 1.0


@pytest.mark.parametrize('level', POSITIVE)
def test_uniqueness(level):
    clean = CLEAN['categorical']
    out = pollute('uniqueness', level)
    assert abs(len(out) - N / (1 - level)) <= 2          # làm tròn theo từng lớp (3 lớp)
    p = profile(out)
    assert p['profile_uniqueness'] == pytest.approx(1 - level, abs=0.005)
    assert p['profile_feature_accuracy'] == 1.0          # bản sao ghép đúng với dòng gốc
    assert p['profile_target_accuracy'] == 1.0
    assert out['t'].value_counts(normalize=True).to_dict() == pytest.approx(
        clean['t'].value_counts(normalize=True).to_dict(), abs=1e-3)


def test_uniqueness_numeric_target():
    level = POSITIVE[-1]
    out = pollute('uniqueness', level, 'numeric')
    assert len(out) == N + round(N * (1 / (1 - level) - 1))
    assert profile(out, 'numeric')['profile_uniqueness'] == pytest.approx(1 - level, abs=0.005)


@pytest.mark.parametrize('level', POSITIVE)
def test_class_balance(level):
    before = CLEAN['categorical']['t'].value_counts()
    out = pollute('class_balance', level)
    after = out['t'].value_counts()
    largest = before.idxmax()
    for cls, n_cls in before.items():
        expected = n_cls if cls == largest else n_cls - round(level * n_cls)
        assert after[cls] == expected
    p = profile(out)
    assert p['profile_completeness'] == 1.0 and p['profile_feature_accuracy'] == 1.0


def test_class_balance_score_decreases_for_two_classes():
    clean = CLEAN['categorical']
    binary = clean[clean['t'] != 3]
    scores = [profile(pollute('class_balance', lv, df=binary), clean=binary)['profile_class_balance']
              for lv in LEVELS]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores)
    n_small, n_large = sorted(binary['t'].value_counts())
    assert scores[-1] == pytest.approx((n_small - round(LEVELS[-1] * n_small)) / n_large)


def test_class_balance_rejects_numeric_target():
    with pytest.raises(ValueError):
        pollute('class_balance', POSITIVE[0], 'numeric')
