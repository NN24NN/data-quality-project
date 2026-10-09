"""Test module downstream: học được dữ liệu dễ, không rò rỉ target, chạy được trên dữ liệu đã làm bẩn, tái lập theo seed.

Chạy: python -m pytest tests/test_downstream.py -q
"""
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from downstream import EVALUATORS, classification, clustering, regression
from downstream.common import split_train_test
from pollution import POLLUTERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(name):
    with open(os.path.join(ROOT, 'configs', name)) as f:
        return yaml.safe_load(f)


SEEDS = _config('seeds.yaml')
TASKS = _config('experiment_matrix.yaml')['tasks']
PARAMS = _config('algorithms.yaml')['algorithms']
MAX_LEVEL = max(_config('pollution_levels.yaml')['levels'])
SEED = SEEDS['runs'][1]
PREFIXES = ['g_']
N = 600
TARGET_TYPE = {'classification': 'categorical', 'regression': 'numeric', 'clustering': 'categorical'}
MODELS = {'classification': classification.MODELS, 'regression': regression.MODELS, 'clustering': clustering.MODELS}
CASES = [(task, algo) for task, cfg in TASKS.items() for algo in cfg['algorithms']]


def _frame(a, b, group, t):
    onehot = np.eye(3, dtype='int8')[group]
    return pd.DataFrame({'a': a, 'b': b, 'g_0': onehot[:, 0], 'g_1': onehot[:, 1], 'g_2': onehot[:, 2], 't': t})


def make_easy(task):
    """Dữ liệu mà target suy ra được rõ ràng từ feature."""
    rng = np.random.default_rng(SEEDS['base_seed'])
    b = rng.normal(0, 1, N)
    if task == 'regression':
        a = rng.normal(0, 1, N)
        return _frame(a, b, rng.integers(0, 3, N), 3 * a - 2 * b + rng.normal(0, 0.1, N))
    k = 2 if task == 'classification' else 3
    t = rng.integers(0, k, N)
    return _frame(10 * t + rng.normal(0, 1, N), b, t, t)          # nhóm one-hot đi theo lớp


def make_unrelated(task):
    """Target độc lập hoàn toàn với feature."""
    rng = np.random.default_rng(SEEDS['base_seed'])
    t = rng.normal(0, 1, N) if task == 'regression' else rng.integers(0, 2 if task == 'classification' else 3, N)
    return _frame(rng.normal(0, 1, N), rng.normal(0, 1, N), rng.integers(0, 3, N), t)


def split(task, df):
    if TASKS[task]['split'] is None:
        return df, None
    return split_train_test(df, 't', TARGET_TYPE[task], TASKS[task]['split']['test_size'], SEEDS['base_seed'])


def score(task, algo, train, test, seed=SEED):
    return EVALUATORS[task](algo, train, test, 't', seed, PARAMS[algo])


# ---------- cấu hình ----------

def test_config_covers_every_algorithm():
    for task, algo in CASES:
        assert algo in MODELS[task], f"{task}: chưa cài {algo}"
        assert algo in PARAMS, f"configs/algorithms.yaml thiếu {algo}"


def test_unknown_algorithm_raises():
    train, test = split('classification', make_easy('classification'))
    with pytest.raises(KeyError):
        EVALUATORS['classification']('khong_co', train, test, 't', SEED, {})


# ---------- chất lượng mô hình ----------

@pytest.mark.parametrize('task,algo', CASES)
def test_learns_easy_data(task, algo):
    train, test = split(task, make_easy(task))
    assert score(task, algo, train, test) > 0.9


@pytest.mark.parametrize('task,algo', CASES)
def test_no_target_leak(task, algo):
    """Target không liên quan feature thì điểm phải ở mức ngẫu nhiên — nếu target lọt vào X, điểm sẽ gần 1."""
    train, test = split(task, make_unrelated(task))
    limit = {'classification': 0.65, 'regression': 0.2, 'clustering': 0.1}[task]   # F1 ngẫu nhiên 2 lớp ~ 0,5
    assert score(task, algo, train, test) < limit


@pytest.mark.parametrize('task,algo', CASES)
def test_reproducible_with_same_seed(task, algo):
    train, test = split(task, make_easy(task))
    assert score(task, algo, train, test) == score(task, algo, train, test)


# ---------- ghép với module pollution ----------

@pytest.mark.parametrize('task,algo,dim', [(t, a, d) for t, a in CASES for d in TASKS[t]['dimensions']])
def test_runs_on_polluted_data(task, algo, dim):
    """Mức bẩn cao nhất của mọi dimension (NaN, nhóm one-hot về 0, index trùng, mất dòng) vẫn ra điểm hữu hạn."""
    def pollute(df):
        return POLLUTERS[dim](df, MAX_LEVEL, SEED, target='t', target_type=TARGET_TYPE[task], onehot_prefixes=PREFIXES)

    train, test = split(task, make_easy(task))
    result = score(task, algo, pollute(train), None if test is None else pollute(test))
    assert np.isfinite(result)


def test_missing_values_hurt_but_do_not_break():
    """Completeness ở mức cao nhất phải làm điểm giảm so với dữ liệu sạch."""
    train, test = split('classification', make_easy('classification'))
    kw = dict(target='t', target_type='categorical', onehot_prefixes=PREFIXES)
    dirty = [POLLUTERS['completeness'](df, MAX_LEVEL, SEED, **kw) for df in (train, test)]
    assert score('classification', 'logistic_regression', *dirty) < score('classification', 'logistic_regression', train, test)


# ---------- chia train/test ----------

def test_split_is_stratified_for_categorical_target():
    df = make_easy('clustering')
    train, test = split_train_test(df, 't', 'categorical', 0.2, SEED)
    assert len(train) + len(test) == N and abs(len(test) - 0.2 * N) <= 1
    assert not set(train.index) & set(test.index)
    ratio = lambda part: part['t'].value_counts(normalize=True).sort_index().to_numpy()
    assert np.abs(ratio(train) - ratio(test)).max() < 0.02


def test_split_is_stratified_for_numeric_target():
    df = make_easy('regression')
    train, test = split_train_test(df, 't', 'numeric', 0.2, SEED)
    assert len(train) + len(test) == N
    edges = df['t'].quantile(np.linspace(0, 1, 11)).to_numpy()            # 10 khoảng phân vị
    share = lambda part: np.histogram(part['t'], edges)[0] / len(part)
    assert np.abs(share(train) - share(test)).max() < 0.02


def test_split_is_reproducible():
    df = make_easy('classification')
    first, second = (split_train_test(df, 't', 'categorical', 0.2, SEED) for _ in range(2))
    assert first[0].index.equals(second[0].index) and first[1].index.equals(second[1].index)
