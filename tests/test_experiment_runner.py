"""Test Experiment Runner trên dữ liệu đồ chơi: đủ số dòng, đúng schema, λ=0 khớp baseline, chạy tiếp được từ chỗ dừng.

Chạy: python -m pytest tests/test_experiment_runner.py -q
"""
import numpy as np
import pandas as pd
import pytest

from downstream import EVALUATORS
from downstream.common import split_train_test
from experiment_runner import run_experiment as runner

CFG = runner.load_configs()
RUN_IDS = list(CFG['seeds']['runs'])[:2]
LEVELS = [0.0, max(CFG['levels'])]
N = 400

# Ma trận thu nhỏ: thuật toán nhanh nhất mỗi tác vụ, 1 tác vụ có split và 1 tác vụ không (clustering)
TOY_TASKS = {
    'classification': {**CFG['tasks']['classification'], 'dataset': 'toy',
                       'dimensions': ['completeness', 'target_accuracy', 'uniqueness'],
                       'algorithms': ['logistic_regression']},
    'clustering': {**CFG['tasks']['clustering'], 'dataset': 'toy',
                   'dimensions': ['feature_accuracy', 'class_balance'], 'algorithms': ['kmeans']},
}
TOY_CFG = {**CFG, 'tasks': TOY_TASKS, 'levels': LEVELS, 'seeds': {**CFG['seeds'], 'runs': {r: CFG['seeds']['runs'][r] for r in RUN_IDS}},
           'datasets': {'toy': {'file': None, 'target': 't', 'target_type': 'categorical', 'onehot_prefixes': ['g_']}}}
N_EXPECTED = sum(len(t['dimensions']) * len(LEVELS) * len(t['algorithms']) * len(RUN_IDS) for t in TOY_TASKS.values())


def make_toy():
    rng = np.random.default_rng(CFG['seeds']['base_seed'])
    t = rng.integers(0, 3, N)
    onehot = np.eye(3, dtype='int8')[t]
    return pd.DataFrame({'a': 5 * t + rng.normal(0, 1, N), 'b': rng.normal(0, 1, N),
                         'g_0': onehot[:, 0], 'g_1': onehot[:, 1], 'g_2': onehot[:, 2], 't': t})


def make_baseline():
    """Baseline sạch theo từng seed — cùng cách cell 11.3 tính baseline_performance.csv."""
    df, base = make_toy(), {}
    for task, tcfg in TOY_TASKS.items():
        if tcfg['split']:
            train, test = split_train_test(df, 't', 'categorical', tcfg['split']['test_size'], CFG['seeds']['base_seed'])
        else:
            train, test = df, None
        for algo in tcfg['algorithms']:
            for r in RUN_IDS:
                base[('toy', algo, r)] = EVALUATORS[task](algo, train, test, 't', CFG['seeds']['runs'][r], CFG['algorithms'][algo])
    return base


BASELINE = make_baseline()          # tính trước khi test nào đếm số lần huấn luyện


def run(out, **kw):
    return runner.run_experiments(TOY_CFG, lambda name: make_toy(), BASELINE, str(out), **kw)


@pytest.fixture
def count_calls(monkeypatch):
    """Đếm số lần huấn luyện mô hình thật."""
    calls = []
    for task, fn in list(runner.EVALUATORS.items()):
        def wrapped(*a, _fn=fn, **k):
            calls.append(a[0])
            return _fn(*a, **k)
        monkeypatch.setitem(runner.EVALUATORS, task, wrapped)
    return calls


def test_full_matrix_has_expected_rows_and_schema(tmp_path):
    res = run(tmp_path / 'experiment_results.parquet')
    assert list(res.columns) == runner.COLUMNS
    assert len(res) == N_EXPECTED
    assert not res.duplicated(runner.KEY).any()
    assert res[['raw_performance', 'baseline_performance', 'delta_performance']].notna().all().all()
    assert np.allclose(res['delta_performance'], res['baseline_performance'] - res['raw_performance'])
    on_disk = pd.read_parquet(tmp_path / 'experiment_results.parquet')
    assert list(on_disk.columns) == runner.COLUMNS and len(on_disk) == N_EXPECTED


def test_level_zero_matches_per_seed_baseline(tmp_path):
    """λ=0 là dữ liệu sạch, cùng seed với baseline -> Δ đúng bằng 0."""
    res = run(tmp_path / 'out.parquet')
    assert (res.loc[res['pollution_level'] == 0, 'delta_performance'].abs() < 1e-12).all()


def test_pollution_moves_its_own_profile_score(tmp_path):
    res = run(tmp_path / 'out.parquet').set_index(['dimension', 'pollution_level']).sort_index()
    top = max(LEVELS)
    for dim in ['completeness', 'target_accuracy', 'uniqueness', 'feature_accuracy']:
        col = f'profile_{dim}'
        assert res.loc[(dim, 0.0), col].min() == pytest.approx(1.0)
        assert res.loc[(dim, top), col].max() < 0.9


def test_level_zero_trained_once_per_algorithm_and_run(tmp_path, count_calls):
    run(tmp_path / 'out.parquet')
    n_level_zero = sum(len(t['algorithms']) * len(RUN_IDS) for t in TOY_TASKS.values())
    n_other = N_EXPECTED // len(LEVELS) * (len(LEVELS) - 1)
    assert len(count_calls) == n_level_zero + n_other


def test_resume_skips_finished_runs(tmp_path, count_calls):
    out = tmp_path / 'out.parquet'
    first = run(out, tasks=['classification'], levels=[0.0])
    assert len(first) == len(TOY_TASKS['classification']['dimensions']) * len(RUN_IDS)
    n_first = len(count_calls)

    full = run(out)                                     # chạy tiếp toàn bộ
    assert len(full) == N_EXPECTED
    assert len(count_calls) - n_first == N_EXPECTED // len(LEVELS) * (len(LEVELS) - 1) + len(RUN_IDS)  # + λ=0 clustering

    n_before = len(count_calls)
    again = run(out)                                    # đã xong hết -> không huấn luyện lại
    assert len(count_calls) == n_before
    pd.testing.assert_frame_equal(again, full)


def test_unknown_selection_raises(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / 'out.parquet', dimensions=['khong_co'])


def test_rerun_is_reproducible(tmp_path):
    a = run(tmp_path / 'a.parquet')
    b = run(tmp_path / 'b.parquet')
    pd.testing.assert_frame_equal(a, b)
