"""Test meta-model, 3 baseline và SHAP trên bảng kết quả giả lập cùng cấu trúc 312 dòng thật.

Bảng giả lập: điểm của dimension bị làm bẩn = 1 − λ, Δ = λ × hệ số riêng theo (task, dimension), thuật toán thứ hai
chịu ảnh hưởng gấp 1,5 lần; λ = 0 giống hệt nhau giữa các dimension như Experiment Runner thật.

Chạy: python -m pytest tests/test_metamodel.py -q
"""
import numpy as np
import pandas as pd
import pytest

from baselines import DQSOpsPCA, RuleBased, WeightedScore
from baselines.weighted_score import quality_score
from metamodel import train_metamodel as tm
from profiling.common import PROFILE_KEYS

CFG = tm.load_configs()
TASKS = CFG['tasks']
LEVELS = [0.0, 0.2, 0.5, 0.8]
RUN_IDS = list(CFG['seeds']['runs'])
FILL = CFG['meta']['not_applicable_fill']
# Ảnh hưởng của từng dimension khác nhau theo tác vụ; Uniqueness không ảnh hưởng (như kết quả thật)
EFFECT = {'classification': {'completeness': 0.2, 'feature_accuracy': 0.3, 'target_accuracy': 0.4, 'class_balance': 0.1},
          'regression': {'completeness': 0.6, 'feature_accuracy': 0.3, 'target_accuracy': 0.5},
          'clustering': {'completeness': 0.15, 'feature_accuracy': 0.2, 'class_balance': 0.05}}
BASELINE = {'classification': 0.7, 'regression': 0.9, 'clustering': 0.2}


def make_results(noise=0.002):
    rng = np.random.default_rng(CFG['seeds']['base_seed'])
    rows = []
    for task, tcfg in TASKS.items():
        for i, algo in enumerate(tcfg['algorithms']):
            for dim in tcfg['dimensions']:
                for lv in LEVELS:
                    for run_id in RUN_IDS:
                        prof = {k: 1.0 for k in PROFILE_KEYS}
                        if 'class_balance' not in tcfg['dimensions']:
                            prof['profile_class_balance'] = np.nan
                        eps = rng.normal(0, noise, 2) if lv else np.zeros(2)
                        prof[f'profile_{dim}'] = 1 - lv + eps[0]
                        delta = lv * EFFECT[task].get(dim, 0.0) * (1 + 0.5 * i) + eps[1]
                        rows.append({'dataset': tcfg['dataset'], 'task': task, 'algorithm': algo, 'dimension': dim,
                                     'pollution_level': lv, 'run_id': run_id, **prof,
                                     'raw_performance': BASELINE[task] - delta,
                                     'baseline_performance': BASELINE[task], 'delta_performance': delta})
    return pd.DataFrame(rows)


RESULTS = make_results()
N_GROUPS = sum(len(t['dimensions']) * (len(LEVELS) - 1) + 1 for t in TASKS.values())


@pytest.fixture(scope='module')
def preds():
    return tm.cross_validate(RESULTS, CFG)


# --- Đặc trưng, nhãn, cách chia ---

def test_toy_matches_experiment_size():
    assert len(RESULTS) == 312 and N_GROUPS == 42


def test_features_no_nan_and_onehot():
    X = tm.build_features(RESULTS, TASKS, FILL)
    n_algos = sum(len(t['algorithms']) for t in TASKS.values())
    assert X.shape == (len(RESULTS), len(PROFILE_KEYS) + len(TASKS) + n_algos)
    assert X.notna().all().all()
    assert (X.filter(like='task_').sum(axis=1) == 1).all()
    assert (X.filter(like='algo_').sum(axis=1) == 1).all()
    assert (X.loc[RESULTS['task'] == 'regression', 'profile_class_balance'] == FILL).all()


def test_relative_target():
    rel = tm.make_target(RESULTS, 'relative')
    assert np.allclose(rel, RESULTS['delta_performance'] / RESULTS['baseline_performance'])
    with pytest.raises(ValueError):
        tm.make_target(RESULTS, 'khong_co')


def test_groups_merge_level_zero_per_task():
    groups = tm.make_groups(RESULTS)
    assert groups.nunique() == N_GROUPS
    zero = RESULTS['pollution_level'] == 0
    assert (groups[zero].groupby(RESULTS.loc[zero, 'task']).nunique() == 1).all()


@pytest.mark.parametrize('seed', list(CFG['seeds']['runs'].values()))
def test_group_kfold_no_leak_and_full_coverage(seed):
    groups = tm.make_groups(RESULTS).to_numpy()
    splits = tm.group_kfold_splits(RESULTS, CFG['meta']['cv']['n_splits'], seed)
    assert len(splits) == CFG['meta']['cv']['n_splits']
    seen = np.concatenate([te for _, te in splits])
    assert sorted(seen) == list(range(len(RESULTS)))               # mỗi dòng được dự đoán đúng 1 lần
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])               # không nhóm nào bị tách rời
        assert set(RESULTS['task'].iloc[tr]) == set(TASKS)         # train luôn đủ 3 tác vụ


def test_interpolation_split():
    level = CFG['meta']['interpolation_test_level']
    [(tr, te)] = tm.interpolation_split(RESULTS, level)
    assert np.allclose(RESULTS['pollution_level'].iloc[te], level)
    assert not np.isclose(RESULTS['pollution_level'].iloc[tr], level).any()
    assert len(tr) + len(te) == len(RESULTS)


# --- 3 baseline ---

def test_rule_based_ignores_not_applicable():
    df = pd.DataFrame([{k: 1.0 for k in PROFILE_KEYS}] * 3)
    df.loc[1, 'profile_completeness'] = 0.85
    df.loc[2, 'profile_class_balance'] = np.nan
    assert list(RuleBased(0.9).fit(df).predict(df)) == [0.0, 1.0, 0.0]


def test_weighted_score_averages_applicable_only():
    df = RESULTS[RESULTS['task'] == 'regression']
    expected = df[[k for k in PROFILE_KEYS if k != 'profile_class_balance']].mean(axis=1)
    assert np.allclose(quality_score(df).ravel(), expected)
    pred = WeightedScore().fit(RESULTS, RESULTS['delta_performance']).predict(RESULTS)
    assert pred.shape == (len(RESULTS),) and np.isfinite(pred).all()


def test_dqsops_pca_fits_on_train_only():
    tr = RESULTS[RESULTS['pollution_level'] != 0.8]
    model = DQSOpsPCA(FILL).fit(tr, tr['delta_performance'])
    assert np.allclose(model.scaler_.mean_, tr[PROFILE_KEYS].fillna(FILL).mean())
    weights, ratio = model.weights()
    assert weights.sum() > 0 and 0 < ratio <= 1
    assert np.isfinite(model.predict(RESULTS)).all()


# --- Đánh giá chéo ---

def test_cross_validate_shape_and_coverage(preds):
    assert list(preds.columns) == tm.PRED_COLUMNS
    n_methods, n_targets, n_repeats = 4, len(CFG['meta']['targets']), len(CFG['seeds']['runs'])
    kfold = preds[preds['scheme'] == 'group_kfold']
    assert len(kfold) == len(RESULTS) * n_repeats * n_targets * n_methods
    assert not kfold.duplicated(['repeat_seed', 'target', 'method', *tm.KEY]).any()
    n_interp = int(np.isclose(RESULTS['pollution_level'], CFG['meta']['interpolation_test_level']).sum())
    assert len(preds) - len(kfold) == n_interp * n_targets * n_methods
    assert set(preds.loc[preds['method'] == 'rule_based', 'y_pred']) <= {0.0, 1.0}


def test_metamodel_beats_task_agnostic_baselines(preds):
    """Δ phụ thuộc tác vụ -> meta-model (biết task) phải sai ít hơn hai baseline một-công-thức-cho-mọi-tác-vụ."""
    d = preds[(preds['scheme'] == 'group_kfold') & (preds['method'] != 'rule_based')]
    mae = d.assign(err=(d['y_true'] - d['y_pred']).abs()).groupby(['target', 'method'])['err'].mean().unstack()
    assert (mae['metamodel'] < mae['weighted_score']).all()
    assert (mae['metamodel'] < mae['dqsops_pca']).all()


def test_cross_validate_reproducible(preds):
    pd.testing.assert_frame_equal(tm.cross_validate(RESULTS, CFG), preds)


# --- Mô hình cuối và SHAP ---

def test_fit_final_and_pca_weights():
    bundle = tm.fit_final(RESULTS, CFG)
    assert set(bundle['models']) == set(CFG['meta']['targets'])
    assert bundle['feature_names'] == list(tm.build_features(RESULTS, TASKS, FILL).columns)
    w = tm.pca_weights(RESULTS, CFG)
    assert list(w['dimension']) == PROFILE_KEYS and w['explained_variance_ratio'].nunique() == 1


def test_shap_importance_task_aware():
    pytest.importorskip('shap')
    from metamodel.explain_shap import COLUMNS, shap_importance
    imp = shap_importance(tm.fit_final(RESULTS, CFG), RESULTS)
    assert list(imp.columns) == COLUMNS
    prof = imp[imp['kind'] == 'profile']
    # tỷ trọng các dimension áp dụng được cộng lại bằng 1 ở mỗi (nhãn, tác vụ)
    assert np.allclose(prof.dropna(subset=['share']).groupby(['target', 'task'])['share'].sum(), 1)
    na = prof.set_index(['task', 'feature'])['applicable']
    assert not na[('regression', 'profile_class_balance')].any()
    assert not na[('clustering', 'profile_target_accuracy')].any()
    share = prof[prof['target'] == 'absolute'].pivot(index='feature', columns='task', values='share')
    # regression: Completeness (hệ số 0,6) quan trọng hơn Uniqueness (hệ số 0)
    assert share.loc['profile_completeness', 'regression'] > share.loc['profile_uniqueness', 'regression']
