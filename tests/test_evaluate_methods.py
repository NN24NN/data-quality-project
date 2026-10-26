"""Test Giai đoạn 7 (evaluate_methods) và chẩn đoán SHAP: chỉ số có đáp án tính tay, bootstrap lấy lại theo nhóm,
bảng phụ bỏ đúng 24 lượt, đủ dòng và đúng định dạng trên dự đoán out-of-fold của bảng giả lập ở test_metamodel.

Chạy: python -m pytest tests/test_evaluate_methods.py -q
"""
import numpy as np
import pandas as pd
import pytest

from metamodel import evaluate_methods as ev
from metamodel import train_metamodel as tm
from test_metamodel import CFG, RESULTS

EVAL = CFG['meta']['evaluation']
THRESHOLDS = [CFG['meta']['rule_based']['bad_relative_delta'], *CFG['meta']['rule_based']['sensitivity_relative_delta']]
N_TASKS = len(CFG['tasks']) + 1                    # + 'all'


@pytest.fixture(scope='module')
def preds():
    return tm.cross_validate(RESULTS, CFG)


# --- Chỉ số có đáp án tính tay ---

def test_regression_metrics_by_hand():
    y = [0.0, 1.0, 2.0, 3.0]
    assert ev.regression_metrics(y, y) == {'mae': 0.0, 'r2': 1.0, 'spearman': pytest.approx(1.0)}
    m = ev.regression_metrics(y, [1.0, 2.0, 3.0, 4.0])          # lệch đều 1
    assert m['mae'] == 1.0 and m['spearman'] == pytest.approx(1.0) and m['r2'] == pytest.approx(1 - 4 / 5)
    assert np.isnan(ev.regression_metrics(y, [1.0] * 4)['spearman'])   # dự đoán hằng -> không xếp hạng được


def test_classification_metrics_by_hand():
    m = ev.classification_metrics(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 1, 1, 1)
    assert m['accuracy'] == m['f1_bad'] == m['balanced_accuracy'] == 0.5
    good = ev.classification_metrics(np.zeros(4, bool), np.zeros(4, bool))   # không có Bad nào
    assert good['accuracy'] == 1.0 and np.isnan(good['f1_bad']) and np.isnan(good['balanced_accuracy'])


def test_cluster_bootstrap_resamples_groups_not_rows():
    """Nhóm a (1 dòng, 0) và b (9 dòng, 1): lấy lại theo nhóm chỉ cho ra 0, 0,9 hoặc 1."""
    values = np.array([0.0] + [1.0] * 9)
    groups = np.array(['a'] + ['b'] * 9)
    point, boot = ev.cluster_bootstrap(values, groups, np.zeros(10), 500, np.random.default_rng(0))
    assert point == pytest.approx(0.9)
    assert set(np.round(boot, 6)) <= {0.0, 0.9, 1.0}
    _, again = ev.cluster_bootstrap(values, groups, np.zeros(10), 500, np.random.default_rng(0))
    assert np.array_equal(boot, again)


def test_secondary_subset_drops_exactly_class_balance_clustering():
    sub = ev.subsets(RESULTS, CFG)
    assert list(sub) == [ev.ALL, f"excl_{EVAL['secondary_subset_exclude']['task']}_"
                                 f"{EVAL['secondary_subset_exclude']['dimension']}"]
    second = list(sub.values())[1]
    assert len(RESULTS) - len(second) == 24
    assert not ((second['task'] == 'clustering') & (second['dimension'] == 'class_balance')).any()


# --- Bảng kết quả trên dự đoán out-of-fold ---

def test_method_comparison(preds):
    comp = ev.method_comparison(preds, CFG)
    n_targets = len(CFG['meta']['targets'])
    assert len(comp) == 2 * 2 * n_targets * len(ev.REGRESSORS) * N_TASKS     # subset × scheme × nhãn × method × task
    kfold = comp[comp['scheme'] == 'group_kfold']
    assert (kfold['n_repeats'] == len(CFG['seeds']['runs'])).all() and kfold['mae_std'].notna().all()
    assert comp.loc[comp['scheme'] == 'interpolation', 'mae_std'].isna().all()
    # một ô đối chiếu tính tay
    cell = comp.query("subset == 'all' and scheme == 'group_kfold' and target == 'absolute' "
                      "and method == 'metamodel' and task == 'regression'").iloc[0]
    d = preds.query("scheme == 'group_kfold' and target == 'absolute' and method == 'metamodel' "
                    "and task == 'regression'")
    manual = d.assign(e=(d['y_true'] - d['y_pred']).abs()).groupby('repeat_seed')['e'].mean().mean()
    assert cell['mae_mean'] == pytest.approx(manual)
    assert cell['n_rows'] == (RESULTS['task'] == 'regression').sum()


def test_bootstrap_mae_diff(preds):
    boot = ev.bootstrap_mae_diff(preds, CFG)
    n_targets = len(CFG['meta']['targets'])
    assert len(boot) == 2 * 2 * n_targets * (len(ev.REGRESSORS) - 1) * N_TASKS
    assert (boot['ci_low'] <= boot['mae_diff']).all() and (boot['mae_diff'] <= boot['ci_high']).all()
    main = boot.query("subset == 'all' and scheme == 'group_kfold' and task == 'all'")
    assert (main['n_groups'] == tm.make_groups(RESULTS).nunique()).all()
    # bảng giả lập có ảnh hưởng phụ thuộc tác vụ -> meta-model tốt hơn rõ rệt cả 2 baseline
    assert (main['ci_high'] < 0).all()
    pd.testing.assert_frame_equal(boot, ev.bootstrap_mae_diff(preds, CFG))      # cùng seed -> giống hệt


def test_goodbad_comparison(preds):
    gb = ev.goodbad_comparison(preds, CFG)
    assert len(gb) == 2 * 2 * 4 * len(THRESHOLDS) * N_TASKS                       # subset × scheme × 4 method × ngưỡng × task
    counts = gb[['tp_mean', 'fp_mean', 'fn_mean', 'tn_mean']].sum(axis=1)
    assert np.allclose(counts, gb['n_rows'])
    # nhãn thật không phụ thuộc phương pháp
    key = ['subset', 'scheme', 'threshold', 'task']
    assert (gb.groupby(key)['bad_rate_true_mean'].nunique() == 1).all()
    # ngưỡng cao hơn -> ít Bad thật hơn
    rate = gb.query("subset == 'all' and scheme == 'group_kfold' and method == 'metamodel' and task == 'all'") \
             .set_index('threshold')['bad_rate_true_mean']
    assert rate[min(THRESHOLDS)] >= rate[max(THRESHOLDS)]


def test_shap_by_row_dimension():
    pytest.importorskip('shap')
    from metamodel.explain_shap import shap_by_row_dimension
    diag = shap_by_row_dimension(tm.fit_final(RESULTS, CFG), RESULTS)
    reg = diag.query("task == 'regression' and target == 'absolute'").set_index('row_dimension')
    assert 'lambda_0' in reg.index
    assert reg['n_rows'].sum() == (RESULTS['task'] == 'regression').sum()
    # SHAP của Completeness ở regression chủ yếu đến từ chính các lượt làm bẩn Completeness
    assert reg['profile_completeness'].idxmax() == 'completeness'
