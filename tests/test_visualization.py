"""Smoke test cho src/visualization/plots.py: dữ liệu giả đúng schema, kiểm tra hàm chạy được và ghi được .png."""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from visualization import plots  # noqa: E402

LEVELS = [0.0, 0.2, 0.5, 0.8]
TASK_SPEC = {'regression': (['ridge', 'gradient_boosting'], ['completeness', 'feature_accuracy', 'target_accuracy', 'uniqueness']),
             'classification': (['logistic_regression', 'random_forest'], plots.DIMENSIONS),
             'clustering': (['kmeans', 'gmm'], ['completeness', 'feature_accuracy', 'uniqueness', 'class_balance'])}


@pytest.fixture(scope='module')
def results():
    rng = np.random.default_rng(0)
    rows = [{'task': t, 'algorithm': a, 'dimension': d, 'pollution_level': lv, 'run_id': r,
             'baseline_performance': 0.8, 'delta_performance': 0.1 * lv + rng.normal(0, 0.01)}
            for t, (algos, dims) in TASK_SPEC.items() for a in algos for d in dims for lv in LEVELS for r in (1, 2, 3)]
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def preds(results):
    rng = np.random.default_rng(1)
    parts = []
    for method in plots.METHODS:
        y = results['delta_performance'].to_numpy()
        pred = (y > 0.04).astype(float) if method == 'rule_based' else y + rng.normal(0, 0.02, len(y))
        for seed in (42, 43):
            parts.append(results[['task', 'algorithm', 'dimension', 'pollution_level', 'run_id']].assign(
                scheme='group_kfold', repeat_seed=seed, fold=0, target='absolute', method=method, y_true=y, y_pred=pred))
    return pd.concat(parts, ignore_index=True)


def test_h1_h2_h3_h4(results, tmp_path):
    rows = []
    for task, (_, dims) in TASK_SPEC.items():
        for d in dims:
            for lv in LEVELS:
                rows.append({'dataset': task, 'dimension': d, 'level': lv, **{k: 1 - lv for k in plots.DIMENSIONS}})
    fig = plots.plot_h1(pd.DataFrame(rows), {t: t for t in TASK_SPEC})
    base = results.drop_duplicates(['task', 'algorithm', 'run_id']).assign(baseline_performance=lambda d: d['baseline_performance'])
    figs = [fig, plots.plot_h2(base), plots.plot_h3(results), plots.plot_h3(plots.drop_secondary(results), ' (288 lượt)'),
            plots.plot_h4(results)]
    for i, f in enumerate(figs):
        assert os.path.getsize(plots.save_fig(f, f'fig{i}', str(tmp_path), dpi=50)) > 0
        plt.close(f)


def test_drop_secondary(results):
    out = plots.drop_secondary(results)
    assert len(out) == len(results) - 2 * 4 * 3
    assert out[(out['task'] == 'clustering')]['dimension'].ne('class_balance').all()


def test_h5_h6(preds):
    rows = [{'subset': 'all', 'scheme': 'group_kfold', 'target': 'absolute', 'method': m, 'task': t,
             **{f'{k}_{s}': 0.1 for k in ('mae', 'r2', 'spearman') for s in ('mean', 'std')}}
            for m in plots.METHODS[:1] + plots.METHODS[2:] for t in [*TASK_SPEC, 'all']]
    comp = pd.DataFrame(rows)
    f5 = plots.plot_h5(comp)
    assert not plots.h5_table(comp).empty
    f6 = plots.plot_h6(preds)
    assert len(f6.axes) == 4
    plt.close('all')


def test_h7_h8_h9_h10():
    gb = pd.DataFrame([{'subset': 'all', 'scheme': 'group_kfold', 'task': 'all', 'threshold': 0.1, 'method': m,
                        'tp_mean': 50, 'fp_mean': 10, 'fn_mean': 20, 'tn_mean': 100, 'recall_mean': 0.7, 'precision_mean': 0.8}
                       for m in ('rule_based', 'metamodel')])
    plots.plot_h7(gb, 0.10)
    shap = pd.DataFrame([{'target': 'absolute', 'task': t, 'feature': f'profile_{d}', 'kind': 'profile',
                          'applicable': True, 'share': 1 / len(dims)} for t, (_, dims) in TASK_SPEC.items() for d in dims])
    plots.plot_h8(shap, note='ghi chú')
    pca = pd.DataFrame({'dimension': [f'profile_{d}' for d in plots.DIMENSIONS], 'pc1_weight': [0.5, 0.1, -0.2, 0.4, 0.3],
                        'explained_variance_ratio': 0.27})
    plots.plot_h9(pca, shap)
    bench = pd.DataFrame([{'engine': e, 'n_rows': n, 'status': 'oom' if (e == 'pandas' and n == 22) else 'ok',
                           'seconds': n * (1 if e == 'pandas' else 3), 'peak_rss_mb': 100 * n}
                          for e in ('pandas', 'spark') for n in (1, 10, 22) for _ in range(3)])
    plots.plot_h10(bench)
    plt.close('all')
