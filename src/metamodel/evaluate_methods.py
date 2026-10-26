"""Đánh giá và so sánh meta-model với 3 baseline (Giai đoạn 7), từ dự đoán out-of-fold của Giai đoạn 6 — không huấn luyện lại.

Theo docs/khac-biet-so-voi-bai-goc.md mục 4.6. Tham số: configs/metamodel.yaml (mục rule_based, evaluation).

- Chỉ số tính riêng từng lần lặp (repeat_seed) rồi lấy trung bình ± độ lệch chuẩn; theo từng tác vụ và 'all'.
- Bootstrap theo nhóm (task, dimension, λ), giữ số nhóm của mỗi tác vụ: khoảng tin cậy của hiệu MAE
  (meta-model − baseline; âm = meta-model sai ít hơn). Lỗi mỗi dòng lấy trung bình qua các lần lặp trước.
- Good/Bad: nhãn thật Bad nếu Δ tương đối > ngưỡng; 3 phương pháp hồi quy rời rạc hóa dự đoán của bản nhãn
  Δ tương đối với cùng ngưỡng; rule-based dùng nhãn 0/1 của nó.
- Cột `subset`: 'all' (bảng chính) và 'excl_<task>_<dimension>' (bảng phụ, chỉ lọc phần đánh giá).

Chạy trên Colab (sau train_metamodel.py):
    python src/metamodel/evaluate_methods.py --project-dir /content/drive/MyDrive/data-quality-project
Ghi ra <project-dir>/results/: method_comparison.csv, method_bootstrap.csv, goodbad_comparison.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from metamodel.train_metamodel import KEY, load_configs, make_groups  # noqa: E402

REGRESSORS = ['metamodel', 'weighted_score', 'dqsops_pca']   # meta-model đứng đầu: là bên được so với các baseline
ALL = 'all'


def subsets(preds, cfg):
    """{'all': toàn bộ, 'excl_<task>_<dimension>': bỏ các lượt trong evaluation.secondary_subset_exclude}."""
    ex = cfg['meta']['evaluation']['secondary_subset_exclude']
    drop = (preds['task'] == ex['task']) & (preds['dimension'] == ex['dimension'])
    return {ALL: preds, f"excl_{ex['task']}_{ex['dimension']}": preds[~drop]}


def _by_task(df):
    for task, part in df.groupby('task', sort=False):
        yield task, part
    yield ALL, df


def _mean_std(per_repeat, row):
    for m in per_repeat:
        row[f'{m}_mean'] = per_repeat[m].mean()
        row[f'{m}_std'] = per_repeat[m].std()          # 1 lần lặp (nội suy) -> NaN
    return row


def regression_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rho = spearmanr(y, p)[0] if np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan
    return {'mae': float(np.mean(np.abs(y - p))), 'r2': float(r2_score(y, p)), 'spearman': float(rho)}


def method_comparison(preds, cfg):
    rows = []
    reg = preds[preds['method'].isin(REGRESSORS)]
    for subset, part in subsets(reg, cfg).items():
        for (scheme, target, method), d in part.groupby(['scheme', 'target', 'method'], sort=False):
            for task, t in _by_task(d):
                per = pd.DataFrame([regression_metrics(r['y_true'], r['y_pred']) for _, r in t.groupby('repeat_seed')])
                rows.append(_mean_std(per, {'subset': subset, 'scheme': scheme, 'target': target, 'method': method,
                                            'task': task, 'n_repeats': len(per), 'n_rows': len(t) // len(per)}))
    return pd.DataFrame(rows)


def cluster_bootstrap(values, groups, strata, n_boot, rng):
    """Trung bình của `values` và n_boot giá trị bootstrap: lấy lại có hoàn lại các nhóm trong từng stratum."""
    agg = pd.DataFrame({'v': values, 'g': groups, 's': strata}).groupby(['s', 'g'])['v'].agg(['sum', 'count'])
    total, count = np.zeros(n_boot), np.zeros(n_boot)
    for _, a in agg.groupby(level='s'):
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        total += a['sum'].to_numpy()[idx].sum(axis=1)
        count += a['count'].to_numpy()[idx].sum(axis=1)
    return float(np.mean(values)), total / count


def bootstrap_mae_diff(preds, cfg):
    ev = cfg['meta']['evaluation']
    rng = np.random.default_rng(cfg['seeds']['base_seed'])
    q = [(1 - ev['ci']) / 2, (1 + ev['ci']) / 2]
    rows = []
    reg = preds[preds['method'].isin(REGRESSORS)]
    for subset, part in subsets(reg, cfg).items():
        for (scheme, target), d in part.groupby(['scheme', 'target'], sort=False):
            err = (d.assign(err=(d['y_true'] - d['y_pred']).abs())
                    .groupby(['method', *KEY])['err'].mean()          # trung bình qua các lần lặp
                    .unstack('method'))
            keys = err.index.to_frame(index=False)
            groups = make_groups(keys).to_numpy()
            for base in REGRESSORS[1:]:
                diff = (err['metamodel'] - err[base]).to_numpy()
                for task in [*keys['task'].unique(), ALL]:
                    m = np.ones(len(keys), bool) if task == ALL else (keys['task'] == task).to_numpy()
                    point, boot = cluster_bootstrap(diff[m], groups[m], keys['task'].to_numpy()[m],
                                                    ev['n_bootstrap'], rng)
                    lo, hi = np.quantile(boot, q)
                    rows.append({'subset': subset, 'scheme': scheme, 'target': target, 'baseline': base,
                                 'task': task, 'n_groups': len(set(groups[m])), 'mae_diff': point,
                                 'ci_low': lo, 'ci_high': hi, 'p_not_better': float((boot >= 0).mean())})
    return pd.DataFrame(rows)


def classification_metrics(truth, pred):
    """Lớp dương = Bad. Tỷ lệ không xác định (mẫu số 0) -> NaN."""
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    n = tp + fp + fn + tn

    def ratio(a, b):
        return a / b if b else np.nan
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {'accuracy': (tp + tn) / n, 'precision': ratio(tp, tp + fp), 'recall': recall,
            'specificity': specificity, 'balanced_accuracy': (recall + specificity) / 2,
            'f1_bad': ratio(2 * tp, 2 * tp + fp + fn), 'bad_rate_true': (tp + fn) / n,
            'bad_rate_pred': (tp + fp) / n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def goodbad_comparison(preds, cfg):
    rb = cfg['meta']['rule_based']
    thresholds = [rb['bad_relative_delta'], *rb['sensitivity_relative_delta']]
    rows = []
    rel = preds[preds['target'] == 'relative']
    for subset, part in subsets(rel, cfg).items():
        for (scheme, method), d in part.groupby(['scheme', 'method'], sort=False):
            for thr in thresholds:
                for task, t in _by_task(d):
                    per = []
                    for _, r in t.groupby('repeat_seed'):
                        truth = r['y_true'].to_numpy() > thr
                        score = r['y_pred'].to_numpy()
                        pred = score >= 0.5 if method == 'rule_based' else score > thr   # rule-based: 1 = Bad
                        per.append(classification_metrics(truth, pred))
                    rows.append(_mean_std(pd.DataFrame(per),
                                          {'subset': subset, 'scheme': scheme, 'method': method, 'threshold': thr,
                                           'task': task, 'n_repeats': len(per), 'n_rows': len(t) // len(per)}))
    return pd.DataFrame(rows)


def _table(df, index, value):
    return df.pivot(index=index, columns='task', values=value)[[*dict.fromkeys(df['task'])]]


def _pm(df, metric):
    return df[f'{metric}_mean'].map('{:.3f}'.format) + ' ± ' + df[f'{metric}_std'].map('{:.3f}'.format)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', required=True, help='thư mục chứa results/ (Drive)')
    args = p.parse_args(argv)

    cfg = load_configs()
    out_dir = os.path.join(args.project_dir, 'results')
    preds = pd.read_parquet(os.path.join(out_dir, 'metamodel_cv_predictions.parquet'))
    comp, boot, gb = method_comparison(preds, cfg), bootstrap_mae_diff(preds, cfg), goodbad_comparison(preds, cfg)
    for name, df in [('method_comparison', comp), ('method_bootstrap', boot), ('goodbad_comparison', gb)]:
        df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
    print(f"Đọc {len(preds)} dòng dự đoán; đã lưu method_comparison.csv ({len(comp)} dòng), "
          f"method_bootstrap.csv ({len(boot)}), goodbad_comparison.csv ({len(gb)})")
    pd.set_option('display.width', 250, 'display.max_columns', 20)

    second = [s for s in subsets(preds, cfg) if s != ALL][0]
    for scheme in ['group_kfold', 'interpolation']:
        for target in cfg['meta']['targets']:
            d = comp.query('subset == @ALL and scheme == @scheme and target == @target')
            print(f"\n===== {scheme} | nhãn {target} | 312 lượt (trung bình ± độ lệch chuẩn qua các lần lặp) =====")
            for metric in ['mae', 'r2', 'spearman']:
                print(f"-- {metric}")
                print(_table(d.assign(v=_pm(d, metric)), 'method', 'v').to_string())

    print(f"\n===== Bảng phụ {second}: MAE, group_kfold =====")
    d = comp.query('subset == @second and scheme == "group_kfold"')
    print(_table(d.assign(v=_pm(d, 'mae')), ['target', 'method'], 'v').to_string())

    print(f"\n===== Bootstrap theo nhóm: hiệu MAE meta-model − baseline (âm = meta-model tốt hơn), "
          f"CI {cfg['meta']['evaluation']['ci']:.0%} =====")
    b = boot.assign(v=boot['mae_diff'].map('{:+.4f}'.format) + ' [' + boot['ci_low'].map('{:+.4f}'.format)
                    + ', ' + boot['ci_high'].map('{:+.4f}'.format) + '] p=' + boot['p_not_better'].map('{:.3f}'.format))
    for subset in [ALL, second]:
        for scheme in ['group_kfold', 'interpolation']:
            print(f"-- {subset} | {scheme}")
            print(_table(b.query('subset == @subset and scheme == @scheme'), ['target', 'baseline'], 'v').to_string())

    main_thr = cfg['meta']['rule_based']['bad_relative_delta']
    g = gb.query('subset == @ALL and scheme == "group_kfold"')
    print(f"\n===== Good/Bad, ngưỡng Δ tương đối {main_thr:.0%}, group_kfold, 312 lượt =====")
    gm = g[g['threshold'] == main_thr]
    for metric in ['f1_bad', 'balanced_accuracy', 'accuracy', 'bad_rate_true', 'bad_rate_pred']:
        print(f"-- {metric}")
        print(_table(gm.assign(v=gm[f'{metric}_mean'].map('{:.3f}'.format)), 'method', 'v').to_string())
    print("-- ma trận nhầm lẫn (tp / fp / fn / tn, trung bình qua các lần lặp), tất cả tác vụ")
    print(gm[gm['task'] == ALL].set_index('method')[['tp_mean', 'fp_mean', 'fn_mean', 'tn_mean']]
          .round(1).to_string())
    print("-- độ nhạy theo ngưỡng: f1_bad, tất cả tác vụ")
    print(g[g['task'] == ALL].pivot(index='method', columns='threshold', values='f1_bad_mean').round(3).to_string())


if __name__ == '__main__':
    main()
