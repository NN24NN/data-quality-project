"""Meta-model (Giai đoạn 6): Gradient Boosting dự đoán ΔPerformance từ (profile, tác vụ, thuật toán), đánh giá chéo cùng 3 baseline.

Theo docs/kien-truc-mo-hinh-va-ky-thuat.md mục 4–5. Mọi tham số đọc từ configs/metamodel.yaml và seeds.yaml.

- Đặc trưng: 5 điểm profile (NaN -> 1.0) + one-hot 3 tác vụ + one-hot 6 thuật toán (khac-biet-so-voi-bai-goc.md mục 4.5).
- Nhãn: Δ tuyệt đối (chính) và Δ tương đối = Δ / baseline_performance (phụ) — mục 4.4.
- Chia train/test (mục 4.3):
    group_kfold    StratifiedGroupKFold, nhóm = (task, dimension, λ); λ = 0 gộp 1 nhóm mỗi task vì các dòng λ = 0
                   là bản chép giống hệt nhau giữa các dimension. Lặp lại với mỗi seed trong seeds.yaml runs.
    interpolation  train trên các λ còn lại, test trên interpolation_test_level (0,5).
  Cả 4 phương pháp fit lại trên đúng phần train của mỗi lần chia.
- Mô hình cuối (cho SHAP) huấn luyện trên toàn bộ bảng kết quả.

Chạy trên Colab (từ thư mục repo):
    python src/metamodel/train_metamodel.py --project-dir /content/drive/MyDrive/data-quality-project
Ghi ra <project-dir>/results/: metamodel_cv_predictions.parquet, metamodel.pkl, dqsops_pca_weights.csv
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baselines import DQSOpsPCA, RuleBased, WeightedScore  # noqa: E402
from profiling.common import PROFILE_KEYS  # noqa: E402

KEY = ['task', 'algorithm', 'dimension', 'pollution_level', 'run_id']
PRED_COLUMNS = [*KEY, 'scheme', 'repeat_seed', 'fold', 'target', 'method', 'y_true', 'y_pred']


def load_configs(config_dir=os.path.join(REPO_DIR, 'configs')):
    def read(name):
        with open(os.path.join(config_dir, name)) as f:
            return yaml.safe_load(f)
    return {
        'meta': read('metamodel.yaml'),
        'seeds': read('seeds.yaml'),
        'tasks': read('experiment_matrix.yaml')['tasks'],
    }


def make_target(results, kind):
    if kind == 'absolute':
        return results['delta_performance'].to_numpy(dtype=float)
    if kind == 'relative':
        return (results['delta_performance'] / results['baseline_performance']).to_numpy(dtype=float)
    raise ValueError(f"Nhãn không hợp lệ: {kind}")


def build_features(results, tasks, fill_value):
    """Ma trận đặc trưng của meta-model: profile (NaN -> fill_value) + one-hot task + one-hot thuật toán."""
    X = results[PROFILE_KEYS].fillna(fill_value).astype(float)
    for task in tasks:
        X[f'task_{task}'] = (results['task'] == task).astype(float)
    for task, tcfg in tasks.items():
        for algo in tcfg['algorithms']:
            X[f'algo_{algo}'] = (results['algorithm'] == algo).astype(float)
    return X


def make_groups(results):
    """Nhóm không được tách rời khi chia: (task, dimension, λ); λ = 0 là 1 nhóm mỗi task."""
    dim = results['dimension'].where(results['pollution_level'] != 0, '*')
    return results['task'] + '|' + dim + '|' + results['pollution_level'].astype(str)


def group_kfold_splits(results, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(results, results['task'], make_groups(results)))


def interpolation_split(results, level):
    test = np.isclose(results['pollution_level'].to_numpy(dtype=float), level)
    return [(np.flatnonzero(~test), np.flatnonzero(test))]


class MetaModel:
    """GradientBoostingRegressor nhận thẳng bảng kết quả, tự dựng đặc trưng — cùng giao diện với 3 baseline."""

    def __init__(self, params, tasks, fill_value, seed):
        self.params, self.tasks, self.fill_value, self.seed = params, tasks, fill_value, seed

    def features(self, results):
        return build_features(results, self.tasks, self.fill_value)

    def fit(self, results, y):
        self.model_ = GradientBoostingRegressor(**self.params, random_state=self.seed).fit(self.features(results), y)
        return self

    def predict(self, results):
        return self.model_.predict(self.features(results))


def make_methods(cfg):
    meta = cfg['meta']
    return {
        'metamodel': MetaModel(meta['gradient_boosting'], cfg['tasks'], meta['not_applicable_fill'],
                               cfg['seeds']['base_seed']),
        'weighted_score': WeightedScore(),
        'dqsops_pca': DQSOpsPCA(meta['not_applicable_fill']),
        'rule_based': RuleBased(meta['rule_based']['profile_threshold']),   # y_pred = 1 (Bad) / 0 (Good)
    }


def iter_splits(results, cfg):
    """(scheme, repeat_seed, [(train_idx, test_idx), ...]) cho cả hai cách chia."""
    for seed in cfg['seeds']['runs'].values():
        yield 'group_kfold', seed, group_kfold_splits(results, cfg['meta']['cv']['n_splits'], seed)
    yield 'interpolation', cfg['seeds']['base_seed'], interpolation_split(results, cfg['meta']['interpolation_test_level'])


def cross_validate(results, cfg):
    """Dự đoán out-of-fold của 4 phương pháp × 2 nhãn × mọi lần chia, dạng bảng dài (PRED_COLUMNS)."""
    results = results.reset_index(drop=True)
    targets = {kind: make_target(results, kind) for kind in cfg['meta']['targets']}
    parts = []
    for scheme, seed, splits in iter_splits(results, cfg):
        for fold, (tr, te) in enumerate(splits):
            train, test = results.iloc[tr], results.iloc[te]
            for kind, y in targets.items():
                for name, method in make_methods(cfg).items():
                    pred = method.fit(train, y[tr]).predict(test)
                    parts.append(test[KEY].assign(scheme=scheme, repeat_seed=seed, fold=fold, target=kind,
                                                  method=name, y_true=y[te], y_pred=pred))
    return pd.concat(parts, ignore_index=True)[PRED_COLUMNS]


def fit_final(results, cfg):
    """Meta-model cuối cho mỗi nhãn (huấn luyện trên toàn bộ bảng) — đầu vào của explain_shap.py."""
    results = results.reset_index(drop=True)
    models = {kind: make_methods(cfg)['metamodel'].fit(results, make_target(results, kind))
              for kind in cfg['meta']['targets']}
    names = list(build_features(results.head(1), cfg['tasks'], cfg['meta']['not_applicable_fill']).columns)
    return {'models': models, 'tasks': cfg['tasks'], 'feature_names': names}


def pca_weights(results, cfg):
    """Trọng số PC1 của DQSOps-PCA fit trên toàn bộ bảng (PCA không phụ thuộc nhãn) — dữ liệu cho H9."""
    model = DQSOpsPCA(cfg['meta']['not_applicable_fill']).fit(results, make_target(results, 'absolute'))
    w, ratio = model.weights()
    return w.rename_axis('dimension').reset_index().assign(explained_variance_ratio=ratio)


def summarize(preds):
    """MAE trung bình qua các lần lặp, theo phương pháp × tác vụ — chỉ để xem nhanh; số liệu chính tính ở Giai đoạn 7."""
    reg = preds[preds['method'] != 'rule_based'].assign(err=lambda d: (d['y_true'] - d['y_pred']).abs())
    per_repeat = reg.groupby(['scheme', 'target', 'method', 'task', 'repeat_seed'])['err'].mean()
    overall = reg.groupby(['scheme', 'target', 'method', 'repeat_seed'])['err'].mean()
    table = per_repeat.groupby(['scheme', 'target', 'method', 'task']).mean().unstack('task')
    table['tất cả'] = overall.groupby(['scheme', 'target', 'method']).mean()
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', required=True, help='thư mục chứa results/ (Drive)')
    p.add_argument('--input', help='mặc định <project-dir>/results/experiment_results.parquet')
    args = p.parse_args(argv)

    cfg = load_configs()
    out_dir = os.path.join(args.project_dir, 'results')
    results = pd.read_parquet(args.input or os.path.join(out_dir, 'experiment_results.parquet'))
    print(f"Đọc {len(results)} dòng kết quả thí nghiệm")

    t0 = time.time()
    preds = cross_validate(results, cfg)
    preds.to_parquet(os.path.join(out_dir, 'metamodel_cv_predictions.parquet'), index=False)
    joblib.dump(fit_final(results, cfg), os.path.join(out_dir, 'metamodel.pkl'))
    weights = pca_weights(results, cfg)
    weights.to_csv(os.path.join(out_dir, 'dqsops_pca_weights.csv'), index=False)
    print(f"Xong trong {time.time() - t0:.0f} giây: {len(preds)} dòng dự đoán, metamodel.pkl, dqsops_pca_weights.csv")

    pd.set_option('display.width', 250, 'display.float_format', '{:.4f}'.format)
    print("\nMAE (out-of-fold, trung bình qua các lần lặp) — chỉ để xem nhanh, rule-based không có MAE:")
    print(summarize(preds).to_string())
    print("\nTrọng số PC1 của DQSOps-PCA:")
    print(weights.to_string(index=False))


if __name__ == '__main__':
    # Gọi qua tên module đầy đủ: MetaModel trong metamodel.pkl phải mang tên `metamodel.train_metamodel`,
    # không phải `__main__`, thì explain_shap.py mới nạp lại được.
    from metamodel.train_metamodel import main as _main
    _main()
