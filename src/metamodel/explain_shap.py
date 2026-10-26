"""SHAP cho meta-model cuối: "trọng số" của từng dimension chất lượng, tách theo tác vụ (đóng góp task-aware của đề tài).

Theo docs/kien-truc-mo-hinh-va-ky-thuat.md mục 4 bước 4. Đầu vào: metamodel.pkl (train_metamodel.py) và experiment_results.parquet.

- Độ quan trọng = trung bình |SHAP| trên các dòng của tác vụ đó.
- `share` = tỷ trọng của dimension trong tổng độ quan trọng của các dimension **áp dụng được** cho tác vụ
  (theo configs/experiment_matrix.yaml). Dimension không áp dụng (Class Balance của regression, Target Accuracy
  của clustering) có `applicable = False`, `share` = NaN — docs/khac-biet-so-voi-bai-goc.md mục 4.5.
- Đặc trưng task/thuật toán vẫn được ghi (kind = task / algorithm) nhưng không tính vào `share`.

Chạy trên Colab (sau train_metamodel.py):
    python src/metamodel/explain_shap.py --project-dir /content/drive/MyDrive/data-quality-project
Ghi ra <project-dir>/results/shap_importance.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from profiling.common import PROFILE_KEYS  # noqa: E402

COLUMNS = ['target', 'task', 'feature', 'kind', 'applicable', 'mean_abs_shap', 'share']


def _kind(feature):
    if feature in PROFILE_KEYS:
        return 'profile'
    return 'task' if feature.startswith('task_') else 'algorithm'


def shap_importance(bundle, results):
    results = results.reset_index(drop=True)
    rows = []
    for target, model in bundle['models'].items():
        X = model.features(results)
        values = np.asarray(shap.TreeExplainer(model.model_).shap_values(X))
        for task, tcfg in bundle['tasks'].items():
            mask = (results['task'] == task).to_numpy()
            importance = np.abs(values[mask]).mean(axis=0)
            applicable = {f'profile_{d}' for d in tcfg['dimensions']}
            total = sum(v for f, v in zip(X.columns, importance) if f in applicable)
            for feature, value in zip(X.columns, importance):
                ok = feature in applicable
                rows.append({'target': target, 'task': task, 'feature': feature, 'kind': _kind(feature),
                             'applicable': ok if feature in PROFILE_KEYS else True,
                             'mean_abs_shap': float(value),
                             'share': float(value / total) if ok and total > 0 else np.nan})
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', required=True, help='thư mục chứa results/ (Drive)')
    args = p.parse_args(argv)

    out_dir = os.path.join(args.project_dir, 'results')
    bundle = joblib.load(os.path.join(out_dir, 'metamodel.pkl'))
    results = pd.read_parquet(os.path.join(out_dir, 'experiment_results.parquet'))
    imp = shap_importance(bundle, results)
    out = os.path.join(out_dir, 'shap_importance.csv')
    imp.to_csv(out, index=False)
    print("Đã lưu", out)

    pd.set_option('display.width', 250, 'display.float_format', '{:.3f}'.format)
    prof = imp[imp['kind'] == 'profile']
    for target, part in prof.groupby('target', sort=False):
        print(f"\nTỷ trọng SHAP của từng dimension theo tác vụ — nhãn {target} (NaN = không áp dụng):")
        print(part.pivot(index='feature', columns='task', values='share').to_string())


if __name__ == '__main__':
    main()
