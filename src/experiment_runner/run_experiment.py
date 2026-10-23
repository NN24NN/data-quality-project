"""Experiment Runner: làm bẩn -> profile -> huấn luyện/chấm điểm, mỗi lượt 1 dòng trong experiment_results.parquet.

Theo docs/thiet-ke-thi-nghiem-chi-tiet.md mục 4 (quy trình) và mục 5 (schema). Mọi tham số đọc từ configs/.

- ΔPerformance = baseline − raw, baseline lấy theo **đúng seed của lượt chạy** (results/baseline_performance.csv).
- Ghi file sau mỗi đơn vị (dimension, level, run); chạy lại thì bỏ qua các lượt đã có -> tiếp tục được từ chỗ dừng.
- λ = 0: mọi polluter trả về đúng dữ liệu sạch, nên kết quả không phụ thuộc dimension — chỉ chạy thật ở
  dimension đầu tiên rồi chép sang các dimension còn lại (có kiểm tra dữ liệu sau làm bẩn trùng bản sạch).

Chạy trên Colab (từ thư mục repo):
    python src/experiment_runner/run_experiment.py --project-dir /content/drive/MyDrive/data-quality-project \
        [--tasks regression] [--dimensions completeness] [--levels 0.0 0.5] [--runs 1] [--output ...]
"""
import argparse
import os
import sys
import time

import pandas as pd
import yaml

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from downstream import EVALUATORS  # noqa: E402
from downstream.common import split_train_test  # noqa: E402
from pollution import POLLUTERS  # noqa: E402
from profiling.common import PROFILE_KEYS  # noqa: E402
from profiling.pandas_profiler import compute_profile  # noqa: E402

KEY = ['task', 'algorithm', 'dimension', 'pollution_level', 'run_id']
COLUMNS = ['dataset', *KEY, *PROFILE_KEYS, 'raw_performance', 'baseline_performance', 'delta_performance']


def load_configs(config_dir=os.path.join(REPO_DIR, 'configs')):
    def read(name):
        with open(os.path.join(config_dir, name)) as f:
            return yaml.safe_load(f)
    return {
        'tasks': read('experiment_matrix.yaml')['tasks'],
        'datasets': read('datasets.yaml')['datasets'],
        'levels': read('pollution_levels.yaml')['levels'],
        'seeds': read('seeds.yaml'),
        'algorithms': read('algorithms.yaml')['algorithms'],
    }


def load_baseline(path):
    """{(dataset, algorithm, run_id): baseline_performance} từ baseline_performance.csv (cell 11.3)."""
    df = pd.read_csv(path)
    return {(r.dataset, r.algorithm, int(r.run_id)): float(r.baseline_performance) for r in df.itertuples()}


def _pick(selected, available, what):
    if selected is None:
        return list(available)
    unknown = [s for s in selected if s not in available]
    if unknown:
        raise ValueError(f"{what} không có trong configs: {unknown}")
    return [a for a in available if a in selected]          # giữ thứ tự của config


def _write(rows, out_path):
    """Ghi đè cả file qua file tạm — mất phiên giữa chừng cũng không để lại file hỏng."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + '.tmp'
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


def run_experiments(cfg, load_dataset, baseline, out_path,
                    tasks=None, dimensions=None, levels=None, run_ids=None):
    """Chạy các lượt được chọn (mặc định: toàn bộ ma trận), trả về DataFrame mọi dòng đã có trong out_path.

    load_dataset(name) -> bản sạch của dataset; baseline: dict như load_baseline.
    """
    rows = pd.read_parquet(out_path).to_dict('records') if os.path.exists(out_path) else []
    done = {tuple(r[k] for k in KEY): r for r in rows}
    seeds = cfg['seeds']

    for task in _pick(tasks, cfg['tasks'], 'task'):
        tcfg = cfg['tasks'][task]
        name = tcfg['dataset']
        dcfg = cfg['datasets'][name]
        target = dcfg['target']
        kw = dict(target=target, target_type=dcfg['target_type'], onehot_prefixes=dcfg['onehot_prefixes'])
        task_dims = _pick(dimensions, tcfg['dimensions'], 'dimension')
        task_levels = _pick(levels, cfg['levels'], 'level')
        task_runs = _pick(run_ids, seeds['runs'], 'run_id')
        units = [(d, lv, r) for d in task_dims for lv in task_levels for r in task_runs]

        clean, train, test = None, None, None
        for i, (dim, level, run_id) in enumerate(units, 1):
            keys = {algo: (task, algo, dim, level, run_id) for algo in tcfg['algorithms']}
            if all(k in done for k in keys.values()):
                continue
            if clean is None:                                 # chỉ đọc dữ liệu khi còn lượt phải chạy
                clean = load_dataset(name)
                if tcfg['split']:
                    train, test = split_train_test(clean, target, dcfg['target_type'],
                                                   tcfg['split']['test_size'], seeds['base_seed'])
                else:
                    train, test = clean, None                # clustering: làm bẩn toàn bộ, không tách
                n_classes = train[target].nunique() if dcfg['target_type'] == 'categorical' else None

            seed = seeds['runs'][run_id]
            t0 = time.time()
            polluted_train = POLLUTERS[dim](train, level, seed, **kw)
            polluted_test = None if test is None else POLLUTERS[dim](test, level, seed, **kw)

            reuse = {}
            if level == 0:
                reuse = {algo: r for r in rows for algo in keys
                         if (r['task'], r['algorithm'], r['pollution_level'], r['run_id']) == (task, algo, 0, run_id)}
            reused = len(reuse) == len(keys)
            if reused:
                assert polluted_train.equals(train), f"{dim} ở λ=0 phải trả về đúng dữ liệu sạch"
                new_rows = [{**reuse[algo], 'dimension': dim} for algo in keys]
            else:
                profile = compute_profile(polluted_train, clean=train, n_classes=n_classes, **kw)
                new_rows = []
                for algo in keys:
                    perf = EVALUATORS[task](algo, polluted_train, polluted_test, target, seed,
                                            cfg['algorithms'][algo])
                    base = baseline[(name, algo, run_id)]
                    new_rows.append({'dataset': name, 'task': task, 'algorithm': algo, 'dimension': dim,
                                     'pollution_level': level, 'run_id': run_id, **profile,
                                     'raw_performance': perf, 'baseline_performance': base,
                                     'delta_performance': base - perf})

            for r in new_rows:
                if done.get(tuple(r[k] for k in KEY)) is None:
                    rows.append(r)
                    done[tuple(r[k] for k in KEY)] = r
                note = '  (chép từ λ=0 đã chạy)' if reused else ''
                if level == 0 and abs(r['delta_performance']) > 1e-9:
                    note += '  [CẢNH BÁO: λ=0 lệch baseline — config thuật toán đã đổi sau cell 11.3?]'
                print(f"[{task} {i}/{len(units)}] {dim} λ={level} run {run_id} {r['algorithm']}: "
                      f"raw {r['raw_performance']:.4f}  Δ {r['delta_performance']:+.4f}{note}", flush=True)
            _write(rows, out_path)
            print(f"    ({time.time() - t0:.0f} giây, đã có {len(rows)} dòng)", flush=True)

    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', required=True, help='thư mục chứa data/processed và results (Drive)')
    p.add_argument('--output', help='mặc định <project-dir>/results/experiment_results.parquet')
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--dimensions', nargs='+')
    p.add_argument('--levels', nargs='+', type=float)
    p.add_argument('--runs', nargs='+', type=int)
    args = p.parse_args(argv)

    cfg = load_configs()
    baseline = load_baseline(os.path.join(args.project_dir, 'results', 'baseline_performance.csv'))
    out = args.output or os.path.join(args.project_dir, 'results', 'experiment_results.parquet')

    def load_dataset(name):
        return pd.read_parquet(os.path.join(args.project_dir, 'data', 'processed', cfg['datasets'][name]['file']))

    t0 = time.time()
    result = run_experiments(cfg, load_dataset, baseline, out,
                             tasks=args.tasks, dimensions=args.dimensions, levels=args.levels, run_ids=args.runs)
    print(f"\nXong: {len(result)} dòng trong {out} ({(time.time() - t0) / 60:.1f} phút)")


if __name__ == '__main__':
    main()
