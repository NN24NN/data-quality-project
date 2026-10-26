"""Scalability study (Giai đoạn 8): thời gian profiling bằng pandas và PySpark trên HIGGS full ở các mốc kích thước.

Thiết kế: docs/khac-biet-so-voi-bai-goc.md mục 4.7. Tham số: configs/scalability.yaml. Chỉ chạy trên Linux (Colab).

- Profile bản sạch so với chính nó: khối lượng tính toán (đếm ô thiếu, ghép theo row_id cho Accuracy, tìm dòng trùng,
  đếm lớp) như khi dữ liệu bẩn, mà không phải làm bẩn 11M dòng hay ghi dữ liệu bẩn ra đĩa.
- Mỗi lần đo là một tiến trình con riêng (cả nhóm tiến trình, gồm JVM của Spark): hết RAM chỉ làm chết tiến trình con
  -> 'oom'; quá giờ -> 'timeout'. Engine đã thất bại ở một mốc thì các mốc lớn hơn ghi 'skipped'.
- Thời gian = đọc parquet + profile. Spark chạy khởi động một lần trên mốc nhỏ nhất (bỏ kết quả) trước khi bấm giờ.
- Bộ nhớ: chỉ đo đỉnh RSS của pandas; Spark chạy với giới hạn cố định spark_driver_memory.

Chạy trên Colab (từ thư mục repo):
    python src/scalability/spark_scale_study.py prepare --project-dir /content/drive/MyDrive/data-quality-project
    python src/scalability/spark_scale_study.py run --project-dir /content/drive/MyDrive/data-quality-project
prepare: tải HIGGS full về --work-dir, đổi sang parquet trên Drive (bỏ qua nếu đã có).
run: cắt các mốc (kèm row_id) ra --work-dir, đo, ghi dần <project-dir>/results/scalability_benchmark.csv.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from profiling.common import PROFILE_KEYS  # noqa: E402

ID_COL = 'row_id'
ENGINES = ['pandas', 'spark']
RESULT_PREFIX = 'RESULT '
MEMORY_ERROR_EXIT = 3
FAILED = {'oom', 'timeout', 'error'}
COLUMNS = ['engine', 'n_rows', 'repeat', 'status', 'seconds', 'read_seconds', 'peak_rss_mb', 'memory_limit',
           'cpu_count', 'total_ram_gb', *PROFILE_KEYS, 'error']
HIGGS_COLUMNS = ['target'] + [f'f{i:02d}' for i in range(1, 29)]     # cùng tên cột với HIGGS mẫu (cell 7.3)


def load_configs(config_dir=os.path.join(REPO_DIR, 'configs')):
    def read(name):
        with open(os.path.join(config_dir, name)) as f:
            return yaml.safe_load(f)
    return {'scale': read('scalability.yaml'), 'dataset': read('datasets.yaml')['datasets']['higgs_sample']}


def _profile_kw(cfg):
    d = cfg['dataset']
    return dict(target=d['target'], target_type=d['target_type'], onehot_prefixes=d['onehot_prefixes'])


# --- Chuẩn bị dữ liệu ---

def prepare_raw(gz_path, out_path):
    """HIGGS.csv.gz -> parquet, giữ nguyên nội dung (target int8, feature float32). Trả về số dòng."""
    df = pd.read_csv(gz_path, header=None, names=HIGGS_COLUMNS, dtype='float32')
    df['target'] = df['target'].astype('int8')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return len(df)


def make_subsets(raw_path, sizes, out_dir):
    """Cắt n dòng đầu cho mỗi mốc, thêm cột row_id (0..n-1). Bỏ qua file đã có. Trả về {n: đường dẫn}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    table, paths = None, {}
    for n in sorted(sizes):
        path = os.path.join(out_dir, f'higgs_{n}.parquet')
        if not os.path.exists(path):
            table = table if table is not None else pq.read_table(raw_path)
            if n > table.num_rows:
                raise ValueError(f"Mốc {n} lớn hơn số dòng của dữ liệu ({table.num_rows})")
            sub = table.slice(0, n).append_column(ID_COL, pa.array(np.arange(n, dtype='int64')))
            pq.write_table(sub, path + '.tmp')
            os.replace(path + '.tmp', path)
        paths[n] = path
    return paths


# --- Một lần đo (chạy trong tiến trình con) ---

def measure_pandas(path, cfg):
    import resource

    import pyarrow.parquet as pq
    from profiling import pandas_profiler
    cols = [c for c in pq.read_schema(path).names if c != ID_COL]   # pandas ghép theo index, không cần row_id
    t0 = time.perf_counter()
    df = pd.read_parquet(path, columns=cols)
    clean = pd.read_parquet(path, columns=cols)                       # ground truth: bản đọc riêng, như khi dùng thật
    read_seconds = time.perf_counter() - t0
    profile = pandas_profiler.compute_profile(df, clean=clean, **_profile_kw(cfg))
    seconds = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KB -> MB
    return {'seconds': seconds, 'read_seconds': read_seconds, 'peak_rss_mb': peak, **profile}


def measure_spark(path, warmup_path, cfg):
    from pyspark.sql import SparkSession

    from profiling import spark_profiler
    scfg = cfg['scale']
    spark = (SparkSession.builder.master(scfg['spark_master']).appName('scale-study')
             .config('spark.driver.memory', scfg['spark_driver_memory'])
             .config('spark.ui.enabled', 'false').getOrCreate())
    try:
        def run(p):
            return spark_profiler.compute_profile(spark.read.parquet(p), clean=spark.read.parquet(p),
                                                  id_col=ID_COL, **_profile_kw(cfg))
        run(warmup_path)                     # khởi JVM + truy vấn đầu tiên, không tính giờ
        t0 = time.perf_counter()
        profile = run(path)
        seconds = time.perf_counter() - t0
    finally:
        spark.stop()
    return {'seconds': seconds, **profile}


def _worker(args):
    cfg = load_configs()
    try:
        if args.engine == 'pandas':
            res = measure_pandas(args.path, cfg)
        else:
            res = measure_spark(args.path, args.warmup_path, cfg)
    except MemoryError:
        sys.exit(MEMORY_ERROR_EXIT)
    print(RESULT_PREFIX + json.dumps(res), flush=True)


def classify_failure(returncode, stderr):
    """Bị hệ điều hành giết (SIGKILL, thường do hết RAM) hoặc lỗi hết bộ nhớ của Python/JVM -> 'oom'."""
    if returncode in (-signal.SIGKILL, MEMORY_ERROR_EXIT) or 'MemoryError' in stderr:   # gồm OutOfMemoryError của JVM
        return 'oom'
    return 'error'


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_worker(engine, path, warmup_path, timeout):
    """Một lần đo trong tiến trình con riêng. Trả về dict có 'status' ('ok' / 'oom' / 'timeout' / 'error')."""
    cmd = [sys.executable, os.path.abspath(__file__), 'worker', '--engine', engine,
           '--path', path, '--warmup-path', warmup_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)                # giết cả JVM của Spark, không chỉ tiến trình Python
        proc.communicate()
        return {'status': 'timeout'}
    _kill_group(proc.pid)                    # dọn JVM còn sót (nếu có)
    for line in out.splitlines():
        if line.startswith(RESULT_PREFIX):
            return {'status': 'ok', **json.loads(line[len(RESULT_PREFIX):])}
    return {'status': classify_failure(proc.returncode, err), 'error': err.strip()[-500:]}


# --- Cả nghiên cứu ---

def machine_info():
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    return {'cpu_count': os.cpu_count(), 'total_ram_gb': round(ram / 1e9, 1)}


def run_study(cfg, paths, out_path, engines=ENGINES, run=run_worker):
    """Đo mọi (mốc, engine, lần), ghi out_path sau mỗi lần; chạy lại thì bỏ qua các lần đã có."""
    scfg = cfg['scale']
    rows = pd.read_csv(out_path).to_dict('records') if os.path.exists(out_path) else []
    done = {(r['engine'], int(r['n_rows']), int(r['repeat'])) for r in rows}
    failed = {r['engine'] for r in rows if r['status'] in FAILED}
    sizes = sorted(paths)
    machine = machine_info()
    for n in sizes:
        for engine in engines:
            for rep in range(1, scfg['repeats'] + 1):
                if (engine, n, rep) in done:
                    continue
                if engine in failed:
                    res = {'status': 'skipped'}
                else:
                    res = run(engine, paths[n], paths[sizes[0]], scfg['timeout_seconds'])
                limit = f"spark.driver.memory={scfg['spark_driver_memory']}" if engine == 'spark' else 'RAM máy'
                rows.append({**dict.fromkeys(COLUMNS), 'engine': engine, 'n_rows': n, 'repeat': rep,
                             'memory_limit': limit, **machine, **res})
                done.add((engine, n, rep))
                if res['status'] in FAILED:
                    failed.add(engine)
                tmp = out_path + '.tmp'
                pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
                os.replace(tmp, out_path)
                secs = f"{res['seconds']:.1f} giây" if res.get('seconds') is not None else ''
                mem = f", đỉnh RAM {res['peak_rss_mb']:.0f} MB" if res.get('peak_rss_mb') is not None else ''
                print(f"[{engine:6s} {n:>10,} dòng, lần {rep}] {res['status']} {secs}{mem}", flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(df):
    """Trung vị thời gian và bộ nhớ đỉnh (lần 'ok'), cùng trạng thái, theo (mốc, engine)."""
    ok = df[df['status'] == 'ok']
    med = ok.groupby(['n_rows', 'engine']).agg(seconds=('seconds', 'median'), peak_rss_mb=('peak_rss_mb', 'median'),
                                               n_ok=('repeat', 'count'))
    status = df.groupby(['n_rows', 'engine'])['status'].agg(lambda s: '/'.join(sorted(set(s))))
    return med.join(status, how='right').unstack('engine')


def check_parity(df, rtol):
    """Ở mỗi mốc cả hai engine cùng chạy được: sai số tương đối lớn nhất giữa điểm profile của pandas và Spark."""
    ok = df[df['status'] == 'ok'].sort_values('repeat').groupby(['n_rows', 'engine']).first()
    rows = []
    for n in sorted({i[0] for i in ok.index}):
        if (n, 'pandas') not in ok.index or (n, 'spark') not in ok.index:
            continue
        a = ok.loc[(n, 'pandas'), PROFILE_KEYS].to_numpy(dtype=float)
        b = ok.loc[(n, 'spark'), PROFILE_KEYS].to_numpy(dtype=float)
        both = ~(np.isnan(a) & np.isnan(b))
        rel = np.abs(a - b)[both] / np.maximum(np.maximum(np.abs(a), np.abs(b))[both], 1e-12)
        rows.append({'n_rows': n, 'max_rel_diff': float(rel.max()) if rel.size else 0.0,
                     'match': bool(np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True))})
    return pd.DataFrame(rows, columns=['n_rows', 'max_rel_diff', 'match'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'run'):
        s = sub.add_parser(name)
        s.add_argument('--project-dir', required=True, help='thư mục chứa data/ và results/ (Drive)')
        s.add_argument('--work-dir', default='/content/higgs_scale', help='ổ tạm cho file tải về và các mốc')
    w = sub.add_parser('worker')                      # dùng nội bộ: một lần đo trong tiến trình con
    w.add_argument('--engine', choices=ENGINES, required=True)
    w.add_argument('--path', required=True)
    w.add_argument('--warmup-path', required=True)
    args = p.parse_args(argv)

    if args.cmd == 'worker':
        return _worker(args)

    cfg = load_configs()
    raw_path = os.path.join(args.project_dir, cfg['scale']['raw_file'])
    if args.cmd == 'prepare':
        if os.path.exists(raw_path):
            print("HIGGS full đã có trên Drive, bỏ qua:", raw_path)
            return
        import urllib.request
        os.makedirs(args.work_dir, exist_ok=True)
        gz = os.path.join(args.work_dir, 'HIGGS.csv.gz')
        if not os.path.exists(gz):
            print("Đang tải", cfg['scale']['source_url'], "(~2,8GB) ...", flush=True)
            urllib.request.urlretrieve(cfg['scale']['source_url'], gz + '.part')
            os.replace(gz + '.part', gz)
        n = prepare_raw(gz, raw_path)
        print(f"Đã lưu {raw_path}: {n:,} dòng, {os.path.getsize(raw_path) / 1e9:.2f} GB")
        return

    t0 = time.time()
    paths = make_subsets(raw_path, cfg['scale']['sizes'], args.work_dir)
    print(f"Đã có {len(paths)} mốc trong {args.work_dir} ({time.time() - t0:.0f} giây). Máy: {machine_info()}", flush=True)
    out = os.path.join(args.project_dir, 'results', 'scalability_benchmark.csv')
    df = run_study(cfg, paths, out)
    pd.set_option('display.width', 250, 'display.max_columns', 20)
    print(f"\nĐã lưu {out} ({len(df)} dòng, {(time.time() - t0) / 60:.1f} phút)")
    print("\nTrung vị thời gian (giây), đỉnh RAM pandas (MB), trạng thái theo mốc:")
    print(summarize(df).to_string(float_format='{:.1f}'.format))
    print("\nĐối chiếu điểm profile pandas vs PySpark:")
    print(check_parity(df, cfg['scale']['parity_rtol']).to_string(index=False))


if __name__ == '__main__':
    main()
