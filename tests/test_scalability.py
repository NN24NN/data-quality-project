"""Test Scalability study trên HIGGS giả lập nhỏ: chuẩn bị dữ liệu, cắt mốc, đo trong tiến trình con, xử lý thất bại.

Chạy: python -m pytest tests/test_scalability.py -q   (chỉ Linux; phần PySpark cần Java, bỏ qua nếu thiếu pyspark)
"""
import signal

import numpy as np
import pandas as pd
import pytest

from profiling.common import PROFILE_KEYS
from scalability import spark_scale_study as ss

CFG = ss.load_configs()
SIZES = [100, 300]
N_RAW = 400


@pytest.fixture(scope='module')
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp('scale')
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.normal(size=(N_RAW, len(ss.HIGGS_COLUMNS))).astype('float32'), columns=ss.HIGGS_COLUMNS)
    raw['target'] = rng.integers(0, 2, N_RAW).astype('float32')
    raw.iloc[5] = raw.iloc[4]                                          # 1 dòng trùng
    gz = d / 'HIGGS.csv.gz'
    raw.to_csv(gz, header=False, index=False, compression='gzip')
    raw_path = str(d / 'raw' / 'higgs_full_raw.parquet')
    assert ss.prepare_raw(str(gz), raw_path) == N_RAW
    return ss.make_subsets(raw_path, SIZES, str(d / 'work'))


def small_cfg(**scale):
    return {**CFG, 'scale': {**CFG['scale'], 'repeats': 1, 'timeout_seconds': 300, **scale}}


def test_prepare_and_subsets(paths):
    assert sorted(paths) == SIZES
    for n, path in paths.items():
        df = pd.read_parquet(path)
        assert len(df) == n and list(df.columns) == [*ss.HIGGS_COLUMNS, ss.ID_COL]
        assert (df[ss.ID_COL].to_numpy() == np.arange(n)).all()
        assert df['target'].dtype == 'int8'
    small, big = (pd.read_parquet(paths[n]) for n in SIZES)
    pd.testing.assert_frame_equal(small, big.head(SIZES[0]))              # mốc nhỏ là phần đầu của mốc lớn


def test_classify_failure():
    assert ss.classify_failure(-signal.SIGKILL, '') == 'oom'
    assert ss.classify_failure(ss.MEMORY_ERROR_EXIT, '') == 'oom'
    assert ss.classify_failure(1, 'java.lang.OutOfMemoryError: Java heap space') == 'oom'
    assert ss.classify_failure(1, 'ValueError: something else') == 'error'


def test_pandas_worker_in_subprocess(paths):
    res = ss.run_worker('pandas', paths[300], paths[100], timeout=300)
    assert res['status'] == 'ok' and res['peak_rss_mb'] > 0 and 0 < res['read_seconds'] <= res['seconds']
    assert res['profile_completeness'] == 1.0 and res['profile_feature_accuracy'] == 1.0
    assert res['profile_uniqueness'] < 1.0                                 # có 1 dòng trùng


def test_failed_engine_skips_larger_sizes(paths, tmp_path):
    """Pandas hết RAM ở mốc 100 -> mốc 300 ghi 'skipped', không chạy thử."""
    calls = []

    def fake_run(engine, path, warmup, timeout):
        calls.append((engine, path))
        return {'status': 'oom', 'error': 'killed'} if engine == 'pandas' else {'status': 'ok', 'seconds': 1.0}

    df = ss.run_study(small_cfg(repeats=2), paths, str(tmp_path / 'bench.csv'), run=fake_run)
    pandas = df[df['engine'] == 'pandas'].set_index(['n_rows', 'repeat'])['status']
    assert pandas[(100, 1)] == 'oom' and (pandas.drop((100, 1)) == 'skipped').all()
    assert [e for e, _ in calls].count('pandas') == 1 and [e for e, _ in calls].count('spark') == 4


def test_resume_skips_finished_runs(paths, tmp_path):
    out = str(tmp_path / 'bench.csv')
    calls = []

    def fake_run(engine, path, warmup, timeout):
        calls.append(engine)
        return {'status': 'ok', 'seconds': 1.0, **dict.fromkeys(PROFILE_KEYS, 1.0)}

    first = ss.run_study(small_cfg(), paths, out, engines=['pandas'], run=fake_run)
    again = ss.run_study(small_cfg(), paths, out, engines=['pandas'], run=fake_run)
    assert len(calls) == len(SIZES) and len(again) == len(first) == len(SIZES)
    assert list(pd.read_csv(out).columns) == ss.COLUMNS


def test_check_parity():
    base = {'status': 'ok', 'repeat': 1, **dict.fromkeys(PROFILE_KEYS, 0.5)}
    df = pd.DataFrame([{**base, 'engine': 'pandas', 'n_rows': 10}, {**base, 'engine': 'spark', 'n_rows': 10},
                       {**base, 'engine': 'pandas', 'n_rows': 20},
                       {**base, 'engine': 'spark', 'n_rows': 20, 'profile_uniqueness': 0.6}])
    par = ss.check_parity(df, 1e-9).set_index('n_rows')
    assert par.loc[10, 'match'] and not par.loc[20, 'match']
    assert par.loc[20, 'max_rel_diff'] == pytest.approx(0.1 / 0.6)


def test_spark_matches_pandas_end_to_end(paths, tmp_path):
    pytest.importorskip('pyspark')
    df = ss.run_study(small_cfg(), paths, str(tmp_path / 'bench.csv'))
    assert (df['status'] == 'ok').all(), df[['engine', 'n_rows', 'status', 'error']]
    assert df.loc[df['engine'] == 'spark', 'peak_rss_mb'].isna().all()   # không đo bộ nhớ Spark
    par = ss.check_parity(df, CFG['scale']['parity_rtol'])
    assert list(par['n_rows']) == SIZES and par['match'].all()


def test_timeout_kills_worker(paths):
    pytest.importorskip('pyspark')
    res = ss.run_worker('spark', paths[300], paths[100], timeout=0.5)     # JVM chưa kịp khởi động
    assert res['status'] == 'timeout'
