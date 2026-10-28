"""Biểu đồ H1–H10 của báo cáo (docs/ket-qua-thuc-nghiem.md mục 9).

Mỗi hàm nhận DataFrame đã đọc từ file kết quả trên Drive (không chép số bằng tay) và trả về Figure;
`save_fig` ghi .png vào results/figures/. Không đặt tham số thí nghiệm ở đây: tên dimension, tác vụ, ngưỡng
đều lấy từ dữ liệu hoặc từ tham số hàm.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

TASKS = ['regression', 'classification', 'clustering']
DIMENSIONS = ['completeness', 'feature_accuracy', 'target_accuracy', 'uniqueness', 'class_balance']
METHODS = ['metamodel', 'rule_based', 'weighted_score', 'dqsops_pca']
METHOD_LABELS = {'metamodel': 'Meta-model', 'rule_based': 'Rule-based', 'weighted_score': 'Weighted score',
                 'dqsops_pca': 'DQSOps-PCA'}
METRIC_LABELS = {'regression': 'R²', 'classification': 'F1-macro', 'clustering': 'AMI'}

# Bảng màu cố định để cùng một dimension / tác vụ / phương pháp luôn cùng màu ở mọi hình (an toàn cho người mù màu)
DIM_COLORS = dict(zip(DIMENSIONS, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']))
TASK_COLORS = dict(zip(TASKS, ['#0072B2', '#E69F00', '#009E73']))
METHOD_COLORS = dict(zip(METHODS, ['#0072B2', '#999999', '#E69F00', '#D55E00']))

plt.rcParams.update({'figure.dpi': 100, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True, 'font.size': 10})


def save_fig(fig, name, figures_dir, dpi=200):
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


def drop_secondary(df, task='clustering', dimension='class_balance'):
    """Bảng phụ 288 lượt: bỏ Class Balance × clustering (khac-biet-so-voi-bai-goc.md mục 4.6)."""
    return df[~((df['task'] == task) & (df['dimension'] == dimension))]


def _ordered(values, order):
    """Các giá trị theo thứ tự `order`, phần còn lại theo thứ tự xuất hiện trong dữ liệu."""
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + [v for v in present if v not in order]


def _pm(ax, x, mean, std, **kw):
    line, = ax.plot(x, mean, marker='o', **kw)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.18, linewidth=0)


# ---------------------------------------------------------------- H1
def theory_curve(dimension, task, levels):
    """Điểm lý thuyết theo λ (cach-trien-khai-chi-tiet-tung-giai-doan.md, Giai đoạn 3); None nếu không có công thức."""
    lam = np.asarray(levels, dtype=float)
    if dimension in ('completeness', 'uniqueness') or (dimension == 'target_accuracy' and task == 'classification'):
        return 1 - lam
    if dimension == 'feature_accuracy' or (dimension == 'target_accuracy' and task == 'regression'):
        return 1 - np.sqrt(lam) * np.sqrt(2 / np.pi)   # nhiễu Gaussian; chỉ khớp chính xác với cột liên tục
    return None


def plot_h1(check, task_of_dataset):
    """check: bảng cell 10.2 (cột dataset, dimension, level + điểm các dimension); task_of_dataset: {dataset: task}."""
    check = check.assign(task=check['dataset'].map(task_of_dataset))
    check = check.assign(measured=check.apply(lambda r: r[r['dimension']], axis=1))
    tasks = _ordered(check['task'], TASKS)
    dims = _ordered(check['dimension'], DIMENSIONS)
    fig, axes = plt.subplots(len(tasks), len(dims), figsize=(3.0 * len(dims), 2.7 * len(tasks)), sharex=True,
                             sharey=True, squeeze=False)
    for i, task in enumerate(tasks):
        for j, dim in enumerate(dims):
            ax = axes[i, j]
            d = check[(check['task'] == task) & (check['dimension'] == dim)].sort_values('level')
            if d.empty:
                ax.axis('off')
                continue
            ax.plot(d['level'], d['measured'], marker='o', color=DIM_COLORS.get(dim), label='đo được')
            theory = theory_curve(dim, task, d['level'])
            if theory is not None:
                ax.plot(d['level'], theory, '--', color='black', linewidth=1, label='lý thuyết')
            if i == 0:
                ax.set_title(dim)
            if j == 0:
                ax.set_ylabel(f'{task}\nđiểm profile')
            if i == len(tasks) - 1:
                ax.set_xlabel('λ')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('H1 — Điểm profile đo được sau khi làm bẩn theo λ', y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H2
def plot_h2(baseline):
    """baseline: baseline_performance.csv (task, algorithm, run_id, baseline_performance)."""
    tasks = _ordered(baseline['task'], TASKS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        g = baseline[baseline['task'] == task].groupby('algorithm', sort=False)['baseline_performance']
        m, s = g.mean(), g.std()
        ax.bar(m.index, m.values, yerr=s.values, capsize=4, color=TASK_COLORS.get(task), alpha=0.85)
        ax.set_title(task)
        ax.set_ylabel(METRIC_LABELS.get(task, 'điểm'))
        ax.tick_params(axis='x', rotation=20)
        for x, (mean, std) in enumerate(zip(m.values, s.values)):
            ax.text(x, mean + std, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('H2 — Hiệu năng baseline sạch (trung bình ± độ lệch chuẩn qua các seed)', y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H3
def plot_h3(results, title_suffix=''):
    """Δ theo λ: hàng = tác vụ, cột = thuật toán, đường = dimension; dải = độ lệch chuẩn qua các run."""
    tasks = _ordered(results['task'], TASKS)
    algos = {t: _ordered(results.loc[results['task'] == t, 'algorithm'], []) for t in tasks}
    ncol = max(len(a) for a in algos.values())
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(4.2 * ncol, 3.0 * len(tasks)), sharex=True, sharey='row',
                             squeeze=False)
    for i, task in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(algos[task]):
                ax.axis('off')
                continue
            algo = algos[task][j]
            d = results[(results['task'] == task) & (results['algorithm'] == algo)]
            for dim in _ordered(d['dimension'], DIMENSIONS):
                g = d[d['dimension'] == dim].groupby('pollution_level')['delta_performance']
                _pm(ax, g.mean().index.to_numpy(), g.mean().to_numpy(), g.std().fillna(0).to_numpy(),
                    color=DIM_COLORS.get(dim), label=dim)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(f'{task} — {algo}')
            if j == 0:
                ax.set_ylabel(f'Δ{METRIC_LABELS.get(task, "")}')
            if i == len(tasks) - 1:
                ax.set_xlabel('λ')
    seen = {}
    for ax in axes.ravel():          # gom chú giải của mọi ô (class_balance chỉ có ở một số tác vụ)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=len(seen), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'H3 — Đường cong suy giảm hiệu năng theo λ{title_suffix}', y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H4
def plot_h4(results, level=0.8, title_suffix=''):
    """Heatmap Δ tương đối = Δ / baseline: hàng = dimension, cột = thuật toán."""
    d = results[np.isclose(results['pollution_level'], level)].copy()
    d['rel'] = d['delta_performance'] / d['baseline_performance']
    cols = [a for t in _ordered(d['task'], TASKS) for a in _ordered(d.loc[d['task'] == t, 'algorithm'], [])]
    rows = _ordered(d['dimension'], DIMENSIONS)
    table = d.groupby(['dimension', 'algorithm'])['rel'].mean().unstack('algorithm').reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 0.7 * len(rows) + 1.6))
    im = ax.imshow(table.to_numpy(dtype=float), cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha='right')
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(table.to_numpy(dtype=float)):
        ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Δ / baseline')
    ax.set_title(f'H4 — Suy giảm tương đối ở λ = {level}{title_suffix}')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H5
def plot_h5(comparison, subset='all', scheme='group_kfold', target='absolute'):
    """comparison: method_comparison.csv. Cột nhóm theo tác vụ, màu theo phương pháp, thanh = độ lệch chuẩn qua seed."""
    d = comparison[(comparison['subset'] == subset) & (comparison['scheme'] == scheme)
                   & (comparison['target'] == target)]
    tasks = _ordered(d['task'], [*TASKS, 'all'])
    metrics = [('mae', 'MAE (thấp hơn là tốt)'), ('r2', 'R²'), ('spearman', 'Spearman')]
    methods = [m for m in METHODS if m in set(d['method'])]
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (metric, label) in zip(axes, metrics):
        for k, method in enumerate(methods):
            part = d[d['method'] == method].set_index('task').reindex(tasks)
            ax.bar(np.arange(len(tasks)) + k * width, part[f'{metric}_mean'], width, yerr=part[f'{metric}_std'],
                   capsize=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(np.arange(len(tasks)) + width * (len(methods) - 1) / 2, tasks)
        ax.set_title(label)
        ax.axhline(0, color='black', linewidth=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'H5 — So sánh meta-model với baseline ({scheme}, nhãn {target}, tập {subset})', y=1.03)
    fig.tight_layout()
    return fig


def h5_table(comparison, subset='all', scheme='group_kfold', target='absolute'):
    """Bảng 'trung bình ± độ lệch chuẩn' cho H5 (dán thẳng vào báo cáo)."""
    d = comparison[(comparison['subset'] == subset) & (comparison['scheme'] == scheme)
                   & (comparison['target'] == target)].copy()
    for m in ('mae', 'r2', 'spearman'):
        d[m] = d[f'{m}_mean'].map('{:.3f}'.format) + ' ± ' + d[f'{m}_std'].map('{:.3f}'.format)
    return d.set_index(['task', 'method'])[['mae', 'r2', 'spearman']]


# ---------------------------------------------------------------- H6
def plot_h6(preds, scheme='group_kfold', target='absolute'):
    """preds: metamodel_cv_predictions.parquet. Mỗi điểm = một lượt chạy (dự đoán trung bình qua các seed chia).

    Rule-based chỉ cho nhãn Good (0) / Bad (1) nên vẽ dạng dải thay vì scatter.
    """
    d = preds[(preds['scheme'] == scheme) & (preds['target'] == target)]
    key = ['task', 'algorithm', 'dimension', 'pollution_level', 'run_id', 'method']
    d = d.groupby(key, as_index=False)[['y_true', 'y_pred']].mean()
    methods = [m for m in METHODS if m in set(d['method'])]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 4.0), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, method in zip(axes[0], methods):
        for task in _ordered(d['task'], TASKS):
            p = d[(d['method'] == method) & (d['task'] == task)]
            y = p['y_pred'] + (rng.normal(0, 0.02, len(p)) if method == 'rule_based' else 0)
            ax.scatter(p['y_true'], y, s=12, alpha=0.6, color=TASK_COLORS.get(task), label=task)
        if method != 'rule_based':
            lim = [min(d['y_true'].min(), d['y_pred'].min()), max(d['y_true'].max(), d['y_pred'].max())]
            ax.plot(lim, lim, 'k--', linewidth=1)
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel('Δ thật')
        ax.set_ylabel('Good (0) / Bad (1)' if method == 'rule_based' else 'Δ dự đoán')
    axes[0, 0].legend(fontsize=8, markerscale=1.5)
    fig.suptitle(f'H6 — Δ dự đoán so với Δ thật ({scheme}, nhãn {target})', y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H7
def plot_h7(goodbad, threshold, methods=('rule_based', 'metamodel'), subset='all', scheme='group_kfold'):
    """goodbad: goodbad_comparison.csv. Ma trận nhầm lẫn trung bình qua seed ở task='all', lớp dương = Bad."""
    d = goodbad[(goodbad['subset'] == subset) & (goodbad['scheme'] == scheme) & (goodbad['task'] == 'all')
                & np.isclose(goodbad['threshold'], threshold)]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), squeeze=False)
    for ax, method in zip(axes[0], methods):
        r = d[d['method'] == method].iloc[0]
        m = np.array([[r['tn_mean'], r['fp_mean']], [r['fn_mean'], r['tp_mean']]])
        ax.imshow(m, cmap='Blues')
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', color='white' if v > m.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['dự đoán Good', 'dự đoán Bad'])
        ax.set_yticks([0, 1], ['thật Good', 'thật Bad'])
        ax.set_title(f"{METHOD_LABELS[method]}\nrecall Bad = {r['recall_mean']:.2f}, precision = {r['precision_mean']:.2f}")
    fig.suptitle(f'H7 — Ma trận nhầm lẫn Good/Bad (Bad nếu Δ tương đối > {threshold:.0%})', y=1.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H8, H9
def plot_h8(shap, target='absolute', note=None):
    """shap: shap_importance.csv. Tỷ trọng (share) của từng dimension, tính trong các dimension áp dụng được của tác vụ."""
    d = shap[(shap['target'] == target) & (shap['kind'] == 'profile') & shap['applicable'].astype(bool)].copy()
    d['dimension'] = d['feature'].str.replace('profile_', '', regex=False)
    dims = _ordered(d['dimension'], DIMENSIONS)
    tasks = _ordered(d['task'], TASKS)
    width = 0.8 / len(tasks)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, task in enumerate(tasks):
        p = d[d['task'] == task].set_index('dimension').reindex(dims)
        ax.bar(np.arange(len(dims)) + k * width, p['share'], width, color=TASK_COLORS.get(task), label=task)
    ax.set_xticks(np.arange(len(dims)) + width * (len(tasks) - 1) / 2, dims)
    ax.set_ylabel('Tỷ trọng SHAP trung bình |giá trị|')
    ax.legend()
    ax.set_title('H8 — Tầm quan trọng của từng dimension theo tác vụ (meta-model)')
    if note:
        fig.text(0.01, -0.04, note, fontsize=8, ha='left', wrap=True)
    fig.tight_layout()
    return fig


def plot_h9(pca_weights, shap, target='absolute'):
    """Trọng số PC1 của DQSOps (một bộ cho mọi tác vụ) cạnh tỷ trọng SHAP theo tác vụ."""
    w = pca_weights.assign(dimension=lambda x: x['dimension'].str.replace('profile_', '', regex=False))
    s = shap[(shap['target'] == target) & (shap['kind'] == 'profile') & shap['applicable'].astype(bool)].copy()
    s['dimension'] = s['feature'].str.replace('profile_', '', regex=False)
    dims = _ordered(w['dimension'], DIMENSIONS)
    tasks = _ordered(s['task'], TASKS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    ax.bar(dims, w.set_index('dimension').reindex(dims)['pc1_weight'], color='#D55E00')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"DQSOps-PCA: trọng số PC1 (giải thích {w['explained_variance_ratio'].iloc[0]:.1%} phương sai)")
    ax.tick_params(axis='x', rotation=25)
    ax = axes[1]
    width = 0.8 / len(tasks)
    for k, task in enumerate(tasks):
        p = s[s['task'] == task].set_index('dimension').reindex(dims)
        ax.bar(np.arange(len(dims)) + k * width, p['share'], width, color=TASK_COLORS.get(task), label=task)
    ax.set_xticks(np.arange(len(dims)) + width * (len(tasks) - 1) / 2, dims)
    ax.tick_params(axis='x', rotation=25)
    ax.set_title('Meta-model: tỷ trọng SHAP theo tác vụ')
    ax.legend()
    fig.suptitle('H9 — Trọng số cố định (DQSOps) so với trọng số theo ngữ cảnh (meta-model)', y=1.03)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- H10
def plot_h10(bench):
    """bench: scalability_benchmark.csv. Trung vị qua các lần 'ok'; mốc không chạy được đánh dấu ✕ ở cuối đường."""
    engines = ['pandas', 'spark']
    colors = {'pandas': '#0072B2', 'spark': '#D55E00'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, label) in zip(axes, [('seconds', 'Thời gian (giây)'), ('peak_rss_mb', 'RAM đỉnh của tiến trình (MB)')]):
        for engine in engines:
            e = bench[bench['engine'] == engine]
            ok = e[e['status'] == 'ok'].groupby('n_rows')[col].median().sort_index()
            ax.plot(ok.index, ok.values, marker='o', color=colors[engine], label=engine)
            failed = sorted(set(e['n_rows']) - set(ok.index))
            for n in failed:
                ax.axvline(n, color=colors[engine], linestyle=':', alpha=0.6)
                ax.text(n, ax.get_ylim()[1], f'{engine}: hết RAM', rotation=90, va='top', ha='right', fontsize=8,
                        color=colors[engine])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Số dòng')
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle('H10 — Profiling: pandas và PySpark theo số dòng (HIGGS)', y=1.03)
    fig.tight_layout()
    return fig
