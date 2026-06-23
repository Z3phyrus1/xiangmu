# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:35:07 2026

@author: WANGLIANGFU
"""

# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# 路径按你项目结构设置
# =========================
RESULTS_ROOT = os.path.join('..', '..', '..', 'results', 'decoding')
FIG_ROOT = os.path.join('..', '..', '..', 'figures', 'decoding_mean')
os.makedirs(FIG_ROOT, exist_ok=True)

pattern = os.path.join(RESULTS_ROOT, '**', '*_generalization_results.npz')
files = sorted(glob.glob(pattern, recursive=True))
print(f'Found npz: {len(files)}')

if len(files) == 0:
    raise RuntimeError('No *_generalization_results.npz found')

# 收集
acc_diag_all = []
conf_diag_all = []
acc_diag_mean_sub = []
conf_diag_mean_sub = []
sub_names = []
times_common = None

for f in files:
    d = np.load(f, allow_pickle=True)

    # 兼容常见命名
    acc_key = None
    conf_key = None
    for k in d.files:
        lk = k.lower()
        if ('acc' in lk) and (('gen' in lk) or ('score' in lk) or ('matrix' in lk)):
            acc_key = k if acc_key is None else acc_key
        if ('conf' in lk) and (('gen' in lk) or ('score' in lk) or ('matrix' in lk)):
            conf_key = k if conf_key is None else conf_key

    # 更强兜底：按明确优先级找
    for k in ['gen_acc', 'scores_acc', 'generalization_acc', 'acc_matrix']:
        if k in d.files:
            acc_key = k
            break
    for k in ['gen_conf', 'scores_conf', 'generalization_conf', 'conf_matrix']:
        if k in d.files:
            conf_key = k
            break

    # 时间轴
    tkey = None
    for k in ['times_train', 'times_test', 'times', 'time']:
        if k in d.files:
            tkey = k
            break

    if acc_key is None and conf_key is None:
        print(f'[SKIP] no ACC/CONF matrix in: {f}')
        continue

    # subject 名
    base = os.path.basename(f).replace('_generalization_results.npz', '')
    sub_names.append(base)

    # 读取矩阵并取对角线
    if acc_key is not None:
        A = np.array(d[acc_key], dtype=float)
        if A.ndim == 2:
            diag_acc = np.diag(A)
            acc_diag_all.append(diag_acc)
            acc_diag_mean_sub.append(np.nanmean(diag_acc))
        else:
            acc_diag_all.append(None)
            acc_diag_mean_sub.append(np.nan)

    else:
        acc_diag_all.append(None)
        acc_diag_mean_sub.append(np.nan)

    if conf_key is not None:
        C = np.array(d[conf_key], dtype=float)
        if C.ndim == 2:
            diag_conf = np.diag(C)
            conf_diag_all.append(diag_conf)
            conf_diag_mean_sub.append(np.nanmean(diag_conf))
        else:
            conf_diag_all.append(None)
            conf_diag_mean_sub.append(np.nan)
    else:
        conf_diag_all.append(None)
        conf_diag_mean_sub.append(np.nan)

    # time
    if times_common is None and tkey is not None:
        times_common = np.array(d[tkey], dtype=float)

# numpy 化
acc_diag_mean_sub = np.array(acc_diag_mean_sub, dtype=float)
conf_diag_mean_sub = np.array(conf_diag_mean_sub, dtype=float)

# 仅保留有效
acc_valid = np.isfinite(acc_diag_mean_sub)
conf_valid = np.isfinite(conf_diag_mean_sub)

acc_vals = acc_diag_mean_sub[acc_valid]
conf_vals = conf_diag_mean_sub[conf_valid]

print('\n===== Subject-level diagonal mean AUC =====')
print(f'ACC n={len(acc_vals)} mean={np.mean(acc_vals):.4f} std={np.std(acc_vals, ddof=1) if len(acc_vals)>1 else np.nan:.4f}')
print(f'CONF n={len(conf_vals)} mean={np.mean(conf_vals):.4f} std={np.std(conf_vals, ddof=1) if len(conf_vals)>1 else np.nan:.4f}')

# t-test vs chance=0.5
def onesample_report(x, name):
    if len(x) < 2:
        print(f'{name}: n<2, skip t-test')
        return
    t, p = stats.ttest_1samp(x, popmean=0.5, nan_policy='omit')
    # Cohen's d (one-sample)
    d = (np.mean(x) - 0.5) / np.std(x, ddof=1)
    print(f'{name}: t={t:.4f}, p={p:.6f}, cohen_d={d:.4f}')

onesample_report(acc_vals, 'ACC diag-mean vs 0.5')
onesample_report(conf_vals, 'CONF diag-mean vs 0.5')

# 保存文本报告
report_path = os.path.join(FIG_ROOT, 'generalization_diag_stats.txt')
with open(report_path, 'w', encoding='utf-8') as wf:
    wf.write('Subject-level diagonal mean AUC\n')
    wf.write(f'ACC n={len(acc_vals)} mean={np.mean(acc_vals):.6f}\n')
    wf.write(f'CONF n={len(conf_vals)} mean={np.mean(conf_vals):.6f}\n')
    if len(acc_vals) >= 2:
        t, p = stats.ttest_1samp(acc_vals, popmean=0.5, nan_policy='omit')
        d = (np.mean(acc_vals) - 0.5) / np.std(acc_vals, ddof=1)
        wf.write(f'ACC t={t:.6f} p={p:.8f} d={d:.6f}\n')
    if len(conf_vals) >= 2:
        t, p = stats.ttest_1samp(conf_vals, popmean=0.5, nan_policy='omit')
        d = (np.mean(conf_vals) - 0.5) / np.std(conf_vals, ddof=1)
        wf.write(f'CONF t={t:.6f} p={p:.8f} d={d:.6f}\n')
print(f'Report saved: {report_path}')

# =========================
# 图1：箱线图 + 散点
# =========================
plt.figure(figsize=(6,5))
data_plot = []
labels = []
if len(acc_vals) > 0:
    data_plot.append(acc_vals); labels.append('ACC')
if len(conf_vals) > 0:
    data_plot.append(conf_vals); labels.append('CONF')

if len(data_plot) == 0:
    raise RuntimeError('No valid ACC/CONF values to plot')

bp = plt.boxplot(data_plot, labels=labels, showmeans=True)
for i, arr in enumerate(data_plot, start=1):
    x = np.random.normal(i, 0.04, size=len(arr))
    plt.scatter(x, arr, s=22, alpha=0.8)

plt.axhline(0.5, linestyle='--', linewidth=1)
plt.ylabel('Diagonal mean AUC (subject-level)')
plt.title('Generalization diagonal summary')
plt.tight_layout()
fig1 = os.path.join(FIG_ROOT, 'generalization_diag_boxplot.png')
plt.savefig(fig1, dpi=200)
plt.close()
print(f'Saved: {fig1}')

# =========================
# 图2：对角线时间曲线 mean±SEM
# =========================
def stack_valid(diag_list):
    arrs = []
    for x in diag_list:
        if x is None:
            continue
        if np.ndim(x) == 1:
            arrs.append(np.array(x, dtype=float))
    if len(arrs) == 0:
        return None
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.vstack(arrs)

A = stack_valid(acc_diag_all)
C = stack_valid(conf_diag_all)

if (A is not None) or (C is not None):
    plt.figure(figsize=(8,5))
    if A is not None:
        m = np.nanmean(A, axis=0)
        s = np.nanstd(A, axis=0, ddof=1) / np.sqrt(A.shape[0]) if A.shape[0] > 1 else np.zeros_like(m)
        x = np.arange(len(m)) if times_common is None else times_common[:len(m)]
        plt.plot(x, m, label=f'ACC (n={A.shape[0]})')
        plt.fill_between(x, m-s, m+s, alpha=0.2)
    if C is not None:
        m = np.nanmean(C, axis=0)
        s = np.nanstd(C, axis=0, ddof=1) / np.sqrt(C.shape[0]) if C.shape[0] > 1 else np.zeros_like(m)
        x = np.arange(len(m)) if times_common is None else times_common[:len(m)]
        plt.plot(x, m, label=f'CONF (n={C.shape[0]})')
        plt.fill_between(x, m-s, m+s, alpha=0.2)

    plt.axhline(0.5, linestyle='--', linewidth=1)
    plt.axvline(0.0, linestyle='-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('AUC (diagonal)')
    plt.title('Diagonal time-course (mean ± SEM)')
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(FIG_ROOT, 'generalization_diag_timecourse.png')
    plt.savefig(fig2, dpi=200)
    plt.close()
    print(f'Saved: {fig2}')

print('Done.')