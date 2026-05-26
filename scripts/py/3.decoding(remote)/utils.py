# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:23:06 2025

@author: DELL
"""
from mne.stats import permutation_cluster_1samp_test
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import mne
from autoreject import AutoReject
import common_settings as CS
from mne.decoding import SlidingEstimator, cross_val_multiscore, Scaler, Vectorizer, LinearModel, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
def get_session_number(filename):
    import re
    match = re.search(r'sess(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def determine_event_id(subject, session_num):
    sub_num = int(subject.replace('sub', ''))
    if (sub_num + session_num) % 2 == 0:
        event_id_task1 = CS.event_id_line
        event_id_task2 = CS.event_id_dot
    else:
        event_id_task1 = CS.event_id_dot
        event_id_task2 = CS.event_id_line
    return {'task1': event_id_task1, 'task2': event_id_task2}
#原eog = 0.3 muscle = 0.9
def run_ica_cleaning(epochs, picks, figures_dir, session_num, task_name, 
                     threshold_eog=0.4, threshold_muscle=0.9):

    cov = mne.compute_covariance(epochs, tmin=epochs.tmin, tmax=epochs.tmax)
    ica = mne.preprocessing.ICA(noise_cov=cov, **CS.ica_args)
    ica.fit(epochs, picks=picks)
    
    eog_idx, _ = ica.find_bads_eog(epochs,
        ch_name=['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8','F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8'],
        threshold=threshold_eog, measure='correlation')
    muscle_idx, _ = ica.find_bads_muscle(epochs, threshold=threshold_muscle)
    
    ica.exclude.extend(eog_idx + muscle_idx)
    ica.exclude = list(set(ica.exclude))
    
    if len(ica.exclude) > 0:
        figures = ica.plot_properties(epochs, picks=ica.exclude)
        for fig, idx in zip(figures, ica.exclude):
            fig.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_ICA{idx}.png'), 
                       dpi=300, bbox_inches='tight')
    else:
        print(f"{task_name}: 本次没有需要排除的 ICA 成分，跳过成分可视化")
    
    clean_epochs = ica.apply(epochs)
    return clean_epochs


def run_autoreject(epochs, picks, figures_dir, session_num, task_name):

    ar = AutoReject(picks=picks, **CS.autoreject_args)
    ar.fit(epochs)
    ar_epochs, reject_log = ar.transform(epochs, return_log=True)
    
    fig_reject = reject_log.plot('horizontal')
    fig_reject.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_reject.png'), 
                       dpi=300, bbox_inches='tight')
    
    return ar_epochs, reject_log


def save_epochs(clean_epochs, ar_epochs, reject_log, results_dir, file_base):

    clean_epochs.save(os.path.join(results_dir, f'clean_epochs_{file_base}-epo.fif'), overwrite=True)
    ar_epochs.save(os.path.join(results_dir, f'AR_epochs_{file_base}-epo.fif'), overwrite=True)
    reject_log.save(os.path.join(results_dir, f'reject_log_{file_base}.npz'), overwrite=True)


def plot_and_save_joint(epochs, event_id, figures_dir, session_num, task_name):

    for key in event_id.keys():
        fig = epochs[key].average().plot_joint()
        fig.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_clean_{key}.png'), 
                   dpi=300, bbox_inches='tight')


def concatenate_baseline(epochs_data, baseline_data):

    return np.concatenate([baseline_data, epochs_data], axis=2)


def get_event_indices(events, events_per_trial, position, num_trials):

    indices = [trial_idx * events_per_trial + position 
               for trial_idx in range(num_trials)
               if trial_idx * events_per_trial + position < len(events)]
    return indices

def compute_accuracy_labels_from_raw_events(raw_events, events_per_trial=6):
    task1_accuracy = []
    task2_accuracy = []
    
    # 跳过第一个 New Segment 事件
    num_trials = (len(raw_events) - 1) // events_per_trial
    
    for trial_idx in range(num_trials):
        base = 1 + trial_idx * events_per_trial
        
        if base + 5 >= len(raw_events):
            continue
        
        # Task1 事件和 Response1 事件
        task1_event = raw_events[base, -1]          # Task1 呈现
        response1_event = raw_events[base + 2, -1]  # Response1
        
        # Task2 事件和 Response2 事件
        task2_event = raw_events[base + 3, -1]      # Task2 呈现
        response2_event = raw_events[base + 5, -1]  # Response2
        
        # 计算正确性：事件个位数相同表示正确
        task1_correct = 1 if (task1_event % 10) == (response1_event % 10) else 0
        task2_correct = 1 if (task2_event % 10) == (response2_event % 10) else 0
        
        task1_accuracy.append(task1_correct)
        task2_accuracy.append(task2_correct)
    
    return task1_accuracy, task2_accuracy


def get_task1_accuracy_labels(events, events_per_trial):
    labels = []
    num_trials = (len(events) - 1) // events_per_trial
    
    for trial_idx in range(num_trials):
        base = 1 + trial_idx * events_per_trial
        
        if base + 2 >= len(events):
            continue
        
        task1_event = events[base, -1]
        response1_event = events[base + 2, -1]
        
        if task1_event % 10 == response1_event % 10:
            labels.append(1)
        else:
            labels.append(0)
    
    return np.array(labels)


# ============================================================
# 解码器创建函数
# ============================================================
def make_decoder_pipeline():
    clf = make_pipeline(
        # Scaler(epochs_info),
        # Vectorizer(),
        StandardScaler(),
        LogisticRegressionCV(
        **CS.classifier_args
        )
    )
    return clf


def make_sliding_estimator():
    clf = make_decoder_pipeline()
    sliding_estimator = SlidingEstimator(
        clf,
        **CS.temporal_decoding_args
    )
    return sliding_estimator


def make_generalizing_estimator():
    clf = make_decoder_pipeline()
    gen_estimator = GeneralizingEstimator(
        clf,
        **CS.temporal_generalizing_args
    )
    return gen_estimator


# ============================================================
# 解码运行函数
# ============================================================
# def run_cv_decoding(X, y, epochs_info):
#     sliding_estimator = make_sliding_estimator(epochs_info)
#     cv = StratifiedKFold(
#         n_splits=CS.decoding_args['n_splits'],
#         shuffle=True,
#         random_state=CS.decoding_args['random_state']
#     )
#     scores = cross_val_multiscore(
#         sliding_estimator, X, y,
#         cv=cv,
#         n_jobs=CS.decoding_args['n_jobs']
#     )
#     return scores


def run_generalization_decoding(X, y, epochs_info):

    gen_estimator = make_generalizing_estimator(epochs_info)
    cv = StratifiedKFold(
        n_splits=CS.decoding_args['n_splits'],
        shuffle=True,
        random_state=CS.decoding_args['random_state']
    )
    scores = cross_val_multiscore(
        gen_estimator, X, y,
        cv=cv,
        n_jobs=CS.decoding_args['n_jobs']
    )
    return scores
def compute_accuracy_from_epochs(epochs_task, epochs_response):
    task_events = epochs_task.events[:, 2]
    response_events = epochs_response.events[:, 2]
        
    if len(task_events) != len(response_events):
        print(f"[WARN] 事件数量不匹配! task: {len(task_events)}, response: {len(response_events)}")
        
    accuracy = np.array([
            1 if (t % 10) == (r % 10) else 0 
            for t, r in zip(task_events, response_events)
        ])
    return accuracy

# def run_cross_task_decoding(X_train, y_train, X_test, y_test, epochs_info):
#     sliding_estimator = make_sliding_estimator(epochs_info)
#     sliding_estimator.fit(X_train, y_train)
#     scores = sliding_estimator.score(X_test, y_test)
#     return scores


# ============================================================
# 绘图函数
# ============================================================
def plot_time_decoding(times, scores, title, save_path=None, chance_level=0.5):

    
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, label='Decoding score (AUC)', linewidth=2)
    plt.axhline(chance_level, color='k', linestyle='--', label='Chance level')
    plt.axvline(0, color='k', linestyle='-', label='Stimulus onset')
    plt.fill_between(times, chance_level, scores, where=(scores > chance_level),
                     alpha=0.3, color='green')
    plt.fill_between(times, chance_level, scores, where=(scores < chance_level),
                     alpha=0.3, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



    # === 新增：保存试次剔除日志的函数 ===
def save_preprocessing_log(subject, file_identifier, task_name, total_trials, 
                               ica_epochs, ar_epochs, output_dir):
    current_status = []
        
        # 检查 ICA 阶段的剔除
    if ica_epochs.drop_log is not None:
        for d in ica_epochs.drop_log:
            if len(d) == 0:
                current_status.append("OK") # 暂时保留
            else:
                current_status.append("Dropped_ICA") # ICA 阶段（或之前）被剔除
    else:
            # 如果没有 drop_log，默认全是 OK
        current_status = ["OK"] * len(ica_epochs)

        # 2.检查 AutoReject 阶段的剔除
    ok_indices = [i for i, s in enumerate(current_status) if s == "OK"]
        
    if ar_epochs.drop_log is not None:
        limit = min(len(ok_indices), len(ar_epochs.drop_log))
            
        for k in range(limit):
            original_idx = ok_indices[k]
            if len(ar_epochs.drop_log[k]) > 0:
                current_status[original_idx] = "Dropped_AutoReject"
        
        # 3.保存为 CSV
    df_log = pd.DataFrame({
        'Subject': subject,
        'File': file_identifier,
        'Task': task_name,
        'Trial_ID': range(len(current_status)), # 原始第几个试次
        'Status': current_status
        })
        
    log_name = f"{subject}_{file_identifier}_{task_name}_drop_log.csv"
    df_log.to_csv(os.path.join(output_dir, log_name), index=False)
    print(f"  --> 日志已保存: {log_name}")
    
    # === 新增：ICA前ERP绘图函数 ===
def plot_pre_ica_erp(epochs, event_id, figures_dir, session_num, task_name, subject, file_stem):
    """
    在ICA处理前绘制原始ERP图
    """
    print(f"  绘制ICA前ERP图...")
    
    # 创建ICA前的专属目录
    pre_ica_fig_dir = os.path.join(figures_dir, 'pre_ICA_ERPs')
    if not os.path.exists(pre_ica_fig_dir):
        os.makedirs(pre_ica_fig_dir)
    
    # 复制epochs进行基线矫正（不修改原始数据）
    epochs_copy = epochs.copy()
    epochs_copy.apply_baseline(CS.baseline_task1)
    
    # 按条件绘制ERP图
    fig_erp = epochs_copy.plot_image(
        picks=['Fz', 'Cz', 'Pz', 'Oz'],  # 选择中线电极
        combine='mean',
        show=False
    )
    
    # 保存ERP图
    erp_fig_path = os.path.join(pre_ica_fig_dir, 
                                f"{subject}_{file_stem}_{task_name}_pre_ICA_ERP.png")
    fig_erp[0].savefig(erp_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存: {erp_fig_path}")
    
    # 绘制所有通道的平均ERP
    fig_all = epochs_copy.average().plot(spatial_colors=True, show=False)
    all_fig_path = os.path.join(pre_ica_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_pre_ICA_allchannels.png")
    fig_all.savefig(all_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存: {all_fig_path}")
    
    # 绘制地形图序列（时间序列）- 修复时间范围问题
    evoked = epochs_copy.average()
    
    # 方案1：使用自动选择的时间点（MNE会根据数据自动选择合适的时间点）
    fig_topo1 = evoked.plot_topomap(times='auto', show=False)
    topo_fig_path1 = os.path.join(pre_ica_fig_dir,
                                 f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_auto.png")
    fig_topo1.savefig(topo_fig_path1, dpi=300, bbox_inches='tight')
    print(f"    已保存: {topo_fig_path1}")
    
    # 方案2：手动指定在范围内的特定时间点
    # 获取epochs的实际时间范围
    tmin, tmax = epochs_copy.tmin, epochs_copy.tmax
    print(f"    实际时间范围: [{tmin:.2f}, {tmax:.2f}] 秒")
    
    # 创建在范围内的自定义时间点
    # 例如：从-0.3秒到0.3秒，每0.1秒一个点
    custom_times = np.arange(max(tmin, -0.3), min(tmax, 0.3), 0.1)
    
    # 确保至少有一个时间点在范围内
    if len(custom_times) > 0:
        fig_topo2 = evoked.plot_topomap(times=custom_times, show=False)
        topo_fig_path2 = os.path.join(pre_ica_fig_dir,
                                     f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_custom.png")
        fig_topo2.savefig(topo_fig_path2, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path2}")
    else:
        print("    警告：没有可用的自定义时间点在地形图范围内")
    
    # 方案3：使用整数个时间点（如5个等间隔点）
    n_times = 5  # 选择5个时间点
    times_equally_spaced = np.linspace(tmin + 0.1, tmax - 0.1, n_times)
    
    # 过滤掉可能超出范围的时间点
    times_equally_spaced = times_equally_spaced[(times_equally_spaced >= tmin) & (times_equally_spaced <= tmax)]
    
    if len(times_equally_spaced) > 0:
        fig_topo3 = evoked.plot_topomap(times=times_equally_spaced, show=False)
        topo_fig_path3 = os.path.join(pre_ica_fig_dir,
                                     f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_equally_spaced.png")
        fig_topo3.savefig(topo_fig_path3, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path3}")
    
    # 关闭所有图形，避免内存泄漏
    import matplotlib.pyplot as plt
    plt.close('all')
    
    return pre_ica_fig_dir



def plot_post_autoreject_erp(ar_epochs, event_id, figures_dir, session_num, task_name, subject, file_stem):
    """
    在 AutoReject 处理后绘制 ERP 图
    """
    print(f"  绘制 AutoReject 后 ERP 图...")
    
    # 创建 AutoReject 后的专属目录
    post_ar_fig_dir = os.path.join(figures_dir, 'post_AutoReject_ERPs')
    if not os.path.exists(post_ar_fig_dir):
        os.makedirs(post_ar_fig_dir)
    
    # 复制 epochs 进行基线矫正
    epochs_copy = ar_epochs.copy()
    epochs_copy.apply_baseline(CS.baseline_task1)
    
    # 1.绘制中线电极的 ERP 图像
    fig_erp = epochs_copy.plot_image(
        picks=['Fz', 'Cz', 'Pz', 'Oz'],
        combine='mean',
        show=False,
        title=f'{subject} {file_stem} Task{task_name} - AutoReject后 ERP'
    )
    
    erp_fig_path = os.path.join(post_ar_fig_dir, 
                                f"{subject}_{file_stem}_{task_name}_POST_AR_ERP.png")
    fig_erp[0].savefig(erp_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存:  {erp_fig_path}")
    
    # 2.绘制所有通道的平均 ERP（带标题）
    evoked = epochs_copy.average()
    fig_all = evoked.plot(spatial_colors=True, show=False)
    fig_all.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 全通道 ERP', 
                     fontsize=14, fontweight='bold')
    all_fig_path = os.path.join(post_ar_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_POST_AR_allchannels.png")
    fig_all.savefig(all_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存:  {all_fig_path}")
    
    # 3.绘制地形图序列（自动选择时间点）
    fig_topo1 = evoked.plot_topomap(times='auto', show=False)
    fig_topo1.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 地形图（自动时间点）',
                       fontsize=12, fontweight='bold')
    topo_fig_path1 = os.path.join(post_ar_fig_dir,
                                 f"{subject}_{file_stem}_{task_name}_POST_AR_topomap_auto.png")
    fig_topo1.savefig(topo_fig_path1, dpi=300, bbox_inches='tight')
    print(f"    已保存: {topo_fig_path1}")
    
    # 4.绘制等间隔时间点的地形图
    tmin, tmax = epochs_copy.tmin, epochs_copy.tmax
    n_times = 5
    times_equally_spaced = np.linspace(tmin + 0.1, tmax - 0.1, n_times)
    times_equally_spaced = times_equally_spaced[(times_equally_spaced >= tmin) & (times_equally_spaced <= tmax)]
    
    if len(times_equally_spaced) > 0:
        fig_topo2 = evoked.plot_topomap(times=times_equally_spaced, show=False)
        fig_topo2.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 地形图（等间隔时间点）',
                           fontsize=12, fontweight='bold')
        topo_fig_path2 = os.path.join(post_ar_fig_dir,
                                     f"{subject}_{file_stem}_{task_name}_POST_AR_topomap_equally_spaced.png")
        fig_topo2.savefig(topo_fig_path2, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path2}")
    
    # 5.绘制条件对比图（如果有多个条件）
    if len(event_id) > 1:
        fig_compare, axes = plt.subplots(1, len(event_id), figsize=(15, 4))
        
        for idx, (key, _) in enumerate(event_id.items()):
            evoked_cond = epochs_copy[key].average()
            fig_single = evoked_cond.plot(spatial_colors=True, show=False, axes=axes[idx] if len(event_id) > 1 else axes)
            axes[idx].set_title(f'{key}', fontweight='bold')
        
        fig_compare.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 条件对比',
                             fontsize=14, fontweight='bold')
        compare_fig_path = os.path.join(post_ar_fig_dir,
                                       f"{subject}_{file_stem}_{task_name}_POST_AR_conditions.png")
        fig_compare.savefig(compare_fig_path, dpi=300, bbox_inches='tight')
        print(f"    已保存:  {compare_fig_path}")
    
    # 关闭所有图形
    plt.close('all')
    
    return post_ar_fig_dir


# ============================================================
# 获取 W 阈值
# ============================================================
def get_id_w_threshold(fit_par_path, sub_id):
    try:
        df = pd.read_csv(fit_par_path, sep='\s+')
        df.columns = df.columns.str.replace('"', '').str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('"', '').str.strip()
        
        row = df[df['id'].astype(str) == str(sub_id)]
        if len(row) == 0: return None
        
        if 'model' in row.columns:
            sub2_row = row[row['model'] == 'sub2']
            if not sub2_row.empty: return float(sub2_row['w'].values[0])
        return float(row['w'].iloc[0])
    except:
        return None

def get_idsess_w_threshold(fit_par_path, sub_id):
    try:
        df = pd.read_csv(fit_par_path, sep='\s+')
        df.columns = df.columns.str.replace('"', '').str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('"', '').str.strip()
        
        # 修改：提取 id 列中 '_' 前的数字部分与 sub_id 比较
        row = df[df['id'].str.split('_').str[0] == str(sub_id)]
        if len(row) == 0: return None
        
        if 'model' in row.columns:
            sub2_row = row[row['model'] == 'sub2']
            if not sub2_row.empty: return float(sub2_row['w1'].values[0])
        return float(row['w1'].iloc[0])
    except:
        return None

def plot_temporal_generalization(scores, times_train, times_test, title, save_path, vmin=0.4, vmax=0.6):
    fig, ax = plt.subplots(figsize=(8, 8))
        
    im = ax.imshow(
        scores,
        interpolation='lanczos',
        origin='lower',
        cmap='RdBu_r',
            extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        
    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.axvline(0, color='k', linestyle='-', linewidth=1)
        
    ax.set_xlabel('Testing Time - Task2 Stimulus (s)', fontsize=12)
    ax.set_ylabel('Training Time - Task1 Response (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
        
    cbar = plt.colorbar(im, ax=ax, label='AUC')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存至: {save_path}")
    plt.show()
        
    return fig
def plot_temporal_generalization_with_contour(scores, times_train, times_test,
                                              title, save_path,
                                              vmin=0.4, vmax=0.6,
                                              contour_levels=None):
    if contour_levels is None:
        contour_levels = np.linspace(vmin, vmax, 6)  # 5段等高线

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(scores, origin='lower', interpolation='lanczos',
                   extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
                   aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    # 叠加等高线
    cs = ax.contour(times_test, times_train, scores, levels=contour_levels,
                    colors='k', linewidths=0.8, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='AUC')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {save_path}")
# ============================================================
# 新增：绘制带置换检验聚类高亮的图
# ============================================================
def plot_with_cluster_highlight(times, scores_mean, scores_sem, permutation_result, 
                                title, save_path, chance_level=0.5):
    fig, ax = plt.subplots(figsize=(12, 6))
     
    # 绘制AUC曲线
    ax.plot(times, scores_mean, label="Decoding Score", lw=2, color='blue')
    ax.fill_between(times, scores_mean + scores_sem, scores_mean - scores_sem, 
                    color='blue', alpha=0.2, label="SEM")
    
    # 绘制随机水平线
    ax.axhline(chance_level, color="k", linestyle="--", label=f"Chance ({chance_level})")
    ax.axvline(0.0, color="k", linestyle="-", alpha=0.5)
    
    # 处理置换检验结果
    if permutation_result is not None:
        t_vals, clusters, cluster_p_vals, _ = permutation_result
        
        for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_vals)):
            if p_val <= 0.05:  # 只高亮显著的聚类
                
                # --- 情况 A: Cluster 是布尔掩码 (out_type='mask') ---
                if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
                    # 找到掩码为 True 的时间点索引
                    idx = np.where(cluster)[0]
                    if len(idx) > 0:
                        # 可能会有不连续的片段，简单起见我们画出每个连续段
                        # 或者简单地高亮该掩码覆盖的范围
                        # 这里使用 fill_between 直接填充掩码区域
                        ax.fill_between(times, 0, 1, where=cluster, 
                                        transform=ax.get_xaxis_transform(),
                                        color='yellow', alpha=0.2, zorder=0)

                # --- 情况 B: Cluster 是切片元组 (标准输出) ---
                elif isinstance(cluster, tuple) or isinstance(cluster, list):
                    # 通常是 (slice(start, stop, None), )
                    for sl in cluster:
                        if isinstance(sl, slice):
                            start = sl.start if sl.start is not None else 0
                            stop = sl.stop if sl.stop is not None else len(times)
                            ax.axvspan(times[start], times[stop-1], 
                                       alpha=0.2, color='yellow', zorder=0)
    
    # 设置图形属性
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图形已保存至: {save_path}")
    plt.show()
    
    return fig
    return fig
def read_concat(files, baseline, crop_tmin, crop_tmax):
    eps = [mne.read_epochs(f, verbose=False) for f in files]
    for e in eps: e.apply_baseline(baseline)
    epochs = mne.concatenate_epochs(eps)
    epochs.crop(crop_tmin, crop_tmax).resample(500) # 下采样加速矩阵计算
    return epochs
get_idsess_w_threshold