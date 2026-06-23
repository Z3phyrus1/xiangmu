
# -*- coding: utf-8 -*-
"""
Task1 Response confidence decoding with EEG-behavior session alignment

新增：
1. 所有试次中的高/低信心解码：
   label = abs1 > 1

2. 正确试次中的高/低信心解码：
   先筛选 y_acc == 1
   再在正确试次中解码 abs1 > 1

注意：
单被试内的 cluster test 是基于 CV split 的探索性统计。
正式结论仍建议做 group-level 统计。
"""

import os
import re
from glob import glob
from shutil import copyfile

import numpy as np
import pandas as pd
import mne

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from mne.decoding import SlidingEstimator, cross_val_multiscore


# ============================================================
# External settings
# ============================================================
copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')

import common_settings as CS
import utils


# ============================================================
# Global settings
# ============================================================
ABS_THRESHOLD = 1.0
CHANCE_LEVEL = 0.5


# ============================================================
# Helper functions
# ============================================================
def get_behavior_sessions_for_subject(df_pred, sub_id_int):
    df_sub = df_pred[df_pred['id'].astype(int) == int(sub_id_int)].copy()

    if df_sub.empty:
        raise RuntimeError(f'行为表中未找到 id={sub_id_int} 的数据。')

    behavior_sessions = df_sub['id_sess'].astype(str).drop_duplicates().tolist()

    print('\nAvailable behavior id_sess for this subject:')
    for sid in behavior_sessions:
        n = (df_sub['id_sess'].astype(str) == str(sid)).sum()
        print(f'  {sid}: {n} rows')

    return behavior_sessions


def get_behavior_blocks_by_sessions(df_pred, sub_id_int, behavior_sessions):
    df_sub = df_pred[df_pred['id'].astype(int) == int(sub_id_int)].copy()

    if df_sub.empty:
        raise RuntimeError(f'行为表中未找到 id={sub_id_int} 的数据。')

    available_sessions = df_sub['id_sess'].astype(str).unique().tolist()

    missing = [sid for sid in behavior_sessions if sid not in available_sessions]
    if len(missing) > 0:
        raise RuntimeError(f'行为表中缺少这些 id_sess: {missing}')

    blocks = []
    for sid in behavior_sessions:
        block = df_sub[df_sub['id_sess'].astype(str) == str(sid)].copy()
        blocks.append(block)

    beh_aligned = pd.concat(blocks, axis=0, ignore_index=True)
    return beh_aligned


def id_sess_to_eeg_sess(id_sess):
    sid = str(id_sess)

    m = re.search(r'(\d+)$', sid)
    if m is None:
        raise RuntimeError(f'无法从 id_sess={sid} 中提取 session 编号。')

    sess_num = int(m.group(1))
    return f'sess{sess_num}'


def behavior_sessions_to_eeg_sessions(behavior_sessions):
    return [id_sess_to_eeg_sess(sid) for sid in behavior_sessions]


def file_has_exact_session(fname, sess):
    base = os.path.basename(fname)
    pattern = rf'(?<!\d){re.escape(sess)}(?!\d)'
    return re.search(pattern, base) is not None


def select_files_by_sessions(files, keep_sessions, name):
    selected = []

    print(f'\n{name}:')
    print(f'  all files found: {len(files)}')
    for f in sorted(files):
        print(f'    ALL: {os.path.basename(f)}')

    for sess in keep_sessions:
        matches = [
            f for f in files
            if file_has_exact_session(f, sess)
        ]

        if len(matches) == 0:
            raise RuntimeError(f'{name}: 未找到 {sess} 对应的 EEG 文件。')

        if len(matches) > 1:
            print(f'\n{name}: {sess} matched multiple files:')
            for f in matches:
                print(f'  {os.path.basename(f)}')
            raise RuntimeError(f'{name}: {sess} 匹配到多个文件，请检查文件名。')

        selected.append(matches[0])

    print(f'  selected files: {len(selected)}')
    for f in selected:
        print(f'    USE: {os.path.basename(f)}')

    return selected


def read_concat_epochs(files, baseline=None, crop_tmin=None, crop_tmax=None, resample_hz=None):
    epochs_list = [mne.read_epochs(f, verbose=False) for f in files]

    if baseline is not None:
        for ep in epochs_list:
            ep.apply_baseline(baseline)

    epochs = mne.concatenate_epochs(epochs_list)

    if crop_tmin is not None or crop_tmax is not None:
        epochs.crop(crop_tmin, crop_tmax)

    if resample_hz is not None:
        epochs.resample(resample_hz)

    return epochs


def get_selection_array(epochs):
    if hasattr(epochs, 'selection') and epochs.selection is not None:
        return np.asarray(epochs.selection)
    return None


def strict_length_check(ep_t1_stim, ep_t1_resp, beh_aligned):
    lengths = {
        'Task1 Stim epochs': len(ep_t1_stim),
        'Task1 Response epochs': len(ep_t1_resp),
        'Behavior rows': len(beh_aligned),
    }

    print('\n===== LENGTH CHECK =====')
    for k, v in lengths.items():
        print(f'  {k}: {v}')

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            'EEG 和 behavior 长度不一致，停止分析。'
            '请检查 alignment report 和 EEG/behavior session 对应关系。'
        )


def print_label_counts(name, y):
    y = np.asarray(y)
    vals, counts = np.unique(y, return_counts=True)

    print(f'\n{name} label counts:')
    for v, c in zip(vals, counts):
        print(f'  {v}: {c}')

    if len(vals) < 2:
        raise RuntimeError(f'{name} 只有一个类别，无法计算 ROC-AUC。')

    return dict(zip(vals, counts))


def min_class_count(y):
    vals, counts = np.unique(np.asarray(y), return_counts=True)
    if len(vals) < 2:
        return 0
    return int(counts.min())


def make_classifier(random_state, y_for_cv, test_size=0.2):
    """
    根据当前标签数量动态设置 LogisticRegressionCV 的 inner cv。
    避免正确试次太少时 cv=5 报错。
    """
    min_count = min_class_count(y_for_cv)

    if min_count < 3:
        raise RuntimeError(
            f'当前标签中较少类别只有 {min_count} 个样本，'
            f'不适合做 LogisticRegressionCV。'
        )

    # 外层 train 大约是 1-test_size，所以 inner CV 不要超过训练集中少数类数量。
    approx_min_train_count = int(np.floor(min_count * (1.0 - test_size)))
    inner_cv = min(5, max(2, approx_min_train_count))

    print(f'Using LogisticRegressionCV inner cv = {inner_cv}')

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state,
            penalty='l1',
            Cs=np.logspace(-3, 3, 7),
            cv=inner_cv,
            scoring='roc_auc',
            max_iter=3000,
            n_jobs=1,
            verbose=0,
        ),
    )

    return clf


def save_alignment_report(
    save_path,
    subject,
    behavior_sessions,
    keep_sessions,
    task1_stim_files,
    task1_resp_files,
    ep_t1_stim,
    ep_t1_resp,
    beh_aligned,
):
    lines = []

    def log(x=''):
        x = str(x)
        print(x)
        lines.append(x)

    log('\n================ ALIGNMENT REPORT ================')
    log(f'subject = {subject}')
    log(f'behavior id_sess = {behavior_sessions}')
    log(f'EEG sessions kept = {keep_sessions}')

    log('\n[EEG files used] Task1 Stim:')
    for f in task1_stim_files:
        log(f'  {os.path.basename(f)}')

    log('\n[EEG files used] Task1 Response:')
    for f in task1_resp_files:
        log(f'  {os.path.basename(f)}')

    log('\n[Counts after reading]')
    log(f'  Task1 Stim epochs:     {len(ep_t1_stim)}')
    log(f'  Task1 Response epochs: {len(ep_t1_resp)}')
    log(f'  Behavior rows:         {len(beh_aligned)}')

    s_stim = get_selection_array(ep_t1_stim)
    s_resp = get_selection_array(ep_t1_resp)

    log('\n[Selection check: Task1 Stim vs Task1 Response]')
    if s_stim is not None and s_resp is not None:
        log(f'  first 20 Task1 Stim selection = {s_stim[:20]}')
        log(f'  first 20 Task1 Response selection = {s_resp[:20]}')
        log(f'  last 20 Task1 Stim selection = {s_stim[-20:]}')
        log(f'  last 20 Task1 Response selection = {s_resp[-20:]}')
        log('  注意：Task1 Stim selection 可能呈现 6 的间隔，这通常反映 marker 结构，不一定表示 trial 丢失。')
    else:
        log('  selection not available.')

    lengths = [
        len(ep_t1_stim),
        len(ep_t1_resp),
        len(beh_aligned),
    ]

    if len(set(lengths)) == 1:
        log('\nFINAL CHECK: OK. EEG and behavior lengths are equal.')
    else:
        log('\nFINAL CHECK: ERROR. EEG and behavior lengths are NOT equal.')

    log('================ END ALIGNMENT REPORT ================\n')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run_sliding_decoding(
    X,
    y,
    times,
    subject,
    analysis_name,
    figure_prefix,
    results_dir,
    figures_dir,
    test_size,
    n_splits,
    random_state,
):
    """
    运行一套 SlidingEstimator decoding：
    1. cross_val_multiscore
    2. basic plot
    3. cluster plot
    4. 返回结果字典
    """
    print('\n' + '=' * 70)
    print(f'Running decoding: {analysis_name}')
    print('=' * 70)

    print(f'X shape = {X.shape}')
    print(f'y length = {len(y)}')

    print_label_counts(f'{analysis_name} label', y)

    clf = make_classifier(
        random_state=random_state,
        y_for_cv=y,
        test_size=test_size,
    )

    slider = SlidingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=1,
        verbose=False,
    )

    cv = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    scores = cross_val_multiscore(
        slider,
        X,
        y,
        cv=cv,
        n_jobs=1,
        verbose=1,
    )

    scores_mean = scores.mean(axis=0)
    scores_sem = scores.std(axis=0) / np.sqrt(scores.shape[0])

    print(f'scores shape: {scores.shape}')

    basic_fig_path = os.path.join(
        figures_dir,
        f'{subject}_{figure_prefix}_sliding_basic.png',
    )

    utils.plot_time_decoding(
        times=times,
        scores=scores_mean,
        title=f'{analysis_name} ({subject})',
        save_path=basic_fig_path,
        chance_level=CHANCE_LEVEL,
    )

    T_obs, clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(
        scores - CHANCE_LEVEL,
        n_permutations=2000,
        tail=1,
        threshold=None,
        out_type='mask',
        seed=random_state,
        n_jobs=1,
        verbose=False,
    )

    cluster_fig_path = os.path.join(
        figures_dir,
        f'{subject}_{figure_prefix}_sliding_cluster.png',
    )

    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=scores_mean,
        scores_sem=scores_sem,
        permutation_result=(T_obs, clusters, cluster_p_vals, H0),
        title=f'{analysis_name} ({subject})',
        save_path=cluster_fig_path,
        chance_level=CHANCE_LEVEL,
        alpha=0.05,
    )

    print(f'Basic figure saved to: {os.path.abspath(basic_fig_path)}')
    print(f'Cluster figure saved to: {os.path.abspath(cluster_fig_path)}')

    return {
        'scores': scores,
        'scores_mean': scores_mean,
        'scores_sem': scores_sem,
        'T_obs': T_obs,
        'clusters': clusters,
        'cluster_p_vals': cluster_p_vals,
        'H0': H0,
        'basic_fig_path': basic_fig_path,
        'cluster_fig_path': cluster_fig_path,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # Basic settings
    # ============================================================
    subject = 'sub18'

    test_size = 0.2
    n_splits = 100
    random_state = 12345
    resample_hz = 100

    sub_id_int = int(re.search(r'\d+', subject).group())

    working_dir = os.path.join(
        '..',
        '..',
        '..',
        'data',
        'clean_EEG',
        subject,
    )

    behavior_path = os.path.join(
        '..',
        '..',
        '..',
        'data',
        'behavior_for_behavior',
        'data_wide_wmPred.txt',
    )

    figures_dir = os.path.join(
        '..',
        '..',
        '..',
        'figures',
        'decoding_idsess',
        subject,
    )

    results_dir = os.path.join(
        '..',
        '..',
        '..',
        'results',
        'decoding_idsess',
        subject,
    )

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f'\n===== START {subject} =====')
    print(f'EEG working_dir: {os.path.abspath(working_dir)}')
    print(f'Behavior file: {os.path.abspath(behavior_path)}')
    print(f'ABS_THRESHOLD = {ABS_THRESHOLD}')
    print(f'n_splits = {n_splits}, test_size = {test_size}, resample_hz = {resample_hz}')

    # ============================================================
    # Behavior first
    # ============================================================
    df_pred = pd.read_csv(behavior_path, sep=';')

    behavior_sessions = get_behavior_sessions_for_subject(
        df_pred=df_pred,
        sub_id_int=sub_id_int,
    )

    keep_sessions = behavior_sessions_to_eeg_sessions(behavior_sessions)

    beh_aligned = get_behavior_blocks_by_sessions(
        df_pred=df_pred,
        sub_id_int=sub_id_int,
        behavior_sessions=behavior_sessions,
    )

    print(f'\nBehavior sessions: {behavior_sessions}')
    print(f'EEG sessions to keep: {keep_sessions}')
    print(f'Behavior rows before EEG loading: {len(beh_aligned)}')

    # ============================================================
    # EEG files
    # ============================================================
    task1_resp_files_all = sorted(
        glob(os.path.join(
            working_dir,
            'clean_epochs_response1_ICA_*.fif',
        ))
    )

    task1_stim_files_all = sorted(
        glob(os.path.join(
            working_dir,
            'clean_epochs_task1_ICA_*.fif',
        ))
    )

    print(f'\ntask1_resp_files_all: {len(task1_resp_files_all)}')
    print(f'task1_stim_files_all: {len(task1_stim_files_all)}')

    if len(task1_resp_files_all) == 0 or len(task1_stim_files_all) == 0:
        raise RuntimeError('有 Task1 文件没找到，请检查 working_dir 和文件名模式。')

    task1_resp_files = select_files_by_sessions(
        task1_resp_files_all,
        keep_sessions,
        name='Task1 Response files',
    )

    task1_stim_files = select_files_by_sessions(
        task1_stim_files_all,
        keep_sessions,
        name='Task1 Stim files',
    )

    if not (
        len(task1_resp_files)
        == len(task1_stim_files)
        == len(behavior_sessions)
    ):
        raise RuntimeError(
            '过滤后的 EEG 文件数和 behavior session 数不一致。\n'
            f'  Task1 Resp files: {len(task1_resp_files)}\n'
            f'  Task1 Stim files: {len(task1_stim_files)}\n'
            f'  Behavior sessions: {len(behavior_sessions)}'
        )

    # ============================================================
    # Load EEG
    # ============================================================
    print('\nLoading Task1 Response EEG...')

    ep_t1_resp = read_concat_epochs(
        task1_resp_files,
        baseline=CS.baseline_response,
        crop_tmin=-0.5,
        crop_tmax=CS.tmax_response,
        resample_hz=resample_hz,
    )

    print('\nLoading Task1 Stim EEG...')

    ep_t1_stim = read_concat_epochs(
        task1_stim_files,
        baseline=None,
        crop_tmin=None,
        crop_tmax=None,
        resample_hz=None,
    )

    # ============================================================
    # Alignment report and strict check
    # ============================================================
    alignment_report_path = os.path.join(
        results_dir,
        f'{subject}_alignment_report_task1_response_confidence.txt',
    )

    save_alignment_report(
        save_path=alignment_report_path,
        subject=subject,
        behavior_sessions=behavior_sessions,
        keep_sessions=keep_sessions,
        task1_stim_files=task1_stim_files,
        task1_resp_files=task1_resp_files,
        ep_t1_stim=ep_t1_stim,
        ep_t1_resp=ep_t1_resp,
        beh_aligned=beh_aligned,
    )

    strict_length_check(
        ep_t1_stim=ep_t1_stim,
        ep_t1_resp=ep_t1_resp,
        beh_aligned=beh_aligned,
    )

    # ============================================================
    # Save behavior used
    # ============================================================
    beh_used_path = os.path.join(
        results_dir,
        f'{subject}_behavior_used_for_task1_response_confidence.csv',
    )

    beh_aligned.to_csv(
        beh_used_path,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'\nBehavior used saved to: {os.path.abspath(beh_used_path)}')

    # ============================================================
    # Labels
    # ============================================================
    y_acc = utils.compute_accuracy_from_epochs(
        ep_t1_stim,
        ep_t1_resp,
    )

    abs1 = pd.to_numeric(beh_aligned['abs1'], errors='coerce')
    y_conf = (abs1 > ABS_THRESHOLD).astype(int).values

    print_label_counts('Task1 accuracy label', y_acc)
    print_label_counts('Task1 confidence / abs1 label, all trials', y_conf)

    correct_mask = np.asarray(y_acc == 1)

    X_t1 = ep_t1_resp.get_data(copy=False)
    times_t1 = ep_t1_resp.times

    if len(X_t1) != len(y_conf):
        raise RuntimeError(
            f'Task1 Response 和 y_conf 样本数不一致: '
            f'X_t1={len(X_t1)}, y_conf={len(y_conf)}'
        )

    X_t1_correct = X_t1[correct_mask]
    y_conf_correct = y_conf[correct_mask]

    print(f'\nCorrect-trial data:')
    print(f'  correct trials: {correct_mask.sum()} / {len(correct_mask)}')
    print(f'  X_t1_correct shape: {X_t1_correct.shape}')
    print(f'  y_conf_correct length: {len(y_conf_correct)}')

    print_label_counts(
        'Task1 confidence / abs1 label, correct trials only',
        y_conf_correct,
    )

    print('\nConfidence label distribution by id_sess, all trials:')
    print(pd.crosstab(
        beh_aligned['id_sess'],
        y_conf,
        rownames=['id_sess'],
        colnames=['abs1_group'],
    ))

    print('\nConfidence label distribution by id_sess, correct trials only:')
    print(pd.crosstab(
        beh_aligned.loc[correct_mask, 'id_sess'],
        y_conf_correct,
        rownames=['id_sess'],
        colnames=['abs1_group_correct'],
    ))

    print('\nAccuracy label distribution by id_sess:')
    print(pd.crosstab(
        beh_aligned['id_sess'],
        y_acc,
        rownames=['id_sess'],
        colnames=['acc1_from_epochs'],
    ))

    # ============================================================
    # Decode 1: all trials confidence
    # ============================================================
    result_all = run_sliding_decoding(
        X=X_t1,
        y=y_conf,
        times=times_t1,
        subject=subject,
        analysis_name='Task1 Response: Confidence all trials',
        figure_prefix='Task1_Resp_Confidence_all_trials',
        results_dir=results_dir,
        figures_dir=figures_dir,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
    )

    # ============================================================
    # Decode 2: correct trials only confidence
    # ============================================================
    result_correct = run_sliding_decoding(
        X=X_t1_correct,
        y=y_conf_correct,
        times=times_t1,
        subject=subject,
        analysis_name='Task1 Response: Confidence correct trials only',
        figure_prefix='Task1_Resp_Confidence_correct_trials',
        results_dir=results_dir,
        figures_dir=figures_dir,
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
    )

    # ============================================================
    # Save values
    # ============================================================
    npz_path = os.path.join(
        results_dir,
        f'{subject}_Task1_Resp_Confidence_scores.npz',
    )

    np.savez(
        npz_path,

        scores_all_trials=result_all['scores'],
        scores_all_trials_mean=result_all['scores_mean'],
        scores_all_trials_sem=result_all['scores_sem'],
        T_obs_all_trials=result_all['T_obs'],
        cluster_p_vals_all_trials=result_all['cluster_p_vals'],

        scores_correct_trials=result_correct['scores'],
        scores_correct_trials_mean=result_correct['scores_mean'],
        scores_correct_trials_sem=result_correct['scores_sem'],
        T_obs_correct_trials=result_correct['T_obs'],
        cluster_p_vals_correct_trials=result_correct['cluster_p_vals'],

        times=times_t1,

        abs_threshold=ABS_THRESHOLD,
        y_conf=y_conf,
        y_acc=y_acc,
        correct_mask=correct_mask,
        y_conf_correct=y_conf_correct,

        behavior_sessions=np.array(behavior_sessions, dtype=str),
        eeg_sessions=np.array(keep_sessions, dtype=str),
    )

    print(f'\nNPZ saved to: {os.path.abspath(npz_path)}')
    print('DONE')

