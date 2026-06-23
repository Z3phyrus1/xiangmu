# -*- coding: utf-8 -*-
"""
Cross-task temporal generalization decoding
Fixed EEG-behavior alignment version, no trial variable

功能：
1. 对单个被试做 cross-task temporal generalization
2. Train: Task1 Response
3. Test:  Task2 Stimulus
4. 解码三类结果：
   A. Accuracy generalization:
      - train label: Task1 accuracy
      - test label:  Task2 acc2

   B. Confidence / abs generalization, all trials:
      - train label: abs1 > 1
      - test label:  abs2 > 1

   C. Confidence / abs generalization, correct trials only:
      - train samples: Task1 correct trials only
      - train label:   abs1 > 1
      - test samples:  Task2 correct trials only
      - test label:    abs2 > 1

5. EEG-behavior 对齐：
   - 根据 behavior id_sess 自动推断 EEG sess
   - 只读取匹配 session 的 EEG 文件
   - 不再使用 n_min 静默截断
   - 长度不一致直接停止
"""

import os
import re
from glob import glob
from shutil import copyfile

import numpy as np
import pandas as pd
import mne

from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


# ============================================================
# Helper functions
# ============================================================
def get_behavior_sessions_for_subject(df_pred, sub_id_int):
    """
    获取当前被试在 behavior 表中的所有 id_sess。
    顺序按照 behavior 表中出现顺序保留。
    """
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
    """
    按指定 id_sess 顺序拼接 behavior。
    """
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
    """
    将 behavior 的 id_sess 映射到 EEG 文件名中的 sessX。

    示例：
        54fl03 -> sess3
        54fl04 -> sess4
        18cl03 -> sess3
        18cl04 -> sess4
    """
    sid = str(id_sess)

    m = re.search(r'(\d+)$', sid)
    if m is None:
        raise RuntimeError(f'无法从 id_sess={sid} 中提取 session 编号。')

    sess_num = int(m.group(1))
    return f'sess{sess_num}'


def behavior_sessions_to_eeg_sessions(behavior_sessions):
    """
    将多个 behavior id_sess 转成 EEG sessX 列表。
    """
    return [id_sess_to_eeg_sess(sid) for sid in behavior_sessions]


def file_has_exact_session(fname, sess):
    """
    判断文件名是否包含精确 sess 编号。
    避免 sess1 错误匹配 sess10。
    """
    base = os.path.basename(fname)
    pattern = rf'(?<!\d){re.escape(sess)}(?!\d)'
    return re.search(pattern, base) is not None


def select_files_by_sessions(files, keep_sessions, name):
    """
    按 keep_sessions 顺序选择 EEG 文件。

    每个 sess 必须正好对应一个文件。
    没有文件或匹配多个文件都会报错。
    """
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
    """
    读取、拼接 epochs，并可选 baseline / crop / resample。
    """
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


def strict_length_check(ep_t1_stim, ep_t1_resp, ep_t2_stim, beh_aligned):
    """
    严格检查 EEG 和 behavior 长度。
    不再自动 n_min 截断。
    """
    lengths = {
        'Task1 Stim epochs': len(ep_t1_stim),
        'Task1 Response epochs': len(ep_t1_resp),
        'Task2 Stim epochs': len(ep_t2_stim),
        'Behavior rows': len(beh_aligned),
    }

    print('\nLength check:')
    for k, v in lengths.items():
        print(f'  {k}: {v}')

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            'EEG 和 behavior 长度不一致，停止分析。'
            '请检查 alignment report 和 EEG/behavior session 对应关系。'
        )


def save_alignment_report(
    save_path,
    subject,
    behavior_sessions,
    keep_sessions,
    task1_stim_files,
    task1_resp_files,
    task2_stim_files,
    ep_t1_stim,
    ep_t1_resp,
    ep_t2_stim,
    beh_aligned,
):
    """
    保存本次实际使用的 EEG 文件和 behavior block。
    """
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

    log('\n[EEG files used] Task2 Stimulus:')
    for f in task2_stim_files:
        log(f'  {os.path.basename(f)}')

    log('\n[Counts after reading]')
    log(f'  Task1 Stim epochs:     {len(ep_t1_stim)}')
    log(f'  Task1 Response epochs: {len(ep_t1_resp)}')
    log(f'  Task2 Stim epochs:     {len(ep_t2_stim)}')
    log(f'  Behavior rows:         {len(beh_aligned)}')

    s_resp = get_selection_array(ep_t1_resp)
    s_t2 = get_selection_array(ep_t2_stim)

    log('\n[Selection check: Task1 Response vs Task2 Stimulus]')
    if s_resp is not None and s_t2 is not None:
        same = np.array_equal(s_resp, s_t2)
        log(f'  selections identical? {same}')
        log(f'  first 20 Task1 Response selection = {s_resp[:20]}')
        log(f'  first 20 Task2 Stimulus selection = {s_t2[:20]}')
        log(f'  last 20 Task1 Response selection = {s_resp[-20:]}')
        log(f'  last 20 Task2 Stimulus selection = {s_t2[-20:]}')
    else:
        log('  selection not available.')

    lengths = [
        len(ep_t1_stim),
        len(ep_t1_resp),
        len(ep_t2_stim),
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


def print_label_counts(name, y):
    """
    打印标签数量，并检查是否有两个类别。
    """
    y = np.asarray(y)
    vals, counts = np.unique(y, return_counts=True)

    print(f'\n{name} label counts:')
    for v, c in zip(vals, counts):
        print(f'  {v}: {c}')

    if len(vals) < 2:
        raise RuntimeError(f'{name} 只有一个类别，无法计算 ROC-AUC。')


def check_binary_labels_for_auc(name, y):
    """
    检查某个标签是否至少包含两个类别。
    AUC 要求 y_true 至少有 0 和 1 两类。
    """
    y = np.asarray(y)
    vals, counts = np.unique(y, return_counts=True)

    print(f'\n{name} label counts:')
    for v, c in zip(vals, counts):
        print(f'  {v}: {c}')

    if len(vals) < 2:
        raise RuntimeError(
            f'{name} 只有一个类别，无法计算 ROC-AUC。'
        )


def make_classifier():
    """
    构建分类器。
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=3000,
        ),
    )
    return clf


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 1) Basic settings
    # ============================================================
    subject = 'sub18'
    sub_id_int = int(re.search(r'\d+', subject).group())

    working_dir = os.path.join(
        '../../..',
        'data',
        'clean_EEG',
        subject,
    )

    behavior_path = os.path.join(
        '../../..',
        'data',
        'behavior_for_behavior',
        'data_wide_wmPred.txt',
    )

    results_dir = os.path.join(
        '../../..',
        'results',
        'cross_task_generalization',
        subject,
    )

    figures_dir = os.path.join(
        '../../..',
        'figures',
        'cross_task_generalization',
        subject,
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f'Subject: {subject} (ID={sub_id_int})')
    print(f'EEG working_dir: {os.path.abspath(working_dir)}')
    print(f'Behavior file: {os.path.abspath(behavior_path)}')
    print(f'ABS_THRESHOLD = {ABS_THRESHOLD}')

    # ============================================================
    # 2) Behavior first
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
    # 3) EEG files
    # ============================================================
    task1_stim_files_all = sorted(
        glob(os.path.join(
            working_dir,
            'clean_epochs_task1_ICA_*.fif',
        ))
    )

    task1_resp_files_all = sorted(
        glob(os.path.join(
            working_dir,
            'clean_epochs_response1_ICA_*.fif',
        ))
    )

    task2_stim_files_all = sorted(
        glob(os.path.join(
            working_dir,
            'clean_epochs_task2_ICA_*.fif',
        ))
    )

    if len(task1_stim_files_all) == 0:
        raise FileNotFoundError('No Task1 Stim files found.')

    if len(task1_resp_files_all) == 0:
        raise FileNotFoundError('No Task1 Response files found.')

    if len(task2_stim_files_all) == 0:
        raise FileNotFoundError('No Task2 Stim files found.')

    task1_stim_files = select_files_by_sessions(
        task1_stim_files_all,
        keep_sessions,
        name='Task1 Stim files',
    )

    task1_resp_files = select_files_by_sessions(
        task1_resp_files_all,
        keep_sessions,
        name='Task1 Response files',
    )

    task2_stim_files = select_files_by_sessions(
        task2_stim_files_all,
        keep_sessions,
        name='Task2 Stimulus files',
    )

    if not (
        len(task1_stim_files)
        == len(task1_resp_files)
        == len(task2_stim_files)
        == len(behavior_sessions)
    ):
        raise RuntimeError(
            '过滤后的 EEG 文件数和 behavior session 数不一致。\n'
            f'  Task1 Stim files: {len(task1_stim_files)}\n'
            f'  Task1 Resp files: {len(task1_resp_files)}\n'
            f'  Task2 Stim files: {len(task2_stim_files)}\n'
            f'  Behavior sessions: {len(behavior_sessions)}'
        )

    # ============================================================
    # 4) Load EEG and preprocess
    # ============================================================
    print('\nLoading EEG...')

    # Task1 Stim：用于 compute_accuracy_from_epochs
    ep_t1_stim = read_concat_epochs(
        task1_stim_files,
        baseline=None,
        crop_tmin=None,
        crop_tmax=None,
        resample_hz=None,
    )

    # Task1 Response：train
    ep_t1_resp = read_concat_epochs(
        task1_resp_files,
        baseline=CS.baseline_response,
        crop_tmin=-0.5,
        crop_tmax=CS.tmax_response,
        resample_hz=500,
    )

    # Task2 Stimulus：test
    ep_t2_stim = read_concat_epochs(
        task2_stim_files,
        baseline=CS.baseline_task2,
        crop_tmin=CS.tmin_task2,
        crop_tmax=CS.tmax_task2,
        resample_hz=500,
    )

    # ============================================================
    # 5) Alignment report and strict check
    # ============================================================
    alignment_report_path = os.path.join(
        results_dir,
        f'{subject}_alignment_report_cross_task.txt',
    )

    save_alignment_report(
        save_path=alignment_report_path,
        subject=subject,
        behavior_sessions=behavior_sessions,
        keep_sessions=keep_sessions,
        task1_stim_files=task1_stim_files,
        task1_resp_files=task1_resp_files,
        task2_stim_files=task2_stim_files,
        ep_t1_stim=ep_t1_stim,
        ep_t1_resp=ep_t1_resp,
        ep_t2_stim=ep_t2_stim,
        beh_aligned=beh_aligned,
    )

    strict_length_check(
        ep_t1_stim=ep_t1_stim,
        ep_t1_resp=ep_t1_resp,
        ep_t2_stim=ep_t2_stim,
        beh_aligned=beh_aligned,
    )

    print(f'\nAligned samples: {len(beh_aligned)}')

    # 保存最终用于分析的 behavior
    beh_used_path = os.path.join(
        results_dir,
        f'{subject}_behavior_used_for_cross_task_generalization.csv',
    )

    beh_aligned.to_csv(
        beh_used_path,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'Behavior used for decoding saved to: {os.path.abspath(beh_used_path)}')

    # ============================================================
    # 6) Labels
    # ============================================================
    # Accuracy labels
    y_train_acc = utils.compute_accuracy_from_epochs(
        ep_t1_stim,
        ep_t1_resp,
    )

    y_test_acc = beh_aligned['acc2'].astype(int).values

    # Confidence / abs labels
    # 高信心试次定义：
    #   Task1: abs1 > 1
    #   Task2: abs2 > 1
    abs1 = pd.to_numeric(beh_aligned['abs1'], errors='coerce')
    abs2 = pd.to_numeric(beh_aligned['abs2'], errors='coerce')

    y_train_conf = (abs1 > ABS_THRESHOLD).astype(int).values
    y_test_conf = (abs2 > ABS_THRESHOLD).astype(int).values

    # Correct-trial masks
    # Train side: Task1 correct trials
    # Test side: Task2 correct trials
    correct_train_mask = np.asarray(y_train_acc == 1)
    correct_test_mask = np.asarray(y_test_acc == 1)

    # Confidence labels within correct trials only
    y_train_conf_correct = y_train_conf[correct_train_mask]
    y_test_conf_correct = y_test_conf[correct_test_mask]

    print_label_counts('Train accuracy label', y_train_acc)
    print_label_counts('Test accuracy label', y_test_acc)
    print_label_counts('Train confidence / abs1 label, all trials', y_train_conf)
    print_label_counts('Test confidence / abs2 label, all trials', y_test_conf)

    check_binary_labels_for_auc(
        'Train confidence / abs1 label, correct Task1 trials only',
        y_train_conf_correct,
    )

    check_binary_labels_for_auc(
        'Test confidence / abs2 label, correct Task2 trials only',
        y_test_conf_correct,
    )

    print(f'\nCorrect-trial counts:')
    print(f'  Task1 correct train trials: {correct_train_mask.sum()} / {len(correct_train_mask)}')
    print(f'  Task2 correct test trials:  {correct_test_mask.sum()} / {len(correct_test_mask)}')

    # 检查 session 内标签分布，排查 session confound
    print('\nTest accuracy label distribution by id_sess:')
    print(pd.crosstab(
        beh_aligned['id_sess'],
        y_test_acc,
        rownames=['id_sess'],
        colnames=['acc2'],
    ))

    print('\nTrain confidence label distribution by id_sess, all trials:')
    print(pd.crosstab(
        beh_aligned['id_sess'],
        y_train_conf,
        rownames=['id_sess'],
        colnames=['abs1_group'],
    ))

    print('\nTest confidence label distribution by id_sess, all trials:')
    print(pd.crosstab(
        beh_aligned['id_sess'],
        y_test_conf,
        rownames=['id_sess'],
        colnames=['abs2_group'],
    ))

    print('\nTrain confidence label distribution by id_sess, correct Task1 trials only:')
    print(pd.crosstab(
        beh_aligned.loc[correct_train_mask, 'id_sess'],
        y_train_conf_correct,
        rownames=['id_sess'],
        colnames=['abs1_group_correct'],
    ))

    print('\nTest confidence label distribution by id_sess, correct Task2 trials only:')
    print(pd.crosstab(
        beh_aligned.loc[correct_test_mask, 'id_sess'],
        y_test_conf_correct,
        rownames=['id_sess'],
        colnames=['abs2_group_correct'],
    ))

    # ============================================================
    # 7) Data arrays
    # ============================================================
    X_train = ep_t1_resp.get_data(copy=False)
    X_test = ep_t2_stim.get_data(copy=False)

    # Correct-trial data arrays
    X_train_correct = X_train[correct_train_mask]
    X_test_correct = X_test[correct_test_mask]

    times_train = ep_t1_resp.times
    times_test = ep_t2_stim.times

    print(f'\nData shape check:')
    print(f'  X_train all trials:      {X_train.shape}')
    print(f'  X_test all trials:       {X_test.shape}')
    print(f'  X_train correct trials:  {X_train_correct.shape}')
    print(f'  X_test correct trials:   {X_test_correct.shape}')

    # ============================================================
    # 8) Decode: Cross-task temporal generalization
    # ============================================================
    clf = make_classifier()

    # ----------------------------
    # Accuracy generalization
    # ----------------------------
    print('\nRunning Gen: Acc (T1Resp -> T2Stim)...')

    gen_acc = GeneralizingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=True,
    )

    gen_acc.fit(X_train, y_train_acc)
    scores_acc = gen_acc.score(X_test, y_test_acc)

    acc_fig_path = os.path.join(
        figures_dir,
        f'{subject}_gen_acc_cross_task.png',
    )

    utils.plot_temporal_generalization(
        scores=scores_acc,
        times_train=times_train,
        times_test=times_test,
        title=f'Acc Gen ({subject})',
        save_path=acc_fig_path,
        vmin=0.4,
        vmax=0.6,
    )

    # ----------------------------
    # Confidence / abs generalization, all trials
    # ----------------------------
    print('\nRunning Gen: Conf/Abs all trials (T1Resp -> T2Stim)...')

    gen_conf = GeneralizingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=True,
    )

    gen_conf.fit(X_train, y_train_conf)
    scores_conf = gen_conf.score(X_test, y_test_conf)

    conf_fig_path = os.path.join(
        figures_dir,
        f'{subject}_gen_conf_abs_cross_task.png',
    )

    utils.plot_temporal_generalization(
        scores=scores_conf,
        times_train=times_train,
        times_test=times_test,
        title=f'Conf/Abs Gen all trials ({subject})',
        save_path=conf_fig_path,
        vmin=0.4,
        vmax=0.6,
    )

    # ----------------------------
    # Confidence / abs generalization, correct trials only
    # ----------------------------
    print('\nRunning Gen: Conf/Abs correct trials only (T1Resp correct -> T2Stim correct)...')

    gen_conf_correct = GeneralizingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=True,
    )

    gen_conf_correct.fit(
        X_train_correct,
        y_train_conf_correct,
    )

    scores_conf_correct = gen_conf_correct.score(
        X_test_correct,
        y_test_conf_correct,
    )

    conf_correct_fig_path = os.path.join(
        figures_dir,
        f'{subject}_gen_conf_abs_correct_trials_cross_task.png',
    )

    utils.plot_temporal_generalization(
        scores=scores_conf_correct,
        times_train=times_train,
        times_test=times_test,
        title=f'Conf/Abs Gen correct trials only ({subject})',
        save_path=conf_correct_fig_path,
        vmin=0.4,
        vmax=0.6,
    )

    # ============================================================
    # 9) Save
    # ============================================================
    save_dict = {
        'scores_gen_acc': scores_acc,
        'scores_gen_conf': scores_conf,
        'scores_gen_conf_correct_trials': scores_conf_correct,

        'times_train': times_train,
        'times_test': times_test,

        'y_train_acc': y_train_acc,
        'y_test_acc': y_test_acc,

        'y_train_conf': y_train_conf,
        'y_test_conf': y_test_conf,

        'correct_train_mask': correct_train_mask,
        'correct_test_mask': correct_test_mask,

        'y_train_conf_correct': y_train_conf_correct,
        'y_test_conf_correct': y_test_conf_correct,

        'abs_threshold': ABS_THRESHOLD,
        'behavior_sessions': np.array(behavior_sessions, dtype=str),
        'eeg_sessions': np.array(keep_sessions, dtype=str),
    }

    npz_path = os.path.join(
        results_dir,
        f'{subject}_generalization_results.npz',
    )

    np.savez(npz_path, **save_dict)

    print(f'\nAccuracy figure saved to: {os.path.abspath(acc_fig_path)}')
    print(f'Confidence figure saved to: {os.path.abspath(conf_fig_path)}')
    print(f'Correct-trial confidence figure saved to: {os.path.abspath(conf_correct_fig_path)}')
    print(f'NPZ saved to: {os.path.abspath(npz_path)}')
    print('Done.')