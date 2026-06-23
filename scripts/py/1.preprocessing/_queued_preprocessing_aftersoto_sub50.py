  # -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:02:13 2026

@author: WANGLIANGFU
"""

import os
import matplotlib
matplotlib.use('Agg')
from glob import glob
import mne
import numpy as np
import pandas as pd
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils


if __name__ == "__main__":

    subject = 'sub50'
    working_dir = os.path.join('../../..', 'data', 'eeg', subject)
    print(f"被试文件夹: {working_dir}")

    # 获取所有 vhdr 文件
    working_data = np.sort(glob(os.path.join(working_dir, "*", '*vhdr')))
    print(f"找到 {len(working_data)} 个数据文件")

    results_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)
    figures_dir = os.path.join('../../..', 'figures', 'preprocessing_dual', subject)
    for f in [results_dir, figures_dir]:
        if not os.path.exists(f):
            os.makedirs(f)

    # 循环处理每个文件（不合并）
    for raw_file in working_data:
        print(f"\n{'=' * 60}")
        print(f"处理文件: {os.path.basename(raw_file)}")
        print(f"{'=' * 60}")

        # 文件唯一标识
        file_stem = os.path.basename(raw_file).replace('.vhdr', '').replace('-raw', '')
        final_marker = os.path.join(
            results_dir, f'reject_log_response2_ICA_{subject}_{file_stem}.npz'
        )
        if os.path.exists(final_marker):
            print(f'[SKIP] {file_stem} already finished: {final_marker}')
            continue

        # 提取 session 数字
        session_num = utils.get_session_number(os.path.basename(raw_file))
        if session_num is None:
            print("无法识别session，跳过此文件")
            continue

        # 动态决定 event_id
        event_id_tasks = utils.determine_event_id(subject, session_num)
        event_id_task1 = event_id_tasks['task1']
        event_id_task2 = event_id_tasks['task2']
        event_id_belief = {'right': 2, 'left': 1}  # 若后续不用可删除
        event_id_response = {'left': 1, 'right': 2}  # 保持原样，不添加 17, 18

        print(f"Session: {session_num}")
        print(f"  task1 Event ID: {event_id_task1}")
        print(f"  task2 Event ID: {event_id_task2}")

        # ============================================================
        # 读取和预处理原始数据
        # ============================================================
        raw = mne.io.read_raw_brainvision(raw_file, preload=True)
        raw.set_montage('standard_1020')
        raw.set_eeg_reference('average', projection=True)

        picks = mne.pick_types(raw.info, eeg=True)
        raw.notch_filter(np.arange(50, 251, 50), picks=picks, **CS.filter_args)
        raw.filter(CS.fmin, CS.fmax, picks=picks, **CS.filter_args)
        # manual_bads = ['AF7']
        # raw.info['bads'] = manual_bads
        # print(f"[BADS] 手动标记坏道: {raw.info['bads']}")
        # if len(raw.info['bads']) > 0:
        #     raw.interpolate_bads(reset_bads=True)
        #     print("[BADS] 已插值坏道")
        events, _ = mne.events_from_annotations(raw)
        num_trials = (len(events) - 1) // CS.events_per_trial

        # ============================================================
        # Task1 预处理（锁时在刺激呈现）
        # ============================================================
        print("\n--- Task1 预处理 ---")
        epochs_task1 = mne.Epochs(
            raw,
            events=events, event_id=event_id_task1,
            tmin=CS.tmin_task1, tmax=CS.tmax_task1,
            baseline=None, picks=picks, preload=True, detrend=1
        )

        clean_epochs_task1 = utils.run_ica_cleaning(
            epochs_task1, picks, figures_dir, session_num, 'task1'
        )

        # 保存 baseline 用于后续拼接（整个 [-1.2, 0] 区间）
        epoch_task1_baseline = epochs_task1.copy().crop(CS.baseline_task1[0], CS.baseline_task1[1])
        baseline_data = epoch_task1_baseline.get_data(copy=True)

        # 基线矫正和绘图（ICA后）
        clean_epochs_task1_copy = clean_epochs_task1.copy()
        clean_epochs_task1_copy.apply_baseline(CS.baseline_task1)
        utils.plot_and_save_joint(
            clean_epochs_task1_copy, event_id_task1, figures_dir, session_num, 'task1'
        )

        # AutoReject
        ar_epochs_task1, reject_log_task1 = utils.run_autoreject(
            clean_epochs_task1, picks, figures_dir, session_num, 'task1'
        )

        # AutoReject 后 ERP 可视化
        print("\n--- Task1 AutoReject 后 ERP 可视化 ---")
        ar_epochs_task1_baseline = ar_epochs_task1.copy()
        ar_epochs_task1_baseline.apply_baseline(CS.baseline_task1)
        utils.plot_post_autoreject_erp(
            ar_epochs_task1_baseline, event_id_task1,
            figures_dir, session_num, 'task1', subject, file_stem
        )

        # 保存日志 & 文件
        utils.save_preprocessing_log(
            subject, file_stem, 'task1',
            len(epochs_task1), clean_epochs_task1, ar_epochs_task1, results_dir
        )
        utils.save_epochs(
            clean_epochs_task1, ar_epochs_task1, reject_log_task1,
            results_dir, f"task1_ICA_{subject}_{file_stem}"
        )

        # ============================================================
        # Task2 预处理（锁时在刺激呈现）
        # ============================================================
        print("\n--- Task2 预处理 ---")
        epochs_task2 = mne.Epochs(
            raw,
            events=events, event_id=event_id_task2,
            tmin=CS.tmin_task2, tmax=CS.tmax_task2,
            baseline=None, picks=picks, preload=True, detrend=1
        )

        clean_epochs_task2 = utils.run_ica_cleaning(
            epochs_task2, picks, figures_dir, session_num, 'task2'
        )

        # 拼接 baseline
        epochs_data_task2 = clean_epochs_task2.get_data(copy=True)
        epochs_data_task2 = utils.concatenate_baseline(epochs_data_task2, baseline_data)

        picked_conditions = [events[:, -1] == val for val in event_id_task2.values()]
        events_task2 = events[np.logical_or(*picked_conditions)]

        # 拼接后 tmin
        tmin_task2_concat = CS.tmin_task1 + CS.tmin_task2

        clean_epochs_task2 = mne.EpochsArray(
            epochs_data_task2, clean_epochs_task2.info,
            events=events_task2, tmin=tmin_task2_concat, event_id=event_id_task2
        )

        # 基线矫正和绘图（ICA后）
        clean_epochs_task2_copy = clean_epochs_task2.copy()
        clean_epochs_task2_copy.apply_baseline(CS.baseline_task2)
        utils.plot_and_save_joint(
            clean_epochs_task2_copy, event_id_task2, figures_dir, session_num, 'task2'
        )

        # AutoReject（建议使用 baseline 后版本）
        ar_epochs_task2, reject_log_task2 = utils.run_autoreject(
            clean_epochs_task2_copy, picks, figures_dir, session_num, 'task2'
        )

        # AutoReject 后 ERP 可视化
        print("\n--- Task2 AutoReject 后 ERP 可视化 ---")
        ar_epochs_task2_baseline = ar_epochs_task2.copy()
        ar_epochs_task2_baseline.apply_baseline(CS.baseline_task2)
        utils.plot_post_autoreject_erp(
            ar_epochs_task2_baseline, event_id_task2,
            figures_dir, session_num, 'task2', subject, file_stem
        )

        # 保存日志 & 文件
        utils.save_preprocessing_log(
            subject, file_stem, 'task2',
            len(epochs_task2), clean_epochs_task2, ar_epochs_task2, results_dir
        )
        utils.save_epochs(
            clean_epochs_task2, ar_epochs_task2, reject_log_task2,
            results_dir, f"task2_ICA_{subject}_{file_stem}"
        )

        # ============================================================
        # Response1 预处理（锁时在反应）
        # ============================================================
        print("\n--- Response1 预处理 ---")
        response1_indices = utils.get_event_indices(events, CS.events_per_trial, 3, num_trials)
        response1_events = events[response1_indices]

        epochs_response1 = mne.Epochs(
            raw,
            events=response1_events, event_id=event_id_response,
            tmin=CS.tmin_response, tmax=CS.tmax_response,
            baseline=None, picks=picks, preload=True, detrend=1
        )

        clean_epochs_response1 = utils.run_ica_cleaning(
            epochs_response1, picks, figures_dir, session_num, 'response1'
        )

        # 拼接 baseline
        epochs_data_response1 = clean_epochs_response1.get_data(copy=True)
        epochs_data_response1 = utils.concatenate_baseline(epochs_data_response1, baseline_data)

        # 拼接后 tmin
        tmin_response_concat = CS.tmin_task1 + CS.tmin_response

        clean_epochs_response1 = mne.EpochsArray(
            epochs_data_response1, clean_epochs_response1.info,
            events=response1_events, tmin=tmin_response_concat, event_id=event_id_response
        )

        # 基线矫正和绘图（ICA后）
        clean_epochs_response1_copy = clean_epochs_response1.copy()
        clean_epochs_response1_copy.apply_baseline(CS.baseline_response)
        utils.plot_and_save_joint(
            clean_epochs_response1_copy, event_id_response, figures_dir, session_num, 'response1'
        )

        # AutoReject（建议使用 baseline 后版本）
        ar_epochs_response1, reject_log_response1 = utils.run_autoreject(
            clean_epochs_response1_copy, picks, figures_dir, session_num, 'response1'
        )

        # AutoReject 后 ERP 可视化
        print("\n--- Response1 AutoReject 后 ERP 可视化 ---")
        ar_epochs_response1_baseline = ar_epochs_response1.copy()
        ar_epochs_response1_baseline.apply_baseline(CS.baseline_response)
        utils.plot_post_autoreject_erp(
            ar_epochs_response1_baseline, event_id_response,
            figures_dir, session_num, 'response1', subject, file_stem
        )

        # 保存日志 & 文件
        utils.save_preprocessing_log(
            subject, file_stem, 'response1',
            len(epochs_response1), clean_epochs_response1, ar_epochs_response1, results_dir
        )
        utils.save_epochs(
            clean_epochs_response1, ar_epochs_response1, reject_log_response1,
            results_dir, f"response1_ICA_{subject}_{file_stem}"
        )
        # ============================================================
        # Response2 预处理（锁时在反应）
        # ============================================================
        print("\n--- Response2 预处理 ---")
        response2_indices = utils.get_event_indices(events, CS.events_per_trial, 6, num_trials)
        response2_events = events[response2_indices]

        epochs_response2 = mne.Epochs(
            raw,
            events=response2_events, event_id=event_id_response,
            tmin=CS.tmin_response, tmax=CS.tmax_response,
            baseline=None, picks=picks, preload=True, detrend=1
        )

        clean_epochs_response2 = utils.run_ica_cleaning(
            epochs_response2, picks, figures_dir, session_num, 'response2'
        )

        # 拼接 baseline
        epochs_data_response2 = clean_epochs_response2.get_data(copy=True)
        epochs_data_response2 = utils.concatenate_baseline(epochs_data_response2, baseline_data)

        # 拼接后 tmin
        tmin_response_concat = CS.tmin_task1 + CS.tmin_response

        clean_epochs_response2 = mne.EpochsArray(
            epochs_data_response2, clean_epochs_response2.info,
            events=response2_events, tmin=tmin_response_concat, event_id=event_id_response
        )

        # 基线矫正和绘图（ICA后）
        clean_epochs_response2_copy = clean_epochs_response2.copy()
        clean_epochs_response2_copy.apply_baseline(CS.baseline_response)
        utils.plot_and_save_joint(
            clean_epochs_response2_copy, event_id_response, figures_dir, session_num, 'response2'
        )

        # AutoReject（建议使用 baseline 后版本）
        ar_epochs_response2, reject_log_response2 = utils.run_autoreject(
            clean_epochs_response2_copy, picks, figures_dir, session_num, 'response2'
        )

        # AutoReject 后 ERP 可视化
        print("\n--- Response2 AutoReject 后 ERP 可视化 ---")
        ar_epochs_response2_baseline = ar_epochs_response2.copy()
        ar_epochs_response2_baseline.apply_baseline(CS.baseline_response)
        utils.plot_post_autoreject_erp(
            ar_epochs_response2_baseline, event_id_response,
            figures_dir, session_num, 'response2', subject, file_stem
        )

        # 保存日志 & 文件
        utils.save_preprocessing_log(
            subject, file_stem, 'response2',
            len(epochs_response2), clean_epochs_response2, ar_epochs_response2, results_dir
        )
        utils.save_epochs(
            clean_epochs_response2, ar_epochs_response2, reject_log_response2,
            results_dir, f"response2_ICA_{subject}_{file_stem}"
        )

        print(f"\n✓ 文件 {file_stem} 全部预处理完成")
        print(f"  - Task1: ICA前={len(epochs_task1)}, ICA后={len(clean_epochs_task1)}, AR后={len(ar_epochs_task1)}")
        print(f"  - Task2: ICA前={len(epochs_task2)}, ICA后={len(clean_epochs_task2)}, AR后={len(ar_epochs_task2)}")
        print(f"  - Response1: ICA前={len(epochs_response1)}, ICA后={len(clean_epochs_response1)}, AR后={len(ar_epochs_response1)}")
        print(f"  - Response2: ICA前={len(epochs_response2)}, ICA后={len(clean_epochs_response2)}, AR后={len(ar_epochs_response2)}")