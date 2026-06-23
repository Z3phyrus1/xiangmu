# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:06:27 2025
Updated: Single file processing with unique naming & Drop logs
"""

import os
from glob import glob
import mne
import numpy as np
import pandas as pd # 需要用到 pandas 保存日志
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils


if __name__ == "__main__":
    
    subject = 'sub16'
    working_dir = os.path.join('../../..', 'data', 'eeg', subject)
    print(f"被试文件夹: {working_dir}")
    
    # 获取所有 vhdr 文件
    working_data = np.sort(glob(os.path.join(working_dir, "*", '*vhdr')))
    print(f"找到 {len(working_data)} 个数据文件")
    
    results_dir = os.path.join('../..', 'data', 'clean_EEG', subject)
    figures_dir = os.path.join('../..', 'figures', 'preprocessing', subject)
    for f in [results_dir, figures_dir]:
        if not os.path.exists(f):
            os.makedirs(f)

    # 循环处理每个文件（不合并）
    for raw_file in working_data:
        print(f"\n{'='*60}")
        print(f"处理文件: {os.path.basename(raw_file)}")
        print(f"{'='*60}")
        
        # === 关键修改 1: 获取文件唯一标识 ===
        # 例如: zjy2_sess3-raw.vhdr -> zjy2_sess3
        file_stem = os.path.basename(raw_file).replace('.vhdr', '').replace('-raw', '')
        
        # 提取 session 数字
        session_num = utils.get_session_number(os.path.basename(raw_file))
        if session_num is None:
            print("无法识别session，跳过此文件")
            continue
        
        # 动态决定 event_id
        event_id_tasks = utils.determine_event_id(subject, session_num)
        event_id_task1 = event_id_tasks['task1']
        event_id_task2 = event_id_tasks['task2']
        event_id_belief = {'right': 2, 'left': 1}
        event_id_response = {'left': 1, 'right': 2} # 保持原样，不添加 17, 18
        
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
        
        events, _ = mne.events_from_annotations(raw)
        num_trials = (len(events) - 1) // CS.events_per_trial
        
        # ============================================================
        # Task1 预处理（锁时在刺激呈现）
        # ============================================================
        print("\n--- Task1 预处理 ---")
        epochs_task1 = mne.Epochs(raw,
            events=events, event_id=event_id_task1,
            tmin=CS.tmin_task1, tmax=CS.tmax_task1,
            baseline=None, picks=picks, preload=True, detrend=1)
        
        clean_epochs_task1 = utils.run_ica_cleaning(epochs_task1, picks, figures_dir, session_num, 'task1')
        
        # 保存 baseline 用于后续拼接（整个 [-1.2, 0] 区间）
        epoch_task1_baseline = epochs_task1.copy().crop(CS.baseline_task1[0], CS.baseline_task1[1])
        baseline_data = epoch_task1_baseline.get_data(copy=True)
        
        # 基线矫正和绘图
        clean_epochs_task1_copy = clean_epochs_task1.copy()
        clean_epochs_task1_copy.apply_baseline(CS.baseline_task1)
        utils.plot_and_save_joint(clean_epochs_task1_copy, event_id_task1, figures_dir, session_num, 'task1')
        
        # AutoReject
        ar_epochs_task1, reject_log_task1 = utils.run_autoreject(clean_epochs_task1, picks, figures_dir, session_num, 'task1')

        # 传入 ICA 前的 epochs_task1 以获取正确的 drop_log
        utils.save_preprocessing_log(subject, file_stem, 'task1', len(epochs_task1), clean_epochs_task1, ar_epochs_task1, results_dir)
        
        # === 关键修改 2: 保存文件名包含 file_stem ===
        utils.save_epochs(clean_epochs_task1, ar_epochs_task1, reject_log_task1, results_dir, f"task1_ICA_{subject}_{file_stem}")
        
        # ============================================================
        # Task2 预处理（锁时在刺激呈现）
        # ============================================================
        print("\n--- Task2 预处理 ---")
        epochs_task2 = mne.Epochs(raw,
            events=events, event_id=event_id_task2,
            tmin=CS.tmin_task2, tmax=CS.tmax_task2,
            baseline=None, picks=picks, preload=True, detrend=1)
        
        clean_epochs_task2 = utils.run_ica_cleaning(epochs_task2, picks, figures_dir, session_num, 'task2')
        
        # 拼接 baseline
        epochs_data_task2 = clean_epochs_task2.get_data(copy=True)
        epochs_data_task2 = utils.concatenate_baseline(epochs_data_task2, baseline_data)
        
        picked_conditions = [events[:, -1] == val for val in event_id_task2.values()]
        events_task2 = events[np.logical_or(*picked_conditions)]
        
        # 拼接后 tmin
        tmin_task2_concat = CS.tmin_task1 + CS.tmin_task2
        
        clean_epochs_task2 = mne.EpochsArray(
            epochs_data_task2, clean_epochs_task2.info,
            events=events_task2, tmin=tmin_task2_concat, event_id=event_id_task2)
        
        # 基线矫正和绘图
        clean_epochs_task2_copy = clean_epochs_task2.copy()
        clean_epochs_task2_copy.apply_baseline(CS.baseline_task2)
        utils.plot_and_save_joint(clean_epochs_task2_copy, event_id_task2, figures_dir, session_num, 'task2')
        
        # AutoReject
        ar_epochs_task2, reject_log_task2 = utils.run_autoreject(clean_epochs_task2_copy, picks, figures_dir, session_num, 'task2')
        
        # === 保存日志 & 文件名修改 ===
        utils.save_preprocessing_log(subject, file_stem, 'task2', len(epochs_task2), clean_epochs_task2, ar_epochs_task2, results_dir)
        utils.save_epochs(clean_epochs_task2, ar_epochs_task2, reject_log_task2, results_dir, f"task2_ICA_{subject}_{file_stem}")
        
        # ============================================================
        # Response1 预处理（锁时在反应）
        # ============================================================
        print("\n--- Response1 预处理 ---")
        response1_indices = utils.get_event_indices(events, CS.events_per_trial, 3, num_trials)
        response1_events = events[response1_indices]
        
        epochs_response1 = mne.Epochs(raw,
            events=response1_events, event_id=event_id_response,
            tmin=CS.tmin_response, tmax=CS.tmax_response,
            baseline=None, picks=picks, preload=True, detrend=1)
        
        clean_epochs_response1 = utils.run_ica_cleaning(epochs_response1, picks, figures_dir, session_num, 'response1')
        
        # 拼接 baseline
        epochs_data_response1 = clean_epochs_response1.get_data(copy=True)
        epochs_data_response1 = utils.concatenate_baseline(epochs_data_response1, baseline_data)
        
        # 拼接后 tmin
        tmin_response_concat = CS.tmin_task1 + CS.tmin_response
        
        clean_epochs_response1 = mne.EpochsArray(
            epochs_data_response1, clean_epochs_response1.info,
            events=response1_events, tmin=tmin_response_concat, event_id=event_id_response)
        
        # 基线矫正和绘图
        clean_epochs_response1_copy = clean_epochs_response1.copy()
        clean_epochs_response1_copy.apply_baseline(CS.baseline_response)
        utils.plot_and_save_joint(clean_epochs_response1_copy, event_id_response, figures_dir, session_num, 'response1')
        
        # AutoReject
        ar_epochs_response1, reject_log_response1 = utils.run_autoreject(clean_epochs_response1_copy, picks, figures_dir, session_num, 'response1')
        
        # === 保存日志 & 文件名修改 ===
        utils.save_preprocessing_log(subject, file_stem, 'response1', len(epochs_response1), clean_epochs_response1, ar_epochs_response1, results_dir)
        utils.save_epochs(clean_epochs_response1, ar_epochs_response1, reject_log_response1, results_dir, f"response1_ICA_{subject}_{file_stem}")
        
        print(f"\n✓ 文件 {file_stem} 全部预处理完成")