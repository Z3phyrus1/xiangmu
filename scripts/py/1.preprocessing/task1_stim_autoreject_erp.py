# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:02:49 2025

@author: WANGLIANGFU
"""



import os
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
    
    subject = 'sub15'
    working_dir = os.path.join('../..', 'data', 'eeg', subject)
    print(f"被试文件夹: {working_dir}")
    
    # 获取所有 vhdr 文件
    working_data = np.sort(glob(os.path.join(working_dir, "*", '*vhdr')))
    print(f"找到 {len(working_data)} 个数据文件")
    
    results_dir = os.path.join('../..', 'data', 'clean_EEG', subject)
    figures_dir = os.path.join('../..', 'figures', 'preprocessing', subject)
    for f in [results_dir, figures_dir]:
        if not os.path.exists(f):
            os.makedirs(f)

    # 循环处理每个文件
    for raw_file in working_data:
        print(f"\n{'='*60}")
        print(f"处理文件: {os.path.basename(raw_file)}")
        print(f"{'='*60}")
        
        # 获取文件唯一标识
        file_stem = os.path.basename(raw_file).replace('.vhdr', '').replace('-raw', '')
        
        # 提取 session 数字
        session_num = utils.get_session_number(os.path.basename(raw_file))
        if session_num is None:
            print("无法识别session，跳过此文件")
            continue
        
        # 动态决定 event_id
        event_id_tasks = utils.determine_event_id(subject, session_num)
        event_id_task1 = event_id_tasks['task1']
        
        print(f"Session: {session_num}")
        print(f"  task1 Event ID: {event_id_task1}")
        
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
        
        # ============================================================
        # Task1 预处理（锁时在刺激呈现）
        # ============================================================
        print("\n--- Task1 预处理 ---")
        epochs_task1 = mne.Epochs(raw,
            events=events, event_id=event_id_task1,
            tmin=CS.tmin_task1, tmax=CS.tmax_task1,
            baseline=None, picks=picks, preload=True, detrend=1)
        
        print(f"  提取到 {len(epochs_task1)} 个试次")
        
        # ============================================================
        # 新增：ICA前绘制原始ERP图
        # ============================================================
        # print("\n--- ICA前ERP可视化 ---")
        # pre_ica_dir = utils.plot_pre_ica_erp(epochs_task1, event_id_task1, figures_dir, 
        #                                session_num, 'task1', subject, file_stem)
        
        # ============================================================
        # ICA处理
        # ============================================================
        print("\n--- 进行ICA处理 ---")
        clean_epochs_task1 = utils.run_ica_cleaning(epochs_task1, picks, figures_dir, session_num, 'task1')
        
        print(f"  ICA处理后剩余 {len(clean_epochs_task1)} 个试次")
        
        # 保存 baseline 数据（如果需要）
        epoch_task1_baseline = epochs_task1.copy().crop(CS.baseline_task1[0], CS.baseline_task1[1])
        baseline_data = epoch_task1_baseline.get_data(copy=True)
        
        # 基线矫正
        clean_epochs_task1_copy = clean_epochs_task1.copy()
        clean_epochs_task1_copy.apply_baseline(CS.baseline_task1)
        
        # 绘图
        utils.plot_and_save_joint(clean_epochs_task1_copy, event_id_task1, figures_dir, session_num, 'task1')
        
        # AutoReject处理
        ar_epochs_task1, reject_log_task1 = utils.run_autoreject(clean_epochs_task1, picks, figures_dir, session_num, 'task1')
        
        
        # === 新增：绘制 AutoReject 后的 ERP 图 ===
        print("\n--- AutoReject 后 ERP 可视化 ---")
        ar_epochs_task1_baseline = ar_epochs_task1.copy()
        ar_epochs_task1_baseline.apply_baseline(CS.baseline_task1)
        utils.plot_post_autoreject_erp(ar_epochs_task1_baseline, event_id_task1, 
                                       figures_dir, session_num, 'task1', 
                                       subject, file_stem)

        # 保存剔除日志
        utils.save_preprocessing_log(subject, file_stem, 'task1', len(epochs_task1), clean_epochs_task1, ar_epochs_task1, results_dir)
        
        # 保存处理后的epochs数据
        utils.save_epochs(clean_epochs_task1, ar_epochs_task1, reject_log_task1, results_dir, f"task1_ICA_{subject}_{file_stem}")
        
        print(f"\n✓ 文件 {file_stem} Task1 stimulus 预处理完成")
        print(f"  - ICA前试次: {len(epochs_task1)}")
        print(f"  - ICA后试次: {len(clean_epochs_task1)}")
        print(f"  - AutoReject后试次: {len(ar_epochs_task1)}")