# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:32:27 2026

@author: WANGLIANGFU
"""

import os, glob, numpy as np
from scipy.signal import resample

wd = os.path.join('..','..','..','results','decoding_idsess')
files = sorted(glob.glob(os.path.join(wd,'**','*_sliding_scores.npz'), recursive=True))

for fp in files:
    d = np.load(fp, allow_pickle=True)
    t1 = d['times_t1']; t2 = d['times_t2']

    # 只处理 400/360 -> 80/72
    if len(t1)==400 and len(t2)==360:
        out = {}
        for k in d.files:
            x = d[k]
            if isinstance(x, np.ndarray) and x.dtype==object and x.shape==():
                x = x.item()
            out[k] = x

        # 线性时间轴对齐到目标长度
        out['times_t1'] = np.linspace(t1[0], t1[-1], 80)
        out['times_t2'] = np.linspace(t2[0], t2[-1], 72)

        for k, n_new in [('scores_t1_acc',80),('scores_t1_conf',80),('scores_t2_acc',72),('scores_t2_conf',72)]:
            x = out.get(k, None)
            if x is None:
                continue
            x = np.asarray(x)
            if x.ndim==2:
                out[k] = resample(x, n_new, axis=1)
            elif x.ndim==1:
                out[k] = resample(x, n_new, axis=0)

        new_fp = fp.replace('_sliding_scores.npz', '_sliding_scores_rs100.npz')
        np.savez(new_fp, **out)
        print("saved:", new_fp)

    d.close()