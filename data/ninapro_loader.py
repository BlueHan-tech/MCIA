"""
NinaPro DB2/DB3 数据加载与基础预处理

提供统一的数据加载接口：
- DB2: 40 名健康被试，12 通道 sEMG
- DB3: 11 名截肢患者，12 通道 sEMG
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class NinaProDataLoader:
    """NinaPro DB2/DB3 数据加载与预处理"""

    def __init__(self, db2_path, db3_path, fs=2000):
        self.db2_path = Path(db2_path)
        self.db3_path = Path(db3_path)
        self.fs = fs

    def load_db2_subject(self, subject_id, exercises=[1, 2, 3]):
        """加载 DB2 健康被试数据"""
        subject_dir = self.db2_path / f"DB2_s{subject_id}"
        emg_list, stimulus_list, repetition_list = [], [], []

        for ex in exercises:
            mat_file = subject_dir / f"S{subject_id}_E{ex}_A1.mat"
            if mat_file.exists():
                data = sio.loadmat(mat_file)
                emg_list.append(data['emg'])
                stimulus_list.append(data['restimulus'])
                repetition_list.append(data['rerepetition'])

        return {
            'emg': np.vstack(emg_list),
            'stimulus': np.vstack(stimulus_list).flatten(),
            'restimulus': np.vstack(stimulus_list).flatten(),
            'repetition': np.vstack(repetition_list).flatten(),
            'subject_id': subject_id,
            'n_channels': emg_list[0].shape[1]
        }

    def load_db3_subject(self, subject_id, exercises=[1, 2, 3]):
        """加载 DB3 截肢患者数据"""
        subject_dir = self.db3_path / f"s{subject_id}" / f"DB3_s{subject_id}"

        emg_list, stimulus_list, repetition_list = [], [], []

        for ex in exercises:
            mat_file = subject_dir / f"S{subject_id}_E{ex}_A1.mat"
            if mat_file.exists():
                data = sio.loadmat(mat_file)
                emg_list.append(data['emg'])
                stimulus_list.append(data['restimulus'])
                repetition_list.append(data['rerepetition'])

        return {
            'emg': np.vstack(emg_list),
            'stimulus': np.vstack(stimulus_list).flatten(),
            'restimulus': np.vstack(stimulus_list).flatten(),
            'repetition': np.vstack(repetition_list).flatten(),
            'subject_id': subject_id,
            'n_channels': emg_list[0].shape[1]
        }

    def bandpass_filter(self, emg, lowcut=20, highcut=450):
        """甯﹂€氭护娉?"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, emg, axis=0)
        return signal.sosfilt(sos, np.flip(filtered, axis=0), axis=0)[::-1]

    def notch_filter(self, emg, freq=50, Q=30):
        """闄锋尝婊ゆ尝锛堝幓闄ゅ伐棰戝共鎵帮級"""
        nyq = 0.5 * self.fs
        w0 = freq / nyq
        b, a = signal.iirnotch(w0, Q)
        sos = signal.tf2sos(b, a)
        filtered = signal.sosfilt(sos, emg, axis=0)
        return signal.sosfilt(sos, np.flip(filtered, axis=0), axis=0)[::-1]
    def preprocess_emg(self, emg):
        """EMG 标准预处理流程"""
        emg_filtered = self.bandpass_filter(emg)
        emg_filtered = self.notch_filter(emg_filtered)
        scaler = StandardScaler()
        emg_normalized = scaler.fit_transform(emg_filtered)
        return emg_normalized, scaler

    def segment_emg(self, emg, stimulus, repetition, window_size=200, stride=50):
        """
        将 EMG 信号分段

        Returns:
            segments: (n_segments, window_size, n_channels)
            labels: (n_segments,)
        """
        segments = []
        labels = []

        valid_indices = np.where(stimulus > 0)[0]

        for start_idx in range(0, len(valid_indices) - window_size, stride):
            idx = valid_indices[start_idx:start_idx + window_size]
            if len(idx) == window_size:
                seg = emg[idx, :]
                label = stimulus[idx[0]]
                segments.append(seg)
                labels.append(label)

        return np.array(segments), np.array(labels)


