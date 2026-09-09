"""
(实验一) DB2 健康人肌电数据集

功能：
- EMGCompletionDataset：PyTorch Dataset，返回 EMG 片段 + 元数据
- prepare_data_db2：DB2 专用预处理流水线（滤波→整流→包络→μ-law→归一化→分段）
- load_and_cache_data：支持缓存的数据加载
- load_db2_metadata：加载被试元数据（利手、年龄、性别）
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class EMGCompletionDataset(Dataset):
    """
    EMG 补全数据集（课程学习版本）

    mask 在训练时由 mask_generator 动态生成，不在此处生成。
    支持元数据注入（subject_id, side, age, gender）和 Repetition 标签。
    """

    def __init__(self, segments, subject_ids=None, repetitions=None, metadata_dict=None):
        self.segments = torch.FloatTensor(segments)
        self.subject_ids = subject_ids
        self.repetitions = repetitions
        self.metadata_dict = metadata_dict

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        emg_clean = self.segments[idx]
        
        side = 0
        age = 0.3
        gender = 1
        rep = 0
        
        if self.subject_ids is not None and self.metadata_dict is not None:
            sid = self.subject_ids[idx]
            if sid in self.metadata_dict:
                info = self.metadata_dict[sid]
                side = info['side']
                age = info['age']
                gender = info.get('gender', 1)
        
        if self.repetitions is not None:
            rep = int(self.repetitions[idx])
        
        return {
            'data': emg_clean,
            'side': torch.tensor(side, dtype=torch.long),
            'age': torch.tensor(age, dtype=torch.float),
            'gender': torch.tensor(gender, dtype=torch.long),
            'repetition': torch.tensor(rep, dtype=torch.long)
        }


def load_db2_metadata(csv_path):
    """
    加载 DB2 被试元数据
    Returns: {subject_id: {'side': int, 'age': float, 'gender': int}}
    """
    try:
        df = pd.read_csv(csv_path)
        metadata = {}
        
        for _, row in df.iterrows():
            sid = int(row['Subject'])
            is_right = 1 if row['Handedness'].strip().lower() == 'right' else 0
            norm_age = float(row['Age']) / 100.0
            gender_str = str(row.get('Gender', 'Male')).strip().lower()
            is_male = 1 if gender_str == 'male' else 0
            
            metadata[sid] = {
                'side': is_right,
                'age': norm_age,
                'gender': is_male
            }
            
        print(f"  Metadata loaded from CSV ({len(metadata)} subjects total)")
        return metadata
        
    except Exception as e:
        print(f"  Warning: cannot load metadata ({e}), using defaults")
        return None


def moving_average(signal_data, kernel_size):
    """移动平均滤波器，用于包络提取"""
    kernel = np.ones(kernel_size) / kernel_size
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0, arr=signal_data
    )


def prepare_data_db2(data_loader, subject_ids, config, exercises=[1]):
    """
    DB2 专用预处理流水线

    流程: Raw EMG (2000Hz) -> Bandpass -> Rectify -> Envelope -> μ-law -> Downsample -> Normalize -> Segment

    Returns:
        (segments, subject_ids_array, repetitions_array)
    """
    all_segments = []
    all_subject_ids = []
    all_repetitions = []
    downsample_factor = int(config['orig_fs'] / config['target_fs'])
    print(f"\n[DB2 Data Prep] {config['orig_fs']}Hz -> {config['target_fs']}Hz")

    for subject_id in subject_ids:
        try:
            data = data_loader.load_db2_subject(subject_id, exercises)

            emg = data['emg'] * 1000.0
            emg = data_loader.bandpass_filter(emg)
            emg = data_loader.notch_filter(emg)
            emg_rect = np.abs(emg)
            emg_env = moving_average(emg_rect, downsample_factor)
            emg_down = emg_env[::downsample_factor]

            labels_source = data.get('restimulus', data.get('stimulus'))
            labels_down = labels_source[::downsample_factor]
            reps_down = data['repetition'][::downsample_factor]

            # μ-law 变换（压缩长尾分布）
            mu = 255.0
            emg_max = emg_down.max()
            if emg_max > 0:
                emg_normalized_temp = emg_down / emg_max
                emg_compressed = np.log1p(mu * emg_normalized_temp) / np.log1p(mu)
                emg_down = emg_compressed * emg_max

            # 鲁棒 Min-Max 归一化（仅用 Rep 1,3,4,6 计算参数）
            calib_rep_mask = np.isin(reps_down, [1, 3, 4, 6])
            if calib_rep_mask.sum() > 0:
                emg_calib = emg_down[calib_rep_mask]
                q05 = np.percentile(emg_calib, 5)
                q99 = np.percentile(emg_calib, 99)
            else:
                q05 = np.percentile(emg_down, 5)
                q99 = np.percentile(emg_down, 99)
            
            scale = (q99 - q05) + 1e-8
            emg_norm = (emg_down - q05) / scale
            emg_norm = np.clip(emg_norm, 0.0, 1.0)

            # 分段（动作段100% + 过渡段限制10%）
            w_len = config['window_size']
            stride = config['stride']
            n_samples = emg_norm.shape[0]
            context_margin = 50
            transition_ratio_limit = 0.1
            
            active_segments = []
            active_reps = []
            transition_segments = []
            transition_reps = []
            
            for start in range(0, n_samples - w_len, stride):
                end = start + w_len
                segment_data = emg_norm[start:end, :]
                segment_label = labels_down[start:end]
                segment_rep_window = reps_down[start:end]
                center_idx = w_len // 2
                rep_id = int(segment_rep_window[center_idx])
                
                is_active = np.any(segment_label != 0)
                
                if is_active:
                    active_segments.append(segment_data)
                    active_reps.append(rep_id)
                else:
                    look_back = max(0, start - context_margin)
                    look_ahead = min(n_samples, end + context_margin)
                    context_label = labels_down[look_back:look_ahead]
                    if np.any(context_label != 0):
                        transition_segments.append(segment_data)
                        transition_reps.append(rep_id)
            
            n_active = len(active_segments)
            max_transition = int(n_active * transition_ratio_limit / (1 - transition_ratio_limit))
            
            if len(transition_segments) > max_transition:
                indices = np.random.choice(len(transition_segments), max_transition, replace=False)
                transition_segments = [transition_segments[i] for i in indices]
                transition_reps = [transition_reps[i] for i in indices]
            
            subject_segments = active_segments + transition_segments
            subject_reps = active_reps + transition_reps
            
            if len(subject_segments) > 0:
                segments = np.array(subject_segments)
                all_segments.append(segments)
                all_subject_ids.extend([subject_id] * len(segments))
                all_repetitions.extend(subject_reps)
                
                n_trans = len(transition_segments)
                trans_pct = n_trans / (n_active + n_trans) * 100 if (n_active + n_trans) > 0 else 0
                print(f"  S{subject_id:02d}: {len(segments)} segs (active:{n_active} + trans:{n_trans}, {trans_pct:.1f}%)")
        except Exception as exc:
            print(f"  S{subject_id}: FAILED - {exc}")
            continue

    if not all_segments:
        raise ValueError("No valid data")

    segments_concat = np.concatenate(all_segments)
    subject_ids_array = np.array(all_subject_ids, dtype=np.int32)
    repetitions_array = np.array(all_repetitions, dtype=np.int32)
    return segments_concat, subject_ids_array, repetitions_array


def load_and_cache_data(data_loader, subject_ids, config, cache_file, force_reload=False):
    """
    加载并缓存数据

    Returns:
        dict: {'data': ndarray, 'subject_ids': ndarray, 'repetitions': ndarray}
    """
    cache_path = Path(cache_file)
    
    if cache_path.exists() and not force_reload:
        cached_data = torch.load(cache_file, weights_only=False)
        
        # 校验缓存的 subject 列表是否与当前请求一致
        snapshot = cached_data.get('config_snapshot', {}) if isinstance(cached_data, dict) else {}
        cached_sids = snapshot.get('subject_ids', None)
        if cached_sids is not None and sorted(list(cached_sids)) != sorted(list(subject_ids)):
            print(f"\n[Processing] Cache subject mismatch, rebuilding...")
        else:
            if isinstance(cached_data, dict):
                data = cached_data.get('data')
                cached_subject_ids = cached_data.get('subject_ids')
                cached_repetitions = cached_data.get('repetitions')
                
                if cached_subject_ids is None:
                    cached_subject_ids = np.array([sid for sid in subject_ids for _ in range(len(data) // len(subject_ids))])
                if cached_repetitions is None:
                    cached_repetitions = np.zeros(len(data), dtype=np.int32)
            else:
                data = cached_data
                cached_subject_ids = None
                cached_repetitions = np.zeros(len(data), dtype=np.int32)
                
            print(f"\n[Cache] Loaded {data.shape if hasattr(data, 'shape') else len(data)} from: {Path(cache_file).name}")
            return {
                'data': data,
                'subject_ids': cached_subject_ids,
                'repetitions': cached_repetitions
            }
    
    print(f"\n[Processing] Building data from scratch...")
    data, subject_ids_array, repetitions_array = prepare_data_db2(data_loader, subject_ids, config)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'data': data,
        'subject_ids': subject_ids_array,
        'repetitions': repetitions_array,
        'config_snapshot': {
            'window_size': config['window_size'],
            'stride': config['stride'],
            'orig_fs': config['orig_fs'],
            'target_fs': config['target_fs'],
            'subject_ids': subject_ids
        }
    }, cache_file)
    print(f"  Cached to: {cache_file}")
    return {
        'data': data,
        'subject_ids': subject_ids_array,
        'repetitions': repetitions_array
    }
