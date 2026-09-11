"""配对 EMG 到 Key10 运动学数据的准备。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from data.dataset_db2_emg import moving_average
from utils.kinematic_target import KEY10_DIM, assert_key10_target, select_key10_angles
from utils.db3_quality_mask import db3_quality_mask

DEFAULT_TRAIN_REPS = (1, 3, 4)
DEFAULT_VAL_REPS = (6,)
DEFAULT_TEST_REPS = (2, 5)

class KinematicsDataset(Dataset):
    def __init__(self, emg_segments, angle_segments):
        if len(emg_segments) != len(angle_segments):
            raise ValueError(f"EMG ({len(emg_segments)}) and angle ({len(angle_segments)}) count mismatch")
        assert_key10_target(angle_segments, "KinematicsDataset angle segments")
        self.emg = torch.as_tensor(emg_segments, dtype=torch.float32)
        self.angle = torch.as_tensor(angle_segments, dtype=torch.float32)
    def __len__(self):
        return len(self.emg)
    def __getitem__(self, idx):
        return {"emg": self.emg[idx], "angle": self.angle[idx]}

def make_subject_split(n_items, train_ratio=0.6, val_ratio=0.2, seed=42):
    indices = np.arange(n_items)
    np.random.default_rng(seed).shuffle(indices)
    n_train, n_val = int(n_items * train_ratio), int(n_items * val_ratio)
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

def make_rep_split(repetitions, train_reps=DEFAULT_TRAIN_REPS, val_reps=DEFAULT_VAL_REPS,
                   test_reps=DEFAULT_TEST_REPS, val_ratio=None, seed=None):
    """冻结划分：训练 1/3/4，验证 6，测试 2/5。"""
    del val_ratio, seed
    reps = np.asarray(repetitions, dtype=np.int32)
    train_reps = tuple(rep for rep in train_reps if rep not in val_reps and rep not in test_reps)
    train = np.flatnonzero(np.isin(reps, train_reps)).astype(np.int64)
    val = np.flatnonzero(np.isin(reps, val_reps)).astype(np.int64)
    test = np.flatnonzero(np.isin(reps, test_reps)).astype(np.int64)
    if not len(train) or not len(val) or not len(test):
        raise ValueError(f"Fixed repetition split is empty: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test

def _subject_dir(data_loader, subject_id, db):
    if db == "db2":
        return Path(data_loader.db2_path) / f"DB2_s{subject_id}"
    if db == "db3":
        return Path(data_loader.db3_path) / f"s{subject_id}" / f"DB3_s{subject_id}"
    raise ValueError(f"Unsupported database: {db}")

def _load_exercise(data_loader, subject_id, exercise, db):
    path = _subject_dir(data_loader, subject_id, db) / f"S{subject_id}_E{exercise}_A1.mat"
    mat = sio.loadmat(path)
    if "glove" not in mat:
        raise ValueError(f"S{subject_id:02d} E{exercise}: no glove field")
    label_key = "restimulus" if "restimulus" in mat else "stimulus"
    repetition_key = "rerepetition" if "rerepetition" in mat else "repetition"
    emg = np.asarray(mat["emg"], dtype=np.float32)
    glove = np.asarray(mat["glove"], dtype=np.float32)
    labels = np.asarray(mat[label_key]).reshape(-1)
    repetitions = np.asarray(mat[repetition_key]).reshape(-1)
    if len({len(emg), len(glove), len(labels), len(repetitions)}) != 1 or glove.ndim != 2 or glove.shape[1] != 22:
        raise ValueError(f"S{subject_id:02d} E{exercise}: invalid paired EMG/glove shapes")
    return {"exercise": int(exercise), "emg": emg, "glove": glove, "labels": labels, "repetitions": repetitions}

def _preprocess_emg(data_loader, emg, factor):
    filtered = data_loader.bandpass_filter(emg * 1000.0)
    filtered = data_loader.notch_filter(filtered)
    return moving_average(np.abs(filtered), factor)[::factor].astype(np.float32)

def _window_rows(labels, repetitions, window_size, stride):
    """保留静息/动作转换，但窗口绝不跨越两个非零 repetition。"""
    rows = []
    for start in range(0, len(labels) - window_size + 1, stride):
        end = start + window_size
        nonzero_reps = np.unique(repetitions[start:end][repetitions[start:end] > 0])
        if len(nonzero_reps) == 1 and np.any(labels[start:end] != 0):
            rows.append((start, int(nonzero_reps[0])))
    return rows

def prepare_kinematics_data(data_loader, subject_ids, config, exercises=(1,), db="db2",
                            return_metadata=False):
    """构建 exercise 分离的窗口，EMG 和 glove 缩放仅由训练集拟合。"""
    factor = int(config["orig_fs"] / config["target_fs"])
    window_size, stride = int(config["window_size"]), int(config["stride"])
    all_emg, all_angle, all_subjects, all_reps, all_exercises, all_starts, all_masks = [], [], [], [], [], [], []
    print(f"\n[Kinematics Data Prep] {db.upper()}, exercises={list(exercises)}, {config['orig_fs']}Hz -> {config['target_fs']}Hz")
    for subject_id in subject_ids:
        try:
            prepared = []
            for exercise in exercises:
                raw = _load_exercise(data_loader, subject_id, int(exercise), db)
                quality_keep = (db3_quality_mask(raw["emg"], raw["labels"], raw["repetitions"], int(config["target_fs"]))[0]
                                if db == "db3" else np.ones_like(_preprocess_emg(data_loader, raw["emg"], factor), dtype=np.float32))
                emg = _preprocess_emg(data_loader, raw["emg"], factor)
                glove, labels, repetitions = raw["glove"][::factor], raw["labels"][::factor], raw["repetitions"][::factor]
                n = min(len(emg), len(glove), len(labels), len(repetitions))
                prepared.append({"exercise": raw["exercise"], "emg": emg[:n], "quality_mask": quality_keep[:n], "glove": glove[:n],
                                 "rows": _window_rows(labels[:n], repetitions[:n], window_size, stride)})
            train_emg = [item["emg"][start:start+window_size] for item in prepared for start, rep in item["rows"] if rep in DEFAULT_TRAIN_REPS]
            train_glove = [item["glove"][start:start+window_size] for item in prepared for start, rep in item["rows"] if rep in DEFAULT_TRAIN_REPS]
            if not train_emg:
                raise ValueError("no train-repetition paired windows")
            train_emg, train_glove = np.concatenate(train_emg), np.concatenate(train_glove)
            emg_max = float(train_emg.max())
            def compress(values):
                return values if emg_max <= 0 else np.log1p(255.0*values/emg_max)/np.log1p(255.0)*emg_max
            q05, q99 = np.percentile(compress(train_emg), [5, 99])
            glove_min = train_glove.min(0, keepdims=True)
            glove_scale = np.maximum(train_glove.max(0, keepdims=True)-glove_min, 1e-8)
            count = 0
            for item in prepared:
                emg_norm = np.clip((compress(item["emg"])-q05)/(q99-q05+1e-8), 0.0, 1.0)
                angle = select_key10_angles((item["glove"]-glove_min)/glove_scale)
                for start, rep in item["rows"]:
                    end = start + window_size
                    all_emg.append(emg_norm[start:end].astype(np.float32, copy=False))
                    all_angle.append(angle[start:end].astype(np.float32, copy=False))
                    all_subjects.append(int(subject_id)); all_reps.append(int(rep))
                    all_exercises.append(int(item["exercise"])); all_starts.append(int(start)); count += 1
                    all_masks.append(item["quality_mask"][start:end].astype(np.float32, copy=False))
            print(f"  S{subject_id:02d}: {count} paired Key10 ({KEY10_DIM}-D) windows")
        except Exception as exc:
            print(f"  S{subject_id:02d}: FAILED - {exc}")
    if not all_emg:
        raise ValueError("No valid kinematics data")
    values = (np.stack(all_emg), np.stack(all_angle), np.asarray(all_subjects, dtype=np.int32), np.asarray(all_reps, dtype=np.int32))
    if not return_metadata:
        return values
    return (*values, {"exercise": np.asarray(all_exercises, dtype=np.int16),
                      "start": np.asarray(all_starts, dtype=np.int64),
                      "quality_mask": np.stack(all_masks).astype(np.float32),
                      "normalization": "train_repetitions_1_3_4_q5_q99_only",
                      "window_policy": "exercise_separated_single_nonzero_repetition"})
