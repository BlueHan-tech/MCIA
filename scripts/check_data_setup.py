"""有界真实数据环境检查；不创建 run，也不训练完整实验。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np
from scipy.io import loadmat, whosmat

from data.ninapro_loader import NinaProDataLoader
from data.waveform import FS, REP_SPLITS, WAVEFORM_FORMAT, partition_segments, task_windows
from utils.paper_pipeline import load_yaml_config, save_json


def main():
    cfg = load_yaml_config(ROOT)
    loader = NinaProDataLoader(cfg['paths']['db2'], cfg['paths']['db3'], fs=FS)
    db2 = cfg['exp1_mcia']
    subjects = sorted(set(db2['train_subjects'] + db2['val_subjects'] + db2['test_subjects']))
    required = {(db, sid, ex)
                for db, ids, exercises in [
                    ('db2', subjects, cfg['waveform']['db2_exercises']),
                    ('db3', cfg['exp4_gesture']['subjects'], cfg['exp4_gesture']['exercises']),
                    ('db3', cfg['exp3_regressor']['db3_subjects'], cfg['exp3_regressor']['exercises']),
                ]
                for sid in ids for ex in exercises}
    gesture_counts = {}
    for db, subject, exercise in sorted(required):
        path = loader.exercise_path(db, subject, exercise)
        fields = {name: shape for name, shape, _ in whosmat(path)}
        count, channels = fields['emg']
        if (channels != 12 or np.prod(fields['restimulus']) != count
                or np.prod(fields['rerepetition']) != count):
            raise ValueError(f'Invalid raw time grid: {path}')
        if (db == 'db3' and subject in cfg['exp3_regressor']['db3_subjects']
                and exercise in cfg['exp3_regressor']['exercises'] and fields['glove'] != (count, 22)):
            raise ValueError(f'Invalid synchronized glove labels: {path}')
        if db != 'db3' or subject not in cfg['exp4_gesture']['subjects']:
            continue
        # Audit membership only. Do not read or score held-out EMG or model predictions.
        data = loadmat(path, variable_names=['restimulus', 'rerepetition'])
        labels, reps = data['restimulus'].ravel(), data['rerepetition'].ravel()
        windows = task_windows(labels, reps, partition_segments(reps), gesture=True)
        result = gesture_counts.setdefault(subject, np.zeros((3, 49), dtype=np.int64))
        for split, allowed in enumerate(REP_SPLITS):
            subset = windows.actions[np.isin(windows.repetitions, allowed)]
            if ((subset < 1) | (subset > 49)).any():
                raise ValueError(f'Out-of-range action: {path}')
            result[split] += np.bincount(subset, minlength=50)[1:]
    for subject, counts in gesture_counts.items():
        if not (counts > 0).all():
            raise ValueError(f'S{subject:02d} is missing gesture classes')
        print(f'S{subject:02d}: 49/49 classes, windows={counts.sum(axis=1).tolist()}', flush=True)
    report = ROOT / 'outputs' / 'setup_validation' / 'waveform_report.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, {'format': WAVEFORM_FORMAT, 'status': 'passed', 'required_mat_files': len(required),
                       'label_only_gesture_counts': {sid: counts.tolist() for sid, counts in gesture_counts.items()},
                       'split_order': ['train', 'validation', 'evaluation']})
    print(f'Validated {len(required)} MAT schemas and {len(gesture_counts)} gesture subjects: {report}', flush=True)


if __name__ == '__main__':
    main()
