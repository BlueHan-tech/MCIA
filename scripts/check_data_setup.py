"""Bounded real-data setup check; does not create a run or train full experiments."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import yaml
from scipy.io import whosmat
from data.ninapro_loader import NinaProDataLoader
from data.dataset_db2_emg import EMGCompletionDataset, prepare_data_db2, load_db2_metadata
from data.dataset_db3_emg import prepare_data_db3
from data.dataset_kinematics import prepare_kinematics_data, make_rep_split
from utils.paper_pipeline import build_mcia, build_mask_generator, build_structural_loss, train_mcia_epoch, complete_with_mask
from models.prediction.kinematic_regressor import KinematicTCN


def main():
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    flat = {**cfg['signal'], **cfg['exp1_mcia']}
    loader = NinaProDataLoader(cfg['paths']['db2'], cfg['paths']['db3'], fs=flat['orig_fs'])
    out = ROOT / 'outputs' / 'setup_validation'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    report = {'status': 'running', 'python': sys.executable, 'full_pipeline_run': False}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    required = []
    subjects = sorted(set(cfg['exp1_mcia']['train_subjects'] + cfg['exp1_mcia']['val_subjects'] + cfg['exp1_mcia']['test_subjects']))
    for sid in subjects:
        required.append(Path(loader.db2_path) / f'DB2_s{sid}' / f'S{sid}_E1_A1.mat')
    for section, key in [('exp2_transfer', 'db3_subjects'), ('exp3_regressor', 'db3_subjects'), ('exp4_gesture', 'subjects')]:
        for sid in cfg[section][key]:
            for ex in cfg[section]['exercises']:
                required.append(Path(loader.db3_path) / f's{sid}' / f'DB3_s{sid}' / f'S{sid}_E{ex}_A1.mat')
    required = sorted(set(required))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    for path in required:
        fields = {name: shape for name, shape, kind in whosmat(path)}
        assert {'emg', 'restimulus', 'rerepetition'} <= fields.keys(), path
        assert fields['emg'][1] == 12, (path, fields['emg'])
    metadata_path = Path(cfg['paths']['metadata_csv'])
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    metadata = load_db2_metadata(metadata_path)
    assert metadata and set(subjects) <= metadata.keys()
    report['required_mat_files'] = len(required)
    print(f'Validated {len(required)} configured MAT files and metadata', flush=True)
    modules = {}
    for index, filename in enumerate(['01_train_mcia_db2_healthy_prior.py', '02_finetune_mcia_db3_amputee.py', '03_generate_augmented_db3_semg.py', '04_eval_db3_angle_raw_vs_augmented.py', '05_eval_db3_gesture_raw_vs_augmented.py', 'run_all_experiments.py']):
        spec = importlib.util.spec_from_file_location(f'mcia_setup_stage_{index}', ROOT / 'scripts' / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[filename] = module
        print(f'Imported {filename}', flush=True)
    assert modules['run_all_experiments.py']._python_command(['probe.py'])[0] == sys.executable
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report['device'] = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    torch.set_num_threads(2)
    db2, _, reps2 = prepare_data_db2(loader, [subjects[0]], flat, exercises=[1])
    db3, _, meta3 = prepare_data_db3(loader, [cfg['exp3_regressor']['db3_subjects'][0]], flat, exercises=[1], return_metadata=True)
    train2 = np.flatnonzero(np.isin(reps2, (1, 3, 4)))
    train3, val3, test3 = make_rep_split(meta3['repetition'])
    model = build_mcia(flat, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=flat['learning_rate'])
    for name, values, indices, domain in [('db2', db2, train2, 0), ('db3', db3, train3, 1)]:
        assert len(indices) >= 2 and np.isfinite(values).all()
        loss = train_mcia_epoch(model, torch.utils.data.DataLoader(EMGCompletionDataset(values[indices[:2]]), batch_size=2), optimizer, device, build_mask_generator(flat), build_structural_loss(flat, device), scenario='s3', domain_id=domain)
        assert np.isfinite(loss), loss
        report[name] = {'windows': len(values), 'one_batch_loss': loss}
        print(f'{name}: one training batch OK, loss={loss:.6f}', flush=True)
    sid = cfg['exp3_regressor']['db3_subjects'][0]
    emg, angles, _, reps = prepare_kinematics_data(loader, [sid], flat, exercises=[1], db='db3')
    train, validation, test = make_rep_split(reps)
    x = torch.from_numpy(emg[train[:2]]).to(device)
    y = torch.from_numpy(angles[train[:2]]).to(device)
    mask = torch.ones_like(x)
    mask[:, :, :2] = 0
    model.eval()
    enhanced = complete_with_mask(model, x, mask, domain_id=1)
    assert torch.isfinite(enhanced).all() and torch.equal(enhanced[mask.bool()], x[mask.bool()])
    reg = cfg['exp3_regressor']
    tcn = KinematicTCN(hidden_dim=reg['hidden_dim'], n_layers=reg['n_layers'], kernel_size=reg['kernel_size'], dropout=reg['model_dropout']).to(device)
    optimizer_tcn = torch.optim.Adam(tcn.parameters(), lr=reg['learning_rate'])
    loss = torch.nn.functional.mse_loss(tcn(enhanced), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer_tcn.step()
    report['kinematics'] = {'windows': len(emg), 'split_counts': [len(train), len(validation), len(test)], 'one_batch_loss': float(loss)}
    gesture = modules['05_eval_db3_gesture_raw_vs_augmented.py']
    windows = gesture.load_db3_windows(loader, cfg['exp4_gesture']['subjects'][0], 1, flat, 0.8, 0.8)
    split = gesture.split_indices(windows.repetitions)
    report['gesture'] = {'windows': len(windows.emg), 'split_counts': [len(v) for v in split]}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(emg[train[0], :, 0], label='real DB3 input')
    ax.plot(enhanced[0, :, 0].cpu().numpy(), label='smoke model completion (untrained)')
    ax.legend()
    fig.savefig(out / 'pipeline_smoke.png')
    plt.close(fig)
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
