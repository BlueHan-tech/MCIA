"""
【实验三 - 上半场】使用 DB2 训练角度预测模型

流程：
1. 加载 DB2 健康人 EMG + 关节角度(glove) 配对数据
2. 训练 TCN/BiLSTM 回归模型：EMG → 关节角度
3. 保存最佳模型权重

说明：
该模型学习 "健康人 EMG → 关节角度" 的映射关系，
后续在实验四中用于评估 DB3 补全前后的预测精度差异。
"""

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from data.ninapro_loader import NinaProDataLoader
from data.dataset_kinematics import KinematicsDataset, prepare_kinematics_data
from models.prediction.kinematic_regressor import KinematicTCN
from utils.kinematic_target import KEY10_DIM, key10_target_metadata
from utils.metrics_kinematics import evaluate_kinematics


def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    flat = {
        'db2_path': cfg['paths']['db2'],
        'db3_path': cfg['paths']['db3'],
        'output_dir': cfg['paths']['output'],
        'checkpoints_dir': cfg['paths']['checkpoints'],
        'orig_fs': cfg['signal']['orig_fs'],
        'target_fs': cfg['signal']['target_fs'],
        'n_channels': cfg['signal']['n_channels'],
        'window_size': cfg['signal']['window_size'],
        'stride': cfg['signal']['stride'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    flat.update(cfg['exp3_regressor'])
    if int(flat['n_angle_channels']) != KEY10_DIM:
        raise ValueError(f"DB2 development regressor requires fixed Key10 output dimension {KEY10_DIM}.")
    return flat


def train_regressor_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        emg = batch['emg'].to(device)
        angle = batch['angle'].to(device)

        pred = model(emg)
        loss = criterion(pred, angle)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_regressor(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            emg = batch['emg'].to(device)
            angle = batch['angle'].to(device)

            pred = model(emg)
            loss = criterion(pred, angle)
            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(angle.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = evaluate_kinematics(targets, preds)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def main():
    CONFIG = load_config()
    device = CONFIG['device']

    output_dir = Path(CONFIG['output_dir']) / 'exp3_regressor_db2'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("[Exp3a] Train Kinematic Regressor on DB2")
    print("="*80)

    data_loader = NinaProDataLoader(CONFIG['db2_path'], CONFIG['db3_path'], fs=CONFIG['orig_fs'])

    # --- 加载数据 ---
    train_subjects = CONFIG['train_subjects']
    print(f"\nLoading kinematics data for {len(train_subjects)} DB2 subjects...")

    try:
        emg_segs, angle_segs, sids, _reps = prepare_kinematics_data(
            data_loader, train_subjects, CONFIG, exercises=[1], db='db2'
        )
    except Exception as e:
        print(f"FAILED to load kinematics data: {e}")
        print("Note: NinaPro DB2 requires 'glove' field in .mat files.")
        return

    print(f"\nTotal: {emg_segs.shape[0]} paired segments")
    print(f"  EMG: {emg_segs.shape}, Angle: {angle_segs.shape}")

    # --- 划分训练/验证集 (80/20) ---
    n_total = len(emg_segs)
    n_train = int(n_total * 0.8)
    indices = np.random.permutation(n_total)

    train_set = KinematicsDataset(emg_segs[indices[:n_train]], angle_segs[indices[:n_train]])
    val_set = KinematicsDataset(emg_segs[indices[n_train:]], angle_segs[indices[n_train:]])

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    # --- 初始化模型 ---
    model = KinematicTCN(
        n_emg_channels=CONFIG['n_channels'],
        n_angle_channels=CONFIG['n_angle_channels'],
        hidden_dim=CONFIG['hidden_dim'],
        n_layers=CONFIG['n_layers'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['model_dropout']
    ).to(device)

    print(f"\nModel: KinematicTCN Key10 | Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['lr_factor'],
        patience=CONFIG['lr_patience'], min_lr=1e-7
    )

    # --- 训练 ---
    best_val_loss = float('inf')
    no_improve = 0
    start_time = time.time()

    print("\nTraining...")
    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_regressor_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_regressor(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        scheduler.step(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'angle_target': key10_target_metadata(),
            }, output_dir / 'best_model.pth')
            improved = " *"
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} "
                  f"| R2: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f}{improved}")

        if no_improve >= CONFIG['patience']:
            print(f"\nEarly stop after {CONFIG['patience']} epochs without improvement")
            break

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min | Best val loss: {best_val_loss:.6f}")

    # 保存到 checkpoints
    import shutil
    ckpt_dir = Path(CONFIG['checkpoints_dir']) / 'exp3_regressor_db2'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / 'best_model.pth', ckpt_dir / 'best_model.pth')

    print(f"Model saved to: {ckpt_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
