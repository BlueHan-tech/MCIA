"""
Train the signed 2 kHz DB2 healthy prior.
"""

import sys
from pathlib import Path

# 若服务器 locale 是 C/POSIX，sys.stdout.encoding 会被设成 ascii，
# 导致 print 里的中文、希腊字母 γ/β 被替换成 "?"。强制切到 UTF-8，
# 让日志里真实字符被正确写出（终端是否能渲染取决于 SSH 字体）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # 老版本 Python 或已是 UTF-8，忽略

# 必须在任何触发 matplotlib.pyplot 加载的 import 之前切到非交互后端，
# 否则服务器（无 X server / 无 GUI）下 matplotlib 会默认 Qt，报
# `QXcbConnection: Failed to initialize XRandr` 并导致图件保存失败。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import matplotlib
matplotlib.use('Agg')

import torch
from models.completion.mcia_core import derive_ch_mask_from_sample_mask, derive_patch_time_mask


def _assert_mask_contract(mask_binary_btc: torch.Tensor, patch_size: int, tag: str = '') -> None:
    """训练前若干 step 的运行期契约断言。"""
    assert torch.all((mask_binary_btc == 0) | (mask_binary_btc == 1)), \
        f"{tag} mask not strictly 0/1"
    B, T, C = mask_binary_btc.shape
    # 补丁对齐：每个补丁内要么全 0 要么全 1
    assert T % patch_size == 0, f"{tag} T={T} not divisible by patch_size={patch_size}"
    N = T // patch_size
    grouped = mask_binary_btc.permute(0, 2, 1).reshape(B, C, N, patch_size)
    lo = grouped.amin(dim=-1)
    hi = grouped.amax(dim=-1)
    mixed = int(((lo < 0.5) & (hi > 0.5)).sum().item())
    assert mixed == 0, f"{tag} {mixed} mixed patches detected (patch-align violated)"
    # 最小池化无信息损失：原始掩码与补丁级 mask_ratio 必须严格一致
    raw_ratio = 1.0 - float(mask_binary_btc.mean().item())
    patch_mask = derive_patch_time_mask(mask_binary_btc, patch_size)  # (B,C,N)
    patch_ratio = 1.0 - float(patch_mask.mean().item())
    assert abs(raw_ratio - patch_ratio) < 1e-6, \
        f"{tag} raw_ratio={raw_ratio:.6f} != patch_ratio={patch_ratio:.6f}"


def train_epoch_mcia(model, dataloader, optimizer, device, mask_gen, criterion,
                    difficulty=0.5, cfg_dropout_prob=0.1, scenario=None,
                    debug_assert_steps: int = 0, patch_size: int = 8,
                    use_personal_condition: bool = False):
    """Legacy 200 Hz training helper imported by the historical smoke script.

    参数：
        scenario: 强制场景（ScenarioMix 用；None 表示按权重随机逐样本选择）
        difficulty: 遗留课程学习保留入口
        debug_assert_steps: 训练前 N 步 跑运行期契约断言
    """
    model.train()
    total_loss = 0.0
    loss_parts_sum = {}
    step = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            emg_clean = batch['data'].to(device)
            side = batch.get('side', None)
            age = batch.get('age', None)
            gender = batch.get('gender', None)
            if side is not None: side = side.to(device)
            if age is not None: age = age.to(device)
            if gender is not None: gender = gender.to(device)
        else:
            emg_clean = batch.to(device)
            side = age = gender = None
        if not use_personal_condition:
            side = age = gender = None
        B, T, C = emg_clean.shape

        if scenario is not None and hasattr(mask_gen, '_dispatch'):
            mask_soft = mask_gen.generate_batch_masks(
                B, n_channels=C, time_steps=T, device=device, scenario=scenario,
            )
        elif hasattr(mask_gen, '_dispatch'):
            mask_soft = mask_gen.generate_batch_masks(
                B, n_channels=C, time_steps=T, device=device,
            )
        else:
            mask_soft = mask_gen.generate_batch_masks(
                batch_size=B, n_channels=C, time_steps=T,
                device=device, difficulty_level=difficulty,
            )
        mask_soft = mask_soft.transpose(1, 2)          # (B,T,C)
        mask_binary = (mask_soft > 0.5).float()        # (B,T,C) 严格 0/1

        if step < debug_assert_steps:
            _assert_mask_contract(mask_binary, patch_size, tag=f"[step{step}]")

        emg_masked = emg_clean * mask_binary           # 波形级硬零
        mask_1d = derive_ch_mask_from_sample_mask(mask_binary)  # (B,C) 1=通道仍有可见时刻

        drop_condition = torch.rand(1).item() < cfg_dropout_prob

        emg_pred = model(
            emg_masked, mask=mask_1d, x_masked=emg_masked,
            drop_condition=drop_condition,
            side=side, age=age, gender=gender,
            raw_time_mask=mask_binary,
        )

        if criterion is not None:
            loss, loss_parts = criterion(emg_pred, emg_clean, mask_binary)
            for key, value in loss_parts.items():
                loss_parts_sum[key] = loss_parts_sum.get(key, 0.0) + float(value)
        else:
            loss = torch.nn.functional.mse_loss(emg_pred, emg_clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

    train_epoch_mcia.last_loss_parts = {
        key: value / max(step, 1) for key, value in loss_parts_sum.items()
    }
    return total_loss / len(dataloader)


if __name__ == '__main__':
    import argparse
    from utils.waveform_pipeline import train_healthy

    parser = argparse.ArgumentParser(description='Train the 2 kHz DB2 healthy prior')
    parser.add_argument('--smoke', action='store_true')
    train_healthy(Path(__file__).resolve().parents[1], smoke=parser.parse_args().smoke)
