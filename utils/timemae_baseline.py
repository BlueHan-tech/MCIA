"""
TimeMAE Completion Baseline
============================================================
自包含实现（不依赖 TimeMAE-main 目录），用于与 MCIA 模型做 baseline 对比。

基于论文：TimeMAE: Self-Supervised Representations of Time Series with Decoupled
         Masked Autoencoders (https://arxiv.org/abs/2303.00320)

关键改动（相对于原始 TimeMAE）：
  - 增加 recon_head: Linear(d_model → wave_length × C)，支持信号级重建
  - forward_completion(x, mask)：给定显式 mask 做推断，输出补全信号
  - 训练增加 recon_loss (MSE on signal space) 与原始 align+token 损失联合优化
  - 不包含分类 head（predict_head），不包含下游微调
"""

from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple

from scipy.signal import welch
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# 基础模块（复现自 TimeMAE-main/model/layers.py，保持一致）
# ──────────────────────────────────────────────────────────────────────────────

class _PosEmb(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe.weight.unsqueeze(0).expand(x.size(0), -1, -1)


class _MHA(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q, k, v = [l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out(x)


class _SubLayer(nn.Module):
    def __init__(self, size: int, enable_res: bool = True, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.drop = nn.Dropout(dropout)
        self.enable = enable_res
        if enable_res:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        if isinstance(x, list):
            return self.norm(x[1] + self.drop(self.a * sublayer(x)))
        scale = self.a if self.enable else 1.0
        return self.norm(x + self.drop(scale * sublayer(x)))


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x):
        return self.net(x)


class _TRMBlock(nn.Module):
    def __init__(self, d: int, h: int, d_ffn: int, enable_res: bool = True, drop: float = 0.1):
        super().__init__()
        self.attn = _MHA(h, d, drop)
        self.ffn = _FFN(d, d_ffn, drop)
        self.sc1 = _SubLayer(d, enable_res, drop)
        self.sc2 = _SubLayer(d, enable_res, drop)

    def forward(self, x, mask=None):
        x = self.sc1(x, lambda _x: self.attn(_x, _x, _x, mask))
        x = self.sc2(x, self.ffn)
        return x


class _CrossBlock(nn.Module):
    def __init__(self, d: int, h: int, d_ffn: int, enable_res: bool = True, drop: float = 0.1):
        super().__init__()
        self.attn = _MHA(h, d, drop)
        self.ffn = _FFN(d, d_ffn, drop)
        self.sc1 = _SubLayer(d, enable_res, drop)
        self.sc2 = _SubLayer(d, enable_res, drop)

    def forward(self, rep_vis, rep_mask, mask=None):
        x = [rep_vis, rep_mask]
        x = self.sc1(x, lambda _x: self.attn(_x[1], _x[0], _x[0], mask))
        x = self.sc2(x, self.ffn)
        return x


class _Encoder(nn.Module):
    def __init__(self, d: int, h: int, layers: int, drop: float, enable_res: bool):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TRMBlock(d, h, 4 * d, enable_res, drop) for _ in range(layers)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class _Regressor(nn.Module):
    def __init__(self, d: int, h: int, d_ffn: int, enable_res: bool, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CrossBlock(d, h, d_ffn, enable_res) for _ in range(layers)])

    def forward(self, rep_vis, rep_mask):
        for blk in self.layers:
            rep_mask = blk(rep_vis, rep_mask)
        return rep_mask


# ──────────────────────────────────────────────────────────────────────────────
# TimeMAE 补全模块
# ──────────────────────────────────────────────────────────────────────────────

class TimeMAECompletion(nn.Module):
    """TimeMAE + 信号重建头，用于 EMG 补全 Baseline。

    data_shape : (T, C)，T=200，C=12
    wave_length: 时间 patch 大小，将 T 分为 T/wave_length 个 patch
    """

    def __init__(
        self,
        data_shape: Tuple[int, int],
        wave_length: int = 8,
        d_model: int = 64,
        attn_heads: int = 4,
        layers: int = 4,
        reg_layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = 192,
        mask_ratio: float = 0.6,
        momentum: float = 0.99,
    ):
        super().__init__()
        T, C = data_shape
        self.wave_length = wave_length
        self.C = C
        self.T = T
        # 保证 T 能被 wave_length 整除
        self.T_padded = T + (wave_length - T % wave_length) % wave_length
        self.n_patches = self.T_padded // wave_length
        self.d_model = d_model
        self.mask_len = int(mask_ratio * self.n_patches)
        self.momentum = momentum

        self.position = _PosEmb(self.n_patches, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.input_proj = nn.Conv1d(C, d_model, kernel_size=wave_length, stride=wave_length)

        self.encoder = _Encoder(d_model, attn_heads, layers, dropout, True)
        self.momentum_encoder = _Encoder(d_model, attn_heads, layers, dropout, True)

        # Tokenizer（离散 token 预测，原版预训练目标之一）
        self.tok_center = nn.Linear(d_model, vocab_size)

        self.reg = _Regressor(d_model, attn_heads, 4 * d_model, True, reg_layers)

        # 信号重建头：d_model → wave_length × C
        self.recon_head = nn.Linear(d_model, wave_length * C)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for a, b in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                b.data = a.data

    def momentum_update(self):
        with torch.no_grad():
            for a, b in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                b.data = self.momentum * b.data + (1 - self.momentum) * a.data

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T < self.T_padded:
            x = torch.cat([x, x.new_zeros(B, self.T_padded - T, C)], dim=1)
        return x

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T_padded, C) → (B, N, d_model)"""
        return self.input_proj(x.transpose(1, 2)).transpose(1, 2).contiguous()

    # ── 预训练前向（随机 mask）──────────────────────────────────────────────

    def pretrain_forward(self, x: torch.Tensor):
        """原始 TimeMAE 随机 mask 预训练 + 信号重建损失。

        Returns
        -------
        [rep_mask, rep_mask_pred] : 用于 align_loss (MSE in embedding space)
        [token_pred_prob, tokens] : 用于 reconstruct_loss (CE on discrete tokens)
        recon_signal              : (B, n_mask, wave_length, C) 重建的信号片段
        signal_gt                 : (B, n_mask, wave_length, C) 对应的真值
        """
        x = self._pad(x)
        patches = self._to_patches(x)                               # (B, N, D)
        pos = self.position(patches)                                # (B, N, D)

        # 对离散目标做 Gumbel-softmax 分词
        tok_logits = self.tok_center(patches.reshape(-1, self.d_model))
        tokens = F.gumbel_softmax(tok_logits).max(-1)[1].view(patches.shape[0], -1)

        patches_pos = patches + pos
        mask_tokens = self.mask_token.view(1, 1, -1).expand_as(patches_pos) + pos

        idx = list(range(self.n_patches))
        random.shuffle(idx)
        v_idx = sorted(idx[: -self.mask_len])
        m_idx = sorted(idx[-self.mask_len:])

        visible = patches_pos[:, v_idx, :]
        mask_gt_emb = patches[:, m_idx, :]     # 动量编码器目标
        tokens_m = tokens[:, m_idx]
        mask_tok_inp = mask_tokens[:, m_idx, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            rep_mask = self.momentum_encoder(mask_gt_emb)
        rep_mask_pred = self.reg(rep_visible, mask_tok_inp)
        token_pred_prob = self.tok_center(rep_mask_pred)

        # 信号重建
        recon = self.recon_head(rep_mask_pred)                      # (B, n_mask, WL*C)
        B = x.shape[0]
        WL, C = self.wave_length, self.C
        recon_signal = recon.view(B, len(m_idx), WL, C)

        # 对应真值（从原始信号取对应 patch）
        gt_patches = []
        for pi in m_idx:
            t0, t1 = pi * WL, (pi + 1) * WL
            gt_patches.append(x[:, t0:t1, :])                      # (B, WL, C)
        signal_gt = torch.stack(gt_patches, dim=1)                  # (B, n_mask, WL, C)

        return (
            [rep_mask, rep_mask_pred],
            [token_pred_prob, tokens_m],
            recon_signal,
            signal_gt,
        )

    # ── 推断：给定显式 mask 做补全 ─────────────────────────────────────────

    def forward_completion(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, T, C)  原始信号（缺失区值任意）
            mask : (B, T, C)  1=已知，0=缺失

        Returns:
            (B, T, C) 补全结果，已知区使用真值
        """
        B, T, C = x.shape
        WL, N = self.wave_length, self.n_patches

        x_pad = self._pad(x)
        mask_pad = torch.ones(B, self.T_padded, C, device=mask.device)
        mask_pad[:, :T, :] = mask

        patches = self._to_patches(x_pad)       # (B, N, D)
        pos = self.position(patches)            # (B, N, D)
        patches_pos = patches + pos

        # 逐 patch 判断是否缺失：均值 < 0.5 即为缺失 patch
        patch_mask = mask_pad.reshape(B, N, WL, C).mean(dim=[2, 3])   # (B, N)
        # 批内取多数：超过半数样本该 patch 为缺失，则视为缺失
        is_masked = patch_mask.float().mean(0) < 0.5                   # (N,)

        v_idx = (~is_masked).nonzero(as_tuple=True)[0]
        m_idx = is_masked.nonzero(as_tuple=True)[0]

        if len(v_idx) == 0:
            # 无已知区：退化为全零
            return torch.zeros(B, T, C, device=x.device)
        if len(m_idx) == 0:
            # 无缺失区：直接返回输入
            return x

        rep_visible = self.encoder(patches_pos[:, v_idx, :])           # (B, n_vis, D)

        mask_tok = self.mask_token.view(1, 1, -1).expand(B, len(m_idx), -1)
        mask_tok = mask_tok + pos[:, m_idx, :]
        rep_masked = self.reg(rep_visible, mask_tok)                    # (B, n_mask, D)

        # 信号重建
        recon = self.recon_head(rep_masked).view(B, len(m_idx), WL, C)

        # 组装输出：先用重建填满，再将已知区还原为真值
        out = x_pad.clone()
        for i, pi in enumerate(m_idx):
            t0, t1 = int(pi) * WL, int(pi + 1) * WL
            out[:, t0:t1, :] = recon[:, i, :, :]

        out = out[:, :T, :]                     # 去掉 padding
        out = out * (1 - mask.float()) + x * mask.float()
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 训练函数
# ──────────────────────────────────────────────────────────────────────────────

def train_timemae_baseline(
    train_dataloader,
    device,
    data_shape: Tuple[int, int] = (200, 12),
    wave_length: int = 8,
    d_model: int = 64,
    attn_heads: int = 4,
    layers: int = 4,
    reg_layers: int = 2,
    dropout: float = 0.1,
    vocab_size: int = 192,
    mask_ratio: float = 0.6,
    momentum: float = 0.99,
    epochs: int = 30,
    lr: float = 1e-3,
    alpha: float = 5.0,     # embedding align loss 权重
    beta: float = 1.0,      # token prediction loss 权重
    gamma: float = 2.0,     # signal reconstruction loss 权重
    save_path: Optional[Path] = None,
) -> TimeMAECompletion:
    """在训练集上预训练 TimeMAECompletion。

    Returns
    -------
    训练完成的 TimeMAECompletion 模型（已置于 device）。
    """
    model = TimeMAECompletion(
        data_shape=data_shape,
        wave_length=wave_length, d_model=d_model, attn_heads=attn_heads,
        layers=layers, reg_layers=reg_layers, dropout=dropout,
        vocab_size=vocab_size, mask_ratio=mask_ratio, momentum=momentum,
    ).to(device)

    model.copy_weight()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.2)
    mse_loss = nn.MSELoss()

    print(f"\n[TimeMAE] Pre-training {epochs} epochs | "
          f"wave_length={wave_length} d_model={d_model} layers={layers}")

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for batch in train_dataloader:
            x = batch['data'].to(device) if isinstance(batch, dict) else batch.to(device)
            optimizer.zero_grad()

            (rep_mask, rep_mask_pred), (tok_pred, tok_gt), recon_sig, sig_gt = \
                model.pretrain_forward(x)

            loss_align = mse_loss(rep_mask_pred, rep_mask.detach())
            loss_tok = ce_loss(
                tok_pred.view(-1, tok_pred.shape[-1]), tok_gt.view(-1)
            )
            loss_recon = mse_loss(recon_sig, sig_gt)
            loss = alpha * loss_align + beta * loss_tok + gamma * loss_recon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.momentum_update()

            total += loss.item()
            n_batches += 1

        avg = total / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f}")

        if avg < best_loss and save_path is not None:
            best_loss = avg
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    if save_path is not None and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"  [TimeMAE] Loaded best ckpt from {save_path}")

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 评估函数（与 evaluate_subject 同格式）
# ──────────────────────────────────────────────────────────────────────────────

def _rms_env(s, win=16):
    return np.sqrt(np.convolve(s ** 2, np.ones(win) / win, 'same') + 1e-12)


def _mdf(s, fs=200):
    f, psd = welch(s, fs=fs, nperseg=min(64, len(s)))
    cum = np.cumsum(psd)
    return float(f[min(np.searchsorted(cum, cum[-1] / 2), len(f) - 1)])


def _safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    r, _ = stats.pearsonr(a, b)
    return float(r) if not np.isnan(r) else float('nan')


def evaluate_timemae(
    model: TimeMAECompletion,
    dataloader,
    mask_gen,
    device,
    scenario: Optional[str] = None,
    difficulty: float = 1.0,
) -> Dict[str, float]:
    """评估 TimeMAE 补全质量，返回与 evaluate_subject 完全相同格式的指标字典。"""
    from utils.evaluation import _make_batch_mask, _masked_region_metrics

    _fs, _win = 200, 16
    model.eval()

    mse_list, mae_list, corr_list = [], [], []
    region_keys = ['mse_masked', 'mse_known', 'mse_whole',
                   'mae_masked', 'mae_known', 'mae_whole',
                   'corr_masked', 'corr_masked_chmiss', 'corr_masked_partial',
                   'corr_known', 'corr_whole']
    racc = {k: [] for k in region_keys}
    env_corr_list, mdf_err_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['data'].to(device) if isinstance(batch, dict) else batch.to(device)
            B, T, C = x.shape

            mask_soft = _make_batch_mask(mask_gen, B, C, T, device,
                                         scenario=scenario, difficulty=difficulty)
            mask = (mask_soft.transpose(1, 2) > 0.5).float()          # (B,T,C)

            x_masked = x * mask
            pred = model.forward_completion(x_masked, mask)            # (B,T,C)
            completed = pred * (1 - mask) + x * mask

            pred_np = completed.cpu().numpy()
            true_np = x.cpu().numpy()
            mask_np = mask.cpu().numpy()

            mse_list.append(float(np.mean((pred_np - true_np) ** 2)))
            mae_list.append(float(np.mean(np.abs(pred_np - true_np))))

            for b in range(B):
                sc = [_safe_corr(pred_np[b, :, c], true_np[b, :, c]) for c in range(C)]
                sc = [r for r in sc if not np.isnan(r)]
                if sc:
                    corr_list.append(float(np.mean(sc)))

            # 分区指标（与 evaluate_subject 同口径）
            rm = _masked_region_metrics(pred_np, true_np, mask_np)
            for k, v in rm.items():
                if k in racc and not np.isnan(v):
                    racc[k].append(v)

            # 诊断指标
            for b in range(B):
                for c in range(C):
                    ps, ts = pred_np[b, :, c], true_np[b, :, c]
                    r = _safe_corr(_rms_env(ps, _win), _rms_env(ts, _win))
                    if not np.isnan(r):
                        env_corr_list.append(r)
                    try:
                        mdf_err_list.append(abs(_mdf(ps, _fs) - _mdf(ts, _fs)))
                    except Exception:
                        pass

    result = {
        'mse':           float(np.mean(mse_list))        if mse_list        else float('nan'),
        'mae':           float(np.mean(mae_list))         if mae_list        else float('nan'),
        'correlation':   float(np.mean(corr_list))        if corr_list       else float('nan'),
        'envelope_corr': float(np.mean(env_corr_list))    if env_corr_list   else float('nan'),
        'dtw_distance':  float('nan'),
        'mdf_error':     float(np.mean(mdf_err_list))     if mdf_err_list    else float('nan'),
    }
    for k in region_keys:
        result[k] = float(np.mean(racc[k])) if racc[k] else float('nan')
    return result
