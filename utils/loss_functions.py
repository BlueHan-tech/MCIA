"""EMG 补全损失函数。"""

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGImputationLoss(nn.Module):
    """缺失区域 EMG 补全损失。"""

    def __init__(
        self,
        w_charbonnier: float = 1.0,
        w_ncc: float = 0.5,
        w_stft: float = 0.3,
        w_boundary: float = 0.1,
        w_aux: float = 0.1,
        aux_ratio: float = 0.10,
        fft_sizes: Iterable[int] = (32, 64, 128),
        eps: float = 1e-4,
        boundary_radius: int = 4,
        envelope_loss_weight: float = 0.0,
        patch_rms_loss_weight: float = 0.0,
        envelope_kernel_size: int = 25,
        patch_rms_size: int = 8,
        range_penalty_weight: float = 0.0,
        range_low: float = 0.0,
        range_high: float = 1.0,
    ):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ncc = w_ncc
        self.w_stft = w_stft
        self.w_boundary = w_boundary
        self.w_aux = w_aux
        self.aux_ratio = aux_ratio
        self.fft_sizes = tuple(fft_sizes)
        self.eps = eps
        self.boundary_radius = boundary_radius
        self.envelope_loss_weight = float(envelope_loss_weight)
        self.patch_rms_loss_weight = float(patch_rms_loss_weight)
        self.envelope_kernel_size = max(3, int(envelope_kernel_size) | 1)
        self.patch_rms_size = max(1, int(patch_rms_size))
        self.range_penalty_weight = float(range_penalty_weight)
        self.range_low = float(range_low)
        self.range_high = float(range_high)
        self.current_epoch = 0
        self.window_funcs: Dict[Tuple[torch.device, int], torch.Tensor] = {}

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _masked_mean(self, value: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        denom = loss_mask.sum().clamp_min(1.0)
        return (value * loss_mask).sum() / denom

    def _charbonnier(self, pred: torch.Tensor, target: torch.Tensor,
                     loss_mask: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return self._masked_mean(torch.sqrt(diff.square() + self.eps ** 2), loss_mask)

    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = min(self.envelope_kernel_size, T if T % 2 == 1 else max(1, T - 1))
        if k <= 1:
            return x.abs()
        pad = k // 2
        x_flat = x.abs().permute(0, 2, 1).reshape(B * C, 1, T)
        env = F.avg_pool1d(x_flat, kernel_size=k, stride=1, padding=pad)
        return env.reshape(B, C, T).permute(0, 2, 1)

    def _envelope_loss(self, pred: torch.Tensor, target: torch.Tensor,
                       loss_mask: torch.Tensor) -> torch.Tensor:
        # 平滑前将真值复制到观测区，使梯度仍绑定缺失区预测，同时包络仍能看到真实上下文。
        pred_completed = pred * loss_mask + target * (1.0 - loss_mask)
        pred_env = self._envelope(pred_completed)
        target_env = self._envelope(target)
        return self._charbonnier(pred_env, target_env, loss_mask)

    def _patch_rms_loss(self, pred: torch.Tensor, target: torch.Tensor,
                        loss_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = pred.shape
        p = min(self.patch_rms_size, T)
        n_patches = T // p
        if n_patches < 1:
            return pred.new_zeros(())
        T_crop = n_patches * p
        pred_completed = pred * loss_mask + target * (1.0 - loss_mask)
        pred_p = pred_completed[:, :T_crop, :].reshape(B, n_patches, p, C)
        tgt_p = target[:, :T_crop, :].reshape(B, n_patches, p, C)
        mask_p = loss_mask[:, :T_crop, :].reshape(B, n_patches, p, C).mean(dim=2)
        pred_rms = torch.sqrt(pred_p.square().mean(dim=2) + self.eps ** 2)
        tgt_rms = torch.sqrt(tgt_p.square().mean(dim=2) + self.eps ** 2)
        denom = mask_p.sum().clamp_min(1.0)
        return (torch.abs(pred_rms - tgt_rms) * mask_p).sum() / denom

    def _ncc(self, pred: torch.Tensor, target: torch.Tensor,
             loss_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = pred.shape
        p = pred.permute(0, 2, 1).reshape(B * C, T)
        y = target.permute(0, 2, 1).reshape(B * C, T)
        m = loss_mask.permute(0, 2, 1).reshape(B * C, T)

        count = m.sum(dim=1)
        valid = count >= 2
        if not valid.any():
            return pred.new_zeros(())

        denom_count = count.clamp_min(1.0).unsqueeze(1)
        p_mean = (p * m).sum(dim=1, keepdim=True) / denom_count
        y_mean = (y * m).sum(dim=1, keepdim=True) / denom_count
        p = (p - p_mean) * m
        y = (y - y_mean) * m

        num = (p * y).sum(dim=1)
        p_norm = p.square().sum(dim=1).detach().clamp_min(1e-6).sqrt()
        y_norm = y.square().sum(dim=1).detach().clamp_min(1e-6).sqrt()
        corr = num / (p_norm * y_norm).clamp_min(1e-6)
        return (1.0 - corr[valid]).mean()

    def _stft(self, pred: torch.Tensor, target: torch.Tensor,
              loss_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = pred.shape
        pred_flat = (pred * loss_mask).permute(0, 2, 1).reshape(B * C, T)
        target_flat = (target * loss_mask).permute(0, 2, 1).reshape(B * C, T)
        mask_flat = loss_mask.permute(0, 2, 1).reshape(B * C, T)

        total = pred.new_zeros(())
        for n_fft in self.fft_sizes:
            key = (pred.device, pred.dtype, n_fft)
            if key not in self.window_funcs:
                self.window_funcs[key] = torch.hann_window(
                    n_fft, device=pred.device, dtype=pred.dtype
                )
            window = self.window_funcs[key]
            hop = max(1, n_fft // 4)
            pred_spec = torch.stft(
                pred_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True
            ).abs()
            target_spec = torch.stft(
                target_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True
            ).abs()

            pad = n_fft // 2
            frame_weight = F.conv1d(
                F.pad(mask_flat.unsqueeze(1), (pad, pad)),
                torch.ones(1, 1, n_fft, device=pred.device, dtype=pred.dtype) / n_fft,
                stride=hop,
            ).squeeze(1)
            n_frames = min(frame_weight.shape[-1], pred_spec.shape[-1])
            frame_weight = frame_weight[:, :n_frames].clamp(0.0, 1.0).detach()
            pred_spec = pred_spec[:, :, :n_frames]
            target_spec = target_spec[:, :, :n_frames]

            w = frame_weight.unsqueeze(1)
            denom = (w.sum() * pred_spec.shape[1]).clamp_min(1.0)
            linear = (torch.abs(pred_spec - target_spec) * w).sum() / denom
            log = (
                torch.abs(torch.log(pred_spec + 1e-6) - torch.log(target_spec + 1e-6)) * w
            ).sum() / denom
            total = total + linear + log
        return total / len(self.fft_sizes)

    def _boundary_grad(self, pred: torch.Tensor, target: torch.Tensor,
                       loss_mask: torch.Tensor) -> torch.Tensor:
        grad_pred = pred[:, 1:, :] - pred[:, :-1, :]
        grad_target = target[:, 1:, :] - target[:, :-1, :]
        grad_missing = torch.maximum(loss_mask[:, 1:, :], loss_mask[:, :-1, :])
        boundary = (loss_mask[:, 1:, :] - loss_mask[:, :-1, :]).abs()

        B, Tg, C = boundary.shape
        band = boundary.permute(0, 2, 1).reshape(B * C, 1, Tg)
        kernel = torch.ones(
            1, 1, self.boundary_radius * 2 + 1,
            device=pred.device, dtype=pred.dtype,
        )
        band = F.conv1d(band, kernel, padding=self.boundary_radius)
        band = (band > 0).float().reshape(B, C, Tg).permute(0, 2, 1)
        band = band * grad_missing

        denom = band.sum().clamp_min(1.0)
        return (torch.abs(grad_pred - grad_target) * band).sum() / denom

    def _range_penalty(self, pred: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """软范围惩罚：缺失区预测超出 [range_low, range_high] 的二次惩罚。

        与推理期硬 clip 配套，使训练目标与交付约束一致；权重 0 时完全关闭。
        """
        low = torch.relu(self.range_low - pred).square()
        high = torch.relu(pred - self.range_high).square()
        return self._masked_mean(low + high, loss_mask)

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   loss_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_mask.sum() < 1:
            zero = pred.new_zeros(())
            return zero, {
                "charbonnier_loss": zero,
                "ncc_loss": zero,
                "stft_loss": zero,
                "boundary_grad_loss": zero,
                "envelope_loss": zero,
                "patch_rms_loss": zero,
                "range_penalty_loss": zero,
            }

        charbonnier = self._charbonnier(pred, target, loss_mask)
        ncc = self._ncc(pred, target, loss_mask)
        stft = self._stft(pred, target, loss_mask)
        boundary = self._boundary_grad(pred, target, loss_mask)
        envelope = self._envelope_loss(pred, target, loss_mask)
        patch_rms = self._patch_rms_loss(pred, target, loss_mask)
        range_penalty = (
            self._range_penalty(pred, loss_mask)
            if self.range_penalty_weight > 0 else pred.new_zeros(())
        )
        total = (
            self.w_charbonnier * charbonnier
            + self.w_ncc * ncc
            + self.w_stft * stft
            + self.w_boundary * boundary
            + self.envelope_loss_weight * envelope
            + self.patch_rms_loss_weight * patch_rms
            + self.range_penalty_weight * range_penalty
        )
        return total, {
            "charbonnier_loss": charbonnier,
            "ncc_loss": ncc,
            "stft_loss": stft,
            "boundary_grad_loss": boundary,
            "envelope_loss": envelope,
            "patch_rms_loss": patch_rms,
            "range_penalty_loss": range_penalty,
        }

    def forward(self, pred_x0: torch.Tensor, target_x0: torch.Tensor,
                mask: torch.Tensor):
        """
        参数：
            pred_x0: (B, T, C)，或 MCIA(return_aux=True) 的辅助字典
            target_x0: (B, T, C)
            mask: (B, T, C)，1=已知，0=缺失
        """
        aux_pred = pred_x0 if isinstance(pred_x0, dict) else None
        pred_main = aux_pred.get("time", aux_pred.get("pred")) if aux_pred is not None else pred_x0

        loss_mask = (1.0 - mask).float()
        total, parts = self._base_loss(pred_main, target_x0, loss_mask)

        aux_loss = pred_main.new_zeros(())
        if self.w_aux > 0 and self.aux_ratio > 0:
            known = mask.float()
            aux_mask = (torch.rand_like(known) < self.aux_ratio).float() * known
            aux_loss, _ = self._base_loss(pred_main, target_x0, aux_mask)
            total = total + self.w_aux * aux_loss

        out = {k: float(v.detach().item()) for k, v in parts.items()}
        out["aux_loss"] = float(aux_loss.detach().item())
        return total, out


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """DDPM 兼容辅助函数。"""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
