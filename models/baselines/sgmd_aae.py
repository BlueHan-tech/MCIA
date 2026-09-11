"""SGMD-AAE 文献基线（Zou、Cheng 和 Han，2023）。

Reported architecture and hyperparameters are implemented verbatim where the
paper specifies them.  The paper does not specify convolution padding, final
linear dimensions, or epoch count; those are explicit caller choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SGMDAAEConfig:
    generator_lr: float = 1e-3
    discriminator_lr: float = 2e-4
    selfguided_weight: float = 10.0
    style_weight: float = 1.0
    align_weight: float = 0.5
    adversarial_weight: float = 0.6
    dropout: float = 0.5
    leaky_relu_slope: float = 0.2


class SelfMaskPartialConv2d(nn.Module):
    """公式 2--3 的自掩码部分卷积，并使用等效于 LayerNorm 的 GN。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding: tuple[int, int] | str = (0, 0)):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if padding != "same" and (not isinstance(padding, tuple) or len(padding) != 2):
            raise ValueError("padding must be a (height, width) tuple or 'same'.")
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.register_buffer("ones", torch.ones(1, in_channels, *kernel_size), persistent=False)

    def _pad(self, values: torch.Tensor) -> torch.Tensor:
        if self.padding == "same":
            pad_h = self.kernel_size[0] - 1
            pad_w = self.kernel_size[1] - 1
            return F.pad(values, (pad_w // 2, pad_w - pad_w // 2,
                                  pad_h // 2, pad_h - pad_h // 2))
        pad_h, pad_w = self.padding
        return F.pad(values, (pad_w, pad_w, pad_h, pad_h)) if (pad_h or pad_w) else values

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.shape[1] == 1 and x.shape[1] != 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        if mask.shape != x.shape:
            raise ValueError(f"Self-mask shape {mask.shape} must match feature shape {x.shape}")
        masked = self._pad(x * mask)
        padded_mask = self._pad(mask)
        output = F.conv2d(masked, self.conv.weight, bias=None, stride=self.conv.stride)
        counts = F.conv2d(padded_mask, self.ones.to(dtype=x.dtype), stride=self.conv.stride)
        total = float(self.ones[0].numel())
        valid = counts > 0
        output = output * (total / counts.clamp_min(1.0))
        if self.conv.bias is not None:
            output = output + self.conv.bias.view(1, -1, 1, 1)
        output = torch.where(valid, output, torch.zeros_like(output))
        output = self.activation(self.norm(output))
        next_mask = valid.any(dim=1, keepdim=True).to(dtype=x.dtype).expand_as(output)
        return output, next_mask


class SGMDAAEGenerator(nn.Module):
    """表 1 中用于 240 x 12 x 1 sEMG 输入的 U-Net 生成器。"""
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([
            # 表 1：240x12 -> 24x12 -> 8x4 -> 4x2 -> 2x1 -> 1x1。
            SelfMaskPartialConv2d(1, 128, (10, 1), (10, 1), padding=(0, 0)),
            SelfMaskPartialConv2d(128, 256, (3, 3), (3, 3), padding=(0, 0)),
            SelfMaskPartialConv2d(256, 512, (3, 3), (2, 2), padding=(1, 1)),
            SelfMaskPartialConv2d(512, 512, (3, 3), (2, 2), padding=(1, 1)),
            SelfMaskPartialConv2d(512, 512, (2, 1), (2, 1), padding=(0, 0)),
        ])
        self.dec = nn.ModuleList([
            # Pconv 保持上采样后的空间尺寸；偶数卷积核在 PyTorch 中需要
            # 非对称的 SAME 填充。
            SelfMaskPartialConv2d(1024, 512, (2, 1), padding="same"),
            SelfMaskPartialConv2d(1024, 512, (2, 2), padding="same"),
            SelfMaskPartialConv2d(1024, 256, (2, 2), padding="same"),
            SelfMaskPartialConv2d(512, 128, (3, 3), padding="same"),
            SelfMaskPartialConv2d(256, 1, (10, 1), padding="same"),
        ])
        # 表 1 在五次拼接处列出 1024/1024/1024/512/256 通道，而编码器对应
        # 两个尺度列出 256 和 128 通道。这些 1x1 投影在不改变尺度的前提下
        # 调和论文公开的维度不一致。
        self.skip_projections = nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Conv2d(256, 512, 1),
            nn.Conv2d(128, 256, 1), nn.Identity(),
        ])

    @staticmethod
    def _resize(pair: tuple[torch.Tensor, torch.Tensor], size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = pair
        return F.interpolate(x, size=size, mode="nearest"), F.interpolate(mask, size=size, mode="nearest")

    def forward(self, incomplete: torch.Tensor, self_mask: torch.Tensor) -> torch.Tensor:
        if incomplete.ndim != 4 or incomplete.shape[1] != 1:
            raise ValueError("Expected sEMG image input shaped (B,1,time,channels).")
        x, mask = incomplete, self_mask
        encoded: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.enc:
            x, mask = block(x, mask)
            encoded.append((x, mask))
        # 图 2 经由 e4/e3/e2/e1 解码 e5，随后在全分辨率处复用 e1 用于
        # Concate-5。原始单通道输入不是最终跳连，因为表 1 在此指定 256 通道。
        skip_indices = (-2, -3, -4, -5, 0)
        for index, block in enumerate(self.dec):
            skip_x, skip_mask = encoded[skip_indices[index]]
            skip_x = self.skip_projections[index](skip_x)
            if skip_mask.shape[1] != skip_x.shape[1]:
                skip_mask = skip_mask[:, :1].expand_as(skip_x)
            output_size = incomplete.shape[-2:] if index == len(self.dec) - 1 else skip_x.shape[-2:]
            x, mask = self._resize((x, mask), output_size)
            if skip_x.shape[-2:] != output_size:
                skip_x, skip_mask = self._resize((skip_x, skip_mask), output_size)
            x, mask = torch.cat([x, skip_x], dim=1), torch.cat([mask, skip_mask], dim=1)
            x, mask = block(x, mask)
        return x[..., :incomplete.shape[-2], :incomplete.shape[-1]]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, pool, dropout: float):
        super().__init__()
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=tuple(k // 2 for k in kernel_size)),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(pool, ceil_mode=True), nn.Dropout2d(dropout),
        )
    def forward(self, x): return self.net(x)


class SGMDMultiViewDiscriminator(nn.Module):
    """图 3 的原始多尺度与 FFT 频谱图判别器。"""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.raw_branches = nn.ModuleList([
            nn.Sequential(_ConvBlock(1, 32, (kernel, 3), (10, 3), dropout), _ConvBlock(32, 64, (3, 3), (2, 2), dropout))
            for kernel in (10, 20, 30, 40, 50)
        ])
        self.raw_tail = nn.ModuleList([
            _ConvBlock(320, 128, (3, 3), (2, 2), dropout),
            _ConvBlock(128, 128, (3, 3), (2, 2), dropout),
            _ConvBlock(128, 128, (3, 3), (2, 2), dropout),
        ])
        self.spec_tail = nn.ModuleList([_ConvBlock(1 if i == 0 else 128, 128, (3, 3), (2, 2), dropout) for i in range(5)])
        self.classifier = nn.Sequential(nn.LazyLinear(1))

    @staticmethod
    def _spectrogram(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.fft.rfft(x, dim=2))

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        raw = torch.cat([branch(x) for branch in self.raw_branches], dim=1)
        raw_features = []
        for block in self.raw_tail:
            raw = block(raw); raw_features.append(raw)
        spec = self._spectrogram(x)
        spec_features = []
        for block in self.spec_tail:
            spec = block(spec); spec_features.append(spec)
        # 公式 5 选择 conv-4、conv-5、conv-9 和 conv-10。
        return [raw_features[1], raw_features[2], spec_features[3], spec_features[4]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = torch.cat([feature.flatten(1) for feature in (features[1], features[3])], dim=1)
        return self.classifier(flattened).squeeze(1)


def _gram(feature: torch.Tensor) -> torch.Tensor:
    flat = feature.flatten(2)
    return torch.bmm(flat, flat.transpose(1, 2)) / max(flat.shape[-1], 1)


class SGMDAAEObjective(nn.Module):
    """公式 4--12 的生成器目标与交替训练的判别器 BCE。"""
    def __init__(self, config: SGMDAAEConfig):
        super().__init__(); self.config = config

    def generator_loss(self, output, target, observed_mask, discriminator: SGMDMultiViewDiscriminator):
        completed = output * (1.0 - observed_mask) + target * observed_mask
        raw = F.l1_loss(output * (1.0 - observed_mask), target * (1.0 - observed_mask)) + F.l1_loss(output * observed_mask, target * observed_mask)
        generated = discriminator.features(output)
        complete_features = discriminator.features(completed)
        target_features = discriminator.features(target)
        perceptual = sum(F.l1_loss(a, c) + F.l1_loss(b, c) for a, b, c in zip(generated, complete_features, target_features))
        style = sum(F.l1_loss(_gram(a), _gram(c)) + F.l1_loss(_gram(b), _gram(c)) for a, b, c in zip(generated, complete_features, target_features))
        align = sum(F.l1_loss(feature.mean(dim=0), truth.mean(dim=0)) for feature, truth in zip(generated, target_features))
        temporal_tv = torch.abs(completed[:, :, 1:, :] - completed[:, :, :-1, :]) * (1.0 - observed_mask[:, :, 1:, :])
        selfguided = raw + perceptual + temporal_tv.mean()
        adversarial = F.binary_cross_entropy_with_logits(discriminator(output), torch.ones(output.shape[0], device=output.device))
        total = (self.config.selfguided_weight * selfguided + self.config.style_weight * style +
                 self.config.align_weight * align + self.config.adversarial_weight * adversarial)
        return total, {"selfguided": selfguided.detach(), "style": style.detach(), "align": align.detach(), "adversarial": adversarial.detach()}

    @staticmethod
    def discriminator_loss(real_logits, fake_logits):
        return 0.5 * (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


@torch.no_grad()
def sgmd_complete(generator: SGMDAAEGenerator, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    """交付重建结果，同时保留真实观测到的 sEMG。"""
    pred = generator(values * observed_mask, observed_mask)
    return pred * (1.0 - observed_mask) + values * observed_mask
