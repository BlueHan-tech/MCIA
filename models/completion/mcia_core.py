"""MCIA：带轴向注意力的多通道补全。

基于门控轴向注意力块的 12 通道 patch 级 EMG 补全（区别于编解码 MAE）。

前向路径：
  patch_embed -> mask_token -> temp_pos + chan_pos -> 可选 domain_embed
  -> LocalConvBypass -> GatedAxialBlock x n -> norm
  -> 可选 SynergyBottleneck -> 时域 pred_head
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    """逐通道 Conv1d patch 嵌入，卷积核=步长=patch_size。"""

    def __init__(self, seq_len: int = 256, patch_size: int = 8,
                 in_chans: int = 1, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x：(B, C, T) -> (B, C, N, D)。"""
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x.reshape(B, C, self.num_patches, -1)


class PatchRecover1D(nn.Module):
    """(B, C, N, P) -> (B, C, T)."""

    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, N, P = patches.shape
        return patches.reshape(B, C, N * P)


def derive_patch_time_mask(raw_time_mask_btc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """样本掩码 (B,T,C)，1=已知 -> patch 掩码 (B,C,N)，最小池化。"""
    B, T, C = raw_time_mask_btc.shape
    assert T % patch_size == 0, f"T ({T}) must be divisible by patch_size ({patch_size})"
    N = T // patch_size
    m = raw_time_mask_btc.transpose(1, 2).reshape(B, C, N, patch_size)
    return m.min(dim=-1).values


def derive_ch_mask_from_sample_mask(mask_btc: torch.Tensor) -> torch.Tensor:
    """样本掩码 (B,T,C)，1=已知 -> 通道掩码 (B,C)，1=该通道存在任意已知采样点。"""
    return mask_btc.max(dim=1).values


def _safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """避免每个 key 都被掩码的行产生 NaN。"""
    if mask.numel() == 0:
        return mask
    all_masked = mask.all(dim=1)
    if all_masked.any():
        mask = mask.clone()
        mask[all_masked] = False
    return mask


class LocalConvBypass(nn.Module):
    """保留突发尺度结构的局部 patch 卷积旁路。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.pw = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, D = x.shape
        h = x.reshape(B * C, N, D).transpose(1, 2)
        h = self.dw(h)
        h = self.pw(h)
        h = self.act(h)
        h = h.transpose(1, 2).reshape(B, C, N, D)
        h = self.norm(h)
        return x + h


class Adapter(nn.Module):
    """零初始化残差适配器，供后续迁移微调使用。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(1, embed_dim // 4)
        self.norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden, embed_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(self.act(self.down_proj(self.norm(x))))
        return x + h


class SynergyBottleneck(nn.Module):
    """波形恢复前的低维肌肉协同调制。

    从通道池化 token 提取 K 个时变协同因子并映射回通道门控；
    初始为恒等变换，可无缝加入旧检查点而不突变预测结果。
    """

    def __init__(
        self,
        embed_dim: int,
        n_channels: int,
        n_synergies: int = 6,
        gate_scale: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_synergies = max(1, int(n_synergies))
        self.gate_scale = float(gate_scale)
        self.norm = nn.LayerNorm(embed_dim)
        self.to_syn = nn.Linear(embed_dim, self.n_synergies)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.to_gate = nn.Linear(self.n_synergies, n_channels)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens：(B, C, N, D)。池化通道以估计时变协同状态。
        pooled = self.norm(tokens).mean(dim=1)        # (B, N, D)
        synergy = self.dropout(self.act(self.to_syn(pooled)))
        gate = torch.tanh(self.to_gate(synergy))      # (B, N, C)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)    # (B, C, N, 1)
        return tokens * (1.0 + self.gate_scale * gate)


class GatedAxialBlock(nn.Module):
    """时序多头注意力、通道多头注意力、FFN，可选适配器。"""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_gate = nn.Linear(embed_dim * 2, embed_dim)

        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.channel_gate = nn.Linear(embed_dim * 2, embed_dim)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.adapter: Optional[nn.Module] = None

    def forward(
        self,
        x: torch.Tensor,
        known_mask: torch.Tensor,
        chan_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, N, D = x.shape

        h = self.temporal_norm(x).reshape(B * C, N, D)
        time_key_mask = _safe_key_padding_mask((known_mask < 0.5).reshape(B * C, N))
        attn_out, _ = self.temporal_attn(
            h, h, h, key_padding_mask=time_key_mask, need_weights=False
        )
        attn_out = attn_out.reshape(B, C, N, D)
        gate = torch.sigmoid(self.temporal_gate(torch.cat([x, attn_out], dim=-1)))
        x = x + gate * self.dropout(attn_out)

        h = self.channel_norm(x).permute(0, 2, 1, 3).reshape(B * N, C, D)
        channel_key_mask = None
        if chan_valid_mask is not None:
            channel_key_mask = ~chan_valid_mask.bool()
            channel_key_mask = channel_key_mask[:, None, :].expand(B, N, C).reshape(B * N, C)
            channel_key_mask = _safe_key_padding_mask(channel_key_mask)
        attn_out, _ = self.channel_attn(
            h, h, h, key_padding_mask=channel_key_mask, need_weights=False
        )
        attn_out = attn_out.reshape(B, N, C, D).permute(0, 2, 1, 3).contiguous()
        gate = torch.sigmoid(self.channel_gate(torch.cat([x, attn_out], dim=-1)))
        x = x + gate * self.dropout(attn_out)

        x = x + self.ffn(self.ffn_norm(x))
        if self.adapter is not None:
            x = self.adapter(x)
        return x


class MCIA(nn.Module):
    """带轴向注意力的多通道补全（MCIA）。

    每通道 N=32 个 patch token；通过门控轴向注意力实现多通道协同，无摘要池化。
    """

    def __init__(
        self,
        window_size: int = 256,
        n_channels: int = 12,
        patch_size: int = 8,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_domains: int = 1,
        use_synergy_bottleneck: bool = False,
        n_synergies: int = 6,
        synergy_gate_scale: float = 0.5,
        synergy_dropout: float = 0.1,
        # 仅为旧调用点保留；本架构不使用。
        num_heads: Optional[int] = None,
        spatial_depth: Optional[int] = None,
        mlp_ratio: Optional[float] = None,
        **_: object,
    ):
        super().__init__()
        if num_heads is not None:
            n_heads = num_heads
        if spatial_depth is not None:
            n_layers = spatial_depth
        if mlp_ratio is not None and ffn_dim is None:
            ffn_dim = int(embed_dim * mlp_ratio)

        assert window_size % patch_size == 0, (
            f"window_size ({window_size}) must be divisible by patch_size ({patch_size})"
        )
        assert embed_dim % n_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.window_size = window_size
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.num_patches = window_size // patch_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.use_synergy_bottleneck = bool(use_synergy_bottleneck)
        self.n_synergies = int(n_synergies)
        self.synergy_gate_scale = float(synergy_gate_scale)
        self.synergy_dropout = float(synergy_dropout)

        N, D = self.num_patches, embed_dim
        self.patch_embed = PatchEmbed1D(window_size, patch_size, 1, D)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, D))
        self.uncond_token = nn.Parameter(torch.zeros(1, 1, 1, D))
        self.temp_pos = nn.Parameter(torch.zeros(1, 1, N, D))
        self.chan_pos = nn.Parameter(torch.zeros(1, n_channels, 1, D))
        self.domain_embed = nn.Embedding(num_domains, D)

        self.local_bypass = LocalConvBypass(D)
        self.blocks = nn.ModuleList([
            GatedAxialBlock(D, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.synergy_bottleneck = SynergyBottleneck(
            D,
            n_channels,
            n_synergies=self.n_synergies,
            gate_scale=self.synergy_gate_scale,
            dropout=self.synergy_dropout,
        ) if self.use_synergy_bottleneck else None
        self.pred_head = nn.Sequential(
            nn.Linear(D, patch_size),
            nn.Softplus(beta=10),
        )
        self.patch_recover = PatchRecover1D(patch_size)

        self._init_weights()

    def _init_weights(self) -> None:
        def _init_fn(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_fn)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.uncond_token, std=0.02)
        nn.init.normal_(self.temp_pos, std=0.02)
        nn.init.normal_(self.chan_pos, std=0.02)
        nn.init.zeros_(self.domain_embed.weight)
        nn.init.normal_(self.pred_head[0].weight, std=0.02)
        nn.init.zeros_(self.pred_head[0].bias)
        if self.synergy_bottleneck is not None:
            nn.init.zeros_(self.synergy_bottleneck.to_gate.weight)
            nn.init.zeros_(self.synergy_bottleneck.to_gate.bias)

    def _derive_time_mask(self, raw_time_mask_btc: torch.Tensor) -> torch.Tensor:
        return derive_patch_time_mask(raw_time_mask_btc, self.patch_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        x_masked: Optional[torch.Tensor] = None,
        drop_condition: bool = False,
        side: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        gender: Optional[torch.Tensor] = None,
        raw_time_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        chan_valid_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape
        assert T == self.window_size, f"T ({T}) must equal window_size ({self.window_size})"
        assert C == self.n_channels, f"C ({C}) must equal n_channels ({self.n_channels})"

        if raw_time_mask is not None:
            x = x * raw_time_mask
            time_mask = self._derive_time_mask(raw_time_mask).float()
        else:
            time_mask = x.new_ones(B, C, self.num_patches)

        if chan_valid_mask is None and mask is not None:
            chan_valid_mask = mask.float()

        tokens = self.patch_embed(x.transpose(1, 2).contiguous())
        if drop_condition:
            tokens = self.uncond_token.expand(B, C, self.num_patches, self.embed_dim).contiguous()
        else:
            tm = time_mask.unsqueeze(-1)
            mask_tok = self.mask_token.expand(B, C, self.num_patches, self.embed_dim)
            tokens = tokens * tm + mask_tok * (1.0 - tm)

        tokens = tokens + self.temp_pos + self.chan_pos
        if domain_id is not None:
            tokens = tokens + self.domain_embed(domain_id)[:, None, None, :]

        tokens = self.local_bypass(tokens)
        for block in self.blocks:
            tokens = block(tokens, time_mask, chan_valid_mask=chan_valid_mask)
        tokens = self.norm(tokens)
        if self.synergy_bottleneck is not None:
            tokens = self.synergy_bottleneck(tokens)

        pred_patches = self.pred_head(tokens)
        pred = self.patch_recover(pred_patches).transpose(1, 2).contiguous()

        if return_aux:
            return {
                "pred": pred,
                "time": pred,
            }
        return pred

    def compute_mae_loss(self, pred: torch.Tensor, target: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none")
        if mask is None:
            return loss.mean()
        mask_missing = 1.0 - (mask.unsqueeze(1) if mask.dim() == 2 else mask)
        return (loss * mask_missing).sum() / mask_missing.sum().clamp_min(1.0)


class MCIA_Wrapper:
    """MCIA 训练/推理封装器，观测区执行 copy-back。"""

    def __init__(self, model: MCIA):
        self.model = model
        self.timesteps = 1

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_start

    def p_sample(
        self,
        model: MCIA,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        x_clean_masked: torch.Tensor,
        guidance_scale: float = 0.0,
        raw_time_mask: Optional[torch.Tensor] = None,
        side: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        gender: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        chan_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = dict(
            mask=mask,
            x_masked=x_clean_masked,
            raw_time_mask=raw_time_mask,
            side=side,
            age=age,
            gender=gender,
            domain_id=domain_id,
            chan_valid_mask=chan_valid_mask,
        )
        if guidance_scale > 0.0:
            pred_cond = model(x_clean_masked, drop_condition=False, **kwargs)
            pred_uncond = model(x_clean_masked, drop_condition=True, **kwargs)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred = model(x_clean_masked, drop_condition=False, **kwargs)

        # Constrain completion after guidance, before copying observations back.
        pred = pred.clamp(0.0, 1.0)
        if raw_time_mask is not None:
            return pred * (1.0 - raw_time_mask) + x_clean_masked * raw_time_mask
        mask_expanded = mask.unsqueeze(1).expand_as(pred)
        return pred * (1.0 - mask_expanded) + x_clean_masked * mask_expanded
