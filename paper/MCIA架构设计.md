> 当前实现已固定为 S1 短时缺失 20%、S2 单整通道缺失 40%、S3 双整通道缺失 40%。本文后续关于 S4 和 curriculum 的内容仅作历史设计记录，不代表当前主流程默认行为。

Multi-Channel Imputation with Axial-attention
# 现有数据处理
原始采样率：2000
目标采样率：200
通道数为12
任务目标：双向补全，有前后context
DB2: 40名健康被试，12通道 sEMG
DB3: 11名截肢患者，12通道 sEMG
window_size: 256       # 分段窗口长度（采样点）
stride: 128            # 滑动步长

# 整体架构
把目前所有设计决策汇总一下，确认没有遗漏：
```
输入 (B, C, T)
  ↓ × mask_binary → emg_masked
  ↓ PatchEmbed1D（每通道独立 Conv1d, P=8）
(B, C, N=32, D=128)
  ↓ mask_token 替换缺失 patch
  ↓ temp_pos + chan_pos（可学习，分轴）
  ↓ LocalConvBypass（depthwise conv, kernel=3）
  ↓ × 4 层 GatedAxialBlock
      ├─ Temporal MHA（N轴，key_padding_mask 屏蔽缺失 patch）
      ├─ Channel MHA（C轴，无 mask）
      └─ FFN（GELU, dropout=0.1）
  ↓ LayerNorm
  ↓ pred_head Linear(D=128 → P=8)
(B, C, N, P) → reshape → (B, C, T)
  ↓ 推理回填：pred × (1-mask) + input × mask

Loss（只在缺失区）：
  Charbonnier(w=1.0) + NCC(w=0.5) + MultiScaleSTFT(w=0.3) + BoundaryGrad(w=0.1)

训练策略：
  CurriculumMaskScheduler，5阶段，按 val_corr_masked 退出
```
现在架构、loss、训练策略三层都定下来了。
# Patch Embed
将连续的、高维的数据结构，转化为 Transformer 能够处理的离散 Token（词元）序列
(B, C, T=256)
  ↓ patch embed (P=8)
  (B, C, N=32, D)         ← 不 pool，32 个 patch token 全部保留

# Factorized Axial Attention
从不同通道上学习
从不同时间片上学习
### 第一步：用 Factorized Axial Attention 替换 Pool→Spatial→Upsample

#### 核心设计
去掉 pool/upsample，直接在 `(B, C, N, D)` 上做双轴交替 attention：

```
(B, C, N, D)
  ↓ [× L layers]
  Temporal Attention：在 N 轴上，每个通道独立
  Channel Attention： 在 C 轴上，每个时间步独立
  各自带 Pre-LN + residual
  ↓
(B, C, N, D)
  ↓ pred_head: Linear(D, P)
(B, C, N, P) → reshape → (B, C, T)
```

#### 为什么是交替而非并行/concat

文献支撑：**Axial-Attention**（Ho et al., NeurIPS 2019）证明，对二维结构数据（这里是 time × channel），沿两轴交替做 attention 的表达能力等价于联合 attention，但复杂度从 `O((CN)²)` 降到 `O(CN² + C²N)`。

你的规模：`C=12, N=32` → 联合 attention = 384² ≈ 147k，交替 = 12×32² + 12²×32 ≈ 17k，**约 8× 更高效**，且对这个数据量更不容易过拟合。
#### Mask 处理：关键细节

当前你用 `mask_patch` 替换缺失 patch 的 embed，这个做法**保留**，但有一个必须修的地方：

**Temporal attention 的 attention mask 要显式传入。**

缺失区的 token 不应该在 temporal attention 里被其他位置当作 key 来 attend（否则模型会 attend 到 mask token 的伪信息）：
```python
# temporal attention 时
# known_mask: (B, C, N) bool，True=已知
# 对每个通道，已知位置可以作为 key，缺失位置只作为 query
attn_mask = ~known_mask  # 缺失位置的 key 被 mask 掉
```
这是 **BERT-style masked attention** 的标准做法，**MAE**（He et al., CVPR 2022）和 **SAITS** 都用这个，不传 attention mask 是当前架构的另一个隐性问题。
#### 位置编码选择

用可学习位置编码分轴：
```python
self.temp_pos = nn.Parameter(torch.zeros(1, 1, N, D))   # 时间轴，广播到所有通道
self.chan_pos = nn.Parameter(torch.zeros(1, C, 1, D))   # 通道轴，广播到所有时间步
# forward:
x = x + self.temp_pos + self.chan_pos  # 在进 attention 之前加
```
不用 RoPE 或 sinusoidal，原因：窗口固定 T=256，不需要外推能力；可学习编码在小数据集上拟合更快。支撑：**PatchTST**（ICLR 2023）在固定长度时序上用可学习位置编码，优于 sinusoidal。
#### 最小实现骨架

python

```python
class AxialBlock(nn.Module):
    def __init__(self, D, n_heads, C, N):
        # Temporal: 在 N 轴上 attention，C 个独立
        self.temp_attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.temp_ln   = nn.LayerNorm(D)
        # Channel: 在 C 轴上 attention，N 个独立
        self.chan_attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.chan_ln   = nn.LayerNorm(D)
        self.ffn       = FFN(D)
        self.ffn_ln    = nn.LayerNorm(D)

    def forward(self, x, temp_key_mask=None):
        # x: (B, C, N, D)
        B, C, N, D = x.shape

        # --- Temporal attention: 对每个通道，N 个 token 互相 attend ---
        xt = x.reshape(B * C, N, D)
        if temp_key_mask is not None:
            km = temp_key_mask.reshape(B * C, N)  # True=被mask掉的key
        else:
            km = None
        xt = self.temp_ln(xt)
        xt_out, _ = self.temp_attn(xt, xt, xt, key_padding_mask=km)
        x = x + xt_out.reshape(B, C, N, D)

        # --- Channel attention: 对每个时间步，C 个 token 互相 attend ---
        xc = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
        xc = self.chan_ln(xc)
        xc_out, _ = self.chan_attn(xc, xc, xc)  # 通道间无 mask
        x = x + xc_out.reshape(B, N, C, D).permute(0, 2, 1, 3)

        # --- FFN ---
        x = x + self.ffn(self.ffn_ln(x))
        return x
```

注意：这里用 **Pre-LN**（LN 在 sublayer 之前），不是原始 Transformer 的 Post-LN。理由：你数据量小，Pre-LN 训练更稳定，不需要 warmup trick。支撑：**On Layer Normalization in the Transformer Architecture**（Xiong et al., ICML 2020）。
### 双轴 Attention 的本质
它不是「不用 Encoder-Decoder」，而是用了一个**对称结构**：
```
每个token（无论已知/缺失）都参与 attention
已知区 token：提供 key/value，也更新自己的表示
缺失区 token：只从已知区attend，不提供可信的 key（通过 attention mask）
```
这在文献里叫 **in-context imputation**，和 BERT 的 masked language model 是同一个思路：用双向上下文直接在同一序列内补全，不需要压缩再解压。
支撑：**iTransformer**（Liu et al., ICLR 2024）、**TimesNet**（Wu et al., ICLR 2023）都指出对时序数据，保留全序列的 token 表示比 bottleneck 压缩更有利于局部结构恢复。

假设输入特征为 $X$：
1. **主干道（全局视野）：** $X$ 进入 Multi-Head Self-Attention，计算全局的上下文依赖。
$$H_{\text{global}} = \text{Attention}(X)$$
2. **旁路道（局部细节）：** $X$ 同时（或紧接着）进入一个包含局部卷积的网络分支。
$$H_{\text{local}} = \text{Conv}(X)$$

3. **特征融合（残差相加）：** 将两者的结果通过加法（或拼接）融合在一起，并加上原始输入。
$$Y = H_{\text{global}} + H_{\text{local}} + X$$
如何避免attention输出的是直线。特征融合（残差相加）的方式应该如何去考虑
#### 修正一：Gate 替换残差相加（针对缺失区）

不用加法，用 learned gate 控制每层 attention 的贡献比例：
```python
class GatedAxialBlock(nn.Module):
    def __init__(self, D, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.temp_ln   = nn.LayerNorm(D)
        self.temp_attn = nn.MultiheadAttention(D, n_heads, batch_first=True, dropout=dropout)
        self.temp_gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        
        self.chan_ln   = nn.LayerNorm(D)
        self.chan_attn = nn.MultiheadAttention(D, n_heads, batch_first=True, dropout=dropout)
        self.chan_gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())
        
        self.ffn_ln = nn.LayerNorm(D)
        self.ffn    = FFN(D, ffn_dim, dropout)

    def forward(self, x, temp_key_mask=None):
        B, C, N, D = x.shape

        # Temporal
        xt = x.reshape(B * C, N, D)
        xt_norm = self.temp_ln(xt)
        km = temp_key_mask.reshape(B * C, N) if temp_key_mask is not None else None
        xt_out, _ = self.temp_attn(xt_norm, xt_norm, xt_norm, key_padding_mask=km)
        # Gate：由原始表示和 attention 输出共同决定融合比例
        g = self.temp_gate(torch.cat([xt, xt_out], dim=-1))  # (B*C, N, D)
        x = x + (g * xt_out).reshape(B, C, N, D)

        # Channel
        xc = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
        xc_norm = self.chan_ln(xc)
        xc_out, _ = self.chan_attn(xc_norm, xc_norm, xc_norm)
        g = self.chan_gate(torch.cat([xc, xc_out], dim=-1))
        x = x + (g * xc_out).reshape(B, N, C, D).permute(0, 2, 1, 3)

        x = x + self.ffn(self.ffn_ln(x))
        return x
```

Gate 的作用：当 attention 输出和当前表示差异大时（缺失区早期，mask token 还没被校正），gate 自动压低融合比例，避免把均值回归的结果强行写入残差链。

文献支撑：**GRU** 的 update gate 和 **Highway Network**（Srivastava et al., ICML 2015）都证明了 learned gate 比固定残差在信号恢复任务上更稳定。在 Transformer 里的应用见 **Gated Linear Units**（Dauphin et al., ICML 2017）和 **GLU Variants**（Noam, 2020）。
#### 修正二：Local Conv Bypass（针对 burst 高频结构）

这也是我上一步说的那个 EMG 专属组件，现在正好一起放进来。

在 patch embed 之后加一条并行的 depthwise conv 路径：

python

```python
class LocalConvBypass(nn.Module):
    """
    在 attention 之前，给每个通道的 patch token 序列加一条
    局部卷积旁路，保留 burst 的局部高频结构。
    """
    def __init__(self, D, kernel_size=3):
        super().__init__()
        # Depthwise：每个 D 维独立卷积，不混合特征维度
        self.dw_conv = nn.Conv1d(D, D, kernel_size=kernel_size,
                                 padding=kernel_size//2, groups=D)
        self.pw_conv = nn.Conv1d(D, D, kernel_size=1)  # Pointwise 混合
        self.ln      = nn.LayerNorm(D)
        self.act     = nn.GELU()

    def forward(self, x):
        # x: (B, C, N, D)
        B, C, N, D = x.shape
        xr = x.reshape(B * C, N, D).permute(0, 2, 1)  # (B*C, D, N)
        out = self.act(self.pw_conv(self.dw_conv(xr)))
        out = out.permute(0, 2, 1).reshape(B, C, N, D)
        return x + self.ln(out)
```

加在 `AxialTransformerImputer.forward` 里，位置在位置编码之后、进 blocks 之前：

python

```python
tokens = tokens + self.temp_pos + self.chan_pos
tokens = self.local_conv(tokens)   # ← 这里
for block in self.blocks:
    tokens = block(tokens, temp_key_mask)
```

文献支撑：**ConvNeXt**（Liu et al., CVPR 2022）证明 depthwise conv 作为局部归纳偏置的残差旁路，能让 Transformer 在局部结构恢复上显著提升，且参数开销极小。在时序上的对应：**TimesNet**（Wu et al., ICLR 2023）用 conv 旁路保留时序局部特征，防止纯 attention 的过度平滑。

# loss 设计
#### Layer 0：主重建项，用 Charbonnier Loss 替换 MSE

MSE 对异常值（burst 峰值）惩罚过重，会让模型保守地输出均值来规避大误差。MAE 更鲁棒但梯度在 0 处不连续。

Charbonnier loss 是两者的折中：

```
L_char(x, y) = √((x - y)² + ε²)
```

在大误差时行为像 MAE（线性惩罚），在小误差时行为像 MSE（平滑梯度）。

python

```python
def charbonnier(pred, target, mask, eps=1e-3):
    """mask: 1=缺失区（需要计算loss的位置）"""
    diff = torch.sqrt((pred - target) ** 2 + eps ** 2)
    return (diff * mask).sum() / (mask.sum() + 1e-8)
```

文献支撑：**Charbonnier loss** 在图像/时序恢复任务中系统性优于 MSE，见 **EDSR**（Lim et al., CVPRW 2017），在时序上见 **Pyraformer**（Liu et al., ICLR 2022）的重建实验。
#### Layer 1：频域结构项，用多尺度 STFT Loss

Burst 是局部高频事件，时域 loss 对频率结构不敏感。STFT loss 在频域直接惩罚频谱差异：

python

```python
def multiscale_stft_loss(pred, target, mask_point,
                          fft_sizes=[16, 32, 64],
                          hop_sizes=[4, 8, 16],
                          win_sizes=[16, 32, 64]):
    """
    pred, target: (B, C, T)
    mask_point:   (B, C, T) 1=缺失区
    """
    loss = 0.0
    B, C, T = pred.shape
    
    # 只在缺失区计算，且需要足够长的连续段
    # 用 mask 加权而非截断，避免边界 artifact
    pred_m   = pred * mask_point
    target_m = target * mask_point

    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # reshape: (B*C, T)
        p = pred_m.reshape(B * C, T)
        t = target_m.reshape(B * C, T)
        
        p_stft = torch.stft(p, fft_size, hop_size, win_size,
                            torch.hann_window(win_size).to(p.device),
                            return_complex=True)
        t_stft = torch.stft(t, fft_size, hop_size, win_size,
                            torch.hann_window(win_size).to(t.device),
                            return_complex=True)
        
        # 幅度谱 L1
        p_mag = p_stft.abs()
        t_mag = t_stft.abs()
        loss += (p_mag - t_mag).abs().mean()
        
        # log 幅度谱 L1（对低幅值区域更敏感）
        loss += (torch.log(p_mag + 1e-7) - torch.log(t_mag + 1e-7)).abs().mean()

    return loss / len(fft_sizes)
```

文献支撑：**Multi-Resolution STFT Loss**（Yamamoto et al., ICASSP 2020），原用于语音波形生成，被广泛移植到生理信号重建任务，因为两者都有局部高频瞬态结构。

---

#### Layer 2：形态项，用稳定的 Normalized Cross-Correlation 替换 Pearson

你现在的 `alpha_corr=1.2` 用 Pearson，梯度在 `std(pred)→0` 时爆炸，这是训练早期不稳定的根因之一。

用 NCC（Normalized Cross-Correlation），分母加 detach + clamp：

python

```python
def ncc_loss(pred, target, mask_point, eps=1e-6):
    """
    Normalized Cross-Correlation，返回 1 - NCC（越小越好）
    pred, target: (B, C, T)
    mask_point:   (B, C, T)
    """
    # 只在缺失区计算
    p = pred * mask_point
    t = target * mask_point
    n = mask_point.sum(dim=-1, keepdim=True).clamp(min=1)

    p_mean = p.sum(dim=-1, keepdim=True) / n
    t_mean = t.sum(dim=-1, keepdim=True) / n

    p_c = (p - p_mean) * mask_point
    t_c = (t - t_mean) * mask_point

    cov  = (p_c * t_c).sum(dim=-1)
    
    # 关键：分母 detach，避免梯度通过 std 传播导致爆炸
    p_std = p_c.pow(2).sum(dim=-1).clamp(min=eps).sqrt().detach()
    t_std = t_c.pow(2).sum(dim=-1).clamp(min=eps).sqrt().detach()

    ncc = cov / (p_std * t_std + eps)
    return (1 - ncc).mean()
```

文献支撑：**detach on denominator** 是 contrastive learning 里稳定 cosine similarity 梯度的标准做法，见 **SimSiam**（Chen & He, CVPR 2021）。直接用于 NCC 的做法见 **VoxelMorph**（Balakrishnan et al., TMI 2019）在医学图像配准中的实现，效果上显著优于直接 Pearson。

---

#### Layer 3：梯度连续性项，只惩罚缺失区边界

你现在的 `alpha_grad` 在全序列上算梯度 loss，这会惩罚已知区的正常 burst 边缘。应该只在缺失区及其边界附近计算：

python

```python
def boundary_gradient_loss(pred, target, mask_point, boundary_width=4):
    """
    只在缺失区边界 boundary_width 个点内惩罚梯度不连续。
    """
    # 找边界区域：mask 从 1→0 或 0→1 的转变点附近
    mask_diff = (mask_point[:, :, 1:] - mask_point[:, :, :-1]).abs()
    # 膨胀 boundary_width 步
    boundary = mask_diff
    for _ in range(boundary_width - 1):
        boundary = torch.maximum(boundary[:, :, 1:],
                                  boundary[:, :, :-1].clone())
        boundary = torch.cat([boundary, boundary[:, :, -1:]], dim=-1)
    boundary = torch.cat([boundary, boundary[:, :, -1:]], dim=-1)

    pred_grad   = (pred[:, :, 1:]   - pred[:, :, :-1]).abs()
    target_grad = (target[:, :, 1:] - target[:, :, :-1]).abs()
    grad_diff   = (pred_grad - target_grad).abs()

    return (grad_diff * boundary[:, :, :-1]).sum() / (boundary[:, :, :-1].sum() + 1e-8)
```

#### 组合与权重

```python
class EMGImputationLoss(nn.Module):
    def __init__(self, w_char=1.0, w_stft=0.3, w_ncc=0.5, w_grad=0.1):
        super().__init__()
        self.w_char = w_char
        self.w_stft = w_stft
        self.w_ncc  = w_ncc
        self.w_grad = w_grad

    def forward(self, pred, target, mask_point):
        """
        pred, target, mask_point: (B, C, T)
        mask_point: 1=缺失区，0=已知区
        """
        l_char = charbonnier(pred, target, mask_point)
        l_stft = multiscale_stft_loss(pred, target, mask_point)
        l_ncc  = ncc_loss(pred, target, mask_point)
        l_grad = boundary_gradient_loss(pred, target, mask_point)

        loss = (self.w_char * l_char +
                self.w_stft * l_stft +
                self.w_ncc  * l_ncc  +
                self.w_grad * l_grad)
        return loss, {
            'char': l_char.item(),
            'stft': l_stft.item(),
            'ncc':  l_ncc.item(),
            'grad': l_grad.item()
        }
```

**权重调参顺序**：先只开 `w_char`，训练 5 epoch 确认 loss 下降正常；再加 `w_ncc`；最后加 `w_stft` 和 `w_grad`。不要一开始就全开，因为各项的数值范围不同，全开容易被某一项主导。
# 课程学习方法
课程调度的完整实现如下：
```python
class CurriculumMaskScheduler:
    """
    按 epoch 动态调整 ScenarioMix 的场景权重和掩码强度。
    退出条件：val_corr_masked 达标 或 epoch 到期，取先到者。
    """
    def __init__(self, mask_gen, val_metric_fn):
        self.mask_gen = mask_gen
        self.val_metric_fn = val_metric_fn  # 返回 {'s1': corr, 's2': corr, ...}
        self.current_stage = 0

        self.stages = [
            {
                'name':        'warmup',
                'until_epoch': 10,
                'exit_cond':   lambda m: m.get('s1', 0) > 0.40,
                'weights':     {'s1': 1.0, 's2': 0.0, 's3': 0.0, 's4': 0.0},
                's4_params':   None,  # 不用 S4
            },
            {
                'name':        'introduce_s3_s4',
                'until_epoch': 18,
                'exit_cond':   lambda m: m.get('s4', 0) > 0.25,
                'weights':     {'s1': 0.6, 's2': 0.0, 's3': 0.2, 's4': 0.2},
                's4_params':   {'n_blocks': [1, 2], 'block_len_patches': [1, 2]},
            },
            {
                'name':        'ramp_up',
                'until_epoch': 28,
                'exit_cond':   lambda m: m.get('s4', 0) > 0.35,
                'weights':     {'s1': 0.4, 's2': 0.0, 's3': 0.3, 's4': 0.3},
                's4_params':   {'n_blocks': [1, 3], 'block_len_patches': [1, 3]},
            },
            {
                'name':        'introduce_s2',
                'until_epoch': 38,
                'exit_cond':   lambda m: m.get('s2', 0) > 0.20,
                'weights':     {'s1': 0.3, 's2': 0.1, 's3': 0.3, 's4': 0.3},
                's4_params':   {'n_blocks': [2, 3], 'block_len_patches': [2, 3]},
            },
            {
                'name':        'target',
                'until_epoch': 9999,
                'exit_cond':   lambda m: False,  # 不自动退出
                'weights':     {'s1': 0.32, 's2': 0.08, 's3': 0.22, 's4': 0.38},
                's4_params':   {'n_blocks': [2, 3], 'block_len_patches': [2, 4]},
            },
        ]

    def step(self, epoch, val_metrics: dict):
        """
        每个 epoch val 结束后调用。
        val_metrics: {'s1': corr_masked, 's2': ..., 's3': ..., 's4': ...}
        返回是否发生了阶段切换。
        """
        stage = self.stages[self.current_stage]
        should_advance = (
            epoch >= stage['until_epoch'] or
            stage['exit_cond'](val_metrics)
        )

        if should_advance and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self._apply_stage()
            return True  # 发生了切换，可以打 log
        return False

    def _apply_stage(self):
        stage = self.stages[self.current_stage]
        # 更新场景权重
        self.mask_gen.scenario_weights = stage['weights']
        # 更新 S4 掩码强度
        if stage['s4_params'] is not None:
            self.mask_gen.scenario_params['s4'].update(stage['s4_params'])

    def current_stage_name(self):
        return self.stages[self.current_stage]['name']
```

训练循环里的接入点：
```python
scheduler = CurriculumMaskScheduler(mask_gen, val_metric_fn)

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_metrics = evaluate_per_scenario(model, val_loader)  # 返回每场景 corr_masked
    
    advanced = scheduler.step(epoch, val_metrics)
    if advanced:
        print(f"Epoch {epoch}: curriculum → {scheduler.current_stage_name()}")
```
### 各阶段退出条件的依据

|阶段|退出条件|理由|
|---|---|---|
|warmup|S1 corr_masked > 0.40|模型能在最简单场景下有效重建，才有资格面对更难场景|
|introduce_s3_s4|S4 corr_masked > 0.25|S4 开始起效，说明时间连续缺失的基本重建能力已建立|
|ramp_up|S4 corr_masked > 0.35|S4 稳定，可以引入 S2（最难）|
|introduce_s2|S2 corr_masked > 0.20|S2 整通道缺失，0.20 是通道间协同关系开始被利用的信号|

这些阈值是保守估计。你当前 `corr_masked ≈ 0.012`，所以 warmup 阶段会跑满 10 epoch 才靠时间退出，这是正常的——说明课程设计是必要的。

# 迁移到健康人
### 具体需要预留的四个接口

#### 接口一：Domain Embedding（现在加，迁移时激活）

在位置编码之后，加一个域标识 embedding 的注入点，健康人训练时用零向量（等价于不存在），截肢患者时换成可学习的域向量：
```python
class AxialTransformerImputer(nn.Module):
    def __init__(self, ..., num_domains=1):
        ...
        # 域 embedding：健康人=domain 0，截肢=domain 1
        # 现在 num_domains=1，迁移时改成 2
        self.domain_embed = nn.Embedding(num_domains, D)
        nn.init.zeros_(self.domain_embed.weight)  # 健康人时全零，等价于无影响

    def forward(self, x, mask_patch, domain_id=None):
        ...
        tokens = tokens + self.temp_pos + self.chan_pos

        # 域注入：domain_id=None 或 0 时加零向量，不改变任何计算
        if domain_id is not None:
            d_emb = self.domain_embed(domain_id)        # (B, D)
            tokens = tokens + d_emb[:, None, None, :]   # 广播到 (B,C,N,D)

        tokens = self.local_conv(tokens)
        ...
```

迁移时：把 `num_domains` 改为 2，`domain_embed.weight[0]` 冻结（健康人域），只训练 `weight[1]`（截肢域）。

**参数开销**：`D=128` → 新增 128 个参数，可忽略不计。

#### 接口二：Channel Validity Mask（现在加，迁移时真正用起来）

截肢患者有「结构性坏通道」（长期低幅值/接触不良），和健康人的「人工掩码通道」不同——坏通道不应该参与 channel attention 的 key/value，但也不能直接删掉（通道索引要保持对齐）。

现在在 channel attention 里加一个 `channel_valid_mask` 参数，健康人训练时全为 True（所有通道有效）：
```python
def forward(self, x, temp_key_mask=None, chan_valid_mask=None):
    """
    chan_valid_mask: (B, C) bool，False=该通道不可信，不作为 key
                    健康人训练时传 None（等价于全 True）
    """
    B, C, N, D = x.shape

    # Channel attention
    xc = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
    xc_norm = self.chan_ln(xc)

    if chan_valid_mask is not None:
        # (B, C) → (B*N, C)，无效通道不作为 key
        ck = (~chan_valid_mask).unsqueeze(1).expand(-1, N, -1).reshape(B * N, C)
    else:
        ck = None

    xc_out, _ = self.chan_attn(xc_norm, xc_norm, xc_norm,
                               key_padding_mask=ck)
    ...
```

迁移时，`chan_valid_mask` 由你的「弱肌电/漏采/接触不良判定规则」生成，直接传入，不需要改模型结构。

#### 接口三：Adapter 层的插槽（现在加空壳，迁移时填充）

在每个 `GatedAxialBlock` 里预留一个 adapter 插槽，健康人训练时是恒等映射（零参数开销），迁移时替换为真正的 Adapter：
```python
class GatedAxialBlock(nn.Module):
    def __init__(self, D, n_heads, ffn_dim, dropout=0.1, use_adapter=False):
        ...
        # Adapter 插槽：健康人时是 None（恒等），迁移时替换
        self.adapter = (
            Adapter(D, bottleneck_dim=D // 4)
            if use_adapter else None
        )

    def forward(self, x, temp_key_mask=None, chan_valid_mask=None):
        ...
        # FFN 之后，adapter 之前
        x = x + self.ffn(self.ffn_ln(x))

        # 迁移时激活，健康人时跳过
        if self.adapter is not None:
            x = self.adapter(x)
        return x


class Adapter(nn.Module):
    """
    Bottleneck adapter：down-project → GELU → up-project → residual
    参数量 = 2 × D × bottleneck_dim，远小于主干
    """
    def __init__(self, D, bottleneck_dim=32):
        super().__init__()
        self.down = nn.Linear(D, bottleneck_dim)
        self.up   = nn.Linear(bottleneck_dim, D)
        self.act  = nn.GELU()
        self.ln   = nn.LayerNorm(D)
        # 初始化为近似恒等映射
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.ln(x))))
```

迁移时的操作：
```python
# 加载健康人预训练权重
model.load_state_dict(torch.load('healthy_pretrained.pt'))

# 冻结主干
for name, param in model.named_parameters():
    param.requires_grad = False

# 只在最后 1-2 个 block 激活 adapter，解冻
for block in model.blocks[-2:]:
    block.adapter = Adapter(D=128, bottleneck_dim=32)
    for param in block.adapter.parameters():
        param.requires_grad = True

# 解冻 pred_head 和 domain_embed
model.pred_head.requires_grad_(True)
model.domain_embed.weight.requires_grad_(True)
```

文献支撑：**Parameter-Efficient Transfer Learning**（Houlsby et al., ICML 2019），adapter 在小目标域数据（你有 11 个截肢被试）上的迁移效果系统性优于全量 fine-tune，原因是全量 fine-tune 在小数据集上会破坏预训练特征，而 adapter 只在残差路径上添加少量参数。
#### 接口四：无监督重建 loss 的预留（截肢患者没有真值）

健康人训练时，loss 在「人工掩码的缺失区」上算，真值已知。

截肢患者的坏通道/漏采区域没有真值。你需要提前设计一个**自监督 loss**，在健康人阶段就用来做辅助训练，这样迁移时这个 loss 已经收敛稳定。

具体做法：在健康人训练时，额外随机再掩掉一部分「已知区」，用模型对这部分的重建作为自监督信号：
```python
def forward_with_self_supervised_aux(model, x_clean, mask_binary, mask_gen):
    """
    x_clean:     (B, C, T) 干净信号
    mask_binary: (B, C, T) 主掩码（场景掩码）

    额外采一个轻量辅助掩码，只在主掩码的已知区内额外随机掩 10%
    用于自监督预训练，迁移时直接用这个 loss 处理无真值区域
    """
    # 只在已知区内再掩（不影响主任务）
    known_region = mask_binary  # 1=已知
    aux_mask = torch.bernoulli(known_region * 0.10)  # 已知区随机掩 10%
    combined_mask = mask_binary * (1 - aux_mask)     # 合并掩码

    pred = model(x_clean * combined_mask, combined_mask)

    # 主 loss：原始缺失区
    loss_main = criterion(pred, x_clean, 1 - mask_binary)
    # 辅助 loss：额外掩掉的区域（权重小，只是预热）
    loss_aux  = criterion(pred, x_clean, aux_mask) * 0.1

    return loss_main + loss_aux
```

迁移时，`loss_main` 消失（没有真值），只用 `loss_aux` 加上通道间一致性约束来训练。
### 迁移时的操作清单（现在不做，但心里有数）

```
Step 1：加载健康人权重，冻结全部参数
Step 2：激活最后 2 个 block 的 adapter（新增 ~8k 参数）
Step 3：解冻 domain_embed.weight[1]、pred_head
Step 4：用截肢患者数据，loss = self_supervised_aux_loss
Step 5：用 chan_valid_mask 传入判定好的坏通道，屏蔽其 key
Step 6：如果有部分有标注数据（少量截肢被试），混入监督 loss
```

总共新增可训练参数：`~8k adapter + 128 domain_embed + pred_head`，远小于 11 个被试能支撑的参数量。
# 迁移
现状：只覆盖「弱 + 高幅值」两条规则；漏采、接触不良未实现；无异常分类小网络。
数据：DB3 无异常人工标注 → 判定模块只能 无监督/规则，不适合直接上监督小 NN。
粒度：训练/补全主路径是 (T,C) 点级掩码 + patch 对齐；通道级 kill 是场景设计的一部分，不是独立判定器的最终输出形态。
## 异常检测
### 第一步：规则检测器
#### 数据挖掘
**永久死通道（物理断联）：**
- S06 Ch9、Ch10：全程平线，补全有意义
- S07 Ch9、Ch10：全程平线，补全有意义
**间歇性异常通道：**
- S03 Ch7：④游程 P25=2、P50=4、P75=7、P95=14，MAD P25=0（说明经常出现低 MAD 段），但 RMS 很高（P5=0.504）。这是典型的**间歇性断联**：信号大部分时间正常，但频繁出现短暂断线，规则一（漏采检测）会处理这个。
- S01 Ch1/Ch8：④ P95=7，偶发，规则一可以覆盖
**可选择性放进论文里的表述**
截肢患者表面肌电（sEMG）信号的异常模式与残缺特性分析
对 NinaPro DB3 数据集中 11 名截肢患者的 12 通道 sEMG 信号（采样率降至 200Hz，窗口大小 256，Patch 大小 8）进行多维度的量化统计，结果揭示了截肢患者肌电信号在实际采集场景中存在高度的非平稳性与严重的结构性残缺。数据主要呈现出以下三种典型的异常物理分布：
**1. 通道级物理断联（Dead Channels / Complete Failure）**
在多传感器高密度采集方案中，电极脱落或残肢局部组织严重萎缩会导致整个通道的信号彻底丢失。
- **数据实证：** 统计显示，受试者 S06 和 S07 的 Ch9 与 Ch10 在所有分位数（P5 至 P95）下的 RMS（均方根）和 MAD（平均绝对偏差）均为 0.0。更关键的是，在这两个通道上，最长低 MAD 游程（连续 MAD < 0.001 的 Patch 数）达到了窗口的物理上限 32。
- **分析：** 这意味着在长达 1.28 秒（256个采样点）的时间窗内，信号呈现绝对的“死线（Flatline）”状态。这种通道级的系统性缺失要求后续的补全算法必须具备跨通道拓扑推断的能力，而不能仅仅依赖单通道的时序上下文。
**2. 瞬态极弱信号与局部漏采（Transient Dropout & Weak Activation）**
截肢患者的残肢肌肉往往存在异常的激活模式，极易出现短暂的发力中断或瞬态电极接触不良，导致信号呈现碎片化的微弱状态。
- **数据实证：** 在 P5 分位数下，多名受试者的局部 MAD 降至绝对的 0.0（如 S01 的 Ch1、S03 的 Ch7、S08 的 Ch8），表明窗口内存在零星的平线区。从游程分布（P95）来看，S03 的 Ch7 出现了长达 14 个 Patch（约 560 毫秒）的连续极低变异区；S08 的 Ch11 也出现了长度为 9 的局部漏采。
- **分析：** 这种高随机性的时序局部残缺（Burst Dropout），正是传统均值插值或线性平滑方法失效的重灾区，它要求网络必须在不平滑健康高频突变的前提下，对局部空白进行上下文原域生成。
**3. 幅值过载与极端运动伪影（Signal Saturation & Clipping Artifacts）**
由于残肢形态的动态变化，电极与皮肤表面极易发生挤压或相对滑动，导致阻抗瞬间变化，产生极端的幅值突变或截断。
- **数据实证：** 峰值幅值（max|x|）分布显示了严重的过载现象。受试者 S07 在几乎所有有效通道的 P5 级别就已经达到了系统的物理上限（1.0000），说明信号存在持续性的饱和截断（Clipping）。此外，即使是整体 RMS 较低的受试者（如 S03 和 S09），在其 P95 峰值分布中也频繁出现幅值触顶（1.0000）的现象。
- **分析：** 幅值的极度两极分化（要么死线，要么触顶饱和）证明了基于简单全局阈值或 MSE（均方误差）的重建模型会受到离群值的严重干扰。这也为引入 Charbonnier Loss（对大误差呈线性宽容）和 NCC 形态学约束提供了坚实的数据支撑。
**总结：**
截肢患者 sEMG 信号不仅面临常规的时域高频噪声，更掺杂了通道级坏死（长时窗全掩码）、瞬态漏采（局部块掩码）以及幅值过载（极值失真）等复杂的复合型残缺。这表明，在走向临床应用之前，必须构建一种非对称且能够保持局部高频物理烙印的全分辨率上下文推断架构，以应对这种高度异构的数据缺失。
既然真实的残缺模式既包含长达 1 秒以上的“整个通道坏死”，也包含几十毫秒的“局部碎片化漏采”，你在训练 Axial Transformer 时，打算如何设计 Mask 的生成策略（比如块掩码、随机掩码、特定通道掩码的混合比例）来精准模拟并覆盖这两种截然不同的物理分布。
#### 规则设计
```
离线预计算（每被试运行一次）：
  θ_dead[c]  = mean_over_windows(RMS[c]) < 0.01
  θ_weak[c]  = percentile_5(RMS[c]) × 0.5

在线检测（每窗口）：
  Step 0：死通道屏蔽
    if θ_dead[c]: mask[:, c] = 0，跳过后续检测

  Step 1：漏采检测（patch 级）
    for each patch p:
      MAD[p] = mean|x[p×8:(p+1)×8] - mean(...)|
    找连续 MAD < 1e-3 的游程
    if 游程长度 ≥ 4: 对应 patch 的 mask = 0

  Step 2：弱肌电检测（窗口级）
    RMS_window = sqrt(mean(x²))
    if RMS_window < θ_weak[c]: mask[:, c] = 0

  Step 3：高幅值检测（暂时关闭）
    原因：③数据显示 P25 大量通道已达 1.0，触顶是正常激活现象
    暂不启用，避免误判

  输出：mask_rule (T=256, C=12)，patch 对齐
```
### 第二步：1D-CNN Autoencoder
#### 输入设计
每次送入**单通道单窗口**，shape `(1, T=256)`。
不做跨通道，原因：各通道的正常幅值范围不同，跨通道会让 AE 的重建误差尺度混乱，阈值难以统一。每通道独立检测，阈值也独立校准。
#### 架构
轻量是核心要求，参数量控制在 **~10k** 以内：
```
Encoder:
  Conv1d(1,  16, k=8, s=4)  → (16, 64)   # patch 级降采样
  GELU
  Conv1d(16, 32, k=4, s=2)  → (32, 31)
  GELU
  Conv1d(32, 8,  k=4, s=2)  → (8,  14)   # bottleneck，压缩比 256→14×8=112
  GELU

Decoder:
  ConvTranspose1d(8,  32, k=4, s=2)  → (32, 30)
  GELU
  ConvTranspose1d(32, 16, k=4, s=2)  → (16, 62)
  GELU
  ConvTranspose1d(16, 1,  k=8, s=4)  → (1,  256) 或需 crop/pad 对齐
```
参数量约 **8k**，单次前向 < 1ms，适合在线判定。
#### 训练
```
数据：DB2 全量 28 被试的训练窗口，逐通道拆开
     → 28 × ~1875窗/被试 × 12通道 ≈ 630k 单通道样本
Loss：MSE(reconstruction, input)，无标签
训练：batch=256, lr=1e-3, epoch=30, AdamW
     无需 curriculum，无需掩码，就是普通 AE 训练
```
#### 推理时的阈值校准
阈值不能全局固定，因为不同截肢患者的残肢 EMG 幅值差异大。
按被试、按通道独立校准：
```
对每个截肢患者，取第一关规则过滤后的「相对纯净」片段（mask_rule=1 的窗口）
送入 AE，计算重建误差分布
θ_ae[patient][channel] = 该通道重建误差的 95th percentile
推理时：reconstruction_error > θ_ae[patient][channel] → 异常
```
#### 和 MCIA 的接口
```
输入窗口 (B, T, C)
  ↓ 第一关规则 → mask_rule (B, T, C)
  ↓ 第二关 AE → mask_ae (B, T, C)
  ↓ 合并：mask_final = mask_rule * mask_ae  （两关都通过才是已知区）
  ↓ 送入 MCIA：model(x * mask_final, raw_time_mask=mask_final)
  ↓ 推理回填：pred * (1-mask_final) + x * mask_final
```
`chan_valid_mask` 接口（迁移接口二）直接从 `mask_final` 派生：
```python
chan_valid_mask = (mask_final.mean(dim=1) > 0.5)  # (B, C)，通道超过一半时间有效才算有效通道
```
# 评估体系
### 三层评估体系

**第一层：检测器自身的质量（间接评估）**

不评估补全质量，只评估检测器的行为是否合理：

```
指标一：异常检测率 per channel per subject
  = 被判定为异常的时间点占比
  合理范围：5%~30%（视患者情况）
  若某通道 >60% 被判异常 → 该通道可能是永久性坏通道，应整通道屏蔽
  若全部通道 <1% 被判异常 → 检测器阈值太松，几乎没检测到任何东西

指标二：检测结果的时间连续性
  异常段的平均长度（patch 数）
  真实接触不良通常是连续的（几十到几百毫秒），不是随机散点
  若检测结果是大量 1-2 个 patch 的随机散点 → 检测器在误报正常信号
```

**第二层：补全信号的内部一致性（无 GT 也能算）**
利用 12 通道 EMG 之间的肌肉协同关系：
```
指标三：通道间协同一致性（Synergy Consistency）
  正常 EMG 的相邻通道之间有稳定的相关结构（拮抗肌群的协同模式）
  补全前：计算含伪影信号的通道间相关矩阵 R_before
  补全后：计算补全信号的通道间相关矩阵 R_after
  比较 R_after 是否更接近健康人的典型相关结构 R_healthy
  
  具体：用 DB2 健康人数据计算 R_healthy（28被试均值）
       R_after 与 R_healthy 的 Frobenius 距离越小越好
```

这个指标不需要 GT，只需要健康人作为参考分布。

**第三层：实验三的后验评估
```
指标四：关节角度估计的 RMSE / Pearson corr
  原始含伪影信号 → TCN → 角度估计（baseline，你已有）
  补全后信号 → TCN → 角度估计
  
  若补全后的角度估计显著优于原始信号 → 说明补全有效去除了伪影
  这是最有说服力的间接评估
```

# 图片规划
## 实验一：MCIA 在健康人 DB2 上的补全质量
**Figure 1：架构图（方法图，放在 Method 节）**
整体 pipeline 示意：输入含缺失的 EMG → PatchEmbed → LocalConvBypass → 4× GatedAxialBlock（双轴 attention 示意）→ pred_head → 补全输出。标注 mask_token、temp_pos/chan_pos、domain_embed、adapter 插槽的位置。
**Figure 2：补全波形可视化（定性结果）**
选 2-3 个典型窗口，每图展示 4 行：原始信号、掩码位置、模型补全结果、Ground Truth。缺失区用阴影标注，已知区直接复制（mse_known=0）。覆盖 S1/S4 两种场景各一个样本，展示散点掩码和连续段掩码的补全效果。
**Figure 3：场景对比柱状图（定量结果）**
X 轴：S1/S2/S3/S4 四个场景。Y 轴：corr_masked。三组柱子：MCIA（我们）/ TimeMAE / Cubic Spline。误差棒用被试间标准差。
**Figure 4：训练收敛曲线**
X 轴：epoch。Y 轴左：train loss 各分量（char/ncc/stft）。Y 轴右：val corr_masked（S1/S2/S3/S4 四条线）。标注 curriculum stage 切换点（垂直虚线）。
**Table 1：定量指标汇总表**
行：各 test 被试（S33-S40）+ 均值±std。列：corr_masked / corr_masked_partial / RMSE / MAE（缺失区）。对比三个方法：MCIA / TimeMAE / Cubic Spline。
## Experiment 3: continuous Key10 angle estimation after EMG completion
**Figure 7: Key10 trajectory comparison**
Select dynamic test windows and show the fixed ten MCP/IP/PIP channels: thumb MCP/IP plus MCP/PIP pairs for index, middle, ring, and little fingers. Plot Ground Truth and A/B/C on the same test-local windows.
**Figure 8: grouped comparison**
Compare global, MCP, and PIP/IP Pearson correlation for A/B/C. Error bars use between-subject standard deviation.
**Figure 9: subject-level improvement scatter**
X-axis: Group A global correlation. Y-axis: Group B minus Group A global correlation.
**Table 3: Key10 quantitative results**
Rows: subjects plus mean and standard deviation. Report global, MCP, and PIP/IP metrics for A/B/C.
## 一张贯穿全文的总览图
**Figure 0（Introduction 或 Abstract 图）**
整条 pipeline 的流程图：
```
DB3 原始 EMG（含伪影）
  ↓ 规则检测器（Step0/1/2）
  ↓ MCIA 补全（DB2 预训练 → DB3 adapter 微调）
  ↓ Enhanced EMG
  ↓ TCN 角度估计
  → 连续关节角度
```

左侧展示含伪影的原始波形，右侧展示补全后的干净波形，底部是角度曲线对比。这张图是论文的「门面」，审稿人第一眼看的。
## 优先级建议
全量训练结束跑完所有实验后，按以下顺序制图：
Figure 0 → Table 1 → Figure 3 → Figure 8/9（这四个是审稿人最关注的）→ Figure 2/7（定性可视化）→ 其余。
# 最新结果

## 补全效果图
![[whole_ch_rank01_idx8.png]]

![[whole_ch_rank02_idx1.png]]

![[whole_ch_rank03_idx10.png]]

![[whole_ch_rank04_idx2.png]]

![[whole_ch_rank05_idx4.png]]

![[whole_ch_rank06_idx0.png]]

## 架构图
![[模型架构图.png]]

![[迁移学习.png]]
![[实验设计.png]]

# 表格
## 实验一：MCIA 在健康人 DB2 上的补全质量（8 名 held-out 被试，zero-shot）

训练集：S01–S28（28人）；验证集：S29–S32；测试集：S33–S40。指标在缺失区计算。

| **测试被试**     | **缺失区相关系数corr_masked** | **缺失区相关系数（有锚点通道）corr_masked_partial** | **包络相关系数EnvCorr** | **均方误差MSE**         |
| ------------ | ---------------------- | ------------------------------------- | ----------------- | ------------------- |
| **S33**      | 0.233                  | 0.236                                 | 0.751             | 0.0208              |
| **S34**      | 0.330                  | 0.329                                 | 0.795             | 0.0217              |
| **S35**      | 0.348                  | 0.348                                 | 0.771             | 0.0169              |
| **S36**      | 0.417                  | 0.412                                 | 0.817             | 0.0165              |
| **S37**      | 0.300                  | 0.297                                 | 0.778             | 0.0196              |
| **S38**      | 0.360                  | 0.359                                 | 0.782             | 0.0173              |
| **S39**      | 0.273                  | 0.277                                 | 0.745             | 0.0189              |
| **S40**      | 0.360                  | 0.358                                 | 0.809             | 0.0204              |
| **均值 ± 标准差** | **0.328 ± 0.055**      | **0.327 ± 0.054**                     | **0.781 ± 0.024** | **0.0190 ± 0.0018** |

> _注：corr_whole（回填后全段相关）均值 = 0.718，因已知区直接复制导致数值偏高，不作为主要评估指标。_

## 实验一：与 baseline 方法对比

测试集 8 名被试的均值。注意：TimeMAE 和 Cubic Spline 的 Corr 为 corr_whole，与 MCIA 的 corr_masked 口径不同。

| **方法**           | **相关系数（Corr）** | **口径说明**        | **均方误差（MSE）** |
| ---------------- | -------------- | --------------- | ------------- |
| **MCIA（本文）**     | 0.328          | 缺失区 corr_masked | 0.0190        |
| **TimeMAE**      | 0.564          | 全段 corr_whole   | 0.0759        |
| **Cubic Spline** | 0.195          | 全段 corr_whole   | ~10M（非归一化）    |

> _注：公平对比需统一口径。TimeMAE 的 corr_masked（待补充）预计低于其 corr_whole。_

## Experiment 3: fixed Key10 continuous angle estimation (DB3)

A/B use the same KinematicTCN architecture, repetition-based train/validation/test split, Key10 target, and evaluation procedure. The only group difference is the EMG representation: raw or healthy-prior enhanced.

The fixed target is the zero-based CyberGlove subset `[1, 2, 4, 5, 7, 8, 11, 12, 15, 16]`: thumb MCP/IP plus MCP/PIP pairs for index, middle, ring, and little fingers.

| **Subset** | **Group A** | **Group B** |
| ---------- | ----------- | ----------- |
| **Global Key10** | pending new Key10 run | pending new Key10 run |
| **MCP** | pending new Key10 run | pending new Key10 run |
| **PIP/IP** | pending new Key10 run | pending new Key10 run |

> The old 22-dimensional figures and values are historical artifacts only and are not reported as current results. New reports additionally contain a dynamic subset for trace-quality diagnosis; the full Key10 test set remains the primary result.

## 规则检测器：DB3 各被试异常检测率

由 RuleAnomalyDetector 检测后置零的比例（含死通道、漏采、弱肌电）。

| **被试**  | **masked_ratio** | **主要异常类型**     |
| ------- | ---------------- | -------------- |
| **S01** | 14.40%           | 偶发漏采           |
| **S02** | 14.97%           | 偶发漏采           |
| **S03** | 21.52%           | Ch7 间歇性断联      |
| **S04** | 13.10%           | 偶发漏采           |
| **S05** | 15.09%           | 偶发漏采           |
| **S06** | 23.26%           | Ch9/Ch10 永久死通道 |
| **S07** | 26.35%           | Ch9/Ch10 永久死通道 |
| **S08** | 15.85%           | 偶发漏采           |
| **S09** | 13.75%           | 偶发漏采           |
| **S10** | 12.62%           | 偶发漏采           |
| **S11** | 14.61%           | 偶发漏采           |
| **均值**  | **16.05%**       | —              |

> _注：当前规则检测仅覆盖硬件层故障，接触不良/运动伪影未检测，是实验三提升空间有限的主要原因。AE 第二关检测器预计可将 masked_ratio 提升至 25–35%。_
# 分析
## gemini说不行
直接给你一个明确的结论：**在目前的主流顶会和高水平期刊中，几乎没有任何文章直接利用 NinaPro DB3（截肢患者数据集）去做连续的“关节角度估计（Continuous Joint Angle Estimation / Kinematics Regression）”。**

绝大多数（99%以上）使用 DB3 数据集的论文，都在做**离散的手势动作分类（Discrete Gesture Classification / Pattern Recognition）**。

之所以会出现这种“研究盲区”，并不是因为大家没想到，而是因为在 DB3 上做连续角度回归，存在一个**底层物理逻辑上的死结**：

### 1. “Ground Truth” 的错位悖论

在 NinaPro 的采集协议中：

* **健康人（如 DB2 等）：** 数据手套（CyberGlove）戴在发力的那只手上，sEMG 电极也贴在同一只手上。电信号和关节角度是**物理同源、完美同步**的。做连续回归非常合理。
* **截肢患者（DB3）：** 患者的残肢已经没有了手掌和手指，根本无法戴数据手套。因此，DB3 的采集方案是：让患者在健侧手（完整手）**戴上手套，然后大脑想象双手同时做镜像运动（Bilateral Mirrored Movements），系统记录健侧手的角度作为 Ground Truth，同时采集**患侧残肢（断肢）的 sEMG。
* **致命问题：** 残肢表面的肌电信号，和对侧健手的关节角度，在生物力学上是**高度解耦**的。残肢肌肉往往发生了严重的萎缩或代偿，且由于没有实际的物理反馈，双侧运动的相位和幅度存在巨大的天然延迟与误差。如果你强行用回归网络（如 LSTM、Transformer）去将残肢 sEMG 映射到对侧手的连续角度曲线上，模型学到的全是噪声和错位的伪映射，根本无法收敛出有物理意义的连续曲线。

### 2. 肌肉信息源的永久性缺失

连续的关节角度估计（尤其是多指独立运动的连续估计），高度依赖控制特定关节的局部肌肉（例如指浅屈肌、指深屈肌）。
而在经桡骨截肢（Transradial Amputation，DB3 中的主要群体）患者中，控制手腕和手指的关键远端肌肉群要么已经被切除，要么已经与其他肌肉缝合发生了解剖结构的改变。仅靠残存的近端肌肉包络，可以勉强“猜”出患者想做一个宏观的“握拳”动作（这也是为什么分类任务能做到 80% 以上的准确率），但想要精准回归出食指弯曲了 35 度还是 45 度，几乎是天方夜谭。

### 3. 目前学术界是怎么处理的？

如果你在文献中看到关于“连续关节角度/运动轨迹估计”的文章：

* **基于 NinaPro：** 作者几乎清一色使用的是 **DB2、DB5、DB7** 等健康受试者数据集，因为这些数据集具备物理同源的真实 Kinematics 标签。
* **针对截肢患者：** 如果非要对截肢患者做连续估计，前沿团队通常不会去估计“关节角度”，而是去估计“连续的手指发力意图（Neural-Drive / Fingertip Force）”。他们通常会自己采集高密度肌电（HD-sEMG），或者通过在肌电传感器中结合超声成像（Ultrasound），直接解析运动单元的放电频率，而不是去强行拟合一个虚假的对侧手套角度。

**总结来说：** 不用在 DB3 的连续角度估计上死磕，这是一条由于数据集先天采集物理限制而被学术界战略性放弃的路线。如果是需要做连续估计/回归的 Benchmark，建议直接转向 DB2 等健康受试者数据集；如果是处理 DB3 数据，将其作为离散分类任务或底层特征提取底座是更为合理的学术道路。

## Fixed Key10 decision
The DB3 glove remains a contralateral proxy label, so the experiment does not claim reconstruction of a missing limb. The adopted downstream target is a fixed, literature-aligned Key10 subset rather than all 22 glove channels.

Source glove indices are zero-based and fixed: `[1, 2, 4, 5, 7, 8, 11, 12, 15, 16]`.

- Thumb MCP and Thumb IP
- Index MCP and Index PIP
- Middle MCP and Middle PIP
- Ring MCP and Ring PIP
- Little MCP and Little PIP

All A/B/C groups use the same single KinematicTCN, the same repetition split, and the same Key10 target. Full Key10 metrics are the main result; the dynamic subset is a supplementary trace-quality analysis only.

The source 22-channel glove array is retained only for loading and normalizing its original channels before the fixed Key10 selection. New main-experiment artifacts carry target metadata and reject legacy 22-dimensional reports or predictions.
### 回答你的问题：有没有信心
**有信心证明「MCIA 补全改善下游任务」这个核心 claim。**
没有信心的是「在 DB3 上做出 SOTA 级别的高精度连续角度估计」，但这不是你的目标。
你的实验三是一个**消融/验证实验**，不是一个独立的角度估计 paper。它的功能是佐证 MCIA 补全的实用价值，不需要角度估计本身达到 SOTA。
# 其他
### 为什么时序生成常用 Encoder-Decoder？
我本地跑有5090的GPU，再加上我可以在云服务器上跑，上面保底有4个gpu。
但是我对于你提出的架构有疑问（不是质疑，而是想知道你怎么考虑的），大家常用编码器解码器的架构去做时序的生成式补全，为什么我们不选择这个，而是去做双轴attention。
Encoder-Decoder 会对**整个序列**（包括已知区）做压缩再重建。这带来两个问题：
**问题一：已知区的重建是浪费的计算，还会引入重建误差。** 你的 `mse_known=0` 说明模型已经学会「已知区直接 copy」——这正是你不应该用 encoder-decoder 的信号，因为模型在用解码器容量做一件不需要的事。
**问题二：bottleneck 会丢失缺失区重建所需的局部时序细节。** 这就是你 pool 层的问题——pool 本质上就是一个手工设计的 encoder bottleneck。
文献支撑：**SAITS**（Du et al., ESWA 2023）明确对比了 Seq2Seq encoder-decoder 和直接在原始时间步上做 attention 的方案，结论是后者在 imputation 上显著更好，原因正是 bottleneck 的信息损失。
因为 forecasting 的输入（历史）和输出（未来）**天然不对齐**：
```
输入：t=1..L（已知）
输出：t=L+1..L+H（未知，需要生成）
```

Encoder 压缩历史，Decoder 自回归生成未来——这里 bottleneck 是合理的，因为你本来就不需要保留输入的局部细节，只需要保留「趋势/周期」等全局信息。
你的任务不是这个。你的输出时间轴和输入时间轴**完全重合**，只是部分位置有缺失。
