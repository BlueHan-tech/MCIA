"""候选优化选项的合成数据 smoke：不改主链路默认行为。

覆盖：范围惩罚损失、规则对齐掩码生成器、包络分支零初始化兼容性、
patch 边界淡化、多步精炼、MC-Dropout 不确定性门控补全。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import numpy as np
import torch

from models.completion.mask_generators import RuleAlignedMaskGenerator, ScenarioMixMaskGenerator
from models.completion.mcia_core import MCIA
from utils.loss_functions import EMGImputationLoss
from utils.paper_pipeline import (
    _linear_gap_fill,
    _patch_boundary_crossfade,
    complete_with_mask,
    complete_with_mask_uncertainty,
)

torch.manual_seed(0)
B, T, C, P = 2, 256, 12, 8


def build_model(**kwargs) -> MCIA:
    return MCIA(window_size=T, n_channels=C, patch_size=P,
                embed_dim=32, n_layers=2, n_heads=4, ffn_dim=64, dropout=0.1, **kwargs)


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_range_penalty() -> None:
    pred = torch.rand(B, T, C) * 2.0          # 一半样本超 1
    target = torch.rand(B, T, C)
    mask = (torch.rand(B, T, C) > 0.3).float()  # 30% 缺失
    off, parts_off = EMGImputationLoss(w_aux=0.0, range_penalty_weight=0.0)(pred, target, mask)
    on, parts_on = EMGImputationLoss(w_aux=0.0, range_penalty_weight=0.5)(pred, target, mask)
    check("range penalty reported (main branch)", "range_penalty_loss" in parts_off)
    check("range penalty zero when disabled", float(parts_off["range_penalty_loss"]) == 0.0)
    check("range penalty activates", float(parts_on["range_penalty_loss"]) > 0.0)
    check("range penalty changes total", not torch.allclose(off, on))
    _, parts_inr = EMGImputationLoss(w_aux=0.0, range_penalty_weight=0.5)(torch.rand(B, T, C), target, mask)
    check("range penalty zero in range", float(parts_inr["range_penalty_loss"]) == 0.0)
    _, parts_empty = EMGImputationLoss(w_aux=0.0, range_penalty_weight=0.5)(pred, target, torch.ones(B, T, C))
    check("range penalty reported (empty-mask branch)", "range_penalty_loss" in parts_empty)


def test_rule_aligned_mask() -> None:
    base = ScenarioMixMaskGenerator(n_channels=C, time_steps=T, patch_size=P,
                                    rng=np.random.default_rng(0))
    bank = base.generate_batch_masks(16, device="cpu").numpy().transpose(0, 2, 1)  # -> (M,T,C) 1=observed
    gen = RuleAlignedMaskGenerator(mask_bank=bank, patch_size=P, rng=np.random.default_rng(1))
    out = gen.generate_batch_masks(8, n_channels=C, time_steps=T, device="cpu")
    arr = out.numpy()
    check("rule mask binary", np.isin(arr, (0.0, 1.0)).all())
    check("rule mask API signature", out.shape == (8, T, C))
    single = gen._resample_single()
    check("resample is shift/copy of bank member",
          any(np.array_equal(single, np.roll(m, s, axis=0))
              for m in bank for s in range(0, T, P)))
    try:
        RuleAlignedMaskGenerator(mask_bank=bank * 0.5)
        check("non-binary bank rejected", False)
    except ValueError:
        check("non-binary bank rejected", True)


def test_envelope_branch_zero_init() -> None:
    base = build_model().eval()
    x = torch.rand(B, T, C)
    mask = (torch.rand(B, T, C) > 0.3).float()
    with torch.no_grad():
        base_pred = base(x, raw_time_mask=mask)
    with_env = build_model(use_envelope_branch=True).eval()
    missing, unexpected = with_env.load_state_dict(base.state_dict(), strict=False)
    check("envelope branch only adds proj keys",
          set(missing) == {"envelope_branch.proj.weight", "envelope_branch.proj.bias"} and not unexpected)
    with torch.no_grad():
        env_pred = with_env(x, raw_time_mask=mask)
    check("zero-init envelope branch = identity", torch.allclose(base_pred, env_pred, atol=1e-6))
    nn_proj = with_env.envelope_branch.proj
    torch.nn.init.normal_(nn_proj.weight, std=0.02)
    with torch.no_grad():
        env_pred2 = with_env(x, raw_time_mask=mask)
    check("trained envelope branch changes output", not torch.allclose(env_pred2, base_pred, atol=1e-6))


def test_completion_options() -> None:
    model = build_model().eval()
    x = torch.rand(B, T, C)
    mask = (torch.rand(B, T, C) > 0.3).float()

    with torch.no_grad():
        legacy = model(x * mask, x_masked=x * mask, raw_time_mask=mask).clamp(0, 1)
        legacy_out = legacy * (1 - mask) + x * mask
        default_out = complete_with_mask(model, x, mask)
    check("complete_with_mask default unchanged", torch.allclose(legacy_out, default_out, atol=1e-6))
    check("observed copy-back exact", torch.allclose(default_out * mask, x * mask))

    with torch.no_grad():
        smooth_out = complete_with_mask(model, x, mask, patch_boundary_smooth=True, patch_size=P)
        refine_out = complete_with_mask(model, x, mask, refine_steps=2)
    check("boundary smooth runs and differs", not torch.allclose(smooth_out, default_out))
    check("refine runs and differs", not torch.allclose(refine_out, default_out))
    boundary_cols = np.zeros(T, dtype=bool)
    boundary_cols[P - 1::P] = True
    boundary_cols[P::P] = True
    non_boundary = torch.from_numpy(~boundary_cols).float().view(1, T, 1)
    check("smooth preserves non-boundary completed samples",
          torch.allclose(smooth_out * (1 - mask) * non_boundary,
                         default_out * (1 - mask) * non_boundary))

    unc_out = complete_with_mask_uncertainty(model, x, mask, n_samples=4, std_gate=0.0)
    check("uncertainty copy-back exact", torch.allclose(unc_out * mask, x * mask))
    check("uncertainty bounded", float(unc_out.min()) >= 0.0 and float(unc_out.max()) <= 1.0)
    model.train()
    unc_out2 = complete_with_mask_uncertainty(model, x, mask, n_samples=3, std_gate=1.0)
    check("uncertainty restores training mode", model.training)
    check("uncertainty bounded in train mode", float(unc_out2.max()) <= 1.0)
    model.eval()

    gap = np.random.rand(3, 20, 2).astype(np.float32)
    gap_mask = np.ones((3, 20, 2), dtype=np.float32)
    gap_mask[:, 5:10, 0] = 0.0
    gap_mask[:, :, 1] = 0.0                       # 整通道缺失 -> 回退预测值
    filled = _linear_gap_fill(gap, gap_mask)
    expected = np.stack([
        np.interp(np.arange(5, 10),
                  (known := np.flatnonzero(gap_mask[b, :, 0] >= 0.5)),
                  gap[b, known, 0])
        for b in range(gap.shape[0])
    ])
    check("gap fill interpolates observed anchors", np.allclose(filled[:, 5:10, 0], expected, atol=1e-6))
    check("gap fill keeps observed samples", np.allclose(filled[gap_mask >= 0.5], gap[gap_mask >= 0.5]))
    check("gap fill full-missing falls back", np.allclose(filled[:, :, 1], gap[:, :, 1]))
    boundary = _patch_boundary_crossfade(torch.ones(1, T, C), P)
    check("boundary crossfade keeps bounds", float(boundary.max()) <= 1.0 and float(boundary.min()) >= 0.0)


def main() -> None:
    test_range_penalty()
    test_rule_aligned_mask()
    test_envelope_branch_zero_init()
    test_completion_options()
    print("all candidate-option smokes passed")


if __name__ == "__main__":
    main()
