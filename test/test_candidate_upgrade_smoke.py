"""候选优化能力的合成数据 smoke。

覆盖：范围惩罚损失、规则对齐掩码生成器、包络分支零初始化兼容性、
已采纳的 patch 边界淡化交付规则（complete_with_mask 默认行为）。
已否决并删除的路径（多步自精炼、MC-Dropout 不确定性门控）不在此列。
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
from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask
from utils.loss_functions import EMGImputationLoss
from utils.paper_pipeline import (
    complete_with_mask,
    patch_boundary_crossfade,
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


def test_completion_delivery_rule() -> None:
    model = build_model().eval()
    x = torch.rand(B, T, C)
    mask = (torch.rand(B, T, C) > 0.3).float()
    mask_1d = derive_ch_mask_from_sample_mask(mask)

    with torch.no_grad():
        raw_pred = model(x * mask, mask=mask_1d, x_masked=x * mask,
                         raw_time_mask=mask).clamp(0, 1)
        unsmoothed = raw_pred * (1 - mask) + x * mask
        default_out = complete_with_mask(model, x, mask)

    boundary_cols = np.zeros(T, dtype=bool)
    boundary_cols[P - 1::P] = True
    boundary_cols[P::P] = True
    non_boundary = torch.from_numpy(~boundary_cols).float().view(1, T, 1)
    check("delivery keeps non-boundary completed samples",
          torch.allclose(default_out * (1 - mask) * non_boundary,
                         unsmoothed * (1 - mask) * non_boundary, atol=1e-6))
    check("delivery smooths boundary completed samples",
          not torch.allclose(default_out * (1 - mask), unsmoothed * (1 - mask)))
    check("observed copy-back exact", torch.allclose(default_out * mask, x * mask))
    check("delivery bounded", float(default_out.min()) >= 0.0 and float(default_out.max()) <= 1.0)

    constant = torch.full((1, T, C), 0.3)
    check("crossfade constant invariant",
          torch.allclose(patch_boundary_crossfade(constant, P), constant, atol=1e-7))
    check("crossfade keeps bounds",
          float(patch_boundary_crossfade(torch.rand(2, T, C), P).min()) >= 0.0)
    odd = torch.rand(1, T + 1, C)
    check("crossfade returns non-aligned lengths unchanged",
          torch.equal(patch_boundary_crossfade(odd, P), odd))


def main() -> None:
    test_range_penalty()
    test_rule_aligned_mask()
    test_envelope_branch_zero_init()
    test_completion_delivery_rule()
    print("all candidate-option smokes passed")


if __name__ == "__main__":
    main()
