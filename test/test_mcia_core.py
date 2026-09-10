"""MCIA（带轴向注意力的多通道补全）冒烟测试。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.completion.mcia_core import MCIA, derive_ch_mask_from_sample_mask


def test_mcia_forward_and_copy_contract():
    B, T, C = 2, 256, 12
    x = torch.rand(B, T, C)
    raw_mask = torch.ones(B, T, C)
    raw_mask[:, 64:96, 0] = 0.0
    raw_mask[:, :, 3] = 0.0
    x_masked = x * raw_mask
    ch_mask = derive_ch_mask_from_sample_mask(raw_mask)

    model = MCIA(
        window_size=T,
        n_channels=C,
        patch_size=8,
        embed_dim=128,
        n_layers=4,
        n_heads=4,
        ffn_dim=256,
        dropout=0.1,
    )
    pred = model(x_masked, mask=ch_mask, raw_time_mask=raw_mask)

    assert pred.shape == x.shape
    assert torch.isfinite(pred).all()
    assert model.num_patches == 32
    assert model.mask_token.shape == (1, 1, 1, 128)
    assert model.uncond_token.shape == (1, 1, 1, 128)
    assert model.temp_pos.shape == (1, 1, 32, 128)
    assert model.chan_pos.shape == (1, C, 1, 128)


if __name__ == "__main__":
    test_mcia_forward_and_copy_contract()
    print("[OK] MCIA forward")
