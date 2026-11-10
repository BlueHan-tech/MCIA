"""协议 §5 C 组（截肢者供体先验）开发筛查。

主入口训练 2 kHz 供体模型。Adapter 类仅供历史 200 Hz 报告读取 checkpoint。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path
ensure_current_env_dll_path()

import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


if __name__ == "__main__":
    from utils.waveform_adaptation import train_donor_priors

    parser = argparse.ArgumentParser(description='Adapt the 2 kHz prior using other DB3 donors')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    train_donor_priors(PROJECT_ROOT, smoke=args.smoke, audit_only=args.audit_only)
