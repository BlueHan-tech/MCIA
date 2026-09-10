"""打印 MCIA 各子模块参数量。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paper_pipeline import build_mcia, flatten_pipeline_config, load_yaml_config


def main():
    cfg = load_yaml_config(PROJECT_ROOT)
    config = flatten_pipeline_config(cfg)
    model = build_mcia(config, "cpu")

    rows = []
    child_param_ids = {id(p) for c in model.children() for p in c.parameters()}

    for name, child in model.named_children():
        if name == "blocks":
            for i, block in enumerate(child):
                rows.append((f"blocks[{i}]", sum(p.numel() for p in block.parameters())))
        else:
            rows.append((name, sum(p.numel() for p in child.parameters())))

    for n, p in model.named_parameters():
        if id(p) not in child_param_ids:
            rows.append((n, p.numel()))

    total = sum(c for _, c in rows)
    w = max(len(n) for n, _ in rows)
    print(
        "MCIA 参数量明ϸ "
        f"(embed_dim={config['embed_dim']}, n_layers={config['n_layers']}, "
        f"n_heads={config['n_heads']}, ffn_dim={config['ffn_dim']})"
    )
    print("-" * (w + 18))
    for name, cnt in rows:
        print(f"{name:<{w}} -> {cnt:>10,}")
    print("-" * (w + 18))
    print(f"{'TOTAL':<{w}} -> {total:>10,}")


if __name__ == "__main__":
    main()
