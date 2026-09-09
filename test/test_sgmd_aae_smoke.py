"""Short architecture and loss smoke for the SGMD-AAE literature baseline."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.baselines.sgmd_aae import SGMDAAEConfig, SGMDAAEGenerator, SGMDMultiViewDiscriminator, SGMDAAEObjective, sgmd_complete

def main():
    torch.manual_seed(42)
    target = torch.rand(2, 1, 240, 12)
    mask = torch.ones_like(target); mask[:, :, 30:90, 2:5] = 0
    generator, discriminator = SGMDAAEGenerator(), SGMDMultiViewDiscriminator()
    output = generator(target * mask, mask)
    if output.shape != target.shape: raise AssertionError(output.shape)
    objective = SGMDAAEObjective(SGMDAAEConfig())
    total, pieces = objective.generator_loss(output, target, mask, discriminator)
    if not torch.isfinite(total): raise AssertionError("non-finite generator loss")
    total.backward()
    delivered = sgmd_complete(generator, target, mask)
    if not torch.equal(delivered[mask.bool()], target[mask.bool()]): raise AssertionError("observations changed")
    print("ok", {key: round(float(value), 5) for key, value in pieces.items()})
if __name__ == "__main__": main()
