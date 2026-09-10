"""Run one literature completion baseline on a supplied (N,T,C) .npy tensor.

The caller is responsible for supplying the paper-specific data representation.
For SGMD-AAE this is 240 x 12, 120-ms non-overlapping, 2-kHz DB2 windows.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from models.baselines.cp_wopt import CPWOPTConfig, complete_cp_wopt, relative_mean_error
from models.baselines.sgmd_aae import SGMDAAEConfig, SGMDAAEGenerator, SGMDMultiViewDiscriminator, SGMDAAEObjective, sgmd_complete

def mask_like(values, ratio, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(values.shape) >= ratio).astype(np.float32)

def scores(pred, target):
    mse = float(np.mean((pred-target)**2)); rmse = float(np.sqrt(mse)); peak = max(float(np.max(np.abs(target))), 1e-8)
    return {"rmse": rmse, "nrmse": rmse/peak, "psnr": float(20*np.log10(peak/max(rmse,1e-12)))}

def train_sgmd(train, test, ratio, epochs, batch_size, seed, device):
    torch.manual_seed(seed); generator, discriminator = SGMDAAEGenerator().to(device), SGMDMultiViewDiscriminator().to(device)
    objective = SGMDAAEObjective(SGMDAAEConfig()); opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3); opt_d = torch.optim.SGD(discriminator.parameters(), lr=2e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_index, (batch,) in enumerate(loader):
            # SGMD-AAE Table 1 treats each sample as (time=240, channels=12).
            target = batch.to(device).unsqueeze(1)
            mask_seed = seed + epoch * len(loader) + batch_index
            observed = torch.from_numpy(mask_like(batch.numpy(), ratio, mask_seed)).to(device).unsqueeze(1)
            with torch.no_grad(): fake = generator(target*observed, observed)
            opt_d.zero_grad(); objective.discriminator_loss(discriminator(target), discriminator(fake)).backward(); opt_d.step()
            opt_g.zero_grad(); output = generator(target*observed, observed); loss, _ = objective.generator_loss(output, target, observed, discriminator); loss.backward(); opt_g.step()
    observed = mask_like(test, ratio, seed+999)
    x = torch.from_numpy(test).to(device).unsqueeze(1)
    m = torch.from_numpy(observed).to(device).unsqueeze(1)
    pred = sgmd_complete(generator.eval(), x, m).squeeze(1).cpu().numpy()
    return pred, observed

def main():
    p = argparse.ArgumentParser(); p.add_argument("--method", choices=("sgmd_aae","cp_wopt"), required=True); p.add_argument("--input", required=True); p.add_argument("--missing-ratio", type=float, default=.3); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--rank", type=int, default=8); p.add_argument("--max-samples", type=int, default=512); a=p.parse_args()
    run = os.environ.get("MCIA_RUN_DIR");
    if not run: raise RuntimeError("MCIA_RUN_DIR is required for this single-step baseline run.")
    values = np.load(a.input).astype(np.float32)
    if a.max_samples > 0:
        values = values[:a.max_samples]
    if values.ndim != 3: raise ValueError("Input .npy must have shape (N,T,C).")
    indices = np.random.default_rng(a.seed).permutation(len(values))
    split = max(1, int(.8*len(values)))
    train, test = values[indices[:split]], values[indices[split:]]
    if not len(test): raise ValueError("Need at least two input windows.")
    if a.method == "sgmd_aae":
        if values.shape[1:] != (240,12): raise ValueError("Faithful SGMD-AAE requires (N,240,12) input.")
        pred, observed = train_sgmd(train, test, a.missing_ratio, a.epochs, a.batch_size, a.seed, "cuda" if torch.cuda.is_available() else "cpu")
    else:
        observed = mask_like(test, a.missing_ratio, a.seed); result = complete_cp_wopt(test*observed, observed, CPWOPTConfig(rank=a.rank, seed=a.seed)); pred = result.reconstruction.astype(np.float32)
    missing = observed < .5; report = {"method":a.method,"input_shape":list(values.shape),"train_test":"deterministic_random_80_20_window_split","missing_ratio":a.missing_ratio,"masked_metrics":scores(pred[missing],test[missing]),"masked_rme":relative_mean_error(pred[missing],test[missing])}
    out=Path(run)/"06_diagnostics"/"literature_baselines"; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/f"{a.method}_test_completion.npz", target=test,prediction=pred,observed_mask=observed); (out/f"{a.method}_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
