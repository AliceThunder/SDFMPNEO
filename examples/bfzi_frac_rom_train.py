"""Train fractional SOE and exponential baselines on BFZI thermal ROM targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.frac_memory import FracNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--poses", nargs="+", default=("P03", "P07"))
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    report = {}
    for pose in a.poses:
        key = pose.lower()
        with np.load(a.runs / f"bfzi_frac_rom_{key}.npz") as d:
            t = d["time_s"].copy()
            v = d["voltage_v"].copy()
            y = np.column_stack((d["thermal_state"], d["delta_z6"]))
        # Causal source is the known drive; two additional channels encode
        # switching and accumulated electrical exposure.
        vref = max(float(np.max(abs(v))), 1e-12)
        u = np.column_stack((v/vref, (v/vref)**2,
                             np.r_[0., np.diff(v)]/vref))
        scale = np.maximum(np.max(abs(y), axis=0), 1e-9)
        target = torch.tensor(y/scale, dtype=torch.float32)[None]
        src = torch.tensor(u, dtype=torch.float32)[None]
        results = {}
        for mode in ("frac", "exp"):
            torch.manual_seed(a.seed)
            net = FracNet(nout=y.shape[1], mode=mode, hidden=96)
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            best = float("inf")
            best_state = None
            for ep in range(a.epochs):
                opt.zero_grad()
                pred = net(src, t)
                loss = torch.mean((pred-target)**2)
                loss.backward()
                opt.step()
                value = float(loss.detach())
                if value < best:
                    best = value
                    best_state = {k: x.detach().clone()
                                  for k, x in net.state_dict().items()}
            net.load_state_dict(best_state)
            with torch.no_grad():
                pred = net(src, t)[0].numpy()*scale
            dz = y[:, -6:]
            dz_err = float(np.linalg.norm(pred[:, -6:]-dz) /
                           max(np.linalg.norm(dz), 1e-30))
            state_err = float(np.linalg.norm(pred[:, :-6]-y[:, :-6]) /
                              max(np.linalg.norm(y[:, :-6]), 1e-30))
            path = a.runs / f"bfzi_frac_net_{key}_{mode}.pt"
            torch.save(dict(state=net.state_dict(), mode=mode,
                            nout=y.shape[1], scale=scale, vref=vref,
                            geometry=pose), path)
            results[mode] = dict(train_loss=best, delta_Z_rel_L2=dz_err,
                                 thermal_state_rel_L2=state_err,
                                 model_file=str(path))
            print(pose, mode, json.dumps(results[mode]), flush=True)
        report[pose] = results
    (a.runs / "bfzi_frac_train.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
