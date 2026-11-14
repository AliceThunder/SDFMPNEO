"""C stage: BFZI-ROM neural residual training and independent-drive acceptance."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from examples.bfzi_obb_subcell_study import measure
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.frac_memory import FracNet
from sdfmpneo_bfzi.frac_rom import FracROM
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.heat_rom import HeatROM
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D

def build_et(bf, v150, runs, pose):
    stem = f"bfzi_pose_{pose.lower()}"
    meta = json.loads((runs / f"{stem}_anchor.json").read_text())
    geo = build_args(meta)
    old = json.loads((runs / "bfzi_obb_physical_study.json").read_text())
    geo["shift_mm"] = np.asarray(old["geometries"][pose]["new_shift_mm"], float)
    prev = GeomHeat.load(runs / f"{stem}_disjoint_anchor.npz")
    xz, y = GRIDS["extra"]
    data, _ = create(bf, v150, "M4", **geo, sea_void="oriented",
                     physical_box_only=True, grid_xz=xz, grid_y=y)
    em = BFZIV148Adapter(data, bf, heat_mode="port-weak",
                         fv_faces="exact", source_mask="subcell")
    return BFZIV148Skin3D(em, external=prev.ext, pkg_shape=(4, 2, 4),
                          pkg_k=.25, skin_depth=.002, skin_layers=8,
                          skin_ratio=1.4, skin_quad=2, skin_near=4,
                          skin_map=2, pkg_sea_h=20., skin_heat="source")

def input_drive(v, vref):
    x = v/vref
    return np.column_stack((x, x*x, np.r_[0., np.diff(x)]))

def train(runs, pose, fr, rom, epochs, seed):
    with np.load(runs / f"bfzi_frac_rom_{pose.lower()}.npz") as d:
        t = d["time_s"].copy()
        v = d["voltage_v"].copy()
        s = d["thermal_state"].copy()
        dz = d["delta_z6"].copy()
        z0 = d["z0"].copy()
    if not np.allclose(z0, fr.z0, rtol=1e-7, atol=1e-7):
        raise ValueError("training BFZI and cached cold impedance differ")
    if s.shape[1] != rom.rank:
        raise ValueError("training/physical ROM rank mismatch")
    y = dz - np.array([pack(fr.impedance(rom, row)[0])-fr.z0 for row in s])
    scale = np.maximum(np.max(abs(y), axis=0), 1e-10)
    vref = max(float(np.max(abs(v))), 1e-12)
    src = torch.as_tensor(input_drive(v, vref)[None], dtype=torch.float32)
    target = torch.as_tensor((y/scale)[None], dtype=torch.float32)
    out = {}
    for mode in ("frac", "exp"):
        torch.manual_seed(seed)
        net = FracNet(nout=6, mode=mode, hidden=96)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        best = np.inf
        state = None
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.mean((net(src, t)-target)**2)
            loss.backward()
            opt.step()
            val = float(loss.detach())
            if val < best:
                best = val
                state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        net.load_state_dict(state)
        ck = dict(state=state, mode=mode, hidden=96, vref=vref,
                  scale=scale, pose=pose)
        f = runs / f"bfzi_frac_rom_{pose.lower()}_{mode}.pt"
        torch.save(ck, f)
        with torch.no_grad():
            fitted = net(src, t)[0].numpy()*scale
        err = np.linalg.norm(fitted-y)/max(np.linalg.norm(dz), 1e-30)
        out[mode] = dict(train_residual_delta_Z_rel_L2=float(err),
                         train_loss=float(best), model_file=str(f))
        print("TRAIN", pose, mode, json.dumps(out[mode]), flush=True)
    return out

def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = FracNet(nout=6, mode=ck["mode"], hidden=ck["hidden"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck

def drives(t, v0):
    v1 = np.full_like(t, .65*v0)
    v1[(t >= 240.) & (t < 330.)] = 0.
    v2 = np.full_like(t, .85*v0)
    v2[(t >= 90.) & (t < 210.)] = 0.
    v2[(t >= 360.) & (t < 420.)] = 0.
    return {"H1_validation": v1, "H2_test": v2}

def metrics(ref, pred, et, elapsed, direct):
    dz = ref["z6"]-ref["z6"][0]
    pz = pred["z6"]-pred["z6"][0]
    diff = pred["temp_K"]-ref["temp_K"]
    pkg = slice(et.count, et.count+2*et.pkg_n)
    smean = np.array([et.sea_temp(row) for row in diff])
    return dict(delta_Z_rel_L2=float(np.linalg.norm(pz-dz) /
                                    max(np.linalg.norm(dz), 1e-30)),
                pkg_max_abs_K=float(np.max(abs(diff[:, pkg]))),
                copper_max_abs_K=float(np.max(abs(diff[:, et.cu]))),
                sea_mean_max_abs_K=float(np.max(abs(smean@(
                    et.model.vol / et.model.vol.sum())))),
                max_total_budget_W=float(np.max(pred["total_budget_W"])),
                infer_sec=float(elapsed), direct_sec=float(direct),
                speedup=float(direct/max(elapsed, 1e-9)))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--poses", nargs="+", choices=("P03", "P07"),
                   default=("P03", "P07"))
    p.add_argument("--epochs", type=int, default=650)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--infer-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    a.runs.mkdir(parents=True, exist_ok=True)
    bf, v150 = originals(a.bfzi_root)
    result_file = a.runs / "bfzi_frac_rom_eval.json"
    report = json.loads(result_file.read_text()) if result_file.exists() else {}
    for pose in a.poses:
        start = time.perf_counter()
        et = build_et(bf, v150, a.runs, pose)
        rom = HeatROM.load(a.runs/f"bfzi_heat_rom_{pose.lower()}.npz")
        f = a.runs/f"bfzi_frac_tangent_{pose.lower()}.npz"
        cached = f.exists()
        if cached:
            fr = FracROM.load(f)
        else:
            fr = FracROM.build(et, rom)
            fr.save(f)
        print("PREPARED", pose, "sec", round(time.perf_counter()-start, 3),
              "BFZI_tangent_solves", 0 if cached else rom.rank, flush=True)
        item = dict(offline_file=str(f), train={})
        if not a.infer_only:
            item["train"] = train(a.runs, pose, fr, rom, a.epochs, a.seed)
        with np.load(a.runs/f"bfzi_frac_rom_{pose.lower()}.npz") as d:
            t, v0 = d["time_s"].copy(), float(np.max(abs(d["voltage_v"])))
        item["heldout"] = {}
        for label, vv in drives(t, v0).items():
            ref = measure(et, t, vv)
            tests = {}
            for mode in ("physics", "frac", "exp"):
                if mode == "physics":
                    net, ck = None, None
                else:
                    net, ck = load_net(a.runs / (
                        f"bfzi_frac_rom_{pose.lower()}_{mode}.pt"))
                t0 = time.perf_counter()
                pred = fr.rollout(rom, t, vv, net, ck)
                elapsed = time.perf_counter()-t0
                tests[mode] = metrics(ref, pred, et, elapsed, ref["direct_sec"])
                print("CHECK", pose, label, mode,
                      json.dumps(tests[mode]), flush=True)
            item["heldout"][label] = tests
            report[pose] = item
            (a.runs/"bfzi_frac_rom_eval.json").write_text(
                json.dumps(report, indent=2))
        print("DONE", pose, flush=True)

if __name__ == "__main__":
    main()
