"""Train one GeomFrac with sea-Z and neural 32-state heat heads.

Train/validation files must be complete physical time histories for the SAME
cold FV anchor and ROM basis.  The full S5 BFZI teacher can be used after its
per-geometry cold anchors and ROM bases are generated.  This trial does not
train a cross-geometry spatial decoder.

Example:
  python -m examples.bfzi_geom_neural --anchor runs/pose_anchor.npz \
    --rom-file runs/pose_rom.npz --train runs/train_a.npz runs/train_b.npz \
    --val runs/val_a.npz --out runs/geomfrac_et.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_memory import G_SCALE, geom_feat, heat_drive, thermal_summary
from sdfmpneo_bfzi.geom_neural import GeomFracET, rollout
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.s5_bridge import feature, teacher_temp


def read(path, rom):
    """Read one complete history; use file.npz#i for batched BFZI teachers."""
    spec = str(path)
    name, sep, tail = spec.rpartition("#")
    idx = int(tail) if sep and tail.isdigit() else None
    file = Path(name) if idx is not None else Path(spec)
    with np.load(file) as d:
        t = d["time_s"].copy()
        temp = d["temp_K"].copy()
        z6 = d["z6"].copy()
        if temp.ndim == 3:
            if idx is None:
                raise ValueError(f"batched teacher needs #history_index: {file}")
            if idx >= len(temp):
                raise ValueError(f"teacher history index out of range: {spec}")
            temp = temp[idx]
            z6 = z6[idx]
        elif idx not in (None, 0):
            raise ValueError(f"single-history teacher has no case #{idx}: {file}")
        if "voltage_v" in d:
            volts = d["voltage_v"].copy()
            if volts.ndim == 2:
                if idx is None:
                    raise ValueError(f"batched voltage requires #history_index: {file}")
                volts = volts[idx]
        else:
            key = "currents_A" if "currents_A" in d else "current_A"
            currents = d[key].copy()
            if currents.ndim == 3:
                if idx is None:
                    raise ValueError(f"batched currents need #history_index: {file}")
                currents = currents[idx]
            volts = np.array([((zmat(zz)+rom.ext)@ii)[0].real
                              for zz, ii in zip(z6, currents)])
    temp = teacher_temp(temp, rom)
    if (t.ndim != 1 or t[0] != 0 or np.any(np.diff(t) <= 0)
            or temp.shape != (len(t), len(rom.cap))
            or z6.shape != (len(t), 6) or volts.shape != t.shape):
        raise ValueError(f"teacher dimensions / time grid mismatch: {spec}")
    if not np.allclose(z6[0], rom.z0, atol=1e-7, rtol=1e-6):
        raise ValueError(f"teacher cold BFZI impedance differs from anchor: {spec}")
    temp_rise = temp-rom.amb
    a = (temp_rise*rom.cap) @ rom.phi
    cu0 = rom.port.copper(np.full(2, rom.amb))
    dz = []
    src = []
    qproj = []
    for k in range(len(t)):
        z = zmat(z6[k])
        cu = rom.port.copper(temp[k, rom.cu])
        sea = z-zmat(rom.z0)-np.diag(cu-cu0)
        dz.append([sea[0,0].real, sea[0,0].imag,
                   sea[0,1].real, sea[0,1].imag,
                   sea[1,1].real, sea[1,1].imag])
        src.append(heat_drive(float(volts[k]), z, rom.ext, rom.port,
                              temp[k,rom.cu], rom.cap[rom.cu]))
        I = np.linalg.solve(z+rom.ext, np.array([volts[k], 0.], complex))
        sea0 = z-1j*rom.port.omega*rom.port.air-np.diag(cu)
        heat, *_ = rom.source(temp[k]-rom.amb, I, sea0)
        qproj.append(rom.phi.T @ heat)
    temp_sum = thermal_summary(temp, rom.ns, rom.npkg, rom.vol, rom.amb)
    y = np.column_stack((np.asarray(dz), temp_sum))
    return dict(time=t, voltage=volts, src=np.asarray(src), y=y, a=a,
                q=np.asarray(qproj), z6=z6, temp_K=temp, file=spec)


def scales(rows):
    sy = np.concatenate([r["y"][1:] for r in rows])
    su = np.concatenate([r["src"][1:] for r in rows])
    sa = np.concatenate([r["a"][1:] for r in rows])
    yscale = np.maximum(np.std(sy, axis=0), np.r_[np.full(6, 1e-8), np.full(8, 1e-3)])
    sscale = np.maximum(np.max(abs(su), axis=0), 1e-4)
    ascale = np.maximum(np.std(sa, axis=0), 1e-3)
    return sscale, yscale, ascale


def score(net, rows, g, sx, sy, sa, closed=False, rom=None, therm_w=0.):
    total = []
    for row in rows:
        t = row["time"]
        u = torch.as_tensor(row["src"]/sx, dtype=torch.float32)
        y = torch.as_tensor(row["y"]/sy, dtype=torch.float32)
        a = torch.as_tensor(row["a"], dtype=torch.float32)
        asc = torch.as_tensor(sa, dtype=torch.float32)
        h = net.init_state(1)
        pr_z, pr_a = [], []
        pr_phys = []
        cur = a[0:1]
        if therm_w:
            if rom is None:
                raise ValueError("thermal physics loss requires a heat ROM")
            K = torch.as_tensor(rom.K, dtype=a.dtype)
            q = torch.as_tensor(row["q"], dtype=a.dtype)
            # Teacher FV trajectories include truncation and time-sampling
            # error. Compare relative heat residual, not the absolute ROM
            # residual that the teacher itself does not satisfy exactly.

        for k in range(1, len(t)):
            prev = cur if closed else a[k-1:k]
            h, zhat, nxt = net.step_et(h, u[k:k+1], float(t[k]-t[k-1]), g,
                                       prev, asc)
            pr_z.append(zhat[0])
            pr_a.append((nxt[0]-a[k])/asc)
            if therm_w:
                dt = float(t[k]-t[k-1])
                res = (nxt[0]-prev[0])/dt+K @ nxt[0]-q[k]
                ref = (a[k]-a[k-1])/dt+K @ a[k]-q[k]
                scale = (asc*(1/dt+K.abs().sum(dim=1))).clamp_min(1e-6)
                pr_phys.append(((res-ref)/scale).square().mean())
            if closed:
                cur = nxt
        if not pr_z:
            continue
        pz = torch.stack(pr_z)
        pa = torch.stack(pr_a)
        zerr = ((pz-y[1:])**2).mean()
        zextra = ((pz[:, :6]-y[1:, :6])**2).mean()
        heat = pa.square().mean()
        phy = torch.stack(pr_phys).mean() if therm_w else 0.
        total.append(zerr+0.45*zextra+heat+therm_w*phy)
    if not total:
        raise ValueError("teacher histories need at least two times")
    return torch.stack(total).mean()


def eval_rollout(rom, net, ck, rows):
    """Evaluate a genuinely autonomous physics-coupled held-out history."""
    result = []
    net.eval()
    with torch.no_grad():
        for row in rows:
            pred = rollout(rom, net, ck, row["voltage"], row["time"])
            real_z = row["z6"]
            real_t = row["temp_K"]
            dz = np.linalg.norm(real_z-real_z[0])
            result.append(dict(
                file=row["file"],
                dynamic_Z_rel_L2=float(np.linalg.norm(pred["z6"]-real_z)
                                       / max(dz, 1e-12)),
                full_Z_rel_L2=float(np.linalg.norm(pred["z6"]-real_z)
                                    / max(np.linalg.norm(real_z), 1e-12)),
                temperature_max_K=float(np.max(abs(pred["temp_K"]-real_t))),
                copper_max_K=float(np.max(abs(pred["temp_K"][:, rom.cu]
                                               -real_t[:, rom.cu]))),
                package_max_K=float(np.max(abs(
                    pred["temp_K"][:, rom.ns:rom.cu.start]
                    -real_t[:, rom.ns:rom.cu.start]))),
                material_budget_max_W=float(np.max(abs(pred["material_budget_W"]))),
                fv_residual_max_W=float(np.max(abs(pred["fv_node_residual_W"])))))
    return result



def load_init(net, old):
    """Retain a trained 40D GeomFrac when adding five S5 material inputs."""
    if not old.get("port_sea", False):
        raise ValueError("initial GeomFrac must predict seawater delta Z")
    ng = int(old["ng"])
    if ng not in (net.ng, net.ng-5):
        raise ValueError("initial GeomFrac has incompatible geometry dimension")
    src = old["state"]
    state = net.state_dict()
    for name, val in src.items():
        if name not in state:
            continue
        if state[name].shape == val.shape:
            state[name] = val.detach().clone()
        elif name == "net.0.weight" and ng == net.ng-5:
            state[name] = torch.cat((val, val.new_zeros((val.shape[0],5))), dim=1)
        else:
            raise ValueError(f"incompatible GeomFrac weight: {name}")
    net.load_state_dict(state)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--rom-file", type=Path, required=True)
    p.add_argument("--train", type=str, nargs="+", required=True)
    p.add_argument("--val", type=str, nargs="+", required=True)
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/geomfrac_et.pt"))
    p.add_argument("--report", type=Path, default=Path("runs/geomfrac_et.json"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=70)
    p.add_argument("--closed-weight", type=float, default=0.2,
                   help="free-running thermal-state loss, with teacher power inputs")
    p.add_argument("--therm-weight", type=float, default=0.03,
                   help="physics residual of the 32-state heat equation during training")
    a0 = p.parse_args()
    if set(a0.train) & set(a0.val):
        p.error("train and validation histories must be distinct")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    head = GeomHeat.load(a0.anchor)
    rom = GeomHeatROM.load(head, a0.rom_file)
    train = [read(f, rom) for f in a0.train]
    val = [read(f, rom) for f in a0.val]
    sx, sy, sa = scales(train)
    ng = 45 if hasattr(rom,"thermal") else (
        40 if rom.pose is not None else len(G_SCALE))
    g = torch.tensor(feature(rom, ng)[None], dtype=torch.float32)
    net = GeomFracET(ng=g.shape[-1], rank=rom.rank)
    if a0.init_model:
        old = torch.load(a0.init_model, map_location="cpu", weights_only=True)
        try:
            load_init(net, old)
        except ValueError as exc:
            p.error(str(exc))
    opt = torch.optim.Adam(net.parameters(), lr=a0.lr)
    best, saved, best_ep = float("inf"), None, 0
    for ep in range(a0.epochs):
        opt.zero_grad(set_to_none=True)
        loss = score(net, train, g, sx, sy, sa,
                     rom=rom, therm_w=a0.therm_weight)
        if a0.closed_weight and ep >= 20:
            loss = loss+a0.closed_weight*score(
                net, train, g, sx, sy, sa, closed=True,
                rom=rom, therm_w=a0.therm_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step()
        net.eval()
        with torch.no_grad():
            v = float(score(net, val, g, sx, sy, sa,
                            rom=rom, therm_w=a0.therm_weight)
                      +a0.closed_weight*score(net, val, g, sx, sy, sa,
                          closed=True, rom=rom, therm_w=a0.therm_weight))
        if v < best-1e-8:
            best, best_ep = v, ep+1
            saved = {k: x.detach().clone() for k, x in net.state_dict().items()}
        if ep-best_ep >= a0.patience:
            break
        net.train()
    net.load_state_dict(saved)
    ck = dict(state=net.state_dict(), ng=net.ng, rank=net.rank,
              src_scale=torch.as_tensor(sx, dtype=torch.float32),
              y_scale=torch.as_tensor(sy, dtype=torch.float32),
              a_scale=torch.as_tensor(sa, dtype=torch.float32), port_sea=True,
              architecture="GeomFracET", train_files=[r["file"] for r in train],
              val_files=[r["file"] for r in val])
    a0.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a0.out)
    free_val = eval_rollout(rom, net, ck, val)
    result = dict(architecture="GeomFracET", rank=net.rank, ng=net.ng,
                  best_epoch=best_ep, val_loss=best,
                  training_count=len(train), validation_count=len(val),
                  thermal_physics_loss_weight=a0.therm_weight,
                  teacher="stored BFZI full-temperature and impedance trajectories",
                  physics="neural sea delta + neural 32-state step + five budgets",
                  autonomous_validation=free_val,
                  status="experimental; held-out physical acceptance pending")
    a0.report.parent.mkdir(parents=True, exist_ok=True)
    a0.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
