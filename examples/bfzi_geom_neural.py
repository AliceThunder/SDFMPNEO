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
import time
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
    spec = str(path).replace("\\", "/")
    name, sep, tail = spec.rpartition("#")
    idx = int(tail) if sep and tail.isdigit() else None
    file = Path(name) if idx is not None else Path(spec)
    with np.load(file) as d:
        t = d["time_s"].copy()
        temp = d["temp_K"].copy()
        z6 = d["z6"].copy()
        pkg_ref = d["Tpkg_K"].copy() if "Tpkg_K" in d.files else None
        cu_ref = d["Tcu_K"].copy() if "Tcu_K" in d.files else None
        if temp.ndim == 3:
            if idx is None:
                raise ValueError(f"batched teacher needs #history_index: {file}")
            if idx >= len(temp):
                raise ValueError(f"teacher history index out of range: {spec}")
            temp = temp[idx]
            z6 = z6[idx]
            if pkg_ref is not None: pkg_ref = pkg_ref[idx]
            if cu_ref is not None: cu_ref = cu_ref[idx]
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
    if pkg_ref is not None and not np.allclose(
            temp[:,rom.ns:rom.cu.start],pkg_ref,atol=1e-9,rtol=0):
        raise ValueError(f"teacher package cells differ from S5 anchor: {spec}")
    if cu_ref is not None and not np.allclose(
            temp[:,rom.cu],cu_ref,atol=1e-9,rtol=0):
        raise ValueError(f"teacher copper cells differ from S5 anchor: {spec}")
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
    for k in range(len(t)):
        z = zmat(z6[k])
        cu = rom.port.copper(temp[k, rom.cu])
        sea = z-zmat(rom.z0)-np.diag(cu-cu0)
        dz.append([sea[0,0].real, sea[0,0].imag,
                   sea[0,1].real, sea[0,1].imag,
                   sea[1,1].real, sea[1,1].imag])
        if k:
            prev = zmat(z6[k-1])
            src.append(heat_drive(float(volts[k-1]), prev, rom.ext, rom.port,
                                  temp[k-1,rom.cu], rom.cap[rom.cu]))
        else:
            src.append(np.zeros(3))
    temp_sum = thermal_summary(temp, rom.ns, rom.npkg, rom.vol, rom.amb)
    y = np.column_stack((np.asarray(dz), temp_sum))
    return dict(time=t, voltage=volts, src=np.asarray(src), y=y, a=a,
                z6=z6, temp_K=temp, file=spec)


def scales(rows):
    sy = np.concatenate([r["y"][1:] for r in rows])
    su = np.concatenate([r["src"][1:] for r in rows])
    sa = np.concatenate([r["a"][1:] for r in rows])
    yscale = np.maximum(np.std(sy, axis=0), np.r_[np.full(6, 1e-8), np.full(8, 1e-3)])
    sscale = np.maximum(np.max(abs(su), axis=0), 1e-4)
    ascale = np.maximum(np.std(sa, axis=0), 1e-3)
    return sscale, yscale, ascale


def _port_src(row, rom, a, y, sy, voltage):
    """Differentiable causal circuit source from previous predicted Z and Tcu."""
    if "port_t" not in row:
        from sdfmpneo_bfzi.geom_canon import mode_scale
        from sdfmpneo_bfzi.geom_train import zmat as ztorch
        cap=rom.cap[rom.cu]
        amb=rom.amb
        copper=np.asarray(rom.port.copper(np.full(2,amb)),complex)
        tangent=(rom.port.copper(np.full(2,amb+.1))
                 -rom.port.copper(np.full(2,amb-.1)))/.2
        sc=mode_scale(rom,rom.rank) if 'K' in row else np.ones(rom.rank)
        from sdfmpneo_bfzi.geom_rom import groups
        cap_phi=rom.cap[:,None]*rom.phi
        h_phi=rom.H@rom.phi
        b=np.stack([cap_phi[sl].sum(axis=0) for sl in groups(rom)])*sc[None,:]
        d=np.stack([h_phi[sl].sum(axis=0) for sl in groups(rom)])*sc[None,:]
        row["port_t"]=dict(
            B=torch.as_tensor(b,dtype=torch.float32),
            D=torch.as_tensor(d,dtype=torch.float32),
            phi=torch.as_tensor(rom.phi*sc[None,:],dtype=torch.float32),
            cold=ztorch(torch.tensor(rom.z0[None],dtype=torch.float32)),
            cu=torch.as_tensor(copper,dtype=torch.complex64),
            deriv=torch.as_tensor(tangent,dtype=torch.complex64),
            cu_map=torch.as_tensor(rom.phi[rom.cu]*sc[None,:],
                                   dtype=torch.float32),
            sea_air=torch.as_tensor(1j*rom.port.omega*rom.port.air,
                                    dtype=torch.complex64),
            ext=torch.as_tensor(rom.ext,dtype=torch.complex64),
            cap=torch.as_tensor(cap,dtype=torch.float32))
    from sdfmpneo_bfzi.geom_train import zmat as ztorch
    p=row["port_t"]
    tc=a@p["cu_map"].T
    cu=p["cu"][None]+p["deriv"][None]*tc
    dz=ztorch(y[:,:6]*torch.as_tensor(sy[:6],dtype=y.dtype))
    z=p["cold"]+dz+torch.diag_embed(cu-p["cu"])
    sea=p["cold"]+dz-p["sea_air"]-torch.diag_embed(p["cu"])
    v=torch.as_tensor(voltage,dtype=a.dtype).reshape(1)
    drive=torch.stack((v,torch.zeros_like(v)),dim=-1).to(torch.complex64)
    current=torch.linalg.solve(z+p["ext"],drive)
    qsea=(.5*torch.einsum("bi,bij,bj->b",
                         current.conj(),sea,current).real).clamp_min(0)
    qcu=.5*cu.real*current.abs().square()/p["cap"]
    return torch.cat((qsea[:,None],qcu),dim=-1)


def _heat_balance(row, rom, old, pred, y, sy, dt, voltage):
    """Differentiable matching budgets plus nonnegative spatial projection."""
    a=pred
    # Each adjustment to zero-mean modes requires a new material budget step.
    for stage in range(6):
        for _ in range(6):
            q=_port_src(row,rom,a,y,sy,voltage)
            p=row["port_t"]
            qg=torch.stack((q[:,0],torch.zeros_like(q[:,0]),
                            torch.zeros_like(q[:,0]),
                            q[:,1]*p["cap"][0],q[:,2]*p["cap"][1]),dim=-1)
            err=((a-old)@p["B"].T)/dt+a@p["D"].T-qg
            jac=p["B"][:,:5]/dt+p["D"][:,:5]
            corr=torch.linalg.solve(jac,-err.T).T
            a=torch.cat((a[:,:5]+corr,a[:,5:]),dim=-1)
        if stage==5 or rom.rank<=5:
            break
        base=a[:,:5]@p["phi"][:,:5].T
        diff=a[:,5:]@p["phi"][:,5:].T
        ratio=torch.where(diff<0,
                          base.clamp_min(0)/(-diff).clamp_min(1e-12),
                          torch.full_like(diff,float("inf")))
        beta=ratio.min(dim=1).values.clamp(max=1.)
        if bool(torch.all(beta>=1.-1e-9)):
            break
        fac=torch.where(beta<1.-1e-9,beta*.999,torch.ones_like(beta))
        a=torch.cat((a[:,:5],a[:,5:]*fac[:,None]),dim=-1)
    return a

def _hist_mean(vals, spans, mode):
    """Average history losses equally or by observed interval count."""
    v=torch.stack(vals)
    if mode=="equal":
        return v.mean()
    if mode!="step":
        raise ValueError("history weighting must be equal or step")
    w=v.new_tensor(spans)
    return (v*w).sum()/w.sum()


def _zrel_full(rom, row, sea, cu, sy):
    """Relative dynamic full-port impedance, including copper temperature."""
    if sea.shape[0]!=len(cu) or cu.shape!=(len(sea),2):
        raise ValueError("expected two predicted copper temperature rises per time")
    amb=rom.amb
    hot=rom.port.copper(np.full(2,amb+.1))
    cold=rom.port.copper(np.full(2,amb-.1))
    slope=torch.as_tensor((hot-cold)/.2,dtype=torch.complex64,
                          device=sea.device)
    dz=cu.to(torch.complex64)*slope[None]
    zero=torch.zeros_like(dz[:,0].real)
    copper=torch.stack((dz[:,0].real,dz[:,0].imag,zero,zero,
                        dz[:,1].real,dz[:,1].imag),dim=-1)
    zscale=torch.as_tensor(sy[:6],dtype=sea.dtype,device=sea.device)
    pred=sea[:,:6]*zscale+copper
    ref=torch.as_tensor(row["z6"][1:]-row["z6"][0],
                        dtype=sea.dtype,device=sea.device)
    return (pred-ref).square().sum()/ref.square().sum().clamp_min(1e-12)


def score(net, rows, g, sx, sy, sa, closed=False, rom=None,
          therm_w=0., src_mix=0., zrel_w=0., hist_weight="equal"):
    total, spans = [], []
    for row in rows:
        t = row["time"]
        u = torch.as_tensor(row["src"]/sx, dtype=torch.float32)
        y = torch.as_tensor(row["y"]/sy, dtype=torch.float32)
        a = torch.as_tensor(row["a"], dtype=torch.float32)
        asc = torch.as_tensor(sa, dtype=torch.float32)
        h = net.init_state(1)
        pr_z, pr_a = [], []
        pr_phys,pr_pkg,pr_cu,pr_cuz = [],[],[],[]
        cur = a[0:1]
        zy = torch.zeros((1,14),dtype=a.dtype)
        rr=row.get("rom",rom)
        if rr is not None:
            from sdfmpneo_bfzi.geom_canon import mode_scale
            sc=mode_scale(rr,rr.rank) if 'K' in row else np.ones(rr.rank)
            decoder=torch.as_tensor(rr.phi[rr.ns:rr.cu.stop]*sc[None,:],
                                    dtype=a.dtype)
            actual=torch.as_tensor(
                row["temp_K"][:,rr.ns:rr.cu.stop]-rr.amb,dtype=a.dtype)
            n_pkg=2*rr.npkg
        if src_mix and (not closed or rr is None):
            raise ValueError("autonomous source needs closed heat and geometry ROM")
        if therm_w:
            if rom is None and 'K' not in row:
                raise ValueError("thermal physics loss requires a heat ROM")
            K = torch.as_tensor(row['K'] if 'K' in row else rom.K, dtype=a.dtype)
            # Both residuals contain the same physical teacher source q,
            # which cancels exactly when subtracting them. Avoid per-time
            # expensive warm-source reconstruction in teacher ingestion.

        gg = torch.as_tensor(row['geom'][None],dtype=torch.float32) if 'geom' in row else g
        for k in range(1, len(t)):
            dt=float(t[k]-t[k-1])
            prev = cur if closed else a[k-1:k]
            input_u=u[k:k+1]
            if src_mix:
                live=_port_src(row,rr,prev,zy,sy,float(row["voltage"][k-1]))
                input_u=(1-src_mix)*input_u+src_mix*(
                    live/torch.as_tensor(sx,dtype=a.dtype))
            h, zhat, nxt = net.step_et(h,input_u,float(t[k]-t[k-1]),gg,
                                       prev,asc)
            zy=zhat
            state=nxt
            if closed and src_mix:
                physical=_heat_balance(row,rr,prev,nxt,zhat,sy,dt,
                                       float(row["voltage"][k-1]))
                state=(1-src_mix)*nxt+src_mix*physical
            if closed:
                cur=state
            # The training target must describe the same corrected state
            # that autonomous inference returns at the sampling instant.
            pr_z.append(zhat[0])
            pr_a.append((state[0]-a[k])/asc)
            if rr is not None:
                pred_temp=state[0]@decoder.T
                pr_pkg.append((pred_temp[:n_pkg]-actual[k,:n_pkg])
                              .square().mean())
                pr_cu.append((pred_temp[n_pkg:]-actual[k,n_pkg:])
                             .square().mean())
                if zrel_w:
                    pr_cuz.append(pred_temp[n_pkg:])
            if therm_w:
                delta=state[0]-a[k]
                prev_delta=prev[0]-a[k-1]
                res=(delta-prev_delta)/dt+K@delta
                scale=(asc*(1/dt+K.abs().sum(dim=1))).clamp_min(1e-6)
                pr_phys.append((res/scale).square().mean())
        if not pr_z:
            continue
        pz = torch.stack(pr_z)
        pa = torch.stack(pr_a)
        zerr = ((pz-y[1:])**2).mean()
        zextra = ((pz[:, :6]-y[1:, :6])**2).mean()
        heat = pa.square().mean()
        phy = torch.stack(pr_phys).mean() if therm_w else 0.
        temp_loss=(.15*torch.stack(pr_pkg).mean()
                   +.15*torch.stack(pr_cu).mean()) if pr_pkg else 0.
        zrel=0.
        if zrel_w:
            if rr is None:
                raise ValueError("full dynamic impedance needs a heat ROM")
            zrel=_zrel_full(rr,row,pz,torch.stack(pr_cuz),sy)
        total.append(zerr+0.45*zextra+heat+therm_w*phy
                     +temp_loss+zrel_w*zrel)
        spans.append(len(t)-1)
    if not total:
        raise ValueError("teacher histories need at least two times")
    return _hist_mean(total, spans, hist_weight)


def eval_rollout(rom, net, ck, rows, tangent=None):
    """Evaluate a genuinely autonomous physics-coupled held-out history."""
    result = []
    net.eval()
    with torch.no_grad():
        for row in rows:
            rr = row.get('rom',rom)
            begin = time.perf_counter()
            pred = rollout(rr, net, ck, row["voltage"], row["time"], tangent=tangent)
            cpu_s = time.perf_counter() - begin
            real_z = row["z6"]
            real_t = row["temp_K"]
            dz = np.linalg.norm(real_z-real_z[0])
            seg = {}
            for lo,hi in ((0,60),(60,300),(300,600)):
                use=(row["time"]>lo)&(row["time"]<=hi)
                if np.any(use):
                    num=np.linalg.norm((pred["z6"]-real_z)[use])
                    den=np.linalg.norm((real_z-real_z[0])[use])
                    seg[f"dynamic_Z_{lo}_{hi}_rel_L2"]=float(num/max(den,1e-12))
            result.append(dict(
                file=row["file"],
                inference_cpu_s=float(cpu_s),
                dynamic_Z_rel_L2=float(np.linalg.norm(pred["z6"]-real_z)
                                       / max(dz, 1e-12)),
                full_Z_rel_L2=float(np.linalg.norm(pred["z6"]-real_z)
                                    / max(np.linalg.norm(real_z), 1e-12)),
                temperature_max_K=float(np.max(abs(pred["temp_K"]-real_t))),
                copper_max_K=float(np.max(abs(pred["temp_K"][:, rr.cu]
                                               -real_t[:, rr.cu]))),
                package_max_K=float(np.max(abs(
                    pred["temp_K"][:, rr.ns:rr.cu.start]
                    -real_t[:, rr.ns:rr.cu.start]))),
                material_budget_max_W=float(np.max(abs(pred["material_budget_W"]))),
                fv_residual_max_W=float(np.max(abs(pred["fv_node_residual_W"])))))
            result[-1].update(seg)
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
