"""Equal-grid runtime benchmark: original BFZI V150 versus neural rollout."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from examples.bfzi_frac_data import programs,drive,pack_z
from examples.bfzi_frac import load
from sdfmpneo_bfzi.frac_memory import predict
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfzi-root",required=True,type=Path)
    ap.add_argument("--model",type=Path,default=Path("runs/m4_frac_hybrid.pt"))
    ap.add_argument("--out",type=Path,default=Path("runs/m4_frac_bench.json"))
    ap.add_argument("--case",type=int,default=-1)
    ap.add_argument("--step",type=float,default=30.)
    a=ap.parse_args()
    sys.path.insert(0,str(a.bfzi_root.resolve()))
    import bfzi_v148_results as calc
    v,windows,split=programs()[a.case]
    if split!="test":
        raise ValueError("benchmark requires a held-out program")
    if a.step<=0 or 600%a.step:
        ap.error("--step must divide 600")
    times=np.arange(0,600+a.step/2,a.step,dtype=float)
    data=calc.build_circle_data("M4")
    em=BFZIV148Adapter(data,calc,heat_mode="port-weak")
    extra=np.diag([.5-1j*em.omega*em.air[0,0],
                   .5-1j*em.omega*em.air[1,1]])
    et=BFZIV148Pkg3D(em,external=extra,pkg_shape=(4,2,4),pkg_k=.25)
    fn=lambda t: drive(v,windows,t)
    t0=time.perf_counter()
    tr=et.trajectory(times,fn)
    direct_sec=time.perf_counter()-t0
    vals=np.asarray([fn(t)[0].real for t in times])
    torch_mod=__import__("torch")
    torch_mod.set_num_threads(2)
    model,ck=load(a.model)
    t0=time.perf_counter()
    p=predict(model,ck,vals,times)
    neural_sec=time.perf_counter()-t0
    actual=np.stack([x.temp for x in tr])+et.amb
    ztrue=np.stack([pack_z(x.impedance) for x in tr])
    dt=p["temp_K"]-actual
    dz=p["z6"]-ztrue
    ns=et.count; npkg=et.pkg_n*2
    metrics=dict(
        model=str(a.model),heldout_voltage_v=v,heldout_windows_s=windows,
        steps=len(times),step_s=a.step,thermal_states=et.n,
        direct_trajectory_sec=direct_sec,
        neural_trajectory_sec=neural_sec,
        speedup=direct_sec/neural_sec,
        Z_relative_L2=float(np.linalg.norm(dz)/np.linalg.norm(ztrue)),
        dynamic_Z_relative_L2=float(np.linalg.norm(dz)/
                    max(np.linalg.norm(ztrue-ztrue[0]),1e-12)),
        sea_max_abs_K=float(np.max(abs(dt[:,:ns]))),
        package_max_abs_K=float(np.max(abs(dt[:,ns:ns+npkg]))),
        copper_max_abs_K=float(np.max(abs(dt[:,ns+npkg:]))),
        source="independent original BFZI v150 M4, full EM and 3-D heat",
        runtime_scope="same-grid rollout, excludes BFZI geometry/Green setup and neural training")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
