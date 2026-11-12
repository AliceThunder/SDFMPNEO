"""Generate genuine BFZI V150 M4 histories for causal fractional neural learning.

One BFZI geometry/Green/air build is shared across all trajectories.
The training split and the held-out voltage/switching programs are disjoint.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
from sdfmpneo_bfzi.v148_et import circuit_power


def programs():
    train=[]
    for volt in (25.,55.,90.,120.,150.):
        for heat,cool in ((120,180),(240,180),(300,120),(180,240)):
            train.append((volt,[(0,heat),(heat+cool,600)],"train"))
    test=[]
    for volt in (40.,75.,105.,135.):
        for heat,cool in ((150,210),(270,150)):
            test.append((volt,[(0,heat),(heat+cool,600)],"test"))
    # A second independent start-stop pair is excluded from training.
    for volt in (65.,115.):
        test.append((volt,[(0,120),(210,330),(420,510)],"test"))
    return train+test


def drive(v, windows, t):
    on=any((lo<=t<=hi if lo==0 else lo<t<=hi)
           for lo,hi in windows)
    return np.array([v if on else 0.,0.],complex)


def pack_z(z):
    z=np.asarray(z)
    return np.array([z[0,0].real,z[0,0].imag,
                     z[0,1].real,z[0,1].imag,
                     z[1,1].real,z[1,1].imag])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("runs/m4_frac_teacher.npz"))
    p.add_argument("--step",type=int,default=30)
    p.add_argument("--sample",choices=("M1","M2","M3","M4"),default="M4",
                   help="Original BFZI V150 dual-port rounded-square/circular sample")
    p.add_argument("--limit",type=int,default=None,
                   help="First N physical histories, for targeted cross-geometry runs")
    args=p.parse_args(argv)
    if args.step<=0 or 600%args.step:
        p.error("step must divide 600 seconds")
    sys.path.insert(0,str(args.bfzi_root.resolve()))
    import bfzi_v148_results as calc
    import run_v150 as original

    t0=time.perf_counter()
    data=original.build_data(args.sample)
    model=BFZIV148Adapter(data,calc,heat_mode="port-weak")
    zr,*_=calc.base.formal_with_local_mass(data)
    cold=model.solve(np.full(len(model.vol),model.ambient),
                     np.full(2,model.ambient))
    rel=float(np.linalg.norm(cold.sea_z-zr)/np.linalg.norm(zr))
    if rel>1e-5:
        raise RuntimeError(f"BFZI {args.sample} original Z regression failed: {rel}")
    extra=np.diag([.5-1j*model.omega*model.air[0,0],
                   .5-1j*model.omega*model.air[1,1]])
    et=BFZIV148Pkg3D(
        model,external=extra,pkg_shape=(4,2,4),pkg_k=.25)
    times=np.arange(0,600+args.step,args.step,dtype=float)
    temp=[]; zs=[]; vin=[]; eff=[]; cur=[]; split=[]; cases=[]
    allcases=programs()
    if args.limit is not None:
        if args.limit<=0:
            p.error("--limit must be positive")
        allcases=allcases[:args.limit]
    for cid,(v,windows,tag) in enumerate(allcases):
        fn=lambda t: drive(v,windows,t)
        tr=et.trajectory(times,fn)
        temp.append(np.stack([a.temp for a in tr])+et.amb)
        zs.append(np.stack([pack_z(a.impedance) for a in tr]))
        vin.append(np.asarray([fn(t)[0].real for t in times]))
        cur.append(np.stack([a.currents for a in tr]))
        eff.append([circuit_power(fn(t),a.currents,.5,.5)[-1]
                    for t,a in zip(times,tr)])
        split.append(tag=="train")
        cases.append(dict(id=cid,sample=args.sample,voltage_v=v,
                          windows_s=windows,split=tag))
        if (cid+1)%5==0 or cid+1==len(allcases):
            print(f"teacher {cid+1}/{len(allcases)} "
                  f"elapsed={time.perf_counter()-t0:.1f}s",flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        args.out,time_s=times,voltage_v=np.asarray(vin),
        temp_K=np.asarray(temp),z6=np.asarray(zs),
        currents_A=np.asarray(cur),eta=np.asarray(eff),
        train=np.asarray(split,bool),nsea=et.count,
        npkg=et.pkg_n,ambient_K=et.amb,external_z=extra,
        sea_xyz_m=model.points,sea_vol_m3=model.vol,
        pkg_xyz_m=np.asarray(et.pkg_xyz),
        pkg_vol_m3=np.asarray(et.pkg_vol),
        sample=np.asarray(args.sample),
        cases_json=json.dumps(cases),bfzi_isothermal_relative_error=rel)
    print(json.dumps(dict(out=str(args.out),cases=len(allcases),
                          train=int(sum(split)),test=int(len(split)-sum(split)),
                          times=len(times),thermal_states=et.n,
                          original_bfzi_sample=args.sample,
                          original_bfzi_z_regression=rel,
                          elapsed_sec=time.perf_counter()-t0),indent=2))


if __name__=="__main__":
    main()
