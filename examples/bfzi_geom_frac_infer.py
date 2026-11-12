"""Run one geometry-conditioned fractional neural port/thermal history.

Reads only original BFZI cold geometry/port parameters from an existing
physical teacher file. No electromagnetic/thermal solver is used in rollout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from sdfmpneo_bfzi.geom_memory import GeomFrac,rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,
                   default=Path("runs/bfzi_geom_frac_v2.pt"))
    p.add_argument("--geometry",type=Path,
                   default=Path("runs/m2_frac_holdout.npz"))
    p.add_argument("--case-index",type=int,default=None,
                   help="Use one stored drive history, not stored temperatures")
    p.add_argument("--voltage",type=float,default=100.)
    p.add_argument("--heat",type=float,default=240.)
    p.add_argument("--cool",type=float,default=180.)
    p.add_argument("--step",type=float,default=30.)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_geom_infer.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    with np.load(a.geometry) as d:
        geom=np.asarray(d["geometry"],float)
        z0=np.asarray(d["z6"][0,0],float)
        ext=np.asarray(d["external_z"],complex)
        sample=str(d["sample"])
        if a.case_index is not None:
            times=np.asarray(d["time_s"],float)
            vin=np.asarray(d["voltage_v"][a.case_index],float)
        else:
            if a.step<=0 or 600%a.step:
                p.error("--step must divide 600 seconds")
            times=np.arange(0.,600+a.step/2,a.step)
            vin=np.where((times<=a.heat)|
                         (times>a.heat+a.cool),a.voltage,0.)
    pr=rollout(net,ck,vin,times,geom,z0,ext)
    eta=[]
    for vv,I in zip(vin,pr["current_A"]):
        pi=.5*float(np.real(vv*I[0].conj()))
        pl=.5*float(ext[1,1].real)*abs(I[1])**2
        eta.append(pl/pi if pi>0 else 0.)
    labels=["Tsea_mean_rise_K","Tsea_hot_rise_K",
            "Tpkg1_mean_rise_K","Tpkg2_mean_rise_K",
            "Tpkg1_hot_rise_K","Tpkg2_hot_rise_K",
            "Tcu1_rise_K","Tcu2_rise_K"]
    rows=[]
    for k,t in enumerate(times):
        row=dict(time_s=float(t),voltage_V=float(vin[k]),
                 Z6=pr["z6"][k].tolist(),eta=float(eta[k]))
        row.update({key:float(val) for key,val in
                    zip(labels,pr["temp_rise_K"][k])})
        if "Psea_W" in pr:
            row["Psea_W"]=float(pr["Psea_W"][k])
            row["Pcu_W"]=pr["Pcu_W"][k].tolist()
            row["circuit_gap_W"]=float(pr["circuit_gap_W"][k])
        rows.append(row)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(
        model=str(a.model),geometry=str(a.geometry),sample=sample,
        cold_complex_Z6=z0.tolist(),trace=rows),indent=2),encoding="utf-8")
    extras={key:pr[key] for key in
            ("Zsea_ohm","Zcu_ohm","Psea_W","Pcu_W","circuit_gap_W")
            if key in pr}
    np.savez_compressed(a.out.with_suffix(".npz"),time_s=times,
        voltage_v=vin,z6=pr["z6"],
        temp_rise_K=pr["temp_rise_K"],eta=np.asarray(eta),
        current_A=pr["current_A"],**extras)
    print(json.dumps(dict(
        sample=sample,steps=len(times),out=str(a.out),
        max_package_rise_K=float(np.max(pr["temp_rise_K"][:,4:6])),
        max_copper_rise_K=float(np.max(pr["temp_rise_K"][:,6:])),
        max_circuit_gap_W=float(np.max(pr["circuit_gap_W"]))
            if "circuit_gap_W" in pr else None,
        final_eta=float(eta[-1])),indent=2))


if __name__=="__main__":
    main()
