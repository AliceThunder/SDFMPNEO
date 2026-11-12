"""Export original BFZI copper/air physics and verify neural assembly.

BFZI checkout is used only to extract validated M4 wire/air constants.
The saved PortModel runs independently on CPU with no BFZI dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.frac_memory import predict,zmat
from sdfmpneo_bfzi.frac_field import SeaHead,complex_z
from sdfmpneo_bfzi.frac_port import PortModel
from examples.bfzi_frac import load


def err(p,t):
    return float(np.linalg.norm(p-t)/max(np.linalg.norm(t),1e-14))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--field",type=Path,default=Path("runs/m4_frac_field.npz"))
    p.add_argument("--model",type=Path,default=Path("runs/m4_frac_multi_hybrid.pt"))
    p.add_argument("--case",type=Path,default=Path("runs/m4_pkg3d_weak_100v.json"))
    p.add_argument("--port",type=Path,default=Path("runs/m4_frac_port.npz"))
    p.add_argument("--out",type=Path,default=Path("runs/m4_frac_port_metrics.json"))
    a=p.parse_args()
    sys.path.insert(0,str(a.bfzi_root.resolve()))
    import bfzi_v148_results as calc
    data=calc.build_circle_data("M4")
    em=BFZIV148Adapter(data,calc,heat_mode="port-weak")
    port=PortModel.from_adapter(em)
    a.port.parent.mkdir(parents=True,exist_ok=True)
    port.save(a.port)
    port=PortModel.load(a.port)
    head=SeaHead.load(a.field)
    d=json.loads(a.case.read_text(encoding="utf-8"))
    with np.load(a.case.with_suffix(".npz")) as pack:
        temp_sea=np.asarray(pack["Tsea_K"],float)
        temp_cu=np.asarray(pack["Tcu_K"],float)
        times=np.asarray(pack["time_s"],float)
    ref=np.asarray([complex_z(s["Z_ohm"]) for s in d["trace"]])
    sref=np.asarray([complex_z(s["Zsea_ohm"]) for s in d["trace"]])
    # Independent original solver copper and Neumann air assembly.
    exact=np.array([port.total(s,t)
                    for s,t in zip(sref,temp_cu)])
    # Approximate sea head, exact BFZI air/copper, true thermal inputs.
    direct_temp=np.array([port.total(head.sea_z(s),t)
                          for s,t in zip(temp_sea,temp_cu)])
    v=np.asarray([d["source_voltage_peak_V"] if x["drive"]=="on" else 0.
                  for x in d["trace"]],float)
    torch.set_num_threads(2)
    model,ck=load(a.model)
    t0=time.perf_counter()
    neural=predict(model,ck,v,times)
    neutral=predict(model,ck,v,times,field=head,port=port)
    sec=time.perf_counter()-t0
    zn=np.asarray([zmat(x) for x in neural["z6"]])
    zp=np.asarray([zmat(x) for x in neutral["z6"]])
    full_res=[]
    sea_res=[]
    for k,I in enumerate(neutral["current_A"]):
        zs=neutral["Zsea_ohm"][k]
        zc=neutral["Zcu_ohm"][k]
        _,q,zi=head.at(neutral["temp_K"][k,:len(head.vol)],I)
        sea_res.append(abs(float(q@head.vol)-.5*np.vdot(I,zi@I).real))
        Rs=float(ck["external_z"][0,0].real)
        Rl=float(ck["external_z"][1,1].real)
        pin=.5*np.real(np.vdot(np.array([v[k],0],complex),I))
        ps=.5*Rs*abs(I[0])**2
        pl=.5*Rl*abs(I[1])**2
        pc=.5*np.sum(zc.real*abs(I)**2)
        pw=.5*np.vdot(I,zs@I).real
        full_res.append(abs(pin-ps-pl-pc-pw))
    result=dict(
        physical_constants=str(a.port),
        original_m4_case=str(a.case),
        original_z_assembly_max_abs_ohm=float(np.max(abs(exact-ref))),
        original_z_assembly_relative_L2=err(exact,ref),
        sea_head_z_assembly_relative_L2=err(direct_temp,ref),
        network_z_relative_L2=err(zn,ref),
        physical_neural_z_relative_L2=err(zp,ref),
        physical_neural_dynamic_z_relative_L2=err(zp,ref)/max(
            float(np.linalg.norm(ref-ref[0])/np.linalg.norm(ref)),1e-14),
        network_dynamic_z_relative_L2=err(zn,ref)/max(
            float(np.linalg.norm(ref-ref[0])/np.linalg.norm(ref)),1e-14),
        max_circuit_balance_absolute_W=max(full_res),
        max_sea_energy_gap_absolute_W=max(sea_res),
        end_to_end_two_models_sec=sec,
        independent_bfzi_used_for_constants_only=True)
    if result["original_z_assembly_relative_L2"]>1e-11:
        raise RuntimeError("original BFZI copper/air port does not close")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
