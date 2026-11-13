"""Original-BFZI held-out 3-D sea-Z thermal tangent validation.

Use exactly the original BFZI direct-method temperature and port impedance
to isolate the cold-operator tangent accuracy before changing inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.frac_memory import zmat


def score(head,tan,teacher):
    rows=[]
    with np.load(teacher) as d:
        temps=np.asarray(d["temp_K"],float)
        zs=np.asarray(d["z6"],float)
        ids=json.loads(str(d["cases_json"]))
    for idx in range(len(temps)):
        est=[];true=[];seatrue=[];seatangent=[];captures=[]
        for k,tk in enumerate(temps[idx]):
            cu=head.port.copper(tk[head.cu])
            real=zmat(zs[idx,k])
            sea=real-1j*head.port.omega*head.port.air-np.diag(cu)
            pred=head.sea0+tan.delta(tk[:head.ns]-head.amb)
            est.append(pack(pred+1j*head.port.omega*head.port.air+
                            np.diag(cu)))
            true.append(zs[idx,k])
            seatrue.append(pack(sea-head.sea0))
            seatangent.append(pack(pred-head.sea0))
            captures.append(tan.capture(tk[:head.ns]-head.amb))
        est=np.asarray(est);true=np.asarray(true)
        a=np.asarray(seatrue);b=np.asarray(seatangent)
        rows.append(dict(
            case=int(ids[idx]["id"]),
            voltage_V=float(ids[idx]["voltage_v"]),
            tangent_full_dynamic_Z_relative_L2=float(
                np.linalg.norm(est-true)/
                max(np.linalg.norm(true-true[0]),1e-12)),
            tangent_sea_delta_relative_L2=float(
                np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),
            thermal_mode_projection_relative_L2_max=float(max(captures))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True)
    p.add_argument("--tangent",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    head=GeomHeat.load(a.anchor)
    tan=SeaTangent.load(a.tangent,head)
    out=dict(anchor=str(a.anchor),tangent=str(a.tangent),
             teacher=str(a.teacher),rank=len(tan.jac),
             oracle_true_3D_temperature=True,
             results=score(head,tan,a.teacher))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
