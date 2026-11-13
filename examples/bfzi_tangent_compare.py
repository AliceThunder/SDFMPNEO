"""Compare reduced-solve BFZI sea-Z tangents against central reference.

Evaluates the SAME untouched original BFZI held-out 3-D temperature and
complex-port trajectories. The test isolates electromagnetic tangent error;
the complete neural+thermal ROM is evaluated by bfzi_geom_heat.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from examples.bfzi_tangent_eval import score


def compare(head,ref,fast,teacher):
    rs=score(head,ref,teacher)
    fs=score(head,fast,teacher)
    with np.load(teacher) as d:
        t=np.asarray(d["temp_K"],float)
        z=np.asarray(d["z6"],float)
    rows=[]
    for i,(a,b) in enumerate(zip(rs,fs)):
        cand=np.stack([pack(fast.delta(x[:head.ns]-head.amb))
                       for x in t[i]])
        baseline=np.stack([pack(ref.delta(x[:head.ns]-head.amb))
                           for x in t[i]])
        rel=float(np.linalg.norm(cand-baseline)/
                  max(np.linalg.norm(z[i]-z[i,0]),1e-12))
        rows.append(dict(
            case=a["case"],voltage_V=a["voltage_V"],
            central_port_dynamic_Z_relative_L2=a[
                "tangent_full_dynamic_Z_relative_L2"],
            forward_port_dynamic_Z_relative_L2=b[
                "tangent_full_dynamic_Z_relative_L2"],
            central_sea_delta_relative_L2=a[
                "tangent_sea_delta_relative_L2"],
            forward_sea_delta_relative_L2=b[
                "tangent_sea_delta_relative_L2"],
            tangent_difference_over_total_dynamic_Z=rel))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    head=GeomHeat.load(args.anchor)
    ref=SeaTangent.load(args.reference,head)
    candidate=SeaTangent.load(args.candidate,head)
    result=dict(
        teacher=str(args.teacher),
        entirely_heldout_dynamic_history=True,
        using_original_BFZI_true_3D_temperatures=True,
        central=dict(file=str(args.reference),method=ref.method,
                     directional_EM_solves=ref.nsolve),
        candidate=dict(file=str(args.candidate),method=candidate.method,
                       directional_EM_solves=candidate.nsolve,
                       perturbation_K=candidate.amplitude),
        rows=compare(head,ref,candidate,args.teacher))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
