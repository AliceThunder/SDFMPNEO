"""Apply the trained V7 cold surrogate only inside its calibrated core domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdfmpneo_bfzi.geom_cold_domain import check,load
from sdfmpneo_bfzi.geom_coldnet import ColdNet


def infer(case,model,domain):
    gate=check(case,domain)
    if not gate["accepted"]:
        return dict(status="out_of_domain",domain=gate)
    z,air=model.predict(case)
    return dict(status="ok",domain=gate,
                z0_ohm=z.tolist(),air_h=air.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry",type=Path,required=True,
                   help="JSON with geometry keys, or a case containing geometry")
    p.add_argument("--model",type=Path,default=Path("runs/geomfrac_cold_v7.pt"))
    p.add_argument("--domain",type=Path,default=Path("runs/geomfrac_cold_v7_domain.json"))
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    case=json.loads(a.geometry.read_text(encoding="utf-8-sig"))
    net=ColdNet.load(a.model)
    result=infer(case,net,load(a.domain))
    data=json.dumps(result,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(data,encoding="utf-8")
    print(data,end="")


if __name__=="__main__":
    main()
