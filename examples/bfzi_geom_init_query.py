"""Predict cold port and common 32-mode thermal operator without BFZI."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sdfmpneo_bfzi.geom_init import GeomInit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atlas',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--case-file',type=Path,required=True,
                   help="JSON case or JSON physical plan containing --case")
    p.add_argument('--case',type=str,default=None)
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_init_query.npz'))
    a=p.parse_args()
    raw=json.loads(a.case_file.read_text(encoding='utf-8'))
    case=raw['cases'][a.case] if a.case else raw
    model=GeomInit.load(a.atlas)
    start=time.perf_counter()
    pred=model.predict(case)
    sec=time.perf_counter()-start
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**{k:v for k,v in pred.items()
                              if k!='norm_weights'})
    print(json.dumps(dict(status='geometry-only cold and shared operator',
        output=str(a.out),query_s=sec,heat_rank=model.rank,
        cold_Z6_ohm=pred['z0'].tolist(),
        min_thermal_eigen_s=float(np.linalg.eigvalsh(pred['k'])[0]),
        online_BFZI_calls=0,per_geometry_basis_builds=0),indent=2))


if __name__=='__main__':
    main()
