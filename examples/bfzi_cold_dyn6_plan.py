"""Freeze new physically valid cold-test geometries for independent S5 dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def design(data, seed=20261119, count=6):
    rows=list(data['test'])
    if count<2 or len(rows)<count:
        raise ValueError('need at least two held-out cold geometries')
    rng=np.random.default_rng(seed)
    ix=rng.choice(len(rows),size=count,replace=False)
    cases={}
    for j,i in enumerate(ix):
        row=rows[int(i)]
        key=row['name']
        # Physical and excitation parameters are assigned independently
        # of cold-label errors and future dynamic responses.
        v=float(rng.uniform(70,85))
        cases[key]=dict(
            family='continuous',
            donor='g1' if j%2==0 else 'g2',
            geometry=row['geometry'],
            alpha_per_K=round(float(rng.uniform(.018,.025)),5),
            thermal=dict(
                ins_k=round(float(rng.uniform(.19,.33)),4),
                ins_t=round(float(rng.uniform(.00034,.00046)),6),
                pkg_k=round(float(rng.uniform(.19,.34)),4),
                pkg_sea_h=round(float(rng.uniform(17,33)),3),
                skin_m=.002),
            drives=[
                dict(voltage=round(.75*v,2),off=[],duration=60),
                dict(voltage=round(v,2),
                     off=[[45,105],[195,255]],duration=300)])
    return dict(description='New six physically valid BFZI S5 dynamic holdouts '
                'selected from pre-existing cold-only test split',
                train=[],val=[],test=list(cases),cases=cases,seed=seed,
                grid='ultra',inner_step_s=1.875)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,
                   default=Path('runs/geomfrac_cold500/data.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_dyn6_plan.json'))
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError(f'dynamic holdout plan is already frozen: {a.out}')
    data=json.loads(a.data.read_text(encoding='utf-8-sig'))
    plan=design(data)
    a.out.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print('COLD_DYN6',plan['test'],'histories',sum(len(c['drives'])
          for c in plan['cases'].values()),flush=True)


if __name__=='__main__':
    main()
