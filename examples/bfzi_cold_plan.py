"""Deterministic cold-only geometry sampling for BFZI V150.

No thermal history or geometry-specific heat ROM is generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _key(g):
    return json.dumps(g,sort_keys=True,separators=(',',':'))


def _coil(a,core):
    """M4 ten-turn BFZI geometry admissibility from native coil rules."""
    r0=88.85-core
    inner_r=r0-50*a
    inner_chi=core-50*(1-a)
    return inner_r>1.25*.790569415 and inner_chi>=0


def sample(seed=62026,n=500,existing=(),accept=None):
    """Create geometry-disjoint train/dev/test samples with broad S5 coverage."""
    rng=np.random.default_rng(seed)
    occupied={_key(g) for g in existing}
    cases={}
    attempts=0
    # Stratification spreads each geometry scalar across the generated batch.
    cols=[]
    for _ in range(12):
        x=(np.arange(n)+rng.random(n))/n
        rng.shuffle(x)
        cols.append(x)
    x=np.stack(cols,axis=1)
    for i in range(n):
        found=False
        for _ in range(40):
            attempts+=1
            u=x[i] if _==0 else rng.random(12)
            al=[round(float(.70+.30*u[j]),4) for j in (0,1)]
            core=[]
            for j in range(2):
                low=max(0,int(np.ceil(50*(1-al[j])+1)))
                high=min(25,int(np.floor(
                    (88.85-50*al[j]-1.25*.790569415-1)/(1+al[j]))))
                if high<low:break
                c=int(low+np.floor(u[2+j]*(high-low+1)))
                core.append(min(c,high))
            if len(core)!=2 or any(not _coil(a,c) for a,c in zip(al,core)):
                continue
            g=dict(alpha1=al[0],alpha2=al[1],
                   core1_mm=core[0],core2_mm=core[1],
                   shift_mm=[int(round(-16+32*u[4])),
                             int(round(80+36*u[5])),
                             int(round(-12+24*u[6]))],
                   tilt_deg=int(round(-20+40*u[7])),
                   yaw_deg=int(round(-20+40*u[8])),
                   pitch_deg=int(round(-18+36*u[9])),
                   box1_mm=int(round(210+32*u[10])),
                   box2_mm=int(round(210+32*u[11])))
            k=_key(g)
            if k in occupied or (accept is not None and not accept(g)):
                continue
            occupied.add(k)
            name=f'C{i+1:04d}'
            cases[name]=g
            found=True
            break
        if not found:
            raise RuntimeError(f'could not find unique valid geometry {i+1}')
    # Sample membership fixed before expensive reference generation.
    names=list(cases)
    rng.shuffle(names)
    ndev=max(1,round(n*.1))
    ntest=max(1,round(n*.1))
    return dict(description='BFZI V150 cold-only geometry data; no thermal histories',
                version=1,seed=seed,attempts=attempts,
                train=names[ndev+ntest:],val=names[:ndev],
                test=names[ndev:ndev+ntest],cases=cases,
                solver=dict(frequency_hz=200000,grid='ultra',
                            medium='original M4 seawater',
                            source_order=3))


def _existing(paths):
    geom=[]
    for path in paths:
        f=Path(path)
        if not f.exists():continue
        d=json.loads(f.read_text(encoding='utf-8-sig'))
        if 'cases' in d:
            geom.extend([c['geometry'] for c in d['cases'].values()])
        elif 'train' in d and len(d['train']) and 'geometry' in d['train'][0]:
            for part in ('train','dev','bb_diagnostic'):
                geom.extend([r['geometry'] for r in d.get(part,[])])
    return geom


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=62026)
    p.add_argument('--samples',type=int,default=500)
    p.add_argument('--existing',type=Path,nargs='*',default=[
        Path('runs/geomfrac_pose42_plan.json'),
        Path('runs/geomfrac_s5_free11_plan.json'),
        Path('runs/geomfrac_cold_ports_v1.json')])
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_cold500_plan.json'))
    a=p.parse_args()
    if a.samples<20: p.error('--samples must be >=20')
    if a.out.exists():
        raise FileExistsError(f'plan already frozen: {a.out}')
    data=sample(a.seed,a.samples,_existing(a.existing))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print('COLD_PLAN',a.out,'train',len(data['train']),
          'dev',len(data['val']),'test',len(data['test']),
          'attempts',data['attempts'],flush=True)


if __name__=='__main__':
    main()
