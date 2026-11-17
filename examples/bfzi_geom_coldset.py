"""Build compact cold-only BFZI S5 anchors for offline geometry training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS,build
from examples.bfzi_geom_init import pack
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.s5_bridge import from_skin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/cold_atlas'))
    p.add_argument('--keys',nargs='+',required=True)
    p.add_argument('--rank',type=int,choices=(32,48,64),default=32)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    if any(k not in plan['train'] for k in a.keys):
        p.error('only declared cold-only training cases permitted')
    a.out.mkdir(parents=True,exist_ok=True)
    bf,v150=originals(a.bfzi_root)
    xz,y=GRIDS['ultra']
    for key in a.keys:
        start=time.perf_counter()
        dest=a.out/f'cold_{key}_{a.rank}.npz'
        if dest.exists():
            print('COLD_EXISTS',key,flush=True)
            continue
        case=plan['cases'][key]
        parent='p03' if case['donor']=='g1' else 'p07'
        ext=GeomHeat.load(Path('runs')/f'bfzi_pose_{parent}_disjoint_anchor.npz').ext
        data,_=create(bf,v150,'M4',**case['geometry'],
                      sea_void='oriented',physical_box_only=True,
                      grid_xz=xz,grid_y=y,grid_max_nodes=4000)
        mat=case['thermal']
        opts=dict(alpha=float(case['alpha_per_K']),source_order=5,
                  skin=mat.get('skin_m',.002),ins_k=mat.get('ins_k',.25),
                  ins_t=mat.get('ins_t'),pkg_k=mat.get('pkg_k',.25),
                  pkg_sea_h=mat.get('pkg_sea_h',20.),
                  sea_max=373.15,cu_max=1357.77,pkg_max=None)
        et=build(data,bf,ext,opts)
        ti=opts['ins_t'] if opts['ins_t'] is not None else np.mean(et.cu_ins_layers)
        head=from_skin(et.model,et,ext,thermal=[
            opts['alpha'],opts['ins_k'],ti,opts['pkg_k'],opts['pkg_sea_h']])
        r=pack(head,case,key,a.rank)
        np.savez_compressed(dest,**{k:r[k] for k in
                            ('x','y','k','src','ext','guard')})
        print('COLD_TRAIN',key,'cells',len(head.cap),
              'sec',round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':
    main()
