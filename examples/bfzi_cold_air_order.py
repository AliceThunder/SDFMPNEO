"""Check native BFZI cold air inductance with physically continuous lead ordering."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS

def solve(bf,v150,g):
    xz,y=GRIDS['ultra']
    d,_=create(bf,v150,'M4',**g,sea_void='oriented',
               physical_box_only=True,grid_xz=xz,grid_y=y,
               grid_max_nodes=4000)
    cfg=dict(d['cfg'])
    cfg.update({'LMB_REQUIRE_CPP_KERNEL':False,'LMB_LINE_GAUSS_ORDER':6,
                'LMB_FALLBACK_QUAD_ORDER':64,'LMB_PAIR_CHUNK':512})
    mu=float(bf.fs.PhysicalConstants().mu)
    old,_=bf.compute_air_matrix_direct_zlead(d)
    ret=[]
    for fam in d['families']:
        lead,meta=bf.zlead_line_geom_for_coil(fam.coil,cfg)
        n=int(meta['panels_outer'])
        a,b=fam.active,lead
        pos=np.concatenate((b.pos[:n],a.pos,b.pos[n:]))
        dl=np.concatenate((b.dl[:n],a.dl,b.dl[n:]))
        path=bf.fs._line_geom(fam.coil,pos,dl)
        z,_=bf._line_self_gmd_diffusion(path,fam.rg,mu,0j,cfg)
        def start(s,v):return np.asarray(s)-np.asarray(v)*.5
        def end(s,v):return np.asarray(s)+np.asarray(v)*.5
        gap1=float(np.linalg.norm(end(b.pos[n-1],b.dl[n-1])-
                                   start(a.pos[0],a.dl[0])))
        gap2=float(np.linalg.norm(end(a.pos[-1],a.dl[-1])-
                                   start(b.pos[n],b.dl[n])))
        ret.append(dict(ordered_uH=float(z.real*1e6),
                        join_gap_m=[gap1,gap2],
                        lead_parts=[n,len(b.pos)-n],
                        body_segments=len(a.pos)))
    return dict(old_uH=np.diag(old).tolist(),new=ret)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--ids',nargs='+',
                   default=['C0459','C0171','C0013','C0510'])
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_air_order_v1.json'))
    a=p.parse_args()
    cases=json.loads(a.plan.read_text(encoding='utf-8-sig'))['cases']
    bf,v150=originals(a.bfzi_root)
    rows={}
    for key in a.ids:
        t0=time.perf_counter()
        try:
            item=solve(bf,v150,cases[key])
            item['seconds']=round(time.perf_counter()-t0,3)
            rows[key]=item
            print('AIR_ORDER',key,json.dumps(item),flush=True)
        except Exception as e:
            rows[key]=dict(error=f'{type(e).__name__}: {e}')
            print('AIR_ORDER_FAIL',key,rows[key]['error'],flush=True)
        a.out.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
    main()
