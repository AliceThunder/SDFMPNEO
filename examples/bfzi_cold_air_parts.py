"""Partition native BFZI V150 air self inductance into body, leads and cross."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS

def one(bf,v150,geom):
    xz,y=GRIDS['ultra']
    data,_=create(bf,v150,'M4',**geom,sea_void='oriented',
                  physical_box_only=True,grid_xz=xz,grid_y=y,
                  grid_max_nodes=4000)
    cfg=dict(data['cfg'])
    cfg.update({'LMB_REQUIRE_CPP_KERNEL':False,'LMB_LINE_GAUSS_ORDER':6,
                'LMB_FALLBACK_QUAD_ORDER':64,'LMB_PAIR_CHUNK':512})
    mu=float(bf.fs.PhysicalConstants().mu)
    parts=[]
    full,_=bf.compute_air_matrix_direct_zlead(data)
    for fam in data['families']:
        lead,_=bf.zlead_line_geom_for_coil(fam.coil,cfg)
        def selfL(line):
            v,_=bf._line_self_gmd_diffusion(line,fam.rg,mu,0j,cfg)
            return float(v.real*1e6)
        body=selfL(fam.active)
        lead_s=selfL(lead)
        cross,_=bf._line_mutual_diffusion(fam.active,lead,mu,0j,cfg)
        cross=float(2*cross.real*1e6)
        parts.append(dict(body_uH=body,lead_uH=lead_s,
                          body_lead_2x_uH=cross,
                          reconstructed_uH=body+lead_s+cross))
    return dict(full_air_uH=np.diag(full).tolist(),components=parts)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--ids',nargs='+',
                   default=['C0459','C0171','C0013','C0510'])
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_air_parts_v1.json'))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    bf,v150=originals(a.bfzi_root)
    out={}
    for key in a.ids:
        start=time.perf_counter()
        try:
            parts=one(bf,v150,plan['cases'][key])
            path=Path('runs/geomfrac_cold500/samples')/f'{key}.npz'
            if path.exists():
                with np.load(path) as f:
                    aa=np.asarray(f['air'],float)
                    parts['saved_air_uH']=(np.diag(aa)*1e6).tolist()
                    parts['saved_z0_R_X']=np.asarray(f['z0'],float).tolist()
            parts['seconds']=time.perf_counter()-start
            out[key]=parts
            print('AIR_PART',key,json.dumps(parts),flush=True)
        except Exception as exc:
            out[key]=dict(error=f'{type(exc).__name__}: {exc}')
            print('AIR_FAILED',key,out[key]['error'],flush=True)
        a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
    main()
