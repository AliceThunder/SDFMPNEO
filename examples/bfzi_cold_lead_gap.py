"""Investigate native BFZI terminal lead clearance and self-L sensitivity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from examples.bfzi_param import originals,create
from examples.bfzi_physics_recheck import GRIDS

def calc(bf,v150,g):
    xz,y=GRIDS['ultra']
    data,_=create(bf,v150,'M4',**g,sea_void='oriented',
                  physical_box_only=True,grid_xz=xz,grid_y=y,
                  grid_max_nodes=4000)
    out={}
    for sep in (0.,.004,.008):
        d=dict(data,cfg={**data['cfg'],'ENV_ZLEAD_LANE_SEPARATION_X_M':sep})
        L,_=bf.compute_air_matrix_direct_zlead(d)
        info=[]
        for fam in d['families']:
            lead,meta=bf.zlead_line_geom_for_coil(fam.coil,d['cfg'])
            n=int(meta['panels_outer'])
            ext=float(np.linalg.norm(np.asarray(meta['outer_terminal_world_m'])-
                                     np.asarray(meta['inner_terminal_world_m'])))
            at=float(np.linalg.norm(np.asarray(meta['outer_attach_world_m'])-
                                    np.asarray(meta['inner_attach_world_m'])))
            mids=float(cdist(lead.pos[:n],lead.pos[n:]).min())
            info.append(dict(terminal_gap_mm=1000*ext,
                             winding_terminal_gap_mm=1000*at,
                             lead_midpoint_min_gap_mm=1000*mids,
                             wire_diameter_mm=2000*fam.coil.wire_radius_m))
        out[f'{sep*1000:.1f}mm']=dict(air_uH=np.diag(L).tolist(),gaps=info)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--ids',nargs='+',
                   default=['C0459','C0171','C0013','C0510'])
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_lead_gap_v1.json'))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    bf,v150=originals(a.bfzi_root)
    out={}
    for key in a.ids:
        try:
            v=calc(bf,v150,plan['cases'][key])
            out[key]=v
            print('LEAD_GAP',key,json.dumps(v),flush=True)
        except Exception as e:
            out[key]=dict(error=f'{type(e).__name__}: {e}')
            print('LEAD_FAIL',key,out[key]['error'],flush=True)
        a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
