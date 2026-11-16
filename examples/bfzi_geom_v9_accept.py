"""Frozen V9 independent BFZI S5 impedance and temperature acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from examples.bfzi_z_tangent import restore as tan_load
from sdfmpneo_bfzi.geom_ztune import restore as z_load
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_pkg import restore as pkg_load
from sdfmpneo_bfzi.geom_skin import restore as skin_load


def stats(rows):
    return dict(dynamic_Z_mean_rel_L2=float(np.mean([r['dynamic_Z_rel_L2'] for r in rows])),
                dynamic_Z_max_rel_L2=float(max(r['dynamic_Z_rel_L2'] for r in rows)),
                full_Z_mean_rel_L2=float(np.mean([r['full_Z_rel_L2'] for r in rows])),
                temperature_mean_max_K=float(np.mean([r['temperature_max_K'] for r in rows])),
                temperature_max_K=float(max(r['temperature_max_K'] for r in rows)),
                material_budget_max_W=float(max(r['material_budget_max_W'] for r in rows)),
                cpu_mean_s=float(np.mean([r['inference_cpu_s'] for r in rows])),
                results=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,
                   default=Path('runs/geomfrac_et_v9_selection.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_et_v9_final.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    sel=json.loads(a.selection.read_text(encoding='utf-8'))
    names=sel['geometry_ids']
    if names!=['W01','W02','W03','W04'] or sel['test_evaluated'] is not False:
        p.error('frozen W01-W04 V9 selection required')
    tags=['v9a','v9b','v9c','v9d']
    data=[];phys={}
    for name,tag in zip(names,tags):
        mf=Path(f'runs/geomfrac_s5_{tag}_manifest.json')
        rf=Path(f'runs/geomfrac_s5_{tag}_build_report.json')
        if not mf.is_file() or not rf.is_file():
            p.error(f'BFZI physics for {name} is unfinished')
        manifest=json.loads(mf.read_text(encoding='utf-8'))
        report=json.loads(rf.read_text(encoding='utf-8'))['cases'][name]
        specs=manifest['test']
        if len(specs)!=1 or f'_{name}_' not in specs[0]['anchor']:
            p.error(f'geometry mismatch for {name}')
        if len(report['traces'])!=3 or any(
                r['physics']['status']!='candidate' for r in report['traces']):
            p.error(f'incomplete physics histories for {name}')
        data.extend(prepare(specs[0],32))
        phys[name]=dict(seconds=report['seconds'],
                        anchor=specs[0]['anchor'],
                        teacher=specs[0]['teacher'])
    if len(data)!=12:
        p.error('expected 4 geometries and 12 BFZI histories')
    ck=torch.load(sel['heat_model'],map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
                   soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    tan=tan_load(sel['sea_tangent'])
    nonlinear=z_load(sel['nonlinear_tangent'])
    pkg=pkg_load(sel['pkg_model'])
    skin=skin_load(sel['skin_model'])
    with np.load(sel['sea_lift']) as f:
        lift=f['w']*float(sel['sea_gain'])
    results={}
    for name,zt in [('V7',tan),('V9',nonlinear)]:
        out=eval_rollout(None,net,ck,data,tangent=zt,pkg=pkg,
                         lift=lift,skin_out=skin)
        results[name]=stats(out)
        m=results[name]
        print('V9_ACCEPT',name,'dynamic_Z_pct',
              round(100*m['dynamic_Z_mean_rel_L2'],4),
              'temperature_max_mean_K',round(m['temperature_mean_max_K'],4),
              'cpu_seconds',round(m['cpu_mean_s'],4),flush=True)
    a0=results['V7']['results'];b0=results['V9']['results']
    if any(x['file']!=y['file'] for x,y in zip(a0,b0)):
        raise RuntimeError('acceptance history ordering mismatch')
    out=dict(description='Frozen independent V9 BFZI S5 acceptance',
             selection=str(a.selection),test_geometries=names,test_histories=12,
             physical_teacher=phys,models=results,
             relative_dynamic_Z_gain=1-results['V9']['dynamic_Z_mean_rel_L2']/
                                   results['V7']['dynamic_Z_mean_rel_L2'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('V9_INDEPENDENT_COMPLETE',a.out,flush=True)


if __name__=='__main__':
    main()
