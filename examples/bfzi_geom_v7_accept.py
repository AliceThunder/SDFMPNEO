"""Frozen four-geometry BFZI V7 independent EM/thermal acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from examples.bfzi_z_tangent import restore as load_tan
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_pkg import restore as load_pkg
from sdfmpneo_bfzi.geom_skin import restore as load_skin


def stats(rows):
    return dict(
        dynamic_Z_mean_rel_L2=float(np.mean([r['dynamic_Z_rel_L2'] for r in rows])),
        dynamic_Z_max_rel_L2=float(max(r['dynamic_Z_rel_L2'] for r in rows)),
        temperature_mean_max_K=float(np.mean([r['temperature_max_K'] for r in rows])),
        temperature_max_K=float(max(r['temperature_max_K'] for r in rows)),
        material_budget_max_W=float(max(r['material_budget_max_W'] for r in rows)),
        cpu_mean_s=float(np.mean([r['inference_cpu_s'] for r in rows])),
        results=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,
                   default=Path('runs/geomfrac_et_v7_selection.json'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_et_v7_final.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    cfg=json.loads(a.selection.read_text(encoding='utf-8'))
    assert cfg['new_test_results_seen'] is False
    assert cfg['skin_feedback'] is False
    expected=cfg['new_test_cases']
    if expected!=['S01','S02','S03','S04']:
        p.error('frozen V7 test cases must be S01-S04')
    tags=('v7a','v7b','v7c','v7d')
    specs=[]
    physical={}
    for key,tag in zip(expected,tags):
        manifest=Path(f'runs/geomfrac_s5_{tag}_manifest.json')
        report=Path(f'runs/geomfrac_s5_{tag}_build_report.json')
        if not manifest.exists() or not report.exists():
            p.error(f'{key}: BFZI physical teacher unfinished')
        mf=json.loads(manifest.read_text(encoding='utf-8'))
        rp=json.loads(report.read_text(encoding='utf-8'))['cases'][key]
        if len(mf['test'])!=1 or len(rp['traces'])!=3:
            p.error(f'{key}: expected three finished physical histories')
        spec=mf['test'][0]
        if f'_{key}_' not in spec['anchor']:
            p.error(f'{key}: geometry differs from frozen plan')
        if any(r['physics']['status']!='candidate' for r in rp['traces']):
            p.error(f'{key}: invalid physics teacher status')
        specs.append(spec)
        physical[key]=dict(seconds=rp['seconds'],
                           anchor=spec['anchor'],teacher=spec['teacher'])
    ck=torch.load(cfg['model'],map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    rows=[r for spec in specs for r in prepare(spec,int(ck['rank']))]
    if len(rows)!=12:
        p.error('12 unseen BFZI teacher histories required')
    tan=load_tan(cfg['tangent_model'])
    pkg=load_pkg(cfg['pkg_model'])
    skin=load_skin(cfg['skin_output_model'])
    with np.load(cfg['lift_model']) as data:
        lift=data['w']*float(cfg['lift_gain'])
    methods={
        'V5':dict(tangent=tan),
        'V6_coupled':dict(tangent=tan,pkg=pkg,lift=lift),
        'V7_skin_output':dict(tangent=tan,pkg=pkg,lift=lift,skin_out=skin)}
    result={}
    for name,opts in methods.items():
        pred=eval_rollout(None,net,ck,rows,**opts)
        result[name]=stats(pred)
        print('ACCEPT_V7',name,'dynamic_Z_pct',
              round(100*result[name]['dynamic_Z_mean_rel_L2'],3),
              'mean_max_temp_K',round(result[name]['temperature_mean_max_K'],3),
              'cpu_s',round(result[name]['cpu_mean_s'],3),flush=True)
    v6=result['V6_coupled']
    v7=result['V7_skin_output']
    assert all(a['file']==b['file'] for a,b in zip(v6['results'],v7['results']))
    assert abs(v6['dynamic_Z_mean_rel_L2']-v7['dynamic_Z_mean_rel_L2'])<1e-12
    assert abs(v6['material_budget_max_W']-v7['material_budget_max_W'])<1e-13
    report=dict(status='frozen independent V7 test completed',
        selection_file=str(a.selection),test_geometries=expected,
        test_histories=len(rows),physical_teacher=physical,models=result,
        relative_skin_temperature_improvement=(
            1-v7['temperature_mean_max_K']/v6['temperature_mean_max_K']))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FROZEN_V7_COMPLETE',a.out,flush=True)


if __name__=='__main__':
    main()
