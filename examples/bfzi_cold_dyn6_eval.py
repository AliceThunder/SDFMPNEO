"""Compare frozen V5 and cold-only candidates on fresh S5 dynamic holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from examples.bfzi_geom_multi import free_metric, prepare
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_coldres import ColdRes
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def _name(row):
    return Path(row['file'].replace('\\','/')).stem.split('_')[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold_dyn6_plan.json'))
    p.add_argument('--manifest',type=Path,
                   default=Path('runs/geomfrac_s5_cold_dyn6_manifest.json'))
    p.add_argument('--config',type=Path,
                   default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--thermal-model',type=Path,
                   default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--coldnet',type=Path,
                   default=Path('runs/geomfrac_coldnet_v1.pt'))
    p.add_argument('--coldres',type=Path,
                   default=Path('runs/geomfrac_coldres_v1.pt'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_dyn6_eval.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    rows=[r for spec in mf.get('test',[]) for r in prepare(spec,32)]
    if not rows:
        raise RuntimeError('no new independent dynamic reference histories')
    ck=torch.load(a.thermal_model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    atlas=GeomInit.load(cfg['atlas'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    ports=dict(
        v5=PortSelf.load(cfg['port_self_model']),
        coldnet=ColdNet.load(a.coldnet),
        coldres=ColdRes.load(a.coldres))
    names=sorted({_name(r) for r in rows})
    if set(names)-set(plan['test']):
        raise RuntimeError('reference geometry outside frozen test split')
    out=dict(status='complete',geometry_count=len(names),
             history_count=len(rows),selected_geometry=names,
             model='frozen geomfrac_et_v5_closed.pt',
             atlas=str(cfg['atlas']),shape_mix=float(cfg['shape_mix']),
             results={})
    for key,port in ports.items():
        free=(plan,atlas,port,mu,src,tan,float(cfg['shape_mix']))
        result=dict(mean_dynamic_Z_pct=100*free_metric(rows,free,net,eck),
                    per_geometry={})
        for name in names:
            rr=[r for r in rows if _name(r)==name]
            result['per_geometry'][name]=dict(
                histories=len(rr),mean_dynamic_Z_pct=
                100*free_metric(rr,free,net,eck))
        out['results'][key]=result
        print('DYN_HOLDOUT',key,round(result['mean_dynamic_Z_pct'],5),
              result['per_geometry'],flush=True)
    out['decision']='diagnostic only; no network weights or hyperparameters selected using these dynamic labels'
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
