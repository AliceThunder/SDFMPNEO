"""Evaluate rank48 shared thermal model on the frozen six-geometry dynamics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_coldres import ColdRes
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('runs/geomfrac_cold_dyn6_plan.json'))
    p.add_argument('--manifest',type=Path,default=Path('runs/geomfrac_s5_cold_dyn6_manifest.json'))
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--thermal-model',type=Path,required=True)
    p.add_argument('--skin-tan-model',type=Path,default=None,
        help='optional tangent using direct rank48 skin coordinates')
    p.add_argument('--coldnet',type=Path,default=Path('runs/geomfrac_coldnet_v1.pt'))
    p.add_argument('--coldres',type=Path,default=Path('runs/geomfrac_coldres_v1.pt'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    ck=torch.load(a.thermal_model,map_location='cpu',weights_only=True)
    rank=int(ck['rank'])
    net=GeomFracET(ng=int(ck['ng']),rank=rank,
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    atlas=GeomInit.load(cfg['atlas'])
    if atlas.rank!=rank:raise ValueError('thermal model and atlas ranks differ')
    if a.skin_tan_model:
        from sdfmpneo_bfzi.geom_skintan import restore_rank48
        tan=restore_rank48(a.skin_tan_model,cfg['z_fix_model'])
    else:
        tan=load_z(cfg['z_fix_model'])
    mu=MutFix.load(cfg['mutual_model'])
    sf=SrcFix.load(cfg['source_model'])
    ports=dict(v5=PortSelf.load(cfg['port_self_model']),
        coldnet=ColdNet.load(a.coldnet),coldres=ColdRes.load(a.coldres))
    out=dict(status='complete',rank=rank,cases={},summary={})
    keys=('dynamic_z_rel_L2','current_rel_L2','material_mean_rmse_K',
          'material_mean_max_K','sea_mean_max_K','pkg_mean_max_K',
          'copper_max_K','energy_budget_W','sea_power_rel_L2',
          'copper_power_rel_L2')
    for spec in mf['test']:
        rows=prepare(spec,rank)
        if not rows:continue
        name=Path(rows[0]['file'].split('#')[0]).stem.split('_')[2]
        case=plan['cases'][name]
        out['cases'][name]={}
        for pname,port in ports.items():
            init=atlas.predict(case)
            for model in (port,mu,sf):
                init=model.apply(case,init)
            vals=[]
            for row in rows:
                pred=rollout(case,atlas,net,ck,row['time'],row['voltage'],
                    tan,init_override=init,shape_mix=float(cfg['shape_mix']))
                m=metric(pred,row)
                vals.append(m)
                label=Path(row['file'].split('#')[0]).stem.split('_')[-1]
                out['cases'][name].setdefault(label,{})[pname]={
                    k:float(m[k]) for k in keys if k in m}
            out['cases'][name][pname+'_mean']={
                k:float(np.mean([v[k] for v in vals if k in v]))
                for k in keys if all(k in v for v in vals)}
            print('RANK_EVAL',name,pname,out['cases'][name][pname+'_mean'],flush=True)
    for pname in ports:
        vals=[hist[pname+'_mean'] for case in out['cases'].values()
              if (hist:=case).get(pname+'_mean') is not None]
        out['summary'][pname]={
            k:float(np.mean([v[k] for v in vals if k in v]))
            for k in keys if vals and all(k in v for v in vals)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('RANK_EVAL_FINAL',json.dumps(out['summary']),flush=True)


if __name__=='__main__':main()
