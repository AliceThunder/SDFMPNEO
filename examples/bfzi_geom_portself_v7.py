"""Select native TX/RX cold port fit using deployed dynamic-Z, not cold current."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import free_metric,prepare
from examples.bfzi_geom_portself_v6 import load_rows,cold_score
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('runs/geomfrac_pose42_plan.json'))
    p.add_argument('--train-manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v6_42g_manifest.json'))
    p.add_argument('--val-manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v6_36g_hold6_manifest.json'))
    p.add_argument('--bb-plan',type=Path,
                   default=Path('runs/geomfrac_s5_free11_plan.json'))
    p.add_argument('--bb-manifest',type=Path,
                   default=Path('runs/geomfrac_s5_free11_manifest.json'))
    p.add_argument('--config',type=Path,
                   default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--model',type=Path,
                   default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_portself_v7.npz'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_portself_v7.json'))
    p.add_argument('--min-improve',type=float,default=.001)
    p.add_argument('--replay-tol',type=float,default=.005)
    a=p.parse_args()
    torch.set_num_threads(2)
    pl=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    mf=json.loads(a.train_manifest.read_text(encoding='utf-8-sig'))
    vm=json.loads(a.val_manifest.read_text(encoding='utf-8-sig'))
    bp=json.loads(a.bb_plan.read_text(encoding='utf-8-sig'))
    bm=json.loads(a.bb_manifest.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    tr=load_rows(pl,mf,'train')
    drow=load_rows(pl,mf,'val')
    dev=[r for spec in vm['val'] for r in prepare(spec,32)
         if Path(r['file'].replace('\\','/')).stem.split('_')[2] in
            ('T04','D01','D02','D03','D04')]
    rep=[r for spec in vm['val'] for r in prepare(spec,32)
         if Path(r['file'].replace('\\','/')).stem.split('_')[2] not in
            ('T04','D01','D02','D03','D04')]
    if len(tr)!=42 or len(dev)!=15 or len(rep)!=18:
        raise ValueError('unexpected 42G/5G/6G split')
    if {r['name'] for r in tr}&{r['name'] for r in drow}:
        raise ValueError('training/dev geometries overlap')
    atlas=GeomInit.load(cfg['atlas'])
    base=PortSelf.load(cfg['port_self_model'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    mix=float(cfg['shape_mix'])
    def score(rows,plan,port):
        return free_metric(rows,(plan,atlas,port,mu,src,tan,mix),net,eck)
    base_dev=score(dev,pl,base)
    base_rep=score(rep,pl,base)
    print('BASE',round(100*base_dev,5),round(100*base_rep,5),flush=True)
    best=None
    trials=[]
    for dim in (2,3):
        for ell in (1.,2.5,5.):
            for lam in (.01,.1):
                ps=PortSelf.fit(tr,dim,ell,lam)
                d=score(dev,pl,ps)
                n=score(rep,pl,ps)
                ok=d<base_dev-a.min_improve and n<=base_rep+a.replay_tol
                trials.append(dict(dims=dim,ell=ell,lam=lam,
                                   dev_dynamic_Z_pct=100*d,
                                   replay_dynamic_Z_pct=100*n,
                                   passes=ok))
                if ok and (best is None or d<best[0]):
                    best=(d,n,dim,ell,lam,ps)
                print('TRIAL',dim,ell,lam,round(100*d,5),
                      round(100*n,5),'pass',ok,flush=True)
    if best is None:
        accepted=False
        # Keep the best-development diagnostic distinct from the frozen base.
        row=min(trials,key=lambda q:q['dev_dynamic_Z_pct'])
        dim,ell,lam=row['dims'],row['ell'],row['lam']
        selected=PortSelf.fit(tr,dim,ell,lam)
    else:
        accepted=True
        _,_,dim,ell,lam,selected=best
    bb=[r for spec in bm['test'] for r in prepare(spec,32)]
    bb_base=score(bb,bp,base)
    bb_new=score(bb,bp,selected)
    report=dict(status='candidate_for_independent_acceptance'
                if accepted else 'rejected',
                train_geometries=42,dev_geometries=5,replay_geometries=6,
                BB_geometries=6,dev_histories=len(dev),
                replay_histories=len(rep),BB_histories=len(bb),
                model=str(a.out),dims=dim,ell=ell,lam=lam,
                baseline_dev_Z_pct=100*base_dev,
                baseline_replay_Z_pct=100*base_rep,
                selected_dev_Z_pct=100*score(dev,pl,selected),
                selected_replay_Z_pct=100*score(rep,pl,selected),
                baseline_BB_Z_pct=100*bb_base,
                selected_BB_Z_pct=100*bb_new,
                baseline_cold_dev=cold_score(drow,atlas,base,mu),
                selected_cold_dev=cold_score(drow,atlas,selected,mu),
                trials=trials,
                note='BB already opened: diagnostic only, never used for selecting hyperparameters')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    selected.save(a.out)
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('FINAL',json.dumps({k:report[k] for k in
          ('status','dims','ell','lam','selected_dev_Z_pct',
           'selected_replay_Z_pct','baseline_BB_Z_pct','selected_BB_Z_pct')}),
          flush=True)


if __name__=='__main__':
    main()
