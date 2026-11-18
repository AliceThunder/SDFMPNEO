"""Select independent TX/RX cold-port correction by BFZI-free closed loop."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import free_metric,prepare
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_portblend import PortBlend
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def load(mf,part):
    return [r for spec in mf[part] for r in prepare(spec,32)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v6_36g_hold6_manifest.json'))
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_pose42_plan.json'))
    p.add_argument('--bb-manifest',type=Path,
                   default=Path('runs/geomfrac_s5_free11_manifest.json'))
    p.add_argument('--bb-plan',type=Path,
                   default=Path('runs/geomfrac_s5_free11_plan.json'))
    p.add_argument('--config',type=Path,
                   default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--new-port',type=Path,
                   default=Path('runs/geomfrac_portself_v6.npz'))
    p.add_argument('--model',type=Path,
                   default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_portblend_v1.json'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_portblend_v1_dev.json'))
    p.add_argument('--min-improve',type=float,default=.001)
    p.add_argument('--replay-tol',type=float,default=.005)
    a=p.parse_args()
    torch.set_num_threads(2)
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    bm=json.loads(a.bb_manifest.read_text(encoding='utf-8-sig'))
    bp=json.loads(a.bb_plan.read_text(encoding='utf-8-sig'))
    rows=load(mf,'val')
    dev=[r for r in rows if Path(r['file'].replace('\\','/')).stem.split('_')[2]
         in ('T04','D01','D02','D03','D04')]
    replay=[
        r for r in rows if Path(r['file'].replace('\\','/')).stem.split('_')[2]
        not in ('T04','D01','D02','D03','D04')]
    if len(dev)!=15 or len(replay)!=18:
        raise ValueError('expected 15 unseen and 18 N-replay histories')
    atlas=GeomInit.load(cfg['atlas'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    old=PortSelf.load(cfg['port_self_model'])
    new=PortSelf.load(a.new_port)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    def metric(rr,pn,port):
        return free_metric(rr,(pn,atlas,port,mu,src,tan,
                               float(cfg['shape_mix'])),net,eck)
    base_dev=metric(dev,plan,old)
    base_rep=metric(replay,plan,old)
    best=(base_dev,base_rep,0.,0.)
    trials=[]
    for tx in (0.,.25,.5,1.):
        for rx in (0.,.25,.5,1.):
            model=PortBlend(old,new,tx,rx)
            ed=metric(dev,plan,model)
            er=metric(replay,plan,model)
            trials.append(dict(tx=tx,rx=rx,unseen_Z_pct=100*ed,
                               replay_Z_pct=100*er))
            if ed<best[0]-a.min_improve and er<=base_rep+a.replay_tol:
                best=(ed,er,tx,rx)
            print('TRIAL',tx,rx,round(ed*100,4),round(er*100,4),
                  flush=True)
    _,_,tx,rx=best
    sel=PortBlend(old,new,tx,rx)
    bb=load(bm,'test')
    bb0=metric(bb,bp,old)
    bb1=metric(bb,bp,sel)
    by={}
    for name in bp['test']:
        group=[r for r in bb if Path(r['file'].replace('\\','/')).stem.split('_')[2]==name]
        by[name]=dict(v5_Z_pct=100*metric(group,bp,old),
                      selected_Z_pct=100*metric(group,bp,sel))
    report=dict(status='candidate_selected' if tx or rx else 'rejected',
                selection='unseen dynamic Z improvement >=0.1 pp, replay degradation <=0.5 pp',
                baseline_dev_Z_pct=100*base_dev,
                baseline_replay_Z_pct=100*base_rep,
                selected_dev_Z_pct=100*best[0],
                selected_replay_Z_pct=100*best[1],
                selected_tx=tx,selected_rx=rx,
                bb_baseline_Z_pct=100*bb0,bb_selected_Z_pct=100*bb1,
                bb_per_geometry=by,trials=trials)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    sel.save(a.out,cfg['port_self_model'],a.new_port)
    print('FINAL',json.dumps({k:report[k] for k in ('status',
        'selected_tx','selected_rx','selected_dev_Z_pct',
        'selected_replay_Z_pct','bb_baseline_Z_pct',
        'bb_selected_Z_pct')}),flush=True)


if __name__=='__main__':
    main()
