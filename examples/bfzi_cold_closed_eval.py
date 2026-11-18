"""Acceptance of a cold-only model inside unchanged V5 BFZI-free dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from examples.bfzi_geom_multi import prepare,free_metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


UNSEEN={'T04','D01','D02','D03','D04'}


def _name(row):
    return Path(row['file'].split('#')[0].replace('\\','/')).stem.split('_')[2]


def _data(mf,split):
    return [r for s in mf[split] for r in prepare(s,32)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cold-model',type=Path,required=True)
    p.add_argument('--thermal-model',type=Path,
                   default=Path('runs/geomfrac_et_v5_closed.pt'))
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
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_coldnet_closed_v1.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    bm=json.loads(a.bb_manifest.read_text(encoding='utf-8-sig'))
    bp=json.loads(a.bb_plan.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    ck=torch.load(a.thermal_model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    atlas=GeomInit.load(cfg['atlas'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    base=PortSelf.load(cfg['port_self_model'])
    cand=ColdNet.load(a.cold_model)
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    rows=_data(mf,'val')
    dev=[r for r in rows if _name(r) in UNSEEN]
    rep=[r for r in rows if _name(r) not in UNSEEN]
    bb=_data(bm,'test')
    mix=float(cfg['shape_mix'])
    def err(rr,pl,port):
        return 100*free_metric(rr,(pl,atlas,port,mu,src,tan,mix),net,eck)
    base_dev=err(dev,plan,base)
    base_rep=err(rep,plan,base)
    candidate_dev=err(dev,plan,cand)
    candidate_rep=err(rep,plan,cand)
    # BB is previously opened: diagnostic only, not part of checkpoint selection.
    bb_base=err(bb,bp,base)
    bb_cand=err(bb,bp,cand)
    accepted=(base_dev-candidate_dev>=.1 and
              candidate_rep-base_rep<=.5)
    res=dict(status='candidate' if accepted else 'rejected',
             independent_new_cold_test='see coldnet training report',
             original_unseen_geometry_count=5,
             original_unseen_history_count=len(dev),
             original_replay_count=6,replay_histories=len(rep),
             frozen_thermal_model=str(a.thermal_model),
             baseline_original_unseen_Z_pct=base_dev,
             candidate_original_unseen_Z_pct=candidate_dev,
             baseline_replay_Z_pct=base_rep,
             candidate_replay_Z_pct=candidate_rep,
             BB_opened_diagnostic=dict(
                baseline_Z_pct=bb_base,candidate_Z_pct=bb_cand),
             decision=('eligible for independent dynamic acceptance'
                       if accepted else 'retain V5 cold model'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print('COLD_CLOSED',json.dumps(res),flush=True)


if __name__=='__main__':
    main()
