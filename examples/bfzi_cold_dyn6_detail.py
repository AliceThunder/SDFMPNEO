"""Per-history dynamic error breakdown for the hardest new cold holdouts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch

from examples.bfzi_geom_multi import free_metric,prepare
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_coldres import ColdRes
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['C0260','C0725'])
    p.add_argument('--plan',type=Path,default=Path('runs/geomfrac_cold_dyn6_plan.json'))
    p.add_argument('--manifest',type=Path,default=Path('runs/geomfrac_s5_cold_dyn6_manifest.json'))
    p.add_argument('--config',type=Path,default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--thermal-model',type=Path,default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--coldnet',type=Path,default=Path('runs/geomfrac_coldnet_v1.pt'))
    p.add_argument('--coldres',type=Path,default=Path('runs/geomfrac_coldres_v1.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_cold_dyn6_detail.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    ck=torch.load(a.thermal_model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
        a_scale=ck['a_scale'],canonical=True,port_sea=True)
    atlas=GeomInit.load(cfg['atlas'])
    mu=MutFix.load(cfg['mutual_model']);src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    ports=dict(v5=PortSelf.load(cfg['port_self_model']),
        coldnet=ColdNet.load(a.coldnet),coldres=ColdRes.load(a.coldres))
    selected={}
    for spec in mf.get('test',[]):
        rows=prepare(spec,32)
        if not rows:continue
        name=Path(rows[0]['file'].split('#')[0]).stem.split('_')[2]
        if name in a.cases:selected[name]=rows
    missing=set(a.cases)-set(selected)
    if missing:raise RuntimeError(f'missing completed dynamic cases: {sorted(missing)}')
    out=dict(status='complete',cases=a.cases,per_history={})
    for key,port in ports.items():
        free=(plan,atlas,port,mu,src,tan,float(cfg['shape_mix']))
        for name,rows in selected.items():
            out['per_history'].setdefault(name,{})
            vals=[]
            for row in rows:
                pct=100*free_metric([row],free,net,eck)
                label=Path(row['file'].split('#')[0]).stem.split('_')[-1]
                out['per_history'][name].setdefault(label,{})[key]=pct
                vals.append(pct)
            out['per_history'][name].setdefault('mean',{})[key]=sum(vals)/len(vals)
            print('DYN_DETAIL',name,key,out['per_history'][name],flush=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
