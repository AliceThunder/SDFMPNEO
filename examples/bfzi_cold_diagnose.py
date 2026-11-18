"""Inspect cold-only test outliers by physical port components and training neighbors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_coldres import ColdRes
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from examples.bfzi_cold_res_train import score

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('runs/geomfrac_cold500/data.json'))
    p.add_argument('--old',type=Path,default=Path('runs/geomfrac_cold_ports_v1.json'))
    p.add_argument('--config',type=Path,default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_cold500_diagnosis.json'))
    a=p.parse_args()
    new=json.loads(a.data.read_text(encoding='utf-8-sig'))
    old=json.loads(a.old.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    base=PortSelf.load(cfg['port_self_model'])
    plain=ColdNet.load('runs/geomfrac_coldnet_v1.pt')
    res=ColdRes.load('runs/geomfrac_coldres_v1.pt')
    mut=MutFix.load(cfg['mutual_model'])
    tr=new['train']+old['train']
    feats=np.array([[r['geometry'][f'alpha{j}'],
                      r['geometry'][f'core{j}_mm']] for r in tr for j in (1,2)])
    pairs=np.array([j for r in tr for j in (1,2)])
    report=dict(summary={},outliers=[])
    for label,mod in (('base',base),('plain',plain),('residual',res)):
        report['summary'][label]={
            part:score(new[part],mod,mut) if part in new else None
            for part in ('val','test')}
    for row in sorted(new['test'],key=lambda r:report['summary']['base']['test']['per_geometry'][r['name']],reverse=True)[:12]:
        g=row['geometry']
        truths=np.asarray(row['z0'],float)
        air=np.asarray(row['air'])
        if air.shape==(2,2):air=air[[0,0,1],[0,1,1]]
        preds={}
        for label,mod in (('base',base),('plain',plain),('residual',res)):
            ini=mut.apply({'geometry':g},mod.apply({'geometry':g},
                          dict(z0=np.zeros(6),air=np.zeros(3))))
            preds[label]=dict(z0=np.asarray(ini['z0'],float).tolist(),
                              air=np.asarray(ini['air'],float).tolist())
        nei={}
        for port in (1,2):
            target=np.array([g[f'alpha{port}'],g[f'core{port}_mm']])
            v=feats[pairs==port]
            dist=np.linalg.norm((v-target)/[.06,5.],axis=1)
            train_port=[r for r in tr]
            ix=np.argsort(dist)[:5]
            near=[]
            for k in ix:
                r=train_port[int(k)]
                zidx=1 if port==1 else 5
                near.append(dict(name=r['name'],geometry=[
                    r['geometry'][f'alpha{port}'],r['geometry'][f'core{port}_mm']],
                    X_self=float(r['z0'][zidx]),distance=float(dist[k])))
            nei[str(port)]=near
        report['outliers'].append(dict(name=row['name'],geometry=g,
             reference=dict(z0=truths.tolist(),air=air.tolist()),
             predictions=preds,nearest_training=nei))
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for item in report['outliers'][:6]:
        print('OUTLIER',item['name'],item['geometry'],
              'true',np.round(item['reference']['z0'],5).tolist(),
              'baseline',np.round(item['predictions']['base']['z0'],5).tolist(),
              'nearest',item['nearest_training'],flush=True)
    print('SUMMARY',json.dumps({k:{s:{x:v[x] for x in ('cold_Z_pct','air_L_pct')}
                                     for s,v in r.items() if v}
                                for k,r in report['summary'].items()}),flush=True)
if __name__=='__main__':
    main()
