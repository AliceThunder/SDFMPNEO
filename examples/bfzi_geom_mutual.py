"""Fit pose/area-aware cold mutual Z from original 30 training geometries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from examples.bfzi_geom_init import records
from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix


def score(net,rows,base,selfnet):
    cur=[];mu=[];air=[]
    for r in rows:
        pred=base.predict(r['case'])
        pred=selfnet.apply(r['case'],pred)
        pred=net.apply(r['case'],pred) if net is not None else pred
        true=r['y'][:6]
        u=np.linalg.solve(zmat(true)+r['ext'],[1,0])
        v=np.linalg.solve(zmat(pred['z0'])+pred['ext'],[1,0])
        cur.append(np.linalg.norm(v-u)/np.linalg.norm(u))
        mu.append(np.linalg.norm(pred['z0'][[2,3]]-true[[2,3]])/
                  np.linalg.norm(true[[2,3]]))
        air.append(abs(pred['air'][1]-r['y'][7]))
    return dict(current=float(np.mean(cur)),
                mutual=float(np.mean(mu)),
                air=float(np.mean(air)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('runs/geomfrac_mutual_v6.npz'))
    ap.add_argument('--report',type=Path,default=Path('runs/geomfrac_mutual_v6_dev.json'))
    a=ap.parse_args()
    plan=json.loads(Path('runs/geomfrac_s5_v4_plan.json').read_text())
    mf=json.loads(Path('runs/geomfrac_s5_v4_30g_manifest.json').read_text())
    tr=records(plan,mf,'train')
    val=records(plan,mf,'val')
    base=GeomInit.load('runs/geomfrac_init_v1.npz')
    selfnet=PortSelf.load('runs/geomfrac_portself_v5.npz')
    original=score(None,val,base,selfnet)
    dims=((0,1,2,3,4,5,6,7,8,9,10),
          (0,1,2,3,4,5,6,7))
    best=None
    for dim in dims:
        for ell in (5.,8.,12.):
            for lam in (.0001,.001,.01):
                model=MutFix.fit(tr,dim,ell,lam,phys=True)
                metrics=score(model,val,base,selfnet)
                obj=metrics['current']+.03*metrics['mutual']
                if best is None or obj<best[0]:
                    best=(obj,model,dim,ell,lam,metrics)
    _,model,dim,ell,lam,metrics=best
    model.save(a.out)
    a.report.write_text(json.dumps(dict(train_geometries=len(tr),
        dev_geometries=len(val),input_feature_ids=list(dim),
        physical_trend='dipole-factor times both coil enclosed areas',
        ell=ell,lam=lam,baseline=original,chosen=metrics,
        requires_BFZI_online=False),indent=2),encoding='utf-8')
    print('MUTUAL_DEV',len(tr),'dims',len(dim),'ell',ell,'lam',lam,
          'current%',round(100*original['current'],4),'to',round(100*metrics['current'],4),
          'mutual%',round(100*original['mutual'],4),'to',round(100*metrics['mutual'],4))


if __name__=='__main__':
    main()
