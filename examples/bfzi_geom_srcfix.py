"""Train geometry-conditioned sea Joule source on original cold-only anchors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from examples.bfzi_geom_init import records
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def score(net,rows,base):
    errs=[]
    for r in rows:
        src=(net.predict(r['case']) if net is not None else
             base.predict(r['case'])['src'])
        errs.append(np.linalg.norm(src-r['src'])/
                    max(np.linalg.norm(r['src']),1e-15))
    return float(np.mean(errs))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_src_v6.npz'))
    p.add_argument('--report',type=Path,default=Path('runs/geomfrac_src_v6_dev.json'))
    a=p.parse_args()
    plan=json.loads(Path('runs/geomfrac_s5_v4_plan.json').read_text())
    mf=json.loads(Path('runs/geomfrac_s5_v4_30g_manifest.json').read_text())
    tr=records(plan,mf,'train')
    dev=records(plan,mf,'val')
    extra=json.loads(Path('runs/geomfrac_cold_v2_plan.json').read_text())
    for name in extra['train']:
        with np.load(f'runs/cold_atlas/cold_{name}_32.npz') as f:
            src=f['src'].copy()
        tr.append(dict(case=extra['cases'][name],src=src,name=name))
    base=GeomInit.load('runs/geomfrac_init_v1.npz')
    original=score(None,dev,base)
    best=None
    for ell in (4.,6.,10.):
        for lam in (.0001,.001,.01):
            model=SrcFix.fit(tr,ell=ell,lam=lam)
            err=score(model,dev,base)
            if best is None or err<best[0]:
                best=(err,model,ell,lam)
    val,model,ell,lam=best
    model.save(a.out)
    a.report.write_text(json.dumps(dict(train_geometries=len(tr),
      dev_geometries=len(dev),features='first 12 geometry/pose/package scalars',
      ell=ell,lam=lam,baseline_source_rel=original,
      learned_source_rel=val,requires_BFZI_at_query=False),indent=2))
    print('SRC_DEV',len(tr),'ell',ell,'lam',lam,
          'source%',round(100*original,4),'->',round(100*val,4),flush=True)


if __name__=='__main__':
    main()
