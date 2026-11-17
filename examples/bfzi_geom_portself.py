"""Fit separate native TX/RX coil impedances, select on matched circuit."""
import argparse,json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.frac_memory import zmat
from examples.bfzi_geom_init import records


def score(model,base,rows):
    zerr=[];curr=[];each=[]
    for r in rows:
        y=model.apply(r['case'],base.predict(r['case']))
        target=r['y'][:6]
        q=np.linalg.solve(zmat(target)+r['ext'],[1.,0.])
        p=np.linalg.solve(zmat(y['z0'])+y['ext'],[1.,0.])
        curr.append(np.linalg.norm(p-q)/np.linalg.norm(q))
        zerr.append(np.linalg.norm(y['z0']-target)/np.linalg.norm(target))
        each.append(dict(geometry=r['name'],current_rel=float(curr[-1]),
                         cold_Z_rel=float(zerr[-1])))
    return dict(current=float(np.mean(curr)),cold_Z=float(np.mean(zerr)),
                per_geometry=each)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--base',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    pa.add_argument('--plan',type=Path,default=Path('runs/geomfrac_s5_v4_plan.json'))
    pa.add_argument('--manifest',type=Path,
                    default=Path('runs/geomfrac_s5_v4_30g_manifest.json'))
    pa.add_argument('--cold-plan',type=Path,default=None)
    pa.add_argument('--cold-dir',type=Path,default=None)
    pa.add_argument('--out',type=Path,default=Path('runs/geomfrac_portself_v5.npz'))
    pa.add_argument('--report',type=Path,
                    default=Path('runs/geomfrac_portself_v5_dev.json'))
    a=pa.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    tr=records(plan,mf,'train')
    val=records(plan,mf,'val')
    if a.cold_plan:
        extra=json.loads(a.cold_plan.read_text(encoding='utf-8'))
        for name in extra['train']:
            with np.load(a.cold_dir/f'cold_{name}_32.npz') as d:
                y=d['y'].copy(); ext=d['ext'].copy()
            tr.append(dict(name=name,y=y,ext=ext,case=extra['cases'][name]))
    base=GeomInit.load(a.base)
    best=None
    for dims in (2,3):
        for ell in (.5,1.,1.5,2.5,3.5,5.,7.,10.):
            for lam in (.0001,.001,.01,.05,.1,.3):
                model=PortSelf.fit(tr,dims,ell,lam)
                result=score(model,base,val)
                # Select the circuit-current response; retain modest
                # cold-Z weight as a check on physical port consistency.
                obj=result['current']+.05*result['cold_Z']
                if best is None or obj<best[0]:
                    best=(obj,model,dims,ell,lam,result)
    _,model,dims,ell,lam,res=best
    model.save(a.out)
    original=[]
    for r in val:
        z=base.predict(r['case'])
        q=np.linalg.solve(zmat(r['y'][:6])+r['ext'],[1.,0.])
        p=np.linalg.solve(zmat(z['z0'])+z['ext'],[1.,0.])
        original.append((np.linalg.norm(p-q)/np.linalg.norm(q),
                         np.linalg.norm(z['z0']-r['y'][:6])/np.linalg.norm(r['y'][:6])))
    data=dict(method='two native port cold R/X RBF with air/sea separation',
        train_geometries=len(tr),dev_geometries=len(val),
        dims=dims,ell=ell,lam=lam,dev=res,
        base_dev_current=float(np.mean([x[0] for x in original])),
        base_dev_cold_Z=float(np.mean([x[1] for x in original])),
        zero_BFZI_inference=True)
    a.report.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('PORTSELF_MODEL','train',len(tr),'dims',dims,'ell',ell,'lam',lam,
        'dev_current%',round(100*res['current'],4),
        'dev_coldZ%',round(100*res['cold_Z'],4),
        'original_current%',round(100*data['base_dev_current'],4),flush=True)


if __name__=='__main__':
    main()
