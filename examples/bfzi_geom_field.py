"""Train and check geometry-only continuous spatial temperature decoder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_field import learn
from sdfmpneo_bfzi.geom_init import GeomInit


def locations(head):
    xyz=np.concatenate((head.xyz,*head.pkg_xyz),axis=0)
    reg=np.r_[np.zeros(head.nwet,int),np.repeat([5,6],head.nskin//2),
             np.full(head.npkg,1),np.full(head.npkg,2)]
    return xyz,reg


def score(atlas,plan,mf,split):
    result=[]
    for spec in mf[split]:
        name=Path(spec['anchor']).stem.split('_')[2]
        head=GeomHeat.load(spec['anchor'])
        xyz,reg=locations(head)
        base=atlas.predict(plan['cases'][name],xyz,reg)
        for row in prepare(spec,32):
            out=head.amb+row['a']@base.T
            truth=row['temp_K'][:,:len(xyz)]
            result.append(dict(case=name,file=row['file'],
                 max_K=float(np.max(abs(out-truth))),
                 rmse_K=float(np.sqrt(np.mean((out-truth)**2)))))
        print('FIELD_VAL',name,
              [round(z['max_K'],3) for z in result if z['case']==name],
              flush=True)
    return dict(histories=len(result),rows=result,
         mean_max_K=float(np.mean([v['max_K'] for v in result])),
         mean_rmse_K=float(np.mean([v['rmse_K'] for v in result])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_s5_v4_plan.json'))
    p.add_argument('--manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v4_30g_manifest.json'))
    p.add_argument('--init',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_field_v1.npz'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_field_v1_dev.json'))
    a=p.parse_args()
    pl=json.loads(a.plan.read_text(encoding='utf-8'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    atlas=learn(mf['train'],GeomInit.load(a.init))
    atlas.save(a.out)
    dev=score(atlas,pl,mf,'val')
    train=score(atlas,pl,dict(train=mf['train'][:2]),'train')
    a.report.write_text(json.dumps(dict(status='geometry-only continuous spatial fields',
          train_geometries=len(mf['train']),dev_geometries=len(mf['val']),
          input_states='teacher projected 32-mode states, spatial decoder isolated',
          dev=dev,train_subset=train),indent=2),encoding='utf-8')
    print('FIELD_DONE',dev['mean_max_K'],dev['mean_rmse_K'],flush=True)


if __name__=='__main__':
    main()
