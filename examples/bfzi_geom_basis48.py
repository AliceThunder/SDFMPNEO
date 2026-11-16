"""Compare canonical 32/48 thermal reconstruction on physical BFZI teachers."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_canon import basis
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=('train','val','test'),default='val')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    result=[]
    for spec in mf[a.split]:
        row=prepare(spec,32)[-1]
        head=GeomHeat.load(str(spec['anchor']))
        cap=np.asarray(head.cap)
        temp=row['temp_K']-head.amb
        region=groups(head)
        item={'case':Path(spec['anchor']).stem.split('_')[2]}
        for rank in (32,48):
            phi=basis(head,rank)
            recon=(temp*cap)@phi@phi.T
            item[str(rank)]=dict(
                max_K=float(np.max(abs(recon-temp))),
                sea_max_K=float(np.max(abs(recon[:,region[0]]-temp[:,region[0]]))),
                pkg_max_K=float(max(np.max(abs(recon[:,sl]-temp[:,sl]))
                                     for sl in region[1:3])))
        result.append(item)
        print(item['case'],
              round(item['32']['max_K'],3),
              round(item['48']['max_K'],3),
              'sea',round(item['32']['sea_max_K'],3),
              round(item['48']['sea_max_K'],3),flush=True)
    report=dict(split=a.split,cases=result,
                mean32=float(np.mean([x['32']['max_K'] for x in result])),
                mean48=float(np.mean([x['48']['max_K'] for x in result])))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AVG',a.split,round(report['mean32'],3),
          round(report['mean48'],3),flush=True)


if __name__=='__main__':
    main()
