"""Compare rank-32/48/64 canonical thermal temperature projections."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_canon import basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--split',choices=('train','val','test'),default='val')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    records=[]
    for spec in manifest[a.split]:
        row=prepare(spec,32)[-1]
        rom=row['rom']
        temp=row['temp_K']-rom.amb
        item=dict(case=Path(spec['anchor']).stem.split('_')[2])
        for rank in (32,48,64):
            phi=basis(rom,rank)
            diff=(temp*rom.cap)@phi@phi.T-temp
            item[str(rank)]=dict(
                max_K=float(np.max(abs(diff))),
                sea_max_K=float(np.max(abs(diff[:,:rom.ns]))),
                bulk_sea_max_K=float(np.max(abs(diff[:,:rom.nwet]))),
                skin_tx_max_K=float(np.max(abs(diff[:,rom.nwet:rom.nwet+rom.nskin//2]))))
        records.append(item)
        print('THERMAL_BASIS',item['case'],
              [(rank,round(item[str(rank)]['max_K'],3))
               for rank in (32,48,64)],flush=True)
    result=dict(split=a.split,cases=records,
                **{f'mean{rank}':float(np.mean([x[str(rank)]['max_K'] for x in records]))
                   for rank in (32,48,64)})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('BASIS_MEANS',result['mean32'],result['mean48'],result['mean64'],flush=True)


if __name__=='__main__':
    main()
